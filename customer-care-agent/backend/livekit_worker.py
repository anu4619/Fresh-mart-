import sys
import asyncio
import audioop
import logging
import os
import re
from dotenv import load_dotenv
from livekit import rtc, api
from backend.sarvam import speech_to_text, text_to_speech
from backend.llm_agent import run_llm, clear_user_memory, set_user_language
from backend.transcript_store import add_message, clear_messages
from backend.db import archive_shopping_list

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("worker")

LIVEKIT_URL    = os.getenv("LIVEKIT_URL")
API_KEY        = os.getenv("LIVEKIT_API_KEY")
API_SECRET     = os.getenv("LIVEKIT_API_SECRET")
ROOM_NAME      = os.getenv("LIVEKIT_ROOM", "shopping-room")
AGENT_IDENTITY = "freshmart-agent"

TARGET_RATE       = 16000
SILENCE_THRESHOLD = 1500  # heavily increased to avoid ambient noise triggering STT
SILENCE_NEEDED    = 70    # = ~0.7 seconds wait before processing (Faster latency)
MIN_SPEECH_FRAMES = 40    # = ~400ms minimum valid speech (ignores short clicks/pops)
MAX_SPEECH_FRAMES = 500   # = ~5.0 seconds max contiguous speech before forcefully processing!

SAMPLES_PER_FRAME = 1600  # 100ms at 16kHz
BYTES_PER_FRAME   = SAMPLES_PER_FRAME * 2


def _split_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _convert_frame(frame: rtc.AudioFrame) -> bytes:
    raw = bytes(frame.data)
    if frame.num_channels == 2:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
    if frame.sample_rate != TARGET_RATE:
        raw, _ = audioop.ratecv(raw, 2, 1, frame.sample_rate, TARGET_RATE, None)
    return raw


async def _stream_tts(pcm: bytes, source: rtc.AudioSource):
    """
    Stream TTS audio frame by frame.
    We add a small sleep per frame to match real-time playback speed.
    This prevents buffer overflow and audio breaking.
    """
    if not pcm:
        return

    # Pad to frame boundary
    remainder = len(pcm) % BYTES_PER_FRAME
    if remainder:
        pcm = pcm + bytes(BYTES_PER_FRAME - remainder)

    total_frames = len(pcm) // BYTES_PER_FRAME

    for i in range(0, len(pcm), BYTES_PER_FRAME):
        chunk = pcm[i: i + BYTES_PER_FRAME]
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=TARGET_RATE,
            num_channels=1,
            samples_per_channel=SAMPLES_PER_FRAME,
        )
        await source.capture_frame(frame)
        # Sleep exactly 100ms per frame — matches real-time audio rate
        # This is the key fix for voice breaking
        await asyncio.sleep(0.1)


async def _speak(
    text: str,
    language: str,
    source: rtc.AudioSource,
    speaking: asyncio.Event,
):
    """Generate TTS and stream it, with speaking flag set the whole time."""
    if not text.strip():
        return

    try:
        audio = await text_to_speech(text, language=language)
        if not audio:
            logger.warning(f"TTS returned no audio for: {text!r}")
            return

        speaking.set()
        await _stream_tts(audio, source)
        # Small pause after sentence ends before clearing speaking flag
        await asyncio.sleep(0.3)
    except Exception as e:
        logger.error(f"Speak error: {e}")
    finally:
        speaking.clear()


async def _process_utterance(
    user_id: str,
    pcm: bytes,
    source: rtc.AudioSource,
    speaking: asyncio.Event,
    processing: asyncio.Event,
    order_confirmed: asyncio.Event,
):
    if processing.is_set():
        logger.info(f"[{user_id}] Skipping — busy")
        return

    processing.set()
    try:
        # STT
        result     = await speech_to_text(pcm, TARGET_RATE)
        transcript = result[0].strip()
        lang       = result[1]

        if not transcript:
            logger.info(f"[{user_id}] Empty transcript — skip")
            return

        # Whisper hallucinates on silence/noise often with these exact phrases
        hallucinations = [
            "thank you.", "thank you", "bye.", "bye", "goodbye.", "goodbye",
            "i didn't quite catch that, can you please rephrase?", "how can i assist you today?",
            "you're welcome, have a great day", "you're welcome, have a great day."
        ]
        if transcript.lower() in hallucinations:
            logger.info(f"[{user_id}] Ignoring hallucination: '{transcript}'")
            return

        logger.info(f"[{user_id}] 🎤 '{transcript}' | lang={lang}")
        add_message(user_id, "user", transcript)
        set_user_language(user_id, lang)

        # LLM
        reply, confirmed = await run_llm(user_id, transcript)
        logger.info(f"[{user_id}] 🤖 '{reply}'")
        add_message(user_id, "agent", reply)

        # Speak sentence by sentence with speaking flag set
        sentences = _split_sentences(reply)
        for sentence in sentences:
            if sentence:
                await _speak(sentence, lang, source, speaking)
                # Brief gap between sentences
                await asyncio.sleep(0.15)

        if confirmed:
            order_confirmed.set()

    except Exception as e:
        logger.error(f"[{user_id}] Process error: {e}", exc_info=True)
    finally:
        processing.clear()


async def handle_participant(
    track: rtc.Track,
    participant: rtc.RemoteParticipant,
    source: rtc.AudioSource,
    speaking: asyncio.Event,
):
    uid    = participant.identity
    stream = rtc.AudioStream(track)
    logger.info(f"Listening to: {uid}")

    buf             = bytearray()
    silent_frames   = 0
    speech_frames   = 0
    in_speech       = False
    processing      = asyncio.Event()
    order_confirmed = asyncio.Event()

    async for event in stream:

        # Completely ignore mic while agent is speaking — prevents echo/interrupt
        if speaking.is_set():
            buf.clear()
            speech_frames = 0
            silent_frames = 0
            in_speech     = False
            continue

        # After order confirmed — keep room live but stop processing
        if order_confirmed.is_set():
            continue

        pcm = _convert_frame(event.frame)
        rms = audioop.rms(pcm, 2)

        if rms > SILENCE_THRESHOLD:
            buf.extend(pcm)
            speech_frames += 1
            silent_frames  = 0
            in_speech      = True

        elif in_speech:
            buf.extend(pcm)
            silent_frames += 1

            if silent_frames >= SILENCE_NEEDED or speech_frames >= MAX_SPEECH_FRAMES:
                if speech_frames >= MIN_SPEECH_FRAMES:
                    captured = bytes(buf)
                    logger.info(
                        f"[{uid}] Captured {speech_frames} frames "
                        f"({len(captured)//1000}kb)"
                    )
                    asyncio.create_task(
                        _process_utterance(
                            uid, captured, source,
                            speaking, processing, order_confirmed,
                        )
                    )
                buf.clear()
                speech_frames = 0
                silent_frames = 0
                in_speech     = False


async def send_greeting(
    source: rtc.AudioSource,
    speaking: asyncio.Event,
    customer_name: str,
):
    greeting = (
        f"Hello {customer_name}, welcome to FreshMart. "
        f"I am Priya, your shopping assistant. "
        f"Please tell me what you want to add to your list. "
        f"Say confirm order when you are done."
    )
    add_message(customer_name, "agent", greeting)
    for sentence in _split_sentences(greeting):
        await _speak(sentence, "en-IN", source, speaking)
        await asyncio.sleep(0.15)


async def main():
    room  = rtc.Room()
    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(AGENT_IDENTITY)
        .with_name("FreshMart Assistant")
        .with_grants(api.VideoGrants(
            room_join=True,
            room=ROOM_NAME,
            can_publish=True,
            can_subscribe=True,
        ))
        .to_jwt()
    )

    source   = rtc.AudioSource(TARGET_RATE, 1)
    track    = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
    speaking = asyncio.Event()
    tasks: dict = {}

    @room.on("participant_connected")
    def on_join(participant: rtc.RemoteParticipant):
        logger.info(f"Joined: {participant.identity}")
        # Archive any leftover items from a previously crashed session
        archive_shopping_list(participant.identity)
        asyncio.create_task(
            send_greeting(source, speaking, participant.identity)
        )

    @room.on("participant_disconnected")
    def on_leave(participant: rtc.RemoteParticipant):
        t = tasks.pop(participant.identity, None)
        if t:
            t.cancel()
        clear_user_memory(participant.identity)
        clear_messages(participant.identity)
        archive_shopping_list(participant.identity)

    @room.on("track_subscribed")
    def on_track(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            t = asyncio.create_task(
                handle_participant(track, participant, source, speaking)
            )
            tasks[participant.identity] = t

    @room.on("track_unsubscribed")
    def on_unsub(track, pub, participant):
        t = tasks.pop(participant.identity, None)
        if t:
            t.cancel()

    logger.info(f"Connecting to {ROOM_NAME}...")
    await room.connect(LIVEKIT_URL, token)
    logger.info("✅ Agent connected")
    await room.local_participant.publish_track(track)
    logger.info("✅ Audio track published")

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        await room.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")