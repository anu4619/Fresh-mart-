import io
import wave
import base64
import logging
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
STT_URL        = "https://api.sarvam.ai/speech-to-text"
TTS_URL        = "https://api.sarvam.ai/text-to-speech"


def _build_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


def _read_wav(wav_bytes: bytes) -> tuple:
    """Returns (pcm_bytes, sample_rate, channels)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_frames = wf.getnframes()
        frames   = wf.readframes(n_frames)
        rate     = wf.getframerate()
        ch       = wf.getnchannels()
    return frames, rate, ch


def _normalize_pcm(pcm: bytes, rate: int, channels: int) -> bytes:
    """Convert any PCM to 16kHz mono 16-bit."""
    import audioop

    # Stereo → mono
    if channels == 2:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)

    # Resample to 16000 if needed
    if rate != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)

    return pcm


async def speech_to_text(pcm_bytes: bytes, sample_rate: int = 16000) -> tuple:
    """Always returns (transcript_str, language_str). Uses Groq Whisper for instantaneous parsing (demo speed)."""
    if not pcm_bytes or len(pcm_bytes) < 2048:
        return ("", "en-IN")

    wav_bytes = _build_wav_bytes(pcm_bytes, sample_rate)

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        
        file_tuple = ("audio.wav", wav_bytes, "audio/wav")
        response = await client.audio.transcriptions.create(
            file=file_tuple,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
        
        transcript = response.text.strip()
        detected_lang = getattr(response, "language", "").lower()
        
        lang_map = {
            "hi": "hi-IN", "hindi": "hi-IN",
            "ta": "ta-IN", "tamil": "ta-IN",
            "te": "te-IN", "telugu": "te-IN",
            "kn": "kn-IN", "kannada": "kn-IN",
            "ml": "ml-IN", "malayalam": "ml-IN",
            "mr": "mr-IN", "marathi": "mr-IN",
            "gu": "gu-IN", "gujarati": "gu-IN",
            "bn": "bn-IN", "bengali": "bn-IN",
            "pa": "pa-IN", "punjabi": "pa-IN",
            "or": "od-IN", "odia": "od-IN",
            "en": "en-IN", "english": "en-IN"
        }
        
        lang_code = lang_map.get(detected_lang, "unknown")
        
        logger.info(f"⚡ Groq STT result: '{transcript}' | Detected: {detected_lang} -> {lang_code}")
        
        return (transcript, lang_code)

    except Exception as e:
        logger.error(f"STT error: {e}")
        return ("", "en-IN")


async def text_to_speech(
    text: str,
    language: str = "en-IN",
) -> bytes | None:
    """Returns raw 16kHz mono 16-bit PCM bytes."""
    if not text or not text.strip():
        return None

    if language == "unknown":
        language = "en-IN"

    # Truncate if too long — Sarvam has input limits
    text = text.strip()[:500]

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type":         "application/json",
    }
    payload = {
        "inputs":               [text],
        "target_language_code": language,
        "speaker":              "anushka",
        "pitch":                0,
        "pace":                 1.0,     # normal speed — 1.05 can cause distortion
        "loudness":             1.5,
        "speech_sample_rate":   16000,
        "enable_preprocessing": True,
        "model":                "bulbul:v2",
    }

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(TTS_URL, headers=headers, json=payload)

        if resp.status_code != 200:
            logger.error(f"TTS {resp.status_code}: {resp.text[:200]}")
            return None

        audios = resp.json().get("audios", [])
        if not audios:
            logger.error("TTS returned empty audios list")
            return None

        wav_b        = base64.b64decode(audios[0])
        pcm, rate, ch = _read_wav(wav_b)
        pcm          = _normalize_pcm(pcm, rate, ch)

        logger.info(f"TTS ok: {len(pcm)} bytes at 16kHz mono")
        return pcm

    except httpx.TimeoutException:
        logger.error("TTS timeout")
        return None
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        return None
                                                        