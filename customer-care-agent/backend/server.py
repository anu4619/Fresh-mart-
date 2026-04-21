import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from livekit import api as livekit_api
from backend.llm_agent import run_llm, get_shopping_list
from backend.db import load_shopping_list, get_all_users, collection, get_shopping_history
from backend.transcript_store import get_messages

load_dotenv()

app = FastAPI(title="FreshMart Voice Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_front = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_front):
    app.mount("/ui", StaticFiles(directory=_front, html=True), name="frontend")


class ChatRequest(BaseModel):
    user_id: str
    text: str


@app.get("/")
def root():
    return {"status": "FreshMart Agent running", "ui": "/ui"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/token")
def get_token(identity: str):
    if not identity.strip():
        raise HTTPException(400, "identity required")
    token = (
        livekit_api.AccessToken(
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET"),
        )
        .with_identity(identity.strip())
        .with_name(identity.strip())
        .with_grants(livekit_api.VideoGrants(
            room_join=True,
            room=os.getenv("LIVEKIT_ROOM", "shopping-room"),
            can_publish=True,
            can_subscribe=True,
        ))
        .to_jwt()
    )
    return {"token": token, "identity": identity}


@app.post("/chat")
async def chat(req: ChatRequest):
    reply, _ = await run_llm(req.user_id, req.text)
    return {"reply": reply, "shopping_list": get_shopping_list(req.user_id)}


@app.get("/list/{user_id}")
def get_list(user_id: str):
    items = load_shopping_list(user_id)
    return {"user_id": user_id, "shopping_list": items, "count": len(items)}


@app.get("/transcript/{user_id}")
def get_transcript(user_id: str):
    return {"user_id": user_id, "messages": get_messages(user_id)}


@app.get("/history/{user_id}")
def get_history(user_id: str):
    return {"user_id": user_id, "history": get_shopping_history(user_id)}


@app.get("/stream/{user_id}")
async def stream_list(user_id: str):
    """SSE — pushes list and transcript updates in real time."""
    async def event_generator():
        last_list = None
        last_msgs = None
        yield f"data: {json.dumps({'type': 'connected', 'user_id': user_id})}\n\n"
        while True:
            try:
                items   = load_shopping_list(user_id)
                list_js = json.dumps(items)
                if list_js != last_list:
                    last_list = list_js
                    yield f"data: {json.dumps({'type': 'list', 'items': items})}\n\n"

                msgs    = get_messages(user_id)
                msgs_js = json.dumps(msgs)
                if msgs_js != last_msgs:
                    last_msgs = msgs_js
                    yield f"data: {json.dumps({'type': 'transcript', 'messages': msgs})}\n\n"
            except Exception:
                pass
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/all-lists")
def all_lists():
    users = get_all_users()
    return {uid: load_shopping_list(uid) for uid in users}


@app.delete("/list/{user_id}")
def clear_list(user_id: str):
    if collection is not None:
        collection.delete_one({"user_id": user_id})
    return {"status": "cleared", "user_id": user_id}