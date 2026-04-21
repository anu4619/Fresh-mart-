import logging
from backend.db import client, DB_NAME

logger = logging.getLogger(__name__)

try:
    if client is not None:
        db = client[DB_NAME]
        collection = db["transcripts"]
    else:
        collection = None
except Exception:
    collection = None

def add_message(user_id: str, role: str, text: str):
    if collection is None:
        return
    try:
        collection.update_one(
            {"user_id": user_id},
            {"$push": {"messages": {"role": role, "content": text}}},
            upsert=True
        )
    except Exception as e:
        logger.error(f"Save transcript error: {e}")

def get_messages(user_id: str) -> list:
    if collection is None:
        return []
    try:
        doc = collection.find_one({"user_id": user_id})
        if doc and "messages" in doc:
            return doc["messages"]
    except Exception as e:
        logger.error(f"Load transcript error: {e}")
    return []

def clear_messages(user_id: str):
    if collection is None:
        return
    try:
        collection.delete_one({"user_id": user_id})
    except Exception as e:
        logger.error(f"Clear transcript error: {e}")