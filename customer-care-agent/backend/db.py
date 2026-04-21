import os
import logging
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI       = os.getenv("MONGO_URI")
DB_NAME         = os.getenv("DB_NAME", "shopping_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "shopping_items")

try:
    client     = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db         = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.create_index([("user_id", ASCENDING)])
    client.server_info()
    logger.info("✅ MongoDB connected")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    collection = None


def save_shopping_list(user_id: str, items: list):
    if collection is None:
        logger.error("MongoDB not connected")
        return
    try:
        collection.update_one(
            {"user_id": user_id},
            {"$set": {
                "user_id":    user_id,
                "items":      items,
                "updated_at": datetime.utcnow(),
            }},
            upsert=True,
        )
        logger.info(f"[{user_id}] ✅ Saved: {items}")
    except Exception as e:
        logger.error(f"[{user_id}] Save error: {e}")


def load_shopping_list(user_id: str) -> list:
    if collection is None:
        return []
    try:
        doc = collection.find_one({"user_id": user_id})
        if doc and "items" in doc:
            return doc["items"]
        return []
    except Exception as e:
        logger.error(f"[{user_id}] Load error: {e}")
        return []


def archive_shopping_list(user_id: str):
    if collection is None:
        return
    try:
        doc = collection.find_one({"user_id": user_id})
        if doc and doc.get("items"):
            # Only archive if the list actually has items
            history_entry = {
                "created_at": datetime.utcnow().isoformat(),
                "items": doc["items"]
            }
            collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {"history": {"$each": [history_entry], "$position": 0}},
                    "$set": {"items": []}
                }
            )
            logger.info(f"[{user_id}] 🗄️ Archived shopping list.")
    except Exception as e:
        logger.error(f"[{user_id}] Archive error: {e}")


def get_all_users() -> list:
    if collection is None:
        return []
    try:
        return [doc["user_id"] for doc in collection.find({}, {"user_id": 1})]
    except Exception as e:
        logger.error(f"get_all_users error: {e}")
        return []

def get_shopping_history(user_id: str) -> list:
    if collection is None:
        return []
    try:
        doc = collection.find_one({"user_id": user_id})
        if doc and "history" in doc:
            return doc["history"]
        return []
    except Exception as e:
        logger.error(f"[{user_id}] History load error: {e}")
        return []