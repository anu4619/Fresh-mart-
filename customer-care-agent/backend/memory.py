import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    user_id: str
    shopping_list: list = field(default_factory=list)
    conversation_history: list = field(default_factory=list)
    awaiting_action: Optional[str] = None

    def add_item(self, item: str) -> bool:
        item = item.lower().strip()
        if item and item not in self.shopping_list:
            self.shopping_list.append(item)
            return True
        return False

    def remove_item(self, item: str) -> bool:
        item = item.lower().strip()
        if item in self.shopping_list:
            self.shopping_list.remove(item)
            return True
        match = next((i for i in self.shopping_list if item in i or i in item), None)
        if match:
            self.shopping_list.remove(match)
            return True
        return False

    def get_list_str(self) -> str:
        if not self.shopping_list:
            return "Your shopping list is empty."
        return "Your list has: " + ", ".join(self.shopping_list) + "."


_sessions: dict = {}
_lock = asyncio.Lock()


async def get_session(user_id: str) -> UserSession:
    async with _lock:
        if user_id not in _sessions:
            _sessions[user_id] = UserSession(user_id=user_id)
        return _sessions[user_id]


async def delete_session(user_id: str):
    async with _lock:
        _sessions.pop(user_id, None)