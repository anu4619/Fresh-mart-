import os
import json
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
from backend.db import save_shopping_list, load_shopping_list

load_dotenv()

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY_TURNS = 10

_user_state: dict = {}

LANGUAGE_NAMES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "bn-IN": "Bengali",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
}


def _get_state(user_id: str) -> dict:
    if user_id not in _user_state:
        existing = load_shopping_list(user_id)
        _user_state[user_id] = {
            "history": [],
            "shopping_list": existing,
            "language": "en-IN",
        }
    return _user_state[user_id]


def set_user_language(user_id: str, language: str):
    _get_state(user_id)["language"] = language
    logger.info(f"[{user_id}] Language: {language}")


def get_shopping_list(user_id: str) -> list:
    return _get_state(user_id)["shopping_list"]


def clear_user_memory(user_id: str):
    if user_id in _user_state:
        save_shopping_list(user_id, _user_state[user_id]["shopping_list"])
        logger.info(f"[{user_id}] Saved to MongoDB on disconnect")
    _user_state.pop(user_id, None)


def _format_list_for_prompt(shopping_list: list) -> str:
    if not shopping_list:
        return "empty"
    return ", ".join(
        f"{item['name']} ({item['quantity']})" for item in shopping_list
    )


def _build_system_prompt(user_id: str, shopping_list: list, language: str) -> str:
    list_str = _format_list_for_prompt(shopping_list)
    lang_name = LANGUAGE_NAMES.get(language, "English")
    return f"""You are Priya, a polite professional AI shopping assistant at FreshMart.
You are on a voice call with {user_id}.

YOUR CURRENT SHOPPING LIST MEMORY:
[{list_str}]

CRITICAL INSTRUCTIONS:
1. You MUST output a strictly valid JSON object. Do not output anything outside of the JSON format.
2. Structure your JSON exactly like this:
{{
  "adds": [
    {{
      "item": "Name of the item identified, MUST be written purely in {lang_name} (e.g. 'टमाटर' if Hindi, 'tomato' if English)",
      "quantity": "Quantity including unit, e.g. '2 kg', '1 piece' or null"
    }},
    {{
      "item": "Name of another item if multiple were mentioned",
      "quantity": "Quantity for the other item"
    }}
  ],
  "removes": [
    "Name of the item to remove, MUST be written purely in {lang_name}"
  ],
  "updates": [
    {{
      "item": "Name of the existing item to update, MUST be written purely in {lang_name}",
      "new_quantity": "The new quantity for the item"
    }}
  ],
  "is_confirmed": false,
  "reply": "Your conversational reply to the user spoken aloud"
}}

RULES FOR ACTIONS:
- "adds": List items the user specifies to add to their cart. You can add multiple items. Leave empty if none.
- "removes": List items the user specifies to remove from their cart. Leave empty if none.
- "updates": List items the user specifies to change the quantity of. Leave empty if none.
- "is_confirmed": Set to true ONLY when the user explicitly says "confirm order", "order confirm", or indicates they are finished shopping. Otherwise false.
- For general questions or chit-chat, leave adds, removes, and updates empty and is_confirmed false.

RULES FOR THE REPLY FIELD:
- Your "reply" MUST be spoken purely in {lang_name} at all times. Do not output English if the language is Hindi.
- If the user adds an item, proactively suggest 1 complementary item they might need (e.g., if they buy pasta, suggest pasta sauce). Do not be overly pushy.
- If the user asks for ingredients for a specific recipe (e.g., 'ingredients for butter chicken'), automatically infer the common ingredients and add them to the 'adds' array, then mention them in your reply.
- Keep the reply to 1-2 very short sentences (perfect for a fast voice call), unless you are confirming an order with multiple items.
- Keep it plain spoken (no symbols, markdown, acronyms).
- If they want to add an item but gave NO quantity, ask them how much in your reply and do not add it yet.
- If "is_confirmed" is true, your reply MUST read out every single item (and quantity) from the CURRENT SHOPPING LIST MEMORY plus any NEW items you are adding in the 'adds' array. This acts as a final order summary before you warmly thank them. You MUST completely translate this final itemized summary into {lang_name}.
"""


async def run_llm(user_id: str, user_message: str) -> tuple:
    state         = _get_state(user_id)
    shopping_list = state["shopping_list"]
    language      = state.get("language", "en-IN")
    
    # Track the conversational history context
    history = state["history"]
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]
    
    # Note: We must NOT pass the assistant's previous raw JSON into history as "assistant",
    # otherwise it confuses the context window. We only track what they actually "said".
    
    messages = [{"role": "system", "content": _build_system_prompt(user_id, shopping_list, language)}]
    for h in trimmed:
        messages.append(h)
    messages.append({"role": "user", "content": user_message})

    try:
        response = await _client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=250,
            response_format={"type": "json_object"},
        )
        raw_output = response.choices[0].message.content.strip()
        data = json.loads(raw_output)
    except Exception as e:
        logger.error(f"LLM JSON extraction error: {e}")
        # Fallback if parsing completely fails
        return "I'm sorry, I encountered an issue making sense of that. Could you repeat?", False

    adds   = data.get("adds", [])
    removes = data.get("removes", [])
    updates = data.get("updates", [])
    reply  = data.get("reply", "Okay.")
    is_confirmed = data.get("is_confirmed", False)

    logger.info(f"[{user_id}] Adds={adds}, Removes={removes}, Updates={updates}, Confirmed={is_confirmed} | LLM Reply: {reply!r}")

    list_changed = False

    # Process Actions Locally
    if adds:
        for i in adds:
            item = i.get("item")
            qty = i.get("quantity")
            if item and qty:
                entry = {"name": item.strip(), "quantity": qty.strip()}
                shopping_list.append(entry)
                logger.info(f"[{user_id}] ✅ ADDED: {entry}")
                list_changed = True

    if removes:
        missing_items = []
        for item in removes:
            # Handle if LLM incorrectly outputs an object like {"item": "sugar"} instead of "sugar"
            if isinstance(item, dict):
                item = item.get("item") or item.get("name")
            
            if not item or not isinstance(item, str):
                continue
            
            # Fuzzy match to drop
            removed = None
            for entry in shopping_list:
                if item.lower() in entry["name"].lower() or entry["name"].lower() in item.lower():
                    removed = entry
                    break
            
            if removed:
                shopping_list.remove(removed)
                logger.info(f"[{user_id}] ❌ REMOVED: {removed}")
                list_changed = True
            else:
                logger.info(f"[{user_id}] Requested item {item!r} to remove but not found in list.")
                missing_items.append(item)
                
        if missing_items:
            reply = f"I am sorry, but I could not find {', '.join(missing_items)} on your list to remove. " + reply

    if updates:
        missing_updates = []
        for upd in updates:
            if not upd or not isinstance(upd, dict):
                continue
            item = upd.get("item")
            new_qty = upd.get("new_quantity")
            if not item or not new_qty:
                continue
            
            # Fuzzy match to update
            updated = None
            for entry in shopping_list:
                if item.lower() in entry["name"].lower() or entry["name"].lower() in item.lower():
                    entry["quantity"] = new_qty
                    updated = entry
                    break
            
            if updated:
                logger.info(f"[{user_id}] 🔄 UPDATED: {updated}")
                list_changed = True
            else:
                logger.info(f"[{user_id}] Requested item {item!r} to update but not found.")
                missing_updates.append(item)
                
        if missing_updates:
            reply = f"I could not find {', '.join(missing_updates)} on your list to update. " + reply

    if list_changed:
        save_shopping_list(user_id, shopping_list)

    # Record ONLY the spoken conversation string to history, not the JSON block
    state["history"].append({"role": "user", "content": user_message})
    state["history"].append({"role": "assistant", "content": reply})

    return reply, is_confirmed


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        user_id = "test_user_json"
        print("FreshMart JSON Intelligence Shell (type 'quit' to exit)")
        print("Say something conversational like: 'remove the sugar that is in it'")
        while True:
            try:
                user_msg = input("\nYou: ")
                if user_msg.lower().strip() in ('quit', 'exit'):
                    break
                reply, is_confirmed = await run_llm(user_id, user_msg)
                print(f"Priya: {reply}")
                if is_confirmed:
                    print("[System: Order Confirmed!]")
            except EOFError:
                break

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass