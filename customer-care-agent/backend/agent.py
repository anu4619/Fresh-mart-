customer_memory = {}
conversation_state = {}

def shopping_agent(user_id, text):
    text = text.lower().strip()

    if user_id not in customer_memory:
        customer_memory[user_id] = []

    if user_id not in conversation_state:
        conversation_state[user_id] = {
            "awaiting_action": None
        }

    shopping_list = customer_memory[user_id]
    state = conversation_state[user_id]

    # Greeting
    if "hello" in text or "hi" in text:
        state["awaiting_action"] = "action"
        return f"Hello {user_id}! I am your shopping assistant. Would you like to add, remove, or hear your list?"

    # If waiting for action choice
    if state["awaiting_action"] == "action":
        if "add" in text:
            state["awaiting_action"] = "add_item"
            return "Sure. What item would you like to add?"
        elif "remove" in text or "delete" in text:
            state["awaiting_action"] = "remove_item"
            return "Okay. What item would you like to remove?"
        elif "list" in text or "show" in text:
            if not shopping_list:
                return "Your shopping list is empty. Would you like to add something?"
            return "Your shopping list contains: " + ", ".join(shopping_list)
        else:
            return "Please say add, remove, or show list."

    # Adding item
    if state["awaiting_action"] == "add_item":
        shopping_list.append(text)
        state["awaiting_action"] = "action"
        return f"{text} added successfully. Would you like to add more, remove something, or hear your list?"

    # Removing item
    if state["awaiting_action"] == "remove_item":
        if text in shopping_list:
            shopping_list.remove(text)
            reply = f"{text} removed successfully."
        else:
            reply = f"{text} was not found in your list."
        state["awaiting_action"] = "action"
        return reply + " Would you like to do anything else?"

    # Direct commands without greeting
    if "add" in text:
        item = text.replace("add", "").strip()
        if item:
            shopping_list.append(item)
            return f"{item} added. Anything else?"
        return "What would you like to add?"

    if "remove" in text:
        item = text.replace("remove", "").strip()
        if item in shopping_list:
            shopping_list.remove(item)
            return f"{item} removed. Anything else?"
        return f"{item} not found."

    if "list" in text:
        if not shopping_list:
            return "Your list is empty."
        return "Your shopping list contains: " + ", ".join(shopping_list)

    return "I didn't understand that. Would you like to add, remove, or hear your list?"
