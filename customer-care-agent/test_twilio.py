import os
import asyncio
from dotenv import load_dotenv
from backend.db import get_user, users_collection
from backend.twilio_client import send_whatsapp_list

load_dotenv()

async def main():
    # Let's try to get the first user to see if they have a phone number
    if users_collection is not None:
        user = users_collection.find_one({})
        if user:
            print(f"Found user: {user.get('email')} with phone: {user.get('phone')}")
            if user.get("phone"):
                print("Attempting to send WhatsApp message...")
                result = send_whatsapp_list(user["phone"], [{"name": "Test Item", "quantity": "1"}])
                print(f"Send result: {result}")
            else:
                print("User does not have a phone number set.")
        else:
            print("No users found in database.")
    else:
        print("MongoDB not connected.")

if __name__ == "__main__":
    asyncio.run(main())
