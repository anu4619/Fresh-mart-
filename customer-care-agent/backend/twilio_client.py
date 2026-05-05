import os
import logging
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        _client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("✅ Twilio Client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Twilio Client: {e}")


def _format_phone(phone: str) -> str:
    """Ensure the phone number is clean and prefixed with '+'."""
    phone = "".join([c for c in phone if c.isdigit() or c == "+"])
    
    # If the user just typed 10 digits (e.g. 9876543210), assume India (+91)
    if len(phone) == 10 and phone.isdigit():
        return "+91" + phone
        
    if not phone.startswith("+"):
        phone = "+" + phone
        
    return phone


def send_whatsapp_list(phone_number: str, items: list) -> bool:
    """Send the finalized shopping list to the user via WhatsApp."""
    if not _client or not TWILIO_WHATSAPP_NUMBER:
        logger.error("Twilio not configured. Cannot send WhatsApp message.")
        return False

    if not phone_number:
        logger.error("No phone number provided for WhatsApp message.")
        return False

    phone_number = _format_phone(phone_number)

    if not items:
        body = "🛒 *Your FreshMart Order*\n\nYour cart is empty!"
    else:
        body = "🛒 *Your FreshMart Order*\n\n"
        for idx, item in enumerate(items, 1):
            body += f"{idx}. {item['name']} ({item['quantity']})\n"
        body += "\nThank you for shopping with FreshMart! 🌟"

    try:
        message = _client.messages.create(
            from_=f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
            body=body,
            to=f"whatsapp:{phone_number}"
        )
        logger.info(f"📱 WhatsApp sent to {phone_number}. Message SID: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"❌ WhatsApp delivery failed: {e}")
        return False
