import os
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
TWILIO_API_URL = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"


async def send_whatsapp_message(to_number: str, message: str):
    """
    Sends a WhatsApp message using Twilio's API.
    """
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER]):
        logger.error("Twilio credentials are not fully configured. Cannot send message.")
        return

    # Ensure the 'to' number is in the correct format for WhatsApp
    if not to_number.startswith("whatsapp:"):
        to_number = f"whatsapp:{to_number}"
        
    # The 'from' number must also be in the correct format
    from_number = f"whatsapp:{TWILIO_WHATSAPP_NUMBER}"

    data = {
        "To": to_number,
        "From": from_number,
        "Body": message,
    }

    try:
        async with httpx.AsyncClient(auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)) as client:
            response = await client.post(TWILIO_API_URL, data=data)
            response.raise_for_status()
            logger.info(f"Message sent to {to_number}. SID: {response.json().get('sid')}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to send WhatsApp message to {to_number}. Status: {e.response.status_code}, Response: {e.response.text}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending WhatsApp message: {e}", exc_info=True) 