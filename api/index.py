import os
import sys
import uuid
import httpx
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Literal
import random
import pytz # Added: For time zone handling

from fastapi import FastAPI, Request, HTTPException, Depends, Security, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Project Imports & Logging ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from supabase_client import supabase
    logger.info("Supabase client imported successfully.")
except ImportError:
    supabase = None
    logger.error("Supabase client not found.")

try:
    from .utils import send_whatsapp_message
    send_whatsapp_message_available = True
    logger.info("send_whatsapp_message imported.")
except ImportError:
    send_whatsapp_message_available = False
    async def send_whatsapp_message(to: str, body: str): logger.error(f"WhatsApp disabled. To {to}: {body}")

try:
    from . import security, admin_router, auth_router, public_router
    logger.info("Routers and security modules imported.")
except ImportError as e:
    security, admin_router, auth_router, public_router = None, None, None, None
    logger.error(f"Failed to import local modules: {e}")

# --- Configuration & Settings ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    TWILIO_AUTH_TOKEN: str
    TWILIO_ACCOUNT_SID: str
    TWILIO_WHATSAPP_NUMBER: str
    FRONTEND_URL: str = "https://marketmenu.vercel.app"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    PAYSTACK_SECRET_KEY: Optional[str] = None
    PAYSTACK_PUBLIC_KEY: Optional[str] = None
    PAYSTACK_PAYMENT_URL: str = "https://api.paystack.co/transaction/initialize"
    # API_KEY: str # Unused currently, consider removal if not needed
    SESSION_ORDER_EXPIRY_HOURS: int = 1 # New: configurable expiry for web menu sessions
    SESSION_HISTORY_EXPIRY_DAYS: int = 7 # New: configurable expiry for general history sessions

settings = Settings()

# --- Constants, Models, FastAPI App ---
OrderStatus = Literal[
    "pending_confirmation", "awaiting_location", "awaiting_location_confirmation",
    "pending_payment", "processing", "out-for-delivery", "delivered", "cancelled", "failed"
]

# --- Constants for Order Statuses (for improved readability and maintainability) ---
ORDER_STATUS_PENDING_CONFIRMATION = "pending_confirmation"
ORDER_STATUS_AWAITING_LOCATION = "awaiting_location"
ORDER_STATUS_PENDING_PAYMENT = "pending_payment"
ORDER_STATUS_PROCESSING = "processing"
ORDER_STATUS_OUT_FOR_DELIVERY = "out-for-delivery"
ORDER_STATUS_DELIVERED = "delivered"
ORDER_STATUS_CANCELLED = "cancelled"
ORDER_STATUS_FAILED = "failed"
ORDER_STATUS_AWAITING_LOCATION_CONFIRMATION = "awaiting_location_confirmation" # If you use this status

class OrderRequest(BaseModel):
    session_token: str
    items: List[Dict]
    total_amount: float

app = FastAPI(title="WhatsApp MarketBot API (State-First)", version="3.4.0")

# --- CORS Configuration (Improved for Production) ---
if os.getenv("APP_ENV") == "production":
    allowed_origins = [settings.FRONTEND_URL, "https://preview--marketbot-command-center.lovable.app"]
    logger.info(f"CORS set for PRODUCTION, allowing: {allowed_origins}")
else:
    # For development, allow localhost, your Vercel URL, and potentially '*' for local testing ease
    allowed_origins = ["http://localhost:3000", settings.FRONTEND_URL, "https://preview--marketbot-command-center.lovable.app"]
    # Be cautious with '*' in development, but it's common for initial setup
    if os.getenv("ALLOW_ALL_CORS_DEV", "false").lower() == "true": # Use an env var to enable wider access in dev
        allowed_origins.append("*")
    logger.info(f"CORS set for DEVELOPMENT, allowing: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"], # Consider narrowing this down for production if possible, e.g., ["GET", "POST", "PUT", "PATCH", "DELETE"]
    allow_headers=["*"], # Consider narrowing this down for production if possible, e.g., ["Content-Type", "Authorization"]
)

if security and admin_router: app.include_router(admin_router.router, prefix="/admin", tags=["admin"], dependencies=[Depends(security.get_admin_user)])
if auth_router: app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
if public_router: app.include_router(public_router.router)


# --- AI BRAIN (WITH CONTEXT) ---
async def get_intent_with_context(user_message: str, last_bot_message: Optional[str] = None) -> Dict[str, Any]:
    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. Using keyword-based intent detection fallback.")
        lower_msg = user_message.lower()
        
        # Explicitly check for greetings first
        if any(word in lower_msg for word in ["hello", "hi", "hey", "sup", "good morning", "good afternoon", "good evening"]):
            return {"intent": "greet"}
        
        # New: Explicitly check for help command
        if any(word in lower_msg for word in ["help", "assist", "support", "what can you do"]):
            return {"intent": "help"}

        # Expanded keywords for better fallback matching
        if any(word in lower_msg for word in ["buy", "menu", "order food", "shop", "browse", "start new order"]): return {"intent": "start_order"}
        if any(word in lower_msg for word in ["status", "where is my order", "order update", "my order status", "track order"]): return {"intent": "check_status"}
        if any(word in lower_msg for word in ["no", "good", "moment", "all", "nothing else", "i'm done", "that's all", "bye"]): return {"intent": "end_conversation"}
        if any(word in lower_msg for word in ["thank", "ok", "got it", "thanks", "alright", "cool"]): return {"intent": "polite_acknowledgement"}
        if any(word in lower_msg for word in ["cart", "items", "my order", "my items", "what have i selected", "basket", "my selection", "what's in my cart"]): return {"intent": "show_cart"}
        
        # Fallback to a new 'unknown_input' intent if none of the above specific keywords are matched
        return {"intent": "unknown_input"} # Changed from 'greet' to 'unknown_input'

    prompt = f"""
    Analyze the user's message for a grocery bot based on the context of the bot's last message. Respond ONLY with a single, minified JSON object.
    You MUST choose one of the specified intents and NEVER invent your own. If unsure, default to 'unknown_input'.
    CONTEXT: The bot's last message to the user was: "{last_bot_message or 'No previous message.'}"
    User's New Message: "{user_message}"
    Your JSON output MUST contain one key, "intent", with one of these values:
    - `start_order`: User wants to start a new order or see the menu.
    - `check_status`: User is asking about an existing order's status.
    - `show_cart`: User wants to see items in their pending order.
    - `polite_acknowledgement`: User is saying "thanks", "ok", "got it".
    - `end_conversation`: User is clearly ending the conversation (e.g., "no thanks", "I am good", "that's all").
    - `greet`: A general greeting or question.
    - `help`: User is asking for help or what the bot can do.
    - `unknown_input`: The user's intent is unclear or not covered by other categories. This is the default if no clear intent is found.
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "application/json"}}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(settings.GEMINI_API_URL, headers={"Content-Type": "application/json"}, params={"key": settings.GEMINI_API_KEY}, json=payload)
            res.raise_for_status()
            ai_response = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
            # Ensure the AI actually returned a valid intent from our list
            valid_intents = ["start_order", "check_status", "show_cart", "polite_acknowledgement", "end_conversation", "greet", "help", "unknown_input"]
            if ai_response.get("intent") in valid_intents:
                return ai_response
            else:
                logger.warning(f"AI returned invalid intent '{ai_response.get('intent')}'. Defaulting to unknown_input.")
                return {"intent": "unknown_input"}
    except httpx.HTTPStatusError as e:
        logger.error(f"Gemini API HTTP Error ({e.response.status_code}) for user message '{user_message}': {e.response.text}", exc_info=True)
        if e.response.status_code == 429:
            return {"intent": "busy"}
        elif e.response.status_code == 400:
            return {"intent": "error_ai"}
        return {"intent": "unknown_input"} # Fallback for general HTTP errors
    except httpx.RequestError as e:
        logger.error(f"Gemini API Network/Request Error for user message '{user_message}': {e}", exc_info=True)
        return {"intent": "unknown_input"} # Fallback if network issue
    except json.JSONDecodeError as e:
        logger.error(f"Gemini API Response JSON Decode Error for user message '{user_message}': {e}", exc_info=True)
        return {"intent": "unknown_input"} # Fallback if response isn't valid JSON
    except Exception as e:
        logger.error(f"Unexpected error in get_intent_with_context for user message '{user_message}': {e}", exc_info=True)
        return {"intent": "unknown_input"} # Fallback for any other unexpected errors


# --- HELPER FUNCTIONS ---
def generate_order_number(): return f"ORD-{int(datetime.now(timezone.utc).timestamp())}"

def calculate_delivery_fee(lat: float, lon: float) -> float:
    """
    Calculates the delivery fee based on location.
    Currently a fixed value. THIS IS A KEY AREA FOR IMPROVEMENT
    to implement dynamic calculation (e.g., using a mapping API).
    """
    return 15.00

async def generate_paystack_payment_link(order_id: str, order_number: str, amount: float, user_phone: str) -> str:
    if not settings.PAYSTACK_SECRET_KEY:
        logger.warning("PAYSTACK_SECRET_KEY not set, cannot generate real payment link. Using mock link.")
        return f"{settings.FRONTEND_URL}/payment-success?mock=true&order_number={order_number}"

    headers = {
        "Authorization": f"Bearer {settings.PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    # Paystack requires an email, construct a placeholder if only phone is available
    placeholder_email = f"{''.join(filter(str.isdigit, user_phone))}@market.bot"
    unique_reference = f"{order_id}_{int(datetime.now().timestamp())}"

    payload = {
        "email": placeholder_email,
        "amount": int(amount * 100), # Amount in kobo/pesewas
        "currency": "GHS", # Assuming Ghana Cedis
        "reference": unique_reference,
        "callback_url": f"{settings.FRONTEND_URL}/payment-success?order_number={order_number}",
        "metadata": {"order_id": order_id, "phone": user_phone, "reference": unique_reference}
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(settings.PAYSTACK_PAYMENT_URL, headers=headers, json=payload)
            response.raise_for_status() # Raise an exception for 4xx/5xx responses

            data = response.json()
            if data.get("status") is True and data.get("data") and data["data"].get("authorization_url"):
                logger.info(f"Paystack payment link generated for order {order_id}: {data['data']['authorization_url']}")
                return data["data"]["authorization_url"]
            else:
                logger.error(f"Paystack API returned success=true but missing authorization_url or unexpected data for order {order_id}: {data}")
                raise ValueError("Paystack API response format error during link generation.")

    except httpx.RequestError as e:
        logger.error(f"Paystack API request error for order {order_id}: {e}", exc_info=True)
        if isinstance(e, httpx.HTTPStatusError):
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                error_detail = error_json.get('message', error_detail)
            except json.JSONDecodeError:
                pass
            raise Exception(f"Payment gateway error ({e.response.status_code}): {error_detail}") from e
        else:
            raise Exception(f"Could not connect to payment gateway: {e}. Please check your internet connection.") from e
    except Exception as e:
        logger.error(f"Unexpected error during Paystack link generation for order {order_id}: {str(e)}", exc_info=True)
        raise Exception(f"An unexpected error occurred during payment link generation.") from e

async def send_and_save_message(phone: str, message: str, user_id: str):
    """
    DEPRECATED: Use _send_bot_reply_and_update_session instead for full history tracking.
    This function only updates last_bot_message in users table.
    """
    if not send_whatsapp_message_available or not supabase: return
    try:
        await send_whatsapp_message(phone, message)
        supabase.table("users").update({"last_bot_message": message}).eq("id", user_id).execute()
    except Exception as e:
        logger.error(f"Error in send_and_save_message for user {user_id}: {e}")

async def send_user_error_message(phone_number: str, user_id: Optional[str], error_detail: str = "I'm sorry, an unexpected error occurred. Please try again."):
    """
    Sends an error message to the user via WhatsApp and logs the error.
    Handles cases where user_id might not be immediately available.
    """
    if not send_whatsapp_message_available:
        logger.error(f"WhatsApp disabled. User {phone_number} would have received error: {error_detail}")
        return

    log_user_info = f"user_id: {user_id}" if user_id else f"phone: {phone_number}"
    logger.error(f"Sending error message to {log_user_info}: {error_detail}")
    try:
        await send_whatsapp_message(phone_number, error_detail)
        # Optionally, save this error message as the last bot message for the user if needed
        # if user_id and supabase:
        #     supabase.table("users").update({"last_bot_message": error_detail[:1000]}).eq("id", user_id).execute()
    except Exception as e:
        logger.error(f"Failed to send direct error message to {phone_number}: {e}")

async def _send_bot_reply_and_update_session(
    from_number_clean: str,
    reply_message: str,
    user_id: str,
    current_session: Optional[Dict[str, Any]],
    current_conversation_history: List[Dict[str, Any]],
    intent: Optional[str] = None
) -> None:
    """
    Helper to send bot reply, add to history, and save session in a centralized manner.
    This function consolidates logic for consistent message handling and history persistence.
    """
    if not send_whatsapp_message_available or not supabase:
        logger.warning(f"WhatsApp sending or Supabase unavailable for {user_id}. Skipping bot reply and session update.")
        return

    # Add Bot's Reply to History
    bot_entry = {
        "speaker": "bot",
        "message": reply_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "text"
    }
    if intent: # Add intent if provided by the AI
        bot_entry["intent"] = intent
    current_conversation_history.append(bot_entry)
    logger.debug(f"Appended bot message to history for user {user_id}. History length: {len(current_conversation_history)}")

    # Update the user's last_bot_message in the users table
    try:
        # Also update 'updated_at' for the user
        supabase.table("users").update({"last_bot_message": reply_message[:1000], "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", user_id).execute()
        logger.debug(f"Updated last_bot_message and updated_at for user {user_id}.")
    except Exception as e:
        logger.error(f"Failed to update last_bot_message for user {user_id}: {e}", exc_info=True)

    # Send the message to WhatsApp
    try:
        await send_whatsapp_message(from_number_clean, reply_message)
        logger.debug(f"Sent WhatsApp message to {from_number_clean}.")
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message to {from_number_clean}: {e}", exc_info=True)
        # Attempt to proceed with session update even if message sending fails

    # Save the full conversation history back to the session
    try:
        if current_session:
            # Update existing session
            update_data = {
                "conversation_history": json.dumps(current_conversation_history),
                "last_intent": intent if intent else current_session.get("last_intent", "unknown"),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            supabase.table("sessions").update(update_data).eq("id", current_session['id']).execute()
            logger.debug(f"Updated existing session {current_session['id']} with new history and intent.")
        else:
            # This 'else' block will only be hit if, for some reason, current_session is None
            # even though existing_session_record might exist (e.g., if existing_session_record
            # was expired and thus not assigned to current_session).
            # With the current logic, existing_session_record is handled explicitly in start_order.
            # This part is primarily for general conversation history tracking outside of specific states.
            logger.debug(f"No active session found. Creating new session for user {user_id} to store history.")
            new_session_token_for_history = str(uuid.uuid4())
            insert_data = {
                "user_id": user_id,
                "phone_number": from_number_clean,
                "session_token": new_session_token_for_history,
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=settings.SESSION_HISTORY_EXPIRY_DAYS)).isoformat(), # Use setting
                "last_intent": intent if intent else "unstructured_conversation",
                "conversation_history": json.dumps(current_conversation_history),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            insert_res = supabase.table("sessions").insert(insert_data).execute()
            if insert_res.data:
                logger.debug(f"New history session {insert_res.data[0]['id']} created for user {user_id}.")
            else:
                logger.error(f"Failed to create new history session for user {user_id}: {insert_res.error}")
    except Exception as e:
        logger.error(f"Failed to update or create session for user {user_id}: {e}", exc_info=True)


# --- Constants for Varied Bot Responses ---
GREETING_NEW_USER_RESPONSES = [
    "Hello and welcome to Fresh Market GH! 🌿 I'm your personal assistant for ordering fresh groceries. You can say 'menu' to start shopping, or 'status' to check an order.",
    "Hi there! Welcome to Fresh Market GH! I'm here to help you with your groceries. Just say 'menu' to see what's available, or 'status' to check on an existing order.",
    "Greetings from Fresh Market GH! I'm your virtual grocery assistant. To begin shopping, simply type 'menu'. If you have an order, you can ask for its 'status'.",
    "A warm welcome to Fresh Market GH! I'm your go-to for fresh groceries. Ready to explore the 'menu' or check your order 'status'?",
    "Fresh Market GH at your service! I can help you order groceries. Start by typing 'menu' or inquire about your 'status'."
]

GREETING_RETURNING_USER_RESPONSES = [
    "Welcome back! How can I help with your groceries today? (You can say 'menu' or 'status')",
    "Good to see you again! What can I assist you with today regarding your groceries? (Try 'menu' or 'status')",
    "Hey there! Ready for more fresh groceries? Let me know if you want to see the 'menu' or check your 'status'.",
    "Nice to have you back! What's on your grocery list today? 'Menu' or 'status'?",
    "Hello again! How can I assist you with your Fresh Market GH order? 'Menu' or 'status'?"
]

# Time-based greetings
GREETING_MORNING_RESPONSES = [
    "Good morning! How can I help you with your groceries today? (Say 'menu' or 'status')",
    "Good morning! Ready to start your day with fresh groceries? (Type 'menu' or 'status')",
    "Rise and shine! Fresh Market GH is here. What can I do for you this morning? ('menu' or 'status')"
]

GREETING_AFTERNOON_RESPONSES = [
    "Good afternoon! How may I assist you with your grocery needs? ('menu' or 'status')",
    "Good afternoon! Looking for fresh groceries? Just say 'menu' or 'status'.",
    "Hello this afternoon! How can I help you with your order? ('menu' or 'status')"
]

GREETING_EVENING_RESPONSES = [
    "Good evening! What can I help you with before the day ends? ('menu' or 'status')",
    "Good evening! Ready for your dinner groceries? Just type 'menu' or 'status'.",
    "Hello this evening! How can I make your grocery shopping easier? ('menu' or 'status')"
]

MENU_LINK_REMINDERS = [
    "Please select your items from our online menu using this link: {menu_url}\n\nOnce you've confirmed your items on the website, return to this chat!",
    "Your cart is ready on the web menu! Use this link to continue: {menu_url}\n\nRemember to come back here after confirming your choices!",
    "Just a friendly reminder to complete your order through our menu link: {menu_url}\n\nLet me know when you're done there!",
    "To pick your items, click here: {menu_url}\n\nDon't forget to come back to WhatsApp to confirm your order once you've made your selections!",
    "Ready to browse? Here's your personalized menu link: {menu_url}\n\nOnce you've added everything, just head back to our chat."
]

DELIVERY_PICKUP_PROMPTS = [
    "You have an order pending confirmation. Please choose how you'd like to receive your order: *delivery* or *pickup*?",
    "Your order is awaiting your decision: *delivery* or *pickup*? Please let me know your preference.",
    "How would you like to get your order? Reply with *delivery* or *pickup*.",
    "To proceed with your order, tell me: *delivery* or *pickup*?",
    "Are you opting for *delivery* or *pickup* for your current order?"
]

LOCATION_REQUEST_PROMPTS = [
    "Great! To calculate the delivery fee, please share your location using the WhatsApp location feature.\n\nTap the *clip icon 📎*, then choose *'Location' 📍*.",
    "Alright! I'll need your location to figure out the delivery charge. Could you please share it via WhatsApp's location feature? (It's under the *clip icon 📎*).",
    "Perfect! To finalize delivery, please send me your location. Just hit the *clip icon 📎* and select 'Location' 📍.",
    "For delivery, I need your location! Please use the 'Location' feature (the *clip icon 📎*) in WhatsApp.",
    "Kindly share your location so I can calculate delivery. Find the 'Location' option under the *clip icon 📎*."
]

PAYMENT_LINK_REMINDERS = [
    "Of course. Please use the link below to complete your payment for order {order_number}:\n\n{payment_link}",
    "Here's your payment link for order {order_number}. Please complete the transaction to finalize your order: {payment_link}",
    "Ready to pay for order {order_number}? Use this secure link: {payment_link}",
    "To complete order {order_number}, simply click on this payment link: {payment_link}",
    "Your payment link for {order_number} is ready: {payment_link}\n\nPlease proceed with the payment."
]

ORDER_CANCELLED_CONFIRMATIONS = [
    "Your order ({order_number}) has been cancelled. Feel free to start a new one anytime!",
    "Order {order_number} has been successfully cancelled. You can begin a fresh order whenever you like!",
    "Confirmed: Order {order_number} is no longer active. Let me know if you change your mind and want to order again!",
    "The cancellation for order {order_number} is complete. Feel free to browse our menu again!",
    "We've cancelled order {order_number} for you. Just say 'menu' to start fresh!"
]

PENDING_PAYMENT_PROMPTS = [
    "You have a pending order ({order_number}) awaiting payment. Would you like to *pay* now or *cancel* the order?",
    "Order {order_number} is waiting for payment. Should I send the payment link again, or would you like to *cancel*?",
    "Your order {order_number} needs payment. Please *pay* or *cancel*.",
    "Just a reminder for order {order_number}: it's awaiting payment. *Pay* or *cancel*?",
    "Action needed for order {order_number}: complete your *payment* or *cancel*."
]

CART_EMPTY_RESPONSES = [
    "Your cart is currently empty. To start an order, just say 'menu'.",
    "Nothing in your cart right now! Type 'menu' to start adding items.",
    "It looks like your cart is empty. How about we start a new order? Just say 'menu'!",
    "Your shopping cart is empty. Would you like to view our 'menu' and fill it up?",
    "No items in your cart. Time to go shopping! Just type 'menu'."
]

POLITE_ACKNOWLEDGEMENT_RESPONSES_GENERAL = [
    "You're welcome! Is there anything else I can help with?",
    "No problem at all! Anything else I can assist you with today?",
    "My pleasure! Let me know if you need anything else.",
    "Glad to help! Do you have any other questions or needs?",
    "You got it! What else can I do for you?"
]

POLITE_ACKNOWLEDGEMENT_RESPONSES_ENDING = [
    "Alright, have a great day! Feel free to message me anytime you need groceries.",
    "You're welcome!", # Sometimes a simple 'You're welcome!' is best
    "Glad I could help! Enjoy the rest of your day.",
    "Perfect! Have a wonderful day, and I'm here if you need me.",
    "You're all set! Take care and don't hesitate to reach out."
]

END_CONVERSATION_RESPONSES = [
    "Alright, have a great day! Feel free to message me anytime you need groceries.",
    "Understood! Have a wonderful day. Don't hesitate to reach out if you need anything.",
    "Goodbye! Looking forward to helping you with your next order.",
    "Farewell! Take care, and I'll be here for your next grocery run.",
    "Okay, chat soon! Wishing you a great day."
]

AI_BUSY_RESPONSES = [
    "I'm a bit busy at the moment, please try again in a few minutes.",
    "My systems are a little overloaded right now. Could you please wait a moment and try again?",
    "Apologies, I'm experiencing high traffic. Please retry your request shortly.",
    "Currently facing heavy load, please bear with me and try again in a bit.",
    "I'm a bit tied up. Can you give me a few moments and try your request again?"
]

AI_ERROR_RESPONSES = [
    "I seem to be having trouble understanding right now. Could you please rephrase, or type 'menu' to start an order?",
    "My apologies, I'm having a little difficulty processing your request. Could you phrase it differently, or type 'menu'?",
    "I'm sorry, I didn't quite get that. Perhaps try rephrasing, or simply say 'menu' to begin an order.",
    "Something went wrong on my end. Please try rephrasing your message, or type 'menu' to get started.",
    "I'm experiencing a temporary glitch. Please try your message again, or use 'menu' to see our products."
]

# New: Help Responses
HELP_RESPONSES = [
    "I can help you order groceries, check your order status, or show your cart. Just type 'menu', 'status', 'cart', or 'cancel'.",
    "Need assistance? I can guide you through ordering by saying 'menu', check your 'status', or show your 'cart'.",
    "How can I assist you? You can say 'menu' to start an order, 'status' to check an existing one, or 'cart' to review your items, or 'cancel' an order."
]

UNKNOWN_INPUT_RESPONSES = [
    "I'm not sure I understand. Could you try rephrasing your request, or type 'menu' to start an order?",
    "Hmm, I didn't quite get that. Try asking in a different way, or say 'menu' to see our products.",
    "I'm still learning! Could you simplify your request, or type 'menu' to browse?",
    "Apologies, I'm having trouble understanding. Maybe try 'menu' or 'status' to get started."
]

# --- HELPER FUNCTIONS ---
def get_time_of_day_greeting():
    """
    Returns a time-based greeting based on the current GMT time.
    """
    gmt_tz = pytz.timezone('GMT')
    current_gmt_time = datetime.now(gmt_tz).time()

    if current_gmt_time >= datetime.strptime('05:00', '%H:%M').time() and \
       current_gmt_time < datetime.strptime('12:00', '%H:%M').time():
        return random.choice(GREETING_MORNING_RESPONSES)
    elif current_gmt_time >= datetime.strptime('12:00', '%H:%M').time() and \
         current_gmt_time < datetime.strptime('17:00', '%H:%M').time():
        return random.choice(GREETING_AFTERNOON_RESPONSES)
    else:
        return random.choice(GREETING_EVENING_RESPONSES)

# --- PRIMARY WEBHOOK (STATE-FIRST ARCHITECTURE) ---
@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    from_number_clean = "unknown"
    reply_message = "" # Initialize reply_message early
    user_id: Optional[str] = None # Initialize user_id for broader scope, make optional
    current_session: Optional[Dict[str, Any]] = None # To hold the currently active session (not expired)
    current_conversation_history: List[Dict[str, Any]] = [] # To build history

    try:
        form_data = await request.form()
        from_number = form_data.get("From")
        incoming_msg_body = form_data.get("Body", "").strip() # Keep original case for Gemini, lower for keyword matching
        is_location_message = "Latitude" in form_data

        if not from_number or (not incoming_msg_body and not is_location_message):
            logger.debug("Received empty or invalid message, returning 200 OK.")
            return JSONResponse(content={}, status_code=200)

        from_number_clean = from_number.replace("whatsapp:", "")

        if not supabase:
            logger.critical(f"SUPABASE_UNAVAILABLE: Supabase client NOT initialized. Cannot process message from {from_number_clean}.")
            await send_user_error_message(from_number_clean, None, "I'm sorry, I'm experiencing a temporary technical issue. Please try again in a few moments.")
            return JSONResponse(content={"detail": "Database unavailable"}, status_code=200) # Still 200 for Twilio to not retry

        # 1. Find or Create User
        try:
            user_res = supabase.table("users").select("id, last_bot_message").eq("phone_number", from_number_clean).limit(1).execute()
            is_new_user = not user_res.data
            if is_new_user:
                insert_user_res = supabase.table("users").insert({"phone_number": from_number_clean, "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()}).execute()
                if insert_user_res.data:
                    user = insert_user_res.data[0]
                    logger.info(f"New user created: {user['id']}")
                else:
                    logger.error(f"Failed to create new user {from_number_clean}: {insert_user_res.error}")
                    await send_user_error_message(from_number_clean, None, "I couldn't register your number. Please try sending a message again.")
                    return JSONResponse(content={}, status_code=200)
            else:
                user = user_res.data[0]
        except Exception as db_op_e:
            logger.error(f"Database operation failed for user lookup/creation ({from_number_clean}): {db_op_e}", exc_info=True)
            await send_user_error_message(from_number_clean, None, "I'm having trouble accessing my records right now. Please try again shortly.")
            return JSONResponse(content={}, status_code=200)
        
        user_id = user['id'] # Set user_id here as it's now guaranteed
        last_bot_message = user.get('last_bot_message')
        logger.debug(f"Processing message for user {user_id} ({from_number_clean}).")

        # 2. Find/Load ANY Session for this user and determine if it's currently active
        existing_session_record: Optional[Dict[str, Any]] = None
        session_res = supabase.table("sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if session_res.data:
            existing_session_record = session_res.data[0]
            try:
                expires_at = datetime.fromisoformat(existing_session_record['expires_at'].replace('Z', '+00:00')) # Ensure timezone-aware
                if datetime.now(timezone.utc) < expires_at:
                    current_session = existing_session_record # This is the active one for conversation continuity
                    # Load existing conversation history
                    history_data = current_session.get('conversation_history')
                    if history_data:
                        if isinstance(history_data, str):
                            try:
                                current_conversation_history = json.loads(history_data)
                                logger.debug(f"Loaded existing conversation history from string for session {current_session['id']}.")
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse conversation_history JSON for session {current_session.get('id')}. Initializing empty.", exc_info=True)
                                current_conversation_history = []
                        elif isinstance(history_data, list):
                            current_conversation_history = history_data
                            logger.debug(f"Loaded existing conversation history as list for session {current_session['id']}.")
                else:
                    logger.debug(f"Expired session {existing_session_record['id']} found for user {user_id}. Not loading history as active session.")
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing session expires_at or history for user {user_id}, session {existing_session_record.get('id')}: {e}. Treating session as invalid.", exc_info=True)

        # 3. Add Incoming User Message to History
        user_message_entry: Dict[str, Any] = {
            "speaker": "user",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if incoming_msg_body:
            user_message_entry["message"] = incoming_msg_body
            user_message_entry["type"] = "text"
        elif is_location_message:
            lat, lon = float(form_data["Latitude"]), float(form_data["Longitude"])
            user_message_entry["message"] = f"Shared location: {lat},{lon}"
            user_message_entry["type"] = "location"
        else:
            user_message_entry["message"] = f"Received message of type: {form_data.get('MessageType', 'unknown')} (no text/location)"
            user_message_entry["type"] = form_data.get('MessageType', 'unknown')
        
        current_conversation_history.append(user_message_entry)
        logger.debug(f"Appended user message to history. Current history length: {len(current_conversation_history)}")

        # --- STATE-FIRST LOGIC (HIGHEST PRIORITY) ---
        handled_by_state = False
        negation_keywords = ["not", "don't", "dont", "no"] # Defined here for convenience

        # Check if there's an active session related to starting an order
        # and if the user's message is a new attempt to order (e.g., "I want mango")
        if current_session and current_session.get("last_intent") == "start_order":
            # If the user is trying to order directly from chat again, remind them about the link
            menu_url = f"{settings.FRONTEND_URL}?session={current_session['session_token']}"
            reply_message = random.choice(MENU_LINK_REMINDERS).format(menu_url=menu_url)
            await _send_bot_reply_and_update_session(from_number_clean, reply_message, user_id, current_session, current_conversation_history, "remind_menu_link")
            handled_by_state = True
            return JSONResponse(content={}, status_code=200)

        # 1. Check for an order awaiting delivery/pickup choice
        pending_confirm_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", ORDER_STATUS_PENDING_CONFIRMATION).order("created_at", desc=True).limit(1).execute()
        if pending_confirm_res.data:
            order = pending_confirm_res.data[0]
            lower_incoming_msg_body = incoming_msg_body.lower()
            
            if "delivery" in lower_incoming_msg_body and not any(neg in lower_incoming_msg_body for neg in negation_keywords):
                supabase.table("orders").update({"status": ORDER_STATUS_AWAITING_LOCATION}).eq("id", order['id']).execute()
                reply_message = random.choice(LOCATION_REQUEST_PROMPTS)
            elif "pickup" in lower_incoming_msg_body and not any(neg in lower_incoming_msg_body for neg in negation_keywords):
                supabase.table("orders").update({"status": ORDER_STATUS_PENDING_PAYMENT, "delivery_type": "pickup"}).eq("id", order['id']).execute()
                payment_link = await generate_paystack_payment_link(order['id'], order['order_number'], order['total_amount'], from_number_clean)
                reply_message = f"Alright, your order is set for pickup. Your total is *GHS {order['total_amount']:.2f}*. Please complete your payment here:\n\n{payment_link}"
            else:
                reply_message = random.choice(DELIVERY_PICKUP_PROMPTS)
            
            await _send_bot_reply_and_update_session(from_number_clean, reply_message, user_id, current_session, current_conversation_history, "delivery_or_pickup_prompt")
            handled_by_state = True
            return JSONResponse(content={}, status_code=200)

        # 2. Check for an order awaiting a location message
        awaiting_loc_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", ORDER_STATUS_AWAITING_LOCATION).order("created_at", desc=True).limit(1).execute()
        if awaiting_loc_res.data and is_location_message:
            order = awaiting_loc_res.data[0]
            lat, lon = float(form_data["Latitude"]), float(form_data["Longitude"])
            delivery_fee = calculate_delivery_fee(lat, lon)
            total_with_delivery = order['total_amount'] + delivery_fee
            
            update_data = {"status": ORDER_STATUS_PENDING_PAYMENT, "delivery_fee": delivery_fee, "total_with_delivery": total_with_delivery, "delivery_location_lat": lat, "delivery_location_lon": lon}
            supabase.table("orders").update(update_data).eq("id", order['id']).execute()
            
            payment_link = await generate_paystack_payment_link(order['id'], order['order_number'], total_with_delivery, from_number_clean)
            reply_message = f"Thank you! Your delivery fee is GHS {delivery_fee:.2f}. Your new total is *GHS {total_with_delivery:.2f}*.\n\nPlease use this link to pay:\n{payment_link}"
            
            await _send_bot_reply_and_update_session(from_number_clean, reply_message, user_id, current_session, current_conversation_history, "location_received_and_payment_link_sent")
            handled_by_state = True
            return JSONResponse(content={}, status_code=200)
        
        # 3. Check for an order awaiting payment
        unpaid_order_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", ORDER_STATUS_PENDING_PAYMENT).order("created_at", desc=True).limit(1).execute()
        if unpaid_order_res.data:
            order = unpaid_order_res.data[0]
            lower_incoming_msg_body = incoming_msg_body.lower()
            
            if "cancel" in lower_incoming_msg_body or "no" in lower_incoming_msg_body:
                supabase.table("orders").update({"status": ORDER_STATUS_CANCELLED}).eq("id", order['id']).execute()
                if current_session:
                    try:
                        supabase.table("sessions").update({"expires_at": datetime.now(timezone.utc).isoformat(), "last_intent": "order_cancelled", "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", current_session['id']).execute()
                        logger.debug(f"Session {current_session['id']} explicitly expired due to order cancellation.")
                    except Exception as e:
                        logger.error(f"Error expiring session {current_session['id']} after order cancellation: {e}", exc_info=True)
                reply_message = random.choice(ORDER_CANCELLED_CONFIRMATIONS).format(order_number=order.get('order_number'))
            elif any(keyword in lower_incoming_msg_body for keyword in ["pay", "yes", "payment"]) and \
                 not any(negation in lower_incoming_msg_body for negation in negation_keywords):
                total = order.get('total_with_delivery') or order.get('total_amount', 0)
                payment_link = await generate_paystack_payment_link(order['id'], order['order_number'], total, from_number_clean)
                reply_message = random.choice(PAYMENT_LINK_REMINDERS).format(order_number=order.get('order_number'), payment_link=payment_link)
            else:
                reply_message = random.choice(PENDING_PAYMENT_PROMPTS).format(order_number=order.get('order_number'))
            
            await _send_bot_reply_and_update_session(from_number_clean, reply_message, user_id, current_session, current_conversation_history, "payment_prompt")
            handled_by_state = True
            return JSONResponse(content={}, status_code=200)

        # --- INTENT-BASED LOGIC (IF NO PRIORITY STATES ARE ACTIVE) ---
        ai_result = await get_intent_with_context(incoming_msg_body, last_bot_message)
        intent = ai_result.get("intent")
        logger.debug(f"AI determined intent: {intent}")
        
        # Only proceed if no state-based response was generated by the blocks above
        if not handled_by_state:
            if intent == 'start_order':
                new_session_token = str(uuid.uuid4())
                expires_at = (datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_ORDER_EXPIRY_HOURS)).isoformat()
                
                session_payload_data = {
                    "phone_number": from_number_clean,
                    "session_token": new_session_token,
                    "expires_at": expires_at,
                    "last_intent": intent,
                    "conversation_history": json.dumps(current_conversation_history),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }

                if existing_session_record:
                    logger.debug(f"User {user_id} starting new order. Updating existing session record for user_id.")
                    update_res = supabase.table("sessions").update(session_payload_data).eq("user_id", user_id).execute()
                    if update_res.data:
                        current_session = update_res.data[0]
                        logger.debug(f"Existing session record for user {user_id} updated with new order session details.")
                    else:
                        logger.error(f"Failed to update existing session record for user {user_id} on start_order intent. Supabase response: {update_res.error}")
                        reply_message = "Sorry, I'm having trouble starting a new order right now. Please try again in a moment."
                        current_session = None
                else:
                    logger.debug(f"No session record found for user {user_id}. Creating first session for start_order intent.")
                    session_payload_data["user_id"] = user_id
                    session_payload_data["created_at"] = datetime.now(timezone.utc).isoformat()
                    insert_res = supabase.table("sessions").insert(session_payload_data).execute()
                    if insert_res.data:
                        current_session = insert_res.data[0]
                        logger.debug(f"New session {current_session['id']} created for user {user_id}.")
                    else:
                        logger.error(f"Failed to create new session for user {user_id} on start_order intent. Supabase response: {insert_res.error}")
                        reply_message = "Sorry, I'm having trouble starting a new order right now. Please try again in a moment."
                        current_session = None

                if current_session:
                    menu_url = f"{settings.FRONTEND_URL}?session={current_session['session_token']}"
                    reply_message = random.choice(MENU_LINK_REMINDERS).format(menu_url=menu_url)
                else:
                    reply_message = "Sorry, I couldn't start your order. Please try again."

            elif intent == 'check_status':
                paid_order_res = supabase.table("orders").select("status, order_number").eq("user_id", user_id).eq("payment_status", "paid").order("created_at", desc=True).limit(1).execute()
                reply_message = f"Your most recent order ({paid_order_res.data[0]['order_number']}) is currently '{paid_order_res.data[0]['status']}'." if paid_order_res.data else random.choice(CART_EMPTY_RESPONSES)
            
            elif intent == 'show_cart':
                pending_order_res = supabase.table("orders").select("items_json, total_amount, status").eq("user_id", user_id).in_("status", [ORDER_STATUS_PENDING_CONFIRMATION, ORDER_STATUS_AWAITING_LOCATION, ORDER_STATUS_PENDING_PAYMENT]).order("created_at", desc=True).limit(1).execute()
                if pending_order_res.data and pending_order_res.data[0].get("items_json"):
                    order = pending_order_res.data[0]
                    items, total_amount = order["items_json"], order["total_amount"]
                    product_ids = [item.get("product_id") for item in items if item.get("product_id")]
                    product_details_map = {p["id"]: p for p in supabase.table("products").select("id, name, price").in_("id", product_ids).execute().data} if product_ids else {}
                    
                    cart_summary_items = []
                    for item in items:
                        product_id = item.get("product_id", "N/A")
                        product_info = product_details_map.get(product_id, {"name": f"Product ID: {product_id}", "price": 0.0})
                        product_name = product_info["name"]
                        product_price = product_info["price"]
                        quantity = item.get("quantity", 1)
                        item_total = product_price * quantity
                        cart_summary_items.append(f'- {product_name} x {quantity} (GHS {item_total:.2f})')

                    reply_message = f"🛒 *Your Current Cart:*\n" + "\n".join(cart_summary_items) + f"\n\n*Total: GHS {total_amount:.2f}*\n\n"
                    if order.get("status") == ORDER_STATUS_PENDING_CONFIRMATION:
                        reply_message += "Please select *delivery* or *pickup* to proceed with your order."
                    elif order.get("status") == ORDER_STATUS_AWAITING_LOCATION:
                        reply_message += "Please share your location for delivery to finalize your order."
                    elif order.get("status") == ORDER_STATUS_PENDING_PAYMENT:
                        reply_message += "Your order is awaiting payment. Please complete the payment to finalize."
                else:
                    reply_message = random.choice(CART_EMPTY_RESPONSES)

            elif intent == 'polite_acknowledgement':
                if last_bot_message and any(msg in last_bot_message for msg in END_CONVERSATION_RESPONSES):
                    reply_message = random.choice(POLITE_ACKNOWLEDGEMENT_RESPONSES_ENDING)
                else:
                    reply_message = random.choice(POLITE_ACKNOWLEDGEMENT_RESPONSES_GENERAL)

            elif intent == 'end_conversation':
                reply_message = random.choice(END_CONVERSATION_RESPONSES)
            
            elif intent == 'help': # New: Handle 'help' intent
                reply_message = random.choice(HELP_RESPONSES)

            elif intent == 'busy':
                reply_message = random.choice(AI_BUSY_RESPONSES)

            elif intent == 'error_ai':
                reply_message = random.choice(AI_ERROR_RESPONSES)
            
            elif intent == 'unknown_input': # New: Handle unknown input
                reply_message = random.choice(UNKNOWN_INPUT_RESPONSES)
            
            else: # Fallback to greet if none of the above, should rarely hit now
                if is_new_user:
                    reply_message = get_time_of_day_greeting()
                else:
                    reply_message = random.choice(GREETING_RETURNING_USER_RESPONSES)
        
            await _send_bot_reply_and_update_session(from_number_clean, reply_message, user_id, current_session, current_conversation_history, intent)
        
        return JSONResponse(content={}, status_code=200)

    except Exception as e:
        logger.error(f"Unhandled critical webhook error for {from_number_clean}: {e}", exc_info=True)
        await send_user_error_message(from_number_clean, user_id, "I'm sorry, an unexpected error occurred. Please try again.")
        return JSONResponse(content={}, status_code=200)

# --- WEB-BASED ENDPOINTS ---
@app.post("/confirm-items")
async def confirm_items(request: OrderRequest):
    if not supabase: raise HTTPException(500, "Server module unavailable")
    
    session_res = supabase.table("sessions").select("*").eq("session_token", request.session_token).limit(1).execute()
    if not session_res.data: raise HTTPException(404, "Session invalid")
    user_id, phone_number = session_res.data[0]['user_id'], session_res.data[0]['phone_number']

    supabase.table("sessions").update({"expires_at": datetime.now(timezone.utc).isoformat(), "last_intent": "order_confirmed_web", "updated_at": datetime.now(timezone.utc).isoformat()}).eq("session_token", request.session_token).execute()
    logger.debug(f"Session {request.session_token} marked as expired (order_confirmed_web).")

    product_ids = [item.get("product_id") for item in request.items if item.get("product_id")]
    product_details_map = {p["id"]: p for p in supabase.table("products").select("id, name").in_("id", product_ids).execute().data} if product_ids else {}
    
    ordered_items_list = [f'- {product_details_map.get(item["product_id"], {}).get("name", f"ID: {item.get("product_id")}")} x {item["quantity"]}' for item in request.items]

    order_data = {"user_id": user_id, "items_json": [item for item in request.items], "total_amount": request.total_amount, "status": ORDER_STATUS_PENDING_CONFIRMATION, "payment_status": "unpaid", "order_number": generate_order_number(), "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()}
    order_res = supabase.table("orders").insert(order_data).execute()
    if not order_res.data:
        logger.error(f"Failed to create order for user {user_id}: {order_res.error}")
        raise HTTPException(500, "Could not create order")
    
    new_order_id = order_res.data[0]['id']
    logger.info(f"Order {new_order_id} created for user {user_id}.")

    reply = (f"Thank you for confirming your items!\n\n*Your Order:*\n" + "\n".join(ordered_items_list) +
             f"\n\nSubtotal: *GHS {request.total_amount:.2f}*\n\nTo proceed, would you like *delivery* or will you *pickup* the order yourself?")
    
    user_action_entry = {
        "speaker": "user",
        "message": f"Confirmed items on website (session: {request.session_token})",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "web_action"
    }
    web_initiated_history = [user_action_entry]

    await _send_bot_reply_and_update_session(phone_number, reply, user_id, current_session=None, current_conversation_history=web_initiated_history, intent="order_confirmation_prompt")
    
    return {"status": "order_confirmed_on_whatsapp", "order_id": new_order_id}

@app.get("/")
async def root(): return {"message": "WhatsApp MarketBot Backend is running."}

@app.post("/paystack-webhook")
async def paystack_webhook(request: Request):
    if not settings.PAYSTACK_SECRET_KEY:
        logger.error("PAYSTACK_SECRET_KEY is not set. Cannot process Paystack webhook.")
        raise HTTPException(status_code=500, detail="Server not configured for Paystack webhooks.")

    body = await request.body()
    signature = request.headers.get("x-paystack-signature")

    if not signature:
        logger.warning("Paystack webhook received without signature.")
        raise HTTPException(status_code=400, detail="Missing X-Paystack-Signature header.")

    import hmac
    import hashlib
    try:
        hash_object = hmac.new(settings.PAYSTACK_SECRET_KEY.encode('utf-8'), body, hashlib.sha512)
        expected_signature = hash_object.hexdigest()
    except Exception as e:
        logger.error(f"Error generating HMAC signature for Paystack webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during signature verification.")

    if not hmac.compare_digest(expected_signature, signature):
        logger.warning(f"Invalid Paystack webhook signature. Received: {signature}, Expected: {expected_signature}")
        raise HTTPException(status_code=400, detail="Invalid X-Paystack-Signature.")

    event = json.loads(body)
    logger.info(f"Received Paystack webhook event: {event.get('event')}")

    if event.get("event") == "charge.success":
        data = event.get("data", {})
        reference = data.get("reference")
        status = data.get("status")
        amount = data.get("amount") # Amount in kobo/pesewas
        currency = data.get("currency")
        
        logger.info(f"Paystack charge.success event received for reference: {reference}, status: {status}, amount: {amount/100:.2f} {currency}")

        if status == "success":
            order_id = data.get("metadata", {}).get("order_id")
            if not order_id and reference:
                try:
                    order_id = reference.split('_')[0]
                except IndexError:
                    logger.warning(f"Could not parse order_id from reference: {reference}")
            
            if not order_id:
                logger.error(f"Paystack webhook: Could not determine order_id for successful transaction reference: {reference}. Event data: {event}")
                return JSONResponse(content={"message": "Order ID not found in webhook data."}, status_code=400)

            verification_url = f"https://api.paystack.co/transaction/verify/{reference}"
            headers = {"Authorization": f"Bearer {settings.PAYSTACK_SECRET_KEY}"}
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    verify_res = await client.get(verification_url, headers=headers)
                    verify_res.raise_for_status()
                    verify_data = verify_res.json()
                    
                    if verify_data.get("status") is True and verify_data.get("data", {}).get("status") == "success":
                        logger.info(f"Paystack transaction {reference} verified successfully.")
                        try:
                            order_update_res = supabase.table("orders").update({"payment_status": "paid", "status": ORDER_STATUS_PROCESSING, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", order_id).execute()
                            if order_update_res.data:
                                logger.info(f"Order {order_id} status updated to 'processing' (paid) after Paystack webhook.")
                                user_res = supabase.table("users").select("phone_number").eq("id", order_update_res.data[0]['user_id']).single().execute()
                                if user_res.data:
                                    phone_number = user_res.data['phone_number']
                                    await send_whatsapp_message(phone_number, f"🎉 Your payment for order {order_update_res.data[0].get('order_number', order_id)} has been confirmed! Your order is now processing.")
                                    logger.info(f"Sent payment confirmation to user {phone_number} for order {order_id}.")
                                else:
                                    logger.error(f"Could not find user phone for order {order_id} to send payment confirmation.")
                            else:
                                logger.error(f"Failed to update order {order_id} in Supabase after successful Paystack webhook: {order_update_res.error}")
                        except Exception as db_e:
                            logger.error(f"Database error updating order {order_id} after Paystack webhook: {db_e}", exc_info=True)
                    else:
                        logger.warning(f"Paystack transaction {reference} verification failed. Verify data: {verify_data}")
            except httpx.RequestError as verify_e:
                logger.error(f"Error verifying Paystack transaction {reference}: {verify_e}", exc_info=True)
            except json.JSONDecodeError as json_e:
                logger.error(f"JSON decode error from Paystack verification for reference {reference}: {json_e}", exc_info=True)
        else:
            logger.info(f"Paystack webhook received charge.success with status '{status}' for reference {reference}. No order update performed.")
    
    return JSONResponse(content={"message": "Webhook received"}, status_code=200)