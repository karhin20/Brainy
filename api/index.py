import os
import sys
import uuid
import httpx
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Literal

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
    API_KEY: str # Unused currently, consider removal if not needed
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
    allowed_origins = [settings.FRONTEND_URL]
    logger.info(f"CORS set for PRODUCTION, allowing: {allowed_origins}")
else:
    # For development, allow localhost, your Vercel URL, and potentially '*' for local testing ease
    allowed_origins = ["http://localhost:3000", settings.FRONTEND_URL]
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
        if any(word in lower_msg for word in ["buy", "menu", "order food"]): return {"intent": "start_order"}
        if any(word in lower_msg for word in ["status", "where is my order", "order update"]): return {"intent": "check_status"}
        if any(word in lower_msg for word in ["no", "good", "moment", "all", "nothing else"]): return {"intent": "end_conversation"}
        if any(word in lower_msg for word in ["thank", "ok", "got it", "thanks"]): return {"intent": "polite_acknowledgement"}
        if any(word in lower_msg for word in ["cart", "items", "my order", "my items", "what have i selected"]): return {"intent": "show_cart"}
        return {"intent": "greet"}

    prompt = f"""
    Analyze the user's message for a grocery bot based on the context of the bot's last message. Respond ONLY with a single, minified JSON object.
    CONTEXT: The bot's last message to the user was: "{last_bot_message or 'No previous message.'}"
    User's New Message: "{user_message}"
    Your JSON output MUST contain one key, "intent", with one of these values:
    - `start_order`: User wants to start a new order or see the menu.
    - `check_status`: User is asking about an existing order's status.
    - `show_cart`: User wants to see items in their pending order.
    - `polite_acknowledgement`: User is saying "thanks", "ok", "got it".
    - `end_conversation`: User is clearly ending the conversation (e.g., "no thanks", "I am good", "that's all").
    - `greet`: A general greeting or question.
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "application/json"}}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(settings.GEMINI_API_URL, headers={"Content-Type": "application/json"}, params={"key": settings.GEMINI_API_KEY}, json=payload)
            res.raise_for_status()
            return json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
    except httpx.HTTPStatusError as e:
        logger.error(f"Gemini API HTTP Error ({e.response.status_code}) for user message '{user_message}': {e.response.text}", exc_info=True)
        # Specific handling for rate limits or invalid keys could go here
        if e.response.status_code == 429: # Too Many Requests
            return {"intent": "busy"} # Custom intent or message to user
        elif e.response.status_code == 400: # Bad Request, often from prompt issues or invalid key
             return {"intent": "error_ai"}
        return {"intent": "greet"} # Fallback to greet on general HTTP errors
    except httpx.RequestError as e:
        logger.error(f"Gemini API Network/Request Error for user message '{user_message}': {e}", exc_info=True)
        return {"intent": "greet"} # Fallback if network issue (DNS, connection, timeout)
    except json.JSONDecodeError as e:
        logger.error(f"Gemini API Response JSON Decode Error for user message '{user_message}': {e}", exc_info=True)
        return {"intent": "greet"} # Fallback if response isn't valid JSON
    except Exception as e:
        logger.error(f"Unexpected error in get_intent_with_context for user message '{user_message}': {e}", exc_info=True)
        return {"intent": "greet"}


# --- HELPER FUNCTIONS ---
def generate_order_number(): return f"ORD-{int(datetime.now(timezone.utc).timestamp())}"

def calculate_delivery_fee(lat: float, lon: float) -> float:
    """
    Calculates the delivery fee based on location.
    Currently a fixed value. THIS IS A KEY AREA FOR IMPROVEMENT
    to implement dynamic calculation (e.g., using a mapping API).
    """
    return 15.00

async def generate_paystack_payment_link(order_id: str, amount: float, user_phone: str) -> str:
    if not settings.PAYSTACK_SECRET_KEY:
        logger.warning("PAYSTACK_SECRET_KEY not set, cannot generate real payment link. Using mock link.")
        return f"{settings.FRONTEND_URL}/payment-success?mock=true&order_id={order_id}"

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
        "callback_url": f"{settings.FRONTEND_URL}/payment-success?order_id={order_id}",
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
            # Create a new session for this conversation if no active session was found/created by start_order
            # This handles cases like 'greet', 'check_status' if no ordering session is active,
            # ensuring conversation history is still captured.
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


# --- PRIMARY WEBHOOK (STATE-FIRST ARCHITECTURE) ---
@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    from_number_clean = "unknown"
    reply_message = "" # Initialize reply_message early
    user_id: Optional[str] = None # Initialize user_id for broader scope, make optional
    current_session: Optional[Dict[str, Any]] = None # To hold the active session dict
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
            logger.error(f"SUPABASE CHECK (WHATSAPP_WEBHOOK): Supabase client NOT available. Cannot process message from {from_number_clean}.")
            # user_id is not available here, so pass None
            await send_user_error_message(from_number_clean, None, "Sorry, I'm currently experiencing technical difficulties. Please try again later.")
            return JSONResponse(content={"detail": "Database unavailable"}, status_code=200)

        # 1. Find or Create User
        user_res = supabase.table("users").select("id, last_bot_message").eq("phone_number", from_number_clean).limit(1).execute()
        is_new_user = not user_res.data
        if is_new_user:
            insert_user_res = supabase.table("users").insert({"phone_number": from_number_clean, "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()}).execute()
            if insert_user_res.data:
                user = insert_user_res.data[0]
                logger.info(f"New user created: {user['id']}")
            else:
                logger.error(f"Failed to create new user {from_number_clean}: {insert_user_res.error}")
                await send_user_error_message(from_number_clean, None, "Failed to register you. Please try again.")
                return JSONResponse(content={}, status_code=200)
        else:
            user = user_res.data[0]
        user_id = user['id']
        last_bot_message = user.get('last_bot_message')
        logger.debug(f"Processing message for user {user_id} ({from_number_clean}).")

        # 2. Find/Load Active Session for this user
        session_res = supabase.table("sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if session_res.data:
            potential_session = session_res.data[0]
            try:
                expires_at = datetime.fromisoformat(potential_session['expires_at'].replace('Z', '+00:00')) # Ensure timezone-aware
                if datetime.now(timezone.utc) < expires_at:
                    current_session = potential_session
                    # Load existing conversation history
                    history_data = current_session.get('conversation_history')
                    if history_data: # history can be string (JSONB) or already parsed list
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
                    logger.debug(f"Expired session {potential_session['id']} found for user {user_id}. Not loading history.")
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing session expires_at or history for user {user_id}, session {potential_session.get('id')}: {e}. Treating session as invalid.", exc_info=True)

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
            # Handle cases where message body is empty but it's not a location (e.g., media messages without text)
            user_message_entry["message"] = f"Received message of type: {form_data.get('MessageType', 'unknown')} (no text/location)"
            user_message_entry["type"] = form_data.get('MessageType', 'unknown')
        
        current_conversation_history.append(user_message_entry)
        logger.debug(f"Appended user message to history. Current history length: {len(current_conversation_history)}")


        # --- STATE-FIRST LOGIC (HIGHEST PRIORITY) ---
        handled_by_state = False

        # 1. Check for an order awaiting delivery/pickup choice
        pending_confirm_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", ORDER_STATUS_PENDING_CONFIRMATION).order("created_at", desc=True).limit(1).execute()
        if pending_confirm_res.data:
            order = pending_confirm_res.data[0]
            if "delivery" in incoming_msg_body:
                supabase.table("orders").update({"status": ORDER_STATUS_AWAITING_LOCATION}).eq("id", order['id']).execute()
                reply_message = "Great! To calculate the delivery fee, please share your location using the WhatsApp location feature.\n\nTap the *clip icon 📎*, then choose *'Location' 📍*."
            elif "pickup" in incoming_msg_body:
                supabase.table("orders").update({"status": ORDER_STATUS_PENDING_PAYMENT, "delivery_type": "pickup"}).eq("id", order['id']).execute()
                payment_link = await generate_paystack_payment_link(order['id'], order['total_amount'], from_number_clean)
                reply_message = f"Alright, your order is set for pickup. Your total is *GHS {order['total_amount']:.2f}*. Please complete your payment here:\n\n{payment_link}"
            else:
                reply_message = f"You have an order pending confirmation. Please choose how you'd like to receive your order: *delivery* or *pickup*?"
            
            # Call centralized helper
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
            
            payment_link = await generate_paystack_payment_link(order['id'], total_with_delivery, from_number_clean)
            reply_message = f"Thank you! Your delivery fee is GHS {delivery_fee:.2f}. Your new total is *GHS {total_with_delivery:.2f}*.\n\nPlease use this link to pay:\n{payment_link}"
            
            # Call centralized helper
            await _send_bot_reply_and_update_session(from_number_clean, reply_message, user_id, current_session, current_conversation_history, "location_received_and_payment_link_sent")
            handled_by_state = True
            return JSONResponse(content={}, status_code=200)
        
        # 3. Check for an order awaiting payment
        unpaid_order_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", ORDER_STATUS_PENDING_PAYMENT).order("created_at", desc=True).limit(1).execute()
        if unpaid_order_res.data:
            order = unpaid_order_res.data[0]
            if incoming_msg_body in ["pay", "yes", "payment"]:
                total = order.get('total_with_delivery') or order.get('total_amount', 0)
                payment_link = await generate_paystack_payment_link(order['id'], total, from_number_clean)
                reply_message = f"Of course. Please use the link below to complete your payment for order {order.get('order_number')}:\n\n{payment_link}"
            elif incoming_msg_body in ["cancel", "no"]:
                supabase.table("orders").update({"status": ORDER_STATUS_CANCELLED}).eq("id", order['id']).execute()
                reply_message = f"Your order ({order.get('order_number')}) has been cancelled. Feel free to start a new one anytime!"
            else:
                reply_message = f"You have a pending order ({order.get('order_number')}) awaiting payment. Would you like to *pay* now or *cancel* the order?"
            
            # Call centralized helper
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
                # If there's an existing active session, expire it gracefully
                if current_session:
                    logger.debug(f"User {user_id} starting new order, expiring current session {current_session['id']}.")
                    supabase.table("sessions").update({"expires_at": datetime.now(timezone.utc).isoformat(), "last_intent": "superseded_by_new_order", "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", current_session['id']).execute()
                    current_session = None # Clear current_session as it's now expired/invalidated for new flow

                # Create a NEW session for the new order flow
                new_session_token = str(uuid.uuid4())
                expires_at = (datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_ORDER_EXPIRY_HOURS)).isoformat() # Use setting
                session_payload = {
                    "user_id": user_id,
                    "phone_number": from_number_clean,
                    "session_token": new_session_token,
                    "expires_at": expires_at,
                    "last_intent": intent, # Store the intent
                    "conversation_history": json.dumps(current_conversation_history), # Save current history (might be empty or contain only user's new message)
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                insert_session_res = supabase.table("sessions").insert(session_payload).execute()
                if insert_session_res.data:
                    current_session = insert_session_res.data[0] # Set the new session as current
                    logger.debug(f"New session {current_session['id']} created for user {user_id}")
                else:
                    logger.error(f"Failed to create new session for user {user_id} on start_order intent. Supabase response: {insert_session_res.error}")
                    reply_message = "Sorry, I'm having trouble starting a new order right now. Please try again in a moment."
                    # No need to return here, let it fall through to final send/save

                if current_session: # Ensure current_session exists before building menu_url
                    menu_url = f"{settings.FRONTEND_URL}?session={current_session['session_token']}" # Use token from current_session
                    reply_message = f"Great! Please use the link below to browse our menu and add items to your cart. Return to this chat after you confirm your items on the website!\n\n{menu_url}"
                else:
                    reply_message = "Sorry, I couldn't start your order. Please try again."

            elif intent == 'check_status':
                paid_order_res = supabase.table("orders").select("status, order_number").eq("user_id", user_id).eq("payment_status", "paid").order("created_at", desc=True).limit(1).execute()
                reply_message = f"Your most recent order ({paid_order_res.data[0]['order_number']}) is currently '{paid_order_res.data[0]['status']}'." if paid_order_res.data else "It looks like you don't have any active orders with us. To start one, just say 'menu'."
            
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
                    # Add current state reminder to cart summary
                    if order.get("status") == ORDER_STATUS_PENDING_CONFIRMATION:
                        reply_message += "Please select *delivery* or *pickup* to proceed with your order."
                    elif order.get("status") == ORDER_STATUS_AWAITING_LOCATION:
                        reply_message += "Please share your location for delivery to finalize your order."
                    elif order.get("status") == ORDER_STATUS_PENDING_PAYMENT:
                        reply_message += "Your order is awaiting payment. Please complete the payment to finalize."
                else:
                    reply_message = "Your cart is currently empty. To start an order, just say 'menu'."

            elif intent == 'polite_acknowledgement':
                ending_messages = [
                    "Alright, have a great day! Feel free to message me anytime you need groceries.",
                    "You're welcome!"
                ]
                if last_bot_message in ending_messages:
                    reply_message = "You're welcome!"
                else:
                    reply_message = "You're welcome! Is there anything else I can help with?"

            elif intent == 'end_conversation':
                reply_message = "Alright, have a great day! Feel free to message me anytime you need groceries."
            
            elif intent == 'busy': # Handle specific AI intent for rate limits/busy
                reply_message = "I'm a bit busy at the moment, please try again in a few minutes."

            elif intent == 'error_ai': # Handle specific AI intent for AI errors
                reply_message = "I seem to be having trouble understanding right now. Could you please rephrase, or type 'menu' to start an order?"
            
            else: # greet or fallback
                if is_new_user:
                    reply_message = "Hello and welcome to Fresh Market GH! 🌿 I'm your personal assistant for ordering fresh groceries. You can say 'menu' to start shopping, or 'status' to check an order."
                else:
                    reply_message = "Welcome back! How can I help with your groceries today? (You can say 'menu' or 'status')"
        
            # Call centralized helper for intent-based replies
            await _send_bot_reply_and_update_session(from_number_clean, reply_message, user_id, current_session, current_conversation_history, intent)
        
        return JSONResponse(content={}, status_code=200)

    except Exception as e:
        logger.error(f"Critical webhook error for {from_number_clean}: {e}", exc_info=True)
        # Use the new centralized error function
        await send_user_error_message(from_number_clean, user_id, "I'm sorry, an unexpected error occurred. Please try again.")
        return JSONResponse(content={}, status_code=200)

# --- WEB-BASED ENDPOINTS ---
@app.post("/confirm-items")
async def confirm_items(request: OrderRequest):
    if not supabase: raise HTTPException(500, "Server module unavailable")
    
    session_res = supabase.table("sessions").select("*").eq("session_token", request.session_token).limit(1).execute()
    if not session_res.data: raise HTTPException(404, "Session invalid")
    user_id, phone_number = session_res.data[0]['user_id'], session_res.data[0]['phone_number']

    # Mark the session as expired/inactive instead of deleting it, to retain conversation history.
    # Set expires_at to now, and update last_intent for admin view.
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
    
    # Use the centralized _send_bot_reply_and_update_session helper here
    # Since this is initiated from the web, there might not be a 'current_session' from webhook logic
    # but the helper will ensure a history session is created/updated for this interaction.
    # We pass None for current_session, and the function will handle creating a new one if needed.
    # We also pass the user's phone number and the reply as a "bot" message.
    # The `user_id` is guaranteed to be available from the session lookup.
    # We also explicitly create a user message entry for the 'confirm-items' trigger
    # since this endpoint is external to the main whatsapp_webhook.
    # First, append the "user" action that led to this endpoint call.
    # Then, let _send_bot_reply_and_update_session handle the bot's reply.

    # Simulate the user's action that led to /confirm-items
    # This is a conceptual entry, as the actual user message came from the web, not WhatsApp.
    # It helps to keep the conversation history coherent.
    user_action_entry = {
        "speaker": "user",
        "message": f"Confirmed items on website (session: {request.session_token})",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "web_action"
    }
    # Create a fresh history for this interaction, starting with the simulated user action
    # This ensures that the conversation history for this specific web-initiated flow
    # is tracked correctly, separate from any ongoing WhatsApp conversation session.
    # The _send_bot_reply_and_update_session will then create a new session if needed.
    web_initiated_history = [user_action_entry]

    await _send_bot_reply_and_update_session(phone_number, reply, user_id, current_session=None, current_conversation_history=web_initiated_history, intent="order_confirmation_prompt")
    
    return {"status": "order_confirmed_on_whatsapp", "order_id": new_order_id}

@app.get("/")
async def root(): return {"message": "WhatsApp MarketBot Backend is running."}