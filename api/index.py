
import os
import sys
import uuid
import httpx
import json
import logging
import hmac
import hashlib
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Literal

from fastapi import FastAPI, Request, HTTPException, Depends, Security, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Project Specific Imports ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from supabase_client import supabase
    if supabase:
        logger.info("Supabase client initialized successfully.")
    else:
         logger.warning("Supabase client did not initialize correctly.")
except ImportError:
    supabase = None
    logger.error("Supabase client not found or failed to import. Database operations will be unavailable.")
except Exception as e:
    supabase = None
    logger.error(f"Unexpected error importing Supabase client: {e}", exc_info=True)


# Import routers and security module
try:
    from . import admin_router, security, auth_router, public_router
    from .utils import send_whatsapp_message
    send_whatsapp_message_available = True
except ImportError as e:
    logger.error(f"Failed to import internal modules (routers, security, utils): {e}", exc_info=True)
    admin_router, security, auth_router, public_router = None, None, None, None
    send_whatsapp_message_available = False
    async def send_whatsapp_message(to: str, body: str):
        logger.error(f"send_whatsapp_message utility is not available. Tried to send to {to}: {body}")


# --- Configuration Management using Pydantic Settings ---
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
    API_KEY: str


settings = Settings()
logger.info("Settings loaded.")


# --- Constants for Order Statuses and Payment Statuses ---
OrderStatus = Literal[
    "pending_confirmation",
    "awaiting_location",
    "awaiting_location_confirmation",
    "pending_payment",
    "processing",
    "out-for-delivery",
    "delivered",
    "cancelled",
    "failed"
]

PaymentStatus = Literal[
    "unpaid",
    "paid",
    "partially_paid",
    "cancelled",
    "failed"
]

class DefaultStatus:
    ORDER_PENDING_CONFIRMATION: OrderStatus = "pending_confirmation"
    ORDER_AWAITing_LOCATION: OrderStatus = "awaiting_location"
    ORDER_AWAITING_LOCATION_CONFIRMATION: OrderStatus = "awaiting_location_confirmation"
    ORDER_PENDING_PAYMENT: OrderStatus = "pending_payment"
    ORDER_PROCESSING: OrderStatus = "processing"
    ORDER_OUT_FOR_DELIVERY: OrderStatus = "out-for-delivery"
    ORDER_DELIVERED: OrderStatus = "delivered"
    ORDER_CANCELLED: OrderStatus = "cancelled"
    ORDER_FAILED: OrderStatus = "failed"

    PAYMENT_UNPAID: PaymentStatus = "unpaid"
    PAYMENT_PAID: PaymentStatus = "paid"
    PAYMENT_PARTIALLY_PAID: PaymentStatus = "partially_paid"
    PAYMENT_CANCELLED: PaymentStatus = "cancelled"
    PAYMENT_FAILED: PaymentStatus = "failed"


# --- FastAPI App Initialization ---
app = FastAPI(
    title="WhatsApp MarketBot API",
    description="API for WhatsApp-based food market ordering system",
    version="1.0.0"
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        request_details = f"{request.method} {request.url.path}"
        logger.error(f"Unhandled server error during request {request_details}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."}
        )


# --- Include Routers ---
if admin_router:
    app.include_router(admin_router.router, prefix="/admin", tags=["admin"])
if auth_router:
    app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
if public_router:
    app.include_router(public_router.router)


# --- Pydantic Models ---
class OrderItem(BaseModel):
    product_id: str = Field(..., description="Unique identifier for the product")
    quantity: int = Field(gt=0, description="Quantity of the product, must be greater than 0")

class OrderRequest(BaseModel):
    session_token: str = Field(..., description="Unique token identifying the user's web session")
    items: List[OrderItem] = Field(..., description="List of items in the order")
    total_amount: float = Field(gt=0, description="Total amount of the items before delivery fee")

class DeliveryStatusUpdate(BaseModel):
    order_id: str = Field(..., description="Unique identifier for the order")
    status: OrderStatus = Field(..., description="New status for the delivery")


# --- Helper Functions ---

@app.get("/")
async def root():
    """Root endpoint to check if the service is running."""
    return {"message": "WhatsApp MarketBot Backend is running."}

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    db_status = "ok" if supabase else "unavailable"
    # Add more robust external service checks here if needed
    return {
        "status": "healthy",
        "database": db_status,
        "external_services": {}, # Placeholder
        "timestamp": datetime.now().isoformat()
    }


# Modified: This function now just calls the AI and returns the raw result or raises specific errors
async def call_gemini_api(message: str) -> Dict[str, Any]:
    """
    Call Gemini API to get raw intent extraction result.
    Raises specific httpx, json, or value errors on failure.
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set") # Raise error if key is missing

    try:
        headers = {"Content-Type": "application/json"}
        params = {"key": settings.GEMINI_API_KEY}

        # Refined prompt
        prompt = (
            f"""
            You are 'Fresh Market GH Assistant', a WhatsApp chatbot for a grocery service in Ghana.
            Analyze the user's message and classify their core intent related to grocery shopping or bot interaction.
            Respond ONLY with a single, minified JSON object. Do not include any other text, explanations, or markdown.

            User message: "{message}"

            Classify the intent into one of the following categories:
            - `buy`: User wants to start a new order, browse products, see the menu, ask what's available, or expresses general interest in purchasing.
            - `check_status`: User is asking about the current status or location of an existing order.
            - `cancel_order`: User explicitly states they want to cancel an order.
            - `greet`: User is initiating conversation politely (e.g., hello, hi, good morning).
            - `help`: User is asking for instructions, how to use the service, or general support.
            - `repeat`: User is asking you to repeat the last message or information you sent.
            - `thank_you`: User is expressing gratitude (e.g., thank you, thanks).
            - `unknown`: The message is irrelevant to grocery shopping, unclear, or outside the bot's capabilities (e.g., telling a joke, asking for personal details, spam).

            Your JSON output MUST contain these two fields:
            - `intent` (string): One of the intent types listed above.
            - `response` (string): A very short, friendly acknowledgement or standard reply appropriate for the intent.
                - `buy`: "Okay, let's start your order."
                - `check_status`: "Checking your order status now."
                - `cancel_order`: "Processing your cancellation request."
                - `greet`: "Hello! How can I assist you today?"
                - `help`: "I can help with that. What do you need?"
                - `repeat`: "Certainly, I can repeat that."
                - `thank_you`: "You're welcome!"
                - `unknown`: "I'm sorry, I can only help with grocery orders."

            Example response format:
            {{"intent": "buy", "response": "Okay, let's start your order."}}
            """
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1, # Lower temperature for predictable JSON
                "maxOutputTokens": 100
            },
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(settings.GEMINI_API_URL, headers=headers, params=params, json=payload)
            response.raise_for_status() # Raises HTTPStatusError

            result = response.json()
            json_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")

            if not json_text:
                 error_msg = result.get("candidates", [{}])[0].get("finishReason", "no text")
                 logger.error(f"Gemini API response missing expected text payload: {result}")
                 raise ValueError(f"Gemini API returned empty text payload or error: {error_msg}")

            return json.loads(json_text) # Raise JSONDecodeError if invalid

    except httpx.RequestError as e:
        logger.error(f"Gemini API communication error: {e}", exc_info=True)
        raise # Re-raise the exception
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding Gemini response JSON: {e}", exc_info=True)
        raise # Re-raise the exception
    except ValueError as e: # From missing text payload check
        logger.error(f"Error processing Gemini response: {e}", exc_info=True)
        raise # Re-raise the exception
    except Exception as e:
        # Catch any other unexpected errors during the API call itself
        logger.error(f"Unexpected error during Gemini API call: {e}", exc_info=True)
        raise # Re-raise the exception


# New helper function to handle Gemini call with graceful fallback
async def get_intent_gracefully(message: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls Gemini API, handles errors gracefully, and provides fallback logic.
    Always returns a dictionary with 'intent' and 'response'.
    """
    try:
        gemini_result = await call_gemini_api(message)
        # Validate minimum expected structure from Gemini
        if "intent" in gemini_result and "response" in gemini_result:
             return gemini_result
        else:
             logger.error(f"Gemini API returned unexpected structure: {gemini_result}")
             # Fallback if structure is valid JSON but wrong keys
             return {"intent": "unknown", "response": "I received an unexpected response from my processing service. Please try again."}

    except ValueError as e:
         if "GEMINI_API_KEY not set" in str(e):
             logger.warning("GEMINI_API_KEY not set, using fallback logic for intent extraction.")
             # --- Fallback logic - simple keyword matching with basic typo tolerance ---
             lower_msg = message.lower().strip()
             if "cancel" in lower_msg: return {"intent": "cancel_order", "response": "Okay, I can help with cancelling an order."}
             if "status" in lower_msg or "track" in lower_msg or "where is my order" in lower_msg: return {"intent": "check_status", "response": "Let me check on your order."}
             if any(word in lower_msg for word in ["buy", "want", "order", "menu", "available", "shop"]):
                  return {"intent": "buy", "response": "Great! I can help with that."}
             if any(word in lower_msg for word in ["hello", "hi", "hey", "hola", "morning", "afternoon"]):
                  return {"intent": "greet", "response": "Hello! How can I help you today?"}
             if "help" in lower_msg or "how" in lower_msg:
                  return {"intent": "help", "response": "I can certainly help with that."}
             if any(word in lower_msg for word in ["repeat", "say again", "last message"]):
                  return {"intent": "repeat", "response": "Certainly, I can repeat that."}
             if any(word in lower_msg for word in ["thank", "thanks"]):
                  return {"intent": "thank_you", "response": "You're welcome!"}
             return {"intent": "unknown", "response": "I'm sorry, I can only assist with grocery orders. Could you please rephrase?"}
         else:
             # Handle other ValueErrors from call_gemini_api (e.g. empty text payload)
             logger.error(f"Error processing Gemini response: {e}", exc_info=True)
             return {"intent": "unknown", "response": f"There was an issue processing the response from my service: {e}. Please try again."}

    except httpx.RequestError as e:
        logger.error(f"Gemini API communication error: {e}", exc_info=True)
        return {"intent": "unknown", "response": "I'm having trouble connecting to my service. Please try again in a moment."}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding Gemini response JSON: {e}", exc_info=True)
        return {"intent": "unknown", "response": "I received an unreadable response from my processing service. Please try again."}
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"Unexpected error during Gemini intent extraction: {e}", exc_info=True)
        return {"intent": "unknown", "response": "I'm sorry, something unexpected went wrong with my processing. Please try again."}


async def generate_paystack_payment_link(order_id: str, amount: float, user_phone: str) -> str:
    """
    Generate a Paystack payment link for the order.
    Raises a specific exception if the API call fails.
    """
    if not settings.PAYSTACK_SECRET_KEY:
        logger.warning("PAYSTACK_SECRET_KEY not set, cannot generate real payment link. Using mock link.")
        if amount <= 0:
             raise ValueError("Amount must be positive to generate payment link.")
        return f"{settings.FRONTEND_URL}/mock-payment-success?order_id={order_id}&amount={amount:.2f}"

    headers = {
        "Authorization": f"Bearer {settings.PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    placeholder_email = f"{''.join(filter(str.isdigit, user_phone))}@market.bot"
    unique_reference = f"{order_id}_{int(datetime.now().timestamp())}"

    payload = {
        "email": placeholder_email,
        "amount": int(amount * 100),
        "currency": "GHS",
        "reference": unique_reference,
        "callback_url": f"{settings.FRONTEND_URL}/payment-success?order_id={order_id}",
        "channels": ["card", "mobile_money"],
        "metadata": {"order_id": order_id, "phone": user_phone}
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(settings.PAYSTACK_PAYMENT_URL, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            if data.get("status") is True and data.get("data") and data["data"].get("authorization_url"):
                 logger.info(f"Successfully generated Paystack link for order {order_id}. Reference: {unique_reference}")
                 return data["data"]["authorization_url"]
            else:
                 logger.error(f"Paystack API returned success=true but missing auth_url or unexpected data: {data}")
                 raise ValueError("Paystack API response format error during link generation.")

    except httpx.RequestError as e:
         logger.error(f"Paystack API request error: {e}", exc_info=True)
         if isinstance(e, httpx.HTTPStatusError):
              if e.response.status_code in [401, 403]:
                  raise Exception("Payment gateway authentication failed. Please contact support.") from e
              elif e.response.status_code == 400:
                   raise Exception(f"Invalid request to payment gateway: {e.response.text}") from e
              else:
                   raise Exception(f"Payment gateway error ({e.response.status_code}). Please try again later.") from e
         else:
              raise Exception(f"Could not connect to payment gateway: {e}. Please check your internet connection.") from e
    except Exception as e:
        logger.error(f"Unexpected error during Paystack link generation for order {order_id}: {str(e)}", exc_info=True)
        raise Exception(f"An error occurred during payment link generation: {e}") from e


def verify_paystack_signature(request_body: bytes, signature: str, secret_key: Optional[str]) -> bool:
    """
    Verifies the Paystack webhook signature using the main Secret Key.
    Returns True if signature is valid, False otherwise.
    """
    if not secret_key:
        logger.error("PAYSTACK_SECRET_KEY is not set. Cannot verify Paystack webhook signature. THIS IS A MAJOR SECURITY RISK.")
        return False

    try:
        expected_signature = hmac.new(
            secret_key.encode('utf-8'),
            request_body,
            hashlib.sha512
        ).hexdigest()

        is_valid = hmac.compare_digest(expected_signature, signature)

        if not is_valid:
            logger.warning("Paystack webhook signature verification failed.")
        else:
            logger.info("Paystack webhook signature verified successfully.")

        return is_valid
    except Exception as e:
        logger.error(f"Error during Paystack signature verification: {e}", exc_info=True)
        return False


async def handle_new_conversation(user: Dict[str, Any], gemini_result: Dict[str, Any], from_number: str, is_new_user: bool, original_message: str) -> str:
    """
    Handles interaction when a user has no active unpaid orders.
    Routes messages based on the intent extracted by Gemini or fallback logic.
    Includes is_new_user flag for tailored responses.
    Accepts original_message for logging/context if needed.
    """
    user_id = user['id']
    intent_data = gemini_result
    intent = intent_data.get("intent")
    ai_response_ack = intent_data.get("response", "Okay.")

    logger.info(f"Handling new conversation for user {user_id}. Intent: {intent}. Is New User: {is_new_user}. Message: '{original_message}'") # CORRECTED LOGGING

    if intent == "buy":
        session_token = str(uuid.uuid4())
        selection_url = f"{settings.FRONTEND_URL}?session={session_token}"
        try:
            supabase.table("sessions").insert({
                "user_id": user_id,
                "phone_number": from_number,
                "session_token": session_token,
                "last_intent": "buy",
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
            }).execute()
            logger.info(f"Created new session {session_token} for user {user_id}")

            # **APPLIED TO BUY:** Combine AI acknowledgement with menu link message
            greeting_part = "Welcome! " if is_new_user else ""
            return (
                f"{ai_response_ack} {greeting_part}"
                f"You can select the fresh items you'd like to purchase from our online menu here:\n"
                f"{selection_url}"
            )
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id} (buy intent): {e}", exc_info=True)
            return "Sorry, I'm having trouble starting a new order right now. Please try again in a moment."

    elif intent == "check_status":
        try:
            active_paid_orders_res = supabase.table("orders").select("status, id, delivery_type").eq("user_id", user_id).in_("payment_status", [DefaultStatus.PAYMENT_PAID, DefaultStatus.PAYMENT_PARTIALLY_PAID]).not_.in_("status", [DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_FAILED]).order("created_at", desc=True).limit(1).execute()

            # **APPLIED TO CHECK_STATUS:** Combine AI acknowledgement with status message
            if active_paid_orders_res.data:
                latest_order = active_paid_orders_res.data[0]
                status_display = latest_order['status'].replace('-', ' ').title()
                delivery_type = latest_order.get('delivery_type', 'N/A').title()
                return f"{ai_response_ack} Your latest active order (ID: {latest_order['id']}) status is: *{status_display}* ({delivery_type})."
            else:
                recent_delivered_res = supabase.table("orders").select("id").eq("user_id", user_id).eq("status", DefaultStatus.ORDER_DELIVERED).order("created_at", desc=True).limit(1).execute()
                if recent_delivered_res.data:
                    return f"{ai_response_ack} Your latest order (ID: {recent_delivered_res.data[0]['id']}) has already been delivered. Can I help you start a new one?"
                else:
                     return f"{ai_response_ack} It looks like you don't have any active orders right now. Can I help you start a new one?"
        except Exception as e:
            logger.error(f"Error checking order status for user {user_id}: {e}", exc_info=True)
            return f"{ai_response_ack} I'm having trouble looking up your order details right now. Please try again in a moment."

    elif intent == "cancel_order":
        # **APPLIED TO CANCEL_ORDER:** Combine AI acknowledgement with the 'no pending' message
        return f"{ai_response_ack} You don't have any pending orders to cancel right now."

    elif intent == "greet":
        # **APPLIED TO GREET:** Combine AI acknowledgement with the welcome/welcome back message
        if is_new_user:
            return (
                f"{ai_response_ack}\n\n" # E.g., "Hello! How can I assist you today?"
                "I can help you order fresh groceries.\n\n"
                "To start, just say 'I want to buy...' or 'Show me the menu'."
            )
        else:
            user_name = user.get('name')
            greeting_name = f", {user_name}!" if user_name else "!"
            return (
                 f"{ai_response_ack}{greeting_name} " # E.g., "Hello! How can I assist you today, Kofi!"
                 "How can I help you with your groceries today?"
            )


    elif intent == "help":
        # **APPLIED TO HELP:** Combine AI ack with help text
         return (
             f"{ai_response_ack}\n\n" # Start with AI's canned "I can help with that. What do you need?"
             "I can help you with the following:\n\n" # Clearly introduce the list
             "🛒 *Starting a new grocery order:*\n Just say 'I want to buy...' or 'Show me the menu'.\n\n"
             "📦 *Checking your order status:*\n Ask 'Where is my order?' or 'What is the status of my order?'.\n\n"
             "❌ *Cancelling a pending order:*\n If you have an order waiting for payment, reply 'cancel'.\n\n"
             "What can I specifically assist you with regarding your groceries?" # Re-prompt
         )

    elif intent == "repeat":
        last_msg = user.get('last_bot_message')
        # **APPLIED TO REPEAT:** Combine AI acknowledgement with the repeated message
        if last_msg:
            return f"{ai_response_ack}\n\nI last said:\n> {last_msg}"
        else:
            return f"{ai_response_ack} I don't have a recent message to repeat right now."

    elif intent == "thank_you":
        # **APPLIED TO THANK_YOU:** Already returns AI ack, fits the pattern.
         return ai_response_ack

    else: # unknown or any other unhandled intent
        logger.info(f"User {user_id} sent message with unknown intent '{intent}'. Message: '{original_message}')")
        return ai_response_ack # This should be "I'm sorry, I can only help with grocery orders."


def calculate_delivery_fee(lat: float, lon: float) -> float:
    """
    Calculates the delivery fee based on the distance from a central point.
    Uses Haversine formula. SHOULD BE REPLACED with a proper geospatial service or zones in production.
    """
    # Coordinates for a central point in Accra (e.g., Kwame Nkrumah Interchange)
    central_lat, central_lon = 5.5560, -0.2057

    R = 6371 # Radius of Earth in kilometers

    lat1_rad = math.radians(central_lat)
    lon1_rad = math.radians(central_lon)
    lat2_rad = math.radians(lat)
    lon2_rad = math.radians(lon)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_km = R * c # Distance in km

    # Define fee based on distance tiers (adjust thresholds and fees as needed)
    if distance_km < 5:
        fee = 15.00
    elif distance_km < 10:
        fee = 25.00
    elif distance_km < 20:
         fee = 35.00
    elif distance_km < 30:
         fee = 45.00
    else:
        fee = 60.00
        if distance_km > 50:
             logger.warning(f"Delivery requested outside standard range: {distance_km:.2f} km from center {central_lat},{central_lon}. Applying max fee.")

    logger.info(f"Calculated delivery fee: GHS {fee:.2f} for location {lat},{lon} ({distance_km:.2f} km from center)")
    return fee


# --- Main Webhook Endpoint ---

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    """
    Main webhook endpoint for incoming WhatsApp messages (text and location).
    Handles user identification, state management (pending orders), and intent routing.
    Returns 200 OK to the webhook provider regardless of internal processing success.
    """
    form_data = await request.form()
    from_number = form_data.get("From")
    incoming_msg = form_data.get("Body", "").strip() # Capture the incoming message

    if not from_number:
        logger.warning("Received webhook without 'From' number field.")
        return JSONResponse(content={}, status_code=status.HTTP_200_OK)

    from_number_clean = from_number.replace("whatsapp:", "")
    logger.info(f"Received message from {from_number_clean}. Form Data Keys: {list(form_data.keys())}")
    if incoming_msg:
        logger.info(f"Message Body: '{incoming_msg}'")


    # Determine if it's a location message
    is_location_message = "Latitude" in form_data and "Longitude" in form_data
    latitude, longitude = None, None
    if is_location_message:
        try:
            latitude = float(form_data.get("Latitude"))
            longitude = float(form_data.get("Longitude"))
            logger.info(f"Received location message from {from_number_clean}: {latitude},{longitude}")
        except (ValueError, TypeError) as e:
            logger.error(f"Could not parse location coordinates from webhook for user {from_number_clean}. Data: {form_data}. Error: {e}", exc_info=True)
            is_location_message = False # Treat as invalid location message

    # Determine if it's a media message (other than location, which is handled separately)
    is_media_message = form_data.get("NumMedia", "0") != "0" and not is_location_message

    reply_message = "" # Initialize reply message

    try:
        # --- 1. Find or Create User ---
        user_res = supabase.table("users").select("*").eq("phone_number", from_number_clean).limit(1).execute()
        user = user_res.data[0] if user_res.data else None
        is_new_user = False

        if not user:
            is_new_user = True
            try:
                # CORRECTED: Removed .select("*") from the insert query
                insert_res = supabase.table("users").insert({"phone_number": from_number_clean}).execute()
                # Assuming execute() on insert returns data in this client version
                user = insert_res.data[0]
                logger.info(f"Created new user with id {user['id']} for phone {from_number_clean}")
            except Exception as e:
                 logger.error(f"Failed to create new user for {from_number_clean}: {e}", exc_info=True)
                 if send_whatsapp_message_available:
                    await send_whatsapp_message(from_number_clean, "Sorry, I'm having trouble setting up your profile right now. Please try again in a moment.")
                 return JSONResponse(content={}, status_code=status.HTTP_200_OK)

        user_id = user['id']
        # Update last_active timestamp (fire and forget, doesn't need to block)
        try:
             supabase.table("users").update({"last_active": datetime.now().isoformat()}).eq("id", user['id']).execute()
        except Exception as e:
             logger.error(f"Failed to update last_active for user {user_id}: {e}", exc_info=True)

        # --- 2. Find and Clean Up Potentially Multiple Pending Orders ---
        latest_pending_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("payment_status", DefaultStatus.PAYMENT_UNPAID).not_.in_("status", [DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_FAILED]).order("created_at", desc=True).limit(1).execute()
        active_pending_order = latest_pending_res.data[0] if latest_pending_res.data else None

        if active_pending_order:
            latest_pending_created_at = active_pending_order['created_at']
            older_pending_res = supabase.table("orders").select("id").eq("user_id", user_id).eq("payment_status", DefaultStatus.PAYMENT_UNPAID).not_.in_("status", [DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_FAILED]).lt("created_at", latest_pending_created_at).execute()

            if older_pending_res.data:
                older_order_ids = [o['id'] for o in older_pending_res.data]
                logger.info(f"Cancelling {len(older_order_ids)} older pending orders for user {user_id}: {older_order_ids}")
                try:
                    supabase.table("orders").update({
                        "status": DefaultStatus.ORDER_CANCELLED,
                        "payment_status": DefaultStatus.PAYMENT_CANCELLED,
                        "cancelled_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                         "cancellation_reason": "superseded by newer pending order"
                    }).in_("id", older_order_ids).execute()
                    logger.info(f"Cancelled older pending orders: {older_order_ids}")
                except Exception as e:
                     logger.error(f"Failed to cancel older pending orders for user {user_id}: {e}", exc_info=True)


        # --- 3. Main Logic Branching: Handle Message Type and Order State ---

        # A. Handle incoming Location message if it's valid
        if is_location_message and latitude is not None and longitude is not None:
             # Check if we are expecting a location for the active pending order
             if active_pending_order and active_pending_order['status'] == DefaultStatus.ORDER_AWAITING_LOCATION:
                 logger.info(f"Processing expected location for order {active_pending_order['id']} from user {user_id}")
                 try:
                    location_to_save = json.dumps({"latitude": str(latitude), "longitude": str(longitude)})
                    supabase.table("users").update({"last_known_location": location_to_save}).eq("id", user_id).execute()
                    logger.info(f"Saved location {latitude},{longitude} for user {user_id}")

                    delivery_fee = calculate_delivery_fee(latitude, longitude)
                    total_with_delivery = active_pending_order['total_amount'] + delivery_fee

                    update_data = {
                        "status": DefaultStatus.ORDER_PENDING_PAYMENT,
                        "delivery_type": "delivery",
                        "delivery_fee": delivery_fee,
                        "total_with_delivery": total_with_delivery,
                        "delivery_location_lat": latitude,
                        "delivery_location_lon": longitude,
                        "updated_at": datetime.now().isoformat()
                    }
                    supabase.table("orders").update(update_data).eq("id", active_pending_order['id']).execute()
                    logger.info(f"Updated order {active_pending_order['id']} with delivery details and status {DefaultStatus.ORDER_PENDING_PAYMENT}")

                    payment_link = await generate_paystack_payment_link(active_pending_order['id'], total_with_delivery, from_number_clean)
                    reply_message = (
                        f"Great, location received! Your delivery fee is GHS {delivery_fee:.2f}.\n\n"
                        f"Your new total is *GHS {total_with_delivery:.2f}*.\n\n"
                        f"Please complete your payment here to confirm your order:\n{payment_link}"
                    )
                 except Exception as e:
                     logger.error(f"Error processing location or finalizing order for user {user_id}, order {active_pending_order['id']}: {e}", exc_info=True)
                     reply_message = f"Sorry, I encountered an issue processing your delivery details. Please try again or contact support."

             elif active_pending_order and active_pending_order['status'] == DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION:
                  logger.info(f"User {user_id} sent location while order {active_pending_order['id']} is {DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION}. Treating as 'provide new'.")
                  try:
                    location_to_save = json.dumps({"latitude": str(latitude), "longitude": str(longitude)})
                    supabase.table("users").update({"last_known_location": location_to_save}).eq("id", user_id).execute()
                    logger.info(f"Saved NEW location {latitude},{longitude} for user {user_id}")

                    delivery_fee = calculate_delivery_fee(latitude, longitude)
                    total_with_delivery = active_pending_order['total_amount'] + delivery_fee

                    update_data = {
                        "status": DefaultStatus.ORDER_PENDING_PAYMENT,
                        "delivery_type": "delivery",
                        "delivery_fee": delivery_fee,
                        "total_with_delivery": total_with_delivery,
                        "delivery_location_lat": latitude,
                        "delivery_location_lon": longitude,
                        "updated_at": datetime.now().isoformat()
                    }
                    supabase.table("orders").update(update_data).eq("id", active_pending_order['id']).execute()
                    logger.info(f"Updated order {active_pending_order['id']} with NEW delivery details and status {DefaultStatus.ORDER_PENDING_PAYMENT}")

                    payment_link = await generate_paystack_payment_link(active_pending_order['id'], total_with_delivery, from_number_clean)
                    reply_message = (
                        f"Okay, using the new location you shared. The delivery fee is GHS {delivery_fee:.2f}.\n\n"
                        f"Your new total is *GHS {total_with_delivery:.2f}*.\n\n"
                        f"Please complete your payment here:\n{payment_link}"
                    )
                  except Exception as e:
                     logger.error(f"Error processing NEW location or finalizing order for user {user_id}, order {active_pending_order['id']} (from confirmation stage): {e}", exc_info=True)
                     reply_message = f"Sorry, I encountered an issue processing your delivery details. Please try again or contact support."

             else:
                 # User sent a location when we weren't expecting one
                 logger.info(f"Received unexpected location message from user {user_id}. No pending order or not awaiting location.")
                 reply_message = "Thanks for sharing your location! I've saved it for future delivery orders, but I wasn't expecting it right now. How else can I help you today?"

        # B. Handle Media Message
        elif is_media_message:
             logger.info(f"Received media message (non-text/non-location) from user {user_id}. NumMedia: {form_data.get('NumMedia')}")
             reply_message = "Thanks for sending that! Currently, I can only process text messages and shared locations. How can I help you with your grocery order?"

        # C. Handle incoming Text Message
        elif incoming_msg:
            lower_incoming_msg = incoming_msg.lower().strip()
            handled_by_pending_state = False

            if active_pending_order:
                order_id = active_pending_order['id']
                current_status: OrderStatus = active_pending_order.get('status', DefaultStatus.ORDER_PENDING_CONFIRMATION)
                logger.info(f"Handling pending order {order_id} (status: {current_status}) text input: '{incoming_msg}'")

                # --- Check for specific commands within the current pending state ---
                if current_status == DefaultStatus.ORDER_PENDING_CONFIRMATION:
                     if lower_incoming_msg in ["1", "delivery"]:
                         handled_by_pending_state = True
                         if user.get("last_known_location"):
                             try:
                                supabase.table("orders").update({"status": DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION, "updated_at": datetime.now().isoformat()}).eq("id", order_id).execute()
                                logger.info(f"Order {order_id} status updated to {DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION}")
                                reply_message = (
                                    "I see you have a saved delivery location with us. Would you like to use it for this order?\n\n"
                                    "Reply *1* to use saved location\n"
                                    "Reply *2* to provide a new one\n\n"
                                    "Or reply 'cancel'."
                                )
                             except Exception as e:
                                  logger.error(f"Failed to update order status for location confirmation {order_id}: {e}", exc_info=True)
                                  reply_message = "Sorry, I had trouble updating your order details. Please try again or reply 'cancel'."
                         else:
                             try:
                                supabase.table("orders").update({"delivery_type": "delivery", "status": DefaultStatus.ORDER_AWAITING_LOCATION, "updated_at": datetime.now().isoformat()}).eq("id", order_id).execute()
                                logger.info(f"Order {order_id} status updated to {DefaultStatus.ORDER_AWAITING_LOCATION}")
                                reply_message = (
                                    "Great! Please share your delivery location using the WhatsApp location sharing feature.\n\n"
                                    "Tap the *clip icon 📎* next to the message box, then choose *'Location' 📍* and select 'Send your current location' or a nearby place.\n\n"
                                    "Or reply 'cancel'."
                                )
                             except Exception as e:
                                  logger.error(f"Failed to update order status to awaiting location {order_id}: {e}", exc_info=True)
                                  reply_message = "Sorry, I had trouble updating your order details. Please try again or reply 'cancel'."

                     elif lower_incoming_msg in ["2", "pickup"]:
                          handled_by_pending_state = True
                          try:
                            supabase.table("orders").update({"delivery_type": "pickup", "status": DefaultStatus.ORDER_PENDING_PAYMENT, "updated_at": datetime.now().isoformat()}).eq("id", order_id).execute()
                            logger.info(f"Order {order_id} status updated to {DefaultStatus.ORDER_PENDING_PAYMENT} (pickup)")
                            payment_link = await generate_paystack_payment_link(order_id, active_pending_order['total_amount'], from_number_clean)
                            reply_message = (
                                f"Alright, your order is set for pickup.\n\n"
                                f"Your total is *GHS {active_pending_order['total_amount']:.2f}*.\n\n"
                                f"Please complete your payment here to confirm your order:\n{payment_link}\n\n"
                                "Or reply 'cancel'."
                            )
                          except Exception as e:
                              logger.error(f"Failed to finalize pickup order {order_id}: {e}", exc_info=True)
                              reply_message = f"Sorry, I encountered an issue setting up your pickup order and generating the payment link. Please try again in a moment, or reply 'cancel'."

                elif current_status == DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION:
                    if lower_incoming_msg == "1": # Use saved location
                        handled_by_pending_state = True
                        location_str = user.get("last_known_location")
                        if not location_str:
                            logger.warning(f"User {user_id} replied '1' to location confirmation but had no saved location. Order {order_id}")
                            try:
                                supabase.table("orders").update({"status": DefaultStatus.ORDER_AWAITING_LOCATION, "updated_at": datetime.now().isoformat()}).eq("id", order_id).execute()
                                reply_message = (
                                    "It seems your saved location wasn't available.\n\n"
                                    "Please share your delivery location using the WhatsApp location sharing feature.\n"
                                    "Tap the *clip icon 📎* next to the message box, then choose *'Location' 📍*.\n\n"
                                    "Or reply 'cancel'."
                                )
                            except Exception as e:
                                logger.error(f"Failed to update status to awaiting location after missing saved loc {order_id}: {e}", exc_info=True)
                                reply_message = "Sorry, I had trouble processing your request. Please try again or reply 'cancel'."
                        else:
                            try:
                                location_data = json.loads(location_str)
                                latitude = float(location_data.get("latitude"))
                                longitude = float(location_data.get("longitude"))

                                delivery_fee = calculate_delivery_fee(latitude, longitude)
                                total_with_delivery = active_pending_order['total_amount'] + delivery_fee

                                update_data = {
                                    "status": DefaultStatus.ORDER_PENDING_PAYMENT,
                                    "delivery_type": "delivery",
                                    "delivery_fee": delivery_fee,
                                    "total_with_delivery": total_with_delivery,
                                    "delivery_location_lat": latitude,
                                    "delivery_location_lon": longitude,
                                    "updated_at": datetime.now().isoformat()
                                }
                                supabase.table("orders").update(update_data).eq("id", order_id).execute()
                                logger.info(f"Order {order_id} status updated to {DefaultStatus.ORDER_PENDING_PAYMENT} using saved location.")

                                payment_link = await generate_paystack_payment_link(order_id, total_with_delivery, from_number_clean)
                                reply_message = (
                                    f"Using your saved location. The delivery fee is GHS {delivery_fee:.2f}.\n\n"
                                    f"Your new total is *GHS {total_with_delivery:.2f}*.\n\n"
                                    f"Please complete your payment here:\n{payment_link}\n\n"
                                    "Or reply 'cancel'."
                                )
                            except (ValueError, TypeError, json.JSONDecodeError) as e:
                                logger.error(f"Invalid saved location data or fee processing error for user {user_id}: {location_str}. Error: {e}", exc_info=True)
                                try:
                                    supabase.table("orders").update({"status": DefaultStatus.ORDER_AWAITING_LOCATION, "updated_at": datetime.now().isoformat()}).eq("id", order_id).execute()
                                    reply_message = (
                                        "It seems there was an issue with your saved location data.\n\n"
                                        "Please share your delivery location again using the WhatsApp location sharing feature.\n"
                                        "Tap the *clip icon 📎* next to the message box, then choose *'Location' 📍*.\n\n"
                                        "Or reply 'cancel'."
                                    )
                                except Exception as update_e:
                                     logger.error(f"Failed to update status to awaiting location after invalid saved loc {order_id}: {update_e}", exc_info=True)
                                     reply_message = "Sorry, I had trouble processing your request. Please try again or reply 'cancel'."

                            except Exception as e:
                                logger.error(f"Error finalizing order {order_id} after using saved location for user {user_id}: {e}", exc_info=True)
                                reply_message = f"Sorry, I encountered an issue processing your delivery details. Please try again or reply 'cancel'."

                    elif lower_incoming_msg == "2": # Provide a new one
                        handled_by_pending_state = True
                        try:
                            supabase.table("orders").update({"status": DefaultStatus.ORDER_AWAITING_LOCATION, "updated_at": datetime.now().isoformat()}).eq("id", order_id).execute()
                            logger.info(f"Order {order_id} status updated to {DefaultStatus.ORDER_AWAITING_LOCATION} (user providing new)")
                            reply_message = (
                                "Okay, no problem.\n\n"
                                "Please share your new delivery location using the WhatsApp location sharing feature.\n"
                                "Tap the *clip icon 📎* next to the message box, then choose *'Location' 📍*.\n\n"
                                "Or reply 'cancel'."
                            )
                        except Exception as e:
                            logger.error(f"Failed to update status to awaiting new location {order_id}: {e}", exc_info=True)
                            reply_message = "Sorry, I had trouble processing your request. Please try again or reply 'cancel'."

                # --- Handle Cancel Command ---
                if lower_incoming_msg == 'cancel':
                    handled_by_pending_state = True
                    if current_status in [DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_FAILED]:
                         reply_message = f"This order (ID: {order_id}) is already marked as *{current_status}* and cannot be cancelled."
                    else:
                         try:
                            supabase.table("orders").update({
                                "status": DefaultStatus.ORDER_CANCELLED,
                                "payment_status": DefaultStatus.PAYMENT_CANCELLED,
                                "cancelled_at": datetime.now().isoformat(),
                                "updated_at": datetime.now().isoformat()
                            }).eq("id", order_id).execute()
                            logger.info(f"Order {order_id} successfully cancelled by user {user_id}.")
                            reply_message = f"Your order (ID: {order_id}) has been cancelled. Please let me know if there's anything else I can help with."
                         except Exception as e:
                            logger.error(f"Failed to cancel order {order_id} for user {user_id}: {e}", exc_info=True)
                            reply_message = "Sorry, I had trouble cancelling your order right now. Please try again or contact support."


                # --- Handle messages that are *not* state-specific commands ---
                if not handled_by_pending_state:
                    user_context = {'has_paid_order': False, 'has_saved_address': bool(user.get("last_known_location"))}
                    gemini_result = await get_intent_gracefully(incoming_msg, user_context)
                    intent = gemini_result.get('intent')
                    ai_ack = gemini_result.get('response', 'Okay.')

                    # Default reminder message based on current pending status
                    reminder_message = ""
                    if current_status == DefaultStatus.ORDER_PENDING_CONFIRMATION:
                         reminder_message = "\n\nBut please first choose '1' for Delivery or '2' for Pickup for your pending order."
                    elif current_status == DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION:
                         reminder_message = "\n\nBut please first reply '1' or '2' regarding your saved location for your pending order."
                    elif current_status == DefaultStatus.ORDER_AWAITING_LOCATION:
                         reminder_message = "\n\nBut I'm still waiting for your delivery location for your pending order. Please share it using the location feature."
                    elif current_status == DefaultStatus.ORDER_PENDING_PAYMENT:
                         total = active_pending_order.get('total_with_delivery') or active_pending_order['total_amount']
                         try:
                             payment_link = await generate_paystack_payment_link(order_id, total, from_number_clean)
                             reminder_message = (
                                 f"\n\nBut you have a pending order (ID: {order_id}) waiting for payment (GHS {total:.2f}). Please pay here: {payment_link}\n"
                                 "Or reply 'cancel'."
                             )
                         except Exception as e:
                              logger.error(f"Failed to re-generate payment link for reminder for order {order_id}: {e}", exc_info=True)
                              reminder_message = f"\n\nBut you have a pending order (ID: {order_id}) waiting for payment. Please reply 'cancel' if you don't want to proceed."


                    if intent in ["greet", "thank_you", "help", "repeat"]:
                         reply_message = ai_ack + reminder_message
                    elif intent in ["buy", "check_status"]:
                         action_needed = ""
                         if current_status == DefaultStatus.ORDER_PENDING_CONFIRMATION: action_needed = "choose delivery/pickup"
                         elif current_status in [DefaultStatus.ORDER_AWAITING_LOCATION, DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION]: action_needed = "provide delivery location"
                         elif current_status == DefaultStatus.ORDER_PENDING_PAYMENT: action_needed = "make payment"

                         if action_needed:
                              reply_message = f"{ai_ack} However, you currently have a pending order (ID: {order_id}) waiting for you to {action_needed}. Please complete that step first, or reply 'cancel'."
                         else:
                              reply_message = f"{ai_ack} However, you have a pending order (ID: {order_id}) in progress. Please complete the next step for that order, or reply 'cancel'."

                    else:
                         reply_message = f"I'm not sure how to help with that right now. You currently have a pending order (ID: {order_id}) in progress. Please complete the next step for that order, or reply 'cancel'."
                         if reminder_message:
                              reply_message += reminder_message


            else: # No active pending order
                user_context = {'has_paid_order': False, 'has_saved_address': bool(user.get("last_known_location"))}
                gemini_result = await get_intent_gracefully(incoming_msg, user_context)
                reply_message = await handle_new_conversation(user, gemini_result, from_number_clean, is_new_user, incoming_msg)

        else:
             logger.info(f"Received an empty or unhandled message type from user {user_id}.")
             if is_new_user:
                 reply_message = "👋 Welcome to Fresh Market GH!\n\nI can help you order fresh groceries.\n\nTo start, just say 'I want to buy...' or 'Show me the menu'."
             else:
                  user_name = user.get('name')
                  greeting_name = f", {user_name}!" if user_name else "!"
                  reply_message = f"Hello! Welcome back to Fresh Market GH{greeting_name} How can I help you with your groceries today?"


        # --- 4. Send the determined reply message and update last_bot_message ---
        if reply_message and send_whatsapp_message_available:
             try:
                await send_whatsapp_message(from_number_clean, reply_message)
                logger.info(f"Sent reply to {from_number_clean}.")
                try:
                    # Save the sent message as last_bot_message for 'repeat' intent
                    # Limit length to avoid excessive storage
                    truncated_message = reply_message[:1000] # Limit to first 1000 chars
                    supabase.table("users").update({"last_bot_message": truncated_message}).eq("id", user_id).execute();
                    logger.info(f"Saved last bot message for user {user_id}.");
                except Exception as e:
                     logger.error(f"Failed to save last bot message for user {user_id}: {e}", exc_info=True);

             except Exception as send_e:
                logger.error(f"Failed to send WhatsApp message to {from_number_clean}: {send_e}", exc_info=True)

        elif reply_message and not send_whatsapp_message_available:
            logger.error(f"send_whatsapp_message is not available. Cannot send reply to {from_number_clean}: {reply_message}")


    except Exception as e:
        logger.error(f"Unhandled critical error in whatsapp_webhook for user {from_number_clean}: {e}", exc_info=True)
        if not reply_message and from_number_clean and send_whatsapp_message_available:
             try:
                await send_whatsapp_message(from_number_clean, "Oh, something went wrong on my end. Please try again in a moment.")
             except Exception as send_e_2:
                 logger.error(f"Failed to send emergency error message to {from_number_clean}: {send_e_2}", exc_info=True)

    return JSONResponse(content={}, status_code=status.HTTP_200_OK)


# --- Frontend/API Endpoints ---
# (These endpoints remain largely the same as they are called from the frontend, not chat)

@app.post("/confirm-items")
async def confirm_items(request: OrderRequest, api_key: str = Depends(security.verify_api_key)):
    """
    Endpoint for the frontend/web menu to confirm items selected by the user.
    Requires API key authentication from the frontend.
    """
    if not supabase:
        logger.error("/confirm-items called but Supabase client is not available.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database connection not available")

    try:
        session_res = supabase.table("sessions").select("user_id, phone_number").eq("session_token", request.session_token).limit(1).execute()

        if not session_res.data:
            logger.warning(f"Session token {request.session_token} not found or expired for confirm-items.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired.")

        session_data = session_res.data[0]
        user_id = session_data["user_id"]
        phone_number = session_data["phone_number"]
        logger.info(f"Session {request.session_token} found for user {user_id} ({phone_number})")

        logger.info(f"Cancelling any existing pending unpaid orders for user_id: {user_id} before creating new one from session {request.session_token}.")
        try:
            supabase.table("orders").update({
                "status": DefaultStatus.ORDER_CANCELLED,
                "payment_status": DefaultStatus.PAYMENT_CANCELLED,
                "cancelled_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "cancellation_reason": "superseded by new order via web menu"
            }).eq("user_id", user_id).eq("payment_status", DefaultStatus.PAYMENT_UNPAID).not_.in_("status", [DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_FAILED]).execute()
            logger.info(f"Cancelled older pending orders for user {user_id} during confirm-items.")
        except Exception as e:
            logger.error(f"Failed to cancel old pending orders for user {user_id} during confirm-items: {e}", exc_info=True)


        items_dict = [item.model_dump() for item in request.items]

        order_data = {
            "user_id": user_id,
            "items_json": items_dict,
            "total_amount": request.total_amount,
            "status": DefaultStatus.ORDER_PENDING_CONFIRMATION,
            "payment_status": DefaultStatus.PAYMENT_UNPAID,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        try:
            # CORRECTED: Removed .select("id") from the insert query
            result = supabase.table("orders").insert(order_data).execute()
            # Assuming execute() on insert returns data in this client version
            if not result.data:
                logger.error(f"Supabase insert returned no data for new order for user {user_id}.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create new order in database.")

            order_id = result.data[0]["id"]
            logger.info(f"Created new order {order_id} for user {user_id} from session {request.session_token}")

        except Exception as e:
            logger.error(f"Failed to insert new order for user {user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create new order in database: {e}")


        try:
            supabase.table("sessions").delete().eq("session_token", request.session_token).execute()
            logger.info(f"Deleted session token {request.session_token} after order creation.")
        except Exception as e:
             logger.error(f"Failed to delete session token {request.session_token}: {e}", exc_info=True)


        delivery_msg = (
            "Thank you for confirming your items!\n\n"
            f"Your order total is *GHS {request.total_amount:.2f}* (excluding delivery).\n\n"
            "Please tell me how you'd like to receive your order by replying with the number:\n\n"
            f"*1* for Delivery (I'll ask for your location to calculate the fee).\n"
            f"*2* for Pickup (no delivery fee)."
        )
        if send_whatsapp_message_available:
            try:
                await send_whatsapp_message(phone_number, delivery_msg)
                logger.info(f"Sent confirmation message to user {user_id} ({phone_number}) for order {order_id}")
                try:
                    truncated_message = delivery_msg[:1000]
                    supabase.table("users").update({"last_bot_message": truncated_message}).eq("id", user_id).execute();
                    logger.info(f"Saved last bot message for user {user_id} from /confirm-items.");
                except Exception as e:
                     logger.error(f"Failed to save last bot message for user {user_id} from /confirm-items: {e}", exc_info=True);

            except Exception as e:
                logger.error(f"Failed to send WhatsApp confirmation message to {phone_number} for order {order_id}: {e}", exc_info=True)
        else:
            logger.error(f"send_whatsapp_message is not available. Cannot send confirmation for order {order_id} to {phone_number}")

        return {"status": "order saved", "order_id": order_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/confirm-items endpoint unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")


@app.post("/payment-success")
async def payment_success_webhook(request: Request):
    """
    Paystack webhook endpoint for successful payments.
    Verifies signature, updates order status, and notifies user.
    MUST return 200 OK to Paystack quickly.
    Also handles other Paystack events like 'charge.failed' if necessary.
    """
    logger.info("Received Paystack webhook call.")

    signature = request.headers.get("x-paystack-signature")
    if not signature:
        logger.warning("Paystack webhook received without signature header.")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "No signature header"})

    try:
        request_body = await request.body()
    except Exception as e:
        logger.error(f"Failed to get raw request body for Paystack webhook: {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Failed to read request body"})

    if not verify_paystack_signature(request_body, signature, settings.PAYSTACK_SECRET_KEY):
        logger.warning("Paystack webhook signature verification failed.")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Invalid signature"})

    try:
        data = json.loads(request_body)
        event_type = data.get("event")
        paystack_reference = data.get("data", {}).get("reference")
        paystack_status = data.get("data", {}).get("status")
        paystack_gateway_response = data.get("data", {}).get("gateway_response")
        paystack_amount_kobo = data.get("data", {}).get("amount")
        order_id = data.get("data", {}).get("metadata", {}).get("order_id")

        logger.info(f"Processing Paystack event '{event_type}' for reference '{paystack_reference}' (Paystack status: {paystack_status}). Order ID: {order_id}")

        if not order_id:
            logger.error(f"Paystack webhook event '{event_type}' received without order_id in metadata. Reference: {paystack_reference}")
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "order_id not found in metadata"})

        if not supabase:
            logger.error("Supabase client not available for payment-success webhook processing.")
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Database connection unavailable"})

        order_res = supabase.table("orders").select("id, user_id, payment_status, status, total_amount, total_with_delivery, delivery_type").eq("id", order_id).limit(1).execute()

        if not order_res.data:
            logger.warning(f"Paystack webhook: No order found in DB for order_id {order_id} (Paystack ref: {paystack_reference}).")
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Order not found in internal database."})

        order = order_res.data[0]
        current_payment_status: PaymentStatus = order.get("payment_status", DefaultStatus.PAYMENT_UNPAID)
        current_order_status: OrderStatus = order.get("status", DefaultStatus.ORDER_PENDING_PAYMENT)

        expected_total_kobo = int((order.get('total_with_delivery') or order.get('total_amount')) * 100)

        update_needed = False
        update_data: Dict[str, Any] = {
             "paystack_reference": paystack_reference,
             "amount_paid_kobo": paystack_amount_kobo,
             "updated_at": datetime.now().isoformat()
        }
        notification_prefix = "ℹ️ Order Update"
        notification_suffix = ""
        new_payment_status = current_payment_status
        new_order_status = current_order_status


        # --- Process specific event types ---
        if event_type == "charge.success":
            if paystack_status == 'success':
                 if paystack_amount_kobo is not None and paystack_amount_kobo >= expected_total_kobo:
                     if current_payment_status != DefaultStatus.PAYMENT_PAID:
                         new_payment_status = DefaultStatus.PAYMENT_PAID
                         new_order_status = DefaultStatus.ORDER_PROCESSING
                         update_needed = True
                         notification_prefix = "✅ Payment confirmed!"
                         notification_suffix = " We are now preparing your items."
                         logger.info(f"Order {order_id}: Full payment success.")
                     else:
                         logger.info(f"Order {order_id}: Already marked PAID, skipping update.")
                         return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "success", "message": "Order already processed"})

                 elif paystack_amount_kobo is not None and paystack_amount_kobo > 0: # Partial payment
                     if current_payment_status not in [DefaultStatus.PAYMENT_PAID, DefaultStatus.PAYMENT_PARTIALLY_PAID]:
                         new_payment_status = DefaultStatus.PAYMENT_PARTIALLY_PAID
                         update_needed = True
                         notification_prefix = "⚠️ Partial Payment Confirmed!"
                         notification_suffix = f" Received GHS {(paystack_amount_kobo/100):.2f} of GHS {(expected_total_kobo/100):.2f}. Please contact support to complete the remaining payment."
                         logger.warning(f"Order {order_id}: Partial payment received.")
                     else:
                          logger.info(f"Order {order_id}: Already marked {current_payment_status}, skipping partial payment update.")
                          return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "success", "message": "Order already processed"})
                 else:
                      logger.error(f"Paystack charge.success event with amount_kobo={paystack_amount_kobo} for order {order_id}.")
                      notification_prefix = "❌ Payment Issue"
                      notification_suffix = " There was an issue with the payment amount received. Please contact support."

            elif paystack_status == 'failed':
                 if current_payment_status not in [DefaultStatus.PAYMENT_PAID, DefaultStatus.PAYMENT_PARTIALLY_PAID, DefaultStatus.PAYMENT_FAILED]:
                     new_payment_status = DefaultStatus.PAYMENT_FAILED
                     new_order_status = DefaultStatus.ORDER_FAILED
                     update_needed = True
                     notification_prefix = "❌ Payment Failed"
                     notification_suffix = " Please try again or contact support if the amount was deducted."
                     logger.info(f"Order {order_id}: Payment explicitly failed according to Paystack.")
                 else:
                      logger.info(f"Order {order_id}: Payment already marked {current_payment_status}, ignoring 'failed' status from Paystack.")
                      return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "success", "message": "Order already processed"})
            else:
                 logger.warning(f"Paystack charge.success event with unexpected status '{paystack_status}' for order {order_id}.")


        # Add handling for other events if they are relevant
        # elif event_type == "charge.dispute.create": ...
        # elif event_type == "transfer.success": ...


        if update_needed:
            update_data["payment_status"] = new_payment_status
            if new_order_status != current_order_status:
                 update_data["status"] = new_order_status

            try:
                supabase.table("orders").update(update_data).eq("id", order_id).execute()
                logger.info(f"Order {order_id} successfully updated to payment_status '{new_payment_status}' and status '{new_order_status}'.")
            except Exception as e:
                logger.error(f"Failed to update order {order_id} in DB after Paystack webhook processing: {e}", exc_info=True)
                return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Failed to update order in database"})
        else:
             logger.info(f"No payment/order status update needed for order {order_id}. Event {event_type}, current payment {current_payment_status}, current order {current_order_status}.")


        # --- Notify User via WhatsApp ---
        # Only send notification if update was needed OR if it was a failure event that needs reporting
        if update_needed or (event_type == "charge.success" and paystack_status == 'failed'):
             user_query = supabase.table("users").select("phone_number").eq("id", order["user_id"]).limit(1).execute()

             if user_query.data and send_whatsapp_message_available:
                 phone_number = user_query.data[0]["phone_number"]
                 total_paid_display = (paystack_amount_kobo / 100.0) if paystack_amount_kobo is not None else "N/A"

                 notification_message = (
                     f"{notification_prefix}\n\n"
                     f"Your Order ID is: {order_id}.\n"
                     f"Amount: *GHS {total_paid_display:.2f}*.\n"
                     f"Delivery Type: *{order.get('delivery_type', 'Not specified').title()}*.\n"
                     f"Reference: {paystack_reference}\n\n"
                     f"{notification_suffix}"
                 )

                 try:
                     await send_whatsapp_message(phone_number, notification_message)
                     logger.info(f"Notified user {order['user_id']} ({phone_number}) about order {order_id} payment status '{new_payment_status}'.")
                     try:
                         supabase.table("users").update({"last_bot_message": notification_message}).eq("id", order['user_id']).execute();
                         logger.info(f"Saved last bot message for user {order['user_id']} from /payment-success.");
                     except Exception as e:
                         logger.error(f"Failed to save last bot message for user {order['user_id']} from /payment-success: {e}", exc_info=True);

                 except Exception as e:
                      logger.error(f"Failed to send WhatsApp notification for paid order {order_id} to {phone_number}: {e}", exc_info=True)

             elif not send_whatsapp_message_available:
                 logger.error(f"send_whatsapp_message is not available. Cannot notify user {order['user_id']} about order {order_id}")
             else:
                 logger.error(f"Could not find user {order['user_id']} to notify for order {order_id}")


        # Return 200 OK to Paystack
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "success", "message": f"Event {event_type} processed successfully"})

    except json.JSONDecodeError:
        logger.error("Paystack webhook received non-JSON body after signature verification.")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Invalid JSON body"})
    except Exception as e:
        logger.error(f"Unexpected error processing Paystack webhook payload for reference {paystack_reference}: {str(e)}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": f"Internal server error during processing: {e}"})


# Export the app for Vercel
app.debug = False

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up.")
    if not supabase:
        logger.error("Supabase client not available at startup.")
    if not send_whatsapp_message_available:
         logger.error("send_whatsapp_message utility is not available. WhatsApp communication will be disabled.")
    if not settings.PAYSTACK_SECRET_KEY:
         logger.warning("PAYSTACK_SECRET_KEY is not set. Payment link generation will use mock links, webhook verification will fail.")
    # Add checks for other critical settings/connections

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down.")