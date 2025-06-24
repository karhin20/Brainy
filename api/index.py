import os
import sys
import uuid
import httpx
import json
import logging
import hmac
import hashlib
import math
from datetime import datetime, timedelta, timezone
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
    # Correct relative import path for modules within the same package
    # Assuming index.py is in the package root or directly next to these modules
    # If index.py is in a subdir like 'api', this needs adjustment
    from . import admin_router, security, auth_router, public_router
    from .utils import send_whatsapp_message
    send_whatsapp_message_available = True
except ImportError as e:
    logger.error(f"Failed to import internal modules (routers, security, utils): {e}", exc_info=True)
    # Assign None only if the specific import failed
    # Re-import if necessary to ensure names exist even if None
    try:
        from . import admin_router, security, auth_router, public_router
    except ImportError:
        admin_router, security, auth_router, public_router = None, None, None, None

    try:
        from .utils import send_whatsapp_message as _swm # Import with alias if possible
        send_whatsapp_message_available = True
        send_whatsapp_message = _swm # Re-assign the imported function
    except ImportError:
         send_whatsapp_message_available = False
         async def send_whatsapp_message(to: str, body: str):
             logger.error(f"send_whatsapp_message utility is not available. Tried to send to {to}: {body}")
    except Exception as ex:
         logger.error(f"Unexpected error assigning send_whatsapp_message: {ex}", exc_info=True)
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
    ORDER_AWAITING_LOCATION: OrderStatus = "awaiting_location"
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
    allow_origins=["*", "http://localhost:8080"], # Restrict this in production
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
# Check if security module was imported successfully before using it in dependencies
if security and admin_router:
    app.include_router(admin_router.router, prefix="/admin", tags=["admin"], dependencies=[Depends(security.get_admin_user)]) # Assuming admin routes need admin auth
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
    whatsapp_util_status = "ok" if send_whatsapp_message_available else "unavailable"
    paystack_key_status = "set" if settings.PAYSTACK_SECRET_KEY else "not set (mock links active)"
    gemini_key_status = "set" if settings.GEMINI_API_KEY else "not set (fallback active)"


    # Add more robust external service checks here if needed
    return {
        "status": "healthy",
        "database": db_status,
        "whatsapp_utility": whatsapp_util_status,
        "paystack_secret_key": paystack_key_status,
        "gemini_api_key": gemini_key_status,
        "external_services": {}, # Placeholder
        "timestamp": datetime.now(timezone.utc).isoformat()
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
            - `affirmative_acknowledgement`: User is confirming information or giving a simple positive reply (e.g., okay, ok, got it, sounds good, sure).
            - `negative_acknowledgement`: User is providing a negative reply, indicating they don't need further help (e.g., no, nope, that's all, I'm good).
            - `multi_intent`: User is asking for two or more things at once (e.g., "I want to buy tomatoes and where is my order?").
            - `modify_order`: User wants to change, add, or remove items from an order they've just confirmed but haven't paid for yet.
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
                - `affirmative_acknowledgement`: "Great! Is there anything else I can help you with?"
                - `negative_acknowledgement`: "Alright, have a great day! Feel free to message me anytime you need groceries."
                - `multi_intent`: "I can help with one thing at a time. Please ask about your order status or placing a new order separately."
                - `modify_order`: "No problem, I can help with that."
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
            # Safely navigate nested dictionary, providing default empty values
            json_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")

            if not json_text:
                 error_msg_detail = "No text payload received."
                 finish_reason = result.get("candidates", [{}])[0].get("finishReason")
                 if finish_reason:
                     error_msg_detail = f"API finished with reason: {finish_reason}"

                 logger.error(f"Gemini API response missing expected text payload: {result}. Detail: {error_msg_detail}")
                 # Include finishReason in the error message if available
                 raise ValueError(f"Gemini API returned empty text payload or error: {finish_reason or 'Unknown API Error'}")


            # Attempt to parse the JSON text
            return json.loads(json_text) # Raise JSONDecodeError if invalid

    except httpx.RequestError as e:
        logger.error(f"Gemini API communication error: {e}", exc_info=True)
        raise # Re-raise the exception
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding Gemini response JSON: {e}. Raw text was: {json_text}", exc_info=True)
        raise # Re-raise the exception
    except ValueError as e: # From missing text payload check or GEMINI_API_KEY not set
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
             
             # Multi-intent check
             buy_present = any(word in lower_msg for word in ["buy", "want", "order", "menu"])
             status_present = any(word in lower_msg for word in ["status", "track", "where is my"])
             if buy_present and status_present:
                 return {"intent": "multi_intent", "response": "I can help with one thing at a time. Please ask about your order status or placing a new order separately."}
             
             if any(word in lower_msg for word in ["add", "change", "modify", "forgot"]):
                return {"intent": "modify_order", "response": "No problem, I can help with that."}

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
             if lower_msg in ["ok", "okay", "k", "ok."]:
                 return {"intent": "affirmative_acknowledgement", "response": "Great! Is there anything else I can help you with?"}
             if lower_msg in ["no", "nope", "no thanks", "no."]:
                 return {"intent": "negative_acknowledgement", "response": "Alright, have a great day! Feel free to message me anytime you need groceries."}
             return {"intent": "unknown", "response": "I'm sorry, I can only assist with grocery orders. Could you please rephrase?"}
         else:
             # Handle other ValueErrors from call_gemini_api (e.g. empty text payload)
             logger.error(f"Error processing Gemini response: {e}", exc_info=True)
             return {"intent": "unknown", "response": f"There was an issue processing the response from my service. Please try again."} # Removed detailed error for user


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
    if amount <= 0:
        logger.warning(f"Attempted to generate payment link for non-positive amount: {amount:.2f} for order {order_id}")
        raise ValueError("Amount must be positive to generate payment link.")

    if not settings.PAYSTACK_SECRET_KEY:
        logger.warning("PAYSTACK_SECRET_KEY not set, cannot generate real payment link. Using mock link.")
        # Using frontend URL as a mock payment success page
        return f"{settings.FRONTEND_URL}/payment-success?order_id={order_id}&amount={amount:.2f}&mock_success=true"

    # --- CORRECTED INDENTATION ---
    # The following block should only execute if PAYSTACK_SECRET_KEY IS set.
    headers = {
        "Authorization": f"Bearer {settings.PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    placeholder_email = f"{''.join(filter(str.isdigit, user_phone))}@market.bot"
    unique_reference = f"{order_id}_{int(datetime.now().timestamp())}"

    payload = {
        "email": placeholder_email,
        "amount": int(amount * 100), # Amount in pesewas/kobo
        "currency": "GHS",
        "reference": unique_reference,
        "callback_url": f"{settings.FRONTEND_URL}/payment-success?order_id={order_id}", # Paystack calls this URL
        "channels": ["card", "mobile_money", "bank", "bank_transfer", "ussd"], # Include common channels
        "metadata": {"order_id": order_id, "phone": user_phone, "reference": unique_reference} # Include ref in metadata too
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(settings.PAYSTACK_PAYMENT_URL, headers=headers, json=payload)
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses

            data = response.json()
            # Paystack returns { "status": true, "message": "...", "data": { ... authorization_url ... } }
            if data.get("status") is True and data.get("data") and data["data"].get("authorization_url"):
                logger.info(f"Successfully generated Paystack link for order {order_id}. Reference: {unique_reference}")
                return data["data"]["authorization_url"]
            else:
                # Log the full response for debugging if expected structure is missing
                logger.error(f"Paystack API returned success=true but missing auth_url or unexpected data for order {order_id}: {data}")
                raise ValueError("Paystack API response format error during link generation.")

    except httpx.RequestError as e:
         logger.error(f"Paystack API request error for order {order_id}: {e}", exc_info=True)
         if isinstance(e, httpx.HTTPStatusError):
              if e.response.status_code in [401, 403]:
                  raise Exception("Payment gateway authentication failed. Please contact support.") from e
              elif e.response.status_code == 400:
                   # Include Paystack's error message if available
                   error_detail = e.response.text
                   try:
                       error_json = e.response.json()
                       error_detail = error_json.get('message', error_detail)
                   except json.JSONDecodeError:
                        pass # Ignore if response isn't JSON
                   raise Exception(f"Invalid request to payment gateway: {error_detail}") from e
              else:
                   raise Exception(f"Payment gateway error ({e.response.status_code}). Please try again later.") from e
         else:
              raise Exception(f"Could not connect to payment gateway: {e}. Please check your internet connection.") from e
    except Exception as e:
        # Catch any other unexpected errors during JSON parsing or data extraction
        logger.error(f"Unexpected error during Paystack link generation for order {order_id}: {str(e)}", exc_info=True)
        raise Exception(f"An unexpected error occurred during payment link generation.") from e # Generic error for user


def verify_paystack_signature(request_body: bytes, signature: str, secret_key: Optional[str]) -> bool:
    """
    Verifies the Paystack webhook signature using the main Secret Key.
    Returns True if signature is valid, False otherwise.
    """
    if not secret_key:
        logger.error("PAYSTACK_SECRET_KEY is not set. Cannot verify Paystack webhook signature. THIS IS A MAJOR SECURITY RISK.")
        return False

    try:
        # CORRECTED: Removed .encode() from request_body as it's already bytes
        expected_signature = hmac.new(
            secret_key.encode('utf-8'), # Secret key needs encoding
            request_body,             # Body is already bytes
            hashlib.sha512
        ).hexdigest()

        is_valid = hmac.compare_digest(expected_signature, signature)

        if not is_valid:
            logger.warning("Paystack webhook signature verification failed.")
        #else: # Keep logging successful verification only at info level if needed elsewhere, or remove for brevity
        #    logger.info("Paystack webhook signature verified successfully.")

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
    Returns the message to be sent back to the user.
    """
    user_id = user['id']
    intent_data = gemini_result
    intent = intent_data.get("intent")
    ai_response_ack = intent_data.get("response", "Okay.") # Default fallback ack

    logger.info(f"Handling new conversation for user {user_id}. Intent: {intent}. Is New User: {is_new_user}. Message: '{original_message}'")

    reply_message = "" # Initialize reply message here

    if intent == "buy":
        # Ensure supabase is available for database operations
        if not supabase:
             logger.error(f"Supabase client not available for 'buy' intent session creation for user {user_id}.")
             return "Sorry, I'm currently having technical difficulties and cannot start a new order. Please try again later."

        session_token = str(uuid.uuid4())
        selection_url = f"{settings.FRONTEND_URL}?session={session_token}"
        try:
            # Add a check for existing active sessions for this user/phone if needed,
            # though current logic relies on the /confirm-items cleanup.
            # For simplicity, proceeding with insert.
            now_utc = datetime.now(timezone.utc)
            insert_res = supabase.table("sessions").insert({
                "user_id": user_id,
                "phone_number": from_number,
                "session_token": session_token,
                "last_intent": "buy", # Store intent for session context if needed later
                "created_at": now_utc.isoformat(),
                # Sessions expire after 24 hours
                "expires_at": (now_utc + timedelta(hours=24)).isoformat()
            }).execute()
            # Check result for success if the supabase client version supports it returning data/status
            if not insert_res.data:
                logger.warning(f"Supabase session insert executed but returned no data for user {user_id}, token {session_token}.")

            logger.info(f"Created new session {session_token} for user {user_id}")

            # Combine AI acknowledgement with menu link message
            greeting_part = "Welcome! " if is_new_user else ""
            reply_message = (
                f"{ai_response_ack} {greeting_part}" # E.g., "Okay, let's start your order. Welcome! "
                f"You can select the fresh items you'd like to purchase from our online menu here:\n"
                f"{selection_url}"
            )
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id} (buy intent): {e}", exc_info=True)
            reply_message = "Sorry, I'm having trouble starting a new order right now. Please try again in a moment."

    elif intent == "check_status":
        if not supabase:
            logger.error(f"Supabase client not available for 'check_status' intent for user {user_id}.")
            return f"{ai_response_ack} I'm having trouble checking your order details because my database is unavailable."

        try:
            # Look for the latest paid order that's not cancelled or delivered.
            # Select ALL relevant fields including status, type, fees, location if available
            active_paid_orders_res = supabase.table("orders").select("status, id, order_number, delivery_type, delivery_fee, total_amount, total_with_delivery, delivery_location_lat, delivery_location_lon").eq("user_id", user_id).in_("payment_status", [DefaultStatus.PAYMENT_PAID, DefaultStatus.PAYMENT_PARTIALLY_PAID]).not_.in_("status", [DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_FAILED]).order("created_at", desc=True).limit(1).execute()

            # Provide detailed message based on the found order's specific status
            if active_paid_orders_res.data:
                latest_order = active_paid_orders_res.data[0]
                order_id = latest_order['id']
                order_number = latest_order.get('order_number', order_id) # Fallback to id
                current_order_status = latest_order['status']
                delivery_type = latest_order.get('delivery_type', 'N/A')
                total_display = latest_order.get('total_with_delivery') or latest_order.get('total_amount')
                total_display_formatted = f"{total_display:.2f}" if total_display is not None else "N/A"

                # Build a more conversational, status-specific message
                reply_message = f"{ai_response_ack} " # Start with the AI acknowledgement

                if current_order_status == DefaultStatus.ORDER_PROCESSING:
                    if delivery_type == 'delivery':
                        reply_message += f"Your order *{order_number}* is currently being prepared. We'll notify you as soon as it's out for delivery!"
                    else: # pickup
                        reply_message += f"Your order *{order_number}* is currently being prepared for pickup. We'll send you a message the moment it's ready!"
                elif current_order_status == DefaultStatus.ORDER_OUT_FOR_DELIVERY:
                     reply_message += f"Great news! Your order *{order_number}* is out for delivery and should be with you soon."
                elif current_order_status == DefaultStatus.ORDER_DELIVERED:
                     # This case is primarily handled by the 'else' block below, but here for defense
                     reply_message += f"Your latest order ({order_number}) has already been delivered."
                elif current_order_status == DefaultStatus.ORDER_PENDING_PAYMENT:
                     # This case should be rare for paid orders, but here for defense
                     reply_message += f"You have a pending order ({order_number}) waiting for payment. Total: GHS {total_display_formatted}."
                else: # A fallback for any other active statuses
                    status_display = current_order_status.replace('-', ' ').title()
                    delivery_type_display = delivery_type.replace('_', ' ').title()
                    reply_message += f"Your latest active order ({order_number}) is currently *{status_display}* ({delivery_type_display})."

            else:
                # No active paid orders found, check for a recent delivered one
                recent_delivered_res = supabase.table("orders").select("id, order_number").eq("user_id", user_id).eq("status", DefaultStatus.ORDER_DELIVERED).order("created_at", desc=True).limit(1).execute()
                if recent_delivered_res.data:
                    order_number = recent_delivered_res.data[0].get('order_number', recent_delivered_res.data[0]['id'])
                    reply_message = f"{ai_response_ack} Your latest order ({order_number}) has already been delivered. Would you like to start a new one?"
                else:
                     reply_message = f"{ai_response_ack} It looks like you don't have any active orders right now. Would you like to place one?"
        except Exception as e:
            logger.error(f"Error checking order status for user {user_id}: {e}", exc_info=True)
            reply_message = f"{ai_response_ack} I'm having trouble looking up your order details right now. Please try again in a moment."

    elif intent == "cancel_order":
        # Combine AI acknowledgement with the 'no pending' message
        reply_message = f"{ai_response_ack} You don't have any pending orders to cancel right now." # This is the case when *no* pending order is found before this function is called.

    elif intent == "greet":
        # Combine AI acknowledgement with the welcome/welcome back message
        if is_new_user:
            reply_message = (
                f"{ai_response_ack}\n\n" # E.g., "Hello! How can I assist you today?"
                "I can help you order fresh groceries.\n\n"
                "To start, just say 'I want to buy...' or 'Show me the menu'."
            )
        else:
            # For returning users, the AI response is already a polite and complete greeting.
            reply_message = ai_response_ack


    elif intent == "help":
        # Combine AI ack with help text
         reply_message = (
             f"{ai_response_ack}\n\n" # Start with AI's canned "I can help with that. What do you need?"
             "I can help you with the following:\n\n" # Clearly introduce the list
             "🛒 *Starting a new grocery order:*\n Just say 'I want to buy...' or 'Show me the menu'.\n\n"
             "📦 *Checking your order status:*\n Ask 'Where is my order?' or 'What is the status of my order?'.\n\n"
             "❌ *Cancelling a pending order:*\n If you have an order waiting for payment, reply 'cancel'.\n\n"
             "What can I specifically assist you with regarding your groceries?" # Re-prompt
         )

    elif intent == "repeat":
        last_msg = user.get('last_bot_message')
        # Combine AI acknowledgement with the repeated message
        if last_msg:
            reply_message = f"{ai_response_ack}\n\nI last said:\n> {last_msg}"
        else:
            reply_message = f"{ai_response_ack} I don't have a recent message to repeat right now."

    elif intent == "thank_you":
        # Already returns AI ack, fits the pattern.
         reply_message = ai_response_ack

    elif intent == "affirmative_acknowledgement":
        reply_message = ai_response_ack

    elif intent == "negative_acknowledgement":
        reply_message = ai_response_ack

    elif intent == "multi_intent":
        reply_message = ai_response_ack

    elif intent == "modify_order":
        # This case is primarily handled inside the pending_order logic.
        # This is a fallback if it's detected in a new conversation.
        reply_message = f"{ai_ack} It looks like you don't have an active order to modify. Would you like to start a new one?"

    else: # unknown or any other unhandled intent by Gemini/fallback
        logger.info(f"User {user_id} sent message with unknown intent '{intent}'. Message: '{original_message}')")
        # The default response from get_intent_gracefully handles the "unknown" case
        reply_message = ai_response_ack # This should be "I'm sorry, I can only help with grocery orders."


    return reply_message # Return the determined message


def calculate_delivery_fee(lat: float, lon: float) -> float:
    """
    Calculates the delivery fee based on the distance from a central point.
    Uses Haversine formula. SHOULD BE REPLACED with a proper geospatial service or zones in production.
    """
    # Coordinates for a central point in Accra (e.g., Kwame Nkrumah Interchange)
    central_lat, central_lon = 5.5560, -0.2057

    # Validate coordinates to prevent mathematical errors with extreme values
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        logger.error(f"Invalid coordinates provided for delivery fee calculation: {lat},{lon}")
        # Return a high default fee or raise an error depending on desired behavior
        return 100.00 # Return a high fee for invalid input

    R = 6371 # Radius of Earth in kilometers

    lat1_rad = math.radians(central_lat)
    lon1_rad = math.radians(central_lon)
    lat2_rad = math.radians(lat)
    lon2_rad = math.radians(lon)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Use abs() for dlat and dlon just in case, though the formula should handle signs
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) # atan2 handles edge cases better than simple atan

    distance_km = R * c # Distance in km

    # Define fee based on distance tiers (adjust thresholds and fees as needed)
    # Using a simple linear increase with thresholds
    fee = 0.0

    if distance_km < 2: # Very close
        fee = 10.00
    elif distance_km < 5:
        fee = 15.00
    elif distance_km < 10:
        fee = 25.00
    elif distance_km < 15:
         fee = 30.00
    elif distance_km < 20:
         fee = 35.00
    elif distance_km < 25:
         fee = 40.00
    elif distance_km < 30:
         fee = 45.00
    else: # Beyond 30km, potentially cap or scale differently
        fee = 50.00 + (distance_km - 30) * 1.5 # Example: Add GHS 1.50 for every KM over 30
        # Set a hard cap if needed
        # fee = min(fee, 100.00) # Example cap at GHS 100

        if distance_km > 50:
             logger.warning(f"Delivery requested significantly outside standard range: {distance_km:.2f} km from center {central_lat},{central_lon}. Calculated fee: {fee:.2f}")


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
    # Initialize reply_message early
    reply_message = ""
    from_number_clean = "unknown" # Default for logging if 'From' is missing

    try:
        form_data = await request.form()
        from_number = form_data.get("From")

        if not from_number:
            logger.warning("Received webhook without 'From' number field.")
            # Return 200 OK to prevent repeated attempts from webhook provider
            return JSONResponse(content={"detail": "No 'From' number"}, status_code=status.HTTP_200_OK)

        from_number_clean = from_number.replace("whatsapp:", "")
        incoming_msg = form_data.get("Body", "").strip() # Capture the incoming message

        logger.info(f"Received message from {from_number_clean}. Form Data Keys: {list(form_data.keys())}")
        if incoming_msg:
            logger.info(f"Message Body: '{incoming_msg}'")

        # --- Ensure Supabase is available ---
        if not supabase:
             logger.error(f"Supabase client not available for processing message from {from_number_clean}.")
             if send_whatsapp_message_available:
                 await send_whatsapp_message(from_number_clean, "Sorry, I'm currently experiencing technical difficulties. Please try again later.")
             return JSONResponse(content={"detail": "Database unavailable"}, status_code=status.HTTP_200_OK)


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

        # --- 1. Find or Create User ---
        user_res = supabase.table("users").select("*").eq("phone_number", from_number_clean).limit(1).execute()
        user = user_res.data[0] if user_res.data else None
        is_new_user = False

        if not user:
            is_new_user = True
            try:
                # CORRECTED: Removed .select("*") from the insert query - execute() might still return data depending on version
                insert_res = supabase.table("users").insert({"phone_number": from_number_clean}).execute()
                # Assuming execute() on insert returns data in this client version
                if insert_res.data:
                    user = insert_res.data[0]
                    logger.info(f"Created new user with id {user['id']} for phone {from_number_clean}")
                else:
                     # Handle case where insert succeeds but returns no data
                     logger.error(f"Supabase insert executed but returned no data for new user {from_number_clean}.")
                     # Attempt to fetch the user again immediately
                     user_res_after_insert = supabase.table("users").select("*").eq("phone_number", from_number_clean).limit(1).execute()
                     if user_res_after_insert.data:
                          user = user_res_after_insert.data[0]
                          logger.info(f"Successfully fetched newly created user {user['id']} after no data return.")
                     else:
                         # Critical failure: User creation failed or fetch failed
                         logger.critical(f"Failed to create AND retrieve new user for {from_number_clean}.")
                         if send_whatsapp_message_available:
                            await send_whatsapp_message(from_number_clean, "Sorry, I'm having trouble setting up your profile right now. Please try again in a moment.")
                         return JSONResponse(content={}, status_code=status.HTTP_200_OK)


            except Exception as e:
                 logger.error(f"Failed to create new user for {from_number_clean}: {e}", exc_info=True)
                 if send_whatsapp_message_available:
                    await send_whatsapp_message(from_number_clean, "Sorry, I'm having trouble setting up your profile right now. Please try again in a moment.")
                 return JSONResponse(content={}, status_code=status.HTTP_200_OK)

        user_id = user['id']
        # Update last_active timestamp (fire and forget, doesn't need to block)
        try:
             supabase.table("users").update({"last_active": datetime.now(timezone.utc).isoformat()}).eq("id", user['id']).execute()
        except Exception as e:
             logger.error(f"Failed to update last_active for user {user_id}: {e}", exc_info=True)

        # --- 2. Find and Clean Up Potentially Multiple Pending Orders ---
        # Find the most recent pending order
        latest_pending_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("payment_status", DefaultStatus.PAYMENT_UNPAID).not_.in_("status", [DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_FAILED]).order("created_at", desc=True).limit(1).execute()
        active_pending_order = latest_pending_res.data[0] if latest_pending_res.data else None

        # If a recent pending order exists, find and cancel any older ones
        if active_pending_order:
            latest_pending_created_at = active_pending_order['created_at']
            older_pending_res = supabase.table("orders").select("id").eq("user_id", user_id).eq("payment_status", DefaultStatus.PAYMENT_UNPAID).not_.in_("status", [DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_FAILED]).lt("created_at", latest_pending_created_at).execute()

            if older_pending_res.data:
                older_order_ids = [o['id'] for o in older_pending_res.data]
                logger.info(f"Cancelling {len(older_order_ids)} older pending orders for user {user_id}: {older_order_ids}")
                try:
                    # Use a single update statement for efficiency
                    supabase.table("orders").update({
                        "status": DefaultStatus.ORDER_CANCELLED,
                        "payment_status": DefaultStatus.PAYMENT_CANCELLED,
                        "cancelled_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "cancellation_reason": "superseded by newer pending order"
                    }).in_("id", older_order_ids).execute()
                    logger.info(f"Cancelled older pending orders: {older_order_ids}")
                except Exception as e:
                     logger.error(f"Failed to cancel older pending orders for user {user_id}: {e}", exc_info=True)


        # --- 3. Main Logic Branching: Handle Message Type and Order State ---

        # A. Handle incoming Location message if it's valid
        if is_location_message and latitude is not None and longitude is not None:
            # Check if we are expecting a location for the active pending order
            if active_pending_order and active_pending_order['status'] in [DefaultStatus.ORDER_AWAITING_LOCATION, DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION]:
                order_id = active_pending_order['id']
                logger.info(f"Processing location for order {order_id} from user {user_id}. Current status: {active_pending_order['status']}")

                try:
                    # Save the received location regardless of whether it's new or confirming saved
                    location_to_save = json.dumps({"latitude": str(latitude), "longitude": str(longitude)})
                    supabase.table("users").update({"last_known_location": location_to_save}).eq("id", user_id).execute()
                    logger.info(f"Saved location {latitude},{longitude} for user {user_id}")

                    delivery_fee = calculate_delivery_fee(latitude, longitude)
                    total_with_delivery = active_pending_order['total_amount'] + delivery_fee

                    update_data = {
                        "status": DefaultStatus.ORDER_PENDING_PAYMENT, # Move to payment stage
                        "delivery_type": "delivery", # Confirm delivery type
                        "delivery_fee": delivery_fee,
                        "total_with_delivery": total_with_delivery,
                        "delivery_location_lat": latitude,
                        "delivery_location_lon": longitude,
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }
                    supabase.table("orders").update(update_data).eq("id", order_id).execute()
                    logger.info(f"Updated order {order_id} with delivery details and status {DefaultStatus.ORDER_PENDING_PAYMENT}")

                    payment_link = await generate_paystack_payment_link(order_id, total_with_delivery, from_number_clean)

                    # Add a note if the location was saved/updated
                    location_note = "Using the location you shared."
                    if active_pending_order['status'] == DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION:
                         location_note = "Using the new location you shared." # Clarify if they were confirming

                    reply_message = (
                        f"{location_note} Your delivery fee is GHS {delivery_fee:.2f}.\n\n"
                        f"Your new total is *GHS {total_with_delivery:.2f}*.\n\n"
                        f"Please complete your payment here to confirm your order:\n{payment_link}\n\n"
                        "Or reply 'cancel' to cancel the order."
                    )
                except Exception as e:
                     logger.error(f"Error processing location or finalizing order for user {user_id}, order {order_id}: {e}", exc_info=True)
                     reply_message = f"Sorry, I encountered an issue processing your delivery details. Please try again or contact support."

            else:
                 # User sent a location when we weren't expecting one or no pending order exists
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
            order_id = active_pending_order['id'] if active_pending_order else None
            order_number = active_pending_order.get('order_number', order_id) if active_pending_order else None
            current_status: Optional[OrderStatus] = active_pending_order.get('status') if active_pending_order else None

            if active_pending_order:
                logger.info(f"Handling text message for pending order {order_id} (status: {current_status}): '{incoming_msg}'")

                # --- Check for specific commands within the current pending state ---
                if current_status == DefaultStatus.ORDER_PENDING_CONFIRMATION:
                     if lower_incoming_msg in ["1", "delivery"]:
                         handled_by_pending_state = True
                         # User selected Delivery. Check if they have a saved location.
                         if user.get("last_known_location"):
                             # User has a saved location, ask if they want to use it
                             try:
                                # Update status to await confirmation for saved location
                                supabase.table("orders").update({"status": DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", order_id).execute()
                                logger.info(f"Order {order_id} status updated to {DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION}")
                                reply_message = (
                                    "I see you have a saved delivery location with us. Would you like to use it for this order?\n\n"
                                    "Reply *1* to use saved location\n"
                                    "Reply *2* to provide a new one (you'll share it via WhatsApp location feature)\n\n"
                                    "Or reply 'cancel'."
                                )
                             except Exception as e:
                                  logger.error(f"Failed to update order status for location confirmation {order_id}: {e}", exc_info=True)
                                  reply_message = "Sorry, I had trouble updating your order details. Please try again or reply 'cancel'."
                         else:
                             # User has no saved location, ask them to share it
                             try:
                                # Update status to await location sharing
                                supabase.table("orders").update({"delivery_type": "delivery", "status": DefaultStatus.ORDER_AWAITING_LOCATION, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", order_id).execute()
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
                            # Update status to pending payment for pickup
                            supabase.table("orders").update({"delivery_type": "pickup", "status": DefaultStatus.ORDER_PENDING_PAYMENT, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", order_id).execute()
                            logger.info(f"Order {order_id} status updated to {DefaultStatus.ORDER_PENDING_PAYMENT} (pickup)")

                            # Generate payment link for the original total amount
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
                            # This case should ideally not happen if the user was prompted based on having a saved location,
                            # but handle defensively.
                            logger.warning(f"User {user_id} replied '1' to location confirmation but had no saved location. Order {order_id}")
                            try:
                                # Revert status to awaiting location sharing
                                supabase.table("orders").update({"status": DefaultStatus.ORDER_AWAITING_LOCATION, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", order_id).execute()
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
                            # Saved location exists, use it to finalize the order
                            try:
                                location_data = json.loads(location_str)
                                latitude = float(location_data.get("latitude"))
                                longitude = float(location_data.get("longitude"))

                                # Calculate fee and total with delivery for the saved location
                                delivery_fee = calculate_delivery_fee(latitude, longitude)
                                total_with_delivery = active_pending_order['total_amount'] + delivery_fee

                                # Update order with delivery details and move to pending payment
                                update_data = {
                                    "status": DefaultStatus.ORDER_PENDING_PAYMENT,
                                    "delivery_type": "delivery",
                                    "delivery_fee": delivery_fee,
                                    "total_with_delivery": total_with_delivery,
                                    "delivery_location_lat": latitude,
                                    "delivery_location_lon": longitude,
                                    "updated_at": datetime.now(timezone.utc).isoformat()
                                }
                                supabase.table("orders").update(update_data).eq("id", order_id).execute()
                                logger.info(f"Order {order_id} status updated to {DefaultStatus.ORDER_PENDING_PAYMENT} using saved location.")

                                # Generate payment link for the total including delivery fee
                                payment_link = await generate_paystack_payment_link(order_id, total_with_delivery, from_number_clean)
                                reply_message = (
                                    f"Using your saved location. The delivery fee is GHS {delivery_fee:.2f}.\n\n"
                                    f"Your new total is *GHS {total_with_delivery:.2f}*.\n\n"
                                    f"Please complete your payment here:\n{payment_link}\n\n"
                                    "Or reply 'cancel'."
                                )
                            except (ValueError, TypeError, json.JSONDecodeError) as e:
                                # Handle errors if the saved location data is invalid or fee calculation fails
                                logger.error(f"Invalid saved location data or fee processing error for user {user_id}: {location_str}. Error: {e}", exc_info=True)
                                try:
                                    # Revert status to awaiting location sharing if saved data was bad
                                    supabase.table("orders").update({"status": DefaultStatus.ORDER_AWAITING_LOCATION, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", order_id).execute()
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
                                # Catch other errors during the finalization process
                                logger.error(f"Error finalizing order {order_id} after using saved location for user {user_id}: {e}", exc_info=True)
                                reply_message = f"Sorry, I encountered an issue processing your delivery details. Please try again or reply 'cancel'."

                    elif lower_incoming_msg == "2": # Provide a new one
                        handled_by_pending_state = True
                        try:
                            # Update status to await location sharing
                            supabase.table("orders").update({"status": DefaultStatus.ORDER_AWAITING_LOCATION, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", order_id).execute()
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
                # Check for 'cancel' regardless of other state-specific inputs
                if lower_incoming_msg == 'cancel':
                    handled_by_pending_state = True # Mark as handled
                    if current_status in [DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_FAILED]:
                         reply_message = f"This order ({order_number}) is already marked as *{current_status.replace('_', ' ').title()}* and cannot be cancelled."
                    else:
                         try:
                            # Update order status to cancelled
                            supabase.table("orders").update({
                                "status": DefaultStatus.ORDER_CANCELLED,
                                "payment_status": DefaultStatus.PAYMENT_CANCELLED, # Also mark payment as cancelled
                                "cancelled_at": datetime.now(timezone.utc).isoformat(),
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                                "cancellation_reason": "user requested cancellation"
                            }).eq("id", order_id).execute()
                            logger.info(f"Order {order_id} successfully cancelled by user {user_id}.")
                            reply_message = f"Your order ({order_number}) has been cancelled. Please let me know if there's anything else I can help with."
                         except Exception as e:
                            logger.error(f"Failed to cancel order {order_id} for user {user_id}: {e}", exc_info=True)
                            reply_message = "Sorry, I had trouble cancelling your order right now. Please try again or contact support."


                # --- Handle messages that are *not* state-specific commands ---
                # If the message wasn't a state-specific command (like 1, 2, cancel)
                if not handled_by_pending_state:
                    user_context = {'has_paid_order': False, 'has_saved_address': bool(user.get("last_known_location"))} # Context might be useful for Gemini
                    gemini_result = await get_intent_gracefully(incoming_msg, user_context)
                    intent = gemini_result.get('intent')
                    ai_ack = gemini_result.get('response', 'Okay.')

                    if intent == 'modify_order':
                        handled_by_pending_state = True
                        logger.info(f"User {user_id} wants to modify pending order {order_id}. Cancelling and creating new session.")
                        try:
                            # 1. Cancel the current pending order
                            supabase.table("orders").update({
                                "status": DefaultStatus.ORDER_CANCELLED,
                                "payment_status": DefaultStatus.PAYMENT_CANCELLED,
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                                "cancellation_reason": "user modified selection"
                            }).eq("id", order_id).execute()

                            # 2. Create a new session and link, like in 'buy' intent
                            session_token = str(uuid.uuid4())
                            selection_url = f"{settings.FRONTEND_URL}?session={session_token}"
                            now_utc = datetime.now(timezone.utc)
                            supabase.table("sessions").insert({
                                "user_id": user_id,
                                "phone_number": from_number_clean,
                                "session_token": session_token,
                                "created_at": now_utc.isoformat(),
                                "expires_at": (now_utc + timedelta(hours=24)).isoformat()
                            }).execute()

                            # 3. Formulate the reply
                            reply_message = (
                                f"{ai_ack} To change your items, please make a new selection with this link. "
                                f"Your previous cart has been cancelled.\n\n{selection_url}"
                            )
                        except Exception as e:
                            logger.error(f"Error modifying pending order {order_id} for user {user_id}: {e}", exc_info=True)
                            reply_message = "I'm sorry, I ran into a problem trying to modify your order. Please try again."
                        
                    else:
                        # Construct a reminder message based on the current pending status
                        reminder_message = ""
                        if current_status == DefaultStatus.ORDER_PENDING_CONFIRMATION:
                             reminder_message = "\n\nBut please first choose '1' for Delivery or '2' for Pickup for your pending order."
                    # ... existing code ...

                # else: # This else block was misplaced and caused issues. Removed.
                #    reply_message = f"I'm not sure how to help with that right now. You currently have a pending order (ID: {order_id}) in progress. Please complete the next step for that order, or reply 'cancel'."


            # --- Handle incoming Text Message when NO active pending order exists ---
            else: # No active pending order
                logger.info(f"Handling text message for user {user_id} with no active pending order: '{incoming_msg}'")
                # Determine intent for a new conversation
                user_context = {'has_paid_order': False, 'has_saved_address': bool(user.get("last_known_location"))} # Provide relevant user context
                gemini_result = await get_intent_gracefully(incoming_msg, user_context)
                # Call handle_new_conversation to get the appropriate response based on intent
                reply_message = await handle_new_conversation(user, gemini_result, from_number_clean, is_new_user, incoming_msg)


        # D. Handle empty or unhandled message type (if no text, location, or media was processed)
        # This block executes if none of the specific message type handlers above set a reply_message
        if not reply_message:
             logger.info(f"Received an empty or unhandled message type from user {user_id}.")
             # Provide a default welcome or help message
             if is_new_user:
                 reply_message = "👋 Welcome to Fresh Market GH!\n\nI can help you order fresh groceries.\n\nTo start, just say 'I want to buy...' or 'Show me the menu'."
             else:
                  user_name = user.get('name')
                  greeting_name = f", {user_name}!" if user_name else "!"
                  reply_message = f"Hello! Welcome back to Fresh Market GH{greeting_name} How can I help you with your groceries today?"


        # --- 4. Send the determined reply message and update last_bot_message ---
        # Only attempt to send if there is a reply message AND the utility is available
        if reply_message and send_whatsapp_message_available:
             try:
                await send_whatsapp_message(from_number_clean, reply_message)
                logger.info(f"Sent reply to {from_number_clean}.")
                try:
                    # Save the sent message as last_bot_message for 'repeat' intent
                    # Limit length to avoid excessive storage
                    truncated_message = reply_message[:1000] # Limit to first 1000 chars
                    # Use upsert or a check if needed, but simple update is fine for last message
                    supabase.table("users").update({"last_bot_message": truncated_message}).eq("id", user_id).execute();
                    logger.info(f"Saved last bot message for user {user_id}.");
                except Exception as e:
                     logger.error(f"Failed to save last bot message for user {user_id}: {e}", exc_info=True);

             except Exception as send_e:
                # Log sending failure, but don't re-raise as webhook must return 200 OK
                logger.error(f"Failed to send WhatsApp message to {from_number_clean}: {send_e}", exc_info=True)

        elif reply_message and not send_whatsapp_message_available:
            # Log that message couldn't be sent due to utility not being available
            logger.error(f"send_whatsapp_message is not available. Cannot send reply to {from_number_clean}: {reply_message}")

        # Webhook must always return 200 OK quickly
        return JSONResponse(content={}, status_code=status.HTTP_200_OK)


    except Exception as e:
        # This catch-all should now only be for truly unexpected errors outside the specific handlers
        logger.error(f"Unhandled critical error in whatsapp_webhook for user {from_number_clean}: {e}", exc_info=True)
        # Attempt to send an emergency error message if possible and no reply was set
        if not reply_message and from_number_clean != "unknown" and send_whatsapp_message_available:
             try:
                await send_whatsapp_message(from_number_clean, "Oh, something went wrong on my end. Please try again in a moment.")
             except Exception as send_e_2:
                 logger.error(f"Failed to send emergency error message to {from_number_clean}: {send_e_2}", exc_info=True)

        # Always return 200 OK to the webhook provider
        return JSONResponse(content={}, status_code=status.HTTP_200_OK)


# --- Frontend/API Endpoints ---
# (These endpoints remain largely the same as they are called from the frontend, not chat)

@app.post("/confirm-items")
async def confirm_items(request: OrderRequest, api_key: str = Depends(security.verify_api_key)):
    """
    Endpoint for the frontend/web menu to confirm items selected by the user.
    Requires API key authentication from the frontend.
    """
    # Ensure Supabase and security are available
    if not supabase:
        logger.error("/confirm-items called but Supabase client is not available.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database connection not available")
    if not security:
         logger.error("/confirm-items called but security module is not available.")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Security module not available")


    try:
        # 1. Validate Session
        session_res = supabase.table("sessions").select("user_id, phone_number, expires_at").eq("session_token", request.session_token).limit(1).execute()

        if not session_res.data:
            logger.warning(f"Session token {request.session_token} not found or expired for confirm-items.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired.")

        session_data = session_res.data[0]
        user_id = session_data["user_id"]
        phone_number = session_data["phone_number"]
        expires_at_str = session_data["expires_at"]

        # Explicitly check session expiry
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now(timezone.utc) > expires_at:
                 logger.warning(f"Session token {request.session_token} expired for user {user_id}.")
                 # Delete the expired session proactively
                 try:
                    supabase.table("sessions").delete().eq("session_token", request.session_token).execute()
                    logger.info(f"Deleted expired session token {request.session_token}.")
                 except Exception as e:
                     logger.error(f"Failed to delete expired session {request.session_token}: {e}", exc_info=True)

                 raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session expired.")
        except ValueError:
             logger.error(f"Could not parse expires_at for session {request.session_token}: {expires_at_str}", exc_info=True)
             # Treat as invalid/expired if date parsing fails
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid session data.")


        logger.info(f"Session {request.session_token} found and valid for user {user_id} ({phone_number})")

        # 2. Cancel any existing pending unpaid orders for this user
        logger.info(f"Cancelling any existing pending unpaid orders for user_id: {user_id} before creating new one from session {request.session_token}.")
        try:
            supabase.table("orders").update({
                "status": DefaultStatus.ORDER_CANCELLED,
                "payment_status": DefaultStatus.PAYMENT_CANCELLED,
                "cancelled_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "cancellation_reason": "superseded by new order via web menu"
            }).eq("user_id", user_id).eq("payment_status", DefaultStatus.PAYMENT_UNPAID).not_.in_("status", [DefaultStatus.ORDER_CANCELLED, DefaultStatus.ORDER_DELIVERED, DefaultStatus.ORDER_FAILED]).execute()
            logger.info(f"Cancelled older pending orders for user {user_id} during confirm-items.")
        except Exception as e:
            logger.error(f"Failed to cancel old pending orders for user {user_id} during confirm-items: {e}", exc_info=True)


        # 3. Create the new order
        items_dict = [item.model_dump() for item in request.items] # Use model_dump() for Pydantic v2+

        order_data = {
            "user_id": user_id,
            "items_json": items_dict,
            "total_amount": request.total_amount,
            "status": DefaultStatus.ORDER_PENDING_CONFIRMATION, # Initial status
            "payment_status": DefaultStatus.PAYMENT_UNPAID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        try:
            # CORRECTED: Removed .select("id") from the insert query - execute() might still return data depending on version
            result = supabase.table("orders").insert(order_data).execute()
            # Assuming execute() on insert returns data in this client version
            if not result.data:
                logger.error(f"Supabase insert returned no data for new order for user {user_id}.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create new order in database (no data returned).")

            order_id = result.data[0]["id"]
            logger.info(f"Created new order {order_id} for user {user_id} from session {request.session_token}")

        except Exception as e:
            logger.error(f"Failed to insert new order for user {user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create new order in database.")


        # 4. Delete the session token
        try:
            supabase.table("sessions").delete().eq("session_token", request.session_token).execute()
            logger.info(f"Deleted session token {request.session_token} after order creation.")
        except Exception as e:
             logger.error(f"Failed to delete session token {request.session_token}: {e}", exc_info=True) # Log error, but don't fail the request


        # 5. Notify user via WhatsApp with next steps
        delivery_msg = (
            "Thank you for confirming your items!\n\n"
            f"Your order total is *GHS {request.total_amount:.2f}* (excluding delivery).\n\n"
            "Please tell me how you'd like to receive your order by replying with the number:\n\n"
            f"*1* for Delivery (I'll ask for your location to calculate the fee).\n"
            f"*2* for Pickup (no delivery fee).\n\n"
            "Or reply 'cancel' to cancel the order."
        )
        if send_whatsapp_message_available:
            try:
                await send_whatsapp_message(phone_number, delivery_msg)
                logger.info(f"Sent confirmation message to user {user_id} ({phone_number}) for order {order_id}")
                try:
                    # Save the sent message as last_bot_message
                    truncated_message = delivery_msg[:1000]
                    supabase.table("users").update({"last_bot_message": truncated_message}).eq("id", user_id).execute();
                    logger.info(f"Saved last bot message for user {user_id} from /confirm-items.");
                except Exception as e:
                     logger.error(f"Failed to save last bot message for user {user_id} from /confirm-items: {e}", exc_info=True);

            except Exception as e:
                logger.error(f"Failed to send WhatsApp confirmation message to {phone_number} for order {order_id}: {e}", exc_info=True)
                # This is a non-critical failure for the API endpoint, just log it.
        else:
            logger.error(f"send_whatsapp_message is not available. Cannot send confirmation for order {order_id} to {phone_number}")

        return {"status": "order saved", "order_id": order_id, "next_step_message_sent": send_whatsapp_message_available}

    except HTTPException:
        # Re-raise HTTPException so FastAPI handles it
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"/confirm-items endpoint unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred.") # Generic error for user


@app.post("/payment-success")
async def payment_success_webhook(request: Request):
    """
    Paystack webhook endpoint for successful payments.
    Verifies signature, updates order status, and notifies user.
    MUST return 200 OK to Paystack quickly.
    Also handles other Paystack events like 'charge.failed' if necessary.
    """
    logger.info("Received Paystack webhook call.")

    # Ensure Supabase is available early for database operations
    if not supabase:
        logger.error("Supabase client not available for payment-success webhook processing.")
        # Return 200 OK to Paystack, but signal internal error
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Internal database unavailable"})


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
        # verification failed, return 200 OK to Paystack but log warning
        logger.warning("Paystack webhook signature verification failed.")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Invalid signature"})

    try:
        # Parse the JSON body now that signature is verified
        data = json.loads(request_body)
        event_type = data.get("event")
        paystack_reference = data.get("data", {}).get("reference")
        paystack_status = data.get("data", {}).get("status")
        paystack_gateway_response = data.get("data", {}).get("gateway_response")
        paystack_amount_kobo = data.get("data", {}).get("amount")
        order_id = data.get("data", {}).get("metadata", {}).get("order_id")
        # Also capture the reference from metadata if it was included (good practice)
        metadata_reference = data.get("data", {}).get("metadata", {}).get("reference")
        # Use the reference from data if available, fallback to metadata
        final_reference = paystack_reference or metadata_reference


        logger.info(f"Processing Paystack event '{event_type}' for reference '{final_reference}' (Paystack status: {paystack_status}). Order ID: {order_id}")

        if not order_id:
            logger.error(f"Paystack webhook event '{event_type}' received without order_id in metadata. Reference: {final_reference}")
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "order_id not found in metadata"})


        # Fetch the order from the database
        order_res = supabase.table("orders").select("id, order_number, user_id, payment_status, status, total_amount, total_with_delivery, delivery_type").eq("id", order_id).limit(1).execute()

        if not order_res.data:
            logger.warning(f"Paystack webhook: No order found in DB for order_id {order_id} (Paystack ref: {final_reference}).")
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Order not found in internal database."})

        order = order_res.data[0]
        current_payment_status: PaymentStatus = order.get("payment_status", DefaultStatus.PAYMENT_UNPAID)
        current_order_status: OrderStatus = order.get("status", DefaultStatus.ORDER_PENDING_PAYMENT) # Assume pending payment if status is null

        # Determine the expected total amount (use total_with_delivery if available, otherwise total_amount)
        expected_total = order.get('total_with_delivery') or order.get('total_amount')
        if expected_total is None:
             logger.error(f"Order {order_id} has no total_amount or total_with_delivery. Cannot process payment webhook.")
             return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Order data missing total amount."})
        expected_total_kobo = int(expected_total * 100)


        update_needed = False
        # Prepare base update data, include reference and amount received
        update_data: Dict[str, Any] = {
             "paystack_reference": final_reference,
             "amount_paid_kobo": paystack_amount_kobo, # Store the amount received
            "updated_at": datetime.now(timezone.utc).isoformat(),
             "paystack_status": paystack_status, # Store paystack's status
             "paystack_gateway_response": paystack_gateway_response # Store gateway response
        }

        # Initialize variables for potential new statuses and notification message parts
        new_payment_status = current_payment_status
        new_order_status = current_order_status
        notification_prefix = "ℹ️ Order Update"
        notification_suffix = ""
        notification_needed = False # Flag to control sending WhatsApp message

        # --- Process specific event types ---
        if event_type == "charge.success":
            notification_needed = True # Always notify on success/failure
            if paystack_status == 'success':
                 # Check if the amount received meets or exceeds the expected total
                 if paystack_amount_kobo is not None and paystack_amount_kobo >= expected_total_kobo:
                     if current_payment_status != DefaultStatus.PAYMENT_PAID:
                         new_payment_status = DefaultStatus.PAYMENT_PAID
                         # If moving to PAID, the order is now ready for processing
                         # Only move to PROCESSING if it was in a payment-related state
                         if current_order_status in [DefaultStatus.ORDER_PENDING_PAYMENT, DefaultStatus.ORDER_AWAITING_LOCATION, DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION, DefaultStatus.ORDER_PENDING_CONFIRMATION]:
                             new_order_status = DefaultStatus.ORDER_PROCESSING

                         update_needed = True
                         notification_prefix = "✅ Payment confirmed!"
                         notification_suffix = " Your order is now being processed."
                         logger.info(f"Order {order_id}: Full payment success. Status updated to PAID/PROCESSING.")
                     else:
                         logger.info(f"Order {order_id}: Already marked PAID. Received duplicate 'charge.success' webhook?")
                         notification_needed = False # Don't send redundant notification

                 # Check for partial payment if amount is less than expected but positive
                 elif paystack_amount_kobo is not None and paystack_amount_kobo > 0 and paystack_amount_kobo < expected_total_kobo:
                     if current_payment_status not in [DefaultStatus.PAYMENT_PAID, DefaultStatus.PAYMENT_PARTIALLY_PAID]:
                         new_payment_status = DefaultStatus.PAYMENT_PARTIALLY_PAID
                         # Keep order status as PENDING_PAYMENT or current unless it was failed/cancelled
                         if current_order_status in [DefaultStatus.ORDER_FAILED, DefaultStatus.ORDER_CANCELLED]:
                              # If it was failed/cancelled, perhaps move to pending payment if partial is allowed?
                              # Decide on your policy for partial payments on failed/cancelled orders.
                              # For now, let's assume partial payment on a failed/cancelled order doesn't automatically 'un-fail'/'un-cancel' it.
                              pass # Status remains failed/cancelled
                         # If it was pending payment, it remains pending payment
                         elif current_order_status != DefaultStatus.ORDER_PENDING_PAYMENT:
                              # This state transition might be unexpected, log it
                              logger.warning(f"Order {order_id} received partial payment (GHS {(paystack_amount_kobo/100):.2f}) but status was '{current_order_status}'.")
                              # Keep current status unless it implies paid (e.g., PROCESSING)
                              if current_order_status not in [DefaultStatus.ORDER_PROCESSING, DefaultStatus.ORDER_OUT_FOR_DELIVERY, DefaultStatus.ORDER_DELIVERED]:
                                  new_order_status = DefaultStatus.ORDER_PENDING_PAYMENT # Revert to pending payment if status implies processing/delivery but only partial paid

                         update_needed = True
                         notification_prefix = "⚠️ Partial Payment Confirmed!"
                         notification_suffix = f" Received GHS {(paystack_amount_kobo/100):.2f} of GHS {(expected_total_kobo/100):.2f}. Please contact support to complete the remaining payment."
                         logger.warning(f"Order {order_id}: Partial payment received.")
                     else:
                          logger.info(f"Order {order_id}: Already marked {current_payment_status}. Received duplicate 'charge.success' webhook with partial amount?")
                          notification_needed = False # Don't send redundant notification
                 else:
                      # Amount received was zero or negative, which is unexpected for charge.success
                      logger.error(f"Paystack charge.success event with invalid amount_kobo={paystack_amount_kobo} for order {order_id}. Expected >= {expected_total_kobo}.")
                      notification_prefix = "❌ Payment Issue"
                      notification_suffix = " There was an issue with the payment amount received. Please contact support."
                      # Do not change payment status unless you have a specific "invalid_amount" status

            elif paystack_status in ['failed', 'reversed', 'chargeback']:
                 # Handle failure/reversals
                 if current_payment_status not in [DefaultStatus.PAYMENT_PAID, DefaultStatus.PAYMENT_PARTIALLY_PAID, DefaultStatus.PAYMENT_FAILED, DefaultStatus.PAYMENT_CANCELLED]:
                     # Only update if the current status isn't already a final state or paid/partially paid
                     new_payment_status = DefaultStatus.PAYMENT_FAILED
                     new_order_status = DefaultStatus.ORDER_FAILED # Also mark the order as failed
                     update_needed = True
                     notification_prefix = "❌ Payment Failed"
                     notification_suffix = " There was an issue processing your payment. Please try again or contact support if the amount was deducted."
                     logger.info(f"Order {order_id}: Payment explicitly failed according to Paystack.")
                 else:
                      logger.info(f"Order {order_id}: Payment already marked {current_payment_status}, ignoring '{paystack_status}' status from Paystack.")
                      notification_needed = False # Don't send notification if already processed

            elif paystack_status in ['abandoned']:
                 # Handle abandoned payments if needed
                 logger.info(f"Paystack charge.success event with status 'abandoned' for order {order_id}. No status change needed.")
                 notification_needed = False # Usually no notification needed for abandoned state via webhook

            else:
                 logger.warning(f"Paystack charge.success event with unexpected status '{paystack_status}' for order {order_id}. Gateway Response: {paystack_gateway_response}")
                 # Decide if you want to notify on unexpected statuses. Maybe log and ignore.
                 notification_needed = False

        # Add handling for other relevant events Paystack sends
        # elif event_type == "transfer.success": ... # If you pay vendors via Paystack
        # elif event_type == "charge.dispute.create": ...
        # elif event_type == "customeridentification.success": ... # If you use this feature


        # --- Update the database if changes are needed ---
        if update_needed:
            update_data["payment_status"] = new_payment_status
            # Only update order status if it's changing
            if new_order_status != current_order_status:
                 update_data["status"] = new_order_status

            try:
                # Use eq("id", order_id) to ensure correct order is updated
                # Consider adding eq("payment_status", current_payment_status) and eq("status", current_order_status)
                # as optimistic concurrency control if race conditions are a concern.
                supabase.table("orders").update(update_data).eq("id", order_id).execute()
                logger.info(f"Order {order_id} successfully updated to payment_status '{new_payment_status}' and status '{new_order_status}'.")
            except Exception as e:
                logger.error(f"Failed to update order {order_id} in DB after Paystack webhook processing: {e}", exc_info=True)
                # Log error but still return 200 OK to Paystack
                return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Failed to update order in database"})
        else:
             logger.info(f"No payment/order status update needed for order {order_id}. Event {event_type}, current payment {current_payment_status}, current order {current_order_status}.")


        # --- Notify User via WhatsApp ---
        # Only send notification if it was deemed necessary based on the event and status changes
        if notification_needed:
             user_query = supabase.table("users").select("phone_number").eq("id", order["user_id"]).limit(1).execute()

             if user_query.data and send_whatsapp_message_available:
                 phone_number = user_query.data[0]["phone_number"]
                 total_paid_display = (paystack_amount_kobo / 100.0) if paystack_amount_kobo is not None else "N/A"
                 # Use the potentially updated statuses for the notification message
                 status_display = new_order_status.replace('-', ' ').title()
                 payment_status_display = new_payment_status.replace('_', ' ').title()
                 order_number_display = order.get('order_number', order_id)


                 notification_message = (
                     f"{notification_prefix}\n\n"
                     f"Your Order ID is: {order_number_display}.\n"
                     f"Amount Paid: *GHS {total_paid_display:.2f}*.\n"
                     f"Current Order Status: *{status_display}*.\n"
                     f"Payment Status: *{payment_status_display}*.\n"
                     f"Reference: {final_reference}\n\n"
                     f"{notification_suffix}"
                 )

                 try:
                     await send_whatsapp_message(phone_number, notification_message)
                     logger.info(f"Notified user {order['user_id']} ({phone_number}) about order {order_id} payment status '{new_payment_status}'.")
                     try:
                         # Save the notification message
                         truncated_message = notification_message[:1000]
                         supabase.table("users").update({"last_bot_message": truncated_message}).eq("id", order['user_id']).execute();
                         logger.info(f"Saved last bot message for user {order['user_id']} from /payment-success.");
                     except Exception as e:
                         logger.error(f"Failed to save last bot message for user {order['user_id']} from /payment-success: {e}", exc_info=True);

                 except Exception as e:
                      logger.error(f"Failed to send WhatsApp notification for order {order_id} to {phone_number}: {e}", exc_info=True)

             elif not user_query.data:
                 logger.error(f"Could not find user {order['user_id']} to notify for order {order_id}")
             elif not send_whatsapp_message_available:
                 logger.error(f"send_whatsapp_message is not available. Cannot notify user {order['user_id']} about order {order_id}")


        # Final step: Return 200 OK to Paystack regardless of processing success
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "success", "message": f"Event {event_type} processed"})

    except json.JSONDecodeError:
        logger.error("Paystack webhook received non-JSON body after signature verification.")
        # Return 200 OK despite error, as per webhook best practices
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": "Invalid JSON body"})
    except Exception as e:
        # Catch any other unexpected errors during payload processing
        logger.error(f"Unexpected error processing Paystack webhook payload for reference {paystack_reference}: {str(e)}", exc_info=True)
        # Return 200 OK despite error, as per webhook best practices
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "error", "message": f"Internal server error during processing."}) # Generic error message


# Export the app for Vercel
app.debug = False # Set debug to False for production

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up.")
    if not supabase:
        logger.error("Supabase client not available at startup.")
    if not send_whatsapp_message_available:
         logger.error("send_whatsapp_message utility is not available. WhatsApp communication will be disabled.")
    if not settings.PAYSTACK_SECRET_KEY:
         logger.warning("PAYSTACK_SECRET_KEY is not set. Payment link generation will use mock links, webhook verification will FAIL.")
    if not settings.GEMINI_API_KEY:
         logger.warning("GEMINI_API_KEY is not set. AI intent analysis will use basic fallback logic.")
    # Add checks for other critical settings/connections

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down.")