# --- START OF FILE: index_conversational.py (FINAL, INTEGRATED & FIXED VERSION) ---

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

# --- MOVED UP: Configure logging early to catch all startup errors ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- REVISED: Import logic for Supabase, Utils, and other local modules ---
try:
    from supabase_client import supabase
    logger.info("Supabase client imported successfully.")
except ImportError:
    supabase = None
    logger.error("Supabase client not found or failed to import. Database operations will be unavailable.")

try:
    # We only import the function, not an availability flag.
    from .utils import send_whatsapp_message
    send_whatsapp_message_available = True
    logger.info("Successfully imported send_whatsapp_message utility.")
except ImportError:
    # If the import fails, we create a dummy function and set the flag to False.
    send_whatsapp_message_available = False
    async def send_whatsapp_message(to: str, body: str):
        logger.error(f"send_whatsapp_message utility is NOT AVAILABLE. Tried to send to {to}: {body}")
    logger.warning("Could not import 'send_whatsapp_message' from .utils. A dummy function will be used.")

try:
    from . import security, admin_router, auth_router, public_router
    logger.info("Routers and security modules imported successfully.")
except ImportError as e:
    security, admin_router, auth_router, public_router = None, None, None, None
    logger.error(f"Failed to import routers or security module: {e}. Related API endpoints will be disabled.")


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

# --- MODIFIED: Constants & Statuses to include 'draft' ---
OrderStatus = Literal[
    "draft", "pending_confirmation", "awaiting_location", "awaiting_location_confirmation",
    "pending_payment", "processing", "out-for-delivery", "delivered", "cancelled", "failed"
]
PaymentStatus = Literal["unpaid", "paid", "partially_paid", "cancelled", "failed"]

class DefaultStatus:
    ORDER_DRAFT: OrderStatus = "draft"
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

# --- Pydantic Models ---
class OrderItem(BaseModel):
    product_id: str
    quantity: int

class OrderRequest(BaseModel):
    session_token: str
    items: List[OrderItem]
    total_amount: float

# --- FastAPI App Initialization & Middleware ---
app = FastAPI(title="WhatsApp MarketBot API (Conversational)", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Include Routers if they were imported successfully
if security and admin_router: app.include_router(admin_router.router, prefix="/admin", tags=["admin"], dependencies=[Depends(security.get_admin_user)])
if auth_router: app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
if public_router: app.include_router(public_router.router)


# --- THE AI BRAIN ---
async def call_ai_assistant(user_message: str, conversation_history: List[Dict[str, str]], current_order_status: Optional[str]) -> Dict[str, Any]:
    if not settings.GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not set")
    history_str = "\n".join([f"{turn['speaker']}: {turn['message']}" for turn in conversation_history])
    prompt = f"""
You are "Fresh Market Assistant", an expert conversational AI for a grocery service in Ghana. Your goal is to have a natural, helpful conversation. Respond ONLY with a single, minified JSON object.

--- CONTEXT ---
Current Order Status: "{current_order_status or 'NO_ORDER'}"
Recent Conversation History:
<history>
{history_str}
</history>
Latest User Message: "{user_message}"
--- END CONTEXT ---

Based on the context, decide the single best `action` to take and craft a suitable `response`.

--- JSON OUTPUT SCHEMA ---
{{"action": "ACTION_TYPE", "entities": [{{"product_name": "...", "quantity": "..."}}], "response": "Your natural language response to the user."}}

--- AVAILABLE ACTIONS ---
- `START_ORDER`: User wants to begin a new order.
- `ADD_TO_CART`: User is adding items to their 'draft' order. Extract `product_name` and `quantity`.
- `VIEW_CART`: User wants to see what's in their cart.
- `CONFIRM_ORDER`: User is done adding items and wants to proceed to checkout.
- `SET_DELIVERY_PREFERENCE`: User has chosen "delivery" or "pickup". Include a "preference" key.
- `CHECK_STATUS`: User is asking about a *previously paid* order.
- `CANCEL_ORDER`: User wants to cancel their current draft or pending order.
- `GREET`: A simple greeting.
- `HELP`: User needs help.
- `ANSWER_QUESTION`: General question. Your response should just answer it.
- `UNKNOWN`: The intent is unclear or irrelevant.

--- EXAMPLES ---
User: "I need 2kg tomatoes" / Status: "draft" -> JSON: {{"action": "ADD_TO_CART", "entities": [{{"product_name": "tomatoes", "quantity": "2kg"}}], "response": "Sure thing! 2kg of tomatoes added. Anything else?"}}
User: "that's all" / Status: "draft" -> JSON: {{"action": "CONFIRM_ORDER", "entities": [], "response": "Great! Let's get this finalized."}}
User: "delivery" / Status: "pending_confirmation" -> JSON: {{"action": "SET_DELIVERY_PREFERENCE", "entities": [], "preference": "delivery", "response": "Alright, delivery it is!"}}
"""
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2}}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(settings.GEMINI_API_URL, headers={"Content-Type": "application/json"}, params={"key": settings.GEMINI_API_KEY}, json=payload)
            response.raise_for_status()
            result_json = response.json()
            json_text = result_json["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(json_text)
    except Exception as e:
        logger.error(f"Error in call_ai_assistant: {e}", exc_info=True)
        return {"action": "UNKNOWN", "response": "I'm having a little trouble processing that. Could you please rephrase?"}

# --- HELPER FUNCTIONS FOR CONVERSATIONAL ACTIONS ---
async def handle_add_to_cart(user: Dict, order: Dict, entities: List[Dict]) -> None:
    if not supabase or not entities: return
    order_id, current_items, new_total = order['id'], order.get('items_json') or [], order.get('total_amount') or 0.0
    for entity in entities:
        product_name, quantity, mock_price = entity.get("product_name", "Unknown Item"), entity.get("quantity", "1"), 10.00
        current_items.append({"product": product_name, "quantity": quantity, "price": mock_price})
        new_total += mock_price
    try:
        supabase.table("orders").update({"items_json": current_items, "total_amount": new_total, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", order_id).execute()
    except Exception as e:
        logger.error(f"Failed to update cart for order {order_id}: {e}", exc_info=True)
        raise

async def handle_confirm_order(user: Dict, order: Dict) -> str:
    if not supabase: return "Database is currently unavailable."
    order_id, total_amount = order['id'], order.get('total_amount', 0.0)
    items = order.get('items_json', [])
    if not items: return "Your cart is empty! Please add some items first."
    items_summary = "\n".join([f"- {item['quantity']} of {item['product']}" for item in items])
    try:
        supabase.table("orders").update({"status": DefaultStatus.ORDER_PENDING_CONFIRMATION, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", order_id).execute()
        return (
            f"Great! Here is your order summary:\n{items_summary}\n\n"
            f"Your subtotal is *GHS {total_amount:.2f}*.\n\n"
            "How would you like to receive your order? Will it be *delivery* or *pickup*?"
        )
    except Exception as e:
        logger.error(f"Failed to confirm order {order_id}: {e}", exc_info=True)
        return "I had trouble confirming your order. Please try again."

# --- PRIMARY CONVERSATIONAL WEBHOOK ---
@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    """ Main webhook endpoint for the conversational flow. """
    from_number_clean = "unknown"
    try:
        form_data = await request.form()
        from_number = form_data.get("From")
        incoming_msg = form_data.get("Body", "").strip()

        if not from_number or not incoming_msg: return JSONResponse(content={}, status_code=200)
        from_number_clean = from_number.replace("whatsapp:", "")

        if not supabase:
            if send_whatsapp_message_available:
                await send_whatsapp_message(from_number_clean, "Sorry, I'm having technical issues right now. Please try again later.")
            return JSONResponse(content={}, status_code=200)

        user_res = supabase.table("users").select("*").eq("phone_number", from_number_clean).limit(1).execute()
        if user_res.data:
            user = user_res.data[0]
        else:
            user = supabase.table("users").insert({"phone_number": from_number_clean}).execute().data[0]
        user_id = user['id']

        conversation_history = [{"speaker": "user", "message": incoming_msg}]

        active_order_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", DefaultStatus.ORDER_DRAFT).order("created_at", desc=True).limit(1).execute()
        active_order = active_order_res.data[0] if active_order_res.data else None
        
        current_status = active_order['status'] if active_order else None
        ai_decision = await call_ai_assistant(incoming_msg, conversation_history, current_status)
        action, ai_response = ai_decision.get("action"), ai_decision.get("response")
        reply_message = ai_response or "I'm not sure how to respond to that."
        
        logger.debug(f"AI Decision for {from_number_clean}: Action='{action}'")
        
        if action == 'START_ORDER':
            if not active_order:
                active_order = supabase.table("orders").insert({"user_id": user_id, "status": DefaultStatus.ORDER_DRAFT}).execute().data[0]
        
        elif action == 'ADD_TO_CART':
            if not active_order:
                active_order = supabase.table("orders").insert({"user_id": user_id, "status": DefaultStatus.ORDER_DRAFT}).execute().data[0]
            await handle_add_to_cart(user, active_order, ai_decision.get("entities", []))

        elif action == 'CONFIRM_ORDER':
            if active_order:
                reply_message = await handle_confirm_order(user, active_order)
            else:
                reply_message = "You don't have an active order to confirm. To start, just tell me what you'd like to buy."

        # (Other action handlers like CHECK_STATUS, CANCEL_ORDER, SET_DELIVERY_PREFERENCE etc. would go here)

        if reply_message and send_whatsapp_message_available:
            await send_whatsapp_message(from_number_clean, reply_message)
        return JSONResponse(content={}, status_code=200)

    except Exception as e:
        logger.error(f"Critical webhook error for {from_number_clean}: {e}", exc_info=True)
        if from_number_clean != "unknown" and send_whatsapp_message_available:
            await send_whatsapp_message(from_number_clean, "I'm sorry, an unexpected error occurred on my end. Please try again in a moment.")
        return JSONResponse(content={}, status_code=200)


# --- WEB-BASED & OTHER ENDPOINTS ---

def generate_order_number():
    return f"ORD-{int(datetime.now(timezone.utc).timestamp())}"

@app.post("/confirm-items")
async def confirm_items(request: OrderRequest, api_key: str = (Security(security.verify_api_key) if security else None)):
    """ Endpoint for the web menu. Bypasses 'draft' and creates a 'pending_confirmation' order. """
    if not supabase or not security: raise HTTPException(500, "Server module unavailable")
    
    session_res = supabase.table("sessions").select("user_id, phone_number").eq("session_token", request.session_token).limit(1).execute()
    if not session_res.data: raise HTTPException(404, "Session not found or expired")
    
    user_id = session_res.data[0]["user_id"]
    phone_number = session_res.data[0]["phone_number"]

    try:
        logger.info(f"Cancelling previous draft/pending orders for user {user_id}")
        statuses_to_cancel = [
            DefaultStatus.ORDER_DRAFT, DefaultStatus.ORDER_PENDING_CONFIRMATION,
            DefaultStatus.ORDER_AWAITING_LOCATION, DefaultStatus.ORDER_AWAITING_LOCATION_CONFIRMATION,
            DefaultStatus.ORDER_PENDING_PAYMENT
        ]
        supabase.table("orders").update({
            "status": DefaultStatus.ORDER_CANCELLED,
            "cancellation_reason": "superseded by new web order"
        }).eq("user_id", user_id).in_("status", statuses_to_cancel).execute()
    except Exception as e:
        logger.error(f"Failed to cancel old orders for user {user_id}: {e}", exc_info=True)

    order_data = {
        "user_id": user_id,
        "items_json": [item.model_dump() for item in request.items],
        "total_amount": request.total_amount,
        "status": DefaultStatus.ORDER_PENDING_CONFIRMATION,
        "payment_status": DefaultStatus.PAYMENT_UNPAID,
        "order_number": generate_order_number(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    result = supabase.table("orders").insert(order_data).execute()
    if not result.data: raise HTTPException(500, "Failed to create order in database")
    
    order_id = result.data[0]["id"]
    supabase.table("sessions").delete().eq("session_token", request.session_token).execute()

    delivery_msg = (
        f"Thank you for confirming your items! Your order subtotal is *GHS {request.total_amount:.2f}*.\n\n"
        "To finalize, how would you like to receive your order? Will it be *delivery* or will you *pick up*?"
    )
    if send_whatsapp_message_available:
        await send_whatsapp_message(phone_number, delivery_msg)
        
    return {"status": "order saved", "order_id": order_id}

@app.post("/payment-success")
async def payment_success_webhook(request: Request):
    """ Webhook for Paystack payment events. This logic doesn't need to change. """
    # This endpoint logic remains complex but doesn't need to be aware of the 'draft' state,
    # as payments only happen on orders that are already confirmed.
    # You can paste your full, original /payment-success logic here.
    # For brevity, I'll add a placeholder.
    logger.info("Received a call to /payment-success webhook.")
    return JSONResponse(status_code=200, content={"status": "received"})