# --- START OF FILE index_conversational.py (FINAL, INTEGRATED VERSION) ---

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

# --- Project Specific Imports (Assumed to be in the same structure) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (Error handling for imports remains the same as your original file)
try:
    from supabase_client import supabase
    from . import security, admin_router, auth_router, public_router
    from .utils import send_whatsapp_message, send_whatsapp_message_available
except ImportError:
    # Fallback definitions for local testing if needed
    supabase = None
    security = None
    admin_router, auth_router, public_router = None, None, None
    send_whatsapp_message_available = False
    async def send_whatsapp_message(to: str, body: str):
        logger.error(f"send_whatsapp_message is not available. Tried to send to {to}")
    logger.error("Failed to import one or more local modules. Running in degraded mode.")


# --- Basic Configuration (Logging, Settings) ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    "draft",
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

PaymentStatus = Literal["unpaid", "paid", "partially_paid", "cancelled", "failed"]

class DefaultStatus:
    ORDER_DRAFT: OrderStatus = "draft"
    ORDER_PENDING_CONFIRMATION: OrderStatus = "pending_confirmation"
    # ... other statuses
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
    # ... other payment statuses

# --- Pydantic Models (Re-integrated for completeness) ---
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
# ... other middleware and router includes

# --- THE AI BRAIN (Unchanged from previous version) ---
async def call_ai_assistant(user_message: str, conversation_history: List[Dict[str, str]], current_order_status: Optional[str]) -> Dict[str, Any]:
    # (The advanced prompt and logic from the previous response goes here, no changes needed)
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
- `PROMPT_FOR_DELIVERY_TYPE`: User has confirmed their cart and needs to choose delivery/pickup.
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
        return {"action": "UNKNOWN", "response": "I'm having a little trouble. Could you rephrase?"}

# --- HELPER FUNCTIONS FOR CONVERSATIONAL ACTIONS (Unchanged) ---
async def handle_add_to_cart(user: Dict, order: Dict, entities: List[Dict]) -> str:
    # (Logic remains the same)
    if not supabase or not entities: return "What items would you like to add?"
    order_id, current_items, new_total = order['id'], order.get('items_json') or [], order.get('total_amount') or 0.0
    for entity in entities:
        product_name, quantity, mock_price = entity.get("product_name", "Unknown"), entity.get("quantity", "1"), 10.00
        current_items.append({"product": product_name, "quantity": quantity, "price": mock_price})
        new_total += mock_price
    try:
        supabase.table("orders").update({"items_json": current_items, "total_amount": new_total}).eq("id", order_id).execute()
        return f"Added. New total: GHS {new_total:.2f}."
    except Exception as e:
        logger.error(f"Failed to update cart for order {order_id}: {e}")
        return "I had trouble adding that."

async def handle_confirm_order(user: Dict, order: Dict) -> str:
    # (Logic remains the same)
    if not supabase: return "DB unavailable."
    order_id, total_amount = order['id'], order.get('total_amount', 0.0)
    items_summary = "\n".join([f"- {item['quantity']} of {item['product']}" for item in order.get('items_json', [])])
    try:
        supabase.table("orders").update({"status": "pending_confirmation"}).eq("id", order_id).execute()
        return f"Great! Summary:\n{items_summary}\n\nSubtotal: *GHS {total_amount:.2f}*.\n\n*Delivery* or *pickup*?"
    except Exception as e:
        logger.error(f"Failed to confirm order {order_id}: {e}")
        return "I had trouble confirming that."

# --- MODIFIED: PRIMARY CONVERSATIONAL WEBHOOK ---
@app.post("/whatsapp-webhook") # MODIFIED: Renamed to be the primary webhook
async def whatsapp_webhook(request: Request):
    """ Main webhook endpoint for the conversational flow. """
    reply_message = "I'm not sure how to respond. Can you rephrase?"
    from_number_clean = "unknown"
    try:
        form_data = await request.form()
        from_number = form_data.get("From")
        incoming_msg = form_data.get("Body", "").strip()
        if not from_number or not incoming_msg: return JSONResponse(content={}, status_code=200)
        from_number_clean = from_number.replace("whatsapp:", "")

        if not supabase:
            await send_whatsapp_message(from_number_clean, "Sorry, I'm having technical issues. Please try again later.")
            return JSONResponse(content={}, status_code=200)

        user_res = supabase.table("users").select("*").eq("phone_number", from_number_clean).limit(1).execute()
        user = user_res.data[0] if user_res.data else supabase.table("users").insert({"phone_number": from_number_clean}).execute().data[0]
        user_id = user['id']

        conversation_history = [{"speaker": "user", "message": incoming_msg}]

        active_order_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", "draft").order("created_at", desc=True).limit(1).execute()
        active_order = active_order_res.data[0] if active_order_res.data else None
        
        ai_decision = await call_ai_assistant(incoming_msg, conversation_history, active_order['status'] if active_order else None)
        action, ai_response = ai_decision.get("action"), ai_decision.get("response")
        
        logger.debug(f"AI Decision: Action='{action}'")
        
        # --- Action Dispatcher ---
        if action == 'START_ORDER':
            if not active_order:
                active_order = supabase.table("orders").insert({"user_id": user_id, "status": "draft"}).execute().data[0]
            reply_message = ai_response or "New order started! What would you like?"
        
        elif action == 'ADD_TO_CART':
            if not active_order:
                active_order = supabase.table("orders").insert({"user_id": user_id, "status": "draft"}).execute().data[0]
            await handle_add_to_cart(user, active_order, ai_decision.get("entities", []))
            reply_message = ai_response

        elif action == 'CONFIRM_ORDER':
            reply_message = await handle_confirm_order(user, active_order) if active_order else "You have no order to confirm."

        # (Other action handlers like CHECK_STATUS, CANCEL_ORDER, etc. would go here)
        else:
            reply_message = ai_response

        if reply_message: await send_whatsapp_message(from_number_clean, reply_message)
        return JSONResponse(content={}, status_code=200)

    except Exception as e:
        logger.error(f"Critical webhook error for {from_number_clean}: {e}", exc_info=True)
        if from_number_clean != "unknown": await send_whatsapp_message(from_number_clean, "An error occurred. Please try again.")
        return JSONResponse(content={}, status_code=200)

# --- RE-INTEGRATED & MODIFIED: Web-based Endpoints ---

def generate_order_number():
    return f"ORD-{int(datetime.now(timezone.utc).timestamp())}"

@app.post("/confirm-items")
async def confirm_items(request: OrderRequest, api_key: str = Security(security.verify_api_key) if security else None):
    """ Endpoint for the web menu. Bypasses 'draft' and creates a 'pending_confirmation' order. """
    if not supabase: raise HTTPException(500, "Database unavailable")
    session_res = supabase.table("sessions").select("user_id, phone_number").eq("session_token", request.session_token).limit(1).execute()
    if not session_res.data: raise HTTPException(404, "Session not found")
    
    user_id = session_res.data[0]["user_id"]
    phone_number = session_res.data[0]["phone_number"]

    # --- MODIFIED: Cancellation logic now includes 'draft' status ---
    try:
        logger.info(f"Cancelling previous draft/pending orders for user {user_id}")
        supabase.table("orders").update({
            "status": "cancelled",
            "cancellation_reason": "superseded by new web order"
        }).eq("user_id", user_id).in_("status", ["draft", "pending_confirmation", "awaiting_location", "awaiting_location_confirmation", "pending_payment"]).execute()
    except Exception as e:
        logger.error(f"Failed to cancel old orders for user {user_id}: {e}")

    order_data = {
        "user_id": user_id,
        "items_json": [item.model_dump() for item in request.items],
        "total_amount": request.total_amount,
        "status": DefaultStatus.ORDER_PENDING_CONFIRMATION, # Correctly bypasses 'draft'
        "payment_status": DefaultStatus.PAYMENT_UNPAID,
        "order_number": generate_order_number(),
    }
    result = supabase.table("orders").insert(order_data).execute()
    order_id = result.data[0]["id"]
    supabase.table("sessions").delete().eq("session_token", request.session_token).execute()

    delivery_msg = (
        "Thank you for confirming your items!\n\n"
        f"Your order total is *GHS {request.total_amount:.2f}* (excluding delivery).\n\n"
        "How would you like to receive your order? Will it be *delivery* or *pickup*?"
    )
    await send_whatsapp_message(phone_number, delivery_msg)
    return {"status": "order saved", "order_id": order_id}

# (The /payment-success webhook can remain as it was in your original file, as it operates on paid/pending orders, not drafts)
@app.post("/payment-success")
async def payment_success_webhook(request: Request):
    # This endpoint logic doesn't need to change significantly because it deals with orders
    # that are already past the 'draft' stage.
    # ... (paste your full, original /payment-success logic here) ...
    return JSONResponse(status_code=200, content={"status": "success"})