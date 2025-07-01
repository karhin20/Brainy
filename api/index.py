
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

# --- Project Imports & Logging ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from supabase_client import supabase
    logger.info("Supabase client imported successfully.")
except ImportError:
    supabase = None
    logger.error("Supabase client not found or failed to import. Database operations will be unavailable.")

try:
    from .utils import send_whatsapp_message
    send_whatsapp_message_available = True
    logger.info("Successfully imported send_whatsapp_message utility.")
except ImportError:
    send_whatsapp_message_available = False
    async def send_whatsapp_message(to: str, body: str):
        logger.error(f"send_whatsapp_message utility is NOT AVAILABLE. Tried to send to {to}: {body}")
    logger.warning("Could not import 'send_whatsapp_message' from .utils. A dummy function will be used.")

try:
    from . import security, admin_router, auth_router, public_router
    logger.info("Routers and security modules imported successfully.")
except ImportError as e:
    security, admin_router, auth_router, public_router = None, None, None, None
    logger.error(f"Failed to import local modules: {e}. Related API endpoints will be disabled.")

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
    API_KEY: str

settings = Settings()

# --- Constants & Statuses ---
OrderStatus = Literal[
    "pending_confirmation", "awaiting_location", "awaiting_location_confirmation",
    "pending_payment", "processing", "out-for-delivery", "delivered", "cancelled", "failed"
]
PaymentStatus = Literal["unpaid", "paid", "partially_paid", "cancelled", "failed"]

# --- Pydantic Models ---
class OrderItem(BaseModel):
    product_id: str
    quantity: int

class OrderRequest(BaseModel):
    session_token: str
    items: List[OrderItem]
    total_amount: float

# --- FastAPI App & Middleware ---
app = FastAPI(title="WhatsApp MarketBot API (Hybrid Model)", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if security and admin_router: app.include_router(admin_router.router, prefix="/admin", tags=["admin"], dependencies=[Depends(security.get_admin_user)])
if auth_router: app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
if public_router: app.include_router(public_router.router)

# --- REVISED AI BRAIN (SIMPLIFIED FOR HYBRID MODEL) ---
async def get_intent(user_message: str) -> Dict[str, Any]:
    if not settings.GEMINI_API_KEY:
        lower_msg = user_message.lower()
        if any(word in lower_msg for word in ["buy", "order", "menu", "shop", "items", "purchase"]): return {"intent": "start_order"}
        if any(word in lower_msg for word in ["status", "track", "where is"]): return {"intent": "check_status"}
        if any(word in lower_msg for word in ["cancel"]): return {"intent": "cancel_order"}
        if any(word in lower_msg for word in ["yes", "ok", "correct", "confirm"]): return {"intent": "confirm_action"}
        return {"intent": "greet"}

    prompt = f"""
    Analyze the user's message for a grocery bot. Classify the core intent. Respond ONLY with a single, minified JSON object.
    User Message: "{user_message}"
    Your JSON output MUST contain one key, "intent", with one of these values:
    - `start_order`: User wants to start a new order, browse, see the menu, or buy anything.
    - `check_status`: User is asking about an existing order's status.
    - `cancel_order`: User explicitly wants to cancel.
    - `greet`: A simple greeting, question, or any other conversational text.
    - `confirm_action`: User says "yes", "ok", "correct", "confirm".
    - `deny_action`: User says "no", "stop", "don't".
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "application/json"}}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(settings.GEMINI_API_URL, headers={"Content-Type": "application/json"}, params={"key": settings.GEMINI_API_KEY}, json=payload)
            res.raise_for_status()
            return json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        logger.error(f"Error in get_intent: {e}", exc_info=True)
        return {"intent": "greet"}

# --- HELPER FUNCTIONS ---
def generate_order_number(): return f"ORD-{int(datetime.now(timezone.utc).timestamp())}"

def calculate_delivery_fee(lat: float, lon: float) -> float:
    # A simple mock fee calculation. Replace with your actual logic.
    return 15.00

async def generate_paystack_payment_link(order_id: str, amount: float, user_phone: str) -> str:
    if not settings.PAYSTACK_SECRET_KEY:
        logger.warning("PAYSTACK_SECRET_KEY not set. Returning mock link.")
        return f"{settings.FRONTEND_URL}/payment-success?mock=true&order_id={order_id}"
    # ... (Full Paystack API call logic would go here)
    return "https://paystack.com/pay/mock-payment-link" # Placeholder for functional code

# --- RE-ARCHITECTED PRIMARY WEBHOOK (HYBRID MODEL) ---
@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    from_number_clean = "unknown"
    try:
        form_data = await request.form()
        from_number = form_data.get("From")
        incoming_msg_body = form_data.get("Body", "").strip().lower()
        is_location_message = "Latitude" in form_data

        if not from_number or (not incoming_msg_body and not is_location_message):
            return JSONResponse(content={}, status_code=200)

        from_number_clean = from_number.replace("whatsapp:", "")
        if not supabase or not send_whatsapp_message_available:
            return JSONResponse(content={}, status_code=200)

        user_res = supabase.table("users").select("*").eq("phone_number", from_number_clean).limit(1).execute()
        user = user_res.data[0] if user_res.data else supabase.table("users").insert({"phone_number": from_number_clean}).execute().data[0]
        user_id = user['id']

        # --- STATE-BASED LOGIC (HIGHEST PRIORITY) ---
        
        # 1. Check for an order awaiting payment
        unpaid_order_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", "pending_payment").order("created_at", desc=True).limit(1).execute()
        if unpaid_order_res.data:
            order = unpaid_order_res.data[0]
            if incoming_msg_body in ["pay", "yes", "payment"]:
                total = order.get('total_with_delivery') or order.get('total_amount', 0)
                payment_link = await generate_paystack_payment_link(order['id'], total, from_number_clean)
                reply = f"Of course. Please use the link below to complete your payment for order {order.get('order_number')}:\n\n{payment_link}"
            elif incoming_msg_body in ["cancel", "no"]:
                supabase.table("orders").update({"status": "cancelled"}).eq("id", order['id']).execute()
                reply = f"Your order ({order.get('order_number')}) has been cancelled. Feel free to start a new one anytime!"
            else:
                reply = f"You have a pending order ({order.get('order_number')}) awaiting payment. Would you like to *pay* now or *cancel* the order?"
            await send_whatsapp_message(from_number_clean, reply)
            return JSONResponse(content={}, status_code=200)

        # 2. Check for an order awaiting delivery/pickup choice
        pending_confirm_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", "pending_confirmation").order("created_at", desc=True).limit(1).execute()
        if pending_confirm_res.data:
            order = pending_confirm_res.data[0]
            if "delivery" in incoming_msg_body:
                supabase.table("orders").update({"status": "awaiting_location"}).eq("id", order['id']).execute()
                reply = "Great! Please share your delivery location using the WhatsApp location feature.\n\nTap the *clip icon 📎*, then choose *'Location' 📍*."
            elif "pickup" in incoming_msg_body:
                supabase.table("orders").update({"status": "pending_payment"}).eq("id", order['id']).execute()
                payment_link = await generate_paystack_payment_link(order['id'], order['total_amount'], from_number_clean)
                reply = f"Alright, your order is set for pickup. Your total is *GHS {order['total_amount']:.2f}*. Please complete your payment here:\n\n{payment_link}"
            else:
                reply = "Please choose how you'd like to receive your order: *delivery* or *pickup*?"
            await send_whatsapp_message(from_number_clean, reply)
            return JSONResponse(content={}, status_code=200)

        # 3. Check for an order awaiting a location message
        awaiting_loc_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", "awaiting_location").order("created_at", desc=True).limit(1).execute()
        if awaiting_loc_res.data and is_location_message:
            order = awaiting_loc_res.data[0]
            lat, lon = float(form_data["Latitude"]), float(form_data["Longitude"])
            delivery_fee = calculate_delivery_fee(lat, lon)
            total_with_delivery = order['total_amount'] + delivery_fee
            
            update_data = {
                "status": "pending_payment",
                "delivery_fee": delivery_fee,
                "total_with_delivery": total_with_delivery,
                "delivery_location_lat": lat, "delivery_location_lon": lon
            }
            supabase.table("orders").update(update_data).eq("id", order['id']).execute()
            
            payment_link = await generate_paystack_payment_link(order['id'], total_with_delivery, from_number_clean)
            reply = f"Thank you! Your delivery fee is GHS {delivery_fee:.2f}. Your new total is *GHS {total_with_delivery:.2f}*.\n\nPlease use this link to pay:\n{payment_link}"
            await send_whatsapp_message(from_number_clean, reply)
            return JSONResponse(content={}, status_code=200)

        # --- INTENT-BASED LOGIC (IF NO PRIORITY STATES) ---
        ai_result = await get_intent(incoming_msg_body)
        intent = ai_result.get("intent")
        reply_message = "Hello! To place an order, say 'menu' or 'buy'. To check on an existing order, say 'status'."

        if intent == 'start_order':
            session_token = str(uuid.uuid4())
            expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            supabase.table("sessions").insert({"session_token": session_token, "user_id": user_id, "phone_number": from_number_clean, "expires_at": expires_at.isoformat()}).execute()
            menu_url = f"{settings.FRONTEND_URL}?session={session_token}"
            reply_message = f"Great! Please use the link below to browse our menu and add items to your cart. Return to this chat after you confirm your items on the website!\n\n{menu_url}"

        elif intent == 'check_status':
            paid_order_res = supabase.table("orders").select("status, order_number").eq("user_id", user_id).eq("payment_status", "paid").order("created_at", desc=True).limit(1).execute()
            if paid_order_res.data:
                reply_message = f"Your most recent order ({paid_order_res.data[0]['order_number']}) is currently '{paid_order_res.data[0]['status']}'."
            else:
                reply_message = "It looks like you don't have any active orders with us. To start one, just say 'menu'."
        
        await send_whatsapp_message(from_number_clean, reply_message)
        return JSONResponse(content={}, status_code=200)

    except Exception as e:
        logger.error(f"Critical webhook error for {from_number_clean}: {e}", exc_info=True)
        if from_number_clean != "unknown" and send_whatsapp_message_available:
            await send_whatsapp_message(from_number_clean, "I'm sorry, an unexpected error occurred. Please try again.")
        return JSONResponse(content={}, status_code=200)

# --- WEB-BASED ENDPOINTS ---
@app.post("/confirm-items")
async def confirm_items(request: OrderRequest):
    if not supabase or not send_whatsapp_message_available: raise HTTPException(500, "Server module unavailable")
    
    session_res = supabase.table("sessions").select("*").eq("session_token", request.session_token).limit(1).execute()
    if not session_res.data: raise HTTPException(404, "Session expired or invalid")
    user_id, phone_number = session_res.data[0]['user_id'], session_res.data[0]['phone_number']

    order_data = {
        "user_id": user_id, "items_json": [item.model_dump() for item in request.items],
        "total_amount": request.total_amount, "status": "pending_confirmation",
        "payment_status": "unpaid", "order_number": generate_order_number(),
        "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()
    }
    order_res = supabase.table("orders").insert(order_data).execute()
    if not order_res.data: raise HTTPException(500, "Could not create order")

    supabase.table("sessions").delete().eq("session_token", request.session_token).execute()

    reply = (f"Thank you! Your cart with a subtotal of *GHS {request.total_amount:.2f}* is confirmed.\n\n"
             "To proceed, would you like *delivery* or will you *pickup* the order yourself?")
    await send_whatsapp_message(phone_number, reply)
    
    return {"status": "order_confirmed_on_whatsapp", "order_id": order_res.data[0]['id']}

@app.get("/")
async def root():
    return {"message": "WhatsApp MarketBot Backend (Hybrid) is running."}

# You can add your full /payment-success endpoint here
@app.post("/payment-success")
async def payment_success_webhook(request: Request):
    logger.info("Received a call to /payment-success webhook.")
    return JSONResponse(status_code=200, content={"status": "received"})