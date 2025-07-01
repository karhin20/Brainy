# --- START OF FILE: index.py (FINAL, WITH IMPROVED CONVERSATIONAL FLOW) ---

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
    TWILIO_AUTH_TOKEN: str; TWILIO_ACCOUNT_SID: str; TWILIO_WHATSAPP_NUMBER: str
    FRONTEND_URL: str = "https://marketmenu.vercel.app"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    PAYSTACK_SECRET_KEY: Optional[str] = None; API_KEY: str

settings = Settings()

# --- Constants, Models, FastAPI App ---
OrderStatus = Literal["pending_confirmation", "awaiting_location", "pending_payment", "processing", "out-for-delivery", "delivered", "cancelled", "failed"]
class OrderRequest(BaseModel):
    session_token: str; items: List[Dict]; total_amount: float

app = FastAPI(title="WhatsApp MarketBot API (Improved Flow)", version="3.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
if security and admin_router: app.include_router(admin_router.router, prefix="/admin", tags=["admin"], dependencies=[Depends(security.get_admin_user)])
if auth_router: app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
if public_router: app.include_router(public_router.router)


# --- REVISED AI BRAIN (WITH END_CONVERSATION INTENT) ---
async def get_intent_with_context(user_message: str, last_bot_message: Optional[str] = None) -> Dict[str, Any]:
    if not settings.GEMINI_API_KEY:
        lower_msg = user_message.lower()
        if any(word in lower_msg for word in ["buy", "menu"]): return {"intent": "start_order"}
        if any(word in lower_msg for word in ["status"]): return {"intent": "check_status"}
        if any(word in lower_msg for word in ["no", "good", "moment", "all"]): return {"intent": "end_conversation"}
        if any(word in lower_msg for word in ["thank", "ok"]): return {"intent": "polite_acknowledgement"}
        return {"intent": "greet"}

    prompt = f"""
    Analyze the user's message for a grocery bot based on the context of the bot's last message. Respond ONLY with a single, minified JSON object.

    CONTEXT:
    The bot's last message to the user was: "{last_bot_message or 'No previous message.'}"

    User's New Message: "{user_message}"

    Your JSON output MUST contain one key, "intent", with one of these values:
    - `start_order`: User wants to start a new order, browse the menu, or buy something.
    - `check_status`: User is asking about an existing order's status.
    - `polite_acknowledgement`: User is saying "thanks", "ok", "got it". This is a polite but potentially open-ended reply.
    - `end_conversation`: User is clearly ending the conversation. Examples: "no thanks", "I am good", "that's all", "not at the moment". This is a closing statement.
    - `greet`: A general greeting or question that doesn't fit other categories. This is a conversation starter.

    Example 1:
    Bot's last message: "Your order is currently 'processing'."
    User's New Message: "okay thanks"
    JSON: {{"intent": "polite_acknowledgement"}}

    Example 2:
    Bot's last message: "Is there anything else I can help with?"
    User's New Message: "no I'm good"
    JSON: {{"intent": "end_conversation"}}
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "application/json"}}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(settings.GEMINI_API_URL, headers={"Content-Type": "application/json"}, params={"key": settings.GEMINI_API_KEY}, json=payload)
            res.raise_for_status()
            return json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        logger.error(f"Error in get_intent_with_context: {e}", exc_info=True)
        return {"intent": "greet"}

# --- HELPER FUNCTIONS ---
async def send_and_save_message(phone: str, message: str, user_id: str):
    if not send_whatsapp_message_available or not supabase: return
    try:
        await send_whatsapp_message(phone, message)
        supabase.table("users").update({"last_bot_message": message}).eq("id", user_id).execute()
    except Exception as e:
        logger.error(f"Error in send_and_save_message for user {user_id}: {e}")

# (Other helpers like generate_order_number, generate_paystack_payment_link, etc.)

# --- PRIMARY WEBHOOK (WITH IMPROVED GREETINGS & EXITS) ---
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
        if not supabase: return JSONResponse(content={}, status_code=200)

        user_res = supabase.table("users").select("id, last_bot_message").eq("phone_number", from_number_clean).limit(1).execute()
        is_new_user = not user_res.data
        if is_new_user:
            user = supabase.table("users").insert({"phone_number": from_number_clean, "last_bot_message": None}).execute().data[0]
        else:
            user = user_res.data[0]
        user_id = user['id']
        last_bot_message = user.get('last_bot_message')

        # --- STATE-BASED LOGIC (REMAINS HIGHEST PRIORITY) ---
        # ... (Your robust logic for pending_payment, pending_confirmation, etc. goes here)

        # --- INTENT-BASED LOGIC (IF NO PRIORITY STATES) ---
        ai_result = await get_intent_with_context(incoming_msg_body, last_bot_message)
        intent = ai_result.get("intent")
        
        reply_message = ""

        if intent == 'start_order':
            new_session_token = str(uuid.uuid4())
            expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            session_payload = {"user_id": user_id, "phone_number": from_number_clean, "session_token": new_session_token, "expires_at": expires_at}
            supabase.table("sessions").upsert(session_payload, on_conflict="user_id").execute()
            menu_url = f"{settings.FRONTEND_URL}?session={new_session_token}"
            reply_message = f"Great! Please use the link below to browse our menu and add items to your cart. Return to this chat after you confirm your items on the website!\n\n{menu_url}"

        elif intent == 'check_status':
            paid_order_res = supabase.table("orders").select("status, order_number").eq("user_id", user_id).eq("payment_status", "paid").order("created_at", desc=True).limit(1).execute()
            reply_message = f"Your most recent order ({paid_order_res.data[0]['order_number']}) is currently '{paid_order_res.data[0]['status']}'." if paid_order_res.data else "It looks like you don't have any active orders with us. To start one, just say 'menu'."
        
        elif intent == 'polite_acknowledgement':
            reply_message = "You're welcome! Is there anything else I can help with?"

        elif intent == 'end_conversation':
            reply_message = "Alright, have a great day! Feel free to message me anytime you need groceries."
        
        else: # This covers 'greet' and any other fallback
            if is_new_user:
                reply_message = "Hello and welcome to Fresh Market GH! 🌿 I'm your personal assistant for ordering fresh groceries. You can say 'menu' to start shopping, or 'status' to check an order."
            else:
                reply_message = "Welcome back! How can I help with your groceries today? (You can say 'menu' or 'status')"

        await send_and_save_message(from_number_clean, reply_message, user_id)
        return JSONResponse(content={}, status_code=200)

    except Exception as e:
        logger.error(f"Critical webhook error for {from_number_clean}: {e}", exc_info=True)
        if from_number_clean != "unknown" and send_whatsapp_message_available:
            await send_whatsapp_message(from_number_clean, "I'm sorry, an unexpected error occurred. Please try again.")
        return JSONResponse(content={}, status_code=200)

# --- WEB-BASED ENDPOINTS ---
@app.post("/confirm-items")
async def confirm_items(request: OrderRequest):
    # This logic remains the same, but we use the send_and_save_message helper
    if not supabase: raise HTTPException(500, "Server module unavailable")
    session_res = supabase.table("sessions").select("*").eq("session_token", request.session_token).limit(1).execute()
    if not session_res.data: raise HTTPException(404, "Session expired or invalid")
    session_data = session_res.data[0]
    user_id, phone_number = session_data['user_id'], session_data['phone_number']

    supabase.table("sessions").update({"session_token": None}).eq("user_id", user_id).execute()

    order_data = { "user_id": user_id, "items_json": [item for item in request.items], "total_amount": request.total_amount, "status": "pending_confirmation", "payment_status": "unpaid", "order_number": f"ORD-{int(datetime.now().timestamp())}", "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat() }
    order_res = supabase.table("orders").insert(order_data).execute()
    if not order_res.data: raise HTTPException(500, "Could not create order")

    reply = (f"Thank you! Your cart with a subtotal of *GHS {request.total_amount:.2f}* is confirmed.\n\n"
             "To proceed, would you like *delivery* or will you *pickup* the order yourself?")
    await send_and_save_message(phone_number, reply, user_id)
    
    return {"status": "order_confirmed_on_whatsapp", "order_id": order_res.data[0]['id']}

@app.get("/")
async def root(): return {"message": "WhatsApp MarketBot Backend is running."}