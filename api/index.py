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
    API_KEY: str

settings = Settings()

# --- Constants, Models, FastAPI App ---
OrderStatus = Literal[
    "pending_confirmation", "awaiting_location", "awaiting_location_confirmation",
    "pending_payment", "processing", "out-for-delivery", "delivered", "cancelled", "failed"
]
class OrderRequest(BaseModel):
    session_token: str
    items: List[Dict]
    total_amount: float

app = FastAPI(title="WhatsApp MarketBot API (State-First)", version="3.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if security and admin_router: app.include_router(admin_router.router, prefix="/admin", tags=["admin"], dependencies=[Depends(security.get_admin_user)])
if auth_router: app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
if public_router: app.include_router(public_router.router)


# --- AI BRAIN (WITH CONTEXT) ---
async def get_intent_with_context(user_message: str, last_bot_message: Optional[str] = None) -> Dict[str, Any]:
    if not settings.GEMINI_API_KEY:
        lower_msg = user_message.lower()
        if any(word in lower_msg for word in ["buy", "menu"]): return {"intent": "start_order"}
        if any(word in lower_msg for word in ["status"]): return {"intent": "check_status"}
        if any(word in lower_msg for word in ["cart", "items"]): return {"intent": "show_cart"}
        if any(word in lower_msg for word in ["no", "good", "moment", "all"]): return {"intent": "end_conversation"}
        if any(word in lower_msg for word in ["thank", "ok"]): return {"intent": "polite_acknowledgement"}
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
    except Exception as e:
        logger.error(f"Error in get_intent_with_context: {e}", exc_info=True)
        return {"intent": "greet"}


# --- HELPER FUNCTIONS ---
def generate_order_number(): return f"ORD-{int(datetime.now(timezone.utc).timestamp())}"
def calculate_delivery_fee(lat: float, lon: float) -> float: return 15.00
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
    if not send_whatsapp_message_available or not supabase: return
    try:
        await send_whatsapp_message(phone, message)
        supabase.table("users").update({"last_bot_message": message}).eq("id", user_id).execute()
    except Exception as e:
        logger.error(f"Error in send_and_save_message for user {user_id}: {e}")

# --- PRIMARY WEBHOOK (STATE-FIRST ARCHITECTURE) ---
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
        user = user_res.data[0] if not is_new_user else supabase.table("users").insert({"phone_number": from_number_clean}).execute().data[0]
        user_id = user['id']
        last_bot_message = user.get('last_bot_message')

        # --- STATE-FIRST LOGIC (HIGHEST PRIORITY) ---

        # 1. Check for an order awaiting delivery/pickup choice
        pending_confirm_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", "pending_confirmation").order("created_at", desc=True).limit(1).execute()
        if pending_confirm_res.data:
            order = pending_confirm_res.data[0]
            reply = ""
            if "delivery" in incoming_msg_body:
                supabase.table("orders").update({"status": "awaiting_location"}).eq("id", order['id']).execute()
                reply = "Great! To calculate the delivery fee, please share your location using the WhatsApp location feature.\n\nTap the *clip icon 📎*, then choose *'Location' 📍*."
            elif "pickup" in incoming_msg_body:
                supabase.table("orders").update({"status": "pending_payment", "delivery_type": "pickup"}).eq("id", order['id']).execute()
                payment_link = await generate_paystack_payment_link(order['id'], order['total_amount'], from_number_clean)
                reply = f"Alright, your order is set for pickup. Your total is *GHS {order['total_amount']:.2f}*. Please complete your payment here:\n\n{payment_link}"
            else:
                reply = "I didn't quite catch that. Please choose how you'd like to receive your order: *delivery* or *pickup*?"
            
            await send_and_save_message(from_number_clean, reply, user_id)
            return JSONResponse(content={}, status_code=200)

        # 2. Check for an order awaiting a location message
        awaiting_loc_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", "awaiting_location").order("created_at", desc=True).limit(1).execute()
        if awaiting_loc_res.data and is_location_message:
            order = awaiting_loc_res.data[0]
            lat, lon = float(form_data["Latitude"]), float(form_data["Longitude"])
            delivery_fee = calculate_delivery_fee(lat, lon)
            total_with_delivery = order['total_amount'] + delivery_fee
            
            update_data = {"status": "pending_payment", "delivery_fee": delivery_fee, "total_with_delivery": total_with_delivery, "delivery_location_lat": lat, "delivery_location_lon": lon}
            supabase.table("orders").update(update_data).eq("id", order['id']).execute()
            
            payment_link = await generate_paystack_payment_link(order['id'], total_with_delivery, from_number_clean)
            reply = f"Thank you! Your delivery fee is GHS {delivery_fee:.2f}. Your new total is *GHS {total_with_delivery:.2f}*.\n\nPlease use this link to pay:\n{payment_link}"
            await send_and_save_message(from_number_clean, reply, user_id)
            return JSONResponse(content={}, status_code=200)
        
        # 3. Check for an order awaiting payment
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
            await send_and_save_message(from_number_clean, reply, user_id)
            return JSONResponse(content={}, status_code=200)

        # --- INTENT-BASED LOGIC (IF NO PRIORITY STATES ARE ACTIVE) ---
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
        
        elif intent == 'show_cart':
            pending_order_res = supabase.table("orders").select("items_json, total_amount").eq("user_id", user_id).in_("status", ["pending_confirmation", "awaiting_location", "pending_payment"]).order("created_at", desc=True).limit(1).execute()
            if pending_order_res.data and pending_order_res.data[0].get("items_json"):
                order = pending_order_res.data[0]
                items, total_amount = order["items_json"], order["total_amount"]
                product_ids = [item.get("product_id") for item in items if item.get("product_id")]
                product_details_map = {p["id"]: p for p in supabase.table("products").select("id, name, price").in_("id", product_ids).execute().data} if product_ids else {}
                
                cart_summary_items = [f'- {product_details_map.get(item["product_id"], {}).get("name", "Unknown Item")} x {item["quantity"]}' for item in items]
                reply_message = f"🛒 *Your Current Cart:*\n" + "\n".join(cart_summary_items) + f"\n\n*Total: GHS {total_amount:.2f}*"
            else:
                reply_message = "You don't have a pending order right now. Say 'menu' to start one!"

        elif intent == 'polite_acknowledgement':
            # Check if the last message from the bot was an "end conversation" message
            if last_bot_message == "Alright, have a great day! Feel free to message me anytime you need groceries.":
                reply_message = "You're welcome!" # Simple acknowledgement, no re-prompt
            else:
                reply_message = "You're welcome! Is there anything else I can help with?"

        elif intent == 'end_conversation':
            reply_message = "Alright, have a great day! Feel free to message me anytime you need groceries."
        
        else: # greet or fallback
            reply_message = "Welcome back! How can I help with your groceries today? (You can say 'menu' or 'status')"
            if is_new_user:
                reply_message = "Hello and welcome to Fresh Market GH! 🌿 I'm your personal assistant for ordering fresh groceries. You can say 'menu' to start shopping, or 'status' to check an order."

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
    if not supabase: raise HTTPException(500, "Server module unavailable")
    
    session_res = supabase.table("sessions").select("*").eq("session_token", request.session_token).limit(1).execute()
    if not session_res.data: raise HTTPException(404, "Session invalid")
    user_id, phone_number = session_res.data[0]['user_id'], session_res.data[0]['phone_number']

    # Delete the session after it's been used to confirm an order.
    # This replaces the line that was attempting to set session_token to None.
    supabase.table("sessions").delete().eq("session_token", request.session_token).execute()

    product_ids = [item.get("product_id") for item in request.items if item.get("product_id")]
    product_details_map = {p["id"]: p for p in supabase.table("products").select("id, name").in_("id", product_ids).execute().data} if product_ids else {}
    
    ordered_items_list = [f'- {product_details_map.get(item["product_id"], {}).get("name", f"ID: {item.get("product_id")}")} x {item["quantity"]}' for item in request.items]

    order_data = {"user_id": user_id, "items_json": [item for item in request.items], "total_amount": request.total_amount, "status": "pending_confirmation", "payment_status": "unpaid", "order_number": generate_order_number(), "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()}
    order_res = supabase.table("orders").insert(order_data).execute()
    if not order_res.data: raise HTTPException(500, "Could not create order")

    reply = (f"Thank you for confirming your items!\n\n*Your Order:*\n" + "\n".join(ordered_items_list) +
             f"\n\nSubtotal: *GHS {request.total_amount:.2f}*\n\nTo proceed, would you like *delivery* or will you *pickup* the order yourself?")
    await send_and_save_message(phone_number, reply, user_id)
    
    return {"status": "order_confirmed_on_whatsapp", "order_id": order_res.data[0]['id']}

@app.get("/")
async def root(): return {"message": "WhatsApp MarketBot Backend is running."}