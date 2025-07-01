# --- START OF FILE: index.py (FINAL, STATE-FIRST ARCHITECTURE) ---

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

# --- (Standard imports, logging, settings, etc. remain the same) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ... (Supabase, utils, security imports) ...
# ... (Settings class) ...
# ... (Constants, Pydantic Models, FastAPI App setup) ...

# --- AI Brain & Helpers (largely the same) ---
# ... (get_intent_with_context, send_and_save_message, generate_order_number, etc.) ...
async def get_intent_with_context(user_message: str, last_bot_message: Optional[str] = None) -> Dict[str, Any]:
    # This function is now only called when no priority state is active.
    # Its definition remains the same as the previous version.
    # ...
    pass # Placeholder for brevity

async def send_and_save_message(phone: str, message: str, user_id: str):
    # ...
    pass # Placeholder for brevity


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
        user = user_res.data[0] if user_res.data else supabase.table("users").insert({"phone_number": from_number_clean}).execute().data[0]
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
                payment_link = "https://paystack.com/pay/mock-link" # await generate_paystack_payment_link(...)
                reply = f"Alright, your order is set for pickup. Your total is *GHS {order['total_amount']:.2f}*. Please complete your payment here:\n\n{payment_link}"
            else:
                reply = "I didn't quite catch that. Please choose how you'd like to receive your order: *delivery* or *pickup*?"
            
            await send_and_save_message(from_number_clean, reply, user_id)
            return JSONResponse(content={}, status_code=200)

        # 2. Check for an order awaiting a location message
        # (This logic block can be added here, similar to previous versions)

        # 3. Check for an order awaiting payment
        # (This logic block can be added here, similar to previous versions)

        # --- INTENT-BASED LOGIC (IF NO PRIORITY STATES ARE ACTIVE) ---
        ai_result = await get_intent_with_context(incoming_msg_body, last_bot_message)
        intent = ai_result.get("intent")
        
        reply_message = "Hello! How can I help? (You can say 'menu' or 'status')"

        if intent == 'show_cart':
            # This logic remains the same - it fetches the latest unpaid order and shows its contents.
            # ...
            pass # Placeholder for brevity

        elif intent == 'start_order':
            # ... (Same logic as before to generate a session and send menu link)
            pass # Placeholder for brevity
        
        # ... (Handle other intents like check_status, polite_acknowledgement, end_conversation)

        await send_and_save_message(from_number_clean, reply_message, user_id)
        return JSONResponse(content={}, status_code=200)

    except Exception as e:
        logger.error(f"Critical webhook error for {from_number_clean}: {e}", exc_info=True)
        # ... (Error handling)
        return JSONResponse(content={}, status_code=200)


# --- WEB-BASED ENDPOINTS ---
@app.post("/confirm-items")
async def confirm_items(request: OrderRequest):
    if not supabase: raise HTTPException(500, "Server module unavailable")
    
    # ... (Session validation)
    session_res = supabase.table("sessions").select("*").eq("session_token", request.session_token).limit(1).execute()
    if not session_res.data: raise HTTPException(404, "Session invalid")
    user_id = session_res.data[0]['user_id']
    phone_number = session_res.data[0]['phone_number']

    # --- FIX FOR PRODUCT NAMES ---
    # 1. Get all product IDs from the incoming request
    product_ids = [item.get("product_id") for item in request.items if item.get("product_id")]
    
    product_details_map = {}
    if product_ids:
        # 2. Fetch the details (name, price) for these products from your DB
        products_res = supabase.table("products").select("id, name, price").in_("id", product_ids).execute()
        if products_res.data:
            product_details_map = {p["id"]: {"name": p["name"], "price": p["price"]} for p in products_res.data}

    # 3. Build the summary message using the fetched names
    ordered_items_list = "Here's what you ordered:\n"
    for item in request.items:
        product_id = item.get("product_id")
        product_name = product_details_map.get(product_id, {}).get("name", f"Product ID: {product_id}")
        quantity = item.get("quantity", 1)
        ordered_items_list += f"- {product_name} x {quantity}\n"

    # Invalidate the session
    supabase.table("sessions").update({"session_token": None}).eq("user_id", user_id).execute()

    # Create the order
    order_data = {
        "user_id": user_id,
        "items_json": [item for item in request.items],
        "total_amount": request.total_amount,
        "status": "pending_confirmation", # This is the crucial next state
        "payment_status": "unpaid",
        # ... other fields
    }
    order_res = supabase.table("orders").insert(order_data).execute()
    if not order_res.data: raise HTTPException(500, "Could not create order")

    # 4. Send the well-formatted message to the user
    reply = (
        f"Thank you for confirming your items!\n\n"
        f"{ordered_items_list}\n"
        f"Your subtotal of *GHS {request.total_amount:.2f}* is confirmed.\n\n"
        "To proceed, would you like *delivery* or will you *pickup* the order yourself?"
    )
    await send_and_save_message(phone_number, reply, user_id)
    
    return {"status": "order_confirmed_on_whatsapp", "order_id": order_res.data[0]['id']}

# ... (Root endpoint)