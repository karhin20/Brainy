from fastapi import FastAPI, Request, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import os
from .supabase_client import supabase
from fastapi import status
import uuid
import httpx
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import json
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WhatsApp MarketBot API",
    description="API for WhatsApp-based food market ordering system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "https://yourdomain.com")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Environment variables
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://yourdomain.com/select-items")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent")
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
PAYSTACK_PAYMENT_URL = os.getenv("PAYSTACK_PAYMENT_URL", "https://api.paystack.co/transaction/initialize")
API_KEY = os.getenv("API_KEY")

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

# Pydantic models for request validation
class OrderItem(BaseModel):
    product_id: str
    quantity: int = Field(gt=0)

class OrderRequest(BaseModel):
    user_id: str
    phone_number: str
    items: list[OrderItem]
    total_amount: float = Field(gt=0)

class DeliveryStatusUpdate(BaseModel):
    order_id: str
    status: str

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

@app.get("/")
async def root():
    return {"message": "WhatsApp MarketBot Backend is running."}

async def call_gemini_intent_extraction(message: str) -> dict:
    """
    Call Gemini API to extract intent and product info from the user's message.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set, using fallback")
        return {"intent": "buy", "products": ["tomatoes", "onions"] if "tomato" in message.lower() or "onion" in message.lower() else []}
    
    try:
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        prompt = (
            "Extract the user's intent (e.g., buy, ask, greet) and a list of food products mentioned from this message. "
            "Respond as JSON: {\"intent\": string, \"products\": string[]}\nMessage: " + message
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
            response.raise_for_status()
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}", exc_info=True)
        return {"intent": "buy", "products": ["tomatoes", "onions"] if "tomato" in message.lower() or "onion" in message.lower() else []}

async def send_whatsapp_message(to_number: str, message: str) -> dict:
    """
    Send a WhatsApp message using Twilio API.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_NUMBER):
        logger.error("Twilio credentials not set")
        return {"error": "Twilio credentials not set"}
    
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        data = {
            "From": f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
            "To": to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}",
            "Body": message
        }
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, data=data, auth=auth)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Twilio API error: {str(e)}", exc_info=True)
        return {"error": str(e)}

async def generate_paystack_payment_link(order_id: str, amount: float, user_phone: str) -> str:
    """
    Generate a Paystack payment link for the order.
    """
    if not PAYSTACK_SECRET_KEY:
        logger.warning("PAYSTACK_SECRET_KEY not set, using mock payment link")
        return f"https://paystack.com/pay/mock-{order_id}"
    
    try:
        headers = {
            "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "amount": int(amount * 100),  # Paystack expects amount in kobo
            "currency": "GHS",
            "reference": order_id,
            "callback_url": f"{FRONTEND_URL}/payment-success?order_id={order_id}",
            "channels": ["card", "mobile_money"],
            "metadata": {"order_id": order_id, "phone": user_phone}
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(PAYSTACK_PAYMENT_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["data"]["authorization_url"]
    except Exception as e:
        logger.error(f"Paystack API error: {str(e)}", exc_info=True)
        return f"https://paystack.com/pay/mock-{order_id}"

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    try:
        data = await request.json()
        from_number = data.get("From")
        body = data.get("Body", "").strip()
        session_token = data.get("session") or str(uuid.uuid4())

        # Validate Twilio request
        if not from_number:
            raise HTTPException(status_code=400, detail="Missing From number")

        # Check if this is a delivery type or location response
        user_query = supabase.table("users").select("id").eq("phone_number", from_number).limit(1).execute()
        if user_query.data:
            user_id = user_query.data[0]["id"]
            order_query = supabase.table("orders").select("*").eq("user_id", user_id).eq("status", "pending").order("created_at", desc=True).limit(1).execute()
            
            if order_query.data:
                order = order_query.data[0]
                
                # Check if this is a delivery type response
                if order.get("delivery_type") is None:
                    if body.lower() in ["delivery", "pickup"]:
                        delivery_type = body.lower()
                        delivery_fee = 0.00 if delivery_type == "pickup" else None
                        total_with_delivery = order["total_amount"] if delivery_type == "pickup" else None
                        
                        update_data = {
                            "delivery_type": delivery_type,
                            "delivery_fee": delivery_fee,
                            "total_with_delivery": total_with_delivery
                        }
                        
                        if delivery_type == "delivery":
                            location_msg = (
                                "📍 Please share your delivery location by sending your address or location pin. "
                                "This will help us calculate the delivery fee."
                            )
                            await send_whatsapp_message(from_number, location_msg)
                        else:
                            # For pickup, generate payment link immediately
                            payment_link = await generate_paystack_payment_link(
                                order["id"], 
                                order["total_amount"], 
                                from_number
                            )
                            pay_msg = f"Thank you for choosing pickup! Please pay here: {payment_link}"
                            await send_whatsapp_message(from_number, pay_msg)
                        
                        supabase.table("orders").update(update_data).eq("id", order["id"]).execute()
                        return {"status": "delivery type saved", "order_id": order["id"]}
                
                # Check if this is a location response (only for delivery type)
                elif order.get("delivery_type") == "delivery" and order.get("delivery_location") is None:
                    # Calculate delivery fee based on location
                    delivery_fee = calculate_delivery_fee(body)
                    total_with_delivery = order["total_amount"] + delivery_fee
                    
                    location_update = {
                        "delivery_location": body,
                        "location_updated_at": datetime.utcnow().isoformat(),
                        "delivery_fee": delivery_fee,
                        "total_with_delivery": total_with_delivery
                    }
                    supabase.table("orders").update(location_update).eq("id", order["id"]).execute()
                    
                    # Send payment link with updated total
                    payment_link = await generate_paystack_payment_link(
                        order["id"], 
                        total_with_delivery, 
                        from_number
                    )
                    
                    confirm_msg = (
                        f"✅ Thank you for sharing your location!\n"
                        f"Delivery fee: ₵{delivery_fee:.2f}\n"
                        f"Total with delivery: ₵{total_with_delivery:.2f}\n"
                        f"Please pay here: {payment_link}"
                    )
                    await send_whatsapp_message(from_number, confirm_msg)
                    return {"status": "location saved", "order_id": order["id"]}

        # If not a delivery/location response, proceed with normal conversation flow
        session_query = supabase.table("sessions").select("*").eq("phone_number", from_number).eq("session_token", session_token).limit(1).execute()
        session = session_query.data[0] if session_query.data else None
        conversation_history = session["conversation_history"] if session else []
        
        # Append new user message
        conversation_history.append({
            "role": "user", 
            "content": body, 
            "timestamp": datetime.utcnow().isoformat()
        })

        # Call Gemini with full conversation history
        gemini_result = await call_gemini_intent_extraction(str(conversation_history))
        intent = gemini_result.get("intent")
        products = gemini_result.get("products", [])
        gemini_response = gemini_result.get("response") or gemini_result.get("message") or "How can I help you today?"
        
        # Append Gemini's response to history
        conversation_history.append({
            "role": "assistant", 
            "content": gemini_response, 
            "timestamp": datetime.utcnow().isoformat()
        })

        # Save session
        session_data = {
            "phone_number": from_number,
            "session_token": session_token,
            "conversation_history": conversation_history,
            "last_intent": intent,
            "updated_at": datetime.utcnow().isoformat()
        }
        if session:
            supabase.table("sessions").update(session_data).eq("id", session["id"]).execute()
        else:
            supabase.table("sessions").insert(session_data).execute()

        # If intent is 'buy' and products, send selection link, else send Gemini's response
        selection_url = f"{FRONTEND_URL}?session={session_token}"
        if intent == "buy" and products:
            reply_message = f"You want to buy: {', '.join(products)}. Please select your items here: {selection_url}"
        else:
            reply_message = gemini_response

        # Send WhatsApp message via Twilio
        await send_whatsapp_message(from_number, reply_message)

        return {
            "status": "reply sent",
            "to": from_number,
            "message": reply_message,
            "selection_url": selection_url if intent == "buy" and products else None,
            "intent": intent,
            "products": products,
            "session_token": session_token,
            "conversation_history": conversation_history
        }
    except Exception as e:
        logger.error(f"Error in whatsapp-webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/confirm-items")
async def confirm_items(request: OrderRequest, api_key: str = Depends(verify_api_key)):
    try:
        items_json = request.items
        order_data = {
            "user_id": request.user_id,
            "items_json": items_json,
            "total_amount": request.total_amount,
            "status": "pending",
            "payment_status": "unpaid",
            "delivery_location": None,
            "location_updated_at": None,
            "delivery_type": None,
            "delivery_fee": None,
            "total_with_delivery": None
        }
        result = supabase.table("orders").insert(order_data).execute()
        if result.error:
            raise HTTPException(status_code=500, detail=str(result.error))
        order_id = result.data[0]["id"] if result.data else None

        # Send delivery type request message
        delivery_msg = (
            "Please choose your delivery option:\n"
            "1. Delivery (we'll ask for your location)\n"
            "2. Pickup (no delivery fee)"
        )
        await send_whatsapp_message(request.phone_number, delivery_msg)

        return {"status": "order saved", "order_id": order_id}
    except Exception as e:
        logger.error(f"Error in confirm-items: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/payment-success")
async def payment_success(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        data = await request.json()
        event = data.get("event")
        payment_data = data.get("data", {})
        order_id = payment_data.get("reference")
        status_paid = payment_data.get("status") == "success"
        phone_number = payment_data.get("metadata", {}).get("phone")

        if not order_id or not status_paid:
            return {"status": "ignored", "reason": "No order_id or not successful payment"}

        # Update order in Supabase
        update_result = supabase.table("orders").update({
            "payment_status": "paid", 
            "status": "processing"
        }).eq("id", order_id).execute()
        
        # Notify user via WhatsApp
        if phone_number:
            msg = f"Payment received for your order! Your order is now being processed. Order ID: {order_id}"
            await send_whatsapp_message(phone_number, msg)
        
        return {"status": "payment updated", "order_id": order_id, "supabase_result": update_result.data}
    except Exception as e:
        logger.error(f"Error in payment-success: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delivery-status")
async def delivery_status(request: DeliveryStatusUpdate, api_key: str = Depends(verify_api_key)):
    try:
        # Update delivery_status table
        delivery_data = {
            "order_id": request.order_id,
            "status": request.status,
            "updated_at": datetime.utcnow().isoformat()
        }
        supabase.table("delivery_status").insert(delivery_data).execute()
        
        # Update order status
        supabase.table("orders").update({"status": request.status}).eq("id", request.order_id).execute()
        
        # Get user phone number
        order_query = supabase.table("orders").select("*").eq("id", request.order_id).limit(1).execute()
        order = order_query.data[0] if order_query.data else None
        phone_number = None
        if order:
            user_id = order.get("user_id")
            user_query = supabase.table("users").select("*").eq("id", user_id).limit(1).execute()
            user = user_query.data[0] if user_query.data else None
            phone_number = user.get("phone_number") if user else None
        
        # Notify user via WhatsApp
        if phone_number:
            msg = f"Your order {request.order_id} status updated: {request.status}"
            await send_whatsapp_message(phone_number, msg)
        
        return {"status": "delivery status updated", "order_id": request.order_id, "status_update": request.status}
    except Exception as e:
        logger.error(f"Error in delivery-status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def calculate_delivery_fee(location: str) -> float:
    """
    Calculate delivery fee based on location.
    This is a mock implementation - you should replace with actual logic.
    """
    # Mock implementation - you should replace with actual distance/zone-based calculation
    base_fee = 5.00  # Base delivery fee
    # Add some randomness to simulate different zones
    import random
    zone_multiplier = random.uniform(1.0, 2.0)
    return round(base_fee * zone_multiplier, 2) 