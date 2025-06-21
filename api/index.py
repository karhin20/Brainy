from fastapi import FastAPI, Request, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import uuid
import httpx
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import json
from pydantic import BaseModel, Field

# Add the parent directory to the path so we can import supabase_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from supabase_client import supabase
except ImportError:
    # Fallback for when supabase_client is not available
    supabase = None

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
    allow_origins=["*"],  # Allow all origins for Vercel deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 

TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://marketmenu.vercel.app")
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
            status_code=401,
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
    return {"message": "WhatsApp MarketBot Backend is running on Vercel."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

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
        # Twilio sends form data, not JSON
        form_data = await request.form()
        
        from_number = form_data.get("From")
        body = form_data.get("Body", "").strip()
        session_token = form_data.get("session") or str(uuid.uuid4())

        # Validate Twilio request
        if not from_number:
            raise HTTPException(status_code=400, detail="Missing From number")

        # Check if this is a delivery type or location response
        if supabase:
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
                            
                            # Update order with delivery type
                            supabase.table("orders").update({
                                "delivery_type": delivery_type,
                                "delivery_fee": delivery_fee,
                                "total_with_delivery": total_with_delivery
                            }).eq("id", order["id"]).execute()
                            
                            if delivery_type == "delivery":
                                response_message = "Please share your delivery location:"
                            else:
                                response_message = "Great! You can pick up your order at our store. Please proceed to payment."
                        else:
                            response_message = "Please choose delivery type: 'delivery' or 'pickup'"
                    else:
                        # Handle location input for delivery
                        if order.get("delivery_type") == "delivery" and not order.get("delivery_location"):
                            delivery_fee = calculate_delivery_fee(body)
                            total_with_delivery = order["total_amount"] + delivery_fee
                            
                            # Update order with location and delivery fee
                            supabase.table("orders").update({
                                "delivery_location": body,
                                "delivery_fee": delivery_fee,
                                "total_with_delivery": total_with_delivery
                            }).eq("id", order["id"]).execute()
                            
                            payment_link = await generate_paystack_payment_link(order["id"], total_with_delivery, from_number)
                            response_message = f"Delivery fee: GHS {delivery_fee:.2f}\nTotal: GHS {total_with_delivery:.2f}\n\nPay here: {payment_link}"
                        else:
                            response_message = "Your order is being processed. You'll receive updates shortly."
                else:
                    # New user or no pending order
                    intent_data = await call_gemini_intent_extraction(body)
                    response_message = f"Welcome! I detected you want to {intent_data['intent']}. Available products: {', '.join(intent_data['products'])}"
            else:
                # Create new user
                new_user = supabase.table("users").insert({
                    "phone_number": from_number,
                    "session_token": session_token
                }).execute()
                
                intent_data = await call_gemini_intent_extraction(body)
                response_message = f"Welcome! I detected you want to {intent_data['intent']}. Available products: {', '.join(intent_data['products'])}"
        else:
            # Fallback when Supabase is not available
            response_message = "Welcome to our food market! How can I help you today?"

        # Send WhatsApp response
        await send_whatsapp_message(from_number, response_message)
        
        return {"status": "success", "message": "Webhook processed successfully"}
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/confirm-items")
async def confirm_items(request: OrderRequest, api_key: str = Depends(verify_api_key)):
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database connection not available")
            
        order_id = str(uuid.uuid4())
        
        # Create order in database
        order_data = {
            "id": order_id,
            "user_id": request.user_id,
            "phone_number": request.phone_number,
            "items": request.items,
            "total_amount": request.total_amount,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        supabase.table("orders").insert(order_data).execute()
        
        # Generate payment link
        payment_link = await generate_paystack_payment_link(order_id, request.total_amount, request.phone_number)
        
        return {
            "order_id": order_id,
            "payment_link": payment_link,
            "status": "pending"
        }
        
    except Exception as e:
        logger.error(f"Confirm items error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/payment-success")
async def payment_success(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        data = await request.json()
        order_id = data.get("order_id")
        
        if not order_id:
            raise HTTPException(status_code=400, detail="Missing order_id")
            
        if not supabase:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Update order status
        supabase.table("orders").update({
            "status": "paid",
            "payment_confirmed_at": datetime.now().isoformat()
        }).eq("id", order_id).execute()
        
        # Get order details for notification
        order_query = supabase.table("orders").select("*").eq("id", order_id).execute()
        if order_query.data:
            order = order_query.data[0]
            notification_message = f"Payment confirmed for order {order_id}. Your order is being prepared!"
            await send_whatsapp_message(order["phone_number"], notification_message)
        
        return {"status": "success", "message": "Payment confirmed"}
        
    except Exception as e:
        logger.error(f"Payment success error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delivery-status")
async def delivery_status(request: DeliveryStatusUpdate, api_key: str = Depends(verify_api_key)):
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Update order status
        supabase.table("orders").update({
            "status": request.status,
            "updated_at": datetime.now().isoformat()
        }).eq("id", request.order_id).execute()
        
        # Get order details for notification
        order_query = supabase.table("orders").select("*").eq("id", request.order_id).execute()
        if order_query.data:
            order = order_query.data[0]
            status_message = f"Order {request.order_id} status: {request.status}"
            await send_whatsapp_message(order["phone_number"], status_message)
        
        return {"status": "success", "message": f"Order status updated to {request.status}"}
        
    except Exception as e:
        logger.error(f"Delivery status error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def calculate_delivery_fee(location: str) -> float:
    """
    Calculate delivery fee based on location.
    """
    # Simple delivery fee calculation
    base_fee = 5.00
    if "accra" in location.lower():
        return base_fee
    elif "tema" in location.lower():
        return base_fee + 2.00
    else:
        return base_fee + 3.00

# Export the app for Vercel
app.debug = False 