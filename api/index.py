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

async def call_gemini_intent_extraction(message: str, user_context: dict) -> dict:
    """
    Call Gemini API to extract intent, products, and a response.
    """
    if not GEMINI_API_KEY:
        # Fallback for development if Gemini key is not set
        if "order" in message: return {"intent": "check_status"}
        if "buy" in message: return {"intent": "buy", "products": ["yam"]}
        return {"intent": "greet", "response": "Hello! Welcome back to Ghana Fresh Market."}

    try:
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        
        # Build a more context-aware prompt
        prompt_context = f"The user's name is {user_context.get('name', 'a returning customer')}."
        if user_context.get('has_paid_order'):
            prompt_context += " They have a paid order that is currently being processed."

        prompt = (
            'You are a friendly and helpful assistant for "Ghana Fresh Market", a WhatsApp-based grocery store in Ghana. '
            'Your primary goal is to help users buy food items or check on their existing orders. '
            f'{prompt_context}\n\n'
            "Analyze the user's message and respond in pure JSON format with three fields: 'intent', 'products' (a list of strings, can be empty), and 'response' (a conversational string). \n"
            "Here are the possible intents:\n"
            '- **buy**: The user wants to buy one or more items. Extract the items into the "products" list.\n'
            '- **check_status**: The user is asking about their order (e.g., "where is my order?").\n'
            '- **greet**: The user is just saying hello.\n'
            '- **cancel_order**: The user wants to cancel their current pending order.\n'
            '- **help**: The user is asking for help or is confused.\n'
            '- **ask**: The user is asking a general question.\n\n'
            "Example Scenarios:\n"
            '1. Message: "hi i want to buy yam and onions" -> {"intent": "buy", "products": ["yam", "onions"], "response": "Great! I can help with that."}\n'
            '2. Message: "where is my stuff" -> {"intent": "check_status", "products": [], "response": "Let me check on your order for you."}\n'
            '3. Message: "hello" -> {"intent": "greet", "products": [], "response": "Hello! Welcome to Ghana Fresh Market. What can I get for you today?"}\n\n'
            "User Message: " + message
        )

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
            response.raise_for_status()
            result = response.json()
            
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            try:
                json_start_index = text.find('{')
                json_end_index = text.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1:
                    json_str = text[json_start_index:json_end_index]
                    return json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON object found", text, 0)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from Gemini response: {text}")
                return {"intent": "ask", "products": [], "response": text} # Return the raw text if parsing fails
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}", exc_info=True)
        return {"intent": "error", "products": [], "response": "Sorry, I'm having a little trouble understanding. Could you try rephrasing?"}

async def send_whatsapp_message(to_number: str, message: str) -> dict:
    """
    Send a WhatsApp message using Twilio API.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_NUMBER):
        logger.error("Twilio credentials not set")
        return {"error": "Twilio credentials not set"}
    
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        
        # Ensure 'whatsapp:' prefix is present for outgoing messages
        to_prefixed = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"
        
        data = {
            "From": f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
            "To": to_prefixed,
            "Body": message
        }
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, data=data, auth=auth)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        # Log the detailed error response from Twilio
        error_response = e.response.json()
        logger.error(f"Twilio API error: {str(e)} - Response: {error_response}", exc_info=True)
        return {"error": str(e), "details": error_response}
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

async def handle_pending_order(order: Dict[str, Any], body: str, from_number: str) -> str:
    """Handles interaction when a user already has an order with 'pending' status."""
    order_id = order["id"]
    
    # Scenario A: Awaiting delivery type choice
    if order.get("delivery_type") is None:
        if body.lower() == "delivery":
            supabase.table("orders").update({"delivery_type": "delivery"}).eq("id", order_id).execute()
            return "Got it. Please share your delivery location so we can calculate the fee."
        elif body.lower() == "pickup":
            total_amount = order["total_amount"]
            payment_link = await generate_paystack_payment_link(order_id, total_amount, from_number)
            supabase.table("orders").update({"delivery_type": "pickup", "delivery_fee": 0, "total_with_delivery": total_amount}).eq("id", order_id).execute()
            return f"Pickup selected. Your total is ₵{total_amount:.2f}. Please pay here to complete your order: {payment_link}"
        else:
            return "Please reply with either *Delivery* or *Pickup* to proceed."

    # Scenario B: Awaiting location for a delivery order
    elif order.get("delivery_type") == "delivery" and order.get("delivery_location") is None:
        delivery_fee = calculate_delivery_fee(body)
        total_with_delivery = order["total_amount"] + delivery_fee
        payment_link = await generate_paystack_payment_link(order_id, total_with_delivery, from_number)
        
        supabase.table("orders").update({
            "delivery_location": body,
            "location_updated_at": datetime.utcnow().isoformat(),
            "delivery_fee": delivery_fee,
            "total_with_delivery": total_with_delivery
        }).eq("id", order_id).execute()
        
        return f"Thanks! Your delivery fee is ₵{delivery_fee:.2f}, bringing your total to ₵{total_with_delivery:.2f}. Please pay here to finalize: {payment_link}"

    # Scenario C: User has a pending order but sends a new message
    else:
        if body.lower() in ['cancel', 'cancel order']:
            supabase.table("orders").update({"status": "cancelled"}).eq("id", order_id).execute()
            return "Your pending order has been cancelled. Feel free to start a new one!"
        
        # If user tries to buy something new while having a pending order
        intent_data = await call_gemini_intent_extraction(body, {})
        if intent_data.get('intent') == 'buy':
            return "You already have a pending order. To make changes, please reply with *Cancel* to cancel the current one first, then you can start a new order."

        total = order.get("total_with_delivery") or order.get("total_amount")
        payment_link = await generate_paystack_payment_link(order_id, total, from_number)
        return f"It looks like you have an unpaid order with us. To complete it, please use this payment link: {payment_link}\n\nIf you want to cancel it, just reply with *Cancel*."

async def handle_new_conversation(user: Dict[str, Any], body: str, from_number: str) -> str:
    """Handles interaction when a user has no pending orders."""
    user_id = user['id']

    # Context for Gemini
    user_context = {
        "name": user.get("name"),
        "has_paid_order": bool(supabase.table("orders").select("id").eq("user_id", user_id).in_("status", ["processing", "out-for-delivery"]).limit(1).execute().data)
    }

    intent_data = await call_gemini_intent_extraction(body, user_context)
    intent = intent_data.get("intent")

    if intent == "buy":
        products = intent_data.get("products", [])
        if not products:
            return "It sounds like you want to buy something, but I couldn't catch what items. Could you please list them?"
        session_token = str(uuid.uuid4())
        selection_url = f"{FRONTEND_URL}?session={session_token}"
        supabase.table("sessions").insert({
            "user_id": user_id,
            "phone_number": from_number,
            "session_token": session_token,
            "last_intent": "buy"
        }).execute()
        return f"Great! Please select the exact items and quantities from this secure link: {selection_url}"
    
    elif intent == "check_status":
        paid_orders = supabase.table("orders").select("status").eq("user_id", user_id).in_("status", ["processing", "out-for-delivery"]).order("created_at", desc=True).limit(1).execute().data
        if paid_orders:
            return f"I've found your latest order. Its current status is: *{paid_orders[0]['status']}*."
        else:
            return "It looks like you don't have any paid orders with us right now. Can I help you start a new one?"
            
    elif intent == "greet":
        return f"Hello! Welcome back to Ghana Fresh Market. What can I help you with today?"

    else: # ask, help, error, etc.
        return intent_data.get("response", "I'm not sure how to help with that. Could you try rephrasing?")



@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    try:
        form_data = await request.form()
        raw_from_number = form_data.get("From")
        body = form_data.get("Body", "").strip()
        from_number = raw_from_number.replace("whatsapp:", "") if raw_from_number else None

        if not from_number:
            raise HTTPException(status_code=400, detail="Missing From number")

        if not supabase:
            await send_whatsapp_message(f"whatsapp:{from_number}", "Sorry, our service is temporarily unavailable. Please try again later.")
            return {"status": "error", "message": "Supabase client not initialized"}

        # Step 1: Find or Create the user
        user_res = supabase.table("users").upsert({"phone_number": from_number}, on_conflict="phone_number").execute()
        if not user_res.data:
            logger.error(f"Could not find or create user for phone number: {from_number}")
            raise HTTPException(status_code=500, detail="User lookup failed")
        user = user_res.data[0]

        # Step 2: Check for a pending order for this user
        pending_order_res = supabase.table("orders").select("*").eq("user_id", user['id']).eq("status", "pending").order("created_at", desc=True).limit(1).execute()
        
        if pending_order_res.data:
            response_message = await handle_pending_order(pending_order_res.data[0], body, from_number)
        else:
            response_message = await handle_new_conversation(user, body, from_number)
    
        await send_whatsapp_message(f"whatsapp:{from_number}", response_message)
        
        return {"status": "success", "message": "Webhook processed successfully"}
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
        # Avoid sending technical errors to the user
        await send_whatsapp_message(f"whatsapp:{from_number}", "Sorry, an unexpected error occurred. Please try again in a moment.")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/confirm-items")
async def confirm_items(request: OrderRequest, api_key: str = Depends(verify_api_key)):
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database connection not available")

        items_dict = [item.dict() for item in request.items]

        order_data = {
            "user_id": request.user_id,
            "items_json": items_dict,
            "total_amount": request.total_amount,
            "status": "pending",
            "payment_status": "unpaid",
        }
        
        result = supabase.table("orders").insert(order_data).execute()

        order_id = result.data[0]["id"] if result.data else None
        
        delivery_msg = (
            "Thank you for your order! Please choose your preferred option:\n\n"
            "1. Reply with *Delivery* (we'll ask for your location to calculate the fee).\n"
            "2. Reply with *Pickup* (no delivery fee)."
        )
        await send_whatsapp_message(request.phone_number, delivery_msg)
        
        return {"status": "order saved", "order_id": order_id}
        
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