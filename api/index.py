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

# Import the new admin router and security modules
from . import admin_router, security, auth_router, public_router
from .utils import send_whatsapp_message

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
 
# Include the routers
app.include_router(admin_router.router)
app.include_router(auth_router.router)
app.include_router(public_router.router)

TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://marketmenu.vercel.app")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent")
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
PAYSTACK_PAYMENT_URL = os.getenv("PAYSTACK_PAYMENT_URL", "https://api.paystack.co/transaction/initialize")
API_KEY = os.getenv("API_KEY")

# API Key security - THIS IS NOW HANDLED IN security.py
# api_key_header = APIKeyHeader(name="X-API-Key")

# async def verify_api_key(api_key: str = Security(api_key_header)):
#     if api_key != API_KEY:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid API key"
#         )
#     return api_key

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
        return {"intent": "greet", "response": "Hello! Welcome back to Fresh Market GH."}
    
    try:
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        
        # Build a more context-aware prompt
        prompt_context = f"The user's name is {user_context.get('name', 'a returning customer')}."
        if user_context.get('has_paid_order'):
            prompt_context += " They have a paid order that is currently being processed."

        prompt = (
            f"""
            You are 'Fresh Market GH Assistant', a friendly, helpful, and efficient virtual assistant for our WhatsApp-based grocery service in Ghana.

            Your primary goals are:
            1. To assist users with placing orders for fresh food items.
            2. To provide information about their existing or past orders.
            3. To guide users through the service.

            You will receive the user's message and potentially additional context ('{prompt_context}') about the current state (e.g., active order, user history) to help you understand the user's request accurately.

            Your task is to analyze the user's message and the provided context to determine the user's primary intent and extract any relevant information (specifically food items).

            You MUST respond in PURE JSON format containing ONLY the following three fields:
            -   `intent` (string): The user's primary goal.
            -   `products` (list of strings): A list of specific food items mentioned by the user, particularly for the 'buy' intent. Extract item names as clearly as possible (e.g., "red onions", "ripe plantain"). This list should be empty `[]` if no specific items are mentioned or the intent is not 'buy'.
            -   `response` (string): A short, friendly conversational acknowledgement of the user's request based on the determined intent. This is *not* the final full response, but a brief confirmation that you understood the request. Keep this brief and relevant.

            Here are the possible intents and their descriptions:

            -   **`buy`**: The user wants to purchase one or more food items. Extract all mentioned food items into the `products` list.
            -   **`check_status`**: The user is asking about the status or location of their current or last order (e.g., "Where is my order?", "Is my delivery coming soon?").
            -   **`show_order_details`**: The user wants to see the items or details of their current or last order (e.g., "What did I order?", "Show me my items", "Can I see my last order?").
            -   **`greet`**: The user is initiating contact with a simple greeting (e.g., "hello", "hi", "good morning").
            -   **`cancel_order`**: The user wants to stop or cancel their pending order (e.g., "Cancel my order", "I want to stop my purchase").
            -   **`help`**: The user is asking for instructions, guidance, or expressing confusion about how to use the service or place an order.
            -   **`ask`**: The user is asking a general question related to the service or products, not covered by other intents (e.g., "Do you sell kontomire?", "What are your opening hours?", "How much is a tuber of yam?").
            -   **`unknown`**: The user message is unclear, irrelevant, or does not match any of the defined intents.

            If the user's message is unclear or doesn't fit any defined intent, use the 'unknown' intent.

            Maintain a friendly, helpful, and approachable tone in the 'response' field.

            Example Scenarios:

            1.  Message: "hi i want to buy some yam and green pepper please"
                Output: ```json
                {{
                "intent": "buy",
                "products": ["yam", "green pepper"],
                "response": "Okay, I can help you with that!"
                }}
                ```
            2.  Message: "where is my delivery?"
                Output: ```json
                {{
                "intent": "check_status",
                "products": [],
                "response": "Let me check the status of your order."
                }}
                ```
            3.  Message: "what was in my last order?"
                Output: ```json
                {{
                "intent": "show_order_details",
                "products": [],
                "response": "Getting the details of your last order now."
                }}
                ```
            4.  Message: "Hello there"
                Output: ```json
                {{
                "intent": "greet",
                "products": [],
                "response": "Hello! Welcome to Fresh Market GH."
                }}
                ```
            5.  Message: "Can I cancel my order from this morning?"
                Output: ```json
                {{
                "intent": "cancel_order",
                "products": [],
                "response": "Okay, I can assist with cancelling your order."
                }}
                ```
            6.  Message: "How do I place an order?"
                Output: ```json
                {{
                "intent": "help",
                "products": [],
                "response": "I can explain how to place an order."
                }}
                ```
            7.  Message: "Do you have fresh fish today?"
                Output: ```json
                {{
                "intent": "ask",
                "products": [],
                "response": "Let me check our availability."
                }}
                ```
            8.  Message: "Tell me a joke"
                Output: ```json
                {{
                "intent": "unknown",
                "products": [],
                "response": "I'm sorry, I can only help with your grocery needs."
                }}
                ```

            User Message: {message}
            """
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

async def generate_paystack_payment_link(order_id: str, amount: float, user_phone: str) -> str:
    """
    Generate a Paystack payment link for the order.
    Raises an exception if the API call fails.
    """
    if not PAYSTACK_SECRET_KEY:
        logger.warning("PAYSTACK_SECRET_KEY not set, using mock payment link")
        return f"https://paystack.com/pay/mock-{order_id}"
    
    headers = {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json"
    }
    
    # Generate a placeholder email as Paystack requires one.
    placeholder_email = f"{user_phone.replace('+', '')}@market.bot"

    payload = {
        "email": placeholder_email,
        "amount": int(amount * 100),  # Paystack expects amount in kobo
        "currency": "GHS",
        "reference": order_id,
        "callback_url": f"{FRONTEND_URL}/payment-success?order_id={order_id}",
        "channels": ["card", "mobile_money"],
        "metadata": {"order_id": order_id, "phone": user_phone}
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(PAYSTACK_PAYMENT_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["data"]["authorization_url"]
    except httpx.HTTPStatusError as e:
        logger.error(f"Paystack API HTTPStatusError: {str(e)} - Response: {e.response.text}", exc_info=True)
        # Re-raise the exception to be handled by the calling function
        raise
    except Exception as e:
        logger.error(f"Paystack API general error: {str(e)}", exc_info=True)
        # Re-raise for the calling function to handle
        raise

async def handle_pending_order(order: Dict[str, Any], body: str, from_number: str, user: Dict[str, Any]) -> str:
    """Handles conversation flow when a user has a pending (unpaid) order."""
    status = order.get('status')

    if status == 'pending_confirmation':
        if body.lower() in ["1", "delivery"]:
            # Check for saved location
            if user.get("last_known_location"):
                supabase.table("orders").update({"status": "awaiting_location_confirmation"}).eq("id", order['id']).execute()
                return (
                    "I see you have a saved location with us. Would you like to use it for this delivery?\\n\\n"
                    "1. Yes, use saved location\\n"
                    "2. No, I'll provide a new one"
                )
            else:
                supabase.table("orders").update({"delivery_type": "delivery", "status": "awaiting_location"}).eq("id", order['id']).execute()
                return "Great! Please share your delivery location so we can calculate the delivery fee."
        elif body.lower() in ["2", "pickup"]:
            supabase.table("orders").update({"delivery_type": "pickup", "status": "pending_payment"}).eq("id", order['id']).execute()
            payment_link = await generate_paystack_payment_link(order['id'], order['total_amount'], from_number)
            return (
                f"Alright, your order is set for pickup.\\n\\n"
                f"Your total is *GHS {order['total_amount']:.2f}*.\\n\\n"
                f"Please complete your payment here to confirm your order:\\n{payment_link}"
            )
        else:
            return "Sorry, I didn't get that. Please choose '1' for Delivery or '2' for Pickup."
    
    elif status == 'awaiting_location_confirmation':
        if body == "1": # Yes, use saved location
            location_str = user.get("last_known_location")
            location_data = json.loads(location_str)
            latitude = float(location_data["latitude"])
            longitude = float(location_data["longitude"])

            delivery_fee = calculate_delivery_fee(latitude, longitude)
            total_with_delivery = order['total_amount'] + delivery_fee
            
            update_data = {
                "status": "pending_payment", 
                "delivery_type": "delivery",
                "delivery_fee": delivery_fee,
                "total_with_delivery": total_with_delivery,
                "delivery_location_lat": latitude,
                "delivery_location_lon": longitude,
            }
            supabase.table("orders").update(update_data).eq("id", order['id']).execute()
            
            payment_link = await generate_paystack_payment_link(order['id'], total_with_delivery, from_number)
            return (
                f"Using your saved location. The delivery fee is GHS {delivery_fee:.2f}.\\n\\n"
                f"Your new total is *GHS {total_with_delivery:.2f}*.\\n\\n"
                f"Please complete your payment here:\\n{payment_link}"
            )
        elif body == "2": # No, use new one
            supabase.table("orders").update({"status": "awaiting_location"}).eq("id", order['id']).execute()
            return "Okay, no problem. Please share your new delivery location."
        else:
            return "Sorry, I didn't understand that. Please reply with '1' to use your saved location or '2' to use a new one."

    elif status in ['pending_payment', 'awaiting_location']:
        if status == 'awaiting_location':
             return "I'm still waiting for you to share your delivery location. Please use the WhatsApp location sharing feature."
        else: # pending_payment
            # Remind the user to pay
            total = order.get('total_with_delivery') or order['total_amount']
            payment_link = await generate_paystack_payment_link(order['id'], total, from_number)
            return (
                f"Just a reminder, you have a pending order with us waiting for payment.\\n\\n"
                f"Total: *GHS {total:.2f}*\\n\\n"
                f"Pay here to confirm: {payment_link}\\n\\n"
                f"If you'd like to cancel this order, just reply 'cancel'."
            )
            
    elif body.lower() == 'cancel':
        supabase.table("orders").update({"status": "cancelled", "payment_status": "cancelled"}).eq("id", order['id']).execute()
        return "Your order has been cancelled. Please let me know if there's anything else I can help with."

    # Fallback for any other message while an order is pending
    total = order.get('total_with_delivery') or order['total_amount']
    payment_link = await generate_paystack_payment_link(order['id'], total, from_number)
    return (
        f"It looks like you have an order in progress waiting for payment.\\n\\n"
        f"Total Amount: *GHS {total:.2f}*.\\n\\n"
        f"You can complete your payment here:\\n{payment_link}\\n\\n"
        f"Or, reply 'cancel' to cancel this order."
    )

async def handle_new_conversation(user: Dict[str, Any], gemini_result: Dict[str, Any], from_number: str) -> str:
    """Handles interaction when a user has no pending orders."""
    user_id = user['id']

    # Context for Gemini
    user_context = {
        "name": user.get("name"),
        "has_paid_order": bool(supabase.table("orders").select("id").eq("user_id", user_id).in_("status", ["processing", "out-for-delivery"]).limit(1).execute().data)
    }

    intent_data = await call_gemini_intent_extraction(gemini_result, user_context)
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
            
    elif intent == "show_order_details":
        # Find the latest order that has been paid for
        latest_order_res = supabase.table("orders").select("items_json, status").eq("user_id", user_id).in_("status", ["processing", "out-for-delivery", "delivered"]).order("created_at", desc=True).limit(1).execute()
        
        if not latest_order_res.data:
            return "I couldn't find any recent completed orders for you. Would you like to start a new one?"
            
        latest_order = latest_order_res.data[0]
        items_json = latest_order.get("items_json", [])
        
        if not items_json:
            return "Your last order seems to have been empty. If you think this is an error, please contact support."

        product_ids = [item['product_id'] for item in items_json]
        
        # Fetch product names for the IDs in the order
        products_res = supabase.table("products").select("id, name").in_("id", product_ids).execute()
        if not products_res.data:
             return "I'm having trouble retrieving the item details for your last order. Please contact support."

        product_map = {p['id']: p['name'] for p in products_res.data}
        
        item_lines = []
        for item in items_json:
            product_name = product_map.get(item['product_id'], "Unknown Item")
            item_lines.append(f"- {item['quantity']} x {product_name}")
            
        items_list_str = "\n".join(item_lines)
        order_status = latest_order.get('status', 'unknown')

        return f"Here are the items for your most recent order (Status: *{order_status}*):\n\n{items_list_str}"

    elif intent == "greet":
        return f"Hello! Welcome back to Fresh Market GH. What can I help you with today?"

    else: # ask, help, error, etc.
        return intent_data.get("response", "I'm not sure how to help with that. Could you try rephrasing?")

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()
    from_number = form_data.get("From")
    incoming_msg = form_data.get("Body", "").strip()

    if from_number:
        # Format number for consistency
        from_number = from_number.replace("whatsapp:", "")

        try:
            # 1. Find or create user
            user_res = supabase.table("users").select("*").eq("phone_number", from_number).limit(1).execute()
            user = user_res.data[0] if user_res.data else None

            if not user:
                # This is a new user, create an entry
                insert_res = supabase.table("users").insert({"phone_number": from_number}).select("*").execute()
                user = insert_res.data[0]
            else:
                # Existing user, update last_active timestamp
                supabase.table("users").update({"last_active": datetime.now().isoformat()}).eq("id", user['id']).execute()

            user_id = user['id']

            # 2. Check for an existing unpaid order for this user
            order_res = supabase.table("orders").select("*").eq("user_id", user_id).eq("payment_status", "pending").order("created_at", desc=True).limit(1).execute()
            order = order_res.data[0] if order_res.data else None

            # --- Main Logic Branching ---

            # A. If the user sends a location (from WhatsApp)
            if "Latitude" in form_data and "Longitude" in form_data:
                if order and order['status'] == 'awaiting_location':
                    try:
                        latitude = float(form_data["Latitude"])
                        longitude = float(form_data["Longitude"])
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse location coordinates from webhook for user {from_number}. Data: {form_data}")
                        await send_whatsapp_message(from_number, "It seems there was an issue with the location you shared. Please try sharing your location again using the WhatsApp feature.")
                        return JSONResponse(content={})

                    # Save the location for future use
                    location_to_save = json.dumps({"latitude": str(latitude), "longitude": str(longitude)})
                    supabase.table("users").update({"last_known_location": location_to_save}).eq("id", user_id).execute()

                    # Calculate delivery fee
                    delivery_fee = calculate_delivery_fee(latitude, longitude)
                    total_with_delivery = order['total_amount'] + delivery_fee
                    
                    # Update order with delivery details
                    update_data = {
                        "status": "pending_payment",
                        "delivery_fee": delivery_fee,
                        "total_with_delivery": total_with_delivery,
                        "delivery_location_lat": latitude,
                        "delivery_location_lon": longitude,
                    }
                    supabase.table("orders").update(update_data).eq("id", order['id']).execute()
                    
                    # Generate payment link and send it
                    payment_link = await generate_paystack_payment_link(order['id'], total_with_delivery, from_number)
                    msg = (
                        f"Great, location received! Your delivery fee is GHS {delivery_fee:.2f}.\\n\\n"
                        f"Your total amount is now *GHS {total_with_delivery:.2f}*.\\n\\n"
                        f"Please complete your payment here to confirm your order:\\n{payment_link}"
                    )
                    await send_whatsapp_message(from_number, msg)
                else:
                    # User sent a location when we weren't expecting one
                    await send_whatsapp_message(from_number, "Thanks for sharing your location, but I wasn't expecting it right now. How can I help you today?")
                return JSONResponse(content={})

            # B. If there's a pending order, guide the user
            if order:
                # This function handles all interactions while an order is active but unpaid
                reply_message = await handle_pending_order(order, incoming_msg, from_number, user)
                await send_whatsapp_message(from_number, reply_message)
            else:
                # C. No pending order, start a new conversation
                user_context = {'has_paid_order': False} # This can be enhanced later
                gemini_result = await call_gemini_intent_extraction(incoming_msg, user_context)
                
                reply_message = await handle_new_conversation(user, gemini_result, from_number)
                await send_whatsapp_message(from_number, reply_message)
        
        except Exception as e:
            logger.error(f"Error in whatsapp_webhook: {e}", exc_info=True)
            # Send a generic error message to the user
            await send_whatsapp_message(from_number, "Oh, something went wrong on my end. Please try again in a moment.")

    return JSONResponse(content={})

@app.post("/confirm-items")
async def confirm_items(request: OrderRequest, api_key: str = Depends(security.verify_api_key)):
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database connection not available")
            
        # CRITICAL FIX: Invalidate any other pending orders for this user to prevent "zombie" orders.
        # This ensures a user can only have one active pending order at a time.
        logger.info(f"Cancelling any existing pending orders for user_id: {request.user_id}")
        supabase.table("orders").update({
            "status": "cancelled",
            # Consider adding a 'notes' column to your DB for internal tracking.
            # "notes": "Superseded by a new order." 
        }).eq("user_id", request.user_id).eq("status", "pending").execute()


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
            "Thank you for your order! Please choose your preferred option by replying with the number:\n\n"
            "*1* for Delivery (we'll ask for your location to calculate the fee).\n"
            "*2* for Pickup (no delivery fee)."
        )
        await send_whatsapp_message(request.phone_number, delivery_msg)
        
        return {"status": "order saved", "order_id": order_id}
        
    except Exception as e:
        logger.error(f"Confirm items error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/payment-success")
async def payment_success(request: Request, api_key: str = Depends(security.verify_api_key)):
    try:
        data = await request.json()
        order_id = data.get("order_id")
        
        if not order_id:
            raise HTTPException(status_code=400, detail="Missing order_id")
            
        if not supabase:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Update order status
        update_res = supabase.table("orders").update({
            "payment_status": "paid",
            "status": "processing", # Set status to processing upon successful payment
            "payment_confirmed_at": datetime.now().isoformat()
        }).eq("id", order_id).execute()
        
        if not update_res.data:
            logger.warning(f"Payment success hook: No order found for order_id {order_id} to update.")
            # Even if the order isn't found, we should return a success to Paystack to prevent retries.
            return {"status": "success", "message": "Order not found, but acknowledged."}


        # Get user phone number for notification
        order = update_res.data[0]
        user_query = supabase.table("users").select("phone_number").eq("id", order["user_id"]).execute()
        
        if user_query.data:
            phone_number = user_query.data[0]["phone_number"]
            notification_message = (
                f"✅ Payment confirmed for your order!\n\n"
                f"Your Order ID is: {order_id}.\n"
                "We are now preparing your items for delivery. We'll let you know once it's on its way."
            )
            await send_whatsapp_message(phone_number, notification_message)
        else:
            logger.error(f"Could not find user to notify for paid order {order_id}")
        
        return {"status": "success", "message": "Payment confirmed and user notified"}
        
    except Exception as e:
        logger.error(f"Payment success webhook error: {str(e)}", exc_info=True)
        # It's crucial to return a success-like status to Paystack to prevent repeated webhook calls.
        return JSONResponse(
            status_code=200,
            content={"status": "error", "message": "Webhook processed with an internal error."}
        )
    


def calculate_delivery_fee(lat: float, lon: float) -> float:
    """
    Calculates the delivery fee based on the distance from a central point.
    This is a simplified example. A real implementation might use a mapping API.
    Our central point is somewhere in Accra, Ghana.
    """
    # Coordinates for a central point in Accra (e.g., Kwame Nkrumah Interchange)
    central_lat, central_lon = 5.5560, -0.2057

    # Simple distance calculation (not highly accurate, but good for this use case)
    distance = ((lat - central_lat)**2 + (lon - central_lon)**2)**0.5

    # Define fee based on distance tiers
    if distance < 0.1:  # Approx < 11km
        return 15.00
    elif distance < 0.2: # Approx < 22km
        return 25.00
    else:
        return 40.00

# Export the app for Vercel
app.debug = False 