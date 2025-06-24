from fastapi import APIRouter, Depends, HTTPException
from typing import List
import logging
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Import from our new security module and the existing supabase_client
# The '..' indicates the parent directory where supabase_client.py is located
from . import security
from .utils import send_whatsapp_message
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from supabase_client import supabase
except ImportError:
    supabase = None

from .security import verify_jwt

router = APIRouter(
    tags=["Admin"],
    dependencies=[Depends(verify_jwt)]
)

logger = logging.getLogger(__name__)

# Pydantic models for admin responses
class AdminOrderItem(BaseModel):
    product_id: str
    quantity: int

class UserInfo(BaseModel):
    phone_number: str

class AdminOrder(BaseModel):
    id: str
    created_at: datetime
    status: str
    payment_status: str
    total_amount: float
    delivery_type: str | None = None
    delivery_fee: float | None = None
    total_with_delivery: float | None = None
    items_json: List[AdminOrderItem]
    user: UserInfo | None = None


@router.get("/orders", response_model=List[AdminOrder])
async def get_all_orders():
    """
    Retrieve all orders from the database for the admin dashboard.
    Joins with the users table to get the phone number.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    try:
        # Foreign table joins in Supabase are done with `select=...`
        # The query gets all orders and for each order, gets the phone_number from the related 'users' table
        response = supabase.table("orders").select("*, user:users(phone_number)").order("created_at", desc=True).execute()

        if response.data:
            # The 'user' field in the response is an object, not a list
            # We need to handle the case where it might be None if the join fails
            for order in response.data:
                if order.get('user') and isinstance(order['user'], list):
                    # This shouldn't happen based on a to-one relationship, but good to be safe
                    order['user'] = order['user'][0] if order['user'] else None
                # Ensure items_json is a list, not a string
                if 'items_json' in order and isinstance(order['items_json'], str):
                    try:
                        order['items_json'] = json.loads(order['items_json'])
                    except Exception as e:
                        logger.error(f"Failed to parse items_json for order {order.get('id')}: {e}")
                        order['items_json'] = []
            return response.data
        return []
    except Exception as e:
        logger.error(f"Could not fetch orders for admin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch orders.")


# Pydantic model for Product data
class Product(BaseModel):
    id: str | None = None
    name: str
    description: str | None = None
    price: float
    category: str | None = None
    image_url: str | None = None
    available_stock: bool

@router.get("/products", response_model=List[Product])
async def get_all_products():
    """ Retrieve all products from the database. """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    try:
        response = supabase.table("products").select("*").order("name").execute()
        return response.data
    except Exception as e:
        logger.error(f"Could not fetch products for admin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch products.")

@router.post("/products", response_model=Product)
async def create_product(product: Product):
    """ Create a new product. """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")
    
    try:
        # Exclude 'id' from the dict as it's auto-generated
        product_data = product.dict(exclude={"id"})
        response = supabase.table("products").insert(product_data).execute()
        return response.data[0]
    except Exception as e:
        logger.error(f"Could not create product: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create product.")

@router.put("/products/{product_id}", response_model=Product)
async def update_product(product_id: str, product: Product):
    """ Update an existing product. """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")
    
    try:
        product_data = product.dict(exclude_unset=True)
        response = supabase.table("products").update(product_data).eq("id", product_id).execute()
        if response.data:
            return response.data[0]
        raise HTTPException(status_code=404, detail="Product not found")
    except Exception as e:
        logger.error(f"Could not update product {product_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update product.")

@router.delete("/products/{product_id}", status_code=204)
async def delete_product(product_id: str):
    """ Delete a product. """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")
    
    try:
        response = supabase.table("products").delete().eq("id", product_id).execute()
        if not response.data:
            # Even if the item does not exist, it's a successful deletion from a client's perspective
            logger.warning(f"Attempted to delete non-existent product with ID: {product_id}")
        return
    except Exception as e:
        logger.error(f"Could not delete product {product_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete product.")

# Pydantic model for User data
class AdminUser(BaseModel):
    id: str
    created_at: datetime
    phone_number: str
    last_active: datetime | None = None
    is_blocked: bool | None = None

class DashboardStats(BaseModel):
    total_revenue: float
    orders_today: int
    total_customers: int

class SalesDataPoint(BaseModel):
    date: str
    sales: float

class TopProductDataPoint(BaseModel):
    name: str
    count: int

class ChartData(BaseModel):
    sales_over_time: list[SalesDataPoint]
    top_products: list[TopProductDataPoint]

class UserBlockStatus(BaseModel):
    is_blocked: bool

@router.get("/users", response_model=List[AdminUser])
async def get_all_users():
    """ Retrieve all users from the database. """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    try:
        response = supabase.table("users").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        logger.error(f"Could not fetch users for admin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch users.")

@router.patch("/users/{user_id}/status", response_model=AdminUser)
async def update_user_status(user_id: str, block_status: UserBlockStatus):
    """
    Update the block status of a user.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    try:
        # 1. Update the user
        supabase.table("users").update({
            "is_blocked": block_status.is_blocked,
            "updated_at": datetime.now().isoformat()
        }).eq("id", user_id).execute()

        # 2. Fetch the updated user
        fetch_res = supabase.table("users").select("*").eq("id", user_id).single().execute()
        if fetch_res.data:
            return fetch_res.data
        raise HTTPException(status_code=404, detail="User not found")
        
    except Exception as e:
        logger.error(f"Could not update block status for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update user block status.")

@router.get("/dashboard-charts-data", response_model=ChartData)
async def get_dashboard_charts_data():
    """
    Get aggregated data for dashboard charts.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    try:
        # --- 1. Sales Over Time (Last 30 days) ---
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        paid_orders_res = supabase.table("orders").select("created_at, total_with_delivery, total_amount").eq("payment_status", "paid").gte("created_at", thirty_days_ago.isoformat()).execute()
        
        sales_by_day = defaultdict(float)
        if paid_orders_res.data:
            for order in paid_orders_res.data:
                total = order.get('total_with_delivery') or order.get('total_amount') or 0.0
                day = datetime.fromisoformat(order['created_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d')
                sales_by_day[day] += total
        
        sales_over_time = [SalesDataPoint(date=day, sales=round(sales, 2)) for day, sales in sorted(sales_by_day.items())]

        # --- 2. Top Selling Products ---
        all_paid_orders_res = supabase.table("orders").select("items_json").eq("payment_status", "paid").execute()
        
        product_counts = defaultdict(int)
        product_ids = set()

        if all_paid_orders_res.data:
            for order in all_paid_orders_res.data:
                items = order.get("items_json") or []
                for item in items:
                    if isinstance(item, str):
                        try:
                            item = json.loads(item)
                        except Exception:
                            continue
                    if isinstance(item, dict):
                        product_id = item.get("product_id")
                        quantity = item.get("quantity")
                        if product_id and quantity:
                            product_counts[product_id] += quantity
                            product_ids.add(product_id)
        
        top_products_data = []
        if product_ids:
            products_res = supabase.table("products").select("id, name").in_("id", list(product_ids)).execute()
            product_map = {p['id']: p['name'] for p in products_res.data} if products_res.data else {}

            for pid, count in product_counts.items():
                top_products_data.append({
                    "name": product_map.get(pid, "Unknown Product"),
                    "count": count
                })
        
        top_products = sorted(top_products_data, key=lambda x: x['count'], reverse=True)[:5]
        top_products_formatted = [TopProductDataPoint(name=p['name'], count=p['count']) for p in top_products]

        return ChartData(sales_over_time=sales_over_time, top_products=top_products_formatted)

    except Exception as e:
        logger.error(f"Could not fetch dashboard chart data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch chart data.")

@router.get("/dashboard-stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """
    Retrieve aggregated statistics for the main dashboard.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    try:
        # 1. Calculate Total Revenue from paid orders
        paid_orders_res = supabase.table("orders").select("total_with_delivery").eq("payment_status", "paid").execute()
        total_revenue = sum(order['total_with_delivery'] for order in paid_orders_res.data if order['total_with_delivery'] is not None)

        # 2. Get Orders Today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        orders_today_res = supabase.table("orders").select("id", count="exact").gte("created_at", today_start.isoformat()).execute()
        orders_today = orders_today_res.count

        # 3. Get Total Customers
        customers_res = supabase.table("users").select("id", count="exact").execute()
        total_customers = customers_res.count

        return {
            "total_revenue": total_revenue,
            "orders_today": orders_today,
            "total_customers": total_customers
        }
    except Exception as e:
        logger.error(f"Could not fetch dashboard stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard stats.")

class OrderStatusUpdate(BaseModel):
    status: str

@router.patch("/orders/{order_id}/status", response_model=AdminOrder)
async def update_order_status(order_id: str, status_update: OrderStatusUpdate):
    """
    Update the status of an order and notify the customer.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    try:
        # 1. Update the order
        supabase.table("orders").update({
            "status": status_update.status,
            "updated_at": datetime.now().isoformat()
        }).eq("id", order_id).execute()

        # 2. Fetch the updated order with join
        fetch_res = supabase.table("orders").select("*, user:users(phone_number)").eq("id", order_id).single().execute()
        updated_order = fetch_res.data
        
        # Ensure items_json is a list, not a string, for the response model
        if updated_order and 'items_json' in updated_order and isinstance(updated_order['items_json'], str):
            try:
                updated_order['items_json'] = json.loads(updated_order['items_json'])
            except Exception as e:
                logger.error(f"Failed to parse items_json for updated order {updated_order.get('id')}: {e}")
                updated_order['items_json'] = [] # Fallback to empty list if parsing fails

        # Notify the customer via WhatsApp
        if updated_order and updated_order.get("user"):
            phone_number = updated_order["user"]["phone_number"]
            # Use order_number if available, otherwise fallback to truncated ID
            display_order_id = updated_order.get('order_number') or f"{order_id[:8]}..."
            message = f"Good news! Your order ({display_order_id}) is now *{status_update.status}*."
            await send_whatsapp_message(phone_number, message)
        
        return updated_order
    except Exception as e:
        logger.error(f"Could not update order status for {order_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update order status.") 