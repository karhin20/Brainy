from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Optional
import logging
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
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
    order_number: str | None = None
    delivery_location_lat: float | None = None
    delivery_location_lon: float | None = None


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
        response = supabase.table("orders").select("*, user:users(phone_number), delivery_location_lat, delivery_location_lon").order("created_at", desc=True).execute()

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

# NEW/MODIFIED Pydantic models for analytics data
class DashboardStats(BaseModel):
    total_revenue: float
    orders_in_period: int # Renamed from orders_today to be more general
    total_customers: int
    average_order_value: float # NEW
    new_customers_in_period: int # NEW

class SalesDataPoint(BaseModel):
    date: str
    sales: float

class TopProductDataPoint(BaseModel):
    name: str
    count: int

class CategorySalesDataPoint(BaseModel): # NEW
    category: str
    revenue: float

class OrderStatusDistributionDataPoint(BaseModel): # NEW
    status: str
    count: int

class ChartData(BaseModel):
    sales_over_time: list[SalesDataPoint]
    top_products: list[TopProductDataPoint]
    revenue_by_category: list[CategorySalesDataPoint] # NEW
    order_status_distribution: list[OrderStatusDistributionDataPoint] # NEW

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
async def get_dashboard_charts_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Get aggregated data for dashboard charts, with optional date range filtering.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    # Parse dates or set defaults
    now_utc = datetime.now(timezone.utc)
    parsed_start_date: datetime
    parsed_end_date: datetime

    if start_date:
        try:
            parsed_start_date = datetime.fromisoformat(start_date).astimezone(timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO 8601.")
    else:
        # Default to 30 days ago if no start_date is provided
        parsed_start_date = now_utc - timedelta(days=30)
    
    if end_date:
        try:
            parsed_end_date = datetime.fromisoformat(end_date).astimezone(timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO 8601.")
    else:
        # Default to now if no end_date is provided
        parsed_end_date = now_utc

    if parsed_start_date >= parsed_end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date.")


    try:
        # --- 1. Sales Over Time ---
        # Filter paid orders by the specified date range
        paid_orders_query_base = supabase.table("orders").select("created_at, total_with_delivery, total_amount, items_json, status").eq("payment_status", "paid") # Added items_json, status for new metrics
        paid_orders_query_base = paid_orders_query_base.gte("created_at", parsed_start_date.isoformat())
        paid_orders_query_base = paid_orders_query_base.lte("created_at", parsed_end_date.isoformat())
        
        all_paid_orders_res_for_charts = paid_orders_query_base.execute() # Fetch once for all chart data

        sales_by_day = defaultdict(float)
        
        # --- For Category Sales and Order Status Distribution ---
        revenue_by_category_raw = defaultdict(float)
        order_status_counts = defaultdict(int)
        
        all_product_ids_in_orders = set()

        if all_paid_orders_res_for_charts.data:
            for order in all_paid_orders_res_for_charts.data:
                total_order_value = order.get('total_with_delivery') or order.get('total_amount') or 0.0
                order_date_str = datetime.fromisoformat(order['created_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d')
                sales_by_day[order_date_str] += total_order_value

                # Collect product IDs for category lookup and calculate category revenue
                items = order.get("items_json") or []
                for item in items:
                    # Handle items_json potentially being a string (legacy data)
                    if isinstance(item, str):
                        try:
                            item = json.loads(item)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse items_json string for order {order.get('id')}: {item}")
                            continue # Skip this item if it's not valid JSON
                    
                    if isinstance(item, dict): # Ensure it's a dictionary after parsing
                        product_id = item.get("product_id")
                        quantity = item.get("quantity")
                        if product_id and quantity:
                            all_product_ids_in_orders.add(product_id)
                            # Actual category revenue calculation will happen after fetching product categories

                # Count order statuses
                status = order.get('status')
                if status:
                    order_status_counts[status] += 1

        sales_over_time = [SalesDataPoint(date=day, sales=round(sales, 2)) for day, sales in sorted(sales_by_day.items())]

        # --- 2. Top Selling Products & Revenue by Category ---
        product_counts = defaultdict(int)
        
        # Fetch product details (name, category, price) for all products involved in paid orders
        product_details_map = {} # Map product_id to {name, category, price}
        if all_product_ids_in_orders:
            products_res = supabase.table("products").select("id, name, category, price").in_("id", list(all_product_ids_in_orders)).execute()
            if products_res.data:
                for p in products_res.data:
                    product_details_map[p['id']] = p

            # Now iterate orders again to calculate product counts and category revenue
            if all_paid_orders_res_for_charts.data:
                for order in all_paid_orders_res_for_charts.data:
                    items = order.get("items_json") or []
                    for item in items:
                        if isinstance(item, str):
                            try:
                                item = json.loads(item)
                            except json.JSONDecodeError:
                                continue
                        
                        if isinstance(item, dict):
                            product_id = item.get("product_id")
                            quantity = item.get("quantity")
                            
                            if product_id and quantity and product_id in product_details_map:
                                # For Top Selling Products
                                product_counts[product_id] += quantity

                                # For Revenue by Category
                                product_info = product_details_map[product_id]
                                category = product_info.get('category')
                                price = product_info.get('price')
                                if category and price is not None:
                                    revenue_by_category_raw[category] += price * quantity
        
        # Format Top Selling Products
        top_products_data = []
        for pid, count in product_counts.items():
            if pid in product_details_map:
                top_products_data.append({
                    "name": product_details_map[pid].get('name', "Unknown Product"),
                    "count": count
                })
        
        top_products = sorted(top_products_data, key=lambda x: x['count'], reverse=True)[:5]
        top_products_formatted = [TopProductDataPoint(name=p['name'], count=p['count']) for p in top_products]

        # Format Revenue by Category
        revenue_by_category_formatted = [
            CategorySalesDataPoint(category=cat, revenue=round(rev, 2)) 
            for cat, rev in sorted(revenue_by_category_raw.items(), key=lambda item: item[1], reverse=True)
        ]

        # Format Order Status Distribution
        order_status_distribution_formatted = [
            OrderStatusDistributionDataPoint(status=stat.replace('_', ' ').title(), count=count) # Format status string
            for stat, count in sorted(order_status_counts.items())
        ]

        return ChartData(
            sales_over_time=sales_over_time,
            top_products=top_products_formatted,
            revenue_by_category=revenue_by_category_formatted, # NEW
            order_status_distribution=order_status_distribution_formatted # NEW
        )

    except Exception as e:
        logger.error(f"Could not fetch dashboard chart data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch chart data.")

@router.get("/dashboard-stats", response_model=DashboardStats)
async def get_dashboard_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Retrieve aggregated statistics for the main dashboard, with optional date range filtering.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    # Parse dates or set defaults for stats
    now_utc = datetime.now(timezone.utc)
    parsed_start_date: datetime
    parsed_end_date: datetime

    if start_date:
        try:
            parsed_start_date = datetime.fromisoformat(start_date).astimezone(timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO 8601.")
    else:
        # Default to 30 days ago if no start_date is provided for stats
        parsed_start_date = now_utc - timedelta(days=30)
    
    if end_date:
        try:
            parsed_end_date = datetime.fromisoformat(end_date).astimezone(timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO 8601.")
    else:
        # Default to now if no end_date is provided for stats
        parsed_end_date = now_utc

    if parsed_start_date >= parsed_end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date.")

    try:
        # 1. Calculate Total Revenue from paid orders within the date range
        paid_orders_query = supabase.table("orders").select("total_with_delivery").eq("payment_status", "paid")
        paid_orders_query = paid_orders_query.gte("created_at", parsed_start_date.isoformat())
        paid_orders_query = paid_orders_query.lte("created_at", parsed_end_date.isoformat())
        
        paid_orders_res = paid_orders_query.execute()
        
        total_revenue = sum(order['total_with_delivery'] for order in paid_orders_res.data if order['total_with_delivery'] is not None)
        
        # 2. Get Orders in selected period
        orders_in_period_query = supabase.table("orders").select("id", count="exact")
        orders_in_period_query = orders_in_period_query.gte("created_at", parsed_start_date.isoformat())
        orders_in_period_query = orders_in_period_query.lte("created_at", parsed_end_date.isoformat())
        orders_in_period_res = orders_in_period_query.execute()
        orders_in_period = orders_in_period_res.count

        # Calculate Average Order Value (NEW KPI)
        average_order_value = total_revenue / orders_in_period if orders_in_period > 0 else 0.0

        # 3. Get Total Customers (overall, not date filtered)
        customers_res = supabase.table("users").select("id", count="exact").execute()
        total_customers = customers_res.count
        
        # 4. Get New Customers in selected period (NEW KPI)
        new_customers_query = supabase.table("users").select("id", count="exact")
        new_customers_query = new_customers_query.gte("created_at", parsed_start_date.isoformat())
        new_customers_query = new_customers_query.lte("created_at", parsed_end_date.isoformat())
        new_customers_res = new_customers_query.execute()
        new_customers_in_period = new_customers_res.count


        return {
            "total_revenue": total_revenue,
            "orders_in_period": orders_in_period, # Renamed from orders_today
            "total_customers": total_customers,
            "average_order_value": round(average_order_value, 2), # NEW
            "new_customers_in_period": new_customers_in_period # NEW
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

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

@router.post("/change-password")
async def change_password(request: PasswordChangeRequest, current_user: dict = Depends(verify_jwt)):
    """
    Allows the authenticated admin user to change their password.
    NOTE: Supabase does not directly support changing password with current_password
    via the client library's update_user. It primarily uses an email OTP flow
    or relies on server-side auth (like a custom JWT check).
    For a secure implementation, you'd typically:
    1. Verify the current password on the backend if not using Supabase's auth service for this.
       (This is difficult with Supabase client-side auth tokens without exposing sensitive info).
    2. Use a password reset flow (send OTP to email) or direct admin DB access if allowed/secure.

    For this project's scope, we'll demonstrate a simplified (less secure) direct update.
    A more secure approach would involve:
    - Supabase's password reset flow (sending an email with a reset link).
    - If direct password verification is absolutely needed on backend, it's complex and risky.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")

    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in token.")

    try:
        # IMPORTANT: Supabase's auth.update_user_by_id does NOT verify the current_password.
        # It directly updates. This means anyone with a valid admin JWT can change the password
        # without knowing the old one, IF ONLY THIS METHOD IS USED.
        # In a production app, for password changes, you'd typically implement:
        # 1. A password reset email flow (Supabase Auth built-in).
        # 2. Or, if current password verification is a MUST for *this endpoint*,
        #    you'd need to create a custom login attempt to verify current_password first,
        #    which is not ideal or easily done without exposing user credentials.

        # This call will update the password for the user_id extracted from the JWT.
        # It DOES NOT verify `request.current_password`.
        response = supabase.auth.admin.update_user_by_id(
            user_id,
            attributes={"password": request.new_password}
        )
        
        if response.user: # Check if the user object is returned, indicating success
            logger.info(f"Password changed for admin user: {user_id}")
            return {"message": "Password updated successfully."}
        else:
            logger.error(f"Supabase update_user_by_id failed for user {user_id}: {response.json()}")
            raise HTTPException(status_code=500, detail="Failed to update password via Supabase.")

    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Error changing password for user {user_id}: {e}", exc_info=True)
        # Attempt to provide more specific error messages if possible
        if "Invalid email or password" in str(e): # Example of catching specific errors
            raise HTTPException(status_code=400, detail="Invalid current password.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while changing password.")

# Pydantic model for Admin profile response
class AdminProfileResponse(BaseModel):
    id: str
    phone_number: str | None = None  # Make phone_number optional
    email: str | None = None
    admin_profile: Dict[str, str] | None = None

@router.get("/me", response_model=AdminProfileResponse)
async def get_current_admin_user(current_user: dict = Depends(security.get_admin_user)):
    """
    Retrieve details of the currently authenticated admin user, including their profile.
    The `security.get_admin_user` dependency already fetches and attaches the admin_profile.
    """
    # Ensure items_json is a list, not a string, for AdminOrder response model
    if 'items_json' in current_user and isinstance(current_user['items_json'], str):
        try:
            current_user['items_json'] = json.loads(current_user['items_json'])
        except Exception as e:
            logger.error(f"Failed to parse items_json for admin user {current_user.get('id')}: {e}")
            current_user['items_json'] = [] # Fallback to empty list if parsing fails

    # Return the current_user object which now contains the admin_profile
    return current_user 