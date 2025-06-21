from fastapi import APIRouter, HTTPException
import logging

try:
    from supabase_client import supabase
except ImportError:
    supabase = None

router = APIRouter(
    prefix="/public",
    tags=["public"]
)
logger = logging.getLogger(__name__)

@router.get("/products")
async def get_public_products():
    """
    A public endpoint to fetch all available (in-stock) products
    for the shareable menu.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")
    try:
        # Only fetch products that are marked as available_stock = true
        response = supabase.table("products").select("*").eq("available_stock", True).order("name").execute()
        return response.data
    except Exception as e:
        logger.error(f"Could not fetch public products: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch products.") 