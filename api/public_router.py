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

@router.get("/public/products", tags=["Public"])
async def get_public_products():
    """
    Public endpoint to fetch all available products.
    No authentication required.
    """
    try:
        if not supabase:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        # Correctly filter for items where stock is greater than 0
        response = supabase.table("products").select("*").gt("available_stock", 0).order("name").execute()
        
        if not response.data:
            logger.warning("No public products found or database query failed.")
            return []

        return response.data
    except Exception as e:
        logger.error(f"Could not fetch public products: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch products.") 