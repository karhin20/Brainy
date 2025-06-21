from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from os import getenv
import logging

try:
    from supabase_client import supabase, SupabaseClient
except ImportError:
    supabase = None
    SupabaseClient = None

API_KEY = getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

logger = logging.getLogger(__name__)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    For server-to-server communication or webhooks where a user is not logged in.
    (e.g., Paystack webhook).
    """
    if not api_key or api_key != API_KEY:
        logger.warning("API key verification failed. The key was either missing or invalid.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key

async def verify_jwt(token: str = Depends(oauth2_scheme), db: SupabaseClient = Depends(lambda: supabase)):
    """
    Verifies the JWT token from a user's session to protect admin endpoints.
    """
    if not token:
        logger.warning("JWT verification failed: Token is missing.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Not authenticated"
        )
    if not db:
        logger.error("JWT verification failed: Database connection not available.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Authentication service is down"
        )
        
    try:
        user_response = db.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token or expired session.")
        
        # You can add role checks here in the future if needed
        # For example: if user.app_metadata.get('role') != 'admin':
        #   raise HTTPException(status_code=403, detail="Not authorized")
            
        return user
    except Exception as e:
        logger.error(f"JWT verification failed with an exception: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        ) 