from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from os import getenv
import logging
import os
import httpx

try:
    from supabase_client import supabase, SupabaseClient
except ImportError:
    supabase = None
    SupabaseClient = None

API_KEY = getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_KEY")

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

async def verify_jwt(token: str = Depends(oauth2_scheme)):
    """
    Verifies the JWT token from a user's session by calling the Supabase Auth REST API.
    """
    if not token:
        logger.warning("JWT verification failed: Token is missing.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Not authenticated"
        )
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        logger.error("JWT verification failed: Supabase URL or Key not configured on the server.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Authentication service is misconfigured"
        )
        
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {token}"
                }
            )
        
        if response.status_code == 200:
            user = response.json()
            # You can add role checks here in the future if needed
            # For example: if user.get('app_metadata', {}).get('role') != 'admin':
            #   raise HTTPException(status_code=403, detail="Not authorized")
            return user
        else:
            logger.warning(f"JWT verification failed with status {response.status_code}: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token or expired session."
            )
            
    except httpx.RequestError as e:
        logger.error(f"JWT verification failed due to a network error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to the authentication service.",
        )
    except Exception as e:
        logger.error(f"JWT verification failed with an unexpected exception: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        ) 

async def get_admin_user(user: dict = Depends(verify_jwt)):
    """
    Depends on verify_jwt, then checks if the user has admin privileges.
    Admin role is expected to be in the 'app_metadata' of the JWT.
    """
    app_metadata = user.get('app_metadata', {})
    
    # Check for 'admin' in a list of roles or if a single role field is 'admin'
    user_roles = app_metadata.get('roles', [])
    user_role = app_metadata.get('role', '')

    if 'admin' not in user_roles and user_role != 'admin':
        logger.warning(f"Admin access denied for user {user.get('id')}. Roles: {user_roles}, Role: '{user_role}'")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resource."
        )
    
    logger.info(f"Admin access granted for user {user.get('id')}.")
    return user 