from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

try:
    from supabase_client import supabase, SupabaseClient
except ImportError:
    supabase = None
    SupabaseClient = None

router = APIRouter(
    prefix="/auth",
    tags=["authentication"]
)
logger = logging.getLogger(__name__)

class AuthRequest(BaseModel):
    email: str
    password: str

@router.post("/signup")
async def signup(request: AuthRequest, db: SupabaseClient = Depends(lambda: supabase)):
    """
    Signs up a new admin user. In a real-world scenario, you might want to restrict
    this endpoint or have an invite-only system.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database connection not available")
    try:
        session = db.auth.sign_up({
            "email": request.email,
            "password": request.password,
        })
        return {"message": "Signup successful! Please check your email to verify.", "session": session}
    except Exception as e:
        logger.error(f"Error during signup: {e}", exc_info=True)
        # Check for specific Supabase errors if possible, e.g., user already exists
        if "User already registered" in str(e):
            raise HTTPException(status_code=400, detail="User with this email already exists.")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login")
async def login(request: AuthRequest, db: SupabaseClient = Depends(lambda: supabase)):
    """
    Logs in an admin user and returns a session object containing the JWT access token.
    """
    if not db:
        raise HTTPException(status_code=500, detail="Database connection not available")
    try:
        session = db.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        return session
    except Exception as e:
        logger.error(f"Error during login: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid login credentials.") 