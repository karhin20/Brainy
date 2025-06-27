import os
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Build the path to the .env file relative to this script
# This ensures it's found regardless of where the app is run from
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

logger.info(f"Supabase Client Init: Attempting to load .env from {dotenv_path}")
logger.info(f"Supabase Client Init: SUPABASE_URL found? {'Yes' if SUPABASE_URL else 'No'}")
logger.info(f"Supabase Client Init: SUPABASE_KEY found? {'Yes' if SUPABASE_KEY else 'No'} (Value hidden for security)")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase Client Init: Missing Supabase configuration in environment variables. Raising ValueError.")
    raise ValueError("Missing Supabase configuration in environment variables.")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase Client Init: Supabase client created successfully.")
except Exception as e:
    logger.critical(f"Supabase Client Init: Failed to create Supabase client: {e}", exc_info=True)
    raise # Re-raise to ensure the app doesn't proceed with a broken client 