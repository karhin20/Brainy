import os
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path

# Build the path to the .env file relative to this script
# This ensures it's found regardless of where the app is run from
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase configuration in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) 