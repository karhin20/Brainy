# WhatsApp MarketBot Backend (FastAPI)

This is the backend for the WhatsApp MarketBot for Food Items in Ghana. It handles WhatsApp messaging, item selection, payment, and delivery status updates.

## Features
- WhatsApp webhook integration (Twilio)
- Item selection confirmation
- Paystack payment webhook
- Delivery status updates
- Supabase (PostgreSQL) integration

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   - Copy `.env.example` to `.env` and fill in your secrets.
4. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

## Project Structure
- `main.py` - FastAPI app and endpoints
- `supabase_client.py` - Supabase client setup
- `.env.example` - Example environment variables
- `requirements.txt` - Python dependencies

## Endpoints
- `POST /whatsapp-webhook` - Twilio WhatsApp webhook
- `POST /confirm-items` - Confirm item selection
- `POST /payment-success` - Paystack webhook
- `POST /delivery-status` - Update delivery status

---

For more details, see the main project plan. 