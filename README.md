# WhatsApp MarketBot Backend (FastAPI)

This is the backend for the WhatsApp MarketBot for Food Items in Ghana. It handles WhatsApp messaging, item selection, payment, and delivery status updates.

## Features
- WhatsApp webhook integration (Twilio)
- Item selection confirmation
- Paystack payment webhook
- Delivery status updates
- Supabase (PostgreSQL) integration
- Vercel deployment ready

## Local Setup

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

## Vercel Deployment

### Prerequisites
- Vercel account
- All environment variables configured

### Deployment Steps

1. **Install Vercel CLI** (optional)
   ```bash
   npm i -g vercel
   ```

2. **Deploy to Vercel**
   ```bash
   vercel
   ```
   
   Or connect your GitHub repository to Vercel for automatic deployments.

3. **Configure Environment Variables**
   - Go to your Vercel project dashboard
   - Navigate to Settings > Environment Variables
   - Add all variables from `.env.example`

4. **Set up Custom Domain** (optional)
   - In Vercel dashboard, go to Settings > Domains
   - Add your custom domain

### Environment Variables for Vercel
Make sure to set these in your Vercel project:
- `TWILIO_AUTH_TOKEN`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_WHATSAPP_NUMBER`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `GEMINI_API_KEY`
- `PAYSTACK_SECRET_KEY`
- `FRONTEND_URL`
- `API_KEY`

## Project Structure
- `api/index.py` - FastAPI app for Vercel deployment
- `main.py` - FastAPI app for local development
- `supabase_client.py` - Supabase client setup
- `vercel.json` - Vercel configuration
- `.env.example` - Example environment variables
- `requirements.txt` - Python dependencies

## Endpoints
- `GET /` - Health check
- `GET /health` - Health status
- `POST /whatsapp-webhook` - Twilio WhatsApp webhook
- `POST /confirm-items` - Confirm item selection
- `POST /payment-success` - Paystack webhook
- `POST /delivery-status` - Update delivery status

## API Documentation
Once deployed, visit:
- `https://your-domain.vercel.app/docs` - Interactive API docs
- `https://your-domain.vercel.app/redoc` - ReDoc documentation

---

For more details, see the main project plan. 