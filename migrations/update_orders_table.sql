-- First, check if columns already exist to avoid errors
DO $$ 
BEGIN
    -- Add delivery_location if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name = 'orders' AND column_name = 'delivery_location') THEN
        ALTER TABLE orders ADD COLUMN delivery_location TEXT;
    END IF;

    -- Add location_updated_at if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name = 'orders' AND column_name = 'location_updated_at') THEN
        ALTER TABLE orders ADD COLUMN location_updated_at TIMESTAMP WITH TIME ZONE;
    END IF;

    -- Add delivery_type if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name = 'orders' AND column_name = 'delivery_type') THEN
        ALTER TABLE orders ADD COLUMN delivery_type VARCHAR(20) DEFAULT 'delivery';
    END IF;

    -- Add delivery_fee if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name = 'orders' AND column_name = 'delivery_fee') THEN
        ALTER TABLE orders ADD COLUMN delivery_fee DECIMAL(10,2) DEFAULT 0.00;
    END IF;

    -- Add total_with_delivery if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name = 'orders' AND column_name = 'total_with_delivery') THEN
        ALTER TABLE orders ADD COLUMN total_with_delivery DECIMAL(10,2);
    END IF;
END $$;

-- Create index if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes 
                  WHERE tablename = 'orders' AND indexname = 'idx_orders_delivery_location') THEN
        CREATE INDEX idx_orders_delivery_location ON orders(delivery_location);
    END IF;
END $$;

-- Add or update comments
COMMENT ON COLUMN orders.delivery_location IS 'Customer''s delivery address or location pin shared via WhatsApp';
COMMENT ON COLUMN orders.location_updated_at IS 'Timestamp when the delivery location was last updated';
COMMENT ON COLUMN orders.delivery_type IS 'Type of delivery: delivery or pickup';
COMMENT ON COLUMN orders.delivery_fee IS 'Delivery fee based on location';
COMMENT ON COLUMN orders.total_with_delivery IS 'Total amount including delivery fee';

-- Update existing orders to have default values
UPDATE orders 
SET 
    delivery_type = COALESCE(delivery_type, 'delivery'),
    delivery_fee = COALESCE(delivery_fee, 0.00),
    total_with_delivery = COALESCE(total_with_delivery, total_amount)
WHERE 
    delivery_type IS NULL 
    OR delivery_fee IS NULL 
    OR total_with_delivery IS NULL; 