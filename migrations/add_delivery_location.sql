-- Add delivery_location column to orders table
ALTER TABLE orders
ADD COLUMN delivery_location TEXT,
ADD COLUMN location_updated_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN delivery_type VARCHAR(20) DEFAULT 'delivery',
ADD COLUMN delivery_fee DECIMAL(10,2) DEFAULT 0.00,
ADD COLUMN total_with_delivery DECIMAL(10,2);

-- Add index for faster queries on delivery_location
CREATE INDEX idx_orders_delivery_location ON orders(delivery_location);

-- Add comment to explain the columns
COMMENT ON COLUMN orders.delivery_location IS 'Customer''s delivery address or location pin shared via WhatsApp';
COMMENT ON COLUMN orders.location_updated_at IS 'Timestamp when the delivery location was last updated';
COMMENT ON COLUMN orders.delivery_type IS 'Type of delivery: delivery or pickup';
COMMENT ON COLUMN orders.delivery_fee IS 'Delivery fee based on location';
COMMENT ON COLUMN orders.total_with_delivery IS 'Total amount including delivery fee'; 