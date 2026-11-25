-- Count items in each order
WITH item_counts AS (
    SELECT
        order_id,
        COUNT(*) AS total_items
    FROM olist.order_items
    GROUP BY order_id
)

-- Get delivery information for multi-item orders
SELECT
    o.order_id,
    ic.total_items,
    o.order_purchase_timestamp,
    o.order_estimated_delivery_date,
    o.order_delivered_customer_date,
    (o.order_delivered_customer_date::date -
        o.order_estimated_delivery_date::date) AS days_from_est,
    (o.order_estimated_delivery_date::date -
        o.order_purchase_timestamp::date) AS est_delivery_days,
    (o.order_delivered_customer_date::date -
        o.order_purchase_timestamp::date) AS delivery_days,
    EXTRACT(YEAR FROM o.order_purchase_timestamp)::int AS order_year,
    EXTRACT(MONTH FROM o.order_purchase_timestamp)::int AS order_month,
    c.customer_state
FROM olist.orders o
JOIN item_counts ic ON ic.order_id = o.order_id
JOIN olist.customers c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
    AND o.order_approved_at IS NOT NULL
    AND o.order_estimated_delivery_date IS NOT NULL
    AND o.order_delivered_carrier_date IS NOT NULL
    AND o.order_purchase_timestamp IS NOT NULL
    AND o.order_delivered_customer_date IS NOT NULL
    AND o.order_delivered_customer_date::date >
        o.order_purchase_timestamp::date
    AND ic.total_items > 1
ORDER BY o.order_id;