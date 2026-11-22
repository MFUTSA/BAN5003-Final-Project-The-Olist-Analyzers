SELECT
    o.order_id,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    o.order_purchase_timestamp,
    o.order_delivered_carrier_date,
    o.order_approved_at, 
    oi.shipping_limit_date,
    (oi.shipping_limit_date::date - o.order_purchase_timestamp::date) AS shipping_limit_days, 
    (o.order_approved_at::date - o.order_purchase_timestamp::date) AS approved_days, 
    (o.order_delivered_customer_date::date - o.order_estimated_delivery_date::date) AS days_from_est,
    (o.order_estimated_delivery_date ::date - o.order_purchase_timestamp::date) AS est_delivery_days,   
	(o.order_delivered_customer_date ::date - o.order_purchase_timestamp::date) AS delivery_days, 
	ROUND(
    	1.0 * (o.order_estimated_delivery_date::date - o.order_purchase_timestamp::date)
        	/ NULLIF(o.order_delivered_customer_date::date - o.order_purchase_timestamp::date, 0)
	, 2) AS est_to_actual_ratio,    
	EXTRACT(YEAR  FROM o.order_purchase_timestamp)::int AS order_year,
    EXTRACT(MONTH FROM o.order_purchase_timestamp)::int AS order_month,
    s.seller_state,
    c.customer_state,
    c.customer_zip_code_prefix,
    pcnt.product_category_name_english,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,
    (p.product_length_cm * p.product_height_cm * p.product_width_cm) AS prod_dim_cm3,
    r.review_score AS avg_review_score
FROM olist.orders o
JOIN olist.order_items oi ON oi.order_id = o.order_id
JOIN olist.customers c ON c.customer_id = o.customer_id
JOIN olist.sellers s ON s.seller_id = oi.seller_id
JOIN olist.products p ON p.product_id = oi.product_id
JOIN olist.product_category_name_translation pcnt ON pcnt.product_category_name = p.product_category_name
LEFT JOIN (
	    SELECT order_id, AVG(review_score) AS review_score
	    FROM olist.order_reviews
	    GROUP BY order_id
	) r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_approved_at IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
  AND o.order_delivered_carrier_date IS NOT NULL
  AND o.order_purchase_timestamp IS NOT NULL
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_delivered_customer_date::date > o.order_purchase_timestamp::date
  AND o.order_id IN (
      SELECT oi2.order_id
      FROM olist.order_items oi2
      GROUP BY oi2.order_id
      HAVING COUNT(*) = 1
      );
