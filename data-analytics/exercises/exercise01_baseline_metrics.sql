-- Exercise 01: Baseline Metrics & Profiling
-- Objective: run foundational counts and explain joins so the team understands table cardinality and potential bottlenecks.

-- 1. Total rows per table (quick sanity check).
SELECT 'customers' AS table_name, COUNT(*) AS row_count FROM analytics_demo.customers
UNION ALL
SELECT 'products', COUNT(*) FROM analytics_demo.products
UNION ALL
SELECT 'orders', COUNT(*) FROM analytics_demo.orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM analytics_demo.order_items
UNION ALL
SELECT 'sessions', COUNT(*) FROM analytics_demo.sessions;

-- 2. Join explain: revenue by customer segment (will also highlight the need for indexes on customer_id).
EXPLAIN FORMAT=JSON
SELECT c.segment,
       COUNT(DISTINCT o.customer_id) AS distinct_customers,
       SUM(o.order_total) AS revenue
FROM analytics_demo.customers c
JOIN analytics_demo.orders o ON c.customer_id = o.customer_id
GROUP BY c.segment
ORDER BY revenue DESC;

-- 3. Channel-level pivot to understand payment mix.
SELECT o.channel,
       o.payment_method,
       COUNT(*) AS orders_count,
       ROUND(AVG(o.order_total), 2) AS avg_ticket
FROM analytics_demo.orders o
GROUP BY o.channel, o.payment_method
ORDER BY orders_count DESC
LIMIT 20;

-- 4. Rolling 7-day revenue by segment (window function).
WITH segment_orders AS (
  SELECT c.segment,
         o.order_date,
         o.order_total
  FROM analytics_demo.customers c
  JOIN analytics_demo.orders o ON c.customer_id = o.customer_id
)
SELECT segment,
       DATE(o.order_date) AS dt,
       ROUND(SUM(o.order_total) OVER (
         PARTITION BY segment
         ORDER BY DATE(o.order_date)
         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ), 2) AS rolling_7d_revenue
FROM segment_orders o
ORDER BY segment, dt
LIMIT 100;

-- 5. Session conversion rate per device (correlated subquery).
SELECT s.device,
       SUM(CASE WHEN EXISTS (
         SELECT 1
         FROM analytics_demo.orders o
         WHERE o.customer_id = s.customer_id
           AND o.order_date >= s.start_time
           AND o.order_date <= s.end_time
       ) THEN 1 ELSE 0 END) AS converted_sessions,
       COUNT(*) AS total_sessions,
       ROUND(100.0 * SUM(CASE WHEN EXISTS (
         SELECT 1
         FROM analytics_demo.orders o
         WHERE o.customer_id = s.customer_id
           AND o.order_date BETWEEN s.start_time AND s.end_time
       ) THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_pct
FROM analytics_demo.sessions s
GROUP BY s.device
ORDER BY conversion_pct DESC;

-- 6. Anomaly profiling summary of negative/future orders.
SELECT issue,
       COUNT(*) AS occurrences,
       MIN(order_id) AS representative_order
FROM (
  SELECT order_id,
         CASE
           WHEN order_total < 0 THEN 'negative_total'
           WHEN order_date > CURRENT_DATE THEN 'future_date'
           ELSE 'other'
         END AS issue
  FROM analytics_demo.orders
) flags
GROUP BY issue;

-- 7. Anomaly profiling dashboard (temporary result sets).
WITH negative_orders AS (
  SELECT order_id, order_total, 'negative_total' AS issue FROM analytics_demo.orders WHERE order_total < 0
), future_orders AS (
  SELECT order_id, order_date, 'future_date' AS issue FROM analytics_demo.orders WHERE order_date > CURRENT_DATE
), invalid_emails AS (
  SELECT customer_id, email, 'bad_email' AS issue FROM analytics_demo.customers WHERE email NOT LIKE '%@%'
), zero_sessions AS (
  SELECT session_id, customer_id, 'zero_duration' AS issue
  FROM analytics_demo.sessions
  WHERE TIMESTAMPDIFF(SECOND, start_time, end_time) <= 0
)
SELECT issue, COUNT(*) AS occurrences FROM (
  SELECT * FROM negative_orders
  UNION ALL
  SELECT * FROM future_orders
  UNION ALL
  SELECT * FROM invalid_emails
  UNION ALL
  SELECT * FROM zero_sessions
) anomaly_summary
GROUP BY issue;

-- 8. Refine orders per payment method vs. segment with averages.
SELECT c.segment,
       o.payment_method,
       COUNT(*) AS orders_count,
       ROUND(AVG(o.order_total), 2) AS avg_ticket
FROM analytics_demo.customers c
JOIN analytics_demo.orders o ON c.customer_id = o.customer_id
GROUP BY c.segment, o.payment_method
HAVING orders_count >= 5
ORDER BY c.segment, avg_ticket DESC
LIMIT 30;

-- 9. Performance explain (JSON) for a heavy aggregation.
EXPLAIN FORMAT=JSON
SELECT c.segment, o.payment_method, AVG(o.order_total)
FROM analytics_demo.customers c
JOIN analytics_demo.orders o ON c.customer_id = o.customer_id
GROUP BY c.segment, o.payment_method;

-- 10. Derived customer_segments view by avg spend.
CREATE OR REPLACE VIEW analytics_demo.customer_segments AS
SELECT c.customer_id,
       CASE
         WHEN AVG(o.order_total) OVER (PARTITION BY c.customer_id) < 100 THEN 'value_100'
         WHEN AVG(o.order_total) OVER (PARTITION BY c.customer_id) BETWEEN 100 AND 500 THEN 'value_500'
         ELSE 'value_plus'
       END AS segment_bucket
FROM analytics_demo.customers c
LEFT JOIN analytics_demo.orders o ON c.customer_id = o.customer_id;

-- 11. Field-level quality checks.
SELECT issue,
       COUNT(*) AS occurrences
FROM (
  SELECT customer_id, 'missing_segment' AS issue FROM analytics_demo.customers WHERE segment IS NULL OR segment = ''
  UNION ALL
  SELECT customer_id, 'invalid_email' FROM analytics_demo.customers WHERE email NOT LIKE '%@%'
  UNION ALL
  SELECT order_id, 'negative_total' FROM analytics_demo.orders WHERE order_total < 0
) issue_counts
GROUP BY issue;

-- 12. SQL vs pandas mapping note (pseudo-query for documentation).
-- SQL window function: `SUM(order_total) OVER (PARTITION BY segment ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`
-- pandas equivalent: `df.groupby('segment').rolling(7)['order_total'].sum()`

-- 13. Weekly cohort analysis.
WITH cohorts AS (
  SELECT customer_id,
         DATE_FORMAT(signup_date, '%Y-%u') AS signup_week
  FROM analytics_demo.customers
), orders_per_week AS (
  SELECT o.customer_id,
         DATE_FORMAT(o.order_date, '%Y-%u') AS order_week
  FROM analytics_demo.orders o
)
SELECT c.signup_week,
       o.order_week,
       COUNT(DISTINCT o.customer_id) AS returning_customers
FROM cohorts c
JOIN orders_per_week o ON c.customer_id = o.customer_id
GROUP BY c.signup_week, o.order_week
ORDER BY c.signup_week, o.order_week;

-- 14. KPI table creation (daily summary).
DROP TABLE IF EXISTS analytics_demo.kpi_daily;
CREATE TABLE analytics_demo.kpi_daily AS
SELECT DATE(o.order_date) AS report_date,
       COUNT(DISTINCT o.customer_id) AS new_customer_count,
       SUM(o.order_total) AS daily_revenue,
       SUM(CASE WHEN o.order_total < 0 THEN 1 ELSE 0 END) AS flagged_negative_orders
FROM analytics_demo.orders o
GROUP BY DATE(o.order_date);

-- 15. Diagnose anomalies via event sourcing union.
WITH anomalies AS (
  SELECT order_id, 'negative_total' AS issue, order_total, order_date FROM analytics_demo.orders WHERE order_total < 0
  UNION ALL
  SELECT order_id, 'future_date', order_total, order_date FROM analytics_demo.orders WHERE order_date > CURRENT_DATE
  UNION ALL
  SELECT customer_id, 'invalid_email', NULL, NULL FROM analytics_demo.customers WHERE email NOT LIKE '%@%'
) SELECT * FROM anomalies ORDER BY issue LIMIT 50;

-- 16. Cleanup preview statements (SELECT ... FOR UPDATE + UPDATE).
START TRANSACTION;
SELECT * FROM analytics_demo.orders WHERE order_total < 0 LIMIT 5 FOR UPDATE;
UPDATE analytics_demo.orders SET status = 'needs_review' WHERE order_total < 0;
ROLLBACK;

-- 17. Artifacts & documentation guidance.
-- Save each of the above SQL snippets in `scratches/data-analytics/exercises` and note timestamp, dataset version, explanation (use doc comment sections).

-- 18. Mini quiz prompts (no SQL) – ask learners to recap top anomalies/time ranges they discovered. Document answers in shared notes.

-- 19. Preparation for cleansing phase (comment block).
-- Summarize: we have identified duplicates, invalid emails, negative/future orders, skewed session durations – these will be cleansed next.

-- 20. Optional labs instructions (comment block).
-- Encourage students to adapt any of the above queries (e.g., filter sessions where `device = 'unknown'`, orders missing shipping info) and present results to peers.

-- Validation comments:
-- After running these queries, confirm that the row counts match the output from generate_test_data.
-- Review the JSON EXPLAIN to ensure the join order aligns with indexed keys (customer_id primary key).
