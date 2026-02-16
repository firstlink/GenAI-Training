-- Exercise 02: Cleansing & Governance workflows
-- Focus: deduplicate, normalize, flag anomalies, and document cleanup plan (Hours 3-6). Each section has validation notes.

SET @orig_mode = @@sql_mode;
SET SESSION sql_mode = REPLACE(@@sql_mode, 'NO_ZERO_DATE', '');

-- 1. Deduplicate customers (keep latest signup) using a row-number derived table.
CREATE TEMPORARY TABLE temp_customers AS
WITH raw_customers AS (
  SELECT customer_id, first_name, last_name, email, segment, country,
         CASE WHEN signup_date = '0000-00-00' THEN NULL ELSE signup_date END AS signup_date
  FROM analytics_demo.customers
)
SELECT customer_id, first_name, last_name, email, segment, country, signup_date
FROM (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY LOWER(email) ORDER BY signup_date DESC) AS rn
  FROM raw_customers
) ranked_raw
WHERE rn = 1;
-- Validate: compare COUNT(*) of temp_customers vs. analytics_demo.customers.

-- 2. Normalize names + segments, store result in a staging table.
CREATE TEMPORARY TABLE customers_normalized AS
SELECT customer_id,
       TRIM(LOWER(first_name)) AS first_name,
       TRIM(LOWER(last_name)) AS last_name,
       TRIM(LOWER(segment)) AS segment,
       LOWER(email) AS email,
       country,
       IF(signup_date = '0000-00-00', NULL, signup_date) AS signup_date
FROM temp_customers;
-- Validation: run `SELECT DISTINCT first_name FROM ...` to ensure lowercase.

-- 3. Flag missing or invalid emails.
ALTER TABLE customers_normalized ADD COLUMN email_status VARCHAR(32) DEFAULT 'valid';
UPDATE customers_normalized
SET email_status = CASE WHEN email NOT LIKE '%@%' THEN 'invalid_email'
                        WHEN email IS NULL OR email = '' THEN 'missing_email'
                        ELSE 'valid' END;
-- Validation: SELECT email_status, COUNT(*) FROM customers_normalized GROUP BY email_status;

-- 4. Replace the production table with cleansed data (upsert pattern).
DELETE FROM analytics_demo.customers WHERE customer_id IN (SELECT customer_id FROM customers_normalized);
INSERT INTO analytics_demo.customers (customer_id, first_name, last_name, email, segment, country, signup_date)
SELECT customer_id, first_name, last_name, email, segment, country, signup_date
FROM customers_normalized;
-- Validation: check the number of rows removed/inserted equals temp_customers count.

-- 5. Flag bad orders (negative/future) in a governance table.
DROP TABLE IF EXISTS analytics_demo.order_flags;
CREATE TABLE analytics_demo.order_flags AS
SELECT order_id,
       customer_id,
       order_date,
       order_total,
       CASE
         WHEN order_total < 0 THEN 'negative_total'
         WHEN order_date > CURRENT_DATE THEN 'future_order'
         ELSE 'ok'
       END AS issue
FROM analytics_demo.orders;
-- Validation: ensure order_flags contains at least one issue (SELECT issue, COUNT(*)...).

-- 6. Balance cleanup preview (no data loss).
SELECT COUNT(*) AS original_customers FROM analytics_demo.customers;
SELECT COUNT(*) AS normalized FROM customers_normalized;
-- Expect same or slightly fewer rows (duplicates collapsed).

-- 7. Document governance (comment block).
-- NOTE: capture these steps (dedup, normalization, flags) in course notes; mention that temp tables simulate staging.

-- 8. Clean session durations and mark anomalies.
ALTER TABLE analytics_demo.sessions ADD COLUMN duration_minutes INT DEFAULT NULL;
UPDATE analytics_demo.sessions
SET duration_minutes = TIMESTAMPDIFF(MINUTE, start_time, end_time)
WHERE start_time IS NOT NULL AND end_time IS NOT NULL;
-- Validation: SELECT COUNT(*) FROM analytics_demo.sessions WHERE duration_minutes < 0;

-- 9. Preview cleanup plan for sessions with negative duration.
SELECT session_id, customer_id, start_time, end_time
FROM analytics_demo.sessions
WHERE duration_minutes < 0;
-- Instructor note: explain why a manual review before updates is best practice.

-- 10. Record final QA metrics (counts for issues).
SELECT 'customers' AS entity, COUNT(*) FROM analytics_demo.customers
UNION ALL
SELECT 'orders', COUNT(*) FROM analytics_demo.orders
UNION ALL
SELECT 'sessions', COUNT(*) FROM analytics_demo.sessions;

SET SESSION sql_mode = @orig_mode;
