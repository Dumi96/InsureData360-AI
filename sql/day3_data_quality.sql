
-- InsureData360 AI
-- Day 3: Data quality and reconciliation checks

-- 1. Verify policies reference existing customers.
SELECT COUNT(*) AS policies_without_customers
FROM policies p
LEFT JOIN customers c
    ON p.customer_id = c.customer_id
WHERE c.customer_id IS NULL;

-- 2. Count policies without payment records by status.
SELECT
    p.policy_status,
    COUNT(*) AS policies_without_payment_records
FROM policies p
WHERE NOT EXISTS (
    SELECT 1
    FROM premium_payments pp
    WHERE pp.policy_id = p.policy_id
)
GROUP BY p.policy_status
ORDER BY p.policy_status;

-- 3. Summarize premium payments by status.
SELECT
    payment_status,
    COUNT(*) AS payment_count,
    ROUND(SUM(payment_amount), 2) AS total_payment_amount
FROM premium_payments
GROUP BY payment_status
ORDER BY payment_status;

-- 4. Summarize annual premiums by product.
SELECT
    product,
    COUNT(*) AS policy_count,
    ROUND(SUM(annual_premium), 2) AS total_annual_premium,
    ROUND(AVG(annual_premium), 2) AS average_annual_premium
FROM policies
GROUP BY product
ORDER BY total_annual_premium DESC;

-- 5. Calculate the simplified claims-to-premium indicator.
WITH premium_summary AS (
    SELECT SUM(annual_premium) AS total_annual_premium
    FROM policies
),
claim_summary AS (
    SELECT SUM(claim_amount) AS total_claim_amount
    FROM claims
)
SELECT
    ROUND(c.total_claim_amount, 2) AS total_claim_amount,
    ROUND(p.total_annual_premium, 2) AS total_annual_premium,
    ROUND(
        100.0 * c.total_claim_amount
        / NULLIF(p.total_annual_premium, 0),
        2
    ) AS claims_to_premium_percentage
FROM claim_summary c
CROSS JOIN premium_summary p;
