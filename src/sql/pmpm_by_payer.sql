-- PMPM by payer and year, with high-cost truncation.
--
-- :trunc is the high-cost truncation threshold. Each claim's allowed amount is
-- winsorized at :trunc (CASE expression) before summing, so a handful of
-- catastrophic claims cannot dominate the trend -- the same logic the pandas
-- metrics engine applies. PMPM = truncated allowed dollars / member-months.
WITH spend AS (
    SELECT
        payer_id,
        year,
        SUM(allowed_amount) AS total_allowed,
        SUM(paid_amount)    AS total_paid,
        SUM(CASE WHEN allowed_amount > :trunc THEN :trunc
                 ELSE allowed_amount END) AS total_allowed_trunc,
        COUNT(*)            AS claims
    FROM analytic_claims
    GROUP BY payer_id, year
),
mm AS (
    SELECT payer_id, year, COUNT(*) AS member_months
    FROM enrollment
    GROUP BY payer_id, year
)
SELECT
    s.payer_id,
    s.year,
    m.member_months,
    s.total_allowed,
    s.total_paid,
    s.total_allowed_trunc,
    s.claims,
    s.total_allowed      * 1.0 / m.member_months AS allowed_pmpm,
    s.total_paid         * 1.0 / m.member_months AS paid_pmpm,
    s.total_allowed_trunc * 1.0 / m.member_months AS truncated_pmpm
FROM spend s
JOIN mm m
  ON s.payer_id = m.payer_id AND s.year = m.year
ORDER BY s.payer_id, s.year;
