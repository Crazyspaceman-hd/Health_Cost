-- Service-category cost and utilization trend by year.
--
-- Category PMPM uses TOTAL member-months for the year as the denominator (every
-- member-month is exposed to every category), so category PMPMs sum to the
-- overall PMPM. Utilization is expressed per 1,000 member-months.
-- :trunc is the high-cost truncation threshold.
WITH cat AS (
    SELECT
        service_category,
        year,
        SUM(allowed_amount) AS total_allowed,
        SUM(CASE WHEN allowed_amount > :trunc THEN :trunc
                 ELSE allowed_amount END) AS total_allowed_trunc,
        COUNT(*) AS claims
    FROM analytic_claims
    GROUP BY service_category, year
),
mm AS (
    SELECT year, COUNT(*) AS member_months
    FROM enrollment
    GROUP BY year
)
SELECT
    c.service_category,
    c.year,
    m.member_months,
    c.claims,
    c.total_allowed,
    c.total_allowed_trunc * 1.0 / m.member_months AS truncated_pmpm,
    c.claims * 1000.0 / m.member_months           AS utilization_per_1000_mm
FROM cat c
JOIN mm m ON c.year = m.year
ORDER BY c.service_category, c.year;
