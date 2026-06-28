-- Member months by payer and year (the PMPM denominator).
-- One enrollment row == one member-month, so a simple COUNT(*) suffices.
SELECT
    payer_id,
    year,
    COUNT(*) AS member_months
FROM enrollment
GROUP BY payer_id, year
ORDER BY payer_id, year;
