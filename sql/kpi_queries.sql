-- ============================================================
-- KPI QUERIES — Premium Chocolate Analysis
-- Target: SQLite / DuckDB / BigQuery compatible (standard SQL)
-- Table: chocolate_ratings
-- ============================================================

-- ─── 1. Top-Level KPIs ────────────────────────────────────────────────────────
SELECT
    COUNT(*)                                          AS total_products,
    COUNT(DISTINCT company)                           AS total_brands,
    COUNT(DISTINCT broad_bean_origin)                 AS total_origins,
    COUNT(DISTINCT company_location)                  AS total_countries,
    ROUND(AVG(rating), 3)                             AS avg_rating,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_share_pct,
    ROUND(AVG(cocoa_percent_clean), 1)                AS avg_cocoa_pct,
    MIN(review_year)                                  AS earliest_review,
    MAX(review_year)                                  AS latest_review
FROM chocolate_ratings;


-- ─── 2. Rating Distribution ───────────────────────────────────────────────────
SELECT
    rating,
    COUNT(*)                                          AS product_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
FROM chocolate_ratings
GROUP BY rating
ORDER BY rating;


-- ─── 3. Annual Trends ─────────────────────────────────────────────────────────
SELECT
    review_year,
    COUNT(*)                  AS total_reviews,
    ROUND(AVG(rating), 3)    AS avg_rating,
    COUNT(DISTINCT company)   AS active_brands,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_pct
FROM chocolate_ratings
GROUP BY review_year
ORDER BY review_year;


-- ─── 4. Premium Market Share by Rating Segment ────────────────────────────────
SELECT
    rating_segment,
    COUNT(*)                                          AS product_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS market_share_pct,
    ROUND(AVG(cocoa_percent_clean), 1)                AS avg_cocoa_pct
FROM chocolate_ratings
GROUP BY rating_segment
ORDER BY
    CASE rating_segment
        WHEN 'Outstanding'    THEN 1
        WHEN 'Good'           THEN 2
        WHEN 'Average'        THEN 3
        WHEN 'Below Average'  THEN 4
        WHEN 'Poor'           THEN 5
    END;
