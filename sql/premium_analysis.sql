-- ============================================================
-- PREMIUM SEGMENT ANALYSIS — Premium Chocolate Analysis
-- ============================================================

-- ─── 1. Top Brands by Average Rating (min 5 reviews) ────────────────────────
SELECT
    company,
    company_location,
    COUNT(*)                  AS review_count,
    ROUND(AVG(rating), 3)    AS avg_rating,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_pct,
    brand_tier
FROM chocolate_ratings
GROUP BY company, company_location, brand_tier
HAVING COUNT(*) >= 5
ORDER BY avg_rating DESC
LIMIT 20;


-- ─── 2. Top Bean Origins by Rating (min 5 reviews) ──────────────────────────
SELECT
    broad_bean_origin,
    COUNT(*)                  AS review_count,
    ROUND(AVG(rating), 3)    AS avg_rating,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_pct,
    ROUND(AVG(origin_score), 1)  AS origin_score
FROM chocolate_ratings
GROUP BY broad_bean_origin
HAVING COUNT(*) >= 5
ORDER BY avg_rating DESC
LIMIT 20;


-- ─── 3. Cocoa Category vs Rating ─────────────────────────────────────────────
SELECT
    cocoa_category,
    COUNT(*)                  AS product_count,
    ROUND(AVG(rating), 3)    AS avg_rating,
    ROUND(MIN(rating), 2)    AS min_rating,
    ROUND(MAX(rating), 2)    AS max_rating,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_pct
FROM chocolate_ratings
GROUP BY cocoa_category
ORDER BY avg_rating DESC;


-- ─── 4. Company Country Performance ──────────────────────────────────────────
SELECT
    company_location,
    COUNT(*)                      AS product_count,
    COUNT(DISTINCT company)       AS unique_brands,
    ROUND(AVG(rating), 3)        AS avg_rating,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_pct
FROM chocolate_ratings
GROUP BY company_location
HAVING COUNT(*) >= 10
ORDER BY avg_rating DESC;


-- ─── 5. Bean Type Performance ────────────────────────────────────────────────
SELECT
    bean_type,
    COUNT(*)                  AS product_count,
    ROUND(AVG(rating), 3)    AS avg_rating,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_pct
FROM chocolate_ratings
WHERE bean_type != 'Unknown'
GROUP BY bean_type
HAVING COUNT(*) >= 10
ORDER BY avg_rating DESC;


-- ─── 6. Cluster Segment Profiles ─────────────────────────────────────────────
SELECT
    cluster_label,
    COUNT(*)                      AS product_count,
    ROUND(AVG(rating), 3)        AS avg_rating,
    ROUND(AVG(cocoa_percent_clean), 1) AS avg_cocoa,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_pct,
    COUNT(DISTINCT company)       AS unique_brands,
    COUNT(DISTINCT broad_bean_origin) AS unique_origins
FROM chocolate_ratings
GROUP BY cluster_label
ORDER BY avg_rating DESC;


-- ─── 7. Underserved Market Opportunities ─────────────────────────────────────
-- High avg rating + low product count = white space opportunity
SELECT
    broad_bean_origin,
    cocoa_category,
    COUNT(*)              AS product_count,
    ROUND(AVG(rating), 3) AS avg_rating
FROM chocolate_ratings
GROUP BY broad_bean_origin, cocoa_category
HAVING COUNT(*) < 10
   AND AVG(rating) >= 3.5
ORDER BY avg_rating DESC, product_count ASC
LIMIT 20;


-- ─── 8. Region-Level Analysis ────────────────────────────────────────────────
SELECT
    company_region,
    COUNT(*)                      AS product_count,
    COUNT(DISTINCT company)       AS unique_brands,
    ROUND(AVG(rating), 3)        AS avg_rating,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_pct
FROM chocolate_ratings
GROUP BY company_region
ORDER BY avg_rating DESC;


-- ─── 9. Year-over-Year Premium Growth ────────────────────────────────────────
SELECT
    review_year,
    SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) AS premium_count,
    COUNT(*) AS total_count,
    ROUND(100.0 * SUM(CASE WHEN premium_flag THEN 1 ELSE 0 END) / COUNT(*), 1) AS premium_pct,
    ROUND(AVG(rating), 3) AS avg_rating
FROM chocolate_ratings
GROUP BY review_year
ORDER BY review_year;


-- ─── 10. Elite Brand × Premium Origin Intersection ───────────────────────────
-- Best of both worlds: elite brands using premium origins
SELECT
    company,
    broad_bean_origin,
    company_location,
    ROUND(AVG(rating), 3)  AS avg_rating,
    COUNT(*)               AS products
FROM chocolate_ratings
WHERE brand_tier = 'Elite'
  AND premium_origin_flag = TRUE
GROUP BY company, broad_bean_origin, company_location
HAVING COUNT(*) >= 2
ORDER BY avg_rating DESC
LIMIT 15;
