"""
EDA and KPI analysis module.

Functions for:
- KPI summaries
- Rating distribution analysis
- Cocoa percentage analysis
- Brand and origin performance
- Statistical tests (ANOVA, correlation)
- Business insight generation
"""

import pandas as pd
import numpy as np
from scipy import stats



# ─── KPI Summary ──────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPIs for the dashboard.

    Returns:
        Dict with all KPI values
    """
    kpis = {
        "total_products": len(df),
        "total_brands": df["company"].nunique(),
        "total_origins": df["broad_bean_origin"].nunique(),
        "total_countries": df["company_location"].nunique(),
        "avg_rating": round(df["rating"].mean(), 3),
        "median_rating": round(df["rating"].median(), 3),
        "premium_share_pct": round(df["premium_flag"].mean() * 100, 1),
        "avg_cocoa_pct": round(df["cocoa_percent_clean"].mean(), 1),
        "rating_std": round(df["rating"].std(), 3),
        "outstanding_products": int((df["rating_segment"] == "Outstanding").sum()),
        "year_range": f"{int(df['review_year'].min())} – {int(df['review_year'].max())}",
    }
    return kpis


# ─── Rating Distribution ──────────────────────────────────────────────────────

def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Value counts of rating values with percentage.

    Returns:
        DataFrame with rating, count, and percentage columns
    """
    counts = df["rating"].value_counts().sort_index().reset_index()
    counts.columns = ["rating", "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(1)
    return counts


def rating_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Group-level summary by rating_segment."""
    order = ["Poor", "Below Average", "Average", "Good", "Outstanding"]
    return (
        df.groupby("rating_segment")
        .agg(count=("rating", "size"), avg_rating=("rating", "mean"))
        .reindex(order)
        .reset_index()
    )


# ─── Cocoa Analysis ───────────────────────────────────────────────────────────

def cocoa_vs_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average rating per cocoa_category bucket.

    Returns:
        DataFrame with cocoa category, count, and mean rating
    """
    return (
        df.groupby("cocoa_category")
        .agg(count=("rating", "size"), avg_rating=("rating", "mean"), std_rating=("rating", "std"))
        .reset_index()
        .sort_values("avg_rating", ascending=False)
    )


def optimal_cocoa_range(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """
    Find the cocoa % range that maximizes average rating.

    Args:
        df: Featured DataFrame
        bins: Number of bins to cut cocoa % into

    Returns:
        DataFrame with bin ranges and avg ratings
    """
    df = df.copy()
    df["cocoa_bin"] = pd.cut(df["cocoa_percent_clean"], bins=bins)
    result = (
        df.groupby("cocoa_bin", observed=True)
        .agg(count=("rating", "size"), avg_rating=("rating", "mean"))
        .reset_index()
        .dropna()
    )
    result["cocoa_bin"] = result["cocoa_bin"].astype(str)
    return result


# ─── Brand Analysis ───────────────────────────────────────────────────────────

def top_brands(df: pd.DataFrame, n: int = 15, min_reviews: int = 5) -> pd.DataFrame:
    """
    Top N brands by average rating with min review threshold.

    Args:
        df: DataFrame
        n: Number of top brands to return
        min_reviews: Minimum reviews required for inclusion

    Returns:
        DataFrame of top brands with stats
    """
    brand_stats = (
        df.groupby("company")
        .agg(
            avg_rating=("rating", "mean"),
            count=("rating", "size"),
            premium_pct=("premium_flag", "mean"),
            location=("company_location", "first"),
        )
        .reset_index()
    )
    brand_stats = brand_stats[brand_stats["count"] >= min_reviews]
    brand_stats["avg_rating"] = brand_stats["avg_rating"].round(3)
    brand_stats["premium_pct"] = (brand_stats["premium_pct"] * 100).round(1)
    return brand_stats.sort_values("avg_rating", ascending=False).head(n)


def premium_brands(df: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
    """
    Brands where premium_share > 75%, with at least min_reviews reviews.
    """
    brand_stats = (
        df.groupby("company")
        .agg(
            avg_rating=("rating", "mean"),
            count=("rating", "size"),
            premium_pct=("premium_flag", "mean"),
        )
        .reset_index()
    )
    brand_stats = brand_stats[brand_stats["count"] >= min_reviews]
    brand_stats["premium_pct"] = brand_stats["premium_pct"] * 100
    return brand_stats[brand_stats["premium_pct"] >= 75].sort_values("premium_pct", ascending=False)


# ─── Origin Analysis ──────────────────────────────────────────────────────────

def top_origins(df: pd.DataFrame, n: int = 15, min_reviews: int = 5) -> pd.DataFrame:
    """
    Top N bean origins by average rating.

    Args:
        df: DataFrame
        n: Number of top origins
        min_reviews: Minimum reviews required

    Returns:
        DataFrame with origin stats
    """
    origin_stats = (
        df.groupby("broad_bean_origin")
        .agg(
            avg_rating=("rating", "mean"),
            count=("rating", "size"),
            premium_pct=("premium_flag", "mean"),
            origin_score=("origin_score", "first"),
        )
        .reset_index()
    )
    origin_stats = origin_stats[origin_stats["count"] >= min_reviews]
    origin_stats["avg_rating"] = origin_stats["avg_rating"].round(3)
    origin_stats["premium_pct"] = (origin_stats["premium_pct"] * 100).round(1)
    return origin_stats.sort_values("avg_rating", ascending=False).head(n)


# ─── Country Analysis ─────────────────────────────────────────────────────────

def country_analysis(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """
    Company location (country) level analysis.

    Returns:
        DataFrame with country stats
    """
    country_stats = (
        df.groupby("company_location")
        .agg(
            avg_rating=("rating", "mean"),
            count=("rating", "size"),
            premium_share=("premium_flag", "mean"),
            unique_brands=("company", "nunique"),
        )
        .reset_index()
    )
    country_stats["avg_rating"] = country_stats["avg_rating"].round(3)
    country_stats["premium_share"] = (country_stats["premium_share"] * 100).round(1)
    return country_stats.sort_values("count", ascending=False).head(n)


# ─── Bean Type Analysis ───────────────────────────────────────────────────────

def bean_type_analysis(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """
    Performance of different bean types.

    Args:
        df: DataFrame
        min_count: Minimum occurrences to include

    Returns:
        DataFrame with bean type stats
    """
    bt_stats = (
        df[df["bean_type"] != "Unknown"]
        .groupby("bean_type")
        .agg(
            avg_rating=("rating", "mean"),
            count=("rating", "size"),
            premium_pct=("premium_flag", "mean"),
        )
        .reset_index()
    )
    bt_stats = bt_stats[bt_stats["count"] >= min_count]
    bt_stats["avg_rating"] = bt_stats["avg_rating"].round(3)
    bt_stats["premium_pct"] = (bt_stats["premium_pct"] * 100).round(1)
    return bt_stats.sort_values("avg_rating", ascending=False)


# ─── Statistical Tests ────────────────────────────────────────────────────────

def anova_cocoa_vs_rating(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA: does cocoa category significantly affect rating?

    Returns:
        Dict with F-stat, p-value, and business interpretation
    """
    groups = [
        df[df["cocoa_category"] == cat]["rating"].values
        for cat in df["cocoa_category"].unique()
    ]
    f_stat, p_value = stats.f_oneway(*groups)

    interpretation = (
        "SIGNIFICANT: Cocoa % category has a statistically significant effect on rating. "
        "Chocolate makers should carefully choose their cocoa range."
        if p_value < 0.05
        else "NOT SIGNIFICANT: No strong statistical evidence that cocoa % category drives rating differences."
    )

    return {
        "test": "One-way ANOVA",
        "factor": "Cocoa Category",
        "target": "Rating",
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "interpretation": interpretation,
    }


def correlation_cocoa_rating(df: pd.DataFrame) -> dict:
    """
    Pearson correlation between cocoa % and rating.

    Returns:
        Dict with correlation, p-value, and interpretation
    """
    corr, p_value = stats.pearsonr(df["cocoa_percent_clean"], df["rating"])

    strength = (
        "strong" if abs(corr) > 0.5
        else "moderate" if abs(corr) > 0.3
        else "weak"
    )
    direction = "positive" if corr > 0 else "negative"

    interpretation = (
        f"There is a {strength} {direction} correlation (r={corr:.3f}) between cocoa % and rating. "
        f"{'Higher cocoa content tends to get better ratings.' if corr > 0 else 'Higher cocoa content tends to get lower ratings.'}"
    )

    return {
        "test": "Pearson Correlation",
        "variable_1": "Cocoa Percent",
        "variable_2": "Rating",
        "correlation": round(float(corr), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "interpretation": interpretation,
    }


def anova_region_vs_rating(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA: does company region significantly affect rating?

    Returns:
        Dict with test results
    """
    groups = [
        df[df["company_region"] == region]["rating"].values
        for region in df["company_region"].unique()
        if len(df[df["company_region"] == region]) >= 10
    ]
    f_stat, p_value = stats.f_oneway(*groups)

    interpretation = (
        "SIGNIFICANT: Company region has a statistically significant effect on product ratings. "
        "Geography of the chocolate maker matters."
        if p_value < 0.05
        else "NOT SIGNIFICANT: Region of the company does not significantly predict rating."
    )

    return {
        "test": "One-way ANOVA",
        "factor": "Company Region",
        "target": "Rating",
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "interpretation": interpretation,
    }


# ─── Underserved Segments ─────────────────────────────────────────────────────

def find_underserved_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify origin × cocoa_category combinations with:
    - High avg rating (>= 3.5)
    - Low product count (< 10 reviews)

    These are market opportunities — quality niche with low competition.

    Returns:
        DataFrame of underserved segments
    """
    seg = (
        df.groupby(["broad_bean_origin", "cocoa_category"])
        .agg(count=("rating", "size"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    underserved = seg[(seg["avg_rating"] >= 3.5) & (seg["count"] < 10)]
    return underserved.sort_values("avg_rating", ascending=False)


# ─── Run All Analysis ─────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame) -> dict:
    """
    Run all EDA functions and return results dict.

    Args:
        df: Featured DataFrame

    Returns:
        Dict of all analysis results
    """
    print("=" * 60)
    print("RUNNING FULL EDA + STATISTICAL ANALYSIS")
    print("=" * 60)

    results = {
        "kpis": compute_kpis(df),
        "rating_distribution": rating_distribution(df),
        "rating_by_segment": rating_by_segment(df),
        "cocoa_vs_rating": cocoa_vs_rating(df),
        "optimal_cocoa_range": optimal_cocoa_range(df),
        "top_brands": top_brands(df),
        "premium_brands": premium_brands(df),
        "top_origins": top_origins(df),
        "country_analysis": country_analysis(df),
        "bean_type_analysis": bean_type_analysis(df),
        "anova_cocoa": anova_cocoa_vs_rating(df),
        "correlation_cocoa": correlation_cocoa_rating(df),
        "anova_region": anova_region_vs_rating(df),
        "underserved_segments": find_underserved_segments(df),
    }

    print("\n--- KPIs ---")
    for k, v in results["kpis"].items():
        print(f"  {k}: {v}")

    print("\n--- Statistical Tests ---")
    for test_key in ["anova_cocoa", "correlation_cocoa", "anova_region"]:
        t = results[test_key]
        print(f"  [{t['test']}] {t.get('factor', t.get('variable_1'))} → p={t['p_value']} | {t['interpretation']}")

    return results


if __name__ == "__main__":
    df = pd.read_csv("data/processed/chocolate_featured.csv")
    results = run_full_eda(df)
    print("\nTop 5 brands:")
    print(results["top_brands"].head())
    print("\nTop 5 origins:")
    print(results["top_origins"].head())
