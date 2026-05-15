"""
Feature engineering module for the Chocolate Ratings Dataset.

Creates:
- premium_flag
- cocoa_category
- brand_tier
- origin_score
- premium_origin_flag
- rating_segment
- company_region
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# ─── Premium Flag ─────────────────────────────────────────────────────────────

def add_premium_flag(df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    Flag chocolates with rating >= threshold as 'premium'.

    Args:
        df: DataFrame with 'rating' column
        threshold: Minimum rating to be considered premium

    Returns:
        DataFrame with 'premium_flag' boolean column
    """
    df = df.copy()
    df["premium_flag"] = df["rating"] >= threshold
    pct = df["premium_flag"].mean() * 100
    print(f"[add_premium_flag] {pct:.1f}% of products flagged as premium (rating >= {threshold})")
    return df


# ─── Cocoa Category ───────────────────────────────────────────────────────────

def add_cocoa_category(
    df: pd.DataFrame,
    low_threshold: float = 60.0,
    high_threshold: float = 75.0,
) -> pd.DataFrame:
    """
    Bin cocoa percentage into Low / Medium / High categories.

    Args:
        df: DataFrame with 'cocoa_percent_clean'
        low_threshold: Upper bound for 'Low' category
        high_threshold: Lower bound for 'High' category

    Returns:
        DataFrame with 'cocoa_category' column
    """
    df = df.copy()

    def categorize(pct: float) -> str:
        if pct < low_threshold:
            return "Low (<60%)"
        elif pct <= high_threshold:
            return "Medium (60-75%)"
        else:
            return "High (>75%)"

    df["cocoa_category"] = df["cocoa_percent_clean"].apply(categorize)
    print(f"[add_cocoa_category] Distribution:\n{df['cocoa_category'].value_counts().to_string()}")
    return df


# ─── Brand Tier ───────────────────────────────────────────────────────────────

def add_brand_tier(df: pd.DataFrame, top_quantile: float = 0.75) -> pd.DataFrame:
    """
    Assign brand tier based on the brand's average rating.

    Tiers:
    - Elite: avg rating >= 75th percentile of brand averages
    - Standard: below 75th percentile
    - Low-Rated: avg rating < 3.0

    Args:
        df: DataFrame with 'company' and 'rating' columns
        top_quantile: Quantile cutoff for 'Elite' tier

    Returns:
        DataFrame with 'brand_avg_rating' and 'brand_tier' columns
    """
    df = df.copy()
    brand_avg = df.groupby("company")["rating"].mean().rename("brand_avg_rating")
    df = df.merge(brand_avg, on="company", how="left")

    cutoff = df["brand_avg_rating"].quantile(top_quantile)

    def assign_tier(avg: float) -> str:
        if avg >= cutoff:
            return "Elite"
        elif avg >= 3.0:
            return "Standard"
        else:
            return "Low-Rated"

    df["brand_tier"] = df["brand_avg_rating"].apply(assign_tier)
    print(f"[add_brand_tier] Elite cutoff: {cutoff:.2f}")
    print(f"[add_brand_tier] Distribution:\n{df['brand_tier'].value_counts().to_string()}")
    return df


# ─── Origin Score ─────────────────────────────────────────────────────────────

def add_origin_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a score (0-100) for each bean origin based on average rating.
    Normalized to the range [0, 100].

    Args:
        df: DataFrame with 'broad_bean_origin' and 'rating'

    Returns:
        DataFrame with 'origin_avg_rating' and 'origin_score' columns
    """
    df = df.copy()
    origin_avg = df.groupby("broad_bean_origin")["rating"].mean().rename("origin_avg_rating")
    df = df.merge(origin_avg, on="broad_bean_origin", how="left")

    min_r = df["origin_avg_rating"].min()
    max_r = df["origin_avg_rating"].max()
    df["origin_score"] = (
        (df["origin_avg_rating"] - min_r) / (max_r - min_r) * 100
    ).round(1)

    return df


# ─── Premium Origin Flag ──────────────────────────────────────────────────────

def add_premium_origin_flag(
    df: pd.DataFrame,
    top_quantile: float = 0.75,
) -> pd.DataFrame:
    """
    Flag origins that consistently produce premium-rated chocolates.
    An origin is 'premium' if its average rating >= top_quantile of all origin averages.

    Args:
        df: DataFrame with 'origin_avg_rating' column
        top_quantile: Quantile threshold

    Returns:
        DataFrame with 'premium_origin_flag' boolean column
    """
    df = df.copy()
    if "origin_avg_rating" not in df.columns:
        raise ValueError("Run add_origin_score() first to get origin_avg_rating")

    cutoff = df["origin_avg_rating"].quantile(top_quantile)
    df["premium_origin_flag"] = df["origin_avg_rating"] >= cutoff
    n = df["premium_origin_flag"].sum()
    pct = n / len(df) * 100
    print(f"[add_premium_origin_flag] {n} rows ({pct:.1f}%) from premium origins (cutoff: {cutoff:.2f})")
    return df


# ─── Rating Segment ───────────────────────────────────────────────────────────

RATING_SEGMENTS = {
    (1.0, 2.0): "Poor",
    (2.0, 2.75): "Below Average",
    (2.75, 3.25): "Average",
    (3.25, 3.75): "Good",
    (3.75, 5.0): "Outstanding",
}


def add_rating_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a qualitative label based on rating value.

    Args:
        df: DataFrame with 'rating' column

    Returns:
        DataFrame with 'rating_segment' column
    """
    df = df.copy()

    def segment(r: float) -> str:
        if r < 2.0:
            return "Poor"
        elif r < 2.75:
            return "Below Average"
        elif r < 3.25:
            return "Average"
        elif r < 3.75:
            return "Good"
        else:
            return "Outstanding"

    df["rating_segment"] = df["rating"].apply(segment)
    print(f"[add_rating_segment] Distribution:\n{df['rating_segment'].value_counts().to_string()}")
    return df


# ─── Company Region ───────────────────────────────────────────────────────────

REGION_MAP = {
    "USA": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "United Kingdom": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Switzerland": "Europe",
    "Italy": "Europe",
    "Belgium": "Europe",
    "Netherlands": "Europe",
    "Spain": "Europe",
    "Austria": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Hungary": "Europe",
    "Ireland": "Europe",
    "Lithuania": "Europe",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Scotland": "Europe",
    "Sweden": "Europe",
    "Wales": "Europe",
    "Czech Republic": "Europe",
    "Amsterdam": "Europe",
    "Brazil": "South America",
    "Colombia": "South America",
    "Ecuador": "South America",
    "Peru": "South America",
    "Venezuela": "South America",
    "Chile": "South America",
    "Bolivia": "South America",
    "Argentina": "South America",
    "Grenada": "Caribbean",
    "Jamaica": "Caribbean",
    "Trinidad": "Caribbean",
    "St. Lucia": "Caribbean",
    "Martinique": "Caribbean",
    "Belize": "Central America",
    "Costa Rica": "Central America",
    "El Salvador": "Central America",
    "Guatemala": "Central America",
    "Honduras": "Central America",
    "Nicaragua": "Central America",
    "Panama": "Central America",
    "Japan": "Asia-Pacific",
    "Australia": "Asia-Pacific",
    "New Zealand": "Asia-Pacific",
    "Philippines": "Asia-Pacific",
    "South Korea": "Asia-Pacific",
    "Taiwan": "Asia-Pacific",
    "Vietnam": "Asia-Pacific",
    "Israel": "Middle East",
    "Ghana": "Africa",
    "Madagascar": "Africa",
    "South Africa": "Africa",
    "Sao Tome": "Africa",
}


def add_company_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map company_location to a broader geographic region.

    Args:
        df: DataFrame with 'company_location' column

    Returns:
        DataFrame with 'company_region' column
    """
    df = df.copy()
    df["company_region"] = df["company_location"].map(REGION_MAP).fillna("Other")
    print(f"[add_company_region] Distribution:\n{df['company_region'].value_counts().to_string()}")
    return df


# ─── Master Pipeline ──────────────────────────────────────────────────────────

def run_feature_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    premium_threshold: float = 3.5,
    low_cocoa: float = 60.0,
    high_cocoa: float = 75.0,
) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.

    Args:
        input_path: Path to cleaned CSV
        output_path: Optional save path
        premium_threshold: Rating cutoff for premium flag
        low_cocoa: Low cocoa % boundary
        high_cocoa: High cocoa % boundary

    Returns:
        DataFrame with all engineered features
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    df = pd.read_csv(input_path)
    print(f"[load] {len(df)} rows")

    df = add_premium_flag(df, threshold=premium_threshold)
    df = add_cocoa_category(df, low_threshold=low_cocoa, high_threshold=high_cocoa)
    df = add_brand_tier(df)
    df = add_origin_score(df)
    df = add_premium_origin_flag(df)
    df = add_rating_segment(df)
    df = add_company_region(df)

    print(f"\n[done] Shape: {df.shape}, Columns: {df.columns.tolist()}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[save] Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = run_feature_pipeline(
        input_path="data/processed/chocolate_clean.csv",
        output_path="data/processed/chocolate_featured.csv",
    )
    print(df[["company", "rating", "premium_flag", "cocoa_category", "brand_tier", "rating_segment", "company_region"]].head())
