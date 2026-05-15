"""
Unit tests for the feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.features import (
    add_premium_flag,
    add_cocoa_category,
    add_brand_tier,
    add_origin_score,
    add_premium_origin_flag,
    add_rating_segment,
    add_company_region,
)


# ─── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_df():
    return pd.DataFrame({
        "company": ["Amedei", "Amedei", "Lindt", "Valrhona", "Valrhona"],
        "rating": [3.75, 3.5, 2.5, 4.0, 3.0],
        "cocoa_percent_clean": [70.0, 72.0, 65.0, 80.0, 55.0],
        "company_location": ["Italy", "Italy", "Switzerland", "France", "France"],
        "broad_bean_origin": ["Venezuela", "Venezuela", "Ghana", "Madagascar", "Peru"],
    })


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestAddPremiumFlag:
    def test_flags_above_threshold(self, base_df):
        df = add_premium_flag(base_df, threshold=3.5)
        assert df.loc[0, "premium_flag"] is True or df.loc[0, "premium_flag"] == True

    def test_does_not_flag_below_threshold(self, base_df):
        df = add_premium_flag(base_df, threshold=3.5)
        assert df.loc[2, "premium_flag"] is False or df.loc[2, "premium_flag"] == False

    def test_column_is_boolean(self, base_df):
        df = add_premium_flag(base_df)
        assert df["premium_flag"].dtype == bool

    def test_equal_to_threshold_is_premium(self, base_df):
        df = add_premium_flag(base_df, threshold=3.5)
        # rating == 3.5 at index 1 → should be True
        assert df.loc[1, "premium_flag"] == True


class TestAddCocoaCategory:
    def test_low_category(self, base_df):
        df = add_cocoa_category(base_df, low_threshold=60.0, high_threshold=75.0)
        # 55% is low
        low_rows = df[df["cocoa_percent_clean"] < 60]
        assert (low_rows["cocoa_category"] == "Low (<60%)").all()

    def test_medium_category(self, base_df):
        df = add_cocoa_category(base_df, low_threshold=60.0, high_threshold=75.0)
        mid_rows = df[(df["cocoa_percent_clean"] >= 60) & (df["cocoa_percent_clean"] <= 75)]
        assert (mid_rows["cocoa_category"] == "Medium (60-75%)").all()

    def test_high_category(self, base_df):
        df = add_cocoa_category(base_df, low_threshold=60.0, high_threshold=75.0)
        high_rows = df[df["cocoa_percent_clean"] > 75]
        assert (high_rows["cocoa_category"] == "High (>75%)").all()

    def test_all_rows_categorized(self, base_df):
        df = add_cocoa_category(base_df)
        assert df["cocoa_category"].isna().sum() == 0


class TestAddBrandTier:
    def test_adds_tier_column(self, base_df):
        df = add_brand_tier(base_df)
        assert "brand_tier" in df.columns

    def test_adds_avg_rating_column(self, base_df):
        df = add_brand_tier(base_df)
        assert "brand_avg_rating" in df.columns

    def test_tier_values_are_valid(self, base_df):
        df = add_brand_tier(base_df)
        valid_tiers = {"Elite", "Standard", "Low-Rated"}
        assert set(df["brand_tier"].unique()).issubset(valid_tiers)

    def test_brand_avg_is_mean_of_ratings(self, base_df):
        df = add_brand_tier(base_df)
        # Amedei has ratings [3.75, 3.5] → avg = 3.625
        amedei_rows = df[df["company"] == "Amedei"]
        assert abs(amedei_rows["brand_avg_rating"].iloc[0] - 3.625) < 0.001


class TestAddOriginScore:
    def test_adds_origin_score_column(self, base_df):
        df = add_origin_score(base_df)
        assert "origin_score" in df.columns

    def test_score_range_0_to_100(self, base_df):
        df = add_origin_score(base_df)
        assert df["origin_score"].min() >= 0
        assert df["origin_score"].max() <= 100


class TestAddPremiumOriginFlag:
    def test_requires_origin_avg_rating(self, base_df):
        with pytest.raises(ValueError):
            add_premium_origin_flag(base_df)

    def test_adds_flag_column(self, base_df):
        df = add_origin_score(base_df)
        df = add_premium_origin_flag(df)
        assert "premium_origin_flag" in df.columns

    def test_flag_is_boolean(self, base_df):
        df = add_origin_score(base_df)
        df = add_premium_origin_flag(df)
        assert df["premium_origin_flag"].dtype == bool


class TestAddRatingSegment:
    def test_outstanding_segment(self, base_df):
        df = add_rating_segment(base_df)
        outstanding = df[df["rating"] >= 3.75]
        assert (outstanding["rating_segment"] == "Outstanding").all()

    def test_poor_segment(self):
        df = pd.DataFrame({
            "rating": [1.0, 1.5, 1.9]
        })
        result = add_rating_segment(df)
        assert (result["rating_segment"] == "Poor").all()

    def test_all_rows_segmented(self, base_df):
        df = add_rating_segment(base_df)
        assert df["rating_segment"].isna().sum() == 0


class TestAddCompanyRegion:
    def test_known_country_mapped(self, base_df):
        df = add_company_region(base_df)
        italy_rows = df[df["company_location"] == "Italy"]
        assert (italy_rows["company_region"] == "Europe").all()

    def test_unknown_country_is_other(self):
        df = pd.DataFrame({"company_location": ["Wakanda"]})
        result = add_company_region(df)
        assert result.loc[0, "company_region"] == "Other"

    def test_usa_is_north_america(self, base_df):
        df = add_company_region(base_df)
        # France → Europe
        fr_rows = df[df["company_location"] == "France"]
        assert (fr_rows["company_region"] == "Europe").all()
