"""
Unit tests for the data cleaning module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cleaning.cleaner import (
    parse_cocoa_percent,
    validate_cocoa_percent,
    validate_ratings,
    handle_missing_values,
    remove_duplicates,
    standardize_country,
    normalize_text_columns,
    flag_outliers_iqr,
    rename_columns,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame simulating cleaned column structure."""
    return pd.DataFrame({
        "company": ["Amedei", "Valrhona", None, "Lindt"],
        "bean_origin_bar": ["Toscano", "Guanaja", "Peru", "Madagascar"],
        "ref": [1, 2, 3, 4],
        "review_year": [2015, 2016, 2017, 2018],
        "cocoa_percent_raw": ["70%", "65%", "85%", "55%"],
        "company_location": ["Italy", "France", "USA", "Switzerland"],
        "rating": [3.75, 3.5, 2.0, 4.0],
        "bean_type": ["Trinitario", None, "Forastero", "Criollo"],
        "broad_bean_origin": ["Venezuela", "Ghana", None, "Madagascar"],
    })


@pytest.fixture
def sample_df_with_cocoa(sample_df):
    df = sample_df.copy()
    df["cocoa_percent_clean"] = parse_cocoa_percent(df["cocoa_percent_raw"])
    return df


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestParseCocoaPercent:
    def test_strips_percent_sign(self):
        s = pd.Series(["70%", "65%", "85%"])
        result = parse_cocoa_percent(s)
        assert list(result) == [70.0, 65.0, 85.0]

    def test_handles_no_percent_sign(self):
        s = pd.Series(["70", "65"])
        result = parse_cocoa_percent(s)
        assert list(result) == [70.0, 65.0]

    def test_returns_float(self):
        s = pd.Series(["72%"])
        result = parse_cocoa_percent(s)
        assert result.dtype == float


class TestValidateCocoaPercent:
    def test_removes_outlier_values(self, sample_df_with_cocoa):
        df = validate_cocoa_percent(sample_df_with_cocoa, min_val=60.0, max_val=100.0)
        assert df["cocoa_percent_clean"].min() >= 60.0
        assert df["cocoa_percent_clean"].max() <= 100.0

    def test_keeps_valid_rows(self, sample_df_with_cocoa):
        df = validate_cocoa_percent(sample_df_with_cocoa, min_val=50.0, max_val=100.0)
        # 55% is >= 50, so all 4 rows should remain
        assert len(df) == 4


class TestValidateRatings:
    def test_removes_invalid_ratings(self):
        df = pd.DataFrame({"rating": [0.5, 1.0, 3.5, 5.0, 6.0]})
        result = validate_ratings(df, min_val=1.0, max_val=5.0)
        assert result["rating"].min() >= 1.0
        assert result["rating"].max() <= 5.0
        assert len(result) == 3

    def test_no_removal_when_all_valid(self):
        df = pd.DataFrame({"rating": [1.0, 2.5, 3.5, 5.0]})
        result = validate_ratings(df)
        assert len(result) == 4


class TestHandleMissingValues:
    def test_fills_bean_type(self, sample_df_with_cocoa):
        df = handle_missing_values(sample_df_with_cocoa)
        assert df["bean_type"].isna().sum() == 0

    def test_fills_broad_bean_origin(self, sample_df_with_cocoa):
        df = handle_missing_values(sample_df_with_cocoa)
        assert df["broad_bean_origin"].isna().sum() == 0

    def test_drops_rows_missing_company(self, sample_df_with_cocoa):
        # row with None company should be dropped
        df = handle_missing_values(sample_df_with_cocoa)
        assert df["company"].isna().sum() == 0
        assert len(df) == 3  # 1 row had None company


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({
            "a": [1, 1, 2],
            "b": ["x", "x", "y"],
        })
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_no_change_when_no_duplicates(self, sample_df):
        result = remove_duplicates(sample_df)
        assert len(result) == len(sample_df)


class TestStandardizeCountry:
    def test_replaces_aliases(self):
        s = pd.Series(["U.S.A.", "UK", "France"])
        result = standardize_country(s)
        assert result.tolist() == ["USA", "United Kingdom", "France"]

    def test_passes_through_unknown(self):
        s = pd.Series(["Wakanda"])
        result = standardize_country(s)
        assert result.tolist() == ["Wakanda"]


class TestNormalizeTextColumns:
    def test_strips_whitespace(self, sample_df_with_cocoa):
        df = sample_df_with_cocoa.copy()
        df.loc[0, "company"] = "  Amedei  "
        result = normalize_text_columns(df)
        assert result.loc[0, "company"] == "Amedei"

    def test_title_cases(self, sample_df_with_cocoa):
        df = sample_df_with_cocoa.copy()
        df.loc[0, "company"] = "amedei"
        result = normalize_text_columns(df)
        assert result.loc[0, "company"] == "Amedei"


class TestFlagOutliersIQR:
    def test_flags_extreme_values(self):
        df = pd.DataFrame({"val": [3.0, 3.1, 3.2, 3.1, 3.0, 100.0]})
        result = flag_outliers_iqr(df, "val", multiplier=1.5)
        # 100.0 should be flagged
        assert result.loc[5, "val_outlier"] is True or result.loc[5, "val_outlier"] == True

    def test_normal_values_not_flagged(self):
        df = pd.DataFrame({"val": [3.0, 3.1, 3.2, 3.1, 3.0, 3.05]})
        result = flag_outliers_iqr(df, "val", multiplier=3.0)
        assert result["val_outlier"].sum() == 0

    def test_adds_correct_column_name(self):
        df = pd.DataFrame({"rating": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = flag_outliers_iqr(df, "rating")
        assert "rating_outlier" in result.columns
