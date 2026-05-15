"""
Data cleaning module for the Chocolate Ratings Dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

COLUMN_MAP = {
    "Company \n(Maker-if known)": "company",
    "Specific Bean Origin\nor Bar Name": "bean_origin_bar",
    "REF": "ref",
    "Review\nDate": "review_year",
    "Cocoa\nPercent": "cocoa_percent_raw",
    "Company\nLocation": "company_location",
    "Rating": "rating",
    "Bean\nType": "bean_type",
    "Broad Bean\nOrigin": "broad_bean_origin",
}

COUNTRY_ALIASES = {
    "U.S.A.": "USA",
    "United States of America": "USA",
    "United States": "USA",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "Sao Tome": "Sao Tome",
    "Congo": "DR Congo",
    "Ivory Coast": "Cote d Ivoire",
    "Niacragua": "Nicaragua",
    "Domincan Republic": "Dominican Republic",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_MAP)


def parse_cocoa_percent(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype(float)
    )


def validate_cocoa_percent(df, col="cocoa_percent_clean", min_val=40.0, max_val=100.0):
    before = len(df)
    df = df[(df[col] >= min_val) & (df[col] <= max_val)].copy()
    removed = before - len(df)
    if removed:
        print(f"[validate_cocoa_percent] Removed {removed} rows outside [{min_val},{max_val}]")
    return df


def validate_ratings(df, col="rating", min_val=1.0, max_val=5.0):
    before = len(df)
    df = df[(df[col] >= min_val) & (df[col] <= max_val)].copy()
    removed = before - len(df)
    if removed:
        print(f"[validate_ratings] Removed {removed} rows with invalid ratings")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bean_type"] = df["bean_type"].fillna("Unknown").str.strip()
    df["broad_bean_origin"] = df["broad_bean_origin"].fillna("Unknown").str.strip()
    core_cols = ["company", "rating", "cocoa_percent_clean", "company_location"]
    before = len(df)
    df = df.dropna(subset=core_cols)
    removed = before - len(df)
    if removed:
        print(f"[handle_missing_values] Dropped {removed} rows with missing core fields")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().copy()
    removed = before - len(df)
    print(f"[remove_duplicates] Removed {removed} fully duplicate rows")
    return df


def standardize_country(series: pd.Series) -> pd.Series:
    return series.str.strip().replace(COUNTRY_ALIASES)


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["company", "bean_type", "broad_bean_origin", "bean_origin_bar"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df


def flag_outliers_iqr(df, col, multiplier=3.0):
    df = df.copy()
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    df[f"{col}_outlier"] = ~df[col].between(lower, upper)
    n = df[f"{col}_outlier"].sum()
    print(f"[flag_outliers_iqr] {n} outliers in {col} (range: {lower:.1f}-{upper:.1f})")
    return df


def run_cleaning_pipeline(
    raw_path: str,
    output_path: Optional[str] = None,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
    cocoa_min: float = 40.0,
    cocoa_max: float = 100.0,
) -> pd.DataFrame:
    print("=" * 60)
    print("CHOCOLATE DATA CLEANING PIPELINE")
    print("=" * 60)

    df = pd.read_csv(raw_path)
    print(f"[load] {len(df)} rows, {len(df.columns)} columns")

    df = rename_columns(df)
    df["cocoa_percent_clean"] = parse_cocoa_percent(df["cocoa_percent_raw"])
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = validate_cocoa_percent(df, min_val=cocoa_min, max_val=cocoa_max)
    df = validate_ratings(df, min_val=rating_min, max_val=rating_max)
    df["company_location"] = standardize_country(df["company_location"])
    df["broad_bean_origin"] = standardize_country(df["broad_bean_origin"])
    df = normalize_text_columns(df)
    df = flag_outliers_iqr(df, "rating", multiplier=3.0)
    df = flag_outliers_iqr(df, "cocoa_percent_clean", multiplier=3.0)
    df = df.drop(columns=["cocoa_percent_raw"], errors="ignore")

    print(f"\n[done] Final shape: {df.shape}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[save] Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = run_cleaning_pipeline(
        raw_path="data/raw/chocolate_ratings.csv",
        output_path="data/processed/chocolate_clean.csv",
    )
    print(df.dtypes)
    print(df.head(3))
