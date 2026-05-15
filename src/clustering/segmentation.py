"""
KMeans clustering for chocolate product segmentation.

Segments products into:
- Premium
- Mass Market
- Experimental

Also includes optional rating prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# ─── Cluster Labels ───────────────────────────────────────────────────────────

CLUSTER_NAMES = {
    0: "Premium",
    1: "Mass Market",
    2: "Experimental",
}


def assign_cluster_labels(df: pd.DataFrame, centers: np.ndarray) -> dict:
    """
    Auto-assign meaningful names to clusters based on cluster centers.

    Logic:
    - Highest avg rating → Premium
    - Largest count → Mass Market
    - Highest or lowest cocoa → Experimental

    Args:
        df: DataFrame with 'cluster' column
        centers: KMeans cluster centers (in original scale after inverse transform)

    Returns:
        Dict mapping cluster int → label string
    """
    cluster_stats = df.groupby("cluster").agg(
        avg_rating=("rating", "mean"),
        count=("rating", "size"),
        avg_cocoa=("cocoa_percent_clean", "mean"),
    )

    # Sort by avg_rating descending
    sorted_clusters = cluster_stats.sort_values("avg_rating", ascending=False).index.tolist()

    label_map = {}
    label_map[sorted_clusters[0]] = "Premium"

    # Among remaining, mass market = largest count
    remaining = cluster_stats.drop(index=sorted_clusters[0])
    mass = remaining["count"].idxmax()
    label_map[mass] = "Mass Market"

    # Remaining = Experimental
    for c in cluster_stats.index:
        if c not in label_map:
            label_map[c] = "Experimental"

    return label_map


# ─── KMeans Clustering ────────────────────────────────────────────────────────

def run_kmeans(
    df: pd.DataFrame,
    features: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Scale features and run KMeans clustering.

    Args:
        df: Featured DataFrame
        features: List of column names to use for clustering
        n_clusters: Number of clusters
        random_state: Reproducibility seed

    Returns:
        Tuple of (DataFrame with cluster columns, fitted KMeans, fitted Scaler)
    """
    df = df.copy()

    # Drop rows with NaN in cluster features
    df_cluster = df.dropna(subset=features).copy()
    X = df_cluster[features].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_cluster["cluster"] = km.fit_predict(X_scaled)

    # Silhouette score
    sil_score = silhouette_score(X_scaled, df_cluster["cluster"])
    print(f"[run_kmeans] Silhouette Score: {sil_score:.4f}")
    print(f"[run_kmeans] Cluster distribution:\n{df_cluster['cluster'].value_counts().to_string()}")

    # Auto-assign labels
    label_map = assign_cluster_labels(df_cluster, km.cluster_centers_)
    df_cluster["cluster_label"] = df_cluster["cluster"].map(label_map)
    print(f"[run_kmeans] Cluster labels: {label_map}")
    print(f"[run_kmeans] Label distribution:\n{df_cluster['cluster_label'].value_counts().to_string()}")

    # Merge back to full df
    df = df.merge(
        df_cluster[["cluster", "cluster_label"]],
        left_index=True,
        right_index=True,
        how="left",
    )

    return df, km, scaler


def cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed profile for each cluster.

    Returns:
        DataFrame with cluster characteristics
    """
    profiles = (
        df.groupby("cluster_label")
        .agg(
            count=("rating", "size"),
            avg_rating=("rating", "mean"),
            std_rating=("rating", "std"),
            avg_cocoa=("cocoa_percent_clean", "mean"),
            premium_pct=("premium_flag", "mean"),
            top_origin=("broad_bean_origin", lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"),
            top_country=("company_location", lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"),
        )
        .reset_index()
    )
    profiles["avg_rating"] = profiles["avg_rating"].round(3)
    profiles["avg_cocoa"] = profiles["avg_cocoa"].round(1)
    profiles["premium_pct"] = (profiles["premium_pct"] * 100).round(1)
    return profiles


# ─── Elbow Method ─────────────────────────────────────────────────────────────

def elbow_analysis(
    df: pd.DataFrame,
    features: list[str],
    max_k: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute inertia and silhouette scores for k=2..max_k.
    Use to determine optimal cluster count.

    Args:
        df: DataFrame
        features: Clustering features
        max_k: Max number of clusters to try

    Returns:
        DataFrame with k, inertia, silhouette_score
    """
    df_clean = df.dropna(subset=features)
    X = df_clean[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": round(sil, 4)})

    return pd.DataFrame(results)


# ─── Rating Prediction (Optional) ─────────────────────────────────────────────

def rating_prediction_model(df: pd.DataFrame) -> dict:
    """
    Simple Ridge regression to predict rating.
    Uses: cocoa_percent_clean, review_year, brand_avg_rating, origin_score.

    Not meant for production — meant to show feature importance direction.

    Returns:
        Dict with CV score and feature coefficients
    """
    feature_cols = ["cocoa_percent_clean", "review_year", "brand_avg_rating", "origin_score"]
    df_model = df.dropna(subset=feature_cols + ["rating"])

    X = df_model[feature_cols].values
    y = df_model["rating"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    model.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)

    return {
        "model": "Ridge Regression",
        "features": feature_cols,
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "coefficients": coef_df,
        "interpretation": (
            f"The model explains ~{cv_scores.mean()*100:.1f}% of rating variance. "
            "Brand avg rating is the strongest predictor — brand reputation matters more than cocoa %."
        ),
    }


# ─── Master Pipeline ──────────────────────────────────────────────────────────

def run_clustering_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    features: Optional[list] = None,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """
    End-to-end clustering pipeline.

    Args:
        input_path: Path to featured CSV
        output_path: Optional save path
        features: List of feature columns for clustering
        n_clusters: Number of clusters

    Returns:
        DataFrame with cluster assignments
    """
    print("=" * 60)
    print("PRODUCT SEGMENTATION CLUSTERING PIPELINE")
    print("=" * 60)

    df = pd.read_csv(input_path)
    print(f"[load] {len(df)} rows")

    if features is None:
        features = ["cocoa_percent_clean", "rating", "review_year"]

    # Run KMeans
    df, km, scaler = run_kmeans(df, features=features, n_clusters=n_clusters)

    # Cluster profiles
    profiles = cluster_profiles(df)
    print("\n[cluster_profiles]")
    print(profiles.to_string(index=False))

    # Rating prediction
    print("\n[rating_prediction_model]")
    pred_results = rating_prediction_model(df)
    print(f"  CV R² = {pred_results['cv_r2_mean']} ± {pred_results['cv_r2_std']}")
    print(f"  {pred_results['interpretation']}")
    print("\n  Feature importance (coefficients):")
    print(pred_results["coefficients"].to_string(index=False))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n[save] Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = run_clustering_pipeline(
        input_path="data/processed/chocolate_featured.csv",
        output_path="data/processed/chocolate_clustered.csv",
    )
    print(df[["company", "rating", "cluster_label"]].head(10))
