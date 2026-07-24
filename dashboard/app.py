"""
Premium Chocolate Analytics Dashboard
=====================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.eda import (
    compute_kpis,
    top_brands,
    top_origins,
    country_analysis,
    bean_type_analysis,
    cocoa_vs_rating,
    optimal_cocoa_range,
    rating_distribution,
    anova_cocoa_vs_rating,
    correlation_cocoa_rating,
    anova_region_vs_rating,
    find_underserved_segments,
    rating_by_segment,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Premium Chocolate Analytics",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #6b4226;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #d4a853;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .insight-box {
        background: #1e1e2e;
        border-left: 4px solid #d4a853;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #d4a853;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the fully processed and featured dataset."""
    clustered_path = ROOT / "data/processed/chocolate_clustered.csv"
    featured_path = ROOT / "data/processed/chocolate_featured.csv"

    if clustered_path.exists():
        return pd.read_csv(clustered_path)
    elif featured_path.exists():
        return pd.read_csv(featured_path)
    else:
        st.error("Processed data not found. Run the pipeline first: `python3 src/cleaning/cleaner.py && python3 src/feature_engineering/features.py && python3 src/clustering/segmentation.py`")
        st.stop()


df_full = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────


st.sidebar.title(" Filters")

# Year filter
years = sorted(df_full["review_year"].unique())
selected_years = st.sidebar.select_slider(
    "Review Year Range",
    options=years,
    value=(min(years), max(years)),
)

# Region filter
regions = ["All"] + sorted(df_full["company_region"].unique().tolist())
selected_region = st.sidebar.selectbox("Company Region", regions)

# Cocoa category filter
cocoa_cats = ["All"] + df_full["cocoa_category"].unique().tolist()
selected_cocoa = st.sidebar.selectbox("Cocoa Category", cocoa_cats)

# Premium only toggle
premium_only = st.sidebar.checkbox("Premium Products Only (Rating ≥ 3.5)")

# Apply filters
df = df_full[
    (df_full["review_year"] >= selected_years[0]) &
    (df_full["review_year"] <= selected_years[1])
]
if selected_region != "All":
    df = df[df["company_region"] == selected_region]
if selected_cocoa != "All":
    df = df[df["cocoa_category"] == selected_cocoa]
if premium_only:
    df = df[df["premium_flag"] == True]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered records:** {len(df):,} / {len(df_full):,}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Premium Chocolate Analysis")
st.sidebar.markdown("**Dataset:** FHSS Chocolate Ratings (2006–2017)")
st.sidebar.markdown("**Built with:** Python · Streamlit · Plotly · Scikit-learn")

# ─── Navigation ───────────────────────────────────────────────────────────────

PAGES = [
    "Overview & KPIs",
    "Rating Analysis",
    "Cocoa Analysis",
    "Brand Performance",
    "Origin & Geography",
    "Product Clusters",
    "Best Product Formula",
    "Business Recommendations",
]

page = st.sidebar.radio("Navigate", PAGES)

# ─── Color Palette ────────────────────────────────────────────────────────────

COLORS = {
    "Premium": "#d4a853",
    "Mass Market": "#6b8e9f",
    "Experimental": "#9b6b9b",
    "Outstanding": "#2ecc71",
    "Good": "#27ae60",
    "Average": "#f39c12",
    "Below Average": "#e67e22",
    "Poor": "#e74c3c",
    "primary": "#d4a853",
    "secondary": "#6b4226",
    "bg": "#1a1a2e",
}

SEGMENT_COLOR_MAP = {
    "Outstanding": "#2ecc71",
    "Good": "#f1c40f",
    "Average": "#e67e22",
    "Below Average": "#e74c3c",
    "Poor": "#8e44ad",
}

CLUSTER_COLOR_MAP = {
    "Premium": "#d4a853",
    "Mass Market": "#6b8e9f",
    "Experimental": "#9b6b9b",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, delta: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {f'<div style="color:#2ecc71;font-size:0.8rem">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight-box"> {text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW & KPIs
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview & KPIs":
    st.title("🍫 Premium Chocolate Analytics")
    st.markdown("*What Makes a Premium Chocolate? — Product Quality, Pricing & Market Strategy Analysis*")
    st.markdown("---")

    kpis = compute_kpis(df)

    # KPI Row
    cols = st.columns(6)
    with cols[0]:
        kpi_card("Total Products", f"{kpis['total_products']:,}")
    with cols[1]:
        kpi_card("Brands", f"{kpis['total_brands']:,}")
    with cols[2]:
        kpi_card("Bean Origins", f"{kpis['total_origins']:,}")
    with cols[3]:
        kpi_card("Avg Rating", f"{kpis['avg_rating']}")
    with cols[4]:
        kpi_card("Premium Share", f"{kpis['premium_share_pct']}%")
    with cols[5]:
        kpi_card("Avg Cocoa %", f"{kpis['avg_cocoa_pct']}%")

    st.markdown("")

    # Two charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Rating Distribution")
        rd = rating_distribution(df)
        fig = px.bar(
            rd, x="rating", y="count",
            color="count",
            color_continuous_scale=["#6b4226", "#d4a853"],
            labels={"rating": "Rating", "count": "Products"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Premium vs Non-Premium")
        seg_counts = df["premium_flag"].value_counts().reset_index()
        seg_counts.columns = ["premium", "count"]
        seg_counts["label"] = seg_counts["premium"].map({True: "Premium (≥3.5)", False: "Standard (<3.5)"})

        fig = px.pie(
            seg_counts, values="count", names="label",
            color_discrete_sequence=["#d4a853", "#4a4a6a"],
            hole=0.45,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rating by segment bar
    st.markdown("#### Market Segments by Rating Quality")
    # Rating by segment bar
    # st.markdown("#### Market Segments by Rating Quality")
    seg_df = rating_by_segment(df)
    fig = px.bar(
        seg_df, x="rating_segment", y="count",
        color="rating_segment",
        color_discrete_map=SEGMENT_COLOR_MAP,
        text="count",
        labels={"count": "Products", "rating_segment": "Segment"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        showlegend=False,
        margin=dict(t=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    insight(
        f"Only {kpis['premium_share_pct']}% of products score 'premium' (≥3.5). "
        f"The market has room for quality improvement — most products cluster around 3.0-3.25."
    )
    


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RATING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Rating Analysis":
    st.title(" Rating Analysis")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Rating Distribution (Histogram)")
        fig = px.histogram(
            df, x="rating", nbins=20,
            color_discrete_sequence=["#d4a853"],
            labels={"rating": "Rating", "count": "Products"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            bargap=0.1,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Rating Statistics")
        stats_data = {
            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Q1", "Q3"],
            "Value": [
                f"{df['rating'].mean():.3f}",
                f"{df['rating'].median():.3f}",
                f"{df['rating'].std():.3f}",
                f"{df['rating'].min():.1f}",
                f"{df['rating'].max():.1f}",
                f"{df['rating'].quantile(0.25):.3f}",
                f"{df['rating'].quantile(0.75):.3f}",
            ],
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # Year trend
    st.markdown("#### Average Rating by Year")
    year_df = df.groupby("review_year").agg(
        avg_rating=("rating", "mean"),
        count=("rating", "size"),
        premium_pct=("premium_flag", "mean"),
    ).reset_index()
    year_df["premium_pct"] = (year_df["premium_pct"] * 100).round(1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=year_df["review_year"], y=year_df["avg_rating"],
        name="Avg Rating", line=dict(color="#d4a853", width=3),
        mode="lines+markers",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=year_df["review_year"], y=year_df["count"],
        name="Reviews", marker_color="#6b4226", opacity=0.5,
    ), secondary_y=True)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_yaxes(title_text="Avg Rating", secondary_y=False)
    fig.update_yaxes(title_text="Review Count", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Box plots by region
    st.markdown("#### Rating Distribution by Company Region")
    fig = px.box(
        df, x="company_region", y="rating",
        color="company_region",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"rating": "Rating", "company_region": "Region"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats test results
    st.markdown("#### Statistical Test: Does Region Affect Rating?")
    if df["company_region"].nunique() >= 2:
        region_test = anova_region_vs_rating(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("F-Statistic", region_test["f_statistic"])
        col2.metric("P-Value", region_test["p_value"])
        col3.metric("Significant?", " Yes" if region_test["significant"] else " No")
        insight(region_test["interpretation"])
    else:
        st.warning("Not enough regions in the filtered data to run ANOVA (need at least 2).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COCOA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Cocoa Analysis":
    st.title(" Cocoa Percentage Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average Rating by Cocoa Category")
        cocoa_df = cocoa_vs_rating(df)
        fig = px.bar(
            cocoa_df, x="cocoa_category", y="avg_rating",
            color="avg_rating",
            color_continuous_scale=["#6b4226", "#d4a853"],
            text="avg_rating",
            error_y="std_rating",
            labels={"avg_rating": "Avg Rating", "cocoa_category": "Cocoa Category"},
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Cocoa % Distribution")
        fig = px.histogram(
            df, x="cocoa_percent_clean", nbins=25,
            color_discrete_sequence=["#6b4226"],
            labels={"cocoa_percent_clean": "Cocoa %"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Rating vs Cocoa % (Scatter)")
    fig = px.scatter(
        df, x="cocoa_percent_clean", y="rating",
        color="cocoa_category",
        opacity=0.6,
        # trendline="ols",
        trendline_color_override="#d4a853",
        labels={"cocoa_percent_clean": "Cocoa %", "rating": "Rating"},
        color_discrete_sequence=["#d4a853", "#6b4226", "#9b6b9b"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Optimal Cocoa Range (10 bins)")
    opt_df = optimal_cocoa_range(df)
    fig = px.bar(
        opt_df, x="cocoa_bin", y="avg_rating",
        color="avg_rating",
        color_continuous_scale=["#6b4226", "#d4a853"],
        text="avg_rating",
        labels={"cocoa_bin": "Cocoa % Range", "avg_rating": "Avg Rating"},
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats
    st.markdown("#### Statistical Tests")
    corr_result = correlation_cocoa_rating(df)
    anova_result = anova_cocoa_vs_rating(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pearson Correlation: Cocoa % vs Rating**")
        st.metric("Correlation (r)", corr_result["correlation"])
        st.metric("P-Value", corr_result["p_value"])
        insight(corr_result["interpretation"])

    with col2:
        st.markdown("**ANOVA: Cocoa Category vs Rating**")
        st.metric("F-Statistic", anova_result["f_statistic"])
        st.metric("P-Value", anova_result["p_value"])
        insight(anova_result["interpretation"])


# ════
