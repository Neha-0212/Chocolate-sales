## Live project link - https://chocolate-sales-zwhtdbycd9h7gnnuy9agro.streamlit.app/
# What Makes a Premium Chocolate?
### Product Quality, Pricing & Market Strategy Analysis

---

## Project Overview

A **production-ready end-to-end data analytics project** that analyzes 1,795 expert chocolate ratings to answer: *What makes a chocolate bar score premium?*

Built as a **Senior Product Analyst** at a fictional premium chocolate marketplace, this project delivers:

- Actionable product insights
- Data-backed new product recommendation
- Interactive Streamlit dashboard
- Modular, tested, production-quality Python codebase

---

## Business Problem

Premium chocolate is a $15B+ global market with significant quality variance. As a product analyst, the goal is to:

1. Identify what drives higher chocolate ratings
2. Discover characteristics of premium chocolates
3. Recommend a new premium product with specific attributes
4. Optimize cocoa percentage strategy
5. Analyze top-performing brands and origins
6. Segment products into market categories
7. Find underserved market opportunities

---

## Dataset

**Source:** FHSS Flavors of Cacao Chocolate Bar Ratings

| Field | Description |
|-------|-------------|
| Company | Chocolate maker/brand |
| Bean Origin | Where the cocoa beans came from |
| REF | Reviewer/company identifier |
| Review Year | Year of review (2006–2017) |
| Cocoa % | Cocoa percentage of the bar |
| Company Location | Country where the maker is based |
| Rating | Expert rating (1.0–5.0 scale) |
| Bean Type | Variety of cacao bean |
| Broad Bean Origin | General origin region |

**Size:** 1,795 reviews | 416 brands | 100 origins | 60 countries

---

## Key Insights

- **Only 39% of products are 'premium'** (rating ≥ 3.5) — the market is quality-constrained
- **Higher cocoa % does NOT mean better ratings** (r = -0.165, significant). The 68-72% range is the sweet spot.
- **Brand reputation is the #1 predictor** of product quality (explains ~45% of rating variance)
- **Italy, Belgium, Switzerland** produce the highest-rated chocolates on average
- **Venezuela, Haiti, Madagascar** are the top-rated bean origins
- **Trinitario** is the highest-rated bean variety
- **Underserved segments:** Haiti × Medium cocoa, Honduras × Medium cocoa — high quality, low competition

---

## Folder Structure

```
premium-chocolate-analysis/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned + featured + clustered CSVs
│
├── src/
│   ├── cleaning/
│   │   └── cleaner.py          # Full data cleaning pipeline
│   ├── feature_engineering/
│   │   └── features.py         # Feature creation pipelines
│   ├── analysis/
│   │   └── eda.py              # EDA, KPIs, statistical tests
│   ├── clustering/
│   │   └── segmentation.py     # KMeans clustering + rating prediction
│   └── utils/
│       ├── logger.py           # Centralized logging
│       └── config_loader.py    # YAML config loader
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard (8 pages)
│
├── sql/
│   ├── kpi_queries.sql         # Standard KPI SQL queries
│   └── premium_analysis.sql    # Deep premium segment SQL
│
├── tests/
│   ├── test_cleaning.py        # 19 unit tests for cleaning
│   └── test_features.py        # 23 unit tests for features
│
├── config/
│   └── config.yaml             # Central project configuration
│
├── outputs/
│   ├── charts/                 # Generated chart files
│   └── reports/                # Generated report files
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Neha-0212/premium-chocolate-analysis.git
cd premium-chocolate-analysis

# 2. Create virtual environment
python3 -m venv venv
Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the raw data file
# Copy chocolate_ratings.csv to data/raw/

# 5. Run the full pipeline
python3 src/cleaning/cleaner.py
python3 src/feature_engineering/features.py
python3 src/clustering/segmentation.py

# 6. Launch the dashboard
streamlit run dashboard/app.py
```

---

## Running Tests

```bash
# Run all 42 unit tests
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ -v --cov=src
```

**Test results:** 42/42 passing 

---

## Dashboard

The Streamlit dashboard includes 8 interactive pages:

| Page | Description |
|------|-------------|
| Overview & KPIs | Top-level metrics, rating distribution |
| Rating Analysis | Year trends, regional comparison, ANOVA |
| Cocoa Analysis | Cocoa % vs rating scatter, optimal range |
| Brand Performance | Top brands, tier analysis, downloadable table |
| Origin & Geography | Origin map, country analysis, white space finder |
| Product Clusters | KMeans segments: Premium / Mass Market / Experimental |
| Best Product Formula | Data-backed ideal product specification |
| Business Recommendations | GTM strategy, pricing, geographic expansion |

### Features
- Dynamic filters: year, region, cocoa category, premium-only
- Interactive Plotly charts
- Statistical test results with business interpretation
- Downloadable CSV exports

---

## Statistical Analysis

| Test | Factor | Result |
|------|--------|--------|
| Pearson Correlation | Cocoa % vs Rating | r = -0.165 (weak negative, significant) |
| One-way ANOVA | Cocoa Category vs Rating | F significant (p < 0.001) |
| One-way ANOVA | Company Region vs Rating | F significant (p = 0.0006) |
| Ridge Regression | Rating Prediction | R² ≈ 0.45 (cross-validated) |

---

## Machine Learning

**KMeans Clustering (k=3):**
- **Premium:** Higher avg rating (~3.3), moderate cocoa (~70%), 48% premium products
- **Mass Market:** Middle tier, 68% avg cocoa, broad geography
- **Experimental:** Highest cocoa (~79%), lowest ratings (~2.7), niche audience

**Rating Prediction (Ridge Regression):**
- Brand avg rating → strongest predictor
- Origin score → moderate impact
- Cocoa % → slight negative impact
- Review year → minimal impact

---

## Best Product Recommendation

| Attribute | Recommendation |
|-----------|---------------|
| Cocoa % | 68–72% |
| Bean Type | Trinitario |
| Origin | Venezuela or Haiti |
| Maker Location | Europe (Italy / Belgium / Switzerland) |
| Target Rating | ≥ 3.75 (Outstanding segment) |
| Retail Price | $14–22 per 70g bar |
| Channel | Specialty retail + D2C |

---

## Business Recommendations

1. **New Product Launch:** Medium-cocoa (68-72%), single-origin, Trinitario — target 75%+ premium rating
2. **Don't compete on darkness:** >75% cocoa consistently underperforms
3. **Build brand reputation relentlessly** — it's the #1 rating predictor
4. **Source from Haiti and Honduras** — high quality, low competition, first-mover opportunity
5. **Target Asia-Pacific as an expansion market** — under-represented as maker, strong as consumer
6. **Price at $14–22 for outstanding tier** — 2-3× premium vs commodity bars is justified by data

---

## Future Improvements

- Add NLP analysis on tasting notes for flavor profile clustering
- Build a rating prediction API (FastAPI) from the Ridge model
- Expand to live data from FlavorMap or Chocolate.org
- Add supplier cost modeling for margin analysis
- Add geospatial origin mapping with real lat/lng database
- Integrate with CRM for brand market share tracking

---


---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Stats | SciPy, Statsmodels |
| ML | Scikit-learn |
| Visualization | Plotly, Seaborn, Matplotlib |
| Dashboard | Streamlit |
| Config | PyYAML |
| Testing | Pytest |
| SQL | SQLite  |

---

## Author

Neha Kanaki

*"In chocolate, as in business: brand consistency beats individual ingredient optimization."*
