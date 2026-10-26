# SmartDeal AI — E-Commerce Price Prediction & Smart Deal Finder

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit%201.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20Pipeline-F7931E.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Storage-SQLite3%20Parameterized-003B57.svg)](https://www.sqlite.org/)
[![Testing](https://img.shields.io/badge/Tests-Pytest%20(28%20Passed)-brightgreen.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Rigor Audit](https://img.shields.io/badge/Audit-Leak--Free%20Verified-0ea5e9.svg)](ml/evaluate.py)
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=kowshiii1208/smartdeal-ai&branch=main&mainModule=app.py)

An intelligent, production-style e-commerce decision intelligence platform developed for BTech CSE placement presentations, hackathons, and academic project reviews. **SmartDeal AI** protects consumers against artificial "fake discounts" by using Machine Learning to predict fair market reference prices, tracking historical price trends with moving averages, and scoring deals from 0 to 100 using a transparent, multi-factor mathematical formula.

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Cloud Deployment (1-Click)](#-cloud-deployment-1-click)
- [Machine Learning Methodology](#-machine-learning-methodology)
- [Deal Scoring Formula](#-transparent-deal-scoring-formula)
- [Navigation Pages Guide](#-navigation-pages-guide)
- [Technology Stack](#-technology-stack)
- [Windows Setup & Quickstart](#-windows-setup--quickstart)
- [Automated Testing](#-automated-testing)

- [Viva & Placement Interview Q&A](#-viva--placement-interview-qa-guide)
- [Ethical Data Provenance & Limitations](#-ethical-data-provenance--limitations)
- [Project Directory Structure](#-project-directory-structure)

---

## 🎯 Problem Statement

In modern e-commerce marketplaces (Amazon, Flipkart, Croma, Reliance Digital), shoppers face constant marketing deception:

1. **Inflated MRP Anchoring**: Retailers frequently double the manufacturer's suggested retail price (MRP) so that an ordinary everyday price appears to carry an eye-catching "60% OFF" badge.
2. **Deceptive Deal Badges**: Algorithmic badges such as "Lightning Deal" or "Limited Time Offer" create artificial urgency without reflecting genuine price cuts.
3. **Lack of Intrinsic Valuation**: Buyers lack an independent benchmark to determine what a product with specific specifications, brand reputation, and consumer satisfaction *ought* to cost.

**SmartDeal AI** decouples deal detection from retailer-manipulated MRPs. By learning fair benchmark reference prices through regression models and comparing current offers to historical price distributions, it reveals whether a discount is a genuine bargain or marketing smoke.

---

## ✨ Key Features

- **Algorithmic Deal Ranking Engine**: Evaluates deals from **0 to 100** and categorizes them into `Great Deal` (≥80), `Good Deal` (65-79), `Fair Price` (45-64), and `Above Reference` (<45).
- **Leak-Free Machine Learning Valuation**: Trains Baseline (Dummy Median), Linear Regression, Random Forest, and Gradient Boosting Regressors. Selects the champion model based strictly on held-out test RMSE.
- **Time-Series Price Trajectory & Forecasting**: Interactive Plotly timeline charts featuring 7-day moving averages, all-time highs/lows, and trend forecasts with 95% expanding confidence intervals.
- **Chronological Holdout Validation**: Avoids "look-ahead bias" by evaluating forecasting models strictly on temporal chronological holdouts rather than random cross-validation.
- **Out-of-Distribution Guardrails**: Detects unseen categories, brands, or extreme price inputs, clamps predictions within physical bounds $[0.10 \times \text{MSRP}, 0.98 \times \text{MSRP}]$, and dynamically inflates uncertainty bands (+35% per warning).
- **Side-by-Side Product Comparison**: Compares 2 to 4 products simultaneously, explicitly distinguishing the **Lowest Sticker Price** from the **Best Supported Deal**.
- **Self-Healing SQLite Persistence**: Automatically initializes and seeds products, price histories, user watchlists, and prediction audit trails using parameterized queries.
- **Fintech Dark Navy UI/UX**: Polished desktop-and-mobile-responsive layout, custom CSS cards, and Indian Rupee (`₹`) formatting with standard Lakh/Crore comma grouping.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Data_Tier["1. Data Ingestion & Storage Tier"]
        A[Catalog CSV / User Upload] --> B[Data Validation & Schema Normalization]
        B --> C[(SQLite Database: smartdeal.db)]
        C --> D[(products Table)]
        C --> E[(price_history Table)]
        C --> F[(prediction_logs Table)]
        C --> G[(watchlist Table)]
    end

    subgraph ML_Tier["2. Machine Learning & Valuation Pipeline"]
        D --> H[Strict Feature/Target Split: prepare_features_and_target]
        H -->|Features X: MSRP, Rating, Reviews, Cat, Brand| I[X_train 80% / X_test 20% Isolation]
        H -->|Target y: fair_reference_price| I
        I --> J[Preprocess Pipeline: Imputer + Scaler + OneHotEncoder]
        J --> K[Train 4 Regressors: Dummy, Linear, RF, GBDT]
        K --> L[Evaluate on Out-of-Sample Held-Out Test Set]
        L --> M[Model Selection by RMSE: Champion Linear Regression]
        M --> N[Joblib Serialization: best_model_pipeline.joblib]
    end

    subgraph Forecast_Tier["3. Time-Series Price Analytics"]
        E --> O[History Service: Rolling 7-day / 14-day Moving Averages]
        O --> P[Dampened Linear Trend Extrapolation]
        P --> Q[Expanding 95% Confidence Intervals]
        O --> R[Adaptive Chronological Holdout Validation]
    end

    subgraph Deal_Tier["4. Business Intelligence & Scoring Engine"]
        N --> S[Live Fair Price Predictor]
        S -->|OOD Input Check| T[Dynamic Uncertainty Inflation & Bounding]
        T --> U[Deal Scoring Formula: 0 to 100 Rating]
        O --> U
        U --> V[Deal Tier: Great Deal, Good Deal, Fair Price, Above Ref]
    end

    subgraph UI_Tier["5. Interactive Presentation Tier (Streamlit)"]
        V --> W[Overview Dashboard & Catalog Analytics]
        V --> X[Smart Deal Finder & Watchlist CRUD]
        S --> Y[Live Prediction Form & Tolerance Bands]
        Q --> Z[Price History Visualizer & Forecast Chart]
        V --> AA[Side-by-Side Product Comparison]
        L --> AB[Model Performance Leaderboard & Parity Plots]
    end
```

---

## 🌐 Cloud Deployment (1-Click)

SmartDeal AI is production-ready and configured for instant cloud deployment:

### Option 1: Streamlit Community Cloud (Recommended — Free & Direct)
Deploy directly from your GitHub repository in 60 seconds:
1. Click the **[Deploy to Streamlit](https://share.streamlit.io/deploy?repository=kowshiii1208/smartdeal-ai&branch=main&mainModule=app.py)** link.
2. Sign in with your GitHub account (`kowshiii1208`).
3. Streamlit Cloud automatically detects:
   - **Repository:** `kowshiii1208/smartdeal-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!** — Your live app will be accessible worldwide at `https://smartdeal-ai.streamlit.app`.

### Option 2: Docker Container Deployment (Render, Railway, Fly.io)
SmartDeal AI includes a multi-stage production [`Dockerfile`](Dockerfile) and `.dockerignore`:
```bash
# Build the Docker image
docker build -t smartdeal-ai .

# Run the container locally or on cloud VM
docker run -p 8501:8501 smartdeal-ai
```
Visit `http://localhost:8501` in your browser.

---

## ⚙️ Machine Learning Methodology


### 1. Target Variable Definition
- **Target**: `fair_reference_price` — the median historical equilibrium market benchmark of the product.
- **Why not predict live selling price?** Live current prices include temporary flash sale fluctuations. Predicting the equilibrium fair benchmark gives a stable, objective anchor against which the current price can be audited.

### 2. Preventing Data Leakage
- `current_price` and `discount_percentage` are **STRICTLY EXCLUDED** from the training feature set $X$.
- *Reasoning*: If `current_price` were fed to the regressor, the model would simply learn the identity function ($y = x$), achieving trivial 0 loss while failing completely to evaluate whether the current price is a good deal.
- All imputation (`SimpleImputer`), scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`) are encapsulated inside a `Pipeline` fitted strictly on training data.
- Feature domain bounds (`training_bounds`) are extracted strictly from $X_{train}$ post-split to eliminate distribution leakage.

### 3. Empirical Model Benchmark (Held-Out 20% Test Split)

| Candidate Model | MAE (₹) | RMSE (₹) | $R^2$ Score | MAPE (%) | Test Split | Benchmark Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Dummy Median)** | ₹32,502.50 | ₹51,360.17 | -0.1012 | 100.99% | 24 records | Academic Baseline |
| **Linear Regression** | **₹3,226.15** | **₹5,050.92** | **0.9894** | **13.11%** | 24 records | **👑 Selected Champion** |
| **Random Forest Regressor** | ₹6,394.77 | ₹13,255.18 | 0.9267 | 23.36% | 24 records | Tree Ensemble |
| **Gradient Boosting Regressor** | ₹6,349.82 | ₹14,674.70 | 0.9101 | 24.08% | 24 records | Gradient Boosting |

*Note: All metrics are computed dynamically on held-out test data. Zero fabricated claims.*

---


## 📐 Transparent Deal Scoring Formula

Rather than relying on proprietary black-box discount percentages, SmartDeal AI calculates deal value via a documented formula:

$$\text{Diff}_{\%} = \frac{P_{ref} - P_{curr}}{P_{ref}} \times 100$$
$$\text{MRP}_{\%} = \frac{P_{orig} - P_{curr}}{P_{orig}} \times 100$$

$$\text{Score}_{raw} = 50.0 + \text{clip}(2.5 \times \text{Diff}_{\%}, -35, 35) + \text{clip}(0.25 \times \text{MRP}_{\%}, 0, 12) + \text{clip}(5.0 \times (R - 4.0), -10, 5)$$

$$\text{Score}_{final} = 50.0 + (\text{Score}_{raw} - 50.0) \times W_{confidence}$$

- $W_{confidence} = 1.00$ for products with $\ge 1,000$ reviews and historical records.
- $W_{confidence} = 0.90$ for products with $100 - 999$ reviews.
- $W_{confidence} = 0.75$ for products with $< 100$ reviews (regresses score toward fair equilibrium 50 due to weak evidence).

---

## 🖥️ Navigation Pages Guide

1. **Overview Dashboard (`pages/1_📊_Overview.py`)**: Macro market analytics, category price comparisons, MRP-vs-selling-price grouped bars, category distributions, and retailer market shares.
2. **Smart Deal Finder (`pages/2_🏷️_Smart_Deals.py`)**: Multi-attribute filtering (category, brand, price slider, minimum rating, deal tier), deal cards with badges, estimated savings in ₹, and one-click SQLite watchlist addition.
3. **Product Price Prediction (`pages/3_🔮_Price_Prediction.py`)**: Interactive specification form, real-time ML inference, tolerance bands, feature driver explanations, extrapolation warnings, and SQLite audit logs.
4. **Price History & Forecast (`pages/4_📈_Price_History.py`)**: Plotly time-series visualizer showing observed prices (cyan), 7-day moving averages (amber), all-time lows/highs, listed price benchmark (green), and future forecasts with 90% confidence bands. Includes chronological holdout evaluation metrics.
5. **Compare Products (`pages/5_⚖️_Compare_Products.py`)**: Side-by-side matrix for 2 to 4 items. Separately highlights the **Lowest Current Price** vs the **Best Supported Deal**. Warns if cross-category items are selected.
6. **ML Model Performance (`pages/6_🎯_Model_Performance.py`)**: Benchmark leaderboard comparing 4 models, actual-vs-predicted scatter with 45° line, residual distribution plot, tree feature importances, and a one-click "Retrain All Models" action.
7. **Dataset Explorer (`pages/7_📁_Dataset_Explorer.py`)**: Searchable data inspector, descriptive summary statistics, missing-value profile, CSV upload with schema validation, and export downloads.
8. **About & Documentation (`pages/8_📖_About.py`)**: System architecture diagrams, mathematical equations, data provenance policies, and placement viva interview prep guide.

---

## 💻 Windows Setup & Quickstart

### Prerequisites
- Python 3.10 to 3.13 installed on Windows.
- Git (optional, for cloning).

### 1. Clone or Open Workspace
```powershell
cd c:\Users\kowsh\OneDrive\Desktop\e_commerce
```

### 2. Set Up Virtual Environment
```powershell
py -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Train Models & Initialize Database
```powershell
python -m ml.train
```
*This validates the dataset, initializes SQLite, trains all 4 regression models, evaluates held-out metrics, and persists the champion model pipeline to `models/best_model_pipeline.joblib`.*

### 5. Launch Application
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

SmartDeal AI includes a comprehensive Pytest test suite covering data cleaning, deal formulas, ML pipelines, time-series forecasting, and database transactions:

### Run Entire Test Suite:
```powershell
python -m pytest -v
```

### Run Specific Test Modules:
```powershell
# Run ML pipeline, target leakage, and reproducibility tests
python -m pytest tests/test_ml_pipeline.py -v

# Run time-series forecasting and adaptive holdout tests
python -m pytest tests/test_forecasting.py -v

# Run deal classification and scoring formula tests
python -m pytest tests/test_deal_engine.py -v

# Run data validation, schema cleaning, and bounds tests
python -m pytest tests/test_data_validation.py -v

# Run SQLite parameterized persistence and watchlist tests
python -m pytest tests/test_database.py -v
```

### Test Coverage Highlights (28 Tests, 100% Passing):
- `tests/test_data_validation.py` (6 tests): Schema validation, mandatory columns, duplicate product ID deduplication, negative price rejection, rating boundary clamping ([1.0, 5.0]), and chronological date sorting.
- `tests/test_deal_engine.py` (5 tests): Mathematical deal score verification, tier classifications (`Great Deal`, `Above Reference`, `Fair Price`), savings calculation, and evidence confidence regression.
- `tests/test_ml_pipeline.py` (8 tests): Target leakage checks (asserting `current_price` is strictly absent from $X$), handling unseen categories/brands in `OneHotEncoder`, metric calculations (MAE, RMSE, R²), live prediction pipeline, deterministic reproducibility (`RANDOM_SEED = 42`), out-of-distribution input detection, dynamic uncertainty inflation, and baseline comparison superiority.
- `tests/test_forecasting.py` (5 tests): Minimum history threshold (< 5 records raises `ValueError`), forecast confidence interval bounds ($Upper \ge Forecast \ge Lower$), chronological holdout evaluation, adaptive holdout for short series ($N=5$), and horizon uncertainty expansion.
- `tests/test_database.py` (4 tests): SQLite schema creation, bulk product insertion, price history ingestion, prediction audit logging, and watchlist CRUD operations.

---

## 🛡️ Ethical Data Provenance & Limitations

### Dataset Provenance
- **Demonstration Dataset**: 120 e-commerce product records spanning 8 key retail categories (Laptops, Smartphones, Audio & Headphones, Smartwatches & Wearables, Televisions, Tablets, Cameras, Home Appliances).
- **Time-Series Price Trajectories**: 90 days of chronological price observations generated every 2–3 days for all 120 catalog products (totaling over 4,000 temporal observations).
- **100% Deterministic Reproducibility**: Generated via `data/data_generator.py` with a fixed pseudo-random seed (`RANDOM_SEED = 42`). All demonstration records are tagged with `is_synthetic = 1`.

### Explicit Limitations & Boundary Conditions
1. **Extrapolation Uncertainty**: When querying prices or specifications far outside the training distribution (e.g. an earphone listed at ₹5,00,000 or an unseen brand), linear models must extrapolate. The system detects this, applies physical clamping $[0.10 \cdot \text{MSRP}, 0.98 \cdot \text{MSRP}]$, and dynamically inflates the confidence interval by $+35\%$ per warning.
2. **Unannounced Retail Dynamics**: Statistical and regression models evaluate intrinsic product value and historical trendlines; they cannot anticipate unannounced manufacturer cashback programs, flash bank credit card coupons, or warehouse clearance liquidations.
3. **No Unconsented Live Scraping**: Live scraping of retailer websites is disabled by default to respect website Terms of Service, rate limits, and `robots.txt` directives. The system architecture is API-ready to consume authorized merchant affiliate feeds (e.g., Amazon Product Advertising API).


---

## 📁 Project Directory Structure

```
e_commerce/
├── app.py                      # Main landing page & executive hub
├── config.py                   # Configuration, paths, seeds, thresholds
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation & interview guide
├── .gitignore                  # Git ignore rules for Python, SQLite, venv
├── database/
│   ├── schema.sql              # SQLite DDL schema
│   └── db.py                   # Transaction manager & parameterized queries
├── data/
│   ├── data_generator.py       # Reproducible demonstration data generator
│   ├── sample_products.csv     # 120+ synthetic product records
│   ├── sample_price_history.csv# 90-day time-series price observations
│   └── cleaned_products.csv    # Normalized, cleaned catalog
├── ml/
│   ├── preprocessing.py        # ColumnTransformer & leakage safeguards
│   ├── features.py             # Feature schema & input bounds validation
│   ├── train.py                # 4-model training & model selection script
│   ├── evaluate.py             # MAE, RMSE, R2, MAPE, residual extractors
│   ├── predict.py              # Model loading, inference & confidence bounds
│   └── forecast.py             # Moving average, trend forecast & holdouts
├── services/
│   ├── deal_engine.py          # 0-100 deal scoring & tier classification
│   ├── product_service.py      # Catalog queries, filters, sorting & comparison
│   └── history_service.py      # Time-series Plotly chart generator
├── pages/
│   ├── 1_📊_Overview.py        # Macro catalog analytics
│   ├── 2_🏷️_Smart_Deals.py     # Deal search, filter & watchlist
│   ├── 3_🔮_Price_Prediction.py # Interactive specification form & inference
│   ├── 4_📈_Price_History.py   # Historical trends & future forecasts
│   ├── 5_⚖️_Compare_Products.py# 2-4 product side-by-side comparison
│   ├── 6_🎯_Model_Performance.py# Held-out evaluation & residual plots
│   ├── 7_📁_Dataset_Explorer.py# Schema validation, CSV upload & export
│   └── 8_📖_About.py           # Technical architecture & viva Q&A
├── utils/
│   ├── formatting.py           # Indian Rupee formatting (₹) & HTML badges
│   ├── validation.py           # Schema validator & anomaly cleaner
│   └── ui_components.py        # Reusable dark navy CSS & hero cards
├── models/
│   ├── best_model_pipeline.joblib # Persisted champion model pipeline
│   └── model_metrics.json      # Evaluation metrics & feature importances
└── tests/
    ├── test_data_validation.py # Schema & anomaly tests
    ├── test_deal_engine.py     # Deal scoring & classification tests
    ├── test_ml_pipeline.py     # Preprocessing, leakage & inference tests
    ├── test_forecasting.py     # Time-series forecast & holdout tests
    └── test_database.py        # SQLite persistence & watchlist tests
```

---

## 📜 License

This project is licensed under the MIT License — feel free to use and adapt it for academic evaluations, coursework, and personal learning.
