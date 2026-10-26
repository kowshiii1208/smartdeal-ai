"""
Page 8: About & Technical Documentation.
Comprehensive technical report, architectural diagrams, mathematical formulations,
and BTech CSE viva/interview presentation guide.
"""

import streamlit as st
from utils.ui_components import apply_custom_css, render_hero_banner

st.set_page_config(page_title="About & Docs — SmartDeal AI", page_icon="📖", layout="wide")
apply_custom_css()

render_hero_banner(
    title="About & Technical Documentation",
    subtitle="System architecture, mathematical methodologies, machine learning pipeline defensibility, and placement interview Q&A guide.",
    badge="Engineering Docs",
    icon="📖"
)

tab_overview, tab_arch, tab_ml, tab_viva = st.tabs([
    "🎯 Project Overview & Problem Statement",
    "🏗️ System Architecture & Data Flow",
    "⚙️ ML Methodology & Deal Scoring",
    "🎓 Student Viva & Placement Interview Q&A",
])

with tab_overview:
    st.markdown(
        """
        ### 1. Project Background & Objective
        In contemporary e-commerce marketplaces (Amazon, Flipkart, Croma, etc.), consumers face significant informational asymmetry:
        
        - **Inflated Artificial MRPs**: Retailers frequently inflate the original manufacturer sticker price (MRP) so that ordinary prices appear to carry an enticing "50% OFF" badge.
        - **Price Volatility & Dynamic Pricing**: Algorithms alter prices daily or hourly based on festive sales, browsing spikes, and inventory levels.
        - **Lack of Fair Value Benchmarks**: Buyers struggle to assess whether a product currently listed at ₹54,990 is genuinely a great deal or simply at its routine equilibrium price.
        
        **SmartDeal AI** solves this problem by decoupling deal scoring from artificial MRP discounts. Instead, it builds an end-to-end price intelligence pipeline that:
        
        1. Learns fair market benchmark prices using machine learning without target leakage.
        2. Evaluates offers using an objective, multi-factor scoring formula (0–100).
        3. Tracks historical price volatility and statistical moving averages.
        4. Provides explainable forecasts with 90% confidence bands.
        
        ---
        
        ### 2. Technology Stack & Design Decisions
        - **Programming Language**: Python 3.13
        - **Data Processing**: Pandas, NumPy
        - **Machine Learning**: Scikit-Learn (Dummy Baseline, Linear Regression, Random Forest, Gradient Boosting)
        - **Time-Series Analysis**: Rolling Moving Averages, Polynomial Trend Extrapolation, Chronological Holdouts
        - **Frontend & UI**: Streamlit with custom Fintech Dark Navy CSS design system
        - **Interactive Plotting**: Plotly Graph Objects
        - **Relational Storage**: SQLite with parameterized queries and transaction pooling
        - **Model Persistence**: Joblib serialization
        - **Testing**: Pytest unit and integration test suite
        """
    )

with tab_arch:
    st.markdown(
        """
        ### 🏛️ Architecture & End-to-End Data Pipeline
        
        ```mermaid
        graph TD
            A[Raw Product Catalogs / CSV Ingestion] --> B[Data Validation & Schema Normalization]
            B --> C[(SQLite Database / Local Storage)]
            C --> D[Feature Engineering & ColumnTransformer]
            D --> E[Model Training Split: 80% Train / 20% Test]
            E --> F[Model Comparison: Baseline, Linear, RF, GBDT]
            F --> G[Select Best Pipeline by Test RMSE]
            G --> H[Joblib Model Persistence]
            H --> I[Smart Deal Engine: 0-100 Scoring]
            C --> J[Historical Observations Service]
            J --> K[Time-Series Forecast Engine]
            I --> L[Streamlit UI & Multi-Page Dashboard]
            K --> L
        ```
        
        #### Modular Project Structure
        - `config.py`: Centralized configuration, file paths, random seeds, and deal thresholds.
        - `database/`: SQLite schema (`schema.sql`) and transactional manager (`db.py`).
        - `data/`: Demonstration data generator (`data_generator.py`) and stored sample datasets.
        - `ml/`: Preprocessing pipeline, feature schema, model training (`train.py`), evaluation (`evaluate.py`), inference (`predict.py`), and forecasting (`forecast.py`).
        - `services/`: Deal scoring engine (`deal_engine.py`), catalog query service (`product_service.py`), and history visualizer (`history_service.py`).
        - `utils/`: Indian Rupee formatting (`formatting.py`), validation engine (`validation.py`), and UI components (`ui_components.py`).
        - `pages/`: 8 dedicated interactive Streamlit pages.
        - `tests/`: Pytest automated test suite.
        """
    )

with tab_ml:
    st.markdown(
        """
        ### 🔬 Machine Learning Pipeline & Mathematical Formulations
        
        #### 1. Target Definition & Preventing Data Leakage
        - **The Target Variable**: `fair_reference_price` — represents the long-term equilibrium benchmark market price of the item.
        - **Critical Data Leakage Rule**: `current_price` and `discount_percentage` are **STRICTLY EXCLUDED** from the training feature set. 
          If `current_price` was included, the model would simply memorize `predicted_price = current_price`, achieving trivial near-zero loss while being completely useless for evaluating whether `current_price` is a good deal!
        
        #### 2. Feature Preprocessing
        Built using `sklearn.compose.ColumnTransformer`:
        - **Numerical Features** (`original_price`, `rating`, `review_count`):
          - Imputed via `SimpleImputer(strategy='median')`
          - Scaled via `StandardScaler()`
        - **Categorical Features** (`category`, `brand`):
          - Imputed with constant `'Unknown'`
          - Encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
        
        #### 3. Transparent Deal Scoring Equation
        Unlike proprietary black-box discount algorithms, SmartDeal AI uses an explainable formula:
        
        $$\\text{Diff}_{\\%} = \\frac{P_{ref} - P_{curr}}{P_{ref}} \\times 100$$
        $$\\text{MRP}_{\\%} = \\frac{P_{orig} - P_{curr}}{P_{orig}} \\times 100$$
        
        $$\\text{Score}_{raw} = 50.0 + \\text{clip}(2.5 \\times \\text{Diff}_{\\%}, -35, 35) + \\text{clip}(0.25 \\times \\text{MRP}_{\\%}, 0, 12) + \\text{clip}(5.0 \\times (R - 4.0), -10, 5)$$
        
        $$\\text{Score}_{final} = 50.0 + (\\text{Score}_{raw} - 50.0) \\times W_{confidence}$$
        
        Where $W_{confidence} \\in [0.75, 1.0]$ based on verified review volume and historical depth.
        """
    )

with tab_viva:
    st.markdown(
        """
        ### 🎓 Placement Interview & Viva Presentation Q&A Guide
        
        #### Q1. Why didn't you feed `current_price` into your regression model?
        > **Answer**: *"That would cause catastrophic target leakage. In our problem domain, the purpose of the model is to learn what a product with these specifications, brand equity, and customer acclaim ought to cost at market equilibrium. If we provided the current listed price, the model would merely learn the identity function without extracting intrinsic value. By hiding the current price, we can use the model's prediction as an independent, objective reference benchmark to evaluate whether the current price is a genuine bargain."*
        
        #### Q2. Why is R² alone not always sufficient for price regression?
        > **Answer**: *"R² measures the proportion of variance explained by the model compared to a naive horizontal mean baseline. On small or clustered test datasets with low intrinsic variance, R² can appear artificially low or even negative despite low absolute errors. Therefore, in financial and e-commerce applications, we report Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in Indian Rupees (₹), which give tangible, actionable bounds on price prediction error."*
        
        #### Q3. How does your pipeline handle unseen brands or categories during live user inference?
        > **Answer**: *"We configure Scikit-Learn's `OneHotEncoder` with `handle_unknown='ignore'`. When an unknown brand is passed into the pipeline, the encoder generates an all-zeros vector for the one-hot columns without throwing a dimensionality mismatch exception. The model falls back on category characteristics, original MSRP, and customer rating to provide a stable, conservative price estimate."*
        
        #### Q4. Why did you use chronological holdout rather than random k-fold cross-validation for price forecasting?
        > **Answer**: *"In time-series forecasting, observations have an inherent temporal sequence. If we used random cross-validation, future price points would leak into the training partition to predict past price points ('look-ahead bias'). Evaluating forecasting models strictly on chronological out-of-sample data points ensures realistic, leak-free evaluation."*
        
        #### Q5. How does your deal engine distinguish genuine discounts from fake inflated MRPs?
        > **Answer**: *"Retailers often inflate MRP to show fake '70% OFF' banners. Our Smart Deal engine assigns only a small secondary weight (up to 12 points) to the MRP discount, while assigning the dominant weight (up to 35 points) to the difference between the listed price and the predicted fair market reference price. If a product has a 50% discount from MRP but is still listed higher than its fair market price, its score will be penalized into 'Above Reference'."*
        """
    )
