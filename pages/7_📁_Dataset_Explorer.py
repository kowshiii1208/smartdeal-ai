"""
Page 7: Dataset Explorer.
Search, inspect, and analyze raw and cleaned product datasets.
Provides schema validation, missing value diagnostics, CSV upload/replacement,
and downloads for cleaned datasets and evaluation results.
"""

import io
import json
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.ui_components import apply_custom_css, render_hero_banner, render_kpi, render_html
from utils.formatting import format_inr
from utils.validation import validate_and_clean_products
from services.product_service import get_all_products
from database.db import insert_or_replace_products
from config import MODEL_METRICS_PATH, CLEANED_PRODUCTS_PATH

st.set_page_config(page_title="Dataset Explorer — SmartDeal AI", page_icon="📁", layout="wide")
apply_custom_css()

render_hero_banner(
    title="Dataset Explorer & Data Quality Studio",
    subtitle="Audit schema integrity, inspect missing value profiles, upload custom product catalogs, and export cleaned CSVs.",
    badge="Data Pipeline Studio"
)

# Fetch current catalog
df = get_all_products()

# Tabs for Explorer vs Upload & Export
tab_explore, tab_upload, tab_export = st.tabs([
    "🔍 Browse & Analyze Catalog",
    "📤 Upload & Validate Custom CSV",
    "💾 Export Cleaned Datasets & Metrics",
])

with tab_explore:
    st.markdown("#### 📋 Catalog Data Inspector")
    
    # Quick filter and column selection
    f_col1, f_col2 = st.columns([2, 2])
    with f_col1:
        search_kw = st.text_input("Filter by Keyword", placeholder="Search by name, brand, or category...")
    with f_col2:
        visible_cols = st.multiselect(
            "Visible Columns",
            options=df.columns.tolist(),
            default=["product_id", "product_name", "category", "brand", "current_price", "original_price", "discount_percentage", "rating", "deal_score", "deal_category"]
        )

    view_df = df.copy()
    if search_kw:
        kw = search_kw.lower()
        view_df = view_df[
            view_df["product_name"].str.lower().str.contains(kw, na=False) |
            view_df["brand"].str.lower().str.contains(kw, na=False) |
            view_df["category"].str.lower().str.contains(kw, na=False)
        ]

    st.dataframe(
        view_df[visible_cols] if visible_cols else view_df,
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Showing {len(view_df)} of {len(df)} total product records.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistical Diagnostics
    st.markdown("#### 🔬 Statistical Summary & Distribution")
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        st.markdown("##### Numerical Descriptive Statistics")
        num_summary = df[["current_price", "original_price", "discount_percentage", "rating", "review_count", "deal_score"]].describe().T
        num_summary = num_summary.rename(columns={
            "mean": "Mean", "std": "Std Dev", "min": "Min", "50%": "Median", "max": "Max"
        })
        st.dataframe(num_summary[["Mean", "Std Dev", "Min", "Median", "Max"]].round(2), use_container_width=True)

    with s_col2:
        st.markdown("##### Missing Value Profile")
        null_counts = df.isna().sum().reset_index()
        null_counts.columns = ["Column Name", "Missing Count"]
        null_counts["Missing %"] = (null_counts["Missing Count"] / len(df) * 100).round(1)
        st.dataframe(null_counts, use_container_width=True, hide_index=True)


with tab_upload:
    st.markdown("#### 📤 Upload and Ingest Product CSV")
    st.caption("Upload your own custom e-commerce product catalog. Files will be strictly validated for types, ranges, and negative values before acceptance.")

    uploaded_file = st.file_uploader("Select a CSV File (Max 10MB)", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.info(f"Loaded CSV with {len(uploaded_df)} rows and {len(uploaded_df.columns)} columns.")

            # Validate uploaded data
            is_valid, errors, warnings, stats, cleaned_uploaded = validate_and_clean_products(uploaded_df)

            if is_valid:
                st.success("✅ Schema Validation PASSED! File complies with SmartDeal AI data standards.")
                
                # Show cleaning stats
                v1, v2, v3, v4 = st.columns(4)
                with v1:
                    render_kpi("Valid Rows", str(stats["valid_rows"]), "Cleaned & verified", "#10B981", icon="✅")
                with v2:
                    render_kpi("Duplicates Removed", str(stats["duplicates_removed"]), "Deduplicated by product_id", "#3B82F6", icon="🧹")
                with v3:
                    render_kpi("Imputed Fields", str(stats["missing_values_imputed"]), "Ratings/discounts repaired", "#8B5CF6", icon="🛠️")
                with v4:
                    render_kpi("Price Warnings", str(stats["anomalous_prices_detected"]), "Current > MSRP anomalies", "#F59E0B", icon="⚠️")

                if warnings:
                    st.markdown("##### ⚠️ Data Quality Warnings")
                    for w in warnings:
                        st.markdown(f"- {w}")

                st.markdown("##### Preview Cleaned Dataset")
                st.dataframe(cleaned_uploaded.head(10), use_container_width=True)

                if st.button("📥 Save & Ingest Into SQLite Catalog", key="save_csv_btn"):
                    try:
                        insert_or_replace_products(cleaned_uploaded)
                        cleaned_uploaded.to_csv(CLEANED_PRODUCTS_PATH, index=False)
                        st.success("Successfully ingested uploaded products into SQLite database!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Failed to ingest records into database: {e}")

            else:
                st.error("❌ Schema Validation FAILED! Please address the following blocking errors:")
                for err in errors:
                    st.markdown(f"- 🔴 **{err}**")

        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")


with tab_export:
    st.markdown("#### 💾 Export Cleaned Data & Model Evaluation Results")
    st.caption("Download verified datasets and evaluation artifacts for offline analysis, research papers, or hackathon reports.")

    e_col1, e_col2 = st.columns(2)

    with e_col1:
        render_html(
            """
            <div class="content-box">
                <h5 style="color: #60A5FA; margin-top: 0;">Cleaned Catalog Dataset (CSV)</h5>
                <p style="color: #94A3B8; font-size: 0.85rem;">
                    Export the current deduplicated, normalized, and deal-enriched product catalog including computed fair reference benchmarks.
                </p>
            </div>
            """
        )
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Cleaned Catalog CSV",
            data=csv_buffer.getvalue(),
            file_name="smartdeal_cleaned_products.csv",
            mime="text/csv",
            use_container_width=True
        )

    with e_col2:
        render_html(
            """
            <div class="content-box">
                <h5 style="color: #34D399; margin-top: 0;">Model Metrics & Benchmark Results (JSON)</h5>
                <p style="color: #94A3B8; font-size: 0.85rem;">
                    Export out-of-sample held-out test evaluation scores (MAE, RMSE, R²), split details, and feature importances for college review documentation.
                </p>
            </div>
            """
        )
        if MODEL_METRICS_PATH.exists():
            with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_json_str = f.read()
            st.download_button(
                label="⬇️ Download Model Evaluation JSON",
                data=metrics_json_str,
                file_name="smartdeal_model_metrics.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("Train the model first to generate downloadable metrics.")
