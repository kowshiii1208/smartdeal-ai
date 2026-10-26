"""
Page 3: Product Price Prediction.
Interactive inference portal using the fitted Scikit-Learn pipeline.
Predicts fair reference market prices, checks training domain bounds,
explains key factors, and simulates deal scores.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from utils.ui_components import apply_custom_css, render_hero_banner, render_kpi, render_empty_state, render_html
from utils.formatting import format_inr, format_percent, get_deal_badge_html
from ml.predict import predict_fair_price, load_model_pipeline
from services.deal_engine import calculate_deal_metrics
from database.db import log_prediction, fetch_df
from config import PRODUCT_CATEGORIES, POPULAR_BRANDS

st.set_page_config(page_title="Price Prediction — SmartDeal AI", page_icon="🔮", layout="wide")
apply_custom_css()

render_hero_banner(
    title="Product Price Prediction & Valuation",
    subtitle="Estimate the fair benchmark market price of any product configuration using our trained machine learning regression pipeline.",
    badge="ML Inference Engine"
)

# Load model pipeline to inspect available models & bounds
try:
    pipeline, metrics = load_model_pipeline()
    best_model_name = metrics.get("best_model_name", "Linear Regression")
    model_stats = metrics.get("model_comparison", {}).get(best_model_name, {})
except Exception as e:
    st.error(f"Failed to load trained model pipeline: {e}")
    st.info("Please run `python -m ml.train` or use the Model Performance page to train candidate models.")
    st.stop()

# Two-column layout: Form on Left, Prediction Output on Right
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown("#### 📝 Product Specification Form")
    st.caption("Provide product characteristics to estimate its equilibrium market price.")

    with st.form("price_prediction_form"):
        # Category Selector
        category = st.selectbox("Category", options=PRODUCT_CATEGORIES, index=0)

        # Brand Selector based on category
        available_brands = POPULAR_BRANDS.get(category, ["Apple", "Samsung", "Sony", "Dell", "HP", "Other"])
        brand = st.selectbox("Brand", options=available_brands, index=0)

        # Original MRP
        original_price = st.number_input(
            "Original Manufacturer Price (MRP in ₹)",
            min_value=500.0,
            max_value=1_000_000.0,
            value=79900.0,
            step=1000.0,
            help="Official retail sticker price before any discounts."
        )

        # Consumer Rating
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            rating = st.slider("Customer Rating (Stars)", min_value=1.0, max_value=5.0, value=4.4, step=0.1)
        with col_r2:
            review_count = st.number_input(
                "Review Count",
                min_value=0,
                max_value=100_000,
                value=3500,
                step=50,
                help="Total verified reviews submitted for this product."
            )

        # Optional Deal Simulation field
        st.markdown("##### 🛒 Deal Evaluation (Optional)")
        offered_price = st.number_input(
            "Current Listed Selling Price (₹) — Optional",
            min_value=0.0,
            max_value=1_000_000.0,
            value=69990.0,
            step=500.0,
            help="Enter an offered price to evaluate whether it represents a Great Deal against the predicted fair price."
        )

        submitted = st.form_submit_button("🔮 Predict Fair Price & Evaluate Deal", use_container_width=True)

with col_result:
    st.markdown("#### 🎯 Valuation & Deal Analysis Result")

    if submitted:
        input_data = {
            "category": category,
            "brand": brand,
            "original_price": original_price,
            "rating": rating,
            "review_count": review_count,
        }

        try:
            pred_result = predict_fair_price(input_data)
            predicted_price = pred_result["predicted_price"]
            min_exp = pred_result["min_expected_price"]
            max_exp = pred_result["max_expected_price"]
            rmse = pred_result["rmse"]
            r2 = pred_result["r2"]
            warnings = pred_result["warnings"]
            factors = pred_result["factors"]

            # Display Warnings if inputs fell outside training bounds
            for w in warnings:
                st.markdown(f'<div class="callout-warning">⚠️ <b>Extrapolation Warning:</b> {w}</div>', unsafe_allow_html=True)

            # Main Prediction Result Box
            render_html(
                f"""
                <div style="background: linear-gradient(135deg, rgba(20, 32, 57, 0.8) 0%, rgba(12, 19, 34, 0.95) 100%); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 14px; padding: 26px; text-align: center; margin-bottom: 20px; box-shadow: 0 12px 30px -8px rgba(6, 182, 212, 0.2);">
                    <div style="font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; color: #94A3B8;">
                        🎯 Estimated Fair Market Valuation
                    </div>
                    <div style="font-size: 2.6rem; font-weight: 800; color: #38BDF8; margin: 8px 0; letter-spacing: -0.02em;">
                        {format_inr(predicted_price)}
                    </div>
                    <div style="background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px; margin: 12px 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #94A3B8; margin-bottom: 4px;">
                            <span>Lower Bound: <b>{format_inr(min_exp)}</b></span>
                            <span>Upper Bound: <b>{format_inr(max_exp)}</b></span>
                        </div>
                        <div style="background: rgba(255,255,255,0.06); height: 6px; border-radius: 9999px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #3B82F6, #38BDF8); height: 100%; width: 100%;"></div>
                        </div>
                    </div>
                    <div style="font-size: 0.78rem; color: #64748B;">
                        Fitted Engine: <b style="color: #F8FAFC;">{pred_result['model_name']}</b> • 
                        {"Tolerance Band: <b style='color: #F59E0B;'>±" + format_inr(pred_result.get('effective_rmse', rmse)) + " (OOD Inflated)</b> • " if pred_result.get('is_ood') else "Held-Out RMSE: <b style='color: #F8FAFC;'>±" + format_inr(rmse) + "</b> • "}
                        R²: <b style="color: #34D399;">{r2:.4f}</b>
                    </div>
                </div>
                """
            )

            # Deal Simulation Evaluation
            if offered_price > 0:
                deal_metrics = calculate_deal_metrics(
                    current_price=offered_price,
                    reference_price=predicted_price,
                    original_price=original_price,
                    rating=rating,
                    review_count=review_count,
                    has_history=False,
                )

                st.markdown("##### 🏷️ Evaluated Deal Rating")
                render_html(
                    f"""
                    <div class="product-card" style="margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 0.88rem; color: #94A3B8;">Offered Listing: <b style="color: #F8FAFC; font-size: 1.05rem;">{format_inr(offered_price)}</b></span>
                            {get_deal_badge_html(deal_metrics['deal_score'], deal_metrics['deal_category'])}
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: {'#10B981' if deal_metrics['absolute_difference'] > 0 else '#EF4444'};">
                            {'Estimated Savings: ' + format_inr(deal_metrics['estimated_savings']) + f" ({deal_metrics['percentage_difference']:.1f}% vs fair benchmark)" if deal_metrics['absolute_difference'] > 0 else 'Above Fair Benchmark by ' + format_inr(abs(deal_metrics['absolute_difference']))}
                        </div>
                        <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 6px;">
                            📝 <i>{deal_metrics['explanation']}</i>
                        </div>
                    </div>
                    """
                )

                # Log to SQLite
                log_prediction(
                    category=category,
                    brand=brand,
                    original_price=original_price,
                    rating=rating,
                    review_count=review_count,
                    predicted_price=predicted_price,
                    model_name=pred_result["model_name"],
                    deal_score=deal_metrics["deal_score"],
                    deal_rating=deal_metrics["deal_category"],
                )

            # Factors explanation
            st.markdown("##### 🔍 Value Driver Breakdown")
            render_html(
                f"""
                <div style="background: rgba(19, 30, 54, 0.5); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 14px 16px; font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                    <div>• <b>MSRP Scale Factor:</b> Original price of {format_inr(original_price)} sets the primary base valuation anchor.</div>
                    <div>• <b>Equilibrium Category Discount:</b> Model calculates an expected equilibrium discount of <b>{factors['implied_fair_discount_pct']}%</b> for {brand} in {category}.</div>
                    <div>• <b>Customer Feedback Influence:</b> Rating of {rating}★ with {review_count:,} reviews ({factors['rating_impact']}) maintains price resilience.</div>
                </div>
                """
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        render_empty_state(
            "Ready for Prediction",
            "Fill in the product specification form on the left and click 'Predict Fair Price' to run inference.",
            icon="🔮"
        )

# Recent Predictions Log Section
st.markdown("<br><hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
st.markdown("### 📋 **Recent Prediction Audit Log**")
try:
    recent_preds = fetch_df("SELECT created_at, category, brand, original_price, rating, predicted_price, model_name, deal_score, deal_rating FROM predictions ORDER BY prediction_id DESC LIMIT 5")
    if not recent_preds.empty:
        recent_preds["original_price"] = recent_preds["original_price"].apply(format_inr)
        recent_preds["predicted_price"] = recent_preds["predicted_price"].apply(format_inr)
        recent_preds["created_at"] = pd.to_datetime(recent_preds["created_at"]).dt.strftime("%d %b %Y, %H:%M")
        recent_preds.columns = ["Timestamp", "Category", "Brand", "MSRP", "Rating", "Predicted Fair Price", "Model", "Deal Score", "Deal Tier"]
        st.dataframe(recent_preds, use_container_width=True, hide_index=True)
    else:
        st.caption("No prediction audit records logged yet in this session.")
except Exception:
    st.caption("Prediction audit log initializing...")
