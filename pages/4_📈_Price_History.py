"""
Page 4: Price History & Forecast.
Interactive Plotly timeline visualizer with moving averages,
all-time price peaks/troughs, and statistical trend forecasting with confidence intervals.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from utils.ui_components import apply_custom_css, render_hero_banner, render_kpi, render_empty_state
from utils.formatting import format_inr, format_percent
from services.product_service import get_all_products
from services.history_service import get_price_history_for_product, build_price_history_chart
from ml.forecast import generate_price_forecast, compute_history_summary

st.set_page_config(page_title="Price History & Forecast — SmartDeal AI", page_icon="📈", layout="wide")
apply_custom_css()

render_hero_banner(
    title="Price History & Trend Forecasting",
    subtitle="Analyze historical price volatility, track festive sale dips, and evaluate statistical future price projections.",
    badge="Time-Series Intelligence"
)

# Load catalog products for selection
products_df = get_all_products()

if products_df.empty:
    render_empty_state("No Products Found", "Catalog is empty. Please populate demonstration data.")
    st.stop()

# Product Selection Bar
product_options = {
    f"{row['product_name']} ({row['brand']} • {format_inr(row['current_price'])})": row['product_id']
    for _, row in products_df.iterrows()
}

selected_label = st.selectbox(
    "Select a Product to Inspect Price Trajectory",
    options=list(product_options.keys()),
    index=0
)

selected_id = product_options[selected_label]
product_meta = products_df[products_df["product_id"] == selected_id].iloc[0]

# Controls row
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])

with ctrl_col1:
    history_window = st.selectbox(
        "Historical Window",
        options=["All Available (90 Days)", "Past 60 Days", "Past 30 Days"],
        index=0
    )

with ctrl_col2:
    enable_forecast = st.toggle("Enable Trend Forecast", value=True)

with ctrl_col3:
    forecast_horizon = st.slider("Forecast Horizon (Days Ahead)", min_value=7, max_value=30, value=14, step=7)

# Fetch historical data
raw_history = get_price_history_for_product(selected_id)

if raw_history.empty:
    render_empty_state(
        "No Historical Observations",
        f"Product '{product_meta['product_name']}' does not have any recorded price observations yet. "
        "Historical observations can be imported via CSV in the Dataset Explorer.",
        icon="📉"
    )
    st.stop()

# Filter by window if requested
sorted_history = raw_history.sort_values(by="observed_date").reset_index(drop=True)
if "Past 30 Days" in history_window:
    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    view_history = sorted_history[sorted_history["observed_date"] >= cutoff]
elif "Past 60 Days" in history_window:
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    view_history = sorted_history[sorted_history["observed_date"] >= cutoff]
else:
    view_history = sorted_history

if view_history.empty:
    view_history = sorted_history

# Compute Summary Statistics
stats = compute_history_summary(view_history)

# Metrics Cards
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    render_kpi("Current Price", format_inr(product_meta["current_price"]), f"MRP: {format_inr(product_meta['original_price'])}", "#10B981", icon="🏷️")
with m2:
    render_kpi("All-Time Low", format_inr(stats.get("min_price", 0)), "Observed in window", "#3B82F6", icon="⭐")
with m3:
    render_kpi("All-Time High", format_inr(stats.get("max_price", 0)), "Observed in window", "#EF4444", icon="🔺")
with m4:
    render_kpi("Average Price", format_inr(stats.get("avg_price", 0)), "Historical mean", "#8B5CF6", icon="📊")
with m5:
    change_pct = stats.get("price_change_pct", 0)
    color = "#10B981" if change_pct < 0 else ("#EF4444" if change_pct > 0 else "#94A3B8")
    drift_icon = "📉" if change_pct < 0 else ("📈" if change_pct > 0 else "➖")
    render_kpi("Net Price Drift", f"{change_pct:+.1f}%", "Since earliest record", color, icon=drift_icon)

st.markdown("<br>", unsafe_allow_html=True)

# Build and display interactive chart
fig = build_price_history_chart(
    history_df=view_history,
    product_name=product_meta["product_name"],
    current_price=product_meta["current_price"],
    include_forecast=enable_forecast,
    horizon_days=forecast_horizon,
)
st.plotly_chart(fig, use_container_width=True)

# Forecasting Holdout Evaluation & Methodology Section
if enable_forecast and len(view_history) >= 5:
    forecast_results = generate_price_forecast(view_history, horizon_days=forecast_horizon)
    holdout_eval = forecast_results.get("holdout_metrics", {})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Forecasting Methodology & Chronological Holdout Evaluation")

    eval_col1, eval_col2 = st.columns([1, 1])

    with eval_col1:
        st.markdown("##### 📏 Model Error Comparison on Chronological Holdout (Last 5 Observations)")
        if holdout_eval:
            eval_rows = []
            for model_name, errs in holdout_eval.items():
                eval_rows.append({
                    "Forecasting Model": model_name,
                    "MAE (₹)": format_inr(errs["mae"]),
                    "RMSE (₹)": format_inr(errs["rmse"]),
                })
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Insufficient historical density to compute multi-model holdout splits.")

    with eval_col2:
        st.markdown("##### ⚠️ Forecast Limitations & Safe Usage")
        st.markdown(
            """
            - **Trend vs Flash Events**: Statistical models extrapolate price persistence and cyclical drift, but cannot predict unannounced flash sales or sudden promotional coupons.
            - **Expanding Uncertainty**: The shaded band indicates a 90% confidence envelope that naturally widens over longer forecast horizons.
            - **Evaluation Protocol**: Models are evaluated on chronological out-of-sample data points, never random splits, to avoid future-to-past data leakage.
            """
        )
