"""
Page 6: ML Model Performance & Evaluation Dashboard.
Presents genuine held-out test split evaluation metrics (MAE, RMSE, R²),
actual vs predicted scatter plots with 45-degree reference lines,
residual distributions, tree feature importances, and data leakage safeguards.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.ui_components import apply_custom_css, render_hero_banner, render_kpi
from utils.formatting import format_inr
from ml.predict import load_model_pipeline
from ml.train import run_training_pipeline
from config import MODEL_METRICS_PATH

st.set_page_config(page_title="Model Performance — SmartDeal AI", page_icon="🎯", layout="wide")
apply_custom_css()

render_hero_banner(
    title="Machine Learning Model Performance",
    subtitle="Rigorous empirical evaluation on a 20% held-out test split. Verified metrics without data leakage.",
    badge="Model Registry & Evaluation"
)

# Load metrics
try:
    pipeline, metrics = load_model_pipeline()
except Exception as e:
    st.error(f"Error loading model metrics: {e}")
    st.stop()

# Header Action: Retrain Button
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    st.markdown("#### 🏆 Benchmark Leaderboard")
    st.caption(f"Selected Champion Model: **{metrics.get('best_model_name')}** (Selection Criteria: {metrics.get('selection_metric')})")

with top_col2:
    if st.button("🔄 Retrain All Models", use_container_width=True):
        with st.spinner("Retraining Baseline, Linear, Random Forest, and Gradient Boosting pipelines..."):
            best_m, updated_meta = run_training_pipeline()
            pipeline, metrics = load_model_pipeline(force_reload=True)
            st.success(f"Retraining complete! Best model: {best_m}")
            st.rerun()

# Model Comparison Table
model_comp = metrics.get("model_comparison", {})
comp_rows = []

for m_name, vals in model_comp.items():
    comp_rows.append({
        "Candidate Model": m_name,
        "MAE (₹)": format_inr(vals["mae"]),
        "RMSE (₹)": format_inr(vals["rmse"]),
        "R² Score": f"{vals['r2']:.4f}",
        "MAPE (%)": f"{vals.get('mape', 0.0):.2f}%",
        "Status": "👑 Champion Pipeline" if m_name == metrics.get("best_model_name") else "Evaluated"
    })

df_comp = pd.DataFrame(comp_rows)
st.dataframe(df_comp, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# Train/Test Split & Leakage Summary Cards
split_c1, split_c2, split_c3, split_c4 = st.columns(4)
with split_c1:
    render_kpi("Training Samples", f"{metrics.get('train_samples', 96)} Records", "80% partition", "#3B82F6")
with split_c2:
    render_kpi("Held-out Test Samples", f"{metrics.get('test_samples', 24)} Records", "Strict 20% out-of-sample", "#10B981")
with split_c3:
    render_kpi("Target Variable", metrics.get("target_variable", "fair_reference_price"), "Equilibrium market benchmark", "#8B5CF6")
with split_c4:
    render_kpi("Reproducibility Seed", f"Seed = {metrics.get('random_seed', 42)}", "Zero stochastic drift", "#F59E0B")

st.markdown("<br>", unsafe_allow_html=True)

# Plots: Actual vs Predicted & Residuals
actuals = np.array(metrics.get("test_actuals", []))
predictions_dict = metrics.get("test_predictions", {})
best_model_name = metrics.get("best_model_name", "Linear Regression")
best_preds = np.array(predictions_dict.get(best_model_name, []))

plot_col1, plot_col2 = st.columns(2)

if len(actuals) > 0 and len(best_preds) > 0:
    with plot_col1:
        st.markdown(f"#### 🎯 Actual vs Predicted Price ({best_model_name})")
        
        fig_actual_pred = go.Figure()

        # Scatter of actual vs pred
        fig_actual_pred.add_trace(go.Scatter(
            x=actuals,
            y=best_preds,
            mode="markers",
            marker=dict(size=9, color="#38BDF8", line=dict(width=1, color="#F8FAFC")),
            name="Test Observations",
            hovertemplate="<b>Actual:</b> ₹%{x:,.0f}<br><b>Predicted:</b> ₹%{y:,.0f}<extra></extra>"
        ))

        # 45-degree ideal fit line
        min_val = min(actuals.min(), best_preds.min())
        max_val = max(actuals.max(), best_preds.max())
        fig_actual_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#10B981", dash="dash", width=2),
            name="Ideal 45° Parity Line"
        ))

        fig_actual_pred.update_layout(
            paper_bgcolor="rgba(19, 30, 54, 0.5)",
            plot_bgcolor="rgba(11, 17, 32, 0.85)",
            margin=dict(l=25, r=25, t=35, b=45),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, font=dict(color="#94A3B8", family="Plus Jakarta Sans", size=11)),
            xaxis=dict(title="Actual Held-Out Fair Price (₹)", tickprefix="₹", gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans")),
            yaxis=dict(title="Predicted Fair Price (₹)", tickprefix="₹", gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans")),
            hoverlabel=dict(bgcolor="#0F172A", font=dict(family="Plus Jakarta Sans", color="#F8FAFC")),
        )
        st.plotly_chart(fig_actual_pred, use_container_width=True)

    with plot_col2:
        st.markdown(f"#### 📉 Residual Plot (Prediction Errors)")
        residuals = actuals - best_preds

        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=best_preds,
            y=residuals,
            mode="markers",
            marker=dict(size=8, color="#F59E0B"),
            name="Residuals",
            hovertemplate="<b>Fitted:</b> ₹%{x:,.0f}<br><b>Residual:</b> ₹%{y:,.0f}<extra></extra>"
        ))
        # Zero error horizontal line
        fig_res.add_hline(y=0, line_dash="solid", line_color="#EF4444", line_width=1.5)

        fig_res.update_layout(
            paper_bgcolor="rgba(19, 30, 54, 0.5)",
            plot_bgcolor="rgba(11, 17, 32, 0.85)",
            margin=dict(l=25, r=25, t=35, b=45),
            xaxis=dict(title="Fitted / Predicted Price (₹)", tickprefix="₹", gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans")),
            yaxis=dict(title="Residual Error [Actual - Pred] (₹)", tickprefix="₹", gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans")),
            hoverlabel=dict(bgcolor="#0F172A", font=dict(family="Plus Jakarta Sans", color="#F8FAFC")),
        )
        st.plotly_chart(fig_res, use_container_width=True)

# Feature Importances Section
st.markdown("<br>", unsafe_allow_html=True)
importances = metrics.get("feature_importances", [])

if importances:
    st.markdown("#### 🌳 Top Feature Contributions (Tree-Based Importance)")
    st.caption("Relative weight of input features in predicting fair market reference price.")
    df_imp = pd.DataFrame(importances).head(10)

    fig_imp = px.bar(
        df_imp,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        labels={"importance": "Importance Weight", "feature": "Engineered Feature"}
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(19, 30, 54, 0.5)",
        plot_bgcolor="rgba(11, 17, 32, 0.85)",
        margin=dict(l=25, r=25, t=25, b=35),
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed", tickfont=dict(color="#94A3B8", size=11, family="Plus Jakarta Sans")),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94A3B8", family="Plus Jakarta Sans")),
        hoverlabel=dict(bgcolor="#0F172A", font=dict(family="Plus Jakarta Sans", color="#F8FAFC")),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# Data Leakage Precautions and Limitations Section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 🛡️ Data Leakage Precautions & Academic Defensibility")

leak_c1, leak_c2 = st.columns(2)

with leak_c1:
    st.markdown(
        """
        <div class="content-box">
            <h5 style="color: #60A5FA; margin-top: 0;">Target Leakage Prevention</h5>
            <ul style="color: #94A3B8; font-size: 0.88rem; line-height: 1.6;">
                <li><b>No Current Price in Feature Matrix:</b> <code>current_price</code> and <code>discount_percentage</code> are strictly excluded from the regression features. If fed, the model would trivially reconstruct current prices rather than learning intrinsic fair market value.</li>
                <li><b>Isolated Preprocessing:</b> Imputers and Scalers are fitted inside a <code>scikit-learn Pipeline</code> solely on the training split, preventing statistical leakage from test sets.</li>
                <li><b>Strict Chronological History:</b> Time-series forecasting uses strictly past-to-future chronological holdout sets, never random train-test splitting.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with leak_c2:
    st.markdown(
        """
        <div class="content-box">
            <h5 style="color: #F87171; margin-top: 0;">Known Limitations & Failure Modes</h5>
            <ul style="color: #94A3B8; font-size: 0.88rem; line-height: 1.6;">
                <li><b>Extrapolation Risk:</b> Querying prices far outside the observed training domain (e.g. ₹5,00,000 for budget earphones) will produce wider uncertainty intervals.</li>
                <li><b>Small Sample R² Interpretation:</b> On very small or highly homogenous test slices, R² can fluctuate significantly; MAE and RMSE in Indian Rupees remain the primary operational metric.</li>
                <li><b>Exogenous Retail Factors:</b> Models cannot anticipate unannounced manufacturer cashback or bank-specific credit card instant discounts.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
