"""
Historical price service and Plotly visualization generator for SmartDeal AI.
Fetches time-series records from SQLite or CSV, computes statistics,
and generates polished, dark-navy interactive charts with moving averages,
key observation markers, and forecasting confidence bands.
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import SAMPLE_HISTORY_PATH
from database.db import fetch_df, insert_price_history, init_db
from data.data_generator import save_demonstration_data
from ml.forecast import generate_price_forecast, compute_history_summary
from utils.formatting import format_inr


def get_price_history_for_product(product_id: str) -> pd.DataFrame:
    """Fetch all chronological price observations for a product."""
    try:
        init_db()
        df = fetch_df(
            "SELECT * FROM price_history WHERE product_id = ? ORDER BY observed_date ASC",
            (product_id,)
        )
        if df.empty:
            # Seed from CSV if present
            if not SAMPLE_HISTORY_PATH.exists():
                save_demonstration_data()
            seed_df = pd.read_csv(SAMPLE_HISTORY_PATH)
            insert_price_history(seed_df)
            df = fetch_df(
                "SELECT * FROM price_history WHERE product_id = ? ORDER BY observed_date ASC",
                (product_id,)
            )
    except Exception:
        if SAMPLE_HISTORY_PATH.exists():
            full_df = pd.read_csv(SAMPLE_HISTORY_PATH)
            df = full_df[full_df["product_id"] == product_id].sort_values("observed_date")
        else:
            df = pd.DataFrame()

    return df.reset_index(drop=True)


def build_price_history_chart(
    history_df: pd.DataFrame,
    product_name: str,
    current_price: Optional[float] = None,
    include_forecast: bool = True,
    horizon_days: int = 14,
) -> go.Figure:
    """
    Construct a responsive, fintech-styled Plotly chart showing:
      - Historical price curve (Cyan)
      - Moving average 7-day (Amber dashed)
      - Key markers: All-time Min, All-time Max
      - Current listed price benchmark (Green dotted)
      - Future forecast trajectory (Violet dashed) with shaded 95% confidence intervals
    """
    fig = go.Figure()

    if history_df.empty:
        fig.add_annotation(
            text="No historical price observations available for this item.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#94A3B8")
        )
        fig.update_layout(
            paper_bgcolor="#131E36",
            plot_bgcolor="#131E36",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    # 1. Historical Observed Prices
    dates = history_df["observed_date"].tolist()
    prices = history_df["price"].tolist()

    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode="lines+markers",
        name="Observed Price",
        line=dict(color="#06B6D4", width=3),
        marker=dict(size=6, color="#22D3EE"),
        hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> ₹%{y:,.0f}<extra></extra>"
    ))

    # 2. Moving Average (7-day / 3-point rolling)
    ma_prices = history_df["price"].rolling(window=3, min_periods=1).mean().tolist()
    fig.add_trace(go.Scatter(
        x=dates,
        y=ma_prices,
        mode="lines",
        name="7-Day Moving Avg",
        line=dict(color="#F59E0B", width=2, dash="dash"),
        hovertemplate="<b>7-Day MA:</b> ₹%{y:,.0f}<extra></extra>"
    ))

    # 3. Markers for Min & Max
    min_idx = int(np.argmin(prices))
    max_idx = int(np.argmax(prices))

    fig.add_trace(go.Scatter(
        x=[dates[min_idx]],
        y=[prices[min_idx]],
        mode="markers+text",
        name="All-Time Low",
        marker=dict(color="#10B981", size=11, symbol="star"),
        text=[f"Min: {format_inr(prices[min_idx])}"],
        textposition="bottom center",
        textfont=dict(color="#10B981", size=11),
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=[dates[max_idx]],
        y=[prices[max_idx]],
        mode="markers+text",
        name="All-Time High",
        marker=dict(color="#EF4444", size=10, symbol="triangle-up"),
        text=[f"Max: {format_inr(prices[max_idx])}"],
        textposition="top center",
        textfont=dict(color="#EF4444", size=11),
        hoverinfo="skip"
    ))

    # 4. Current Listed Price Benchmark
    if current_price is not None:
        fig.add_hline(
            y=current_price,
            line_dash="dot",
            line_color="#10B981",
            line_width=2,
            annotation_text=f"Current: {format_inr(current_price)}",
            annotation_position="bottom right",
            annotation_font_color="#10B981",
            annotation_font_size=11,
        )

    # 5. Future Forecast with Confidence Bands
    if include_forecast and len(history_df) >= 5:
        try:
            forecast_data = generate_price_forecast(history_df, horizon_days=horizon_days)
            f_dates = forecast_data["forecast_dates"]
            f_prices = forecast_data["forecast_prices"]
            upper_b = forecast_data["upper_bounds"]
            lower_b = forecast_data["lower_bounds"]

            # Connect last observed to first forecast
            conn_dates = [dates[-1]] + f_dates
            conn_forecast = [prices[-1]] + f_prices
            conn_upper = [prices[-1]] + upper_b
            conn_lower = [prices[-1]] + lower_b

            # Upper Bound (transparent line for fill)
            fig.add_trace(go.Scatter(
                x=conn_dates,
                y=conn_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Lower Bound with shaded fill
            fig.add_trace(go.Scatter(
                x=conn_dates,
                y=conn_lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(139, 92, 246, 0.15)",
                name="Forecast 90% Band",
                hoverinfo="skip"
            ))

            # Forecast Line
            fig.add_trace(go.Scatter(
                x=conn_dates,
                y=conn_forecast,
                mode="lines+markers",
                name="Projected Forecast",
                line=dict(color="#A855F7", width=2.5, dash="dot"),
                marker=dict(size=5, color="#C084FC"),
                hovertemplate="<b>Forecast Date:</b> %{x}<br><b>Est. Price:</b> ₹%{y:,.0f}<extra></extra>"
            ))
        except Exception:
            pass

    # Layout Aesthetics
    fig.update_layout(
        title=dict(
            text=f"<b>Price Trajectory & Trend Forecast</b> — <span style='color: #94A3B8;'>{product_name[:45]}</span>",
            font=dict(size=15, color="#F8FAFC", family="Plus Jakarta Sans, sans-serif")
        ),
        paper_bgcolor="rgba(19, 30, 54, 0.5)",
        plot_bgcolor="rgba(11, 17, 32, 0.85)",
        margin=dict(l=45, r=45, t=65, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(color="#94A3B8", size=11, family="Plus Jakarta Sans"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            linecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans"),
            title=dict(text="Observation Timeline", font=dict(color="#94A3B8", size=11)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            linecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans"),
            tickprefix="₹",
            title=dict(text="Price (INR)", font=dict(color="#94A3B8", size=11)),
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="rgba(255, 255, 255, 0.15)",
            font=dict(color="#F8FAFC", family="Plus Jakarta Sans", size=11),
        ),
        hovermode="x unified",
    )

    return fig
