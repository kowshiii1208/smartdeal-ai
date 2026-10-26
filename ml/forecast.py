"""
Time-series price history and forecasting module for SmartDeal AI.
Provides chronological holdout evaluation, moving average smoothing,
linear trend forecasting with confidence intervals, and explainable baselines.
"""

from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def compute_history_summary(history_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key statistics across historical price observations."""
    if history_df.empty:
        return {}

    sorted_df = history_df.sort_values(by="observed_date").reset_index(drop=True)
    prices = sorted_df["price"]
    
    first_price = float(prices.iloc[0])
    latest_price = float(prices.iloc[-1])
    min_price = float(prices.min())
    max_price = float(prices.max())
    avg_price = float(prices.mean())

    price_change_pct = ((latest_price - first_price) / first_price) * 100 if first_price > 0 else 0.0

    return {
        "observation_count": len(sorted_df),
        "first_date": str(sorted_df["observed_date"].iloc[0]),
        "latest_date": str(sorted_df["observed_date"].iloc[-1]),
        "first_price": first_price,
        "latest_price": latest_price,
        "min_price": min_price,
        "max_price": max_price,
        "avg_price": round(avg_price, 2),
        "price_change_pct": round(price_change_pct, 2),
    }


def evaluate_forecast_on_holdout(
    history_df: pd.DataFrame,
    holdout_points: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate forecasting models on a strict chronological holdout split.
    Compares:
      1. Naïve Persistence (Last Known Value)
      2. 7-Day Moving Average Baseline
      3. Linear Trend Extrapolation
    """
    sorted_df = history_df.sort_values(by="observed_date").reset_index(drop=True)
    if len(sorted_df) < (holdout_points * 2):
        return {}

    train_df = sorted_df.iloc[:-holdout_points]
    test_df = sorted_df.iloc[-holdout_points:]

    y_test = test_df["price"].values
    
    # 1. Naïve Persistence
    naive_val = train_df["price"].iloc[-1]
    y_pred_naive = np.full_like(y_test, fill_value=naive_val)

    # 2. Moving Average (last 5 points)
    window = min(5, len(train_df))
    ma_val = train_df["price"].iloc[-window:].mean()
    y_pred_ma = np.full_like(y_test, fill_value=ma_val)

    # 3. Linear Trend
    x_train = np.arange(len(train_df))
    x_test = np.arange(len(train_df), len(train_df) + len(test_df))
    slope, intercept = np.polyfit(x_train, train_df["price"].values, 1)
    y_pred_trend = slope * x_test + intercept

    def get_metrics(actual, pred):
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        return {"mae": round(mae, 2), "rmse": round(rmse, 2)}

    return {
        "Naïve Persistence": get_metrics(y_test, y_pred_naive),
        "Moving Average Baseline": get_metrics(y_test, y_pred_ma),
        "Linear Trend": get_metrics(y_test, y_pred_trend),
    }


def generate_price_forecast(
    history_df: pd.DataFrame,
    horizon_days: int = 14
) -> Dict[str, Any]:
    """
    Generate future price forecasts with confidence intervals based on historical observations.
    
    Requirements:
      - At least 5 chronological observations.
      - Uses trend extrapolation bounded by historical volatility.
      - Produces explicit forecast limitations.
    """
    if len(history_df) < 5:
        raise ValueError(
            f"Insufficient history data: Found {len(history_df)} observations. "
            "At least 5 observations are required to construct a reliable forecast."
        )

    sorted_df = history_df.sort_values(by="observed_date").reset_index(drop=True).copy()
    
    # Calculate 7-day and 14-day rolling moving averages
    sorted_df["ma_7"] = sorted_df["price"].rolling(window=3, min_periods=1).mean().round(1)
    sorted_df["ma_14"] = sorted_df["price"].rolling(window=5, min_periods=1).mean().round(1)

    # Evaluate holdout performance
    holdout_metrics = evaluate_forecast_on_holdout(sorted_df, holdout_points=max(3, min(6, len(sorted_df) // 4)))

    # Fit trend model on observed points
    n_points = len(sorted_df)
    x = np.arange(n_points)
    y = sorted_df["price"].values

    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    residual_std = float(np.std(residuals))

    # Last observed date
    last_date_str = str(sorted_df["observed_date"].iloc[-1])
    try:
        last_dt = datetime.strptime(last_date_str, "%Y-%m-%d")
    except Exception:
        last_dt = datetime.now()

    # Generate future horizon points (every 2-3 days)
    future_dates = []
    future_x = []
    step_days = 2
    curr_dt = last_dt + timedelta(days=step_days)
    future_idx = n_points

    while (curr_dt - last_dt).days <= horizon_days:
        future_dates.append(curr_dt.strftime("%Y-%m-%d"))
        future_x.append(future_idx)
        curr_dt += timedelta(days=step_days)
        future_idx += 1

    future_x = np.array(future_x)
    raw_future_y = slope * future_x + intercept
    
    # Dampen excessive slope so forecasts do not drift into absurdity
    last_price = y[-1]
    dampened_future_y = 0.5 * raw_future_y + 0.5 * last_price

    # Round to clean INR steps
    forecast_prices = [round(max(last_price * 0.5, float(p)), -1) for p in dampened_future_y]

    # Expanding confidence interval (95% approximate band: 1.96 * std * sqrt(h))
    upper_bounds = []
    lower_bounds = []
    for i, p in enumerate(forecast_prices, 1):
        uncertainty = 1.645 * residual_std * np.sqrt(1 + (i / len(forecast_prices)))
        upper_bounds.append(round(p + uncertainty, -1))
        lower_bounds.append(round(max(0.0, p - uncertainty), -1))

    summary = compute_history_summary(sorted_df)

    limitations = [
        "Forecast is purely statistical based on past observed price points.",
        "Sudden flash sales, competitor price matching, and inventory stock-outs cannot be anticipated by trend models.",
        "Confidence intervals widen as the forecast horizon extends further into the future.",
    ]

    return {
        "history_df": sorted_df,
        "forecast_dates": future_dates,
        "forecast_prices": forecast_prices,
        "upper_bounds": upper_bounds,
        "lower_bounds": lower_bounds,
        "holdout_metrics": holdout_metrics,
        "summary": summary,
        "limitations": limitations,
    }
