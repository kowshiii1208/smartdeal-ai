"""
Unit tests for price history analysis, moving average smoothing,
and time-series forecasting in SmartDeal AI.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ml.forecast import (
    generate_price_forecast,
    compute_history_summary,
    evaluate_forecast_on_holdout,
)


def test_insufficient_history_rejection():
    """Test that fewer than 5 observations raises ValueError with actionable explanation."""
    short_history = pd.DataFrame({
        "product_id": ["P1", "P1", "P1"],
        "observed_date": ["2026-10-01", "2026-10-03", "2026-10-05"],
        "price": [50000, 49000, 49500],
    })

    with pytest.raises(ValueError) as excinfo:
        generate_price_forecast(short_history, horizon_days=14)

    assert "Insufficient history data" in str(excinfo.value)
    assert "At least 5 observations are required" in str(excinfo.value)


def test_valid_price_forecast_generation():
    """Test full forecast generation for a product with 15 historical points."""
    dates = [(datetime(2026, 9, 1) + timedelta(days=i * 3)).strftime("%Y-%m-%d") for i in range(15)]
    prices = [50000 - (i * 200) + (100 if i % 2 == 0 else -100) for i in range(15)]

    history_df = pd.DataFrame({
        "product_id": ["PROD-1001"] * 15,
        "observed_date": dates,
        "price": prices,
    })

    forecast_res = generate_price_forecast(history_df, horizon_days=14)

    assert "forecast_dates" in forecast_res
    assert len(forecast_res["forecast_dates"]) > 0
    assert len(forecast_res["forecast_prices"]) == len(forecast_res["forecast_dates"])
    assert len(forecast_res["upper_bounds"]) == len(forecast_res["forecast_dates"])
    assert len(forecast_res["lower_bounds"]) == len(forecast_res["forecast_dates"])

    # Bounds sanity
    for p, u, l in zip(forecast_res["forecast_prices"], forecast_res["upper_bounds"], forecast_res["lower_bounds"]):
        assert u >= p
        assert l <= p

    # Summary sanity
    summary = forecast_res["summary"]
    assert summary["min_price"] <= summary["max_price"]
    assert summary["observation_count"] == 15


def test_chronological_holdout_evaluation():
    """Test out-of-sample holdout evaluation compares naive, moving average, and trend."""
    dates = [(datetime(2026, 8, 1) + timedelta(days=i * 2)).strftime("%Y-%m-%d") for i in range(20)]
    prices = [30000 + (i * 150) for i in range(20)]

    history_df = pd.DataFrame({
        "product_id": ["P1"] * 20,
        "observed_date": dates,
        "price": prices,
    })

    metrics = evaluate_forecast_on_holdout(history_df, holdout_points=5)

    assert "Naïve Persistence" in metrics
    assert "Moving Average Baseline" in metrics
    assert "Linear Trend" in metrics
    assert "mae" in metrics["Linear Trend"]
    assert "rmse" in metrics["Linear Trend"]


def test_adaptive_holdout_for_small_series():
    """Verify that a short time series (5 observations) successfully evaluates chronological holdout."""
    dates = [(datetime(2026, 10, 1) + timedelta(days=i * 2)).strftime("%Y-%m-%d") for i in range(5)]
    prices = [40000, 39500, 39000, 38800, 38500]

    history_df = pd.DataFrame({
        "product_id": ["P-SHORT"] * 5,
        "observed_date": dates,
        "price": prices,
    })

    metrics = evaluate_forecast_on_holdout(history_df)
    assert len(metrics) == 3
    assert "Naïve Persistence" in metrics
    assert "Linear Trend" in metrics
    assert metrics["Linear Trend"]["mae"] >= 0


def test_forecast_confidence_interval_widening():
    """Verify that statistical confidence interval width increases into the future horizon."""
    dates = [(datetime(2026, 9, 1) + timedelta(days=i * 3)).strftime("%Y-%m-%d") for i in range(12)]
    prices = [50000 + (i * 100) + (50 if i % 2 == 0 else -50) for i in range(12)]

    history_df = pd.DataFrame({
        "product_id": ["P-WIDE"] * 12,
        "observed_date": dates,
        "price": prices,
    })

    forecast_res = generate_price_forecast(history_df, horizon_days=14)
    upper = forecast_res["upper_bounds"]
    lower = forecast_res["lower_bounds"]

    widths = [u - l for u, l in zip(upper, lower)]
    # As horizon extends, uncertainty width should generally expand or stay constant due to rounding
    assert widths[-1] >= widths[0]

