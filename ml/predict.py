"""
Inference and prediction service for SmartDeal AI.
Loads persisted model pipelines safely, executes single and batch predictions,
and generates confidence ranges and explanatory factor summaries.
"""

import json
import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from config import (
    MODEL_PIPELINE_PATH,
    MODEL_METRICS_PATH,
)
from ml.features import validate_inference_input, create_inference_dataframe

logger = logging.getLogger(__name__)

# Cached model and metrics in memory
_CACHED_PIPELINE = None
_CACHED_METRICS = None


def load_model_pipeline(force_reload: bool = False):
    """
    Safely load the fitted model pipeline.
    Returns (pipeline, metrics_dict).
    Raises FileNotFoundError if model has not been trained yet.
    """
    global _CACHED_PIPELINE, _CACHED_METRICS

    if _CACHED_PIPELINE is not None and not force_reload:
        return _CACHED_PIPELINE, _CACHED_METRICS

    if not MODEL_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PIPELINE_PATH}. "
            f"Please run 'python -m ml.train' to train and save the model."
        )

    try:
        pipeline = joblib.load(MODEL_PIPELINE_PATH)
        metrics = {}
        if MODEL_METRICS_PATH.exists():
            with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        _CACHED_PIPELINE = pipeline
        _CACHED_METRICS = metrics
        return pipeline, metrics
    except Exception as e:
        logger.error(f"Error loading model pipeline: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load trained model pipeline: {e}")


def predict_fair_price(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict fair market reference price for a given product feature specification.
    
    Returns:
        {
            "predicted_price": float,
            "min_expected_price": float,
            "max_expected_price": float,
            "model_name": str,
            "mae": float,
            "rmse": float,
            "r2": float,
            "warnings": List[str],
            "factors": Dict[str, Any]
        }
    """
    pipeline, metrics = load_model_pipeline()
    training_bounds = metrics.get("training_bounds", {})

    # Validate inputs
    is_valid, errors, warnings = validate_inference_input(input_data, training_bounds)
    if not is_valid:
        raise ValueError(f"Input validation error: {'; '.join(errors)}")

    # Format into DataFrame
    df_input = create_inference_dataframe(input_data)

    # Inference
    raw_pred = pipeline.predict(df_input)[0]
    predicted_price = round(max(0.0, float(raw_pred)), -1)

    # Compute expected tolerance band using model RMSE
    best_model_name = metrics.get("best_model_name", "Trained ML Model")
    model_stats = metrics.get("model_comparison", {}).get(best_model_name, {})
    rmse = model_stats.get("rmse", predicted_price * 0.05)
    mae = model_stats.get("mae", predicted_price * 0.04)
    r2 = model_stats.get("r2", 0.85)

    min_expected = max(0.0, round(predicted_price - rmse, -1))
    max_expected = round(min(input_data["original_price"], predicted_price + rmse), -1)

    # Explanation factors
    orig_price = float(input_data["original_price"])
    implied_discount_pct = round(((orig_price - predicted_price) / orig_price) * 100, 1)

    factors = {
        "original_price": orig_price,
        "implied_fair_discount_pct": implied_discount_pct,
        "rating_impact": "Positive (above 4.2★)" if float(input_data.get("rating", 4.0)) >= 4.2 else "Moderate/Standard",
        "market_confidence": "High" if r2 > 0.80 else "Moderate",
    }

    return {
        "predicted_price": predicted_price,
        "min_expected_price": min_expected,
        "max_expected_price": max_expected,
        "model_name": best_model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "warnings": warnings,
        "factors": factors,
    }
