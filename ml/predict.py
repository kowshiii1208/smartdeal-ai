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

    if not MODEL_PIPELINE_PATH.exists() or not MODEL_METRICS_PATH.exists():
        logger.info(f"Trained model pipeline not found at {MODEL_PIPELINE_PATH}. Auto-training now...")
        try:
            from ml.train import run_training_pipeline
            run_training_pipeline()
        except Exception as e:
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PIPELINE_PATH} and auto-training failed: {e}. "
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
    orig_price = float(input_data["original_price"])

    # Inference
    raw_pred = pipeline.predict(df_input)[0]

    # Enforce physical domain constraints:
    # A fair reference price cannot be negative, near zero, or exceed original price (MSRP)
    min_physical = orig_price * 0.10
    max_physical = orig_price * 0.98
    bounded_pred = max(min_physical, min(max_physical, float(raw_pred)))
    predicted_price = round(bounded_pred, -1)

    # Compute expected tolerance band using model RMSE
    best_model_name = metrics.get("best_model_name", "Trained ML Model")
    model_stats = metrics.get("model_comparison", {}).get(best_model_name, {})
    rmse = model_stats.get("rmse", predicted_price * 0.05)
    mae = model_stats.get("mae", predicted_price * 0.04)
    r2 = model_stats.get("r2", 0.85)

    # Dynamic Uncertainty Inflation for Out-of-Distribution (OOD) inputs
    # If the input was extrapolated beyond training bounds or contains unseen categories/brands,
    # expand the uncertainty tolerance band by 35% per warning.
    is_ood = len(warnings) > 0
    inflation_factor = 1.0 + (0.35 * len(warnings))
    effective_rmse = round(rmse * inflation_factor, 1)

    min_expected = max(min_physical, round(predicted_price - effective_rmse, -1))
    max_expected = min(orig_price, round(predicted_price + effective_rmse, -1))

    # Explanation factors
    implied_discount_pct = round(((orig_price - predicted_price) / orig_price) * 100, 1)

    factors = {
        "original_price": orig_price,
        "implied_fair_discount_pct": implied_discount_pct,
        "rating_impact": "Positive (above 4.2★)" if float(input_data.get("rating", 4.0)) >= 4.2 else "Moderate/Standard",
        "market_confidence": "High (In-Domain)" if not is_ood and r2 > 0.80 else ("Moderate" if not is_ood else "Cautious (OOD Extrapolation)"),
        "uncertainty_status": "Standard (In-Distribution)" if not is_ood else f"Inflated (±Rs.{effective_rmse:,.0f} due to {len(warnings)} extrapolation warning(s))",
    }

    return {
        "predicted_price": predicted_price,
        "min_expected_price": min_expected,
        "max_expected_price": max_expected,
        "model_name": best_model_name,
        "mae": mae,
        "rmse": rmse,
        "effective_rmse": effective_rmse,
        "is_ood": is_ood,
        "r2": r2,
        "warnings": warnings,
        "factors": factors,
    }

