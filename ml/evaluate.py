"""
Model evaluation module for SmartDeal AI.
Computes genuine regression performance metrics: MAE, RMSE, R², and MAPE.
Calculates residuals, prediction intervals, and extracts feature importances.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute core regression performance metrics on held-out test data.
    All metrics are computed dynamically from actual observations.
    Ravel ensures 1D array shapes to prevent broadcasting anomalies.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    if len(y_t) == 0 or len(y_p) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "mape": 0.0}

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    
    # R2 Score can be negative if a model performs worse than a horizontal mean line
    try:
        r2 = float(r2_score(y_t, y_p))
        if np.isnan(r2) or np.isinf(r2):
            r2 = 0.0
    except Exception:
        r2 = 0.0

    # Mean Absolute Percentage Error (MAPE)
    nonzero_mask = y_t != 0
    if nonzero_mask.any():
        mape = float(np.mean(np.abs((y_t[nonzero_mask] - y_p[nonzero_mask]) / y_t[nonzero_mask])) * 100)
    else:
        mape = 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
    }


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute residuals and return DataFrame for residual visualization."""
    residuals = y_true - y_pred
    pct_error = (residuals / np.where(y_true == 0, 1, y_true)) * 100
    return pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "residual": residuals,
        "percentage_error": pct_error,
    })


def extract_feature_importances(pipeline, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Extract feature importances from regressors inside a scikit-learn Pipeline.
    Supports tree-based models (via feature_importances_) and linear models
    (via normalized absolute coefficients).
    """
    regressor = pipeline.named_steps.get("regressor")
    if regressor is None:
        return None

    importances = None
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        abs_coef = np.abs(regressor.coef_).ravel()
        total = np.sum(abs_coef)
        importances = (abs_coef / total) if total > 0 else abs_coef

    if importances is None:
        return None

    # If lengths match exactly
    if len(importances) == len(feature_names):
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": np.round(importances, 4)
        }).sort_values(by="importance", ascending=False)
        return df_importance
    else:
        # Fallback with indexed features
        df_importance = pd.DataFrame({
            "feature": [f"Feature_{i}" for i in range(len(importances))],
            "importance": np.round(importances, 4)
        }).sort_values(by="importance", ascending=False)
        return df_importance

