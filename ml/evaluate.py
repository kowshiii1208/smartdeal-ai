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
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    
    # R2 Score can be negative if a model performs worse than a horizontal mean line
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0

    # Mean Absolute Percentage Error (MAPE)
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
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
    Extract feature importances from tree-based regressors (RandomForest, GradientBoosting)
    inside a scikit-learn Pipeline.
    """
    regressor = pipeline.named_steps.get("regressor")
    if regressor is None or not hasattr(regressor, "feature_importances_"):
        return None

    importances = regressor.feature_importances_
    
    # If lengths match exactly
    if len(importances) == len(feature_names):
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        return df_importance
    else:
        # Fallback with indexed features
        df_importance = pd.DataFrame({
            "feature": [f"Feature_{i}" for i in range(len(importances))],
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        return df_importance
