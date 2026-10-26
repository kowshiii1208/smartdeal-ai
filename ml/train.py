"""
Model training and pipeline selection module for SmartDeal AI.
Trains and compares Baseline, Linear Regression, Random Forest, and Gradient Boosting.
Selects best performing model based on held-out test metrics and persists it with Joblib.
"""

import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

from config import (
    RANDOM_SEED,
    TEST_SIZE,
    MODEL_PIPELINE_PATH,
    MODEL_METRICS_PATH,
    SAMPLE_PRODUCTS_PATH,
    CLEANED_PRODUCTS_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_VARIABLE,
)
from utils.validation import validate_and_clean_products
from ml.preprocessing import build_preprocessor, prepare_features_and_target
from ml.evaluate import evaluate_regression_model, extract_feature_importances
from ml.features import get_feature_names_from_preprocessor
from data.data_generator import save_demonstration_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate regression models and baseline."""
    return {
        "Baseline (Dummy Median)": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=RANDOM_SEED,
        ),
    }


def train_and_evaluate_models(
    df: pd.DataFrame
) -> Tuple[str, Pipeline, Dict[str, Any]]:
    """
    Train and evaluate all candidate models using a strict held-out test split.
    Selects the best model based on held-out RMSE.
    """
    logger.info("Preparing features and target...")
    X, y = prepare_features_and_target(df, target_col=TARGET_VARIABLE)

    # Train/Test Split (80/20) with reproducible random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    logger.info(f"Split data into {len(X_train)} training and {len(X_test)} testing records.")

    # Calculate training feature bounds strictly from X_train to prevent data leakage
    training_bounds = {
        "numerical": {col: (float(X_train[col].min()), float(X_train[col].max())) for col in NUMERICAL_FEATURES},
        "known_categories": sorted(X_train["category"].dropna().unique().tolist()),
        "known_brands": sorted(X_train["brand"].dropna().unique().tolist()),
    }

    preprocessor = build_preprocessor()

    candidate_models = get_candidate_models()
    model_results = {}
    fitted_pipelines = {}

    best_model_name = None
    best_rmse = float("inf")

    # Store predictions for the test set for all models
    test_actuals = y_test.tolist()
    model_test_predictions = {}

    for name, regressor in candidate_models.items():
        logger.info(f"Training {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", regressor),
            ]
        )
        # Fit on train split only
        pipeline.fit(X_train, y_train)

        # Predict on held-out test split
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_regression_model(y_test.values, y_pred)

        model_results[name] = metrics
        fitted_pipelines[name] = pipeline
        model_test_predictions[name] = [round(float(p), 1) for p in y_pred]

        logger.info(f"{name} -> MAE: Rs.{metrics['mae']:,.2f}, RMSE: Rs.{metrics['rmse']:,.2f}, R2: {metrics['r2']:.4f}")

        # Choose best based on RMSE (excluding dummy baseline)
        if name != "Baseline (Dummy Median)" and metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model_name = name

    # If no candidate beat dummy (unlikely), default to Random Forest
    if not best_model_name:
        best_model_name = "Random Forest Regressor"

    winning_pipeline = fitted_pipelines[best_model_name]
    logger.info(f"Selected Best Model: {best_model_name} (RMSE: Rs.{best_rmse:,.2f})")

    # Extract feature importances if available
    fitted_preprocessor = winning_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names_from_preprocessor(fitted_preprocessor)
    df_importances = extract_feature_importances(winning_pipeline, feature_names)
    importances_list = df_importances.to_dict(orient="records") if df_importances is not None else []

    # Compile comprehensive metrics payload
    evaluation_metadata = {
        "best_model_name": best_model_name,
        "selection_metric": "RMSE on held-out 20% test split",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_variable": TARGET_VARIABLE,
        "features_used": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "training_bounds": training_bounds,
        "model_comparison": model_results,
        "test_actuals": test_actuals,
        "test_predictions": model_test_predictions,
        "feature_importances": importances_list[:15],  # top 15 features
        "random_seed": RANDOM_SEED,
    }

    return best_model_name, winning_pipeline, evaluation_metadata


def run_training_pipeline() -> Tuple[str, Dict[str, Any]]:
    """Complete end-to-end training routine: load, clean, train, evaluate, and save."""
    # Ensure demonstration data exists
    if not SAMPLE_PRODUCTS_PATH.exists():
        logger.info("Demonstration data not found. Generating now...")
        save_demonstration_data()

    # Load and clean product data
    logger.info(f"Loading products from {SAMPLE_PRODUCTS_PATH}...")
    raw_df = pd.read_csv(SAMPLE_PRODUCTS_PATH)
    is_valid, errors, warnings, stats, cleaned_df = validate_and_clean_products(raw_df)

    if not is_valid:
        raise ValueError(f"Product dataset validation failed: {errors}")

    for w in warnings:
        logger.warning(f"Data warning: {w}")

    # Save cleaned data
    cleaned_df.to_csv(CLEANED_PRODUCTS_PATH, index=False)
    logger.info(f"Saved cleaned dataset to {CLEANED_PRODUCTS_PATH}")

    # Train and evaluate models
    best_name, best_pipeline, metadata = train_and_evaluate_models(cleaned_df)

    # Persist best model pipeline
    joblib.dump(best_pipeline, MODEL_PIPELINE_PATH)
    logger.info(f"Saved best pipeline model to {MODEL_PIPELINE_PATH}")

    # Persist metrics metadata
    with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metrics to {MODEL_METRICS_PATH}")

    return best_name, metadata


if __name__ == "__main__":
    best_model, meta = run_training_pipeline()
    print("\n" + "=" * 60)
    print(f"SUCCESS: Training Complete! Best Model: {best_model}")
    print("=" * 60)
    for model_name, metrics in meta["model_comparison"].items():
        print(f"{model_name:30} | MAE: Rs.{metrics['mae']:>10,.2f} | RMSE: Rs.{metrics['rmse']:>10,.2f} | R2: {metrics['r2']:>6.4f}")
    print("=" * 60)
