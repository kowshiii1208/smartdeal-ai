"""
Unit tests for ML preprocessing, pipeline training, target leakage safeguards,
and inference prediction in SmartDeal AI.
"""

import pytest
import pandas as pd
import numpy as np

from ml.preprocessing import build_preprocessor, prepare_features_and_target
from ml.evaluate import evaluate_regression_model
from ml.features import validate_inference_input, create_inference_dataframe
from ml.predict import predict_fair_price, load_model_pipeline
from config import TARGET_VARIABLE


def test_target_leakage_prevention():
    """Verify that current_price and discount_percentage are strictly excluded from X."""
    df = pd.DataFrame({
        "original_price": [100000, 50000],
        "current_price": [90000, 45000],          # Potential leakage column!
        "discount_percentage": [10.0, 10.0],       # Potential leakage column!
        "rating": [4.5, 4.2],
        "review_count": [1000, 500],
        "category": ["Laptops", "Smartphones"],
        "brand": ["Apple", "Samsung"],
        "fair_reference_price": [92000, 46000],    # Target column
    })

    X, y = prepare_features_and_target(df, target_col="fair_reference_price")

    # Critical assertions
    assert "current_price" not in X.columns
    assert "discount_percentage" not in X.columns
    assert "fair_reference_price" not in X.columns
    assert "original_price" in X.columns
    assert len(y) == 2


def test_unseen_categorical_feature_handling():
    """Test that OneHotEncoder handles unseen categories and brands without failing."""
    preprocessor = build_preprocessor()

    train_df = pd.DataFrame({
        "original_price": [80000, 40000],
        "rating": [4.5, 4.2],
        "review_count": [1000, 500],
        "category": ["Laptops", "Smartphones"],
        "brand": ["Apple", "Samsung"],
    })
    preprocessor.fit(train_df)

    # Test with totally unseen brand and category
    test_df = pd.DataFrame({
        "original_price": [60000],
        "rating": [4.0],
        "review_count": [200],
        "category": ["UnseenCategoryX"],
        "brand": ["UnknownBrandY"],
    })

    transformed = preprocessor.transform(test_df)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_evaluate_regression_model():
    """Test regression metric calculations (MAE, RMSE, R2, MAPE)."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])

    metrics = evaluate_regression_model(y_true, y_pred)

    assert metrics["mae"] == 10.0
    assert metrics["rmse"] == 10.0
    assert metrics["r2"] > 0.95
    assert metrics["mape"] > 0


def test_inference_input_validation():
    """Test input validation for live prediction form."""
    # Valid input
    valid_input = {
        "category": "Laptops",
        "brand": "Apple",
        "original_price": 114900,
        "rating": 4.5,
        "review_count": 2500,
    }
    is_valid, errors, warnings = validate_inference_input(valid_input)
    assert is_valid is True
    assert len(errors) == 0

    # Invalid price (<= 0)
    invalid_input = valid_input.copy()
    invalid_input["original_price"] = -500
    is_valid, errors, warnings = validate_inference_input(invalid_input)
    assert is_valid is False
    assert any("greater than zero" in e for e in errors)

    # Invalid rating (> 5)
    invalid_rating = valid_input.copy()
    invalid_rating["rating"] = 5.5
    is_valid, errors, warnings = validate_inference_input(invalid_rating)
    assert is_valid is False


def test_predict_fair_price_pipeline():
    """Test live prediction through the fitted pipeline."""
    sample_input = {
        "category": "Laptops",
        "brand": "Apple",
        "original_price": 114900,
        "rating": 4.6,
        "review_count": 3500,
    }
    pred_result = predict_fair_price(sample_input)

    assert "predicted_price" in pred_result
    assert pred_result["predicted_price"] > 0
    assert pred_result["min_expected_price"] <= pred_result["predicted_price"]
    assert pred_result["max_expected_price"] >= pred_result["predicted_price"]
    assert pred_result["r2"] > 0
