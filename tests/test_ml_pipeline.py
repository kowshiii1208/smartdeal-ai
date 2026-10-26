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
    assert pred_result["is_ood"] is False


def test_ml_reproducibility():
    """Test that training with fixed RANDOM_SEED yields 100% reproducible metrics."""
    from ml.train import train_and_evaluate_models
    from config import CLEANED_PRODUCTS_PATH, SAMPLE_PRODUCTS_PATH

    path = CLEANED_PRODUCTS_PATH if CLEANED_PRODUCTS_PATH.exists() else SAMPLE_PRODUCTS_PATH
    df = pd.read_csv(path)

    best_1, _, meta_1 = train_and_evaluate_models(df)
    best_2, _, meta_2 = train_and_evaluate_models(df)

    assert best_1 == best_2
    for m in meta_1["model_comparison"]:
        assert meta_1["model_comparison"][m]["mae"] == meta_2["model_comparison"][m]["mae"]
        assert meta_1["model_comparison"][m]["rmse"] == meta_2["model_comparison"][m]["rmse"]
        assert meta_1["model_comparison"][m]["r2"] == meta_2["model_comparison"][m]["r2"]


def test_ood_detection_and_uncertainty_inflation():
    """
    Test that inputs outside training distribution (extreme MSRP, unseen category, unseen brand)
    trigger OOD warnings, inflate uncertainty bounds, and enforce physical price boundaries.
    """
    ood_input = {
        "category": "QuantumComputingExotics",  # Unseen category
        "brand": "FuturisticAlienTech",         # Unseen brand
        "original_price": 1_500_000,            # 7x maximum training MSRP (approx 200,000)
        "rating": 4.9,
        "review_count": 50,
    }

    pred_result = predict_fair_price(ood_input)

    # 1. Verification of OOD flags and warnings
    assert pred_result["is_ood"] is True
    assert len(pred_result["warnings"]) >= 2
    assert any("QuantumComputingExotics" in w for w in pred_result["warnings"])
    assert any("FuturisticAlienTech" in w for w in pred_result["warnings"])
    assert any("original_price" in w for w in pred_result["warnings"])

    # 2. Dynamic uncertainty inflation check
    base_rmse = pred_result["rmse"]
    effective_rmse = pred_result["effective_rmse"]
    assert effective_rmse > base_rmse
    # With 3 warnings, inflation should be 1 + 3 * 0.35 = 2.05x
    assert effective_rmse >= base_rmse * 2.0

    # 3. Physical domain bounds check
    # Predicted price must never exceed 98% of MSRP or fall below 10% of MSRP
    orig_price = ood_input["original_price"]
    assert pred_result["predicted_price"] <= orig_price * 0.98
    assert pred_result["predicted_price"] >= orig_price * 0.10
    assert pred_result["max_expected_price"] <= orig_price
    assert pred_result["min_expected_price"] >= orig_price * 0.10


def test_baseline_comparison_superiority():
    """Verify that all supervised learning models outperform the dummy median baseline."""
    from ml.predict import load_model_pipeline

    _, metrics = load_model_pipeline()
    comp = metrics["model_comparison"]

    assert "Baseline (Dummy Median)" in comp
    dummy_rmse = comp["Baseline (Dummy Median)"]["rmse"]

    for model_name, res in comp.items():
        if model_name != "Baseline (Dummy Median)":
            # Real models must have substantially lower RMSE than dummy median
            assert res["rmse"] < dummy_rmse
            assert res["r2"] > 0.85

