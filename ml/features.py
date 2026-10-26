"""
Feature engineering and schema management for SmartDeal AI.
Provides utilities to validate inference inputs against training domains,
check out-of-range numerical features, and extract transformed feature names.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def validate_inference_input(
    input_data: Dict[str, Any],
    training_bounds: Dict[str, Tuple[float, float]] = None
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate user inputs before passing to ML inference.
    Returns:
        (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check required fields
    for field in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        if field not in input_data or input_data[field] is None:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return False, errors, warnings

    # Validate original_price
    try:
        orig_price = float(input_data["original_price"])
        if orig_price <= 0:
            errors.append("Original price must be greater than zero.")
        elif orig_price > 10_000_000:
            errors.append("Original price exceeds realistic platform limit (₹1,00,00,000).")
    except (ValueError, TypeError):
        errors.append("Original price must be a valid number.")

    # Validate rating
    try:
        rating = float(input_data["rating"])
        if rating < 1.0 or rating > 5.0:
            errors.append("Customer rating must be between 1.0 and 5.0 stars.")
    except (ValueError, TypeError):
        errors.append("Rating must be a valid numeric rating.")

    # Validate review_count
    try:
        review_count = int(input_data["review_count"])
        if review_count < 0:
            errors.append("Review count cannot be negative.")
    except (ValueError, TypeError):
        errors.append("Review count must be a non-negative integer.")

    # Check bounds against training distributions if provided
    if training_bounds and not errors:
        # Extract numerical bounds (support both nested dict and flat dict formats)
        num_bounds = training_bounds.get("numerical", training_bounds) if isinstance(training_bounds, dict) else {}
        
        for num_feat in ["original_price", "rating", "review_count"]:
            if num_feat in num_bounds and num_feat in input_data:
                bound_val = num_bounds[num_feat]
                if isinstance(bound_val, (list, tuple)) and len(bound_val) == 2:
                    min_val, max_val = bound_val
                    val = float(input_data[num_feat])
                    if val < min_val:
                        warnings.append(
                            f"Value for '{num_feat}' ({val:,.0f}) is lower than minimum training data ({min_val:,.0f}). Extrapolation may occur."
                        )
                    elif val > max_val:
                        warnings.append(
                            f"Value for '{num_feat}' ({val:,.0f}) is higher than maximum training data ({max_val:,.0f}). Extrapolation may occur."
                        )

        # Check for unobserved categorical inputs (Out-of-Distribution categories and brands)
        known_cats = training_bounds.get("known_categories") if isinstance(training_bounds, dict) else None
        if known_cats and input_data.get("category") not in known_cats:
            warnings.append(
                f"Category '{input_data.get('category')}' was not in training dataset. Inference will use baseline category weights."
            )

        known_brands = training_bounds.get("known_brands") if isinstance(training_bounds, dict) else None
        if known_brands and input_data.get("brand") not in known_brands:
            warnings.append(
                f"Brand '{input_data.get('brand')}' was not observed during training. Valuation will rely on category and MSRP features."
            )

    return len(errors) == 0, errors, warnings


def create_inference_dataframe(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """Format single input dictionary into a 1-row DataFrame ready for pipeline inference."""
    row = {
        "original_price": float(input_dict["original_price"]),
        "rating": float(input_dict.get("rating", 4.0)),
        "review_count": int(input_dict.get("review_count", 100)),
        "category": str(input_dict["category"]),
        "brand": str(input_dict["brand"]),
    }
    return pd.DataFrame([row])


def get_feature_names_from_preprocessor(preprocessor) -> List[str]:
    """Extract human-readable feature names after one-hot encoding and scaling."""
    feature_names = []
    try:
        for name, transformer, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                onehot = transformer.named_steps.get("onehot")
                if onehot and hasattr(onehot, "get_feature_names_out"):
                    encoded_cols = onehot.get_feature_names_out(cols)
                    feature_names.extend(encoded_cols)
                else:
                    feature_names.extend(cols)
    except Exception:
        # Fallback
        feature_names = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    return feature_names
