"""
Data validation and schema cleaning module for SmartDeal AI.
Ensures uploaded and generated datasets conform to required types, ranges,
and constraints, detecting invalid prices, negative numbers, duplicates, and malformed dates.
"""

from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


REQUIRED_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "brand",
    "current_price",
    "original_price",
]

OPTIONAL_COLUMNS = [
    "rating",
    "review_count",
    "discount_percentage",
    "retailer",
    "timestamp",
    "fair_reference_price",
    "is_synthetic",
]


def validate_and_clean_products(df: pd.DataFrame) -> Tuple[bool, List[str], List[str], Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Validate and clean a product catalog DataFrame.
    Returns:
        (is_valid, errors, warnings, summary_stats, cleaned_df)
    """
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {
        "raw_rows": len(df),
        "valid_rows": 0,
        "dropped_rows": 0,
        "duplicates_removed": 0,
        "missing_values_imputed": 0,
        "anomalous_prices_detected": 0,
    }

    if df.empty:
        errors.append("Dataset is empty. At least one valid product record is required.")
        return False, errors, warnings, stats, None

    # 1. Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing mandatory columns: {', '.join(missing_cols)}")
        return False, errors, warnings, stats, None

    cleaned = df.copy()

    # 2. Check and remove duplicates on product_id
    initial_count = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["product_id"], keep="first")
    duplicates_count = initial_count - len(cleaned)
    stats["duplicates_removed"] = duplicates_count
    if duplicates_count > 0:
        warnings.append(f"Removed {duplicates_count} duplicate product_id entries (kept first occurrence).")

    # 3. Clean and validate numeric price columns
    for price_col in ["current_price", "original_price"]:
        # Attempt conversion to numeric
        cleaned[price_col] = pd.to_numeric(cleaned[price_col], errors="coerce")
        nan_prices = cleaned[price_col].isna().sum()
        if nan_prices > 0:
            errors.append(f"Found {nan_prices} non-numeric or missing values in '{price_col}'.")

    if errors:
        return False, errors, warnings, stats, None

    # Check for negative or zero prices
    invalid_negatives = (cleaned["current_price"] <= 0) | (cleaned["original_price"] <= 0)
    if invalid_negatives.any():
        neg_count = int(invalid_negatives.sum())
        errors.append(f"Found {neg_count} records with negative or zero prices.")
        return False, errors, warnings, stats, None

    # Check for anomalous current_price > original_price (MSRP)
    higher_current = cleaned["current_price"] > (cleaned["original_price"] * 1.05)
    if higher_current.any():
        higher_count = int(higher_current.sum())
        stats["anomalous_prices_detected"] = higher_count
        warnings.append(
            f"{higher_count} products have current_price greater than original_price (MRP). "
            f"This can happen during price gouging or currency mismatch."
        )

    # 4. Clean and normalize optional columns
    # Rating
    if "rating" in cleaned.columns:
        cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
        out_of_bounds = (cleaned["rating"] < 0) | (cleaned["rating"] > 5)
        if out_of_bounds.any():
            warnings.append("Some rating values were outside [0, 5]. Clamped to valid range.")
            cleaned["rating"] = cleaned["rating"].clip(0.0, 5.0)
        impute_rating_count = cleaned["rating"].isna().sum()
        if impute_rating_count > 0:
            stats["missing_values_imputed"] += int(impute_rating_count)
            cleaned["rating"] = cleaned["rating"].fillna(4.0)
    else:
        cleaned["rating"] = 4.0

    # Review count
    if "review_count" in cleaned.columns:
        cleaned["review_count"] = pd.to_numeric(cleaned["review_count"], errors="coerce").fillna(0).astype(int)
        cleaned["review_count"] = cleaned["review_count"].clip(lower=0)
    else:
        cleaned["review_count"] = 0

    # Discount percentage
    if "discount_percentage" in cleaned.columns:
        cleaned["discount_percentage"] = pd.to_numeric(cleaned["discount_percentage"], errors="coerce")
        # Impute or recompute if missing
        missing_disc = cleaned["discount_percentage"].isna()
        if missing_disc.any():
            computed_disc = ((cleaned["original_price"] - cleaned["current_price"]) / cleaned["original_price"] * 100).clip(0, 100)
            cleaned.loc[missing_disc, "discount_percentage"] = computed_disc.loc[missing_disc]
    else:
        cleaned["discount_percentage"] = ((cleaned["original_price"] - cleaned["current_price"]) / cleaned["original_price"] * 100).clip(0, 100)

    cleaned["discount_percentage"] = cleaned["discount_percentage"].round(1)

    # Retailer
    if "retailer" in cleaned.columns:
        cleaned["retailer"] = cleaned["retailer"].fillna("Standard Retailer").astype(str).str.strip()
    else:
        cleaned["retailer"] = "Standard Retailer"

    # Category and Brand string normalization
    cleaned["category"] = cleaned["category"].astype(str).str.strip().str.title()
    cleaned["brand"] = cleaned["brand"].astype(str).str.strip()
    cleaned["product_name"] = cleaned["product_name"].astype(str).str.strip()

    # Timestamp
    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = cleaned["timestamp"].astype(str)
    else:
        cleaned["timestamp"] = datetime.now().isoformat()

    # Synthetic flag
    if "is_synthetic" not in cleaned.columns:
        cleaned["is_synthetic"] = 1

    stats["valid_rows"] = len(cleaned)
    stats["dropped_rows"] = stats["raw_rows"] - stats["valid_rows"]

    return True, errors, warnings, stats, cleaned


def validate_price_history(df: pd.DataFrame) -> Tuple[bool, List[str], List[str], Optional[pd.DataFrame]]:
    """Validate price history observation DataFrame."""
    errors = []
    warnings = []
    required = ["product_id", "observed_date", "price"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Price history is missing required columns: {', '.join(missing)}")
        return False, errors, warnings, None

    cleaned = df.copy()
    cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce")
    if cleaned["price"].isna().any():
        errors.append("Invalid or missing values found in price column.")
        return False, errors, warnings, None

    if (cleaned["price"] <= 0).any():
        errors.append("Price observations must be strictly positive.")
        return False, errors, warnings, None

    # Parse and validate dates
    try:
        cleaned["observed_date"] = pd.to_datetime(cleaned["observed_date"]).dt.strftime("%Y-%m-%d")
    except Exception as e:
        errors.append(f"Malformed dates in price history: {e}")
        return False, errors, warnings, None

    # Sort chronologically
    cleaned = cleaned.sort_values(by=["product_id", "observed_date"]).reset_index(drop=True)

    if "retailer" not in cleaned.columns:
        cleaned["retailer"] = "Market Average"
    if "in_stock" not in cleaned.columns:
        cleaned["in_stock"] = 1
    if "is_synthetic" not in cleaned.columns:
        cleaned["is_synthetic"] = 1

    return True, errors, warnings, cleaned
