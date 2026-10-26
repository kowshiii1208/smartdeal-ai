"""
Unit tests for data validation, schema cleaning, and anomaly detection in SmartDeal AI.
"""

import pytest
import pandas as pd
import numpy as np

from utils.validation import validate_and_clean_products, validate_price_history


def test_valid_product_catalog():
    """Test that a well-formed catalog passes validation with correct summary stats."""
    data = {
        "product_id": ["P1", "P2"],
        "product_name": ["MacBook Air M2", "Sony WH-1000XM5"],
        "category": ["Laptops", "Audio & Headphones"],
        "brand": ["Apple", "Sony"],
        "original_price": [114900, 29990],
        "current_price": [99990, 24990],
        "rating": [4.7, 4.6],
        "review_count": [1200, 3400],
        "retailer": ["Amazon India", "Flipkart"],
    }
    df = pd.DataFrame(data)
    is_valid, errors, warnings, stats, cleaned = validate_and_clean_products(df)

    assert is_valid is True
    assert len(errors) == 0
    assert stats["valid_rows"] == 2
    assert "discount_percentage" in cleaned.columns


def test_missing_mandatory_columns():
    """Test that missing required columns fails validation."""
    data = {
        "product_id": ["P1"],
        "product_name": ["MacBook Air"],
        # Missing original_price, current_price, brand, category
    }
    df = pd.DataFrame(data)
    is_valid, errors, warnings, stats, cleaned = validate_and_clean_products(df)

    assert is_valid is False
    assert any("Missing mandatory columns" in e for e in errors)
    assert cleaned is None


def test_negative_or_zero_prices():
    """Test that non-positive prices are rejected."""
    data = {
        "product_id": ["P1"],
        "product_name": ["Faulty Product"],
        "category": ["Laptops"],
        "brand": ["Apple"],
        "original_price": [50000],
        "current_price": [-100],  # Negative price!
    }
    df = pd.DataFrame(data)
    is_valid, errors, warnings, stats, cleaned = validate_and_clean_products(df)

    assert is_valid is False
    assert any("negative or zero prices" in e for e in errors)


def test_duplicate_product_id_handling():
    """Test that duplicate product IDs are detected and deduplicated."""
    data = {
        "product_id": ["P1", "P1"],  # Duplicate ID
        "product_name": ["Item 1", "Item 1 Duplicate"],
        "category": ["Laptops", "Laptops"],
        "brand": ["Dell", "Dell"],
        "original_price": [60000, 60000],
        "current_price": [52000, 52000],
    }
    df = pd.DataFrame(data)
    is_valid, errors, warnings, stats, cleaned = validate_and_clean_products(df)

    assert is_valid is True
    assert stats["duplicates_removed"] == 1
    assert len(cleaned) == 1


def test_out_of_bounds_rating_clamping():
    """Test that customer ratings outside [0, 5] are clamped safely."""
    data = {
        "product_id": ["P1", "P2"],
        "product_name": ["Item A", "Item B"],
        "category": ["Smartphones", "Smartphones"],
        "brand": ["Samsung", "OnePlus"],
        "original_price": [40000, 30000],
        "current_price": [35000, 25000],
        "rating": [6.5, -1.0],  # Both out of range
    }
    df = pd.DataFrame(data)
    is_valid, errors, warnings, stats, cleaned = validate_and_clean_products(df)

    assert is_valid is True
    assert cleaned["rating"].iloc[0] == 5.0
    assert cleaned["rating"].iloc[1] == 0.0


def test_price_history_validation():
    """Test price history validation for proper sorting, positive prices, and dates."""
    data = {
        "product_id": ["P1", "P1"],
        "observed_date": ["2026-10-15", "2026-10-10"],  # Out of order
        "price": [45000, 48000],
    }
    df = pd.DataFrame(data)
    is_valid, errors, warnings, cleaned = validate_price_history(df)

    assert is_valid is True
    # Should be sorted chronologically
    assert cleaned["observed_date"].iloc[0] == "2026-10-10"
    assert cleaned["observed_date"].iloc[1] == "2026-10-15"
