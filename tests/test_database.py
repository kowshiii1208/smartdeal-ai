"""
Unit tests for SQLite database transactions, schema initialization, and persistence.
"""

import pytest
import pandas as pd
from datetime import datetime

from database.db import (
    init_db,
    insert_or_replace_products,
    insert_price_history,
    log_prediction,
    fetch_df,
    fetch_all,
)
from services.product_service import add_to_watchlist, get_watchlist, remove_from_watchlist


def test_database_initialization():
    """Test database schema creation and table existence."""
    success = init_db(force_recreate=False)
    assert success is True

    tables = fetch_all("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = [t["name"] for t in tables]

    assert "products" in table_names
    assert "price_history" in table_names
    assert "predictions" in table_names
    assert "watchlist" in table_names


def test_insert_and_fetch_products():
    """Test bulk product insertion and query retrieval."""
    test_df = pd.DataFrame([{
        "product_id": "TEST-PROD-999",
        "product_name": "Test Mechanical Keyboard",
        "category": "Accessories",
        "brand": "Keychron",
        "original_price": 8999.0,
        "current_price": 7499.0,
        "discount_percentage": 16.7,
        "rating": 4.8,
        "review_count": 450,
        "retailer": "Amazon India",
        "timestamp": datetime.now().isoformat(),
        "fair_reference_price": 7600.0,
        "is_synthetic": 1,
    }])

    rows_affected = insert_or_replace_products(test_df)
    assert rows_affected >= 1

    fetched = fetch_df("SELECT * FROM products WHERE product_id = 'TEST-PROD-999'")
    assert not fetched.empty
    assert fetched["brand"].iloc[0] == "Keychron"
    assert fetched["current_price"].iloc[0] == 7499.0


def test_prediction_logging():
    """Test logging user inference into the predictions audit table."""
    row_count = log_prediction(
        category="Smartphones",
        brand="Apple",
        original_price=79900.0,
        rating=4.5,
        review_count=1200,
        predicted_price=71500.0,
        model_name="Gradient Boosting Regressor",
        deal_score=78.5,
        deal_rating="Good Deal"
    )
    assert row_count == 1

    recent = fetch_df("SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT 1")
    assert not recent.empty
    assert recent["brand"].iloc[0] == "Apple"
    assert recent["predicted_price"].iloc[0] == 71500.0


def test_watchlist_workflow():
    """Test adding, querying, and removing products from user watchlist."""
    prod_id = "TEST-PROD-999"
    # Add to watchlist
    add_success = add_to_watchlist(prod_id, target_price=7000.0, notes="Target for Diwali Sale")
    assert add_success is True

    # Retrieve watchlist
    w_df = get_watchlist()
    assert not w_df.empty
    assert prod_id in w_df["product_id"].values

    # Remove from watchlist
    del_success = remove_from_watchlist(prod_id)
    assert del_success is True

    w_df_after = get_watchlist()
    assert prod_id not in w_df_after["product_id"].values
