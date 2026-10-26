"""
Unit tests for the Smart Deal Engine and rating classifications in SmartDeal AI.
"""

import pytest
import pandas as pd
from services.deal_engine import calculate_deal_metrics, enrich_products_with_deals
from utils.formatting import get_deal_tier


def test_great_deal_classification():
    """Test that a substantial discount below reference price achieves a Great Deal score (>= 80)."""
    metrics = calculate_deal_metrics(
        current_price=70000,
        reference_price=85000,
        original_price=100000,
        rating=4.7,
        review_count=5000,
        has_history=True
    )

    assert metrics["deal_score"] >= 80
    assert metrics["deal_category"] == "Great Deal"
    assert metrics["estimated_savings"] == 15000
    assert metrics["percentage_difference"] > 0
    assert "below estimated fair market reference" in metrics["explanation"]


def test_above_reference_classification():
    """Test that a product listed above its reference price is classified as Above Reference (< 45)."""
    metrics = calculate_deal_metrics(
        current_price=92000,
        reference_price=80000,
        original_price=100000,
        rating=4.0,
        review_count=300,
        has_history=True
    )

    assert metrics["deal_score"] < 45
    assert metrics["deal_category"] == "Above Reference"
    assert metrics["estimated_savings"] == 0.0
    assert metrics["percentage_difference"] < 0
    assert "above estimated fair reference price" in metrics["explanation"]


def test_fair_price_equilibrium():
    """Test that a product priced right at its reference benchmark receives a Fair Price rating (~50)."""
    metrics = calculate_deal_metrics(
        current_price=50000,
        reference_price=50000,
        original_price=55000,
        rating=4.0,
        review_count=500,
        has_history=True
    )

    assert 45 <= metrics["deal_score"] <= 65
    assert metrics["deal_category"] == "Fair Price"
    assert metrics["percentage_difference"] == 0.0


def test_evidence_confidence_regression():
    """Test that low review volume causes the score to regress towards fair equilibrium (50)."""
    high_evidence = calculate_deal_metrics(
        current_price=70000,
        reference_price=85000,
        original_price=100000,
        rating=4.5,
        review_count=10000,
        has_history=True
    )

    low_evidence = calculate_deal_metrics(
        current_price=70000,
        reference_price=85000,
        original_price=100000,
        rating=4.5,
        review_count=10,  # Very low reviews
        has_history=False
    )

    # Low evidence score should be damped towards 50
    assert low_evidence["deal_score"] < high_evidence["deal_score"]
    assert low_evidence["confidence_level"] == "Low Evidence"


def test_enrich_products_with_deals():
    """Test bulk DataFrame enrichment with deal attributes."""
    df = pd.DataFrame([{
        "product_id": "P1",
        "product_name": "Test Laptop",
        "current_price": 50000.0,
        "original_price": 60000.0,
        "fair_reference_price": 55000.0,
        "rating": 4.5,
        "review_count": 800,
    }])

    enriched = enrich_products_with_deals(df)

    assert "deal_score" in enriched.columns
    assert "deal_category" in enriched.columns
    assert "estimated_savings" in enriched.columns
    assert enriched["estimated_savings"].iloc[0] == 5000.0
