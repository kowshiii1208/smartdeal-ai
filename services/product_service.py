"""
Product catalog and query service for SmartDeal AI.
Provides filtered querying, multi-attribute sorting, side-by-side comparison logic,
and SQLite watchlist persistence.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from config import (
    CLEANED_PRODUCTS_PATH,
    SAMPLE_PRODUCTS_PATH,
    PRODUCT_CATEGORIES,
)
from database.db import (
    fetch_df,
    fetch_all,
    execute_query,
    insert_or_replace_products,
    init_db,
)
from services.deal_engine import enrich_products_with_deals
from data.data_generator import save_demonstration_data


def get_all_products(force_refresh: bool = False) -> pd.DataFrame:
    """
    Retrieve all products from SQLite database or fallback to CSV,
    enriched with real-time deal metrics.
    """
    # Try fetching from SQLite first
    try:
        init_db()
        df = fetch_df("SELECT * FROM products")
        if df.empty or len(df) < 10:
            # Seed from demonstration CSV
            if not SAMPLE_PRODUCTS_PATH.exists():
                save_demonstration_data()
            seed_df = pd.read_csv(SAMPLE_PRODUCTS_PATH)
            insert_or_replace_products(seed_df)
            df = fetch_df("SELECT * FROM products")
    except Exception:
        # Fallback to local CSV
        if CLEANED_PRODUCTS_PATH.exists():
            df = pd.read_csv(CLEANED_PRODUCTS_PATH)
        elif SAMPLE_PRODUCTS_PATH.exists():
            df = pd.read_csv(SAMPLE_PRODUCTS_PATH)
        else:
            p_df, _ = save_demonstration_data()
            df = p_df

    # Calculate real-time deal metrics
    enriched = enrich_products_with_deals(df)
    return enriched


def filter_and_sort_products(
    df: pd.DataFrame,
    search_query: str = "",
    categories: Optional[List[str]] = None,
    brands: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    deal_tiers: Optional[List[str]] = None,
    min_rating: Optional[float] = None,
    sort_by: str = "Best Deal Score",
) -> pd.DataFrame:
    """Filter and sort product catalog based on user criteria."""
    filtered = df.copy()

    # Search Query
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        mask = (
            filtered["product_name"].str.lower().str.contains(q, na=False) |
            filtered["brand"].str.lower().str.contains(q, na=False) |
            filtered["category"].str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]

    # Category Filter
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]

    # Brand Filter
    if brands:
        filtered = filtered[filtered["brand"].isin(brands)]

    # Price Filter
    if min_price is not None:
        filtered = filtered[filtered["current_price"] >= min_price]
    if max_price is not None:
        filtered = filtered[filtered["current_price"] <= max_price]

    # Deal Tier Filter
    if deal_tiers:
        filtered = filtered[filtered["deal_category"].isin(deal_tiers)]

    # Rating Filter
    if min_rating is not None and min_rating > 0:
        filtered = filtered[filtered["rating"] >= min_rating]

    # Sorting
    if sort_by == "Best Deal Score":
        filtered = filtered.sort_values(by="deal_score", ascending=False)
    elif sort_by == "Highest Savings (₹)":
        filtered = filtered.sort_values(by="estimated_savings", ascending=False)
    elif sort_by == "Lowest Price":
        filtered = filtered.sort_values(by="current_price", ascending=True)
    elif sort_by == "Highest Price":
        filtered = filtered.sort_values(by="current_price", ascending=False)
    elif sort_by == "Highest Rated":
        filtered = filtered.sort_values(by="rating", ascending=False)
    elif sort_by == "Largest Discount %":
        filtered = filtered.sort_values(by="discount_percentage", ascending=False)

    return filtered.reset_index(drop=True)


def get_product_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full product details for a given product_id."""
    df = get_all_products()
    matched = df[df["product_id"] == product_id]
    if not matched.empty:
        return matched.iloc[0].to_dict()
    return None


def compare_selected_products(product_ids: List[str]) -> Dict[str, Any]:
    """
    Generate side-by-side comparison matrix for 2-4 products.
    Identifies Lowest Price product vs Best Supported Deal separately.
    """
    all_df = get_all_products()
    selected = all_df[all_df["product_id"].isin(product_ids)].copy()

    if selected.empty:
        return {"products": [], "lowest_price_id": None, "best_deal_id": None}

    # Lowest price product
    lowest_price_row = selected.sort_values(by="current_price", ascending=True).iloc[0]
    lowest_price_id = lowest_price_row["product_id"]

    # Best supported deal product (highest deal score)
    best_deal_row = selected.sort_values(by="deal_score", ascending=False).iloc[0]
    best_deal_id = best_deal_row["product_id"]

    return {
        "products": selected.to_dict(orient="records"),
        "lowest_price_id": lowest_price_id,
        "best_deal_id": best_deal_id,
        "categories_differ": len(selected["category"].unique()) > 1,
    }


def add_to_watchlist(product_id: str, target_price: float, notes: str = "") -> bool:
    """Add a product to SQLite watchlist."""
    from datetime import datetime
    try:
        query = """
        INSERT OR REPLACE INTO watchlist (product_id, target_price, added_date, notes)
        VALUES (?, ?, ?, ?)
        """
        execute_query(query, (product_id, target_price, datetime.now().strftime("%Y-%m-%d"), notes))
        return True
    except Exception:
        return False


def get_watchlist() -> pd.DataFrame:
    """Get all watchlisted products joined with current prices."""
    query = """
    SELECT 
        w.watchlist_id,
        w.product_id,
        w.target_price,
        w.added_date,
        w.notes,
        p.product_name,
        p.category,
        p.brand,
        p.current_price,
        p.original_price,
        p.rating,
        p.retailer
    FROM watchlist w
    JOIN products p ON w.product_id = p.product_id
    ORDER BY w.added_date DESC
    """
    try:
        df = fetch_df(query)
        if not df.empty:
            df["price_met"] = df["current_price"] <= df["target_price"]
        return df
    except Exception:
        return pd.DataFrame()


def remove_from_watchlist(product_id: str) -> bool:
    """Remove product from watchlist."""
    try:
        execute_query("DELETE FROM watchlist WHERE product_id = ?", (product_id,))
        return True
    except Exception:
        return False
