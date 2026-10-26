"""
Page 2: Smart Deal Finder.
Algorithmic bargain detection engine that evaluates listed prices against fair reference prices,
providing multi-criteria filtering, sorting, deal scoring (0-100), and evidence justification.
"""

import streamlit as st
import pandas as pd

from utils.ui_components import apply_custom_css, render_hero_banner, render_kpi, render_empty_state, render_product_card_html, render_html
from utils.formatting import format_inr, format_percent, get_deal_badge_html
from services.product_service import get_all_products, filter_and_sort_products, add_to_watchlist

st.set_page_config(page_title="Smart Deal Finder — SmartDeal AI", page_icon="🏷️", layout="wide")
apply_custom_css()

render_hero_banner(
    title="Smart Deal Finder",
    subtitle="Detect authentic bargains by comparing listed prices against fair market benchmarks rather than inflated MSRP discounts.",
    badge="Algorithmic Deal Ranking"
)

# Fetch all products
all_products = get_all_products()

if all_products.empty:
    render_empty_state("No Products Found", "The product catalog appears to be empty. Please populate sample data.")
    st.stop()

# Sidebar Filters
with st.sidebar:
    st.markdown("### 🔍 **Filter & Search Deals**")
    
    search_query = st.text_input("Search Product / Brand", placeholder="e.g. MacBook, Sony, Galaxy...")
    
    all_categories = sorted(all_products["category"].unique().tolist())
    selected_categories = st.multiselect("Category", options=all_categories, default=[])
    
    # Filter brands by selected categories
    if selected_categories:
        available_brands = sorted(all_products[all_products["category"].isin(selected_categories)]["brand"].unique().tolist())
    else:
        available_brands = sorted(all_products["brand"].unique().tolist())
    selected_brands = st.multiselect("Brand", options=available_brands, default=[])

    # Price Range Slider
    min_catalog_price = float(all_products["current_price"].min())
    max_catalog_price = float(all_products["current_price"].max())
    if min_catalog_price >= max_catalog_price:
        max_catalog_price = min_catalog_price + 1000.0
    
    price_range = st.slider(
        "Price Range (₹)",
        min_value=int(min_catalog_price),
        max_value=int(max_catalog_price),
        value=(int(min_catalog_price), int(max_catalog_price)),
        step=max(100, int((max_catalog_price - min_catalog_price) // 50) or 500)
    )

    # Deal Category Tiers
    deal_tiers_options = ["Great Deal", "Good Deal", "Fair Price", "Above Reference"]
    selected_deal_tiers = st.multiselect("Deal Rating", options=deal_tiers_options, default=["Great Deal", "Good Deal"])

    # Minimum Rating
    min_rating = st.slider("Minimum Customer Rating", min_value=3.0, max_value=5.0, value=3.5, step=0.1)

    # Sorting
    sort_options = [
        "Best Deal Score",
        "Highest Savings (₹)",
        "Lowest Price",
        "Highest Price",
        "Highest Rated",
        "Largest Discount %",
    ]
    selected_sort = st.selectbox("Sort By", options=sort_options, index=0)

# Filter dataset
filtered_df = filter_and_sort_products(
    all_products,
    search_query=search_query,
    categories=selected_categories if selected_categories else None,
    brands=selected_brands if selected_brands else None,
    min_price=float(price_range[0]),
    max_price=float(price_range[1]),
    deal_tiers=selected_deal_tiers if selected_deal_tiers else None,
    min_rating=min_rating,
    sort_by=selected_sort,
)

# Results Header & Metrics Bar
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    render_kpi("Deals Matching", f"{len(filtered_df)} Items", "Filtered results count", "#3B82F6", icon="🔍")
with col_m2:
    avg_score = filtered_df["deal_score"].mean() if not filtered_df.empty else 0
    render_kpi("Average Deal Score", f"{avg_score:.1f} / 100", "Quality of matched deals", "#10B981", icon="⭐")
with col_m3:
    max_savings = filtered_df["estimated_savings"].max() if not filtered_df.empty else 0
    render_kpi("Max Potential Savings", format_inr(max_savings), "Single item highest savings", "#8B5CF6", icon="💰")
with col_m4:
    great_count = len(filtered_df[filtered_df["deal_category"] == "Great Deal"])
    render_kpi("Great Deals Found", f"{great_count} Items", "Score >= 80/100", "#06B6D4", icon="🔥")

st.markdown("<br>", unsafe_allow_html=True)

# Transparent Deal Formula Explainer Expander
with st.expander("ℹ️ **How Deal Scores are Computed (Mathematical Formula)**"):
    st.markdown(
        """
        SmartDeal AI does not rely on subjective or fabricated discount badges. Deal scores are computed objectively:
        
        1. **Fair Price Difference**: `diff_pct = (reference_price - current_price) / reference_price * 100`
        2. **Base Score**: 50.0 (indicates fair market equilibrium)
        3. **Reference Discount Contribution**: `+ 2.5 * diff_pct` (clipped between -35 and +35)
        4. **MSRP Discount Contribution**: `+ 0.25 * mrp_discount_pct` (clipped between 0 and +12)
        5. **Rating Boost**: `+ 5.0 * (rating - 4.0)` (if rating ≥ 4.0)
        6. **Confidence Dampening**: Scores regress slightly toward 50 if review volume or historical observations are low.
        
        *Result*: 0 to 100 Bounded Score. Classified into: **Great Deal (80-100)**, **Good Deal (65-79)**, **Fair Price (45-64)**, **Above Reference (<45)**.
        """
    )

# Product Cards Grid
if filtered_df.empty:
    render_empty_state("No Deals Found", "No products match your current filter combination. Try broadening your price range or deal tier filters.")
else:
    # Display in 2-column responsive layout
    card_cols = st.columns(2)
    for idx, (_, row) in enumerate(filtered_df.iterrows()):
        col_to_use = card_cols[idx % 2]
        with col_to_use:
            # HTML Card Container
            render_html(render_product_card_html(row, show_score_bar=True))
            # Add to Watchlist quick button
            btn_key = f"watch_{row['product_id']}_{idx}"
            if st.button(f"⭐ Add {row['product_id']} to Watchlist", key=btn_key):
                success = add_to_watchlist(row["product_id"], target_price=row["current_price"] * 0.95, notes="Added from Smart Deal Finder")
                if success:
                    st.success(f"Added {row['product_name'][:30]}... to your Watchlist!")
                else:
                    st.info("Item is already in your Watchlist.")
