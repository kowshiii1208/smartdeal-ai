"""
Page 5: Product Comparison.
Allows users to compare 2 to 4 comparable products side-by-side,
evaluating current price, fair benchmark, savings, deal scores, and ratings,
while clearly distinguishing the Lowest Price from the Best Supported Deal.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.ui_components import apply_custom_css, render_hero_banner, render_kpi, render_empty_state, render_html
from utils.formatting import format_inr, format_percent, get_deal_badge_html
from services.product_service import get_all_products, compare_selected_products

st.set_page_config(page_title="Compare Products — SmartDeal AI", page_icon="⚖️", layout="wide")
apply_custom_css()

render_hero_banner(
    title="Side-by-Side Product Comparison",
    subtitle="Evaluate comparable alternatives. Distinguish the lowest absolute sticker price from the highest value-for-money deal.",
    badge="Multi-Item Matrix"
)

# Fetch products
all_products = get_all_products()

if all_products.empty:
    render_empty_state("Catalog Empty", "Please populate products to compare.")
    st.stop()

# Category Pre-filter for meaningful comparison
st.markdown("#### 1. Filter by Category to Compare Equivalent Items")
category_options = ["All Categories"] + sorted(all_products["category"].unique().tolist())
selected_cat = st.selectbox("Category Filter", options=category_options, index=1 if len(category_options) > 1 else 0)

if selected_cat != "All Categories":
    selectable_products = all_products[all_products["category"] == selected_cat]
else:
    selectable_products = all_products

# Product Multiselect
product_display_map = {
    f"{row['product_name']} ({row['brand']} • {format_inr(row['current_price'])})": row['product_id']
    for _, row in selectable_products.iterrows()
}

default_selection = list(product_display_map.keys())[:3] if len(product_display_map) >= 3 else list(product_display_map.keys())

selected_labels = st.multiselect(
    "Select 2 to 4 Products to Compare",
    options=list(product_display_map.keys()),
    default=default_selection,
    max_selections=4,
)

if len(selected_labels) < 2:
    st.info("💡 Please select at least 2 products to view the side-by-side comparison.")
    st.stop()

selected_ids = [product_display_map[lbl] for lbl in selected_labels]
comparison = compare_selected_products(selected_ids)

# Warning if comparing items across different categories
if comparison.get("categories_differ", False):
    render_html(
        """
        <div class="callout-warning">
            ⚠️ <b>Cross-Category Comparison Detected:</b><br>
            You have selected products from different categories (e.g., a Smartphone vs a Television). 
            While prices and deal scores can be viewed, please note that comparing prices across different product domains does not imply equivalent utility or value.
        </div>
        """
    )

prods = comparison["products"]
lowest_price_id = comparison["lowest_price_id"]
best_deal_id = comparison["best_deal_id"]

st.markdown("<br>", unsafe_allow_html=True)

# Winner Highlights Banner
h_col1, h_col2 = st.columns(2)

lowest_item = next(p for p in prods if p["product_id"] == lowest_price_id)
best_deal_item = next(p for p in prods if p["product_id"] == best_deal_id)

with h_col1:
    render_html(
        f"""
        <div style="background: rgba(59, 130, 246, 0.1); border: 2px solid #3B82F6; border-radius: 12px; padding: 18px;">
            <div style="font-size: 0.80rem; text-transform: uppercase; font-weight: 700; color: #60A5FA; letter-spacing: 0.05em;">
                💰 Lowest Current Price
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">
                {lowest_item['product_name'][:40]}...
            </div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #38BDF8;">
                {format_inr(lowest_item['current_price'])}
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8;">
                Deal Score: {lowest_item['deal_score']:.0f}/100 • {lowest_item['deal_category']}
            </div>
        </div>
        """
    )

with h_col2:
    render_html(
        f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 2px solid #10B981; border-radius: 12px; padding: 18px;">
            <div style="font-size: 0.80rem; text-transform: uppercase; font-weight: 700; color: #34D399; letter-spacing: 0.05em;">
                ⭐ Best Supported Deal (Highest Value)
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">
                {best_deal_item['product_name'][:40]}...
            </div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #10B981;">
                {format_inr(best_deal_item['current_price'])}
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8;">
                Deal Score: <b>{best_deal_item['deal_score']:.0f}/100</b> ({best_deal_item['deal_category']}) • Est. Savings: {format_inr(best_deal_item['estimated_savings'])}
            </div>
        </div>
        """
    )

st.markdown("<br>", unsafe_allow_html=True)

# Side-by-Side Comparison Columns
st.markdown("#### 📋 Detailed Specification & Valuation Matrix")

comp_cols = st.columns(len(prods))

for idx, p in enumerate(prods):
    with comp_cols[idx]:
        is_lowest = p["product_id"] == lowest_price_id
        is_best_deal = p["product_id"] == best_deal_id

        badge_tags = []
        if is_lowest:
            badge_tags.append('<span style="background: #2563EB; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">LOWEST PRICE</span>')
        if is_best_deal:
            badge_tags.append('<span style="background: #10B981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">BEST DEAL</span>')

        tags_html = " ".join(badge_tags)

        render_html(
            f"""
            <div class="product-card" style="border-top: 3px solid {'#10B981' if is_best_deal else ('#3B82F6' if is_lowest else '#334155')};">
                <div style="margin-bottom: 8px;">{tags_html}</div>
                <div class="product-brand">{p['brand']} • {p['category']}</div>
                <div class="product-name" style="min-height: 48px;">{p['product_name']}</div>
                
                <div style="background: #0B1120; border-radius: 8px; padding: 12px; margin: 12px 0;">
                    <div style="font-size: 0.78rem; color: #94A3B8;">CURRENT PRICE</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #10B981;">{format_inr(p['current_price'])}</div>
                    <div style="font-size: 0.80rem; color: #64748B;">MRP: <del>{format_inr(p['original_price'])}</del> ({p['discount_percentage']:.0f}% Off)</div>
                </div>

                <div style="font-size: 0.85rem; line-height: 1.8;">
                    <div>⚖️ <b>Fair Price:</b> {format_inr(p['fair_reference_price'])}</div>
                    <div>💵 <b>Est. Savings:</b> <span style="color: #10B981; font-weight: 600;">{format_inr(p['estimated_savings'])}</span></div>
                    <div>🎯 <b>Deal Score:</b> {p['deal_score']:.0f}/100</div>
                    <div>🏷️ <b>Tier:</b> {get_deal_badge_html(p['deal_score'], p['deal_category'])}</div>
                    <div>★ <b>Rating:</b> {p['rating']} ({p['review_count']:,} reviews)</div>
                    <div>🏪 <b>Retailer:</b> {p['retailer']}</div>
                </div>

                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 12px; border-top: 1px solid #1E293B; padding-top: 8px; line-height: 1.4;">
                    📝 {p['deal_explanation']}
                </div>
            </div>
            """
        )

# Visual Comparison Chart
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 📊 Comparative Price vs Fair Benchmark")

chart_names = [p["product_name"][:25] + "..." for p in prods]
chart_curr = [p["current_price"] for p in prods]
chart_ref = [p["fair_reference_price"] for p in prods]

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    x=chart_names,
    y=chart_curr,
    name="Current Price",
    marker_color="#10B981"
))
fig_comp.add_trace(go.Bar(
    x=chart_names,
    y=chart_ref,
    name="Fair Reference Benchmark",
    marker_color="#3B82F6"
))

fig_comp.update_layout(
    barmode="group",
    paper_bgcolor="rgba(19, 30, 54, 0.5)",
    plot_bgcolor="rgba(11, 17, 32, 0.85)",
    margin=dict(l=25, r=25, t=35, b=45),
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, font=dict(color="#94A3B8", family="Plus Jakarta Sans", size=11)),
    xaxis=dict(tickfont=dict(color="#94A3B8", size=11, family="Plus Jakarta Sans")),
    yaxis=dict(tickprefix="₹", tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans"), gridcolor="rgba(255, 255, 255, 0.05)"),
    hoverlabel=dict(bgcolor="#0F172A", font=dict(family="Plus Jakarta Sans", color="#F8FAFC")),
)

st.plotly_chart(fig_comp, use_container_width=True)
