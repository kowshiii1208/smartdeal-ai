"""
Page 1: Overview Dashboard.
Provides high-level catalog analytics, category pricing distributions,
retailer distribution, and interactive summary charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.ui_components import apply_custom_css, render_hero_banner, render_kpi
from utils.formatting import format_inr, format_percent
from services.product_service import get_all_products

st.set_page_config(page_title="Overview Dashboard — SmartDeal AI", page_icon="📊", layout="wide")
apply_custom_css()

render_hero_banner(
    title="Overview Dashboard & Macro Analytics",
    subtitle="Macro-level e-commerce market intelligence: category pricing patterns, discount distributions, and catalog health.",
    badge="Market Analytics",
    icon="📊"
)

# Fetch current catalog
df = get_all_products()

if df.empty:
    st.warning("Product catalog is currently empty. Please initialize demonstration data.")
    st.stop()

# Macro KPI Cards
col1, col2, col3, col4 = st.columns(4)

total_items = len(df)
avg_discount = df["discount_percentage"].mean()
median_price = df["current_price"].median()
unique_brands = df["brand"].nunique()

with col1:
    render_kpi("Tracked Items", f"{total_items:,}", "Active catalog inventory", "#3B82F6", icon="📦")
with col2:
    render_kpi("Avg. Market Discount", f"{avg_discount:.1f}%", "Against manufacturer MRP", "#10B981", icon="🏷️")
with col3:
    render_kpi("Median Price", format_inr(median_price), "Typical product price", "#8B5CF6", icon="💵")
with col4:
    render_kpi("Brands Tracked", f"{unique_brands} Brands", f"Across {df['category'].nunique()} categories", "#F59E0B", icon="🏢")

st.markdown("<br>", unsafe_allow_html=True)

# Charts Section: Row 1
chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    st.markdown("#### 📊 Average Listed Price vs MRP by Category")
    cat_summary = df.groupby("category")[["current_price", "original_price"]].mean().reset_index()
    cat_summary = cat_summary.sort_values(by="current_price", ascending=False)

    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        x=cat_summary["category"],
        y=cat_summary["original_price"],
        name="Original MRP",
        marker=dict(color="#475569", line=dict(width=0))
    ))
    fig_cat.add_trace(go.Bar(
        x=cat_summary["category"],
        y=cat_summary["current_price"],
        name="Current Selling Price",
        marker=dict(color="#38BDF8", line=dict(width=0))
    ))
    fig_cat.update_layout(
        barmode="group",
        paper_bgcolor="rgba(19, 30, 54, 0.5)",
        plot_bgcolor="rgba(11, 17, 32, 0.85)",
        margin=dict(l=25, r=25, t=35, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, font=dict(color="#94A3B8", family="Plus Jakarta Sans", size=11)),
        xaxis=dict(tickangle=-25, tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans")),
        yaxis=dict(tickprefix="₹", tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans"), gridcolor="rgba(255, 255, 255, 0.05)"),
        hoverlabel=dict(bgcolor="#0F172A", font=dict(family="Plus Jakarta Sans", color="#F8FAFC")),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with chart_col2:
    st.markdown("#### 🍩 Category Distribution")
    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]

    fig_pie = px.pie(
        cat_counts,
        values="Count",
        names="Category",
        hole=0.60,
        color_discrete_sequence=px.colors.sequential.Tealgrn_r
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(19, 30, 54, 0.5)",
        plot_bgcolor="rgba(19, 30, 54, 0.5)",
        margin=dict(l=15, r=15, t=35, b=35),
        font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
        showlegend=False,
        hoverlabel=dict(bgcolor="#0F172A", font=dict(family="Plus Jakarta Sans", color="#F8FAFC")),
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0B1120', width=2)))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Charts Section: Row 2
chart_col3, chart_col4 = st.columns([1, 1])

with chart_col3:
    st.markdown("#### 📦 Price Distribution by Category (Box Plot)")
    fig_box = px.box(
        df,
        x="category",
        y="current_price",
        color="category",
        points="outliers",
        labels={"category": "Category", "current_price": "Price (₹)"}
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(19, 30, 54, 0.5)",
        plot_bgcolor="rgba(11, 17, 32, 0.85)",
        margin=dict(l=25, r=25, t=35, b=55),
        showlegend=False,
        xaxis=dict(tickangle=-25, tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans")),
        yaxis=dict(tickprefix="₹", tickfont=dict(color="#94A3B8", size=10, family="Plus Jakarta Sans"), gridcolor="rgba(255, 255, 255, 0.05)"),
        hoverlabel=dict(bgcolor="#0F172A", font=dict(family="Plus Jakarta Sans", color="#F8FAFC")),
    )
    st.plotly_chart(fig_box, use_container_width=True)

with chart_col4:
    st.markdown("#### 🏪 Retailer Inventory Distribution")
    retailer_summary = df["retailer"].value_counts().reset_index()
    retailer_summary.columns = ["Retailer", "Listings"]

    fig_ret = px.bar(
        retailer_summary,
        x="Listings",
        y="Retailer",
        orientation="h",
        color="Listings",
        color_continuous_scale="Tealgrn",
        labels={"Listings": "Number of Offers", "Retailer": "Retail Platform"}
    )
    fig_ret.update_layout(
        paper_bgcolor="rgba(19, 30, 54, 0.5)",
        plot_bgcolor="rgba(11, 17, 32, 0.85)",
        margin=dict(l=25, r=25, t=35, b=35),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94A3B8", family="Plus Jakarta Sans")),
        yaxis=dict(tickfont=dict(color="#94A3B8", family="Plus Jakarta Sans")),
        hoverlabel=dict(bgcolor="#0F172A", font=dict(family="Plus Jakarta Sans", color="#F8FAFC")),
    )
    st.plotly_chart(fig_ret, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Category Aggregation Table
st.markdown("#### 📋 Category Summary Benchmarks")
cat_agg = df.groupby("category").agg(
    Products=("product_id", "count"),
    Min_Price=("current_price", "min"),
    Median_Price=("current_price", "median"),
    Max_Price=("current_price", "max"),
    Avg_Discount=("discount_percentage", "mean"),
    Avg_Rating=("rating", "mean"),
).reset_index()

# Format currency and percentages for display
cat_agg["Min Price"] = cat_agg["Min_Price"].apply(format_inr)
cat_agg["Median Price"] = cat_agg["Median_Price"].apply(format_inr)
cat_agg["Max Price"] = cat_agg["Max_Price"].apply(format_inr)
cat_agg["Avg Discount"] = cat_agg["Avg_Discount"].apply(lambda v: f"{v:.1f}%")
cat_agg["Avg Rating"] = cat_agg["Avg_Rating"].apply(lambda v: f"★ {v:.2f}")

display_df = cat_agg[["category", "Products", "Min Price", "Median Price", "Max Price", "Avg Discount", "Avg Rating"]].rename(
    columns={"category": "Category"}
)

st.dataframe(display_df, use_container_width=True, hide_index=True)
