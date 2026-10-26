"""
Reusable UI components and custom CSS styling for SmartDeal AI.
Provides fintech-grade dark navy aesthetics, glassmorphism, responsive metric cards,
animated deal badges, and consistent typography across all Streamlit pages.
"""

import streamlit as st
from utils.formatting import format_inr, format_percent, get_deal_badge_html


CUSTOM_CSS = """
<style>
/* SmartDeal AI — Premium Fintech Dark Navy Theme */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Background Atmosphere */
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.08) 0%, transparent 35%),
                radial-gradient(circle at 90% 20%, rgba(139, 92, 246, 0.07) 0%, transparent 40%),
                radial-gradient(circle at 50% 90%, rgba(6, 182, 212, 0.05) 0%, transparent 40%),
                #0B1120 !important;
    color: #F8FAFC;
}

/* Subtle Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0B1120;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3B82F6;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C1322 0%, #070B14 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.06) !important;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(20, 32, 57, 0.75) 0%, rgba(12, 19, 34, 0.9) 100%);
    border: 1px solid rgba(59, 130, 246, 0.22);
    border-left: 4px solid #3B82F6;
    padding: 24px 28px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -8px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}

.hero-banner::after {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 250px;
    height: 100%;
    background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.15), transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.025em;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #94A3B8;
    line-height: 1.55;
    max-width: 880px;
}

/* Metric KPI Container */
.kpi-container {
    background: rgba(19, 30, 54, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
}

.kpi-container:hover {
    border-color: rgba(59, 130, 246, 0.4);
    box-shadow: 0 12px 28px -4px rgba(59, 130, 246, 0.18);
    transform: translateY(-3px);
}

.kpi-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #94A3B8;
}

.kpi-icon {
    font-size: 1.15rem;
    opacity: 0.85;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    line-height: 1.15;
}

.kpi-subtext {
    font-size: 0.80rem;
    color: #64748B;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* Product Card */
.product-card {
    background: rgba(19, 30, 54, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.product-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 14px 32px -4px rgba(6, 182, 212, 0.15);
    transform: translateY(-3px);
}

.product-brand-chip {
    font-size: 0.72rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #38BDF8;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.2);
    padding: 2px 8px;
    border-radius: 6px;
}

.product-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 8px 0 10px 0;
    line-height: 1.45;
}

.product-price-main {
    font-size: 1.55rem;
    font-weight: 800;
    color: #10B981;
    letter-spacing: -0.02em;
}

.product-price-orig {
    font-size: 0.92rem;
    color: #64748B;
    text-decoration: line-through;
    margin-left: 8px;
}

.product-price-ref {
    font-size: 0.85rem;
    color: #94A3B8;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 6px 10px;
    border-radius: 8px;
    margin: 8px 0;
}

/* Micro Progress Bar for Deal Score */
.score-track {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 9999px;
    height: 6px;
    width: 100%;
    margin-top: 6px;
    overflow: hidden;
}

.score-fill {
    height: 100%;
    border-radius: 9999px;
}

/* Glassmorphic Content Box */
.content-box {
    background: rgba(19, 30, 54, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
}

/* Clean Alert Callouts */
.callout-info {
    background: rgba(59, 130, 246, 0.07);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-left: 4px solid #3B82F6;
    padding: 14px 18px;
    border-radius: 10px;
    color: #93C5FD;
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 16px 0;
}

.callout-warning {
    background: rgba(245, 158, 11, 0.07);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-left: 4px solid #F59E0B;
    padding: 14px 18px;
    border-radius: 10px;
    color: #FCD34D;
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 16px 0;
}

/* Streamlit Widget Refinements */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stNumberInput"] > div > div > input {
    background-color: #0F172A !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #F8FAFC !important;
    border-radius: 10px !important;
    font-size: 0.90rem !important;
}

[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stMultiSelect"] > div > div:focus-within,
[data-testid="stTextInput"] > div > div > input:focus,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
}

/* Streamlit Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    box-shadow: 0 6px 18px rgba(59, 130, 235, 0.45) !important;
    transform: translateY(-2px) !important;
}

/* Dataframe Styling */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: #0F172A;
}

/* Tabs Polish */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    color: #94A3B8 !important;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom: 2px solid #38BDF8 !important;
    background: rgba(56, 189, 248, 0.05);
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(19, 30, 54, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
}
</style>
"""


def apply_custom_css():
    """Inject modern fintech dark navy CSS styles into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str, badge: str = "SmartDeal AI", icon: str = "🏷️"):
    """Render a polished, executive hero header with soft lighting and badge."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                <span class="hero-title">
                    <span>{icon}</span> {title}
                </span>
                <span style="background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.35); padding: 4px 12px; border-radius: 9999px; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.04em;">
                    {badge}
                </span>
            </div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, subtext: str = "", border_color: str = "#3B82F6", icon: str = "📊"):
    """Render an elevated, responsive KPI metric card with icon and border accent."""
    st.markdown(
        f"""
        <div class="kpi-container" style="border-top: 3px solid {border_color};">
            <div class="kpi-top-row">
                <span class="kpi-label">{label}</span>
                <span class="kpi-icon">{icon}</span>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_product_card_html(row, show_score_bar: bool = True) -> str:
    """Generate HTML markup for a premium product deal card with score visualizer."""
    score = float(row.get("deal_score", 50))
    tier = str(row.get("deal_category", "Fair Price"))
    badge_html = get_deal_badge_html(score, tier)

    # Color for score bar
    if score >= 80:
        bar_color = "linear-gradient(90deg, #10B981, #34D399)"
    elif score >= 65:
        bar_color = "linear-gradient(90deg, #2563EB, #60A5FA)"
    elif score >= 45:
        bar_color = "linear-gradient(90deg, #D97706, #FBBF24)"
    else:
        bar_color = "linear-gradient(90deg, #DC2626, #F87171)"

    diff_val = float(row.get("price_diff", 0.0))
    diff_pct = float(row.get("diff_percentage", 0.0))
    savings = float(row.get("estimated_savings", 0.0))
    
    if diff_val > 0:
        savings_text = f"Est. Savings: {format_inr(savings)} ({diff_pct:.1f}% below fair benchmark)"
        savings_color = "#34D399"
    elif diff_val < 0:
        savings_text = f"Priced above fair benchmark by {format_inr(abs(diff_val))} ({abs(diff_pct):.1f}%)"
        savings_color = "#F87171"
    else:
        savings_text = "Priced exactly at fair benchmark"
        savings_color = "#94A3B8"

    score_bar_html = ""
    if show_score_bar:
        score_bar_html = f"""
        <div style="margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #94A3B8; font-weight: 600;">
                <span>Deal Quality Rating</span>
                <span>{score:.0f} / 100</span>
            </div>
            <div class="score-track">
                <div class="score-fill" style="width: {score}%; background: {bar_color};"></div>
            </div>
        </div>
        """

    return f"""
    <div class="product-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span class="product-brand-chip">{row['brand']} • {row['category']}</span>
            {badge_html}
        </div>
        <div class="product-name">{row['product_name']}</div>
        <div style="display: flex; align-items: baseline; gap: 8px; margin: 8px 0;">
            <span class="product-price-main">{format_inr(row['current_price'])}</span>
            <span class="product-price-orig">{format_inr(row['original_price'])}</span>
            <span style="font-size: 0.80rem; color: #10B981; font-weight: 700; background: rgba(16, 185, 129, 0.12); border-radius: 4px; padding: 1px 6px;">
                {row['discount_percentage']:.0f}% OFF
            </span>
        </div>
        <div class="product-price-ref">
            ⚖️ Estimated Fair Benchmark: <b style="color: #F8FAFC;">{format_inr(row['fair_reference_price'])}</b>
        </div>
        <div style="font-size: 0.84rem; color: {savings_color}; font-weight: 600; margin-top: 6px;">
            {savings_text}
        </div>
        {score_bar_html}
        <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 10px; line-height: 1.45; background: rgba(0,0,0,0.15); padding: 8px 10px; border-radius: 8px;">
            📝 <i>{row.get('deal_explanation', '')}</i>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px; font-size: 0.78rem; color: #64748B; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            <span>★ {row['rating']} ({row['review_count']:,} reviews) • {row['retailer']}</span>
            <span style="color: #38BDF8; font-weight: 600;">{row.get('confidence_level', 'High Confidence')}</span>
        </div>
    </div>
    """


def render_empty_state(title: str, description: str, icon: str = "🔍"):
    """Render a clean, user-friendly empty state container."""
    st.markdown(
        f"""
        <div style="text-align: center; padding: 48px 24px; background: rgba(19, 30, 54, 0.5); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 14px; margin: 20px 0;">
            <div style="font-size: 2.8rem; margin-bottom: 12px;">{icon}</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 0.90rem; color: #94A3B8; max-width: 480px; margin: 0 auto; line-height: 1.55;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
