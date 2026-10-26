"""
Formatting utilities for SmartDeal AI.
Implements Indian currency formatting (₹ with proper Lakh/Crore grouping),
percentage formatting, deal rating badges, and card stylers.
"""

from typing import Union, Optional
import math
from config import CURRENCY_SYMBOL, DEAL_TIERS


def format_inr(amount: Optional[Union[int, float]], include_symbol: bool = True, decimals: int = 0) -> str:
    """
    Format a number into Indian Rupee format with appropriate commas.
    Example: 124500 -> '₹1,24,500' or 1500000 -> '₹15,00,000'.
    Handles negative numbers, zero, None, and NaN safely.
    """
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return "-" if not include_symbol else f"{CURRENCY_SYMBOL}0"

    is_negative = amount < 0
    abs_amount = abs(amount)

    if decimals > 0:
        base_int = int(abs_amount)
        fraction = f"{abs_amount - base_int:.{decimals}f}"[1:]  # e.g., '.50'
    else:
        base_int = int(round(abs_amount))
        fraction = ""

    s = str(base_int)
    if len(s) <= 3:
        formatted_int = s
    else:
        last3 = s[-3:]
        remaining = s[:-3]
        # Group preceding digits in pairs of 2
        groups = []
        while remaining:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        formatted_int = ",".join(groups) + "," + last3

    result = f"{formatted_int}{fraction}"
    prefix = f"-{CURRENCY_SYMBOL}" if (is_negative and include_symbol) else ("-" if is_negative else (f"{CURRENCY_SYMBOL}" if include_symbol else ""))
    return f"{prefix}{result}"


def format_percent(val: Optional[Union[int, float]], decimals: int = 1, show_sign: bool = False) -> str:
    """Format float into a percentage string, e.g. 15.4%."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "0.0%"
    sign = "+" if (show_sign and val > 0) else ""
    return f"{sign}{val:.{decimals}f}%"


def get_deal_tier(score: float) -> str:
    """Classify a 0-100 score into a standard deal category."""
    if score >= DEAL_TIERS["Great Deal"]["min_score"]:
        return "Great Deal"
    elif score >= DEAL_TIERS["Good Deal"]["min_score"]:
        return "Good Deal"
    elif score >= DEAL_TIERS["Fair Price"]["min_score"]:
        return "Fair Price"
    else:
        return "Above Reference"


def get_deal_badge_html(score: float, category: Optional[str] = None) -> str:
    """Return an elevated HTML badge styled for the deal score and tier with glowing indicators."""
    tier = category or get_deal_tier(score)
    
    badge_styles = {
        "Great Deal": {
            "bg": "rgba(16, 185, 129, 0.12)",
            "border": "rgba(16, 185, 129, 0.35)",
            "color": "#34D399",
            "dot": "#10B981",
            "glow": "0 0 10px rgba(16, 185, 129, 0.25)"
        },
        "Good Deal": {
            "bg": "rgba(59, 130, 246, 0.12)",
            "border": "rgba(59, 130, 246, 0.35)",
            "color": "#60A5FA",
            "dot": "#3B82F6",
            "glow": "0 0 10px rgba(59, 130, 246, 0.25)"
        },
        "Fair Price": {
            "bg": "rgba(245, 158, 11, 0.12)",
            "border": "rgba(245, 158, 11, 0.35)",
            "color": "#FBBF24",
            "dot": "#F59E0B",
            "glow": "0 0 10px rgba(245, 158, 11, 0.2)"
        },
        "Above Reference": {
            "bg": "rgba(239, 68, 68, 0.12)",
            "border": "rgba(239, 68, 68, 0.35)",
            "color": "#F87171",
            "dot": "#EF4444",
            "glow": "0 0 10px rgba(239, 68, 68, 0.2)"
        },
    }

    style = badge_styles.get(tier, badge_styles["Fair Price"])
    
    return (
        f'<span style="background: {style["bg"]}; border: 1px solid {style["border"]}; '
        f'color: {style["color"]}; box-shadow: {style["glow"]}; padding: 4px 12px; '
        f'border-radius: 9999px; font-weight: 600; font-size: 0.80rem; '
        f'display: inline-flex; align-items: center; gap: 6px; letter-spacing: 0.02em;">'
        f'<span style="width: 7px; height: 7px; border-radius: 50%; background-color: {style["dot"]}; display: inline-block;"></span>'
        f'{tier} • {score:.0f}/100</span>'
    )
