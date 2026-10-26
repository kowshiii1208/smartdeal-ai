"""
Smart Deal Engine for SmartDeal AI.
Implements transparent, mathematical deal scoring (0-100), savings calculations,
deal classification (Great Deal, Good Deal, Fair Price, Above Reference),
evidence confidence assessment, and human-readable justification.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from utils.formatting import format_inr, format_percent, get_deal_tier


def calculate_deal_metrics(
    current_price: float,
    reference_price: float,
    original_price: float,
    rating: float = 4.0,
    review_count: int = 100,
    has_history: bool = True,
) -> Dict[str, Any]:
    """
    Calculate deal score, deal tier, savings, and justification.
    
    Mathematical Scoring:
      1. Diff % vs Fair Reference Price:
         diff_pct = (reference_price - current_price) / reference_price * 100
      2. MRP Discount %:
         mrp_disc_pct = (original_price - current_price) / original_price * 100
      3. Base Score = 50.0
      4. Reference component = clip(2.5 * diff_pct, -35.0, +35.0)
      5. MRP component = clip(0.25 * mrp_disc_pct, 0.0, +12.0)
      6. Rating component = clip(5.0 * (rating - 4.0), -10.0, +5.0)
      7. Evidence Confidence dampener:
         regresses score towards 50 if review count is low or history is absent.
    """
    if reference_price <= 0:
        reference_price = current_price

    abs_diff = reference_price - current_price
    diff_pct = (abs_diff / reference_price) * 100

    mrp_disc_pct = ((original_price - current_price) / original_price * 100) if original_price > 0 else 0.0

    # Components
    base_score = 50.0
    ref_component = float(np.clip(2.5 * diff_pct, -35.0, 35.0))
    mrp_component = float(np.clip(0.25 * mrp_disc_pct, 0.0, 12.0))
    rating_component = float(np.clip(5.0 * (rating - 4.0), -10.0, 5.0))

    raw_score = base_score + ref_component + mrp_component + rating_component

    # Evidence Confidence Assessment
    if review_count >= 1000 and has_history:
        confidence = "High Confidence"
        confidence_factor = 1.0
    elif review_count >= 100:
        confidence = "Moderate Confidence"
        confidence_factor = 0.90
    else:
        confidence = "Low Evidence"
        confidence_factor = 0.75

    # Regress slightly towards equilibrium (50) if evidence is low
    adjusted_score = 50.0 + (raw_score - 50.0) * confidence_factor
    final_score = round(float(np.clip(adjusted_score, 0.0, 100.0)), 1)

    tier = get_deal_tier(final_score)

    # Generate transparent explanation
    reasons = []
    if diff_pct > 0:
        reasons.append(f"Listed {format_inr(abs_diff)} ({diff_pct:.1f}%) below estimated fair market reference price")
    elif diff_pct < 0:
        reasons.append(f"Listed {format_inr(abs(abs_diff))} ({abs(diff_pct):.1f}%) above estimated fair reference price")
    else:
        reasons.append("Priced exactly at fair market benchmark")

    if mrp_disc_pct > 15:
        reasons.append(f"{mrp_disc_pct:.1f}% discount against original MRP")

    if rating >= 4.4:
        reasons.append(f"Strong customer acclaim ({rating}★ across {review_count:,} reviews)")
    elif rating < 3.8:
        reasons.append(f"Sub-par customer satisfaction ({rating}★)")

    explanation = " • ".join(reasons)

    return {
        "current_price": current_price,
        "reference_price": reference_price,
        "absolute_difference": round(abs_diff, 2),
        "percentage_difference": round(diff_pct, 2),
        "deal_score": final_score,
        "deal_category": tier,
        "confidence_level": confidence,
        "explanation": explanation,
        "estimated_savings": max(0.0, round(abs_diff, 2)),
    }


def enrich_products_with_deals(
    df: pd.DataFrame,
    reference_price_col: str = "fair_reference_price"
) -> pd.DataFrame:
    """
    Enrich a products DataFrame with deal score, tier, savings, and metrics.
    """
    enriched = df.copy()
    
    # If fair_reference_price is missing, fallback to current_price
    if reference_price_col not in enriched.columns or enriched[reference_price_col].isna().all():
        enriched["ref_price"] = enriched["current_price"]
    else:
        enriched["ref_price"] = enriched[reference_price_col].fillna(enriched["current_price"])

    scores = []
    tiers = []
    abs_diffs = []
    diff_pcts = []
    confidences = []
    explanations = []
    savings = []

    for _, row in enriched.iterrows():
        c_p = float(row["current_price"])
        r_p = float(row["ref_price"])
        o_p = float(row.get("original_price", c_p))
        rating = float(row.get("rating", 4.0))
        reviews = int(row.get("review_count", 100))

        metrics = calculate_deal_metrics(
            current_price=c_p,
            reference_price=r_p,
            original_price=o_p,
            rating=rating,
            review_count=reviews,
            has_history=True
        )

        scores.append(metrics["deal_score"])
        tiers.append(metrics["deal_category"])
        abs_diffs.append(metrics["absolute_difference"])
        diff_pcts.append(metrics["percentage_difference"])
        confidences.append(metrics["confidence_level"])
        explanations.append(metrics["explanation"])
        savings.append(metrics["estimated_savings"])

    enriched["deal_score"] = scores
    enriched["deal_category"] = tiers
    enriched["price_diff"] = abs_diffs
    enriched["diff_percentage"] = diff_pcts
    enriched["confidence_level"] = confidences
    enriched["deal_explanation"] = explanations
    enriched["estimated_savings"] = savings

    return enriched
