"""
Analysis Agent
--------------
Computes and enriches performance metrics for each trader:

    score = (win_rate × 0.5) + (roi × 0.3) − (risk × 0.2)

Additional metrics:
    - consistency_score  : inverse of return variance (lower volatility → higher score)
    - drawdown           : estimated max drawdown used as risk proxy
    - rank               : ordinal rank after sorting by composite score
    - tier               : Elite / Strong / Average / Developing
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights (match spec exactly)
# ---------------------------------------------------------------------------
WEIGHT_WIN_RATE = 0.50
WEIGHT_ROI      = 0.30
WEIGHT_RISK     = 0.20

# Tier thresholds (composite score)
TIER_MAP = [
    (0.70, "Elite"),
    (0.55, "Strong"),
    (0.40, "Average"),
    (0.00, "Developing"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_consistency(trader: dict[str, Any]) -> float:
    """
    Return a consistency score ∈ [0, 1].
    Uses simulated trade-by-trade returns when raw return series isn't available.
    """
    # In production, pass in ``trader["return_series"]`` directly.
    # Here we derive a plausible series from the trader's aggregate stats.
    rng = random.Random(hash(trader.get("trader_id", "x")))
    win_rate = trader.get("win_rate", 0.5)
    roi      = trader.get("roi",      0.0)
    n        = min(max(trader.get("total_trades", 30), 10), 200)

    returns  = []
    for _ in range(n):
        if rng.random() < win_rate:
            ret = rng.gauss(roi * 1.2, 0.05)
        else:
            ret = rng.gauss(-roi * 0.8, 0.07)
        returns.append(ret)

    if len(returns) < 2:
        return 0.5

    try:
        std = statistics.stdev(returns)
    except statistics.StatisticsError:
        std = 0.1

    # Consistency ∈ [0, 1]: lower std → higher consistency
    consistency = 1.0 / (1.0 + std * 10)
    return round(min(max(consistency, 0.0), 1.0), 4)


def _compute_drawdown(trader: dict[str, Any]) -> float:
    """
    Estimate max drawdown as a risk proxy ∈ [0, 1].
    Derived from roi and win_rate; replace with actual equity curve in production.
    """
    win_rate = trader.get("win_rate", 0.5)
    roi      = trader.get("roi",      0.0)
    loss_rate = 1.0 - win_rate
    # Simplified: expected drawdown rises with loss rate and negative ROI
    drawdown = loss_rate * 0.4 + max(-roi, 0) * 0.6
    return round(min(max(drawdown, 0.0), 1.0), 4)


def _assign_tier(score: float) -> str:
    for threshold, tier in TIER_MAP:
        if score >= threshold:
            return tier
    return "Developing"


def _compute_score(win_rate: float, roi: float, risk: float) -> float:
    """Core scoring formula from the specification."""
    return round(
        win_rate * WEIGHT_WIN_RATE
        + roi    * WEIGHT_ROI
        - risk   * WEIGHT_RISK,
        4,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_traders(state: dict[str, Any]) -> dict[str, Any]:
    """
    Agent entry-point.

    Iterates over ``state["traders"]``, computes/enriches metrics, and
    writes back a ranked, annotated list.

    Parameters
    ----------
    state : dict
        Shared executor state.

    Returns
    -------
    dict
        Updated state with enriched + ranked ``traders``.
    """
    traders = state.get("traders", [])
    logger.info("[AnalysisAgent] Analysing %d traders …", len(traders))

    if not traders:
        logger.warning("[AnalysisAgent] No traders to analyse.")
        return state

    enriched: list[dict[str, Any]] = []

    for t in traders:
        win_rate = float(t.get("win_rate", 0.5))
        roi      = float(t.get("roi",      0.0))

        # Recompute risk as max of existing risk field and estimated drawdown
        raw_risk = float(t.get("risk", 0.2))
        drawdown = _compute_drawdown(t)
        risk     = round(max(raw_risk, drawdown), 4)

        consistency = _compute_consistency(t)
        score       = _compute_score(win_rate, roi, risk)

        t.update({
            "win_rate":          win_rate,
            "roi":               roi,
            "risk":              risk,
            "drawdown":          drawdown,
            "consistency_score": consistency,
            "score":             score,
            "tier":              _assign_tier(score),
        })
        enriched.append(t)

    # Sort by composite score descending, then assign rank
    enriched.sort(key=lambda x: x["score"], reverse=True)
    for rank, t in enumerate(enriched, start=1):
        t["rank"] = rank

    logger.info(
        "[AnalysisAgent] Done. Top scorer: %s  score=%.4f  tier=%s",
        enriched[0]["trader_id"],
        enriched[0]["score"],
        enriched[0]["tier"],
    )

    # Summary stats stored in state for RAG context
    scores = [t["score"] for t in enriched]
    state["analysis_summary"] = {
        "total_traders":    len(enriched),
        "avg_score":        round(statistics.mean(scores),   4),
        "median_score":     round(statistics.median(scores), 4),
        "top_score":        round(max(scores), 4),
        "bottom_score":     round(min(scores), 4),
        "tier_distribution": _tier_distribution(enriched),
    }

    state["traders"] = enriched
    return state


def _tier_distribution(traders: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {"Elite": 0, "Strong": 0, "Average": 0, "Developing": 0}
    for t in traders:
        tier = t.get("tier", "Developing")
        dist[tier] = dist.get(tier, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# Standalone scorer (also used by learning_loop)
# ---------------------------------------------------------------------------

def compute_trader_score(
    win_rate: float,
    roi: float,
    risk: float,
) -> float:
    """
    Pure function exposing the scoring formula for external use.

    score = (win_rate × 0.5) + (roi × 0.3) − (risk × 0.2)
    """
    return _compute_score(win_rate, roi, risk)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    from agents.polymarket_agent import fetch_traders
    from agents.niche_classifier import classify_by_niche

    s: dict[str, Any] = {"niche": "NBA", "traders": []}
    fetch_traders(s)
    classify_by_niche(s)
    analyze_traders(s)

    print("\nTop 5 traders:")
    for t in s["traders"][:5]:
        print(
            f"  #{t['rank']:2d}  {t['trader_id']:15s}  "
            f"score={t['score']:.4f}  tier={t['tier']:12s}  "
            f"win={t['win_rate']:.2%}  roi={t['roi']:.2%}"
        )
    print("\nSummary:", json.dumps(s["analysis_summary"], indent=2))
