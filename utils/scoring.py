"""
scoring.py
Centralised scoring functions used across agents.

Primary formula:
    score = (win_rate * 0.5) + (roi * 0.3) - (risk * 0.2)

where:
    risk  = normalised variance / drawdown measure
    roi   = return on investment (fractional, e.g. 0.25 = 25%)
"""

from __future__ import annotations

import math
import statistics
from typing import Any


# ---------------------------------------------------------------------------
# Primary scoring function
# ---------------------------------------------------------------------------

def compute_trader_score(
    win_rate: float,
    roi: float,
    risk: float,
    *,
    consistency_bonus: float = 0.0,
) -> float:
    """
    Compute the composite trader score.

    Parameters
    ----------
    win_rate           : Fraction of winning trades  [0, 1].
    roi                : Return on investment         (can be negative).
    risk               : Normalised risk score        [0, 1] – higher = riskier.
    consistency_bonus  : Optional bonus for trade count / streaks [0, 0.1].

    Returns
    -------
    float  Score value (unclamped, typical range −0.5 … 1.0+).
    """
    raw = (win_rate * 0.5) + (roi * 0.3) - (risk * 0.2) + consistency_bonus
    return round(raw, 5)


# ---------------------------------------------------------------------------
# Risk / variance helpers
# ---------------------------------------------------------------------------

def compute_risk(
    trade_outcomes: list[float],
    max_drawdown: float | None = None,
) -> float:
    """
    Compute a normalised risk score from a list of per-trade P&L values.

    Risk = weighted blend of:
        - coefficient of variation (std / mean)  [0 … 1]
        - normalised drawdown                    [0 … 1]

    Parameters
    ----------
    trade_outcomes : List of fractional P&L per trade (e.g. [0.2, -0.1, 0.05]).
    max_drawdown   : Optional externally supplied max drawdown fraction [0, 1].

    Returns
    -------
    float  Normalised risk in [0, 1].
    """
    if not trade_outcomes:
        return 0.5  # default to mid-risk if no data

    mean = statistics.mean(trade_outcomes) if trade_outcomes else 0.0
    std = statistics.stdev(trade_outcomes) if len(trade_outcomes) >= 2 else 0.0

    # Coefficient of variation clamped to [0, 1]
    if mean != 0:
        cv = min(abs(std / mean), 1.0)
    else:
        cv = min(std, 1.0)

    # Drawdown component
    dd_component: float
    if max_drawdown is not None:
        dd_component = min(abs(max_drawdown), 1.0)
    else:
        dd_component = _infer_drawdown(trade_outcomes)

    # Blend: 60% CV + 40% drawdown
    risk = (cv * 0.6) + (dd_component * 0.4)
    return round(min(max(risk, 0.0), 1.0), 4)


def _infer_drawdown(outcomes: list[float]) -> float:
    """
    Compute max drawdown from a running equity curve inferred from outcomes.
    Returns normalised value in [0, 1].
    """
    equity = 1.0
    peak = equity
    max_dd = 0.0
    for r in outcomes:
        equity *= (1 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return round(min(max_dd, 1.0), 4)


# ---------------------------------------------------------------------------
# Consistency score
# ---------------------------------------------------------------------------

def compute_consistency(
    trade_outcomes: list[float],
    total_trades: int,
) -> float:
    """
    Consistency bonus based on:
      - Trade count (more trades → more statistically reliable)
      - Sharpe-like ratio (mean / std of outcomes)

    Returns a value in [0, 0.1].
    """
    if not trade_outcomes or total_trades == 0:
        return 0.0

    # Trade-count bonus: asymptotically approaches 0.05 as trades → ∞
    count_bonus = 0.05 * (1 - math.exp(-total_trades / 50))

    if len(trade_outcomes) >= 2:
        mean_r = statistics.mean(trade_outcomes)
        std_r = statistics.stdev(trade_outcomes)
        sharpe = (mean_r / std_r) if std_r > 0 else 0.0
        # Normalise Sharpe to [0, 0.05]
        sharpe_bonus = min(max(sharpe * 0.01, 0.0), 0.05)
    else:
        sharpe_bonus = 0.0

    return round(min(count_bonus + sharpe_bonus, 0.1), 5)


# ---------------------------------------------------------------------------
# ROI helper
# ---------------------------------------------------------------------------

def compute_roi(outcomes: list[float]) -> float:
    """Compound ROI from a list of per-trade fractional returns."""
    if not outcomes:
        return 0.0
    equity = 1.0
    for r in outcomes:
        equity *= (1 + r)
    return round(equity - 1.0, 5)


# ---------------------------------------------------------------------------
# Win-rate helper
# ---------------------------------------------------------------------------

def compute_win_rate(outcomes: list[float]) -> float:
    """Fraction of outcomes > 0."""
    if not outcomes:
        return 0.0
    wins = sum(1 for r in outcomes if r > 0)
    return round(wins / len(outcomes), 4)


# ---------------------------------------------------------------------------
# Full trader score pipeline
# ---------------------------------------------------------------------------

def score_trader(trader: dict[str, Any]) -> dict[str, Any]:
    """
    Accept a raw trader dict, compute derived metrics and inject them.

    Expected keys:
        trade_outcomes : list[float]   (required)
        total_trades   : int           (optional; defaults to len(trade_outcomes))

    Adds / updates keys:
        win_rate, roi, risk, consistency_score, score
    """
    outcomes: list[float] = trader.get("trade_outcomes", [])
    total = trader.get("total_trades", len(outcomes))

    win_rate = trader.get("win_rate") or compute_win_rate(outcomes)
    roi = trader.get("roi") or compute_roi(outcomes)
    risk = compute_risk(outcomes, trader.get("max_drawdown"))
    consistency = compute_consistency(outcomes, total)
    score = compute_trader_score(win_rate, roi, risk, consistency_bonus=consistency)

    return {
        **trader,
        "win_rate": round(win_rate, 4),
        "roi": round(roi, 4),
        "risk": risk,
        "consistency_score": consistency,
        "score": score,
    }


def rank_traders(traders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort traders by descending score and add a rank field."""
    ranked = sorted(traders, key=lambda t: t.get("score", 0), reverse=True)
    for i, t in enumerate(ranked, 1):
        t["rank"] = i
    return ranked
