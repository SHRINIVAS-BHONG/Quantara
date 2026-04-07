"""
feedback.py
Closed learning loop for the prediction-market agent system.

Responsibilities:
- Store user feedback on recommendations (thumbs up/down, outcome).
- Adjust trader scores based on accumulated feedback.
- Persist feedback and adjusted scores to JSON files.
- Expose score adjustment queries for downstream agents.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
FEEDBACK_FILE = DATA_DIR / "feedback.json"
SCORE_ADJUSTMENTS_FILE = DATA_DIR / "score_adjustments.json"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    feedback_id: str
    trader_id: str
    platform: str
    query: str
    recommendation_score: float          # score at recommendation time
    user_rating: Literal["positive", "negative", "neutral"]
    outcome: str                          # "won", "lost", "pending"
    delta: float                          # actual P&L if known, else 0
    timestamp: float = field(default_factory=time.time)
    notes: str = ""


@dataclass
class ScoreAdjustment:
    trader_id: str
    platform: str
    base_score: float
    adjustment: float                     # cumulative adjustment from feedback
    effective_score: float                # base_score + adjustment
    feedback_count: int
    last_updated: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            with path.open() as f:
                return json.load(f)
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
    return default


def _save_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("Could not save %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Core feedback functions
# ---------------------------------------------------------------------------

def store_feedback(
    trader_id: str,
    platform: str,
    query: str,
    recommendation_score: float,
    user_rating: Literal["positive", "negative", "neutral"] = "neutral",
    outcome: str = "pending",
    delta: float = 0.0,
    notes: str = "",
) -> FeedbackRecord:
    """
    Store a feedback record and immediately update the score adjustment.

    Parameters
    ----------
    trader_id            : Unique trader identifier.
    platform             : "polymarket" | "kalshi".
    query                : Original user query that led to this recommendation.
    recommendation_score : The analyst score at the time of recommendation.
    user_rating          : "positive" | "negative" | "neutral".
    outcome              : "won" | "lost" | "pending".
    delta                : Actual P&L (fractional, e.g. 0.12 = +12%).
    notes                : Free-text notes.

    Returns
    -------
    The created FeedbackRecord.
    """
    record = FeedbackRecord(
        feedback_id=str(uuid.uuid4()),
        trader_id=trader_id,
        platform=platform,
        query=query,
        recommendation_score=recommendation_score,
        user_rating=user_rating,
        outcome=outcome,
        delta=delta,
        notes=notes,
    )

    # Load existing feedback
    raw: list[dict] = _load_json(FEEDBACK_FILE, [])
    raw.append(asdict(record))
    _save_json(FEEDBACK_FILE, raw)

    # Update score adjustment
    _update_score_adjustment(trader_id, platform, recommendation_score, record)

    logger.info(
        "Feedback stored: trader=%s rating=%s outcome=%s delta=%.3f",
        trader_id, user_rating, outcome, delta,
    )
    return record


def _update_score_adjustment(
    trader_id: str,
    platform: str,
    base_score: float,
    record: FeedbackRecord,
) -> ScoreAdjustment:
    """Recompute cumulative score adjustment from all feedback for this trader."""
    key = f"{platform}::{trader_id}"
    adjustments: dict[str, dict] = _load_json(SCORE_ADJUSTMENTS_FILE, {})

    # Gather ALL feedback for this trader
    all_feedback: list[dict] = _load_json(FEEDBACK_FILE, [])
    trader_feedback = [
        f for f in all_feedback
        if f["trader_id"] == trader_id and f["platform"] == platform
    ]

    cumulative_adjustment = _compute_adjustment(trader_feedback)
    effective = round(base_score + cumulative_adjustment, 5)

    adj = ScoreAdjustment(
        trader_id=trader_id,
        platform=platform,
        base_score=base_score,
        adjustment=round(cumulative_adjustment, 5),
        effective_score=effective,
        feedback_count=len(trader_feedback),
    )
    adjustments[key] = asdict(adj)
    _save_json(SCORE_ADJUSTMENTS_FILE, adjustments)
    return adj


def _compute_adjustment(feedback_list: list[dict]) -> float:
    """
    Learning formula:
      - Each 'positive' or won outcome nudges score up by 0.02.
      - Each 'negative' or lost outcome nudges score down by 0.03.
      - delta (actual P&L) contributes linearly scaled by 0.1.
      - Adjustments decay with age (exponential decay, half-life ~30 days).
    """
    if not feedback_list:
        return 0.0

    now = time.time()
    HALF_LIFE_SECONDS = 30 * 86400  # 30 days

    total = 0.0
    for fb in feedback_list:
        age = now - fb.get("timestamp", now)
        decay = 0.5 ** (age / HALF_LIFE_SECONDS)

        base = 0.0
        rating = fb.get("user_rating", "neutral")
        outcome = fb.get("outcome", "pending")

        if rating == "positive" or outcome == "won":
            base += 0.02
        elif rating == "negative" or outcome == "lost":
            base -= 0.03

        delta = float(fb.get("delta", 0))
        base += delta * 0.1

        total += base * decay

    # Clamp adjustment to [-0.5, +0.5]
    return max(-0.5, min(0.5, total))


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_adjusted_score(trader_id: str, platform: str, base_score: float) -> float:
    """
    Return the feedback-adjusted score for a trader.
    Falls back to base_score if no feedback exists.
    """
    key = f"{platform}::{trader_id}"
    adjustments: dict[str, dict] = _load_json(SCORE_ADJUSTMENTS_FILE, {})
    if key in adjustments:
        return adjustments[key].get("effective_score", base_score)
    return base_score


def get_all_adjustments() -> dict[str, ScoreAdjustment]:
    """Return all stored score adjustments keyed by 'platform::trader_id'."""
    raw: dict[str, dict] = _load_json(SCORE_ADJUSTMENTS_FILE, {})
    return {k: ScoreAdjustment(**v) for k, v in raw.items()}


def get_trader_feedback_history(trader_id: str, platform: str) -> list[FeedbackRecord]:
    """Return full feedback history for a single trader."""
    all_fb: list[dict] = _load_json(FEEDBACK_FILE, [])
    return [
        FeedbackRecord(**f)
        for f in all_fb
        if f["trader_id"] == trader_id and f["platform"] == platform
    ]


def apply_learning_to_traders(traders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Given a list of trader dicts (each with 'trader_id', 'platform', 'score'),
    return a new list with scores adjusted by the learning loop.
    """
    updated: list[dict[str, Any]] = []
    for t in traders:
        adjusted = get_adjusted_score(
            trader_id=t.get("trader_id", ""),
            platform=t.get("platform", ""),
            base_score=float(t.get("score", 0)),
        )
        t = {**t, "score": round(adjusted, 5), "score_adjusted": True}
        updated.append(t)
    return updated


# ---------------------------------------------------------------------------
# Batch outcome update (called after market resolution)
# ---------------------------------------------------------------------------

def record_market_outcome(
    trader_id: str,
    platform: str,
    outcome: Literal["won", "lost"],
    delta: float,
    query: str = "auto-resolved",
) -> FeedbackRecord:
    """
    Record an objective market outcome without explicit user rating.
    Used by a background reconciler once markets settle.
    """
    return store_feedback(
        trader_id=trader_id,
        platform=platform,
        query=query,
        recommendation_score=0.0,
        user_rating="neutral",
        outcome=outcome,
        delta=delta,
        notes="Auto-recorded market outcome",
    )
