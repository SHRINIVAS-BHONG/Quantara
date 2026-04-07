import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Literal


# -----------------------------
# Storage paths
# -----------------------------

DATA_DIR = Path("data")
FEEDBACK_FILE = DATA_DIR / "feedback.json"
ADJUST_FILE = DATA_DIR / "score_adjustments.json"


# -----------------------------
# Data Models
# -----------------------------

@dataclass
class FeedbackRecord:
    feedback_id: str
    trader_id: str
    platform: str
    query: str
    recommendation_score: float
    user_rating: Literal["positive", "negative", "neutral"]
    outcome: Literal["won", "lost", "pending"]
    delta: float
    timestamp: float = field(default_factory=time.time)
    notes: str = ""


@dataclass
class ScoreAdjustment:
    trader_id: str
    platform: str
    adjustment: float
    effective_score: float
    feedback_count: int
    last_updated: float = field(default_factory=time.time)


# -----------------------------
# Utils
# -----------------------------

def _load(path: Path, default):
    if path.exists():
        try:
            with path.open() as f:
                return json.load(f)
        except:
            return default
    return default


def _save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# Core Logic
# -----------------------------

def store_feedback(
    trader_id: str,
    platform: str,
    query: str,
    recommendation_score: float,
    user_rating: Literal["positive", "negative", "neutral"] = "neutral",
    outcome: Literal["won", "lost", "pending"] = "pending",
    delta: float = 0.0,
    notes: str = "",
) -> FeedbackRecord:

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

    data = _load(FEEDBACK_FILE, [])
    data.append(asdict(record))
    _save(FEEDBACK_FILE, data)

    _update_adjustment(trader_id, platform, recommendation_score)

    return record


def _update_adjustment(trader_id: str, platform: str, base_score: float):
    all_feedback = _load(FEEDBACK_FILE, [])

    trader_feedback = [
        f for f in all_feedback
        if f["trader_id"] == trader_id and f["platform"] == platform
    ]

    adjustment = _compute_adjustment(trader_feedback)
    effective = round(base_score + adjustment, 4)

    key = f"{platform}::{trader_id}"

    adjustments = _load(ADJUST_FILE, {})
    adjustments[key] = asdict(
        ScoreAdjustment(
            trader_id=trader_id,
            platform=platform,
            adjustment=adjustment,
            effective_score=effective,
            feedback_count=len(trader_feedback),
        )
    )

    _save(ADJUST_FILE, adjustments)


def _compute_adjustment(feedback_list: List[Dict[str, Any]]) -> float:
    if not feedback_list:
        return 0.0

    now = time.time()
    HALF_LIFE = 30 * 86400

    total = 0.0

    for f in feedback_list:
        age = now - f.get("timestamp", now)
        decay = 0.5 ** (age / HALF_LIFE)

        score = 0.0

        if f["user_rating"] == "positive" or f["outcome"] == "won":
            score += 0.02
        elif f["user_rating"] == "negative" or f["outcome"] == "lost":
            score -= 0.03

        score += float(f.get("delta", 0)) * 0.1

        total += score * decay

    return max(-0.5, min(0.5, total))


# -----------------------------
# Query Helpers
# -----------------------------

def get_adjusted_score(trader_id: str, platform: str, base_score: float) -> float:
    adjustments = _load(ADJUST_FILE, {})
    key = f"{platform}::{trader_id}"

    if key in adjustments:
        return adjustments[key]["effective_score"]

    return base_score


def apply_learning_to_traders(traders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    updated = []

    for t in traders:
        score = get_adjusted_score(
            t.get("trader_id", ""),
            t.get("platform", ""),
            t.get("score", 0),
        )

        t = {**t, "score": round(score, 4), "adjusted": True}
        updated.append(t)

    return updated


def get_all_adjustments():
    return _load(ADJUST_FILE, {})