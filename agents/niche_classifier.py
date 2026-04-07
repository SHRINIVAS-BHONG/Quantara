"""
Niche Classifier
----------------
Classifies traders (and their markets) into topic niches:
    NBA | politics | weather | crypto | sports | general

Strategy (cascading):
1. Rule-based keyword matching on ``markets_traded`` and ``username`` (fast, free).
2. Optional LLM call via OpenRouter for ambiguous cases (if API key is set).
3. Filters ``state["traders"]`` to keep only those matching the requested niche.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL          = "mistralai/mixtral-8x7b-instruct"
LLM_CLASSIFY_BATCH = 10      # max traders to classify via LLM per call
CONFIDENCE_THRESHOLD = 0.6   # below this → re-classify via LLM if available

# ---------------------------------------------------------------------------
# Rule-based keyword maps
# ---------------------------------------------------------------------------

NICHE_RULES: dict[str, list[str]] = {
    "NBA": [
        "nba", "basketball", "lakers", "celtics", "warriors", "bulls",
        "heat", "knicks", "nets", "76ers", "suns", "bucks", "lebron",
        "curry", "durant", "finals", "playoff", "dunk", "three-point",
    ],
    "politics": [
        "election", "president", "senate", "congress", "vote", "democrat",
        "republican", "governor", "primary", "ballot", "impeach", "policy",
        "whitehouse", "potus", "legislation", "bill", "midterm", "debate",
        "political", "politics", "supreme court", "filibuster",
    ],
    "weather": [
        "hurricane", "tornado", "storm", "rainfall", "temperature", "drought",
        "flood", "blizzard", "snowfall", "el nino", "la nina", "climate",
        "weather", "celsius", "fahrenheit", "precipitation", "forecast",
    ],
    "crypto": [
        "bitcoin", "ethereum", "btc", "eth", "crypto", "defi", "nft",
        "blockchain", "altcoin", "solana", "cardano", "binance", "coinbase",
        "stablecoin", "web3", "dao", "token", "halving", "mining",
    ],
    "sports": [
        "football", "nfl", "soccer", "mlb", "nhl", "tennis", "golf",
        "f1", "mma", "ufc", "boxing", "rugby", "cricket", "olympic",
        "super bowl", "world cup", "champion", "league", "playoff",
    ],
}

_ALL_NICHES = list(NICHE_RULES.keys()) + ["general"]


def _rule_classify(trader: dict[str, Any]) -> tuple[str, float]:
    """
    Score trader against each niche's keyword list.
    Returns (niche, confidence) where confidence ∈ [0, 1].
    """
    text = " ".join([
        trader.get("username", ""),
        trader.get("niche",    ""),
        " ".join(trader.get("markets_traded", [])),
    ]).lower()

    scores: dict[str, int] = {}
    for niche, keywords in NICHE_RULES.items():
        hit = sum(1 for kw in keywords if kw in text)
        if hit:
            scores[niche] = hit

    if not scores:
        return "general", 0.5

    best_niche = max(scores, key=lambda k: scores[k])
    total_hits  = sum(scores.values())
    confidence  = scores[best_niche] / max(total_hits, 1)
    return best_niche, round(min(confidence, 1.0), 3)


# ---------------------------------------------------------------------------
# LLM-based classification (optional enrichment)
# ---------------------------------------------------------------------------

def _llm_classify_batch(traders: list[dict[str, Any]]) -> dict[str, str]:
    """
    Ask LLM to classify a batch of traders.
    Returns {trader_id: niche} dict.
    Falls back gracefully.
    """
    if not OPENROUTER_API_KEY or not traders:
        return {}

    snippets = [
        {
            "id":      t["trader_id"],
            "markets": t.get("markets_traded", [])[:3],
            "name":    t.get("username", ""),
        }
        for t in traders[:LLM_CLASSIFY_BATCH]
    ]

    prompt = (
        f"Classify each trader into exactly one niche from: {_ALL_NICHES}.\n"
        "Use the trader's username and markets_traded for context.\n"
        "Return ONLY a JSON object mapping trader_id → niche. No other text.\n\n"
        f"Traders:\n{json.dumps(snippets, indent=2)}"
    )

    try:
        resp = httpx.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       LLM_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens":  400,
            },
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"```(?:json)?", "", content).strip("` \n")
        return json.loads(content)
    except Exception as exc:
        logger.warning("[NicheClassifier] LLM batch classify failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_by_niche(state: dict[str, Any]) -> dict[str, Any]:
    """
    Agent entry-point.

    1. Rule-classifies every trader in ``state["traders"]``.
    2. LLM re-classifies low-confidence traders (if API key present).
    3. Filters traders to match ``state["niche"]`` (keeps all if "general").

    Parameters
    ----------
    state : dict
        Shared executor state.

    Returns
    -------
    dict
        Updated state with niche field set on each trader + filtered list.
    """
    traders    = state.get("traders", [])
    target     = state.get("niche", "general").strip()
    logger.info("[NicheClassifier] Classifying %d traders → niche='%s'", len(traders), target)

    if not traders:
        return state

    # --- Step 1: rule-based ---
    low_conf: list[dict[str, Any]] = []
    for t in traders:
        niche, conf = _rule_classify(t)
        t["niche"]            = niche
        t["niche_confidence"] = conf
        if conf < CONFIDENCE_THRESHOLD:
            low_conf.append(t)

    logger.info("[NicheClassifier] Rule pass done | low-confidence: %d", len(low_conf))

    # --- Step 2: LLM re-classify low-confidence ---
    if low_conf:
        overrides = _llm_classify_batch(low_conf)
        for t in low_conf:
            if t["trader_id"] in overrides:
                t["niche"]            = overrides[t["trader_id"]]
                t["niche_confidence"] = 0.85   # trust LLM override
        logger.info("[NicheClassifier] LLM re-classified %d traders", len(overrides))

    # --- Step 3: filter ---
    if target.lower() in ("general", "all", ""):
        filtered = traders
    else:
        filtered = [t for t in traders if t["niche"].lower() == target.lower()]
        if len(filtered) < 3:
            # Not enough matches → return all with the target niche annotated but unfiltered
            logger.warning(
                "[NicheClassifier] Only %d traders matched niche '%s'; returning all.",
                len(filtered), target,
            )
            filtered = traders

    logger.info("[NicheClassifier] After filter: %d traders remain", len(filtered))
    state["traders"] = filtered
    return state


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    from agents.polymarket_agent import fetch_traders
    test_state: dict[str, Any] = {"niche": "NBA", "traders": []}
    fetch_traders(test_state)
    classify_by_niche(test_state)

    for t in test_state["traders"][:5]:
        print(f"  {t['trader_id']:15s}  niche={t['niche']:12s}  conf={t['niche_confidence']:.2f}")
    print(f"\nTotal after filter: {len(test_state['traders'])}")
