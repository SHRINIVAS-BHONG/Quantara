"""
Planner Agent
-------------
Converts a natural language user query into a structured JSON execution plan
using an LLM (OpenRouter-compatible interface, with mock fallback).
"""

import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (override via env vars in production)
# ---------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = ""          # set via env: OPENROUTER_API_KEY
LLM_MODEL          = "mistralai/mixtral-8x7b-instruct"   # any OpenRouter model

PLATFORM_KEYWORDS: dict[str, list[str]] = {
    "polymarket": ["polymarket", "poly"],
    "kalshi":     ["kalshi"],
}
NICHE_KEYWORDS: dict[str, list[str]] = {
    "NBA":      ["nba", "basketball", "lakers", "celtics"],
    "politics": ["politics", "election", "president", "senate", "congress", "vote"],
    "weather":  ["weather", "hurricane", "storm", "climate"],
    "crypto":   ["crypto", "bitcoin", "ethereum", "btc", "eth", "defi"],
    "sports":   ["sports", "football", "nfl", "soccer", "mlb", "nhl"],
}
INTENT_KEYWORDS: dict[str, list[str]] = {
    "search":    ["find", "search", "list", "show", "get"],
    "analyze":   ["analyze", "analyse", "performance", "stats", "statistics"],
    "recommend": ["recommend", "best", "top", "should i", "copy", "who to follow"],
}

STEP_MAP: dict[str, list[str]] = {
    "search":    ["search_traders", "filter_by_niche", "analyze_performance", "rank_traders"],
    "analyze":   ["search_traders", "filter_by_niche", "analyze_performance", "enrich_context", "generate_explanation"],
    "recommend": ["search_traders", "filter_by_niche", "analyze_performance", "enrich_context", "rank_traders", "generate_explanation"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_detect(text: str, mapping: dict[str, list[str]], default: str) -> str:
    lower = text.lower()
    for key, kws in mapping.items():
        if any(kw in lower for kw in kws):
            return key
    return default


def _rule_based_plan(query: str) -> dict[str, Any]:
    """Deterministic fallback planner (no LLM required)."""
    lower = query.lower()

    # platform
    if "polymarket" in lower and "kalshi" in lower:
        platform = "both"
    else:
        platform = _keyword_detect(lower, PLATFORM_KEYWORDS, "both")

    niche  = _keyword_detect(lower, NICHE_KEYWORDS, "general")
    intent = _keyword_detect(lower, INTENT_KEYWORDS, "recommend")
    steps  = STEP_MAP.get(intent, STEP_MAP["recommend"])

    return {
        "platform": platform,
        "niche":    niche,
        "intent":   intent,
        "steps":    steps,
        "raw_query": query,
    }


def _llm_plan(query: str) -> dict[str, Any] | None:
    """Call OpenRouter LLM to produce a structured plan. Returns None on failure."""
    if not OPENROUTER_API_KEY:
        return None

    system_prompt = (
        "You are a planning agent for a prediction-market trader analysis platform.\n"
        "Given a user query, output ONLY a valid JSON object with these keys:\n"
        "  platform  : 'polymarket' | 'kalshi' | 'both'\n"
        "  niche     : 'NBA' | 'politics' | 'weather' | 'crypto' | 'sports' | 'general'\n"
        "  intent    : 'search' | 'analyze' | 'recommend'\n"
        "  steps     : ordered list of step names from:\n"
        "              [search_traders, filter_by_niche, analyze_performance,\n"
        "               enrich_context, rank_traders, generate_explanation]\n"
        "  raw_query : the original query string\n"
        "Do not add any explanation. Output JSON only."
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
        "temperature": 0.2,
        "max_tokens":  300,
    }

    try:
        resp = httpx.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # strip markdown code fences if present
        content = re.sub(r"```(?:json)?", "", content).strip("` \n")
        plan = json.loads(content)
        plan.setdefault("raw_query", query)
        return plan
    except Exception as exc:
        logger.warning("LLM planner failed (%s); falling back to rule-based.", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan(query: str) -> dict[str, Any]:
    """
    Convert a natural-language query into a structured execution plan.

    Parameters
    ----------
    query : str
        The raw user question, e.g. "Find best NBA traders in Polymarket".

    Returns
    -------
    dict
        {
            "platform": "polymarket" | "kalshi" | "both",
            "niche":    "NBA" | "politics" | ...,
            "intent":   "search" | "analyze" | "recommend",
            "steps":    ["search_traders", "filter_by_niche", ...],
            "raw_query": query,
        }
    """
    logger.info("[Planner] Received query: %s", query)

    result = _llm_plan(query) or _rule_based_plan(query)

    # --- validation & normalisation ---
    result["platform"] = result.get("platform", "both").lower()
    result["niche"]    = result.get("niche",    "general")
    result["intent"]   = result.get("intent",   "recommend").lower()
    if not result.get("steps"):
        result["steps"] = STEP_MAP.get(result["intent"], STEP_MAP["recommend"])

    logger.info("[Planner] Execution plan: %s", json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        "Find the best NBA traders in Polymarket",
        "Should I copy politics traders on Kalshi?",
        "Analyze crypto traders across both platforms",
    ]
    for q in tests:
        print(json.dumps(plan(q), indent=2))
        print("-" * 60)
