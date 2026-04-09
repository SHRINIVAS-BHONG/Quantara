import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

PLATFORM_KEYWORDS = {
    "polymarket": ["polymarket", "poly"],
    "kalshi": ["kalshi"],
}

NICHE_KEYWORDS = {
    "NBA": ["nba", "basketball"],
    "politics": ["politics", "election"],
    "crypto": ["crypto", "bitcoin"],
    "weather": ["weather", "storm"],
    "sports": ["sports", "football"],
}

INTENT_KEYWORDS = {
    "search": ["find", "search"],
    "analyze": ["analyze", "stats"],
    "recommend": ["best", "top", "recommend"],
}

STEP_MAP = {
    "search": ["search_traders", "filter_by_niche"],
    "analyze": ["search_traders", "filter_by_niche", "analyze_performance"],
    "recommend": ["search_traders", "filter_by_niche", "analyze_performance"],
}


def _keyword_detect(text: str, mapping: dict, default: str) -> str:
    text = text.lower()
    for key, kws in mapping.items():
        if any(kw in text for kw in kws):
            return key
    return default


def plan(query: str) -> dict[str, Any]:
    platform = _keyword_detect(query, PLATFORM_KEYWORDS, "both")
    niche = _keyword_detect(query, NICHE_KEYWORDS, "general")
    intent = _keyword_detect(query, INTENT_KEYWORDS, "recommend")

    return {
        "platform": platform,
        "niche": niche,
        "intent": intent,
        "steps": STEP_MAP[intent],
        "raw_query": query,
    }