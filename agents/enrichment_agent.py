"""
Enrichment Agent
----------------
Enriches trader data with external context:
    - Recent news headlines related to the niche
    - Event context (upcoming events, market catalysts)
    - Social/sentiment signals (mocked)

Production mode: calls Apify REST API.
Fallback / dev mode: returns realistic simulated enrichment data.
"""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APIFY_API_BASE  = "https://api.apify.com/v2"
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
MOCK_MODE: bool = os.getenv("APIFY_MOCK", "true").lower() == "true"
REQUEST_TIMEOUT = 15

# Apify actor IDs for different data types
APIFY_NEWS_ACTOR    = "apify/google-search-scraper"
APIFY_TWITTER_ACTOR = "quacker/twitter-scraper"

# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------

_NEWS_TEMPLATES: dict[str, list[str]] = {
    "NBA": [
        "Lakers vs Celtics: Preview and odds breakdown for tonight's clash",
        "NBA Finals odds shift after star player injury report",
        "Warriors dominate as prediction markets move sharply",
        "Historic 3-pointer night sends Polymarket NBA contracts soaring",
        "Playoff seeding race tightens — what traders need to know",
    ],
    "politics": [
        "Senate race tightens; prediction markets now 50/50",
        "New polling data shifts election forecasts dramatically",
        "Governor's approval rating drops, affecting Kalshi contracts",
        "Supreme Court decision impacts key political market positions",
        "Swing state update: markets reprice ahead of early voting",
    ],
    "weather": [
        "Hurricane forecast upgraded — storm prediction markets surge",
        "El Niño strengthens; weather contract holders watch closely",
        "Tornado season outlook revised; markets adjust accordingly",
        "Drought conditions persist in Midwest; agricultural forecasts shift",
        "Record temperatures drive weather market volatility",
    ],
    "crypto": [
        "Bitcoin ETF inflows hit record; BTC futures markets rally",
        "Ethereum upgrade complete — DeFi prediction markets react",
        "Crypto regulation bill advances in Senate; markets price in impact",
        "Altcoin season signals emerge; traders position in crypto niches",
        "Stablecoin reserves audited; market confidence shifts",
    ],
    "sports": [
        "NFL playoff bracket projections update after Week 14",
        "World Cup qualifying drama reshapes soccer prediction markets",
        "F1 championship race heats up with three rounds remaining",
        "MLB trade deadline moves create sharp market movements",
        "UFC main event odds flip after fighter weigh-in",
    ],
    "general": [
        "Prediction market volume hits all-time high across platforms",
        "Top traders outperform benchmarks in volatile week",
        "New research: consistent traders beat random strategy by 23%",
        "Market efficiency study: where are the biggest edges?",
        "Cross-platform analysis reveals arbitrage opportunities",
    ],
}

_EVENTS_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "NBA": [
        {"event": "NBA Finals Game 5",              "date": "+3 days",  "impact": "High"},
        {"event": "Trade deadline",                  "date": "+12 days", "impact": "Medium"},
        {"event": "All-Star weekend announcement",   "date": "+25 days", "impact": "Low"},
    ],
    "politics": [
        {"event": "Primary election results",        "date": "+7 days",  "impact": "High"},
        {"event": "Presidential debate",             "date": "+14 days", "impact": "High"},
        {"event": "Senate committee vote",           "date": "+5 days",  "impact": "Medium"},
    ],
    "weather": [
        {"event": "Hurricane season peak",           "date": "+2 days",  "impact": "High"},
        {"event": "NOAA seasonal outlook release",   "date": "+10 days", "impact": "Medium"},
        {"event": "La Niña advisory update",         "date": "+20 days", "impact": "Low"},
    ],
    "crypto": [
        {"event": "Bitcoin halving countdown",       "date": "+45 days", "impact": "High"},
        {"event": "Ethereum network upgrade",        "date": "+8 days",  "impact": "High"},
        {"event": "SEC crypto ruling expected",      "date": "+30 days", "impact": "High"},
    ],
    "sports": [
        {"event": "Super Bowl",                      "date": "+60 days", "impact": "High"},
        {"event": "Champions League final",          "date": "+40 days", "impact": "High"},
        {"event": "NBA draft lottery",               "date": "+15 days", "impact": "Medium"},
    ],
    "general": [
        {"event": "Q3 earnings season",              "date": "+5 days",  "impact": "Medium"},
        {"event": "Federal Reserve meeting",         "date": "+12 days", "impact": "High"},
        {"event": "Platform maintenance window",     "date": "+2 days",  "impact": "Low"},
    ],
}

_SENTIMENT_LABELS = ["Bullish", "Bearish", "Neutral", "Mixed"]


def _mock_news(niche: str, n: int = 4) -> list[dict[str, str]]:
    templates = _NEWS_TEMPLATES.get(niche, _NEWS_TEMPLATES["general"])
    chosen    = random.sample(templates, k=min(n, len(templates)))
    results   = []
    for i, headline in enumerate(chosen):
        pub_date = datetime.utcnow() - timedelta(hours=random.randint(1, 72))
        results.append({
            "headline":    headline,
            "source":      random.choice(["Reuters", "Bloomberg", "ESPN", "CoinDesk", "Politico"]),
            "published_at": pub_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "relevance":   round(random.uniform(0.65, 0.98), 2),
            "sentiment":   random.choice(_SENTIMENT_LABELS),
            "url":         f"https://mock-news.example.com/article/{niche.lower()}-{i+1}",
        })
    return results


def _mock_events(niche: str) -> list[dict[str, str]]:
    return _EVENTS_TEMPLATES.get(niche, _EVENTS_TEMPLATES["general"])


def _mock_trader_sentiment(trader_id: str) -> dict[str, Any]:
    rng = random.Random(hash(trader_id))
    return {
        "social_mentions": rng.randint(0, 300),
        "sentiment_score": round(rng.uniform(-1.0, 1.0), 3),
        "sentiment_label": rng.choice(_SENTIMENT_LABELS),
        "followers":       rng.randint(0, 5000),
    }


# ---------------------------------------------------------------------------
# Live Apify integration
# ---------------------------------------------------------------------------

def _apify_run_actor(actor_id: str, input_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Trigger an Apify actor synchronously and return dataset items.
    Returns [] on any failure.
    """
    if not APIFY_API_TOKEN:
        return []

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            # Start actor run
            run_resp = client.post(
                f"{APIFY_API_BASE}/acts/{actor_id}/runs",
                params={"token": APIFY_API_TOKEN},
                json=input_data,
            )
            run_resp.raise_for_status()
            run_id = run_resp.json()["data"]["id"]

            # Poll until finished (max 30 s)
            for _ in range(6):
                time.sleep(5)
                status_resp = client.get(
                    f"{APIFY_API_BASE}/actor-runs/{run_id}",
                    params={"token": APIFY_API_TOKEN},
                )
                status_resp.raise_for_status()
                status = status_resp.json()["data"]["status"]
                if status in ("SUCCEEDED", "FAILED", "ABORTED"):
                    break

            if status != "SUCCEEDED":
                return []

            # Fetch results
            dataset_id = run_resp.json()["data"]["defaultDatasetId"]
            items_resp = client.get(
                f"{APIFY_API_BASE}/datasets/{dataset_id}/items",
                params={"token": APIFY_API_TOKEN, "format": "json"},
            )
            items_resp.raise_for_status()
            return items_resp.json()

    except Exception as exc:
        logger.warning("[EnrichmentAgent] Apify actor '%s' failed: %s", actor_id, exc)
        return []


def _fetch_live_news(niche: str) -> list[dict[str, Any]]:
    items = _apify_run_actor(
        APIFY_NEWS_ACTOR,
        {"queries": [f"{niche} prediction market news"], "maxPagesPerQuery": 1, "resultsPerPage": 5},
    )
    if not items:
        return []
    return [
        {
            "headline":    i.get("title",       ""),
            "source":      i.get("displayLink", ""),
            "published_at": i.get("date",       ""),
            "relevance":   0.80,
            "sentiment":   "Neutral",
            "url":         i.get("link", ""),
        }
        for i in items
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_traders(state: dict[str, Any]) -> dict[str, Any]:
    """
    Agent entry-point.

    Adds ``state["enrichment"]`` with:
        - news       : list of recent headlines
        - events     : upcoming events with impact ratings
        - per-trader sentiment signals (stored in trader["enrichment"])

    Parameters
    ----------
    state : dict
        Shared executor state.

    Returns
    -------
    dict
        Updated state with ``enrichment`` key and per-trader enrichment.
    """
    niche   = state.get("niche", "general")
    traders = state.get("traders", [])
    use_mock = MOCK_MODE

    logger.info("[EnrichmentAgent] Enriching | niche=%s | mock=%s | traders=%d",
                niche, use_mock, len(traders))

    # ── Fetch news ────────────────────────────────────────────────────────
    if not use_mock:
        news = _fetch_live_news(niche)
        if not news:
            use_mock = True
    if use_mock:
        news = _mock_news(niche)

    # ── Events ────────────────────────────────────────────────────────────
    events = _mock_events(niche)

    # ── Per-trader signals ────────────────────────────────────────────────
    for t in traders:
        t["enrichment"] = _mock_trader_sentiment(t["trader_id"])

    # ── Aggregate enrichment block ─────────────────────────────────────────
    state["enrichment"] = {
        "niche":            niche,
        "news":             news,
        "events":           events,
        "news_count":       len(news),
        "event_count":      len(events),
        "enrichment_source": "apify_mock" if use_mock else "apify_live",
        "enriched_at":      datetime.utcnow().isoformat(),
    }

    logger.info(
        "[EnrichmentAgent] Enrichment done | news=%d | events=%d",
        len(news), len(events),
    )
    state["traders"] = traders
    return state


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    test_state: dict[str, Any] = {
        "niche":   "politics",
        "traders": [{"trader_id": "poly_0001", "username": "testTrader"}],
    }
    enrich_traders(test_state)
    print(json.dumps(test_state["enrichment"], indent=2))
    print("\nPer-trader enrichment:", test_state["traders"][0].get("enrichment"))
