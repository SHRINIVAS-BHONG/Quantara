"""
Kalshi Agent
------------
Fetches trader data from Kalshi (regulated US prediction-market exchange).
Production mode: uses Kalshi's REST API v2 (requires API key).
Fallback / dev mode: returns realistic simulated data.
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
KALSHI_API_BASE = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_API_KEY  = os.getenv("KALSHI_API_KEY", "")
REQUEST_TIMEOUT = 10
MAX_TRADERS     = 50
MOCK_MODE: bool = os.getenv("KALSHI_MOCK", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Simulated data
# ---------------------------------------------------------------------------

_NICHES = ["NBA", "politics", "crypto", "weather", "sports", "general"]
_NAMES  = [
    "kalshiKing", "eventEdge", "forecastPro", "ruleTrader", "probabilist",
    "marketMaker", "betaTrader", "sharpCal", "consensusBuster", "oddsLord",
    "regulatedRick", "preciseBet", "signalSeek", "noiseFilter", "contractPro",
    "hedgeHog", "spreadTrader", "volatilityVince", "momentumMark", "trendTom",
]


def _random_date(days_back: int = 365) -> str:
    dt = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
    return dt.strftime("%Y-%m-%d")


def _generate_mock_trader(idx: int) -> dict[str, Any]:
    """Create one realistic Kalshi trader record."""
    rng = random.Random(idx * 17 + 3)

    niche    = rng.choice(_NICHES)
    win_rate = round(rng.uniform(0.38, 0.80), 4)
    roi      = round(rng.uniform(-0.08, 0.42), 4)
    risk     = round(rng.uniform(0.06, 0.38), 4)
    trades   = rng.randint(10, 500)
    vol_usd  = round(rng.uniform(300, 180_000), 2)
    score    = round(win_rate * 0.5 + roi * 0.3 - risk * 0.2, 4)

    contracts = [
        f"KALSHI_{niche.upper()}_Q{rng.randint(1, 4)}_{rng.randint(2023, 2025)}"
        for _ in range(rng.randint(2, 7))
    ]

    return {
        "trader_id":       f"kalshi_{idx:04d}",
        "username":        rng.choice(_NAMES) + str(rng.randint(10, 999)),
        "platform":        "kalshi",
        "niche":           niche,
        "win_rate":        win_rate,
        "roi":             roi,
        "risk":            risk,
        "score":           score,
        "total_trades":    trades,
        "volume_usd":      vol_usd,
        "markets_traded":  contracts,
        "joined_date":     _random_date(730),
        "last_active":     _random_date(30),
        "streak_wins":     rng.randint(0, 10),
        "streak_losses":   rng.randint(0, 6),
        "avg_position_usd": round(vol_usd / max(trades, 1), 2),
        "metadata": {
            "source":      "kalshi_mock",
            "fetched_at":  datetime.utcnow().isoformat(),
        },
    }


def _generate_mock_traders(n: int = 30, niche_filter: str = "") -> list[dict[str, Any]]:
    traders = [_generate_mock_trader(i) for i in range(1, n + 1)]
    if niche_filter and niche_filter.lower() not in ("", "general", "all"):
        traders = [t for t in traders if t["niche"].lower() == niche_filter.lower()]
        if len(traders) < 5:
            traders = [_generate_mock_trader(i) for i in range(1, n + 1)]
    return traders


# ---------------------------------------------------------------------------
# Live API helpers
# ---------------------------------------------------------------------------

def _fetch_live_traders(niche: str) -> list[dict[str, Any]]:
    """
    Pull data from Kalshi v2 /markets endpoint.
    Returns [] on failure so the caller can fall back to mock.
    """
    if not KALSHI_API_KEY:
        logger.warning("[KalshiAgent] No KALSHI_API_KEY set; skipping live fetch.")
        return []

    headers = {
        "Authorization": f"Bearer {KALSHI_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT, headers=headers) as client:
            resp = client.get(
                f"{KALSHI_API_BASE}/markets",
                params={"limit": 100, "status": "open"},
            )
            resp.raise_for_status()
            markets = resp.json().get("markets", [])

        traders: list[dict[str, Any]] = []
        for mkt in markets[:MAX_TRADERS]:
            yes_bid  = float(mkt.get("yes_bid",  0.5)) / 100
            yes_ask  = float(mkt.get("yes_ask",  0.5)) / 100
            win_rate = round((yes_bid + yes_ask) / 2, 4)
            volume   = float(mkt.get("volume",   0))
            roi      = round((win_rate - 0.5) * 0.75, 4)
            risk     = round(1.0 - win_rate, 4)
            score    = round(win_rate * 0.5 + roi * 0.3 - risk * 0.2, 4)

            ticker = mkt.get("ticker", f"kalshi_{len(traders):04d}")
            traders.append({
                "trader_id":      ticker,
                "username":       ticker[:14],
                "platform":       "kalshi",
                "niche":          niche or "general",
                "win_rate":       win_rate,
                "roi":            roi,
                "risk":           risk,
                "score":          score,
                "total_trades":   int(mkt.get("volume", 0)),
                "volume_usd":     volume,
                "markets_traded": [mkt.get("ticker", "unknown")],
                "joined_date":    "",
                "last_active":    mkt.get("close_time", ""),
                "metadata": {
                    "source":       "kalshi_live",
                    "market_ticker": ticker,
                    "fetched_at":   datetime.utcnow().isoformat(),
                },
            })

        logger.info("[KalshiAgent] Fetched %d live traders", len(traders))
        return traders

    except Exception as exc:
        logger.warning("[KalshiAgent] Live fetch failed (%s); will use mock data.", exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_traders(state: dict[str, Any]) -> dict[str, Any]:
    """
    Agent entry-point.  Reads ``state["niche"]`` and appends to ``state["traders"]``.

    Parameters
    ----------
    state : dict
        Shared executor state dict.

    Returns
    -------
    dict
        Updated state with ``traders`` populated (appended, not replaced).
    """
    niche    = state.get("niche", "general")
    use_mock = MOCK_MODE

    logger.info("[KalshiAgent] Fetching traders | niche=%s | mock=%s", niche, use_mock)
    t0 = time.perf_counter()

    if not use_mock:
        traders = _fetch_live_traders(niche)
        if not traders:
            use_mock = True

    if use_mock:
        traders = _generate_mock_traders(n=35, niche_filter=niche)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("[KalshiAgent] Returned %d traders in %.1f ms", len(traders), elapsed)

    existing = state.get("traders", [])
    state["traders"] = existing + traders
    return state


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    sample_state: dict[str, Any] = {"niche": "politics", "traders": []}
    fetch_traders(sample_state)
    print(f"Total traders: {len(sample_state['traders'])}")
    print(json.dumps(sample_state["traders"][0], indent=2))
