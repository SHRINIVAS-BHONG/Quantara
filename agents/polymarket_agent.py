"""
Polymarket Agent
----------------
Fetches trader data from Polymarket.
Production mode: calls the public Polymarket API (CLOB v2).
Fallback / dev mode: returns realistic simulated data when the API
is unreachable or no key is configured.
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
POLYMARKET_API_BASE = "https://clob.polymarket.com"
REQUEST_TIMEOUT     = 10          # seconds
MAX_TRADERS         = 50          # cap returned traders
MOCK_MODE: bool     = os.getenv("POLYMARKET_MOCK", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Simulated data generators
# ---------------------------------------------------------------------------

_NICHES    = ["NBA", "politics", "crypto", "weather", "sports", "general"]
_NAMES     = [
    "apex_trader", "sharpBettor", "quantumOdds", "marketWizard", "edgeFinder",
    "polyPro", "contractKing", "oddsHunter", "valueBet99", "predictionGuru",
    "kalshiKing", "swingTrader", "momentumEdge", "consistentCal", "riskMgr7",
    "statArb", "nlpSignal", "trendRider", "contrarian88", "alphaSeeker",
]

def _random_date(days_back: int = 365) -> str:
    dt = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
    return dt.strftime("%Y-%m-%d")


def _generate_mock_trader(idx: int) -> dict[str, Any]:
    """Create a single realistic-looking trader record."""
    seed = idx * 31 + 7
    rng  = random.Random(seed)

    niche      = rng.choice(_NICHES)
    win_rate   = round(rng.uniform(0.40, 0.82), 4)
    roi        = round(rng.uniform(-0.05, 0.45), 4)
    risk       = round(rng.uniform(0.05, 0.40), 4)
    trades     = rng.randint(15, 600)
    volume_usd = round(rng.uniform(500, 250_000), 2)

    # composite score (matches analysis_agent formula)
    score = round(win_rate * 0.5 + roi * 0.3 - risk * 0.2, 4)

    markets_traded = [
        f"{niche}_market_{rng.randint(1, 50)}" for _ in range(rng.randint(2, 8))
    ]

    return {
        "trader_id":       f"poly_{idx:04d}",
        "username":        rng.choice(_NAMES) + str(rng.randint(10, 999)),
        "platform":        "polymarket",
        "niche":           niche,
        "win_rate":        win_rate,
        "roi":             roi,
        "risk":            risk,
        "score":           score,
        "total_trades":    trades,
        "volume_usd":      volume_usd,
        "markets_traded":  markets_traded,
        "joined_date":     _random_date(730),
        "last_active":     _random_date(30),
        "streak_wins":     rng.randint(0, 12),
        "streak_losses":   rng.randint(0, 5),
        "avg_position_usd": round(volume_usd / max(trades, 1), 2),
        "metadata": {
            "source":      "polymarket_mock",
            "fetched_at":  datetime.utcnow().isoformat(),
        },
    }


def _generate_mock_traders(n: int = 30, niche_filter: str = "") -> list[dict[str, Any]]:
    traders = [_generate_mock_trader(i) for i in range(1, n + 1)]
    if niche_filter and niche_filter.lower() not in ("", "general", "all"):
        traders = [t for t in traders if t["niche"].lower() == niche_filter.lower()]
        # If filter is too strict, loosen it so we always return ≥ 5 results
        if len(traders) < 5:
            traders = [_generate_mock_trader(i) for i in range(1, n + 1)]
    return traders


# ---------------------------------------------------------------------------
# Live API helpers
# ---------------------------------------------------------------------------

def _fetch_live_traders(niche: str) -> list[dict[str, Any]]:
    """
    Attempt to pull trader/market data from Polymarket's CLOB API.
    Returns an empty list if the request fails (caller falls back to mock).
    """
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(f"{POLYMARKET_API_BASE}/markets", params={"limit": 100})
            resp.raise_for_status()
            markets = resp.json().get("data", [])

        traders: list[dict[str, Any]] = []
        for mkt in markets[:MAX_TRADERS]:
            # Polymarket's public CLOB doesn't expose per-trader leaderboards;
            # we derive a synthetic record from market-level data.
            trader_id = mkt.get("maker_address") or mkt.get("condition_id", f"poly_{len(traders):04d}")
            win_rate  = float(mkt.get("bestBid", 0.5))
            volume    = float(mkt.get("volume", 0))
            roi       = round((win_rate - 0.5) * 0.8, 4)
            risk      = round(1.0 - win_rate, 4)
            score     = round(win_rate * 0.5 + roi * 0.3 - risk * 0.2, 4)

            traders.append({
                "trader_id":      trader_id,
                "username":       trader_id[:12],
                "platform":       "polymarket",
                "niche":          niche or "general",
                "win_rate":       round(win_rate, 4),
                "roi":            roi,
                "risk":           risk,
                "score":          score,
                "total_trades":   int(mkt.get("tradesCount", 0)),
                "volume_usd":     volume,
                "markets_traded": [mkt.get("condition_id", "unknown")],
                "joined_date":    "",
                "last_active":    mkt.get("endDateIso", ""),
                "metadata": {
                    "source":     "polymarket_live",
                    "market_id":  mkt.get("condition_id"),
                    "fetched_at": datetime.utcnow().isoformat(),
                },
            })
        logger.info("[PolymarketAgent] Fetched %d live traders", len(traders))
        return traders

    except Exception as exc:
        logger.warning("[PolymarketAgent] Live fetch failed (%s); will use mock data.", exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_traders(state: dict[str, Any]) -> dict[str, Any]:
    """
    Agent entry-point.  Reads ``state["niche"]`` and writes ``state["traders"]``.

    Parameters
    ----------
    state : dict
        Shared executor state dict.

    Returns
    -------
    dict
        Updated state with ``traders`` key populated.
    """
    niche    = state.get("niche", "general")
    use_mock = MOCK_MODE

    logger.info("[PolymarketAgent] Fetching traders | niche=%s | mock=%s", niche, use_mock)
    t0 = time.perf_counter()

    if not use_mock:
        traders = _fetch_live_traders(niche)
        if not traders:
            use_mock = True  # fallback

    if use_mock:
        traders = _generate_mock_traders(n=40, niche_filter=niche)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("[PolymarketAgent] Returned %d traders in %.1f ms", len(traders), elapsed)

    # Preserve any existing traders (e.g., from a previous platform agent)
    existing = state.get("traders", [])
    state["traders"] = existing + traders
    return state


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    sample_state: dict[str, Any] = {"niche": "NBA", "traders": []}
    fetch_traders(sample_state)
    print(f"Total traders: {len(sample_state['traders'])}")
    print(json.dumps(sample_state["traders"][0], indent=2))
