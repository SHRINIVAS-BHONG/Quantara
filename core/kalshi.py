import random
from typing import Any


def fetch_traders(state: dict[str, Any]) -> dict[str, Any]:
    niche = state.get("niche", "general")

    traders = []
    for i in range(20):
        traders.append({
            "trader_id": f"kalshi_{i}",
            "platform": "kalshi",
            "niche": niche,
            "win_rate": round(random.uniform(0.4, 0.8), 2),
            "roi": round(random.uniform(-0.1, 0.4), 2),
            "risk": round(random.uniform(0.1, 0.4), 2),
        })

    state["traders"] = state.get("traders", []) + traders
    return state