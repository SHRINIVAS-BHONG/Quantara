from typing import Any


def enrich_traders(state: dict[str, Any]) -> dict[str, Any]:
    traders = state.get("traders", [])

    for t in traders:
        t["enrichment"] = {
            "sentiment": "neutral",
            "mentions": 10
        }

    state["enrichment"] = {"status": "added"}
    state["traders"] = traders
    return state