from typing import Any


def classify_by_niche(state: dict[str, Any]) -> dict[str, Any]:
    niche = state.get("niche", "general")

    traders = state.get("traders", [])

    filtered = [
        t for t in traders
        if niche == "general" or t["niche"].lower() == niche.lower()
    ]

    state["traders"] = filtered
    return state