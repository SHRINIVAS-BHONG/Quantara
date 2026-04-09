from typing import Any


def classify_by_niche(traders: list[dict[str, Any]], niche: str = "general") -> list[dict[str, Any]]:
    filtered = [
        t for t in traders
        if niche == "general" or t.get("niche", "").lower() == niche.lower()
    ]

    return filtered