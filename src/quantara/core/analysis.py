from typing import Any


def compute_score(win_rate: float, roi: float, risk: float) -> float:
    return round(win_rate * 0.5 + roi * 0.3 - risk * 0.2, 4)


def analyze_traders(traders: list[dict[str, Any]]) -> dict[str, Any]:
    for t in traders:
        score = compute_score(
            t.get("win_rate", 0),
            t.get("roi", 0),
            t.get("risk", 0),
        )
        t["score"] = score

    traders.sort(key=lambda x: x.get("score", 0), reverse=True)

    return {"traders": traders, "analysis_summary": {"highest_score": traders[0]["score"] if traders else 0}}