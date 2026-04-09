from typing import Any, Dict, List
from quantara.rag.retriever import retrieve_traders, ingest_traders


def run_rag_agent(
    query: str,
    niche: str | None = None,
    top_k: int = 5,
    extra_context: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:

    # 1. Ingest latest traders
    if extra_context:
        ingest_traders(extra_context)

    # 2. Retrieve relevant traders
    retrieved = retrieve_traders(query, k=top_k, niche=niche)

    # 3. Generate simple explanation (no external LLM)
    explanation = _generate_explanation(query, retrieved)

    # 4. Extract traders
    top_traders = [
        {
            "trader_id": r["metadata"].get("trader_id"),
            "platform": r["metadata"].get("platform"),
            "niche": r["metadata"].get("niche"),
            "score": r["metadata"].get("score"),
        }
        for r in retrieved
    ]

    return {
        "recommendation": explanation,
        "top_traders": top_traders,
        "reasoning": _build_reasoning(retrieved),
        "context_used": len(retrieved),
    }


def _generate_explanation(query: str, docs: List[dict]) -> str:
    if not docs:
        return "No strong traders found."

    return (
        f"Based on analysis, the top traders for '{query}' show strong performance "
        f"with high win rates and ROI. Consider following top-ranked traders."
    )


def _build_reasoning(docs: List[dict]) -> str:
    if not docs:
        return "No data available."

    top = docs[0]["metadata"]

    return (
        f"Top trader: {top.get('trader_id')} "
        f"with score {top.get('score')} "
        f"in niche {top.get('niche')}."
    )