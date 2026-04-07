"""
rag_agent.py
Retrieval-Augmented Generation agent.
Retrieves relevant trader/market context and uses an LLM to produce
a natural-language recommendation or explanation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from rag.retriever import retrieve_traders, retrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM interface (OpenRouter-compatible; falls back to mock)
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Calls an OpenRouter-compatible LLM.
    Falls back to deterministic mock when OPENROUTER_API_KEY is absent.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            import httpx  # type: ignore

            payload = {
                "model": os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 800,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("LLM call failed (%s) – using mock.", exc)

    # ---- Mock LLM (no API key needed) ------------------------------------
    return _mock_llm_response(user_prompt)


def _mock_llm_response(prompt: str) -> str:
    """Deterministic mock that generates a plausible-sounding response."""
    if "nba" in prompt.lower() or "basketball" in prompt.lower():
        domain = "NBA basketball"
    elif "politic" in prompt.lower():
        domain = "political"
    elif "crypto" in prompt.lower():
        domain = "crypto"
    elif "weather" in prompt.lower():
        domain = "weather"
    else:
        domain = "prediction market"

    return (
        f"Based on historical performance data and current market signals, "
        f"the top {domain} traders show strong consistency metrics. "
        f"The leading traders exhibit win rates above 60% and positive ROI "
        f"over the last 30-day window. "
        f"Key factors driving their success include disciplined position sizing, "
        f"early market entry, and leveraging niche expertise. "
        f"Consider copy-trading the top-ranked traders with a diversified allocation "
        f"strategy to mitigate individual position risk."
    )


# ---------------------------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert prediction-market analyst assistant.
Your job is to recommend traders, explain performance data, and guide users
in copy-trading decisions.

When given context documents and a user question:
1. Synthesize the trader data precisely.
2. Highlight the best candidates with specific metrics.
3. Explain *why* those traders stand out.
4. Flag any notable risks (low trade count, high variance, etc.).
5. Be concise, factual, and actionable.

Context is provided in JSON-like blocks. Always ground your answer in the data.
"""


def run_rag_agent(
    query: str,
    niche: str | None = None,
    top_k: int = 5,
    extra_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Main entry point for the RAG agent.

    Parameters
    ----------
    query         : User's natural-language query.
    niche         : Optional niche filter for retrieval.
    top_k         : Number of trader documents to retrieve.
    extra_context : Additional context dicts (e.g. from analysis_agent).

    Returns
    -------
    {
        "recommendation": str,
        "top_traders":    list[dict],
        "reasoning":      str,
        "context_used":   int,
    }
    """
    logger.info("RAG agent: query=%r, niche=%r", query, niche)

    # 1. Retrieve relevant trader documents
    retrieved = retrieve_traders(query, k=top_k, niche=niche)

    # 2. Optionally blend in extra_context passed from executor
    if extra_context:
        for item in extra_context:
            if isinstance(item, dict) and item.get("trader_id"):
                retrieved.append(
                    {
                        "text": _build_context_text(item),
                        "metadata": item,
                        "score": item.get("score", 0),
                        "combined_score": item.get("score", 0),
                    }
                )
        # Re-sort
        retrieved.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        retrieved = retrieved[:top_k]

    # 3. Build user prompt with retrieved context
    context_block = _format_context(retrieved)
    user_prompt = (
        f"User question: {query}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"Please provide a recommendation."
    )

    # 4. Call LLM
    llm_response = _call_llm(SYSTEM_PROMPT, user_prompt)

    # 5. Extract top traders for structured output
    top_traders = _extract_top_traders(retrieved)

    return {
        "recommendation": llm_response,
        "top_traders": top_traders,
        "reasoning": _build_reasoning(retrieved),
        "context_used": len(retrieved),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context_text(trader: dict[str, Any]) -> str:
    return (
        f"Trader {trader.get('trader_id')} | "
        f"Platform: {trader.get('platform')} | "
        f"Niche: {trader.get('niche')} | "
        f"Win Rate: {trader.get('win_rate', 0):.2%} | "
        f"ROI: {trader.get('roi', 0):.2%} | "
        f"Score: {trader.get('score', 0):.3f}"
    )


def _format_context(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "No relevant traders found in the vector store."
    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{i}] {doc['text']} (retrieval_score={doc.get('combined_score', doc.get('score', 0)):.3f})")
    return "\n".join(lines)


def _extract_top_traders(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traders: list[dict[str, Any]] = []
    for doc in docs:
        meta = doc.get("metadata", {})
        if meta.get("trader_id"):
            traders.append(
                {
                    "trader_id": meta["trader_id"],
                    "platform": meta.get("platform", "unknown"),
                    "niche": meta.get("niche", "general"),
                    "analyst_score": round(float(meta.get("score", 0)), 4),
                    "retrieval_score": round(float(doc.get("combined_score", doc.get("score", 0))), 4),
                }
            )
    return traders


def _build_reasoning(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "Insufficient data to generate reasoning."
    top = docs[0]
    meta = top.get("metadata", {})
    return (
        f"Top match: trader '{meta.get('trader_id', 'N/A')}' on {meta.get('platform', 'N/A')} "
        f"(niche={meta.get('niche', 'N/A')}, analyst_score={meta.get('score', 0):.3f}). "
        f"Retrieved {len(docs)} contextually relevant trader profiles."
    )
