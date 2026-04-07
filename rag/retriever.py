"""
retriever.py
High-level retrieval interface used by rag_agent.
Handles document ingestion from trader / market data and semantic search.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rag.vector_store import Document, get_vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _trader_to_text(trader: dict[str, Any]) -> str:
    """Serialize a trader record into a searchable text blob."""
    parts = [
        f"Trader: {trader.get('trader_id', 'unknown')}",
        f"Platform: {trader.get('platform', 'unknown')}",
        f"Niche: {trader.get('niche', 'general')}",
        f"Win Rate: {trader.get('win_rate', 0):.2%}",
        f"ROI: {trader.get('roi', 0):.2%}",
        f"Score: {trader.get('score', 0):.3f}",
        f"Total Trades: {trader.get('total_trades', 0)}",
        f"Markets: {', '.join(trader.get('markets', [])[:5])}",
    ]
    if trader.get("news_context"):
        parts.append(f"Context: {trader['news_context']}")
    return " | ".join(parts)


def ingest_traders(traders: list[dict[str, Any]]) -> int:
    """
    Ingest a list of trader dicts into the vector store.
    Returns the number of documents added.
    """
    store = get_vector_store()
    texts = [_trader_to_text(t) for t in traders]
    metadatas = [
        {
            "trader_id": t.get("trader_id"),
            "platform": t.get("platform"),
            "niche": t.get("niche"),
            "score": t.get("score", 0),
        }
        for t in traders
    ]
    ids = store.add_documents(texts, metadatas)
    logger.info("Retriever: ingested %d traders → %d doc IDs", len(traders), len(ids))
    return len(ids)


def ingest_market_events(events: list[dict[str, Any]]) -> int:
    """
    Ingest market event records (news, enrichment) into the vector store.
    """
    store = get_vector_store()
    texts: list[str] = []
    metas: list[dict] = []
    for ev in events:
        text = (
            f"Event: {ev.get('title', '')} | "
            f"Category: {ev.get('category', '')} | "
            f"Summary: {ev.get('summary', '')} | "
            f"Source: {ev.get('source', '')}"
        )
        texts.append(text)
        metas.append({"type": "market_event", "category": ev.get("category", "")})
    ids = store.add_documents(texts, metas)
    logger.info("Retriever: ingested %d market events", len(ids))
    return len(ids)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = 5, filter_niche: str | None = None) -> list[dict[str, Any]]:
    """
    Retrieve top-k documents relevant to *query*.

    Parameters
    ----------
    query        : Natural language query string.
    k            : Number of results to return.
    filter_niche : If provided, only return docs whose metadata niche matches.

    Returns
    -------
    List of dicts with keys: text, metadata, score.
    """
    store = get_vector_store()
    results = store.similarity_search(query, k=k * 3)   # over-fetch then filter

    output: list[dict[str, Any]] = []
    for doc, score in results:
        if filter_niche and doc.metadata.get("niche", "").lower() != filter_niche.lower():
            continue
        output.append(
            {
                "text": doc.text,
                "metadata": doc.metadata,
                "score": round(score, 4),
            }
        )
        if len(output) >= k:
            break

    logger.info("Retriever: query=%r → %d results", query, len(output))
    return output


def retrieve_traders(query: str, k: int = 5, niche: str | None = None) -> list[dict[str, Any]]:
    """
    Convenience wrapper that only returns trader-type documents,
    sorted by a combined retrieval + analyst score.
    """
    raw = retrieve(query, k=k * 2, filter_niche=niche)
    traders = [r for r in raw if r["metadata"].get("trader_id")]

    # Re-rank by blending cosine score with stored analyst score
    for r in traders:
        cosine = r["score"]
        analyst = float(r["metadata"].get("score", 0))
        r["combined_score"] = round(0.6 * cosine + 0.4 * analyst, 4)

    traders.sort(key=lambda x: x["combined_score"], reverse=True)
    return traders[:k]
