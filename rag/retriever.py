from typing import Any, List, Dict
from rag.vector_store import get_vector_store


def _to_text(trader: dict) -> str:
    return (
        f"Trader {trader.get('trader_id')} | "
        f"{trader.get('platform')} | "
        f"{trader.get('niche')} | "
        f"Win {trader.get('win_rate')} | "
        f"ROI {trader.get('roi')} | "
        f"Score {trader.get('score')}"
    )


def ingest_traders(traders: List[dict]) -> int:
    store = get_vector_store()

    texts = [_to_text(t) for t in traders]
    metas = traders

    store.add_documents(texts, metas)
    return len(texts)


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    store = get_vector_store()

    results = store.similarity_search(query, k=k)

    output = []
    for doc, score in results:
        output.append({
            "text": doc.text,
            "metadata": doc.metadata,
            "score": score,
        })

    return output


def retrieve_traders(query: str, k: int = 5, niche: str | None = None):
    docs = retrieve(query, k=k * 2)

    traders = []
    for d in docs:
        meta = d["metadata"]

        if niche and meta.get("niche") != niche:
            continue

        d["combined_score"] = 0.6 * d["score"] + 0.4 * meta.get("score", 0)
        traders.append(d)

    traders.sort(key=lambda x: x["combined_score"], reverse=True)

    return traders[:k]