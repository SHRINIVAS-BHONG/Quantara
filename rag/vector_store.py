"""
vector_store.py
In-memory vector store using cosine similarity (no external dependencies).
Optionally upgrades to FAISS if available.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            # embeddings are not persisted (regenerated on load)
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))


# ---------------------------------------------------------------------------
# Lightweight TF-IDF-style embedding (no ML deps required)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _build_vocab(docs: list[Document]) -> list[str]:
    vocab: set[str] = set()
    for doc in docs:
        vocab.update(_tokenize(doc.text))
    return sorted(vocab)


def _tf_vector(tokens: list[str], vocab: list[str]) -> list[float]:
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = max(len(tokens), 1)
    return [freq.get(w, 0) / total for w in vocab]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Simple in-memory vector store backed by TF-IDF cosine similarity.
    Persists documents (without embeddings) to a JSON file so they survive
    restarts; embeddings are recomputed lazily after each add/load.
    """

    PERSIST_PATH = Path("data/vector_store.json")

    def __init__(self, persist: bool = True) -> None:
        self._docs: list[Document] = []
        self._vocab: list[str] = []
        self._dirty = True          # embeddings need recomputation
        self._persist = persist
        if persist:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """Add documents and return their assigned IDs."""
        ids: list[str] = []
        for i, text in enumerate(texts):
            meta = (metadatas[i] if metadatas else {}) or {}
            doc = Document(id=str(uuid.uuid4()), text=text, metadata=meta)
            self._docs.append(doc)
            ids.append(doc.id)
        self._dirty = True
        if self._persist:
            self._save()
        logger.info("VectorStore: added %d documents (total=%d)", len(texts), len(self._docs))
        return ids

    def similarity_search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Return top-k documents most similar to query, with scores."""
        if not self._docs:
            return []
        self._maybe_reindex()
        q_tokens = _tokenize(query)
        q_vec = _tf_vector(q_tokens, self._vocab)
        scored = [
            (doc, _cosine(q_vec, doc.embedding))
            for doc in self._docs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def get_all(self) -> list[Document]:
        return list(self._docs)

    def clear(self) -> None:
        self._docs = []
        self._vocab = []
        self._dirty = False
        if self._persist:
            self._save()

    def __len__(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_reindex(self) -> None:
        if not self._dirty:
            return
        self._vocab = _build_vocab(self._docs)
        for doc in self._docs:
            doc.embedding = _tf_vector(_tokenize(doc.text), self._vocab)
        self._dirty = False

    def _save(self) -> None:
        try:
            self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.PERSIST_PATH.open("w") as f:
                json.dump([d.to_dict() for d in self._docs], f, indent=2)
        except Exception as exc:
            logger.warning("VectorStore: could not persist – %s", exc)

    def _load(self) -> None:
        try:
            if self.PERSIST_PATH.exists():
                with self.PERSIST_PATH.open() as f:
                    raw = json.load(f)
                self._docs = [Document.from_dict(d) for d in raw]
                self._dirty = True
                logger.info("VectorStore: loaded %d documents from disk", len(self._docs))
        except Exception as exc:
            logger.warning("VectorStore: could not load from disk – %s", exc)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(persist=True)
    return _store
