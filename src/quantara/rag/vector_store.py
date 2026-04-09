import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _build_vocab(docs: List[Document]) -> List[str]:
    vocab = set()
    for doc in docs:
        vocab.update(_tokenize(doc.text))
    return sorted(vocab)


def _tf_vector(tokens: List[str], vocab: List[str]) -> List[float]:
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    total = max(len(tokens), 1)
    return [freq.get(w, 0) / total for w in vocab]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class VectorStore:
    PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "vector_store.json"

    def __init__(self):
        self.docs: List[Document] = []
        self.vocab: List[str] = []
        self.dirty = True
        self._load()

    def add_documents(self, texts: List[str], metas: List[dict]) -> List[str]:
        ids = []
        for i, text in enumerate(texts):
            doc = Document(
                id=str(uuid.uuid4()),
                text=text,
                metadata=metas[i] if metas else {},
            )
            self.docs.append(doc)
            ids.append(doc.id)

        self.dirty = True
        self._save()
        return ids

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self.docs:
            return []

        self._reindex()

        q_vec = _tf_vector(_tokenize(query), self.vocab)

        scored = [(doc, _cosine(q_vec, doc.embedding)) for doc in self.docs]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:k]

    def _reindex(self):
        if not self.dirty:
            return

        self.vocab = _build_vocab(self.docs)

        for doc in self.docs:
            doc.embedding = _tf_vector(_tokenize(doc.text), self.vocab)

        self.dirty = False

    def _save(self):
        self.PATH.parent.mkdir(parents=True, exist_ok=True)
        with self.PATH.open("w") as f:
            json.dump([{"id": d.id, "text": d.text, "metadata": d.metadata} for d in self.docs], f)

    def _load(self):
        if self.PATH.exists():
            with self.PATH.open() as f:
                raw = json.load(f)
            self.docs = [Document(**d) for d in raw]
            self.dirty = True


_store = None


def get_vector_store():
    global _store
    if _store is None:
        _store = VectorStore()
    return _store