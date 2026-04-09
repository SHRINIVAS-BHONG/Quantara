"""
Seed script for Quantara RAG vector store.
Run this once before starting the agent, or whenever you want to refresh 
the historical trader context.
"""
import json
from pathlib import Path
from quantara.core.polymarket import fetch_traders as poly_fetch
from quantara.core.kalshi import fetch_traders as kalshi_fetch
from quantara.core.analysis import analyze_traders
from quantara.rag.retriever import ingest_traders

# ── 1. Clear old vector store so we start fresh ─────────────────────────────
vs_path = Path(__file__).parent / "data" / "vector_store.json"
if vs_path.exists():
    vs_path.unlink()
    print("Cleared old vector store.")
else:
    vs_path.parent.mkdir(parents=True, exist_ok=True)

# Also reset the in-memory singleton
import quantara.rag.vector_store as _vs_module
_vs_module._store = None

# ── 2. Fetch + score + ingest for each niche ─────────────────────────────────
NICHES = ["politics", "crypto", "sports", "NBA"]

print("\n=== Seeding RAG vector store ===\n")

all_scored = []
for niche in NICHES:
    poly  = poly_fetch(niche=niche)
    kalshi = kalshi_fetch(niche=niche)
    combined = poly + kalshi

    result = analyze_traders(combined)
    scored = result.get("traders", [])
    all_scored.extend(scored)

    ingest_traders(scored)
    print(f"  [{niche:10s}] ingested {len(scored)} scored traders "
          f"(top score: {scored[0]['score'] if scored else 'N/A'})")

# ── 3. Verify ─────────────────────────────────────────────────────────────────
from quantara.rag.retriever import retrieve_traders

print(f"\nTotal stored: {len(all_scored)} traders across {len(NICHES)} niches")
print("\nVerifying retrieval...")

test_queries = [
    ("politics election 2026", "politics"),
    ("crypto bitcoin market", "crypto"),
    ("NBA basketball season", "NBA"),
]

for query, niche in test_queries:
    docs = retrieve_traders(query, k=2, niche=niche)
    print(f"\n  Query: '{query}'")
    for d in docs:
        m = d["metadata"]
        print(f"    -> {m.get('trader_id')} | score={m.get('score')} | "
              f"platform={m.get('platform')} | niche={m.get('niche')}")

print("\n=== RAG ready! Next agent run will use historical context. ===")
