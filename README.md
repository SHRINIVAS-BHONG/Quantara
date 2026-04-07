# Prediction Market Multi-Agent System

A production-grade Python backend for finding, classifying, analysing, and recommending consistent traders on **Polymarket** and **Kalshi** using a dynamic Planner → Router → Executor architecture.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────┐
│  Planner    │  LLM-powered: query → JSON execution plan
│  Agent      │  (platform, niche, intent, ordered steps)
└──────┬──────┘
       │ plan (JSON)
       ▼
┌─────────────┐
│  Router     │  Maps each plan step → concrete agent function
│  Agent      │
└──────┬──────┘
       │ routed step map
       ▼
┌─────────────────────────────────────────────────────┐
│                   Executor Engine                   │
│  Runs agents in order, threading state through each │
│                                                     │
│  search_traders → polymarket_agent / kalshi_agent   │
│  filter_by_niche → niche_classifier                 │
│  analyze_performance → analysis_agent               │
│  enrich_context → enrichment_agent                  │
│  rank_traders → scoring (utils/scoring.py)          │
│  generate_explanation → rag_agent                   │
└──────┬──────────────────────────────────────────────┘
       │ result
       ▼
┌─────────────┐       ┌──────────────────────┐
│  RAG Agent  │◄──────│  Vector Store (TF-IDF│
│             │       │  or FAISS)            │
└──────┬──────┘       └──────────────────────┘
       │
       ▼
┌─────────────┐
│  Learning   │  Stores feedback, adjusts scores over time
│  Loop       │  (decay-weighted, persisted to JSON)
└─────────────┘
```

### Key Design Decisions

| Concern | Solution |
|---------|----------|
| LLM dependency | OpenRouter-compatible + deterministic mock fallback |
| Vector DB | In-memory TF-IDF cosine (no deps); drop-in FAISS upgrade |
| Data sources | Simulated realistic data; Apify mock with real interface |
| Persistence | JSON files in `data/` (swap for SQLite/Postgres easily) |
| Learning | Decay-weighted feedback accumulation, per-trader adjustments |

---

## Project Structure

```
project/
├── agents/
│   ├── planner.py          # Query → JSON plan (LLM or mock)
│   ├── router.py           # Plan steps → agent functions
│   ├── executor.py         # Step-by-step dynamic execution
│   ├── polymarket_agent.py # Polymarket trader data
│   ├── kalshi_agent.py     # Kalshi trader data
│   ├── niche_classifier.py # Market → niche classification
│   ├── analysis_agent.py   # Win rate / ROI / consistency
│   └── enrichment_agent.py # Apify / news context
│
├── rag/
│   ├── vector_store.py     # TF-IDF in-memory store + JSON persistence
│   ├── retriever.py        # Ingestion + semantic retrieval
│   └── rag_agent.py        # RAG pipeline + LLM response generation
│
├── learning/
│   └── feedback.py         # Feedback storage + decay-weighted score updates
│
├── api/
│   └── main.py             # FastAPI app (POST /query, POST /feedback, etc.)
│
├── utils/
│   └── scoring.py          # Scoring formula, risk, consistency helpers
│
├── data/                   # Auto-created: vector_store.json, feedback.json, etc.
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone <repo>
cd project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables (optional)

Create a `.env` file:

```env
# LLM (OpenRouter key for real responses; omit for mock mode)
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=openai/gpt-4o-mini

# Apify (for real enrichment; omit for mock mode)
APIFY_API_KEY=apify_api_...
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## API Usage

### `POST /query`

Run the full agent pipeline for a natural-language query.

**Request:**
```json
{
  "query": "Find best NBA traders in Polymarket"
}
```

**Response:**
```json
{
  "recommendation": "Based on historical performance data...",
  "top_traders": [
    {
      "trader_id": "poly_nba_001",
      "platform": "polymarket",
      "niche": "NBA",
      "analyst_score": 0.6814,
      "retrieval_score": 0.5921
    }
  ],
  "reasoning": "Top match: trader 'poly_nba_001' on polymarket (niche=NBA, score=0.681).",
  "plan": {
    "platform": "polymarket",
    "niche": "NBA",
    "intent": "recommend",
    "steps": ["search_traders", "filter_by_niche", "analyze_performance", "rank_traders", "generate_explanation"]
  },
  "execution_log": ["[1/5] search_traders ...", "..."],
  "latency_ms": 142.3
}
```

---

### Sample Queries & Expected Behaviour

#### Query 1 — NBA Traders on Polymarket
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find best NBA traders in Polymarket"}'
```
**Expected plan:**
```json
{
  "platform": "polymarket",
  "niche": "NBA",
  "intent": "recommend",
  "steps": ["search_traders", "filter_by_niche", "analyze_performance", "rank_traders", "generate_explanation"]
}
```
**Expected output:** Top 5 NBA traders from Polymarket, ranked by composite score (win_rate×0.5 + roi×0.3 − risk×0.2), with LLM-generated recommendation.

---

#### Query 2 — Politics Traders on Kalshi
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Should I copy traders in politics on Kalshi?"}'
```
**Expected plan:**
```json
{
  "platform": "kalshi",
  "niche": "politics",
  "intent": "recommend",
  "steps": ["search_traders", "filter_by_niche", "analyze_performance", "enrich_context", "rank_traders", "generate_explanation"]
}
```
**Expected output:** Ranked political traders on Kalshi + news enrichment context + copy-trading advice.

---

#### Query 3 — Cross-platform Crypto
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who are the top consistent crypto traders across both platforms?"}'
```
**Expected plan:**
```json
{
  "platform": "both",
  "niche": "crypto",
  "intent": "recommend",
  "steps": ["search_traders", "filter_by_niche", "analyze_performance", "rank_traders", "generate_explanation"]
}
```
**Expected output:** Combined Polymarket + Kalshi crypto traders, deduplicated and ranked.

---

### `POST /feedback`

Submit feedback to improve future recommendations.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "trader_id": "poly_nba_001",
    "platform": "polymarket",
    "query": "Find best NBA traders in Polymarket",
    "recommendation_score": 0.68,
    "user_rating": "positive",
    "outcome": "won",
    "delta": 0.15
  }'
```

---

### `GET /adjustments`

View all learning-loop score adjustments:

```bash
curl http://localhost:8000/adjustments
```

---

## Scoring Formula

```
score = (win_rate × 0.5) + (roi × 0.3) − (risk × 0.2) + consistency_bonus
```

| Component | Weight | Notes |
|-----------|--------|-------|
| `win_rate` | 0.5 | Fraction of winning trades |
| `roi` | 0.3 | Compound return on investment |
| `risk` | −0.2 | Normalised variance + drawdown |
| `consistency_bonus` | +0–0.1 | Trade count + Sharpe ratio bonus |

---

## Learning Loop

1. Every `/feedback` call stores a `FeedbackRecord` to `data/feedback.json`.
2. A decay-weighted adjustment is computed:
   - `+0.02` per positive/won event
   - `−0.03` per negative/lost event
   - `+delta × 0.1` from actual P&L
   - Exponential decay with 30-day half-life
3. Adjustments are stored in `data/score_adjustments.json`.
4. Every `/query` response applies these adjustments to the returned `top_traders`.

---

## Extending the System

| Goal | Where to change |
|------|----------------|
| Add a real Polymarket API | `agents/polymarket_agent.py` — replace mock with HTTP call |
| Add FAISS | `rag/vector_store.py` — swap `_cosine` section for `faiss.IndexFlatL2` |
| Add a real Apify scraper | `agents/enrichment_agent.py` — replace mock with `httpx` call to Apify actor |
| Use a real LLM | Set `OPENROUTER_API_KEY` in `.env` |
| Persist to PostgreSQL | Replace JSON helpers in `learning/feedback.py` |
