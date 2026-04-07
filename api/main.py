"""
main.py
FastAPI application — entry point for the multi-agent prediction-market system.

Endpoints
---------
POST /query          Run a full planner → router → executor pipeline.
POST /feedback       Submit feedback for a previous recommendation.
GET  /traders        List all traders currently in the vector store.
GET  /adjustments    View learning-loop score adjustments.
GET  /health         Health check.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

# ── ensure project root is on sys.path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.planner import run_planner
from agents.router import route_plan
from agents.executor import run_executor
from learning.feedback import (
    store_feedback,
    get_all_adjustments,
    apply_learning_to_traders,
)
from rag.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prediction Market Multi-Agent API",
    description=(
        "AI-powered system for finding, classifying, and analysing "
        "consistent traders on Polymarket and Kalshi."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural-language user query")
    debug: bool = Field(False, description="Include intermediate agent outputs")


class QueryResponse(BaseModel):
    recommendation: str
    top_traders: list[dict[str, Any]]
    reasoning: str
    plan: dict[str, Any]
    execution_log: list[str]
    latency_ms: float


class FeedbackRequest(BaseModel):
    trader_id: str
    platform: str
    query: str
    recommendation_score: float = 0.0
    user_rating: Literal["positive", "negative", "neutral"] = "neutral"
    outcome: Literal["won", "lost", "pending"] = "pending"
    delta: float = 0.0
    notes: str = ""


class FeedbackResponse(BaseModel):
    feedback_id: str
    trader_id: str
    adjustment_applied: bool
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "prediction-market-agents"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """
    Full pipeline: Planner → Router → Executor → RAG response.

    Example body:
    ```json
    { "query": "Find best NBA traders in Polymarket" }
    ```
    """
    start = time.perf_counter()
    try:
        # 1. Plan
        plan = run_planner(req.query)
        logger.info("Plan generated: %s", plan)

        # 2. Route
        routed = route_plan(plan)
        logger.info("Routed steps: %s", list(routed.keys()))

        # 3. Execute
        result = run_executor(
            plan=plan,
            routed_steps=routed,
            original_query=req.query,
        )

        # 4. Apply learning-loop adjustments to top_traders
        top_traders = apply_learning_to_traders(result.get("top_traders", []))

        elapsed = (time.perf_counter() - start) * 1000

        return QueryResponse(
            recommendation=result.get("recommendation", ""),
            top_traders=top_traders,
            reasoning=result.get("reasoning", ""),
            plan=plan,
            execution_log=result.get("execution_log", []),
            latency_ms=round(elapsed, 2),
        )

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
    """
    Submit feedback for a trader recommendation.
    Updates the learning-loop score adjustment.
    """
    try:
        record = store_feedback(
            trader_id=req.trader_id,
            platform=req.platform,
            query=req.query,
            recommendation_score=req.recommendation_score,
            user_rating=req.user_rating,
            outcome=req.outcome,
            delta=req.delta,
            notes=req.notes,
        )
        return FeedbackResponse(
            feedback_id=record.feedback_id,
            trader_id=record.trader_id,
            adjustment_applied=True,
            message=f"Feedback stored. Score adjustment updated for {req.trader_id}.",
        )
    except Exception as exc:
        logger.exception("Feedback error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/traders")
async def list_traders(limit: int = 20) -> dict[str, Any]:
    """Return documents currently stored in the vector store (for debugging)."""
    store = get_vector_store()
    docs = store.get_all()[:limit]
    return {
        "total": len(store),
        "shown": len(docs),
        "documents": [
            {"text": d.text[:120] + "…", "metadata": d.metadata}
            for d in docs
        ],
    }


@app.get("/adjustments")
async def list_adjustments() -> dict[str, Any]:
    """Return all learning-loop score adjustments."""
    adjs = get_all_adjustments()
    return {
        "count": len(adjs),
        "adjustments": {k: vars(v) for k, v in adjs.items()},
    }


# ---------------------------------------------------------------------------
# Sample queries (returned from root for discoverability)
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    {
        "query": "Find best NBA traders in Polymarket",
        "description": "Search + classify + analyse NBA niche traders",
    },
    {
        "query": "Should I copy traders in politics on Kalshi?",
        "description": "Recommend political traders on Kalshi with risk analysis",
    },
    {
        "query": "Who are the top consistent crypto traders across both platforms?",
        "description": "Cross-platform crypto trader recommendation",
    },
]


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "message": "Prediction Market Multi-Agent API",
        "docs": "/docs",
        "sample_queries": SAMPLE_QUERIES,
    }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
