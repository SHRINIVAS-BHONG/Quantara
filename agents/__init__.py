"""
agents/
=======
Multi-agent pipeline for prediction-market trader analysis.

Exported modules
----------------
planner          – Query → structured execution plan
router           – Plan step → agent function mapping
executor         – Step-by-step pipeline orchestrator
polymarket_agent – Polymarket trader data fetcher
kalshi_agent     – Kalshi trader data fetcher
niche_classifier – Trader niche classification (rule + LLM)
analysis_agent   – Performance metrics & scoring
enrichment_agent – External context via Apify
"""

from agents.planner          import plan
from agents.router           import Router
from agents.executor         import Executor, run_pipeline
from agents.polymarket_agent import fetch_traders as polymarket_fetch
from agents.kalshi_agent     import fetch_traders as kalshi_fetch
from agents.niche_classifier import classify_by_niche
from agents.analysis_agent   import analyze_traders, compute_trader_score
from agents.enrichment_agent import enrich_traders

__all__ = [
    "plan",
    "Router",
    "Executor",
    "run_pipeline",
    "polymarket_fetch",
    "kalshi_fetch",
    "classify_by_niche",
    "analyze_traders",
    "compute_trader_score",
    "enrich_traders",
]
