"""
Router Agent
------------
Maps each step name from the Planner's execution plan to the concrete
callable that should handle it.  The router is platform-aware and
injects the correct market agent (Polymarket / Kalshi / both).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
AgentFn = Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Step → handler registry
# ---------------------------------------------------------------------------

def _import_agents() -> dict[str, Any]:
    """Lazy-import agents to avoid circular dependencies at module load."""
    from agents.polymarket_agent  import fetch_traders as poly_fetch
    from agents.kalshi_agent      import fetch_traders as kalshi_fetch
    from agents.niche_classifier  import classify_by_niche
    from agents.analysis_agent    import analyze_traders
    from agents.enrichment_agent  import enrich_traders

    # rag_agent and learning are imported elsewhere (rag/ and learning/)
    try:
        from rag.rag_agent import generate_explanation
    except ImportError:
        def generate_explanation(state: dict) -> dict:  # type: ignore[misc]
            state.setdefault("explanation", "RAG module not loaded.")
            return state

    return {
        "poly_fetch":         poly_fetch,
        "kalshi_fetch":       kalshi_fetch,
        "classify_by_niche":  classify_by_niche,
        "analyze_traders":    analyze_traders,
        "enrich_traders":     enrich_traders,
        "generate_explanation": generate_explanation,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Router:
    """
    Resolves each step name in the execution plan to a callable agent function.

    The router also handles the 'both' platform case by composing the two
    market-agent fetchers into a single merged call.
    """

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan    = plan
        self.platform = plan.get("platform", "both")
        self._agents  = _import_agents()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_traders_handler(self) -> AgentFn:
        """Return the right fetch function based on platform."""
        poly   = self._agents["poly_fetch"]
        kalshi = self._agents["kalshi_fetch"]

        if self.platform == "polymarket":
            return poly
        if self.platform == "kalshi":
            return kalshi

        # "both" → merge results
        def _combined(state: dict[str, Any]) -> dict[str, Any]:
            poly_state   = poly(dict(state))
            kalshi_state = kalshi(dict(state))
            merged = poly_state.get("traders", []) + kalshi_state.get("traders", [])
            state["traders"] = merged
            logger.info("[Router] Combined traders from both platforms: %d total", len(merged))
            return state

        return _combined

    def _make_rank_handler(self) -> AgentFn:
        """Rank traders already in state by their computed score (desc)."""
        def _rank(state: dict[str, Any]) -> dict[str, Any]:
            traders = state.get("traders", [])
            ranked  = sorted(traders, key=lambda t: t.get("score", 0.0), reverse=True)
            state["traders"] = ranked
            logger.info("[Router] Ranked %d traders by score.", len(ranked))
            return state

        return _rank

    # ------------------------------------------------------------------
    # Core resolution
    # ------------------------------------------------------------------

    def resolve(self, step: str) -> AgentFn:
        """
        Map a step name → callable.

        Parameters
        ----------
        step : str
            One of the step names produced by the Planner.

        Returns
        -------
        AgentFn
            A function with signature ``fn(state: dict) -> dict``.

        Raises
        ------
        ValueError
            If the step name is not recognised.
        """
        mapping: dict[str, AgentFn | Callable[[], AgentFn]] = {
            "search_traders":      lambda: self._search_traders_handler(),
            "filter_by_niche":     lambda: self._agents["classify_by_niche"],
            "analyze_performance": lambda: self._agents["analyze_traders"],
            "enrich_context":      lambda: self._agents["enrich_traders"],
            "rank_traders":        lambda: self._make_rank_handler(),
            "generate_explanation": lambda: self._agents["generate_explanation"],
        }

        factory = mapping.get(step)
        if factory is None:
            raise ValueError(
                f"[Router] Unknown step '{step}'. "
                f"Valid steps: {list(mapping.keys())}"
            )

        handler: AgentFn = factory()  # type: ignore[operator]
        logger.info("[Router] Step '%s' → %s", step, getattr(handler, "__name__", repr(handler)))
        return handler

    def route_all(self) -> list[tuple[str, AgentFn]]:
        """
        Resolve every step in the plan and return an ordered list of
        (step_name, handler) tuples.
        """
        return [(step, self.resolve(step)) for step in self.plan.get("steps", [])]


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_plan = {
        "platform": "polymarket",
        "niche":    "NBA",
        "intent":   "recommend",
        "steps":    ["search_traders", "filter_by_niche", "analyze_performance", "rank_traders", "generate_explanation"],
        "raw_query": "Find best NBA traders in Polymarket",
    }
    router = Router(sample_plan)
    for name, fn in router.route_all():
        print(f"  {name:30s} → {getattr(fn, '__name__', repr(fn))}")
