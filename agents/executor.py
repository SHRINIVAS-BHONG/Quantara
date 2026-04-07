"""
Executor Engine
---------------
Dynamically runs each (step, handler) pair produced by the Router,
maintaining a shared state dictionary that flows between agents.
Every step is logged with timing and error handling.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agents.router import Router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Orchestrates the end-to-end execution of an agent plan.

    Usage
    -----
    ::
        plan     = planner.plan(query)
        executor = Executor(plan)
        result   = executor.run()
    """

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan   = plan
        self.router = Router(plan)

        # Shared mutable state passed through every agent
        self.state: dict[str, Any] = {
            "plan":      plan,
            "platform":  plan.get("platform", "both"),
            "niche":     plan.get("niche",    "general"),
            "intent":    plan.get("intent",   "recommend"),
            "raw_query": plan.get("raw_query", ""),
            "traders":   [],
            "enrichment": {},
            "explanation": "",
            "step_logs":  [],          # audit trail
            "errors":     [],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_step(self, step: str, handler) -> None:  # type: ignore[type-arg]
        """Execute a single agent step, updating shared state in-place."""
        log_entry: dict[str, Any] = {"step": step, "status": "pending", "duration_ms": 0}
        self.state["step_logs"].append(log_entry)

        logger.info("━━━ [Executor] Running step: %-30s ━━━", step)
        t0 = time.perf_counter()

        try:
            result = handler(self.state)

            # Handlers may return a new dict or mutate state in-place
            if isinstance(result, dict):
                self.state.update(result)

            duration = round((time.perf_counter() - t0) * 1000, 1)
            log_entry.update({"status": "success", "duration_ms": duration})
            logger.info("    ✓ Step '%s' completed in %.1f ms", step, duration)

        except Exception as exc:
            duration = round((time.perf_counter() - t0) * 1000, 1)
            log_entry.update({"status": "error", "error": str(exc), "duration_ms": duration})
            self.state["errors"].append({"step": step, "error": str(exc)})
            logger.error("    ✗ Step '%s' FAILED: %s", step, exc, exc_info=True)
            # Non-fatal: continue to next step with degraded data

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Execute all steps in order and return the final state.

        Returns
        -------
        dict
            The accumulated state after all agents have run.  Always
            contains at minimum:
            - ``traders``      : list of trader dicts (possibly empty)
            - ``explanation``  : LLM-generated text (possibly empty)
            - ``step_logs``    : per-step audit trail
            - ``errors``       : list of {step, error} dicts
        """
        total_start = time.perf_counter()
        steps = self.plan.get("steps", [])
        logger.info("[Executor] Starting execution — %d steps for query: '%s'",
                    len(steps), self.state["raw_query"])

        routed = self.router.route_all()
        for step, handler in routed:
            self._run_step(step, handler)

        total_ms = round((time.perf_counter() - total_start) * 1000, 1)
        logger.info("[Executor] Pipeline complete in %.1f ms  |  errors=%d",
                    total_ms, len(self.state["errors"]))

        self.state["total_duration_ms"] = total_ms
        return self.state

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Return a lean summary of the final result suitable for API responses.
        """
        traders = self.state.get("traders", [])
        top_n   = traders[:10]   # cap at 10 for API response

        return {
            "recommendation": self.state.get("explanation", ""),
            "top_traders":    top_n,
            "reasoning": {
                "query":    self.state.get("raw_query"),
                "platform": self.state.get("platform"),
                "niche":    self.state.get("niche"),
                "intent":   self.state.get("intent"),
                "steps_run": [
                    {"step": s["step"], "status": s["status"], "duration_ms": s["duration_ms"]}
                    for s in self.state.get("step_logs", [])
                ],
                "errors": self.state.get("errors", []),
            },
        }


# ---------------------------------------------------------------------------
# Pipeline entry-point (convenience function)
# ---------------------------------------------------------------------------

def run_pipeline(query: str) -> dict[str, Any]:
    """
    High-level helper: plan → route → execute → summarise.

    Parameters
    ----------
    query : str
        Raw user question.

    Returns
    -------
    dict
        API-ready summary dict.
    """
    from agents.planner import plan as make_plan

    exec_plan = make_plan(query)
    executor  = Executor(exec_plan)
    executor.run()
    return executor.summary()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    result = run_pipeline("Find the best NBA traders in Polymarket")
    print(json.dumps(result, indent=2, default=str))
