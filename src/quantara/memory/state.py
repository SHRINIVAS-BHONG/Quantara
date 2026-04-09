from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentState:
    raw_query: str = ""
    platform: str = "both"
    niche: str = "general"
    intent: str = "recommend"

    traders: List[Dict[str, Any]] = field(default_factory=list)
    enrichment: Dict[str, Any] = field(default_factory=dict)
    analysis_summary: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    step_logs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def log_step(self, step: str, status: str):
        self.step_logs.append({"step": step, "status": status})

    def add_error(self, step: str, error: str):
        self.errors.append({"step": step, "error": error})

    def summary(self):
        return {
            "recommendation": self.explanation,
            "top_traders": self.traders[:10],
            "reasoning": {
                "query": self.raw_query,
                "platform": self.platform,
                "niche": self.niche,
                "intent": self.intent,
                "steps": self.step_logs,
                "errors": self.errors,
            },
        }