from hermes_app.memory.state import AgentState

from hermes_app.tools.planner_tool import planner_tool
from hermes_app.tools.polymarket_tool import polymarket_tool
from hermes_app.tools.kalshi_tool import kalshi_tool
from hermes_app.tools.niche_tool import niche_tool
from hermes_app.tools.analysis_tool import analysis_tool
from hermes_app.tools.enrichment_tool import enrichment_tool
from hermes_app.tools.rag_tool import rag_tool


def run_agent(query: str) -> dict:
    state = AgentState(raw_query=query)

    planner_tool(state)

    if state.platform in ["polymarket", "both"]:
        polymarket_tool(state)

    if state.platform in ["kalshi", "both"]:
        kalshi_tool(state)

    niche_tool(state)
    analysis_tool(state)
    enrichment_tool(state)

    rag_tool(state)

    return state.summary()