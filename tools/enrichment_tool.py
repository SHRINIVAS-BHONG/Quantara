from hermes_app.core.enrichment import enrich_traders
from hermes_app.memory.state import AgentState


def enrichment_tool(state: AgentState) -> AgentState:
    try:
        result = enrich_traders({
            "niche": state.niche,
            "traders": state.traders
        })

        state.traders = result.get("traders", state.traders)
        state.enrichment = result.get("enrichment", {})

        state.log_step("enrichment", "success")

    except Exception as e:
        state.add_error("enrichment", str(e))
        state.log_step("enrichment", "error")

    return state