from hermes_app.core.niche import classify_by_niche
from hermes_app.memory.state import AgentState


def niche_tool(state: AgentState) -> AgentState:
    try:
        result = classify_by_niche({
            "niche": state.niche,
            "traders": state.traders
        })

        state.traders = result.get("traders", state.traders)

        state.log_step("niche_classification", "success")

    except Exception as e:
        state.add_error("niche_classification", str(e))
        state.log_step("niche_classification", "error")

    return state