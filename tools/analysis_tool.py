from hermes_app.core.analysis import analyze_traders
from hermes_app.memory.state import AgentState


def analysis_tool(state: AgentState) -> AgentState:
    try:
        result = analyze_traders({
            "traders": state.traders
        })

        state.traders = result.get("traders", state.traders)
        state.analysis_summary = result.get("analysis_summary", {})

        state.log_step("analysis", "success")

    except Exception as e:
        state.add_error("analysis", str(e))
        state.log_step("analysis", "error")

    return state