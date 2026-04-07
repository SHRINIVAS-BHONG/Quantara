from hermes_app.core.polymarket import fetch_traders
from hermes_app.memory.state import AgentState


def polymarket_tool(state: AgentState) -> AgentState:
    try:
        result = fetch_traders({
            "niche": state.niche,
            "traders": state.traders
        })

        state.traders = result.get("traders", state.traders)

        state.log_step("polymarket_fetch", "success")

    except Exception as e:
        state.add_error("polymarket_fetch", str(e))
        state.log_step("polymarket_fetch", "error")

    return state