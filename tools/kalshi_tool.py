from hermes_app.core.kalshi import fetch_traders
from hermes_app.memory.state import AgentState


def kalshi_tool(state: AgentState) -> AgentState:
    try:
        result = fetch_traders({
            "niche": state.niche,
            "traders": state.traders
        })

        state.traders = result.get("traders", state.traders)

        state.log_step("kalshi_fetch", "success")

    except Exception as e:
        state.add_error("kalshi_fetch", str(e))
        state.log_step("kalshi_fetch", "error")

    return state