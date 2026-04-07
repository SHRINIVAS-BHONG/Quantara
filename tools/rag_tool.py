from hermes_app.rag.rag_agent import run_rag_agent
from hermes_app.memory.state import AgentState


def rag_tool(state: AgentState) -> AgentState:
    try:
        result = run_rag_agent(
            query=state.raw_query,
            niche=state.niche,
            extra_context=state.traders
        )

        state.explanation = result.get("recommendation", "")
        state.traders = result.get("top_traders", state.traders)

        state.log_step("rag", "success")

    except Exception as e:
        state.add_error("rag", str(e))
        state.log_step("rag", "error")

    return state