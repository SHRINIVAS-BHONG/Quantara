from hermes_app.core.planner import plan
from hermes_app.memory.state import AgentState


def planner_tool(state: AgentState) -> AgentState:
    try:
        result = plan(state.raw_query)

        state.platform = result.get("platform", "both")
        state.niche = result.get("niche", "general")
        state.intent = result.get("intent", "recommend")

        state.log_step("planner", "success")

    except Exception as e:
        state.add_error("planner", str(e))
        state.log_step("planner", "error")

    return state