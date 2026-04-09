from quantara.core.planner import plan
from tools.registry import registry, tool_result, tool_error

PLANNER_SCHEMA = {
    "name": "planner_analyze",
    "description": "Analyze a user query to determine platform, niche, and intent.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The user query."}
        },
        "required": ["query"]
    }
}

def planner_handler(args: dict) -> str:
    query = args.get("query", "")
    try:
        result = plan(query)
        return tool_result(result)
    except Exception as e:
        return tool_error(f"Planner failed: {e}")

registry.register(
    name="planner_analyze", 
    toolset="quantara", 
    schema=PLANNER_SCHEMA, 
    handler=planner_handler
)