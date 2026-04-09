from quantara.core.analysis import analyze_traders
from tools.registry import registry, tool_result, tool_error

ANALYSIS_SCHEMA = {
    "name": "trader_analysis",
    "description": "Analyze and score a list of traders.",
    "parameters": {
        "type": "object",
        "properties": {
            "traders": {
                "type": "array", 
                "description": "List of traders to analyze.",
                "items": {"type": "object"}
            }
        },
        "required": ["traders"]
    }
}

def trader_analysis_handler(args: dict) -> str:
    traders = args.get("traders", [])
    try:
        result = analyze_traders(traders=traders)
        return tool_result(result)
    except Exception as e:
        return tool_error(f"Failed to analyze traders: {e}")

registry.register(
    name="trader_analysis", 
    toolset="quantara", 
    schema=ANALYSIS_SCHEMA, 
    handler=trader_analysis_handler
)