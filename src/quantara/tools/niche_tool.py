from quantara.core.niche import classify_by_niche
from tools.registry import registry, tool_result, tool_error

NICHE_SCHEMA = {
    "name": "niche_classifier",
    "description": "Filter traders by niche.",
    "parameters": {
        "type": "object",
        "properties": {
            "traders": {
                "type": "array", 
                "description": "List of traders to filter.",
                "items": {"type": "object"}
            },
            "niche": {"type": "string", "description": "The niche to filter by."}
        },
        "required": ["traders", "niche"]
    }
}

def niche_classifier_handler(args: dict, **kwargs) -> str:
    traders = args.get("traders", [])
    niche = args.get("niche", "general")
    try:
        result = classify_by_niche(traders=traders, niche=niche)
        return tool_result(traders=result)
    except Exception as e:
        return tool_error(f"Failed to filter traders by niche: {e}")

registry.register(
    name="niche_classifier", 
    toolset="quantara", 
    schema=NICHE_SCHEMA, 
    handler=niche_classifier_handler
)
