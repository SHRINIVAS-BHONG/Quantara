from quantara.core.enrichment import enrich_traders
from tools.registry import registry, tool_result, tool_error

ENRICHMENT_SCHEMA = {
    "name": "trader_enrichment",
    "description": "Enrich traders with additional fields like sentiment.",
    "parameters": {
        "type": "object",
        "properties": {
            "traders": {
                "type": "array", 
                "description": "List of traders to enrich.",
                "items": {"type": "object"}
            },
            "niche": {"type": "string"}
        },
        "required": ["traders"]
    }
}

def trader_enrichment_handler(args: dict) -> str:
    traders = args.get("traders", [])
    niche = args.get("niche", "general")
    try:
        result = enrich_traders(traders=traders, niche=niche)
        return tool_result(result)
    except Exception as e:
        return tool_error(f"Failed to enrich traders: {e}")

registry.register(
    name="trader_enrichment", 
    toolset="quantara", 
    schema=ENRICHMENT_SCHEMA, 
    handler=trader_enrichment_handler
)