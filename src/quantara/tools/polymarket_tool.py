from quantara.core.polymarket import fetch_traders
from quantara.rag.retriever import ingest_traders
from tools.registry import registry, tool_result, tool_error

POLYMARKET_FETCH_SCHEMA = {
    "name": "polymarket_fetch",
    "description": "Fetch top predictive market traders and metrics from Polymarket given a specific niche.",
    "parameters": {
        "type": "object",
        "properties": {
            "niche": {
                "type": "string", 
                "description": "The target niche (e.g., 'crypto', 'politics', 'sports')."
            }
        },
        "required": ["niche"]
    }
}

def polymarket_fetch_handler(args: dict, **kwargs) -> str:
    niche = args.get("niche", "general")
    try:
        traders = fetch_traders(niche=niche)
        # Closed Learning Loop: ingest into RAG for future queries
        try:
            ingest_traders(traders)
        except Exception:
            pass  # RAG ingestion failure should never block results
        return tool_result(traders=traders)
    except Exception as e:
        return tool_error(f"Failed to fetch Polymarket traders: {e}")

registry.register(
    name="polymarket_fetch",
    toolset="quantara",
    schema=POLYMARKET_FETCH_SCHEMA,
    handler=polymarket_fetch_handler,
)
