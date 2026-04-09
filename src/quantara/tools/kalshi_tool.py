from quantara.core.kalshi import fetch_traders
from quantara.rag.retriever import ingest_traders
from tools.registry import registry, tool_result, tool_error

KALSHI_FETCH_SCHEMA = {
    "name": "kalshi_fetch",
    "description": "Fetch top predictive market traders and metrics from Kalshi given a specific niche.",
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

def kalshi_fetch_handler(args: dict, **kwargs) -> str:
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
        return tool_error(f"Failed to fetch Kalshi traders: {e}")

registry.register(
    name="kalshi_fetch",
    toolset="quantara",
    schema=KALSHI_FETCH_SCHEMA,
    handler=kalshi_fetch_handler,
)
