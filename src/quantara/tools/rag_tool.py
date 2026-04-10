from quantara.rag.rag_agent import run_rag_agent
from tools.registry import registry, tool_result, tool_error

RAG_SCHEMA = {
    "name": "rag_search",
    "description": "Access the Quantara Historical Memory. ALWAYS search here first for consistent trader profiles, historical win rates, and saved market context before querying live APIs.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "niche": {"type": "string", "description": "Optional niche constraint."},
            "top_k": {"type": "integer", "description": "Number of results to return.", "default": 5}
        },
        "required": ["query"]
    }
}

def rag_search_handler(args: dict, **kwargs) -> str:
    query = args.get("query", "")
    niche = args.get("niche")
    top_k = args.get("top_k", 5)
    try:
        result = run_rag_agent(query=query, niche=niche, top_k=top_k)
        return tool_result(result)
    except Exception as e:
        return tool_error(f"RAG search failed: {e}")

registry.register(
    name="rag_search", 
    toolset="quantara", 
    schema=RAG_SCHEMA, 
    handler=rag_search_handler
)
