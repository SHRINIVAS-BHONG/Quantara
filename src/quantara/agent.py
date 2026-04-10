import os
import logging
from dotenv import load_dotenv

# Import quantara tools so they register with the Hermes registry
import quantara.tools

try:
    from run_agent import AIAgent
except ImportError:
    AIAgent = None

logger = logging.getLogger(__name__)

def run_quantara_agent(query: str):
    """
    Runs the Quantara agent using the Hermes framework loop.
    All state, memory, and LLM orchestration is offloaded to Hermes.
    """
    load_dotenv()
    
    if AIAgent is None:
        raise ImportError(
            "hermes-agent is not installed. Please install it via:\n"
            "pip install 'git+https://github.com/NousResearch/hermes-agent.git'"
        )

    logger.info("Delegating to Hermes with query: %s", query)

    # STRICT REQUIREMENT: Use OpenRouter
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "missing_key")
    # Using the generic free router for maximum availability
    os.environ["LLM_MODEL"] = "openrouter/free"
    
    # STRICT REQUIREMENT: Closed Learning Loop
    os.environ["ENABLE_MEMORY"] = "true"

    # Enable our custom toolset in Hermes
    os.environ["AUTO_TOOLSET"] = "quantara"

    system_prompt = (
        "You are the Quantara Trading Expert. Provide deep, data-driven insights. "
        "Check RAG (rag_search) first, then Live APIs. Present data in a professional table."
    )

    # Initialize the agent
    agent = AIAgent(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        enabled_toolsets=["quantara"],
        ephemeral_system_prompt=system_prompt,
        quiet_mode=True  # Reduce console noise for a cleaner 'proper' output
    )

    # Execute the hermes agent workflow
    return agent.chat(query)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Find the best Polymarket crypto traders."
    run_quantara_agent(query)