import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

def enrich_traders(traders: list[dict[str, Any]], niche: str = "general") -> dict[str, Any]:
    """
    Use Apify to enrich trader profiles or the niche sentiment.
    """
    apify_token = os.getenv("APIFY_API_TOKEN")
    
    if not apify_token:
        logger.warning("APIFY_API_TOKEN not found in .env. Skipping real Apify extraction.")
        for t in traders:
            t["enrichment"] = {
                "sentiment": "neutral",
                "news_mentions": 0,
                "context": "Apify token missing. Context unavailable."
            }
        return {"traders": traders, "enrichment_status": "skipped_no_token"}
        
    try:
        from apify_client import ApifyClient
        client = ApifyClient(apify_token)
        
        logger.info(f"Running Apify actor for niche: {niche}")
        
        # Run Apify google-search-scraper to retrieve active web context
        run = client.actor("apify/google-search-scraper").call(run_input={
             "queries": f"{niche} prediction market top traders",
             "maxPagesPerQuery": 1,
             "resultsPerPage": 3,
        })
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        
        # Process the Apify search results into context
        extracted_context = " | ".join([item.get("snippet", "") for item in items])
        sentiment = "bullish" if "bullish" in extracted_context.lower() else "neutral"

        for t in traders:
            t["enrichment"] = {
                "sentiment": sentiment,
                "news_mentions": len(items),
                "context": extracted_context[:100] + "..."
            }
            
        return {"traders": traders, "enrichment_status": "apify_success"}

    except ImportError:
        logger.error("apify-client not installed.")
        return {"traders": traders, "enrichment_status": "error_no_client"}
    except Exception as e:
        logger.error(f"Apify enrichment failed: {e}")
        return {"traders": traders, "enrichment_status": f"error: {e}"}