# Examples and Expected Deliverables

## 1. Finding Consistent Traders in Polymarket
**Input Prompt**: "Search for the top Polymarket traders in the Politics niche to copy trade."

**Agent Thought Process (Hermes learning loop)**:
1. `planner_analyze({"query": "Search for the top Polymarket traders in the Politics niche to copy trade."})` -> Extract niche `politics`.
2. `polymarket_fetch({"niche": "politics"})` -> The agent queries the Gamma API in `core/polymarket.py`. It pulls the active Trump/Election markets. For those active markets, it discovers the high-liquidity volume wallet identifiers.
3. `trader_analysis({...})` -> Sorts the discovered wallets by their risk/roi metric.

**Output**:
> "I scanned active Polymarket events under the `politics` tag. \n\nTarget Event: presidential-election-winner\nTop Wallets:\n1. 0xPolyWalletPolitics0A8F (Win Rate: 50%, ROI: 10%)\n2. 0xPolyWalletPolitics1A8F (Win Rate: 55%, ROI: 15%)\n\nI recommend copying the highest ROI wallet here."

---

## 2. Using Apify context inside RAG
**Input Prompt**: "Who should I copy trade for the upcoming Fed Rate prediction on Kalshi?"

**Agent Thought Process**:
1. `planner_analyze` -> Extracted niche `crypto/finance`.
2. `kalshi_fetch({"niche": "finance"})` -> Hits Kalshi's `v2/events` API for open finance events.
3. `trader_enrichment({"traders": [list], "niche": "finance"})` -> Starts an active Apify web-scraper using `apify-client`. It pulls live Google News sentiment surrounding the upcoming Fed Rate cuts to append to the trader's risk profiles.
4. `rag_search({"query": "upcoming Fed Rate", "niche": "finance"})` -> Checks past agent memory to avoid recommending previously failed rate-cut traders.

**Output**:
> "I fetched active Kalshi finance markets. Apify web sentiment currently shows 'highly bullish' expectations for a rate-cut. Based on this augmented sentiment, I strongly recommend shadowing KalshiWhale_FINANCE_0 as they have a 66% historical win rate against bullish external sentiment scenarios."
