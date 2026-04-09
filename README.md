# 🧠 CrowdWisdomTrading AI Agent
> A Multi-Agent Engine built on the **Hermes-Agent** framework to orchestrate prediction market discovery, sentiment enrichment, and copy-trading analytics.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Hermes%20Agent-orange)](https://github.com/nousresearch/hermes-agent)
[![LLM](https://img.shields.io/badge/LLM-Meta--Llama--3--8b--Instruct-green)](https://openrouter.ai/)
[![Enrichment](https://img.shields.io/badge/Enrichment-Apify-yellow)](https://apify.com/)

---

## 📖 Project Overview

This project is a powerful AI-driven backend searching and research tool for global predictions markets. It utilizes **Hermes-Agent** as the core library orchestrator, offloading all cognitive routing, tool execution, and state memory to the LLM. 

By defining specialized sub-agents natively plugged into the `tools.registry`, the system is capable of aggregating real-time market data across **Polymarket** and **Kalshi**, analyzing web sentiment via **Apify**, and returning conversational trading recommendations supported by an autonomous **Closed Learning Loop**.

---

## 📡 Agent Architecture & Flow Diagram

The entire cognitive execution is driven using `run_agent(use_memory=True)` pointing at OpenRouter's Llama endpoint. When a user queries the application, Hermes intelligently maps and coordinates the following tool schemas:

```mermaid
flowchart TD
    %% Main Title
    Main["CrowdWisdomTrading Prediction AI Agent<br/>(Hermes Framework – LLM Reasoning & Search)"]

    %% Top Level Branches
    Main -- Consists of --> UserAgent["User Input Sequence<br/>• Perceives Query<br/>• Maintains Chat Context<br/>• Triggers Execution Search"]
    Main -- Consists of --> MarketEnv["Prediction Market Environment<br/>• Polymarket / Kalshi / Apify<br/>• Attributes: Volume, Odds, Traders, Sentiment"]
    Main -- Uses for --> Output["Output<br/>• Ranked Trader Recommendations<br/>• Explanations & Sentiment Context<br/>• Specific Event Copy-Targets"]

    %% Perception & Market Data Flow
    UserAgent -- Analyzes --> Perception["Perception Module & Planner<br/>• Intent & Niche Extraction<br/>• Market Categorization (Crypto/Politics)<br/>• Constraint Identification"]
    MarketEnv -- Provides Data --> KnowledgeBase["Environment Knowledge Base<br/>• Polymarket Gamma API<br/>• Kalshi REST Market Depth<br/>• Apify Live Web Sentiment"]
    
    %% Core Engine
    Perception -- Extracted Features --> Reasoning["LLM Reasoning Engine<br/>(Hermes Core + OpenRouter)<br/>• Tool Selection & Delegation<br/>• Trader Analysis & Scoring<br/>• RAG Historical Context Merge"]
    KnowledgeBase -- Retrieves Market Info --> Reasoning

    %% Decision & Action
    Goal(["Goal<br/>Recommend the highest-ROI<br/>trader aligning with requested<br/>prediction market event."]) -- Guides --> Reasoning
    
    Reasoning -- Selects Best --> Decision["Decision / Action<br/>• Execute specific Market APIs<br/>• Choose Top Traders to Copy<br/>• Formulate Trade Thesis"]

    %% Learning Loop (Memory)
    Decision -- Updates --> NewState["New State (Closed Learning Loop)<br/>• Update Hermes Memory SQLite<br/>• Store Interaction Context<br/>• Update Vector RAG"]
    NewState --> UserUpdated(["User & Agent<br/>(Updated Knowledge)"])
    UserUpdated -- Refines Future Perception --> Goal
    UserUpdated --> UserAgent
```

---

## ✨ Core Features & Scope Fulfillment

1. **Polymarket Discovery**: Hooks into the Gamma API (`core/polymarket.py`) to scrape high-liquidity active target events matching the requested intent.
2. **Kalshi Integrations**: Hooks into the Kalshi v2 REST API (`core/kalshi.py`) to discover volume spikes and extract order footprints dynamically.
3. **Niche Routing Engine**: Classifies raw events seamlessly into strict categories (NBA, Politics, Weather, Crypto) (`core/niche.py`).
4. **Apify Web Sentiment Plugin**: Bootstraps the `apify-client` and triggers the live `google-search-scraper` actor to grab breaking news items relative directly to the trader's prediction event (`core/enrichment.py`).
5. **RAG Chat Layer**: An onboard intelligence retriever pushing context into the Hermes context-window to ground the LLM's conversation over copy-trading safety (`rag/rag_agent.py`).
6. **Closed Learning Loop**: By wrapping the pipeline natively in `hermes_agent.run_agent()` and passing explicitly generated context back to the user, we retain conversation memory across sequential searches.

---

## 🛠️ Project Structure

```text
├── src/quantara/                # Core application package
│   ├── agent.py                 # Hermes framework configuration & entry wrapper
│   ├── main.py                  # Terminal CLI Entrypoint
│   ├── core/                    # Pure Python domain logic & API connectors
│   │   ├── polymarket.py        # Polymarket Gamma API connector
│   │   ├── kalshi.py            # Kalshi API connector
│   │   ├── enrichment.py        # Apify news scraper definition
│   │   ├── niche.py             # Event classifier
│   │   ├── analysis.py          # Return/Risk calculator
│   │   └── planner.py           # NLP scope parsing
│   ├── tools/                   # Hermes-Agent Tool Registry Plugins
│   │   └── *_tool.py            # 7 distinct @tool adapters exposing Core logic
│   └── rag/                     # Embedded Data retrieval mechanics
│       ├── rag_agent.py         # Main querying agent
│       ├── retriever.py         
│       └── vector_store.py      # TF-IDF persistent data engine
├── examples.md                  # Workflow Input/Output demonstration transcripts
├── requirements.txt             # Dependency constraints
├── setup.py                     # Module packaging logic
└── .env                         # Secrets configuration
```

---

## 🚀 Installation & Setup

1. **Clone the Repository**
```bash
git clone <your-intern-repo-link-here>
cd Quantara
```

2. **Configure the Environment**
Create a `.env` file at the root containing your authorized credentials:
```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
APIFY_API_TOKEN=your_apify_api_token_here
```

3. **Virtual Environment & Dependencies**
This project acts as an installable package (`-e .`) to prevent cross-path import crashes.
```bash
python -m venv venv
source venv/Scripts/activate     # Windows: .\venv\Scripts\activate
pip install -r requirements.txt  # Installs apify, kalshi-sdk, clob, hermes
```

---

## 🖥️ Execution

Our entrypoint simplifies all underlying Hermes logic:

```bash
python main.py
```

### Example Input
```text
Enter query: Find the most consistent copy traders to shadow for the upcoming US Election on Polymarket.
```

### 📈 Detailed Transcripts
For comprehensive, end-to-end outputs detailing exactly how the Agent plans, invokes Apify, grabs wallets, and provides conversation insights, please view the [`examples.md`](./examples.md) file!

