# Quantara: LangGraph-Powered Prediction Market Trading Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Orchestration: LangGraph](https://img.shields.io/badge/orchestration-langgraph-orange.svg)](https://github.com/langchain-ai/langgraph)

A sophisticated, production-grade multi-agent prediction market trading agent built on **LangGraph**. By organizing specialized agents into an acyclic stateful graph, Quantara coordinates complex, concurrent prediction analysis workflows—covering Market Discovery (Polymarket & Kalshi), Sentiment Analysis, Risk Assessment, and Trader Performance scoring. It integrates a custom TF-IDF local RAG database to serve as a self-improving vector memory (Closed Learning Loop) that learns from previous runs.

## 🎯 Features

- **Multi-Agent Orchestration**: Specialized agents for market discovery, sentiment analysis, risk assessment, and trader analysis
- **Real-Time Streaming**: Live updates during analysis with concurrent session support
- **Automatic Checkpointing**: State persistence and recovery from failures
- **Market Integration**: Support for Polymarket and Kalshi prediction markets
- **Sentiment Analysis**: Real-time news enrichment and sentiment correlation
- **Risk Management**: Portfolio optimization and position sizing recommendations
- **Continuous Monitoring**: Real-time market tracking with adaptive alerts
- **Production-Ready**: 95%+ test coverage with comprehensive validation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/quantara.git
cd quantara

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY
# Optional: POLYMARKET_API_KEY, KALSHI_API_KEY, APIFY_API_KEY
```

### First Run

```python
import asyncio
from src.quantara.langgraph.main import TradingGraph
from src.quantara.langgraph.config import UserPreferences

async def main():
    graph = TradingGraph()
    preferences = UserPreferences(
        risk_tolerance=0.7,
        preferred_platforms=["polymarket"],
        enable_monitoring=False
    )
    
    recommendations = await graph.analyze_trading_opportunity(
        query="Find the best crypto traders",
        preferences=preferences
    )
    
    for trader in recommendations.traders[:5]:
        print(f"{trader.trader_id}: {trader.score:.1%}")

asyncio.run(main())
```

## 🏗️ Architecture

```text
                         +------------------------+
                         |       User Query       |
                         +------------------------+
                                     |
                                     v
                         +------------------------+
                         |   Query Router Node    |
                         +------------------------+
                                     |
                                     v
                  . - - - - - - - - - - - - - - - - - .
                 '        RAG Memory Check             '
                 ' (Searches local TF-IDF Vector DB)  ' - - - - - -> [ Local Vector Store DB ]
                  ' - - - - - - - - - - - - - - - - - '              [   (vector_store.json)   ]
                                     |                                          ^
                                     v                                          |
                         +------------------------+                             |
                         | Market Discovery Node  |                             |
                         | (Polymarket & Kalshi)  |                             |
                         +------------------------+                             |
                                     |                                          |
                                     +-------------------+                      |
                                     |                   |                      |
                                     v                   v                      |
                         +-----------------------+ +-----------------------+    |
                         |  Sentiment Analysis   | |    Risk Assessment    |    |
                         |         Agent         | |         Agent         |    |
                         +-----------------------+ +-----------------------+    |
                                     |                   |                      |
                                     +---------+---------+                      |
                                               |                                |
                                               v                                |
                         +------------------------+                             |
                         |  Trader Analysis Node  |                             |
                         +------------------------+                             |
                                     |                                          |
                                     v                                          |
                  . - - - - - - - - - - - - - - - - - .                         |
                 '        Recall RAG Context           '                        |
                 '   (Applies +15% Score Boost)       '                         |
                  ' - - - - - - - - - - - - - - - - - '                         |
                                     |                                          |
                                     v                                          |
                  . - - - - - - - - - - - - - - - - - .                         |
                 '      Closed Learning Ingestion      '                        |
                 '   (Saves top performers to RAG)     ' - - - - - - - - - - - -+
                  ' - - - - - - - - - - - - - - - - - '
                                     |
                                     v
                         +------------------------+
                         | Report Generator Node  |
                         +------------------------+
                                     |
                                     v
                         +------------------------+
                         | Real-time Stream Out   |
                         +------------------------+
```

### Core Components

#### TradingState
Central state object maintaining:
- Query context and parsed intent
- Market data from multiple sources
- Analysis results (sentiment, risk, trader scores)
- Workflow progress and checkpoints

#### Specialized Agents
- **Market Discovery**: Identifies and classifies prediction markets
- **Sentiment Analysis**: Analyzes market sentiment from news and social media
- **Risk Assessment**: Evaluates trader risk profiles and portfolio risk
- **Trader Analysis**: Scores and ranks traders based on performance

#### Advanced Features
- **Streaming Output**: Real-time event streaming with concurrent sessions
- **Checkpoint Manager**: Automatic state persistence and recovery
- **Error Handling**: Exponential backoff and graceful degradation
- **Data Validation**: Comprehensive validation at every step

## 📊 Performance

- **Interactive Queries**: < 5 seconds
- **Complex Analysis**: < 30 seconds
- **Real-time Alerts**: < 2 seconds
- **Concurrent Sessions**: 100+ supported
- **Test Coverage**: 95%+ of critical components

## 🔐 Security

- AES-256 encryption for sensitive data
- Secure API key storage and rotation
- Input validation and sanitization
- TLS 1.3 for all external communications
- Comprehensive audit logging

## 📁 Project Structure

```
quantara/
├── src/quantara/
│   ├── core/                    # Core trading logic
│   ├── langgraph/               # LangGraph orchestration
│   │   ├── agents/              # Specialized agents
│   │   ├── core/                # Core components
│   │   └── main.py              # Entry point
│   ├── learning/                # Learning & feedback
│   ├── memory/                  # Memory management
│   ├── rag/                     # RAG system
│   └── tools/                   # Tool implementations
├── README.md                    # This file
├── QUICKSTART.md                # Quick start guide
├── CONTRIBUTING.md              # Contributing guidelines
├── CHANGELOG.md                 # Version history
├── .env.example                 # Configuration template
├── requirements.txt             # Dependencies
└── setup.py                     # Package setup
```

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/quantara

# Run specific test
pytest tests/test_state_consistency.py -v
```

**Test Coverage**:
- Property-based tests: 17 comprehensive validations
- Integration tests: 25+ end-to-end workflows
- Unit tests: Comprehensive component testing
- Pass rate: 100%

## 🔄 Workflow Examples

### Find Best Traders in Niche

```python
query = "Find the best crypto prediction market traders with high ROI"
preferences = UserPreferences(
    risk_tolerance=0.8,
    preferred_platforms=["polymarket"],
    min_trader_roi=0.2
)

recommendations = await graph.analyze_trading_opportunity(
    query=query,
    preferences=preferences
)
```

### Analyze Specific Market

```python
query = "Analyze the 2024 US Presidential Election market"
preferences = UserPreferences(
    risk_tolerance=0.5,
    preferred_platforms=["polymarket", "kalshi"]
)

recommendations = await graph.analyze_trading_opportunity(
    query=query,
    preferences=preferences
)
```

### Stream Results in Real-Time

```python
session_id = await graph.create_stream_session(
    query="Find best traders",
    preferences=preferences
)

async for event in graph.stream_events(session_id):
    print(f"[{event.event_type}] {event.data}")
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and features

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🙏 Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration
- [Polymarket](https://polymarket.com) - Market data
- [Kalshi](https://kalshi.com) - Market data
- [Apify](https://apify.com) - News enrichment

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/SHRINIVAS-BHONG/quantara/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SHRINIVAS-BHONG/quantara/discussions)
- **Documentation**: See [QUICKSTART.md](QUICKSTART.md) and [CONTRIBUTING.md](CONTRIBUTING.md)

## 🗺️ Roadmap

- [ ] Web UI dashboard for real-time monitoring
- [ ] Mobile app for alerts and recommendations
- [ ] Advanced ML models for trader scoring
- [ ] Integration with additional prediction markets
- [ ] Backtesting framework for strategy validation
- [ ] Paper trading simulation
- [ ] Live trading integration (with caution)

---

**Made with ❤️ for prediction market traders**
