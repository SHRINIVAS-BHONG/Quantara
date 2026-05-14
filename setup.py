from setuptools import setup, find_packages

setup(
    name="quantara",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Core LangGraph Dependencies
        "langgraph>=0.2.0",
        "langchain>=0.3.0", 
        "langchain-openai>=0.2.0",
        "langchain-core>=0.3.0",
        
        # Market Data Dependencies
        "apify-client",
        "py_clob_client",
        "kalshi-python",
        "requests",
        "python-dotenv",
        
        # Infrastructure Dependencies
        "aiohttp>=3.9.0",
        "redis>=5.0.0",
        "sqlalchemy>=2.0.0",
        "asyncpg>=0.29.0",
        
        # Monitoring and Observability
        "prometheus-client>=0.20.0",
        "structlog>=24.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "hypothesis>=6.98.0",
            "black>=24.0.0",
            "mypy>=1.8.0",
        ]
    },
    python_requires=">=3.8",
    description="LangGraph-powered prediction market trading agent",
    long_description="A sophisticated multi-agent system for analyzing prediction markets and identifying profitable trading opportunities using LangGraph orchestration.",
)