"""
Main entry point for the LangGraph Quantara trading system.

This module provides the primary interface for running trading analysis
using the LangGraph-based multi-agent system.
"""

import asyncio
import sys
from typing import Dict, Any, Optional
from .core.graph import TradingGraph
from .config import get_config, create_user_preferences, UserPreferences


async def run_langgraph_analysis(
    query: str,
    user_preferences: Optional[Dict[str, Any]] = None,
    stream_results: bool = False
) -> Dict[str, Any]:
    """
    Run trading analysis using the LangGraph system.
    
    Args:
        query: User's trading query
        user_preferences: Optional user preferences dictionary
        stream_results: Whether to stream results in real-time
        
    Returns:
        Analysis results with recommendations
    """
    # Get configuration
    config = get_config()
    
    # Create user preferences
    if user_preferences is None:
        user_preferences = {}
    
    preferences = create_user_preferences(**user_preferences)
    
    # Initialize the trading graph
    trading_graph = TradingGraph(config.to_dict())
    
    try:
        if stream_results:
            # Stream results in real-time
            print("🚀 Starting LangGraph trading analysis (streaming)...")
            print(f"📊 Query: {query}")
            print("=" * 60)
            
            results = []
            async for update in trading_graph.stream_analysis(query, preferences.__dict__):
                step = update.get("step", "")
                node = update.get("node", "")
                progress = update.get("progress", 0.0)
                data = update.get("data", {})
                
                print(f"🔄 [{progress:.1%}] {node}: {step}")
                if data.get("market_count", 0) > 0:
                    print(f"   📈 Markets found: {data['market_count']}")
                if data.get("trader_count", 0) > 0:
                    print(f"   👥 Traders analyzed: {data['trader_count']}")
                if data.get("confidence", 0) > 0:
                    print(f"   🎯 Confidence: {data['confidence']:.1%}")
                
                results.append(update)
            
            # Get final results
            final_result = await trading_graph.analyze_trading_opportunity(query, preferences.__dict__)
            
        else:
            # Standard analysis
            print("🚀 Starting LangGraph trading analysis...")
            print(f"📊 Query: {query}")
            print("=" * 60)
            
            final_result = await trading_graph.analyze_trading_opportunity(query, preferences.__dict__)
        
        return final_result
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return {
            "error": str(e),
            "recommendations": [],
            "confidence_score": 0.0,
            "explanation": f"Analysis failed: {str(e)}"
        }


def format_results(results: Dict[str, Any]) -> str:
    """
    Format analysis results for display.
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        Formatted results string
    """
    if "error" in results:
        return f"❌ Analysis Error: {results['error']}"
    
    output = []
    
    # Header
    output.append("🎯 LANGGRAPH TRADING ANALYSIS RESULTS")
    output.append("=" * 50)
    
    # Overall metrics
    confidence = results.get("confidence_score", 0.0)
    output.append(f"📊 Overall Confidence: {confidence:.1%}")
    
    # Market data summary
    market_data = results.get("market_data", {})
    polymarket_count = len(market_data.get("polymarket", []))
    kalshi_count = len(market_data.get("kalshi", []))
    output.append(f"📈 Markets Analyzed: {polymarket_count + kalshi_count} (Polymarket: {polymarket_count}, Kalshi: {kalshi_count})")
    
    # Recommendations
    recommendations = results.get("recommendations", [])
    output.append(f"\n🏆 TOP TRADER RECOMMENDATIONS ({len(recommendations)} found):")
    output.append("-" * 40)
    
    for i, rec in enumerate(recommendations[:5], 1):  # Top 5
        trader_id = rec.get("trader_id", "Unknown")
        platform = rec.get("platform", "Unknown")
        score = rec.get("score", 0.0)
        roi = rec.get("roi", 0.0)
        risk_level = rec.get("risk_level", 0.0)
        
        output.append(f"{i}. {trader_id} ({platform})")
        output.append(f"   Score: {score:.1%} | ROI: {roi:+.1%} | Risk: {risk_level:.1%}")
    
    # Analysis breakdown
    analysis = results.get("analysis", {})
    if analysis:
        output.append(f"\n📋 ANALYSIS BREAKDOWN:")
        output.append("-" * 30)
        
        sentiment = analysis.get("sentiment", {})
        if sentiment:
            overall_sentiment = sentiment.get("overall_sentiment", "neutral")
            output.append(f"😊 Market Sentiment: {overall_sentiment.title()}")
        
        risk = analysis.get("risk", {})
        if risk:
            portfolio_risk = risk.get("portfolio_risk", {})
            if portfolio_risk:
                overall_risk = portfolio_risk.get("overall_risk_score", 0.0)
                output.append(f"⚠️  Portfolio Risk: {overall_risk:.1%}")
        
        trader_scores = analysis.get("trader_scores", [])
        if trader_scores:
            avg_score = sum(t.get("score", 0) for t in trader_scores) / len(trader_scores)
            output.append(f"📊 Average Trader Score: {avg_score:.1%}")
    
    # Explanation
    explanation = results.get("explanation", "")
    if explanation:
        output.append(f"\n💡 EXPLANATION:")
        output.append("-" * 20)
        output.append(explanation)
    
    return "\n".join(output)


async def main():
    """Main entry point for command-line usage."""
    print("🤖 LangGraph Quantara Trading System")
    print("=" * 40)
    
    # Get user input
    query = input("Enter your trading query: ").strip()
    
    if not query:
        print("❌ Please provide a valid query.")
        return
    
    # Ask for streaming preference
    stream_choice = input("Stream results in real-time? (y/n): ").strip().lower()
    stream_results = stream_choice in ['y', 'yes', '1', 'true']
    
    # Run analysis
    try:
        results = await run_langgraph_analysis(query, stream_results=stream_results)
        
        print("\n" + "=" * 60)
        print(format_results(results))
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    # Handle Windows UTF-8 encoding
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    
    # Run the async main function
    asyncio.run(main())