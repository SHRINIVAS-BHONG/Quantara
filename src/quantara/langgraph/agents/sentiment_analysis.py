"""
Sentiment Analysis Agent for the LangGraph trading system.

This agent specializes in market sentiment analysis, news enrichment,
and correlation of sentiment data with trading performance.

Requirements implemented:
- 2.2: Sentiment_Analysis_Agent for news and sentiment enrichment
- 6.1: Integrate with Apify for real-time news scraping
- 6.2: Analyze sentiment from multiple data sources including social media
- 6.3: Detect significant sentiment shifts and trend changes
- 6.4: Correlate sentiment data with trader performance metrics
- 6.5: Update TradingState with enriched data
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from ..core.state import TradingState

# Apify client for news scraping
try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False
    logging.warning("Apify client not available. Install with: pip install apify-client")


class SentimentAnalysisAgent:
    """
    Agent specialized in market sentiment analysis and news enrichment.
    
    This agent performs comprehensive sentiment analysis, integrates news
    and social media data, detects sentiment shifts, and correlates
    sentiment with trading performance.
    
    Requirements implemented:
    - 6.1: Apify integration for real-time news scraping
    - 6.2: Multi-source sentiment analysis (news, social media, market data)
    - 6.3: Sentiment shift detection with trend analysis
    - 6.4: Sentiment-performance correlation algorithms
    - 6.5: TradingState enrichment with sentiment data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Sentiment Analysis Agent.
        
        Args:
            config: Optional configuration for agent behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Sentiment configuration
        self.sentiment_sources = ["news", "social_media", "market_data"]
        self.sentiment_categories = ["bullish", "bearish", "neutral", "volatile"]
        
        # Apify configuration
        self.apify_api_key = os.getenv("APIFY_API_KEY", self.config.get("apify_api_key", ""))
        self.apify_client = None
        if APIFY_AVAILABLE and self.apify_api_key:
            try:
                self.apify_client = ApifyClient(self.apify_api_key)
                self.logger.info("Apify client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Apify client: {e}")
        else:
            self.logger.warning("Apify client not initialized - API key missing or client unavailable")
        
        # Sentiment analysis thresholds
        self.sentiment_shift_threshold = self.config.get("sentiment_shift_threshold", 0.3)
        self.high_correlation_threshold = self.config.get("high_correlation_threshold", 0.7)
        self.sentiment_confidence_threshold = self.config.get("sentiment_confidence_threshold", 0.6)
        
        # News scraping configuration
        self.max_news_articles = self.config.get("max_news_articles", 20)
        self.news_relevance_threshold = self.config.get("news_relevance_threshold", 0.5)
        self.news_age_days = self.config.get("news_age_days", 7)
        
        # Sentiment keywords for enhanced analysis
        self.sentiment_keywords = {
            "bullish": {
                "strong": ["surge", "soar", "boom", "rally", "breakthrough", "skyrocket"],
                "moderate": ["rise", "gain", "increase", "growth", "positive", "optimistic"],
                "weak": ["slight", "modest", "gradual", "steady"]
            },
            "bearish": {
                "strong": ["crash", "plunge", "collapse", "plummet", "disaster", "crisis"],
                "moderate": ["fall", "decline", "drop", "loss", "negative", "pessimistic"],
                "weak": ["dip", "slip", "ease", "soften"]
            },
            "volatile": {
                "strong": ["volatile", "turbulent", "chaotic", "unstable", "erratic"],
                "moderate": ["fluctuate", "swing", "vary", "uncertain", "mixed"],
                "weak": ["change", "shift", "adjust"]
            }
        }
        
        # Social media sentiment indicators
        self.social_indicators = {
            "engagement_metrics": ["likes", "shares", "comments", "retweets"],
            "sentiment_signals": ["trending", "viral", "buzz", "hype"],
            "risk_signals": ["fud", "panic", "fear", "doubt", "scam"]
        }
    
    async def analyze_sentiment(self, state: TradingState) -> TradingState:
        """
        Analyze market sentiment from multiple sources with enhanced algorithms.
        
        Implements Requirements 6.2 and 6.5:
        - Analyze sentiment from multiple data sources including social media
        - Update TradingState with enriched data
        
        Args:
            state: Current TradingState with market data
            
        Returns:
            Updated TradingState with comprehensive sentiment analysis
        """
        self.logger.info("Starting comprehensive multi-source sentiment analysis")
        updated_state = state.copy()
        
        # Get market data for sentiment analysis
        all_markets = state.get("polymarket_data", []) + state.get("kalshi_data", [])
        niche = state.get("niche", "")
        
        if not all_markets:
            self.logger.warning("No markets available for sentiment analysis")
            updated_state["sentiment_analysis"] = {
                "error": "No markets available",
                "overall_sentiment": "neutral",
                "confidence_level": 0.0
            }
            return updated_state
        
        # Perform sentiment analysis for each market with enhanced scoring
        market_sentiments = []
        sentiment_metrics = {
            "total_markets_analyzed": len(all_markets),
            "analysis_timestamp": datetime.now().isoformat(),
            "sources_used": [],
            "processing_time_ms": 0
        }
        
        start_time = datetime.now()
        
        for market in all_markets:
            try:
                sentiment_data = await self._analyze_market_sentiment_enhanced(market, niche)
                market_sentiments.append(sentiment_data)
            except Exception as e:
                self.logger.error(f"Failed to analyze sentiment for market {market.get('event_id', 'unknown')}: {e}")
                # Continue with other markets
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        sentiment_metrics["processing_time_ms"] = processing_time
        
        # Calculate comprehensive overall sentiment metrics
        overall_sentiment = self._calculate_overall_sentiment_enhanced(market_sentiments)
        sentiment_distribution = self._calculate_sentiment_distribution(market_sentiments)
        confidence_level = self._calculate_sentiment_confidence_enhanced(market_sentiments)
        
        # Identify sentiment trends and patterns
        sentiment_trends = self._identify_sentiment_trends(market_sentiments)
        
        # Calculate sentiment strength and momentum
        sentiment_strength = self._calculate_sentiment_strength(market_sentiments)
        sentiment_momentum = self._calculate_sentiment_momentum(market_sentiments)
        
        # Store comprehensive sentiment analysis results
        sentiment_analysis = {
            "overall_sentiment": overall_sentiment,
            "sentiment_category": self._categorize_sentiment(overall_sentiment),
            "market_sentiments": market_sentiments,
            "sentiment_distribution": sentiment_distribution,
            "confidence_level": confidence_level,
            "sentiment_trends": sentiment_trends,
            "sentiment_strength": sentiment_strength,
            "sentiment_momentum": sentiment_momentum,
            "analysis_timestamp": datetime.now().isoformat(),
            "metrics": sentiment_metrics,
            "sources_analyzed": self.sentiment_sources
        }
        
        updated_state["sentiment_analysis"] = sentiment_analysis
        
        self.logger.info(
            f"Sentiment analysis completed: {overall_sentiment} sentiment "
            f"({confidence_level:.2%} confidence) across {len(market_sentiments)} markets"
        )
        
        return updated_state
    
    async def enrich_with_news(self, state: TradingState) -> TradingState:
        """
        Enrich analysis with real-time news data using Apify integration.
        
        Implements Requirements 6.1 and 6.5:
        - Integrate with Apify for real-time news scraping
        - Update TradingState with enriched data
        
        Args:
            state: Current TradingState with sentiment analysis
            
        Returns:
            Updated TradingState with comprehensive news enrichment
        """
        self.logger.info("Starting news enrichment with Apify integration")
        updated_state = state.copy()
        
        niche = state.get("niche", "")
        markets = state.get("polymarket_data", []) + state.get("kalshi_data", [])
        
        if not markets:
            self.logger.warning("No markets available for news enrichment")
            return updated_state
        
        # Extract search queries from markets and niche
        search_queries = self._generate_news_search_queries(niche, markets)
        
        # Fetch news data using Apify (with fallback to mock data)
        news_data = await self._fetch_news_data_apify(search_queries, niche, markets)
        
        # Analyze news sentiment with enhanced NLP
        news_sentiment = await self._analyze_news_sentiment_enhanced(news_data)
        
        # Extract key topics and entities
        key_topics = self._extract_key_topics_enhanced(news_data)
        key_entities = self._extract_key_entities(news_data)
        
        # Detect news-driven sentiment shifts
        sentiment_analysis = updated_state.get("sentiment_analysis", {})
        market_sentiment = sentiment_analysis.get("overall_sentiment", "neutral")
        sentiment_shift = self._detect_news_sentiment_shift_enhanced(
            news_sentiment, market_sentiment, sentiment_analysis
        )
        
        # Calculate news impact score
        news_impact = self._calculate_news_impact(news_data, markets)
        
        # Identify trending narratives
        trending_narratives = self._identify_trending_narratives(news_data)
        
        # Update sentiment analysis with comprehensive news enrichment
        sentiment_analysis["news_enrichment"] = {
            "news_articles_count": len(news_data),
            "news_sentiment": news_sentiment,
            "key_topics": key_topics,
            "key_entities": key_entities,
            "sentiment_shift": sentiment_shift,
            "news_impact_score": news_impact,
            "trending_narratives": trending_narratives,
            "search_queries_used": search_queries,
            "data_source": "apify" if self.apify_client else "mock",
            "enrichment_timestamp": datetime.now().isoformat()
        }
        
        updated_state["sentiment_analysis"] = sentiment_analysis
        
        self.logger.info(
            f"News enrichment completed: {len(news_data)} articles analyzed, "
            f"impact score: {news_impact:.2f}, shift detected: {sentiment_shift['shift_detected']}"
        )
        
        return updated_state
    
    async def detect_sentiment_shifts(self, state: TradingState) -> TradingState:
        """
        Detect significant sentiment changes and trends with advanced algorithms.
        
        Implements Requirements 6.3 and 6.5:
        - Detect significant sentiment shifts and trend changes
        - Update TradingState with enriched data
        
        Args:
            state: Current TradingState with sentiment data
            
        Returns:
            Updated TradingState with comprehensive sentiment shift analysis
        """
        self.logger.info("Starting advanced sentiment shift detection")
        updated_state = state.copy()
        
        sentiment_analysis = state.get("sentiment_analysis", {})
        
        if not sentiment_analysis:
            self.logger.warning("No sentiment analysis available for shift detection")
            return updated_state
        
        # Get current sentiment metrics
        current_sentiment = sentiment_analysis.get("overall_sentiment", "neutral")
        market_sentiments = sentiment_analysis.get("market_sentiments", [])
        sentiment_distribution = sentiment_analysis.get("sentiment_distribution", {})
        
        # Calculate sentiment shift metrics
        shift_metrics = {
            "detection_timestamp": datetime.now().isoformat(),
            "analysis_method": "multi_factor_shift_detection"
        }
        
        # Detect temporal sentiment shifts (comparing with historical baseline)
        temporal_shift = self._detect_temporal_shift(sentiment_analysis)
        
        # Detect cross-market sentiment divergence
        market_divergence = self._detect_market_divergence(market_sentiments)
        
        # Calculate sentiment volatility index
        volatility_index = self._calculate_sentiment_volatility_enhanced(sentiment_analysis)
        
        # Detect sentiment momentum changes
        momentum_change = self._detect_momentum_change(sentiment_analysis)
        
        # Identify sentiment inflection points
        inflection_points = self._identify_inflection_points(market_sentiments)
        
        # Calculate overall shift magnitude and direction
        shift_magnitude = self._calculate_shift_magnitude(
            temporal_shift, market_divergence, volatility_index, momentum_change
        )
        
        shift_direction = self._determine_shift_direction(
            current_sentiment, temporal_shift, momentum_change
        )
        
        # Determine shift classification
        shift_classification = self._classify_sentiment_shift(shift_magnitude, volatility_index)
        
        # Calculate trend strength and persistence
        trend_strength = self._calculate_trend_strength(market_sentiments, sentiment_distribution)
        trend_persistence = self._calculate_trend_persistence(sentiment_analysis)
        
        # Comprehensive sentiment shift analysis
        sentiment_shifts = {
            "shift_detected": shift_magnitude > self.sentiment_shift_threshold,
            "shift_magnitude": shift_magnitude,
            "shift_direction": shift_direction,
            "shift_classification": shift_classification,
            "temporal_shift": temporal_shift,
            "market_divergence": market_divergence,
            "volatility_index": volatility_index,
            "momentum_change": momentum_change,
            "inflection_points": inflection_points,
            "trend_strength": trend_strength,
            "trend_persistence": trend_persistence,
            "shift_confidence": self._calculate_shift_confidence(
                shift_magnitude, volatility_index, trend_strength
            ),
            "risk_level": self._assess_shift_risk_level(shift_magnitude, volatility_index),
            "metrics": shift_metrics
        }
        
        # Update sentiment analysis with shift detection results
        sentiment_analysis["sentiment_shifts"] = sentiment_shifts
        updated_state["sentiment_analysis"] = sentiment_analysis
        
        self.logger.info(
            f"Sentiment shift detection completed: "
            f"{'Shift detected' if sentiment_shifts['shift_detected'] else 'No significant shift'} "
            f"(magnitude: {shift_magnitude:.2f}, direction: {shift_direction})"
        )
        
        return updated_state
    
    async def correlate_sentiment_performance(self, state: TradingState) -> TradingState:
        """
        Correlate sentiment data with trader performance using advanced statistical methods.
        
        Implements Requirements 6.4 and 6.5:
        - Correlate sentiment data with trader performance metrics
        - Update TradingState with enriched data
        
        Args:
            state: Current TradingState with sentiment and trader data
            
        Returns:
            Updated TradingState with comprehensive sentiment-performance correlation
        """
        self.logger.info("Starting advanced sentiment-performance correlation analysis")
        updated_state = state.copy()
        
        sentiment_analysis = state.get("sentiment_analysis", {})
        trader_scores = state.get("trader_scores", [])
        
        if not sentiment_analysis or not trader_scores:
            self.logger.warning("Insufficient data for sentiment-performance correlation")
            return updated_state
        
        # Calculate comprehensive correlations for each trader
        correlations = []
        correlation_metrics = {
            "total_traders_analyzed": len(trader_scores),
            "correlation_method": "multi_factor_statistical",
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        for trader_data in trader_scores:
            trader = trader_data.get("trader", {})
            trader_id = trader.get("trader_id", "unknown")
            
            try:
                # Calculate multi-dimensional correlation
                correlation_result = self._calculate_comprehensive_correlation(
                    sentiment_analysis, trader, trader_data
                )
                
                correlations.append({
                    "trader_id": trader_id,
                    "platform": trader.get("platform", "unknown"),
                    "sentiment_correlation": correlation_result["overall_correlation"],
                    "correlation_breakdown": correlation_result["breakdown"],
                    "sentiment_alignment": correlation_result["alignment"],
                    "performance_sensitivity": correlation_result["sensitivity"],
                    "risk_sentiment_correlation": correlation_result["risk_correlation"],
                    "confidence_level": correlation_result["confidence"]
                })
                
            except Exception as e:
                self.logger.error(f"Failed to correlate sentiment for trader {trader_id}: {e}")
                # Add default correlation for failed analysis
                correlations.append({
                    "trader_id": trader_id,
                    "platform": trader.get("platform", "unknown"),
                    "sentiment_correlation": 0.5,
                    "sentiment_alignment": "unknown",
                    "error": str(e)
                })
        
        # Calculate aggregate correlation statistics
        valid_correlations = [c["sentiment_correlation"] for c in correlations 
                             if "error" not in c]
        
        if valid_correlations:
            avg_correlation = statistics.mean(valid_correlations)
            median_correlation = statistics.median(valid_correlations)
            correlation_std_dev = statistics.stdev(valid_correlations) if len(valid_correlations) > 1 else 0.0
        else:
            avg_correlation = 0.5
            median_correlation = 0.5
            correlation_std_dev = 0.0
        
        # Identify high-correlation traders
        high_correlation_traders = [
            c for c in correlations 
            if c.get("sentiment_correlation", 0) > self.high_correlation_threshold
        ]
        
        # Identify sentiment-aligned trading opportunities
        aligned_opportunities = self._identify_sentiment_aligned_opportunities(
            correlations, sentiment_analysis, trader_scores
        )
        
        # Calculate sentiment-performance predictive power
        predictive_power = self._calculate_predictive_power(correlations, sentiment_analysis)
        
        # Comprehensive performance correlation analysis
        performance_correlations = {
            "trader_correlations": correlations,
            "aggregate_statistics": {
                "avg_correlation": avg_correlation,
                "median_correlation": median_correlation,
                "correlation_std_dev": correlation_std_dev,
                "high_correlation_count": len(high_correlation_traders),
                "correlation_range": {
                    "min": min(valid_correlations) if valid_correlations else 0.0,
                    "max": max(valid_correlations) if valid_correlations else 0.0
                }
            },
            "high_correlation_traders": high_correlation_traders,
            "sentiment_aligned_opportunities": aligned_opportunities,
            "predictive_power": predictive_power,
            "correlation_confidence": self._calculate_correlation_confidence(
                valid_correlations, len(trader_scores)
            ),
            "metrics": correlation_metrics
        }
        
        # Update sentiment analysis with correlation results
        sentiment_analysis["performance_correlations"] = performance_correlations
        updated_state["sentiment_analysis"] = sentiment_analysis
        
        self.logger.info(
            f"Sentiment-performance correlation completed: "
            f"{len(correlations)} traders analyzed, "
            f"avg correlation: {avg_correlation:.2f}, "
            f"{len(high_correlation_traders)} high-correlation traders identified"
        )
        
        return updated_state
    
    
    # ==================== Enhanced Helper Methods ====================
    
    async def _analyze_market_sentiment_enhanced(self, market: Dict[str, Any], niche: str) -> Dict[str, Any]:
        """
        Analyze sentiment for a specific market with enhanced NLP and multi-factor scoring.
        
        Args:
            market: Market data dictionary
            niche: Market niche category
            
        Returns:
            Comprehensive sentiment analysis for the market
        """
        await asyncio.sleep(0.02)  # Simulate processing time
        
        # Extract market characteristics
        title = market.get("title", "").lower()
        description = market.get("description", "").lower()
        volume = market.get("total_volume", 0)
        liquidity = market.get("liquidity", 0)
        odds = market.get("current_odds", {})
        category = market.get("category", "").lower()
        
        # Multi-factor sentiment scoring
        sentiment_factors = {
            "keyword_sentiment": 0.0,
            "volume_sentiment": 0.0,
            "odds_sentiment": 0.0,
            "market_activity_sentiment": 0.0
        }
        
        # Factor 1: Enhanced keyword-based sentiment analysis
        keyword_score = 0.0
        keyword_matches = {"bullish": 0, "bearish": 0, "volatile": 0}
        
        text_content = f"{title} {description} {category}"
        
        for sentiment_type, keyword_groups in self.sentiment_keywords.items():
            for strength, keywords in keyword_groups.items():
                weight = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}.get(strength, 0.5)
                matches = sum(1 for keyword in keywords if keyword in text_content)
                
                if sentiment_type == "bullish":
                    keyword_score += matches * weight * 0.3
                    keyword_matches["bullish"] += matches
                elif sentiment_type == "bearish":
                    keyword_score -= matches * weight * 0.3
                    keyword_matches["bearish"] += matches
                elif sentiment_type == "volatile":
                    keyword_matches["volatile"] += matches
        
        sentiment_factors["keyword_sentiment"] = max(-1.0, min(1.0, keyword_score))
        
        # Factor 2: Volume-based sentiment (higher volume = stronger sentiment)
        if volume > 0:
            volume_score = min(0.3, (volume / 50000.0) * 0.3)
            sentiment_factors["volume_sentiment"] = volume_score
        
        # Factor 3: Odds-based sentiment analysis
        if odds:
            max_odds = max(odds.values())
            min_odds = min(odds.values())
            odds_spread = max_odds - min_odds
            
            # High confidence (extreme odds) indicates strong sentiment
            if max_odds > 0.8:
                sentiment_factors["odds_sentiment"] = 0.2
            elif max_odds < 0.6:
                sentiment_factors["odds_sentiment"] = -0.1
            
            # Wide spread indicates volatile sentiment
            if odds_spread > 0.5:
                keyword_matches["volatile"] += 1
        
        # Factor 4: Market activity sentiment
        if liquidity > 0:
            activity_score = min(0.2, (liquidity / 25000.0) * 0.2)
            sentiment_factors["market_activity_sentiment"] = activity_score
        
        # Calculate composite sentiment score
        sentiment_score = sum(sentiment_factors.values())
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Determine sentiment category with volatility consideration
        if keyword_matches["volatile"] > 2 or (odds and odds_spread > 0.6):
            sentiment_category = "volatile"
        elif sentiment_score > 0.3:
            sentiment_category = "bullish"
        elif sentiment_score < -0.3:
            sentiment_category = "bearish"
        else:
            sentiment_category = "neutral"
        
        # Calculate confidence based on data quality and consistency
        confidence = self._calculate_market_sentiment_confidence(
            sentiment_factors, keyword_matches, volume, liquidity
        )
        
        return {
            "market_id": market.get("event_id", "unknown"),
            "market_title": market.get("title", "Unknown"),
            "sentiment_score": sentiment_score,
            "sentiment_category": sentiment_category,
            "sentiment_factors": sentiment_factors,
            "keyword_matches": keyword_matches,
            "confidence": confidence,
            "volume_factor": volume,
            "liquidity_factor": liquidity,
            "odds_factor": max(odds.values()) if odds else 0.5,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_market_sentiment_confidence(self, sentiment_factors: Dict[str, float],
                                               keyword_matches: Dict[str, int],
                                               volume: float, liquidity: float) -> float:
        """Calculate confidence level for market sentiment analysis."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence with more data points
        active_factors = sum(1 for v in sentiment_factors.values() if abs(v) > 0.1)
        confidence += active_factors * 0.1
        
        # Increase confidence with keyword matches
        total_matches = sum(keyword_matches.values())
        confidence += min(0.2, total_matches * 0.05)
        
        # Increase confidence with volume and liquidity
        if volume > 1000:
            confidence += 0.1
        if liquidity > 500:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    
    def _calculate_overall_sentiment_enhanced(self, market_sentiments: List[Dict[str, Any]]) -> str:
        """Calculate enhanced overall sentiment from individual market sentiments."""
        if not market_sentiments:
            return "neutral"
        
        # Weighted average based on confidence and volume
        weighted_score = 0.0
        total_weight = 0.0
        
        for sentiment in market_sentiments:
            score = sentiment.get("sentiment_score", 0.0)
            confidence = sentiment.get("confidence", 0.5)
            volume = sentiment.get("volume_factor", 0)
            
            # Weight by confidence and volume (logarithmic scaling for volume)
            weight = confidence * (1.0 + min(1.0, volume / 10000.0))
            weighted_score += score * weight
            total_weight += weight
        
        avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Categorize with tighter thresholds for weighted average
        if avg_score > 0.25:
            return "bullish"
        elif avg_score < -0.25:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_sentiment_distribution(self, market_sentiments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of sentiment categories."""
        if not market_sentiments:
            return {"bullish": 0.0, "bearish": 0.0, "neutral": 1.0, "volatile": 0.0}
        
        categories = [s["sentiment_category"] for s in market_sentiments]
        total = len(categories)
        
        return {
            "bullish": categories.count("bullish") / total,
            "bearish": categories.count("bearish") / total,
            "neutral": categories.count("neutral") / total,
            "volatile": categories.count("volatile") / total
        }
    
    
    def _calculate_sentiment_confidence_enhanced(self, market_sentiments: List[Dict[str, Any]]) -> float:
        """Calculate enhanced overall confidence in sentiment analysis."""
        if not market_sentiments:
            return 0.0
        
        # Calculate weighted average confidence
        confidences = [s.get("confidence", 0.5) for s in market_sentiments]
        volumes = [s.get("volume_factor", 0) for s in market_sentiments]
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for conf, vol in zip(confidences, volumes):
            weight = 1.0 + min(1.0, vol / 10000.0)
            weighted_confidence += conf * weight
            total_weight += weight
        
        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        # Adjust confidence based on consistency
        if len(confidences) > 1:
            confidence_std = statistics.stdev(confidences)
            # Lower confidence if there's high variance
            consistency_factor = max(0.5, 1.0 - confidence_std)
            avg_confidence *= consistency_factor
        
        return min(1.0, avg_confidence)
    
    def _categorize_sentiment(self, sentiment_value: str) -> str:
        """Categorize sentiment value into standard categories."""
        # Already categorized in this case
        return sentiment_value
    
    def _identify_sentiment_trends(self, market_sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify sentiment trends and patterns across markets."""
        if not market_sentiments:
            return {"trend": "unknown", "strength": 0.0}
        
        # Analyze sentiment distribution
        categories = [s.get("sentiment_category", "neutral") for s in market_sentiments]
        category_counts = {cat: categories.count(cat) for cat in self.sentiment_categories}
        
        dominant_category = max(category_counts, key=category_counts.get)
        dominant_ratio = category_counts[dominant_category] / len(categories)
        
        # Determine trend strength
        trend_strength = dominant_ratio if dominant_ratio > 0.5 else 0.0
        
        return {
            "dominant_sentiment": dominant_category,
            "trend_strength": trend_strength,
            "category_distribution": category_counts,
            "consistency": dominant_ratio,
            "trend_direction": "strengthening" if trend_strength > 0.6 else "mixed"
        }
    
    def _calculate_sentiment_strength(self, market_sentiments: List[Dict[str, Any]]) -> float:
        """Calculate overall sentiment strength (magnitude regardless of direction)."""
        if not market_sentiments:
            return 0.0
        
        # Average absolute sentiment scores
        abs_scores = [abs(s.get("sentiment_score", 0.0)) for s in market_sentiments]
        return statistics.mean(abs_scores) if abs_scores else 0.0
    
    def _calculate_sentiment_momentum(self, market_sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sentiment momentum (rate of change)."""
        # In a real system, this would compare with historical data
        # For now, we estimate based on current data characteristics
        
        if not market_sentiments:
            return {"momentum": 0.0, "direction": "neutral"}
        
        scores = [s.get("sentiment_score", 0.0) for s in market_sentiments]
        avg_score = statistics.mean(scores)
        
        # Estimate momentum from score variance and strength
        if len(scores) > 1:
            score_variance = statistics.variance(scores)
            momentum = min(1.0, score_variance * 2.0)  # Higher variance = higher momentum
        else:
            momentum = 0.0
        
        direction = "bullish" if avg_score > 0.1 else "bearish" if avg_score < -0.1 else "neutral"
        
        return {
            "momentum": momentum,
            "direction": direction,
            "velocity": abs(avg_score),
            "acceleration": "increasing" if momentum > 0.5 else "stable"
        }
    
    
    def _generate_news_search_queries(self, niche: str, markets: List[Dict[str, Any]]) -> List[str]:
        """Generate optimized search queries for news scraping."""
        queries = []
        
        # Add niche-based query
        if niche and niche != "other":
            queries.append(f"{niche} prediction market news")
            queries.append(f"{niche} market analysis")
        
        # Add market-specific queries (top 3 by volume)
        sorted_markets = sorted(markets, key=lambda m: m.get("total_volume", 0), reverse=True)
        for market in sorted_markets[:3]:
            title = market.get("title", "")
            if title and len(title) > 10:
                # Extract key terms from title
                key_terms = " ".join(title.split()[:5])
                queries.append(key_terms)
        
        return queries[:5]  # Limit to 5 queries
    
    async def _fetch_news_data_apify(self, search_queries: List[str], niche: str, 
                                     markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetch news data using Apify client with fallback to mock data.
        
        Implements Requirement 6.1: Integrate with Apify for real-time news scraping
        """
        if not self.apify_client or not search_queries:
            self.logger.info("Using mock news data (Apify client not available or no queries)")
            return await self._fetch_news_data_mock(niche, markets)
        
        try:
            self.logger.info(f"Fetching news via Apify for queries: {search_queries}")
            
            all_articles = []
            
            for query in search_queries[:3]:  # Limit to 3 queries to avoid rate limits
                try:
                    # Use Apify's Google News Scraper or similar actor
                    # Actor ID: "apify/google-news-scraper" or similar
                    run_input = {
                        "searchQuery": query,
                        "maxItems": self.max_news_articles // len(search_queries),
                        "language": "en",
                        "sortBy": "relevance"
                    }
                    
                    # Run the actor and wait for results
                    run = self.apify_client.actor("apify/google-news-scraper").call(
                        run_input=run_input,
                        timeout_secs=30
                    )
                    
                    # Fetch results from dataset
                    items = list(self.apify_client.dataset(run["defaultDatasetId"]).iterate_items())
                    
                    # Transform Apify results to our format
                    for item in items:
                        article = {
                            "title": item.get("title", ""),
                            "content": item.get("description", "") or item.get("snippet", ""),
                            "source": item.get("source", {}).get("name", "Unknown"),
                            "url": item.get("link", ""),
                            "timestamp": item.get("publishedAt", datetime.now().isoformat()),
                            "sentiment": "neutral",  # Will be analyzed separately
                            "relevance_score": 0.8,
                            "query": query
                        }
                        all_articles.append(article)
                    
                    self.logger.info(f"Fetched {len(items)} articles for query: {query}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to fetch news for query '{query}': {e}")
                    continue
            
            if all_articles:
                self.logger.info(f"Successfully fetched {len(all_articles)} articles via Apify")
                return all_articles[:self.max_news_articles]
            else:
                self.logger.warning("No articles fetched via Apify, falling back to mock data")
                return await self._fetch_news_data_mock(niche, markets)
                
        except Exception as e:
            self.logger.error(f"Apify news fetching failed: {e}, falling back to mock data")
            return await self._fetch_news_data_mock(niche, markets)
    
    async def _fetch_news_data_mock(self, niche: str, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch mock news data when Apify is not available."""
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Generate realistic mock news articles
        mock_articles = [
            {
                "title": f"Latest developments in {niche} prediction markets",
                "content": f"Recent analysis shows significant activity in {niche} prediction markets with increased trading volume and liquidity.",
                "source": "Market News Daily",
                "url": f"https://example.com/news/{niche}-markets-1",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "sentiment": "positive",
                "relevance_score": 0.85
            },
            {
                "title": f"Expert analysis: {niche} market trends and forecasts",
                "content": f"Industry experts weigh in on current {niche} market conditions, highlighting key opportunities and risks for traders.",
                "source": "Trading Insights",
                "url": f"https://example.com/news/{niche}-analysis-1",
                "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
                "sentiment": "neutral",
                "relevance_score": 0.75
            },
            {
                "title": f"Market volatility increases in {niche} sector",
                "content": f"Traders report increased volatility in {niche} prediction markets as new information emerges.",
                "source": "Financial Times",
                "url": f"https://example.com/news/{niche}-volatility-1",
                "timestamp": (datetime.now() - timedelta(hours=8)).isoformat(),
                "sentiment": "negative",
                "relevance_score": 0.70
            }
        ]
        
        # Add market-specific articles
        for market in markets[:2]:
            title = market.get("title", "Unknown Market")
            mock_articles.append({
                "title": f"Analysis: {title[:50]}...",
                "content": f"Detailed analysis of the prediction market: {title}",
                "source": "Market Analysis Pro",
                "url": f"https://example.com/news/market-{market.get('event_id', 'unknown')}",
                "timestamp": (datetime.now() - timedelta(hours=12)).isoformat(),
                "sentiment": "neutral",
                "relevance_score": 0.80
            })
        
        return mock_articles[:self.max_news_articles]
    
    async def _analyze_news_sentiment_enhanced(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment from news articles with enhanced NLP.
        
        Args:
            news_data: List of news articles
            
        Returns:
            Comprehensive news sentiment analysis
        """
        if not news_data:
            return {
                "overall": "neutral",
                "confidence": 0.0,
                "article_count": 0,
                "distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            }
        
        # Analyze each article's sentiment
        sentiment_scores = []
        
        for article in news_data:
            title = article.get("title", "").lower()
            content = article.get("content", "").lower()
            text = f"{title} {content}"
            
            # Calculate sentiment score using keyword analysis
            score = 0.0
            
            # Positive keywords
            for strength, keywords in self.sentiment_keywords["bullish"].items():
                weight = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}.get(strength, 0.5)
                matches = sum(1 for keyword in keywords if keyword in text)
                score += matches * weight * 0.2
            
            # Negative keywords
            for strength, keywords in self.sentiment_keywords["bearish"].items():
                weight = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}.get(strength, 0.5)
                matches = sum(1 for keyword in keywords if keyword in text)
                score -= matches * weight * 0.2
            
            # Normalize to -1 to 1 range
            score = max(-1.0, min(1.0, score))
            sentiment_scores.append(score)
            
            # Update article sentiment
            if score > 0.2:
                article["sentiment"] = "positive"
            elif score < -0.2:
                article["sentiment"] = "negative"
            else:
                article["sentiment"] = "neutral"
        
        # Calculate aggregate sentiment
        avg_score = statistics.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Categorize overall sentiment
        if avg_score > 0.2:
            overall = "positive"
        elif avg_score < -0.2:
            overall = "negative"
        else:
            overall = "neutral"
        
        # Calculate distribution
        sentiments = [article.get("sentiment", "neutral") for article in news_data]
        total = len(sentiments)
        
        distribution = {
            "positive": sentiments.count("positive") / total,
            "negative": sentiments.count("negative") / total,
            "neutral": sentiments.count("neutral") / total
        }
        
        # Calculate confidence based on consistency
        if len(sentiment_scores) > 1:
            score_std = statistics.stdev(sentiment_scores)
            confidence = max(0.3, 1.0 - score_std)
        else:
            confidence = 0.5
        
        return {
            "overall": overall,
            "average_score": avg_score,
            "confidence": confidence,
            "article_count": total,
            "distribution": distribution,
            "sentiment_range": {
                "min": min(sentiment_scores) if sentiment_scores else 0.0,
                "max": max(sentiment_scores) if sentiment_scores else 0.0
            }
        }
    
    def _extract_key_topics_enhanced(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key topics from news articles with frequency and relevance scoring."""
        if not news_data:
            return []
        
        topic_frequency = defaultdict(int)
        topic_relevance = defaultdict(float)
        
        for article in news_data:
            title = article.get("title", "").lower()
            content = article.get("content", "").lower()
            relevance = article.get("relevance_score", 0.5)
            
            # Extract words (simple tokenization)
            words = (title + " " + content).split()
            
            for word in words:
                # Filter meaningful words
                if len(word) > 4 and word not in ["market", "markets", "analysis", "latest", "shows", "report"]:
                    topic_frequency[word] += 1
                    topic_relevance[word] += relevance
        
        # Calculate topic scores
        topics = []
        for topic, freq in topic_frequency.items():
            if freq > 1:  # Only topics mentioned multiple times
                avg_relevance = topic_relevance[topic] / freq
                score = freq * avg_relevance
                topics.append({
                    "topic": topic,
                    "frequency": freq,
                    "relevance": avg_relevance,
                    "score": score
                })
        
        # Sort by score and return top topics
        topics.sort(key=lambda x: x["score"], reverse=True)
        return topics[:10]
    
    def _extract_key_entities(self, news_data: List[Dict[str, Any]]) -> List[str]:
        """Extract key entities (names, organizations) from news articles."""
        # Simplified entity extraction - in production, use NER models
        entities = set()
        
        for article in news_data:
            title = article.get("title", "")
            # Extract capitalized words as potential entities
            words = title.split()
            for i, word in enumerate(words):
                if word and word[0].isupper() and len(word) > 2:
                    # Check if it's not a sentence start
                    if i > 0 or (i == 0 and len(words) > 1 and words[1][0].isupper()):
                        entities.add(word)
        
        return list(entities)[:15]
    
    def _detect_news_sentiment_shift_enhanced(self, news_sentiment: Dict[str, Any], 
                                              market_sentiment: str,
                                              sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect enhanced sentiment shift between news and market data."""
        news_overall = news_sentiment.get("overall", "neutral")
        news_score = news_sentiment.get("average_score", 0.0)
        news_confidence = news_sentiment.get("confidence", 0.5)
        
        # Map sentiment categories to numeric scores for comparison
        sentiment_map = {"positive": 1.0, "bullish": 1.0, "neutral": 0.0, "negative": -1.0, "bearish": -1.0}
        market_score = sentiment_map.get(market_sentiment, 0.0)
        
        # Calculate shift magnitude
        shift_magnitude = abs(news_score - market_score)
        shift_detected = shift_magnitude > self.sentiment_shift_threshold
        
        # Determine shift direction
        if news_score > market_score:
            shift_direction = "news_more_positive"
        elif news_score < market_score:
            shift_direction = "news_more_negative"
        else:
            shift_direction = "aligned"
        
        # Calculate alignment score
        alignment_score = 1.0 - (shift_magnitude / 2.0)  # Normalize to 0-1
        
        return {
            "shift_detected": shift_detected,
            "magnitude": shift_magnitude,
            "direction": shift_direction,
            "news_sentiment": news_overall,
            "market_sentiment": market_sentiment,
            "alignment": "aligned" if not shift_detected else "divergent",
            "alignment_score": alignment_score,
            "confidence": news_confidence,
            "risk_level": "high" if shift_detected and shift_magnitude > 0.5 else "low"
        }
    
    def _calculate_news_impact(self, news_data: List[Dict[str, Any]], 
                               markets: List[Dict[str, Any]]) -> float:
        """Calculate the potential impact of news on markets."""
        if not news_data:
            return 0.0
        
        # Factors: article count, recency, relevance, sentiment strength
        article_count_factor = min(1.0, len(news_data) / 10.0)
        
        # Recency factor (newer articles have more impact)
        recency_scores = []
        now = datetime.now()
        for article in news_data:
            try:
                timestamp = datetime.fromisoformat(article.get("timestamp", now.isoformat()).replace('Z', '+00:00'))
                age_hours = (now - timestamp).total_seconds() / 3600
                recency = max(0.0, 1.0 - (age_hours / 168.0))  # Decay over 1 week
                recency_scores.append(recency)
            except:
                recency_scores.append(0.5)
        
        recency_factor = statistics.mean(recency_scores) if recency_scores else 0.5
        
        # Relevance factor
        relevance_scores = [article.get("relevance_score", 0.5) for article in news_data]
        relevance_factor = statistics.mean(relevance_scores) if relevance_scores else 0.5
        
        # Sentiment strength factor
        sentiment_counts = defaultdict(int)
        for article in news_data:
            sentiment_counts[article.get("sentiment", "neutral")] += 1
        
        max_sentiment_count = max(sentiment_counts.values()) if sentiment_counts else 0
        sentiment_strength = max_sentiment_count / len(news_data) if news_data else 0.0
        
        # Composite impact score
        impact = (
            article_count_factor * 0.25 +
            recency_factor * 0.30 +
            relevance_factor * 0.25 +
            sentiment_strength * 0.20
        )
        
        return min(1.0, impact)
    
    def _identify_trending_narratives(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify trending narratives from news articles."""
        if not news_data:
            return []
        
        # Group articles by similar topics/themes
        narratives = defaultdict(list)
        
        for article in news_data:
            title = article.get("title", "").lower()
            # Simple narrative grouping by key terms
            if any(term in title for term in ["election", "vote", "political"]):
                narratives["political"].append(article)
            elif any(term in title for term in ["crypto", "bitcoin", "ethereum"]):
                narratives["crypto"].append(article)
            elif any(term in title for term in ["market", "trading", "price"]):
                narratives["market_dynamics"].append(article)
            elif any(term in title for term in ["risk", "volatility", "uncertainty"]):
                narratives["risk_sentiment"].append(article)
            else:
                narratives["general"].append(article)
        
        # Calculate narrative strength
        trending = []
        for narrative, articles in narratives.items():
            if len(articles) >= 2:  # At least 2 articles for a trend
                trending.append({
                    "narrative": narrative,
                    "article_count": len(articles),
                    "strength": len(articles) / len(news_data),
                    "sample_titles": [a.get("title", "") for a in articles[:3]]
                })
        
        trending.sort(key=lambda x: x["strength"], reverse=True)
        return trending

    
    # ==================== Sentiment Shift Detection Methods ====================
    
    def _detect_temporal_shift(self, sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect temporal sentiment shifts compared to baseline."""
        # In production, this would compare with historical data
        # For now, we estimate based on current volatility and momentum
        
        volatility = sentiment_analysis.get("sentiment_shifts", {}).get("volatility_index", 0.0) if "sentiment_shifts" in sentiment_analysis else 0.0
        momentum = sentiment_analysis.get("sentiment_momentum", {}).get("momentum", 0.0)
        
        shift_magnitude = (volatility + momentum) / 2.0
        
        return {
            "shift_detected": shift_magnitude > 0.4,
            "magnitude": shift_magnitude,
            "direction": "increasing" if momentum > 0.5 else "stable",
            "confidence": 0.6
        }
    
    def _detect_market_divergence(self, market_sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect divergence in sentiment across different markets."""
        if len(market_sentiments) < 2:
            return {"divergence_detected": False, "magnitude": 0.0}
        
        scores = [s.get("sentiment_score", 0.0) for s in market_sentiments]
        
        # Calculate standard deviation as divergence measure
        if len(scores) > 1:
            divergence = statistics.stdev(scores)
        else:
            divergence = 0.0
        
        return {
            "divergence_detected": divergence > 0.5,
            "magnitude": min(1.0, divergence),
            "score_range": {
                "min": min(scores),
                "max": max(scores),
                "spread": max(scores) - min(scores)
            }
        }
    
    def _calculate_sentiment_volatility_enhanced(self, sentiment_analysis: Dict[str, Any]) -> float:
        """Calculate enhanced sentiment volatility index."""
        distribution = sentiment_analysis.get("sentiment_distribution", {})
        market_sentiments = sentiment_analysis.get("market_sentiments", [])
        
        # Factor 1: Distribution-based volatility
        volatile_ratio = distribution.get("volatile", 0.0)
        neutral_ratio = distribution.get("neutral", 0.0)
        distribution_volatility = volatile_ratio + (1.0 - neutral_ratio) * 0.3
        
        # Factor 2: Score variance-based volatility
        if market_sentiments and len(market_sentiments) > 1:
            scores = [s.get("sentiment_score", 0.0) for s in market_sentiments]
            score_variance = statistics.variance(scores)
            variance_volatility = min(1.0, score_variance * 2.0)
        else:
            variance_volatility = 0.0
        
        # Composite volatility
        volatility = (distribution_volatility * 0.6 + variance_volatility * 0.4)
        
        return min(1.0, volatility)
    
    def _detect_momentum_change(self, sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in sentiment momentum."""
        momentum_data = sentiment_analysis.get("sentiment_momentum", {})
        
        current_momentum = momentum_data.get("momentum", 0.0)
        direction = momentum_data.get("direction", "neutral")
        
        # In production, compare with historical momentum
        # For now, classify based on current momentum
        
        if current_momentum > 0.6:
            change = "accelerating"
        elif current_momentum > 0.3:
            change = "moderate"
        else:
            change = "stable"
        
        return {
            "change_detected": current_momentum > 0.4,
            "change_type": change,
            "momentum_value": current_momentum,
            "direction": direction
        }
    
    def _identify_inflection_points(self, market_sentiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify sentiment inflection points across markets."""
        if len(market_sentiments) < 3:
            return []
        
        inflection_points = []
        scores = [s.get("sentiment_score", 0.0) for s in market_sentiments]
        
        # Simple inflection point detection: look for sign changes in differences
        for i in range(1, len(scores) - 1):
            prev_diff = scores[i] - scores[i-1]
            next_diff = scores[i+1] - scores[i]
            
            # Inflection if direction changes
            if prev_diff * next_diff < 0 and abs(prev_diff) > 0.2:
                inflection_points.append({
                    "market_index": i,
                    "market_id": market_sentiments[i].get("market_id", "unknown"),
                    "score": scores[i],
                    "type": "peak" if prev_diff > 0 else "trough"
                })
        
        return inflection_points
    
    def _calculate_shift_magnitude(self, temporal_shift: Dict, market_divergence: Dict,
                                   volatility_index: float, momentum_change: Dict) -> float:
        """Calculate overall sentiment shift magnitude from multiple factors."""
        # Weighted combination of shift indicators
        temporal_mag = temporal_shift.get("magnitude", 0.0)
        divergence_mag = market_divergence.get("magnitude", 0.0)
        momentum_mag = momentum_change.get("momentum_value", 0.0)
        
        magnitude = (
            temporal_mag * 0.35 +
            divergence_mag * 0.25 +
            volatility_index * 0.20 +
            momentum_mag * 0.20
        )
        
        return min(1.0, magnitude)
    
    def _determine_shift_direction(self, current_sentiment: str, temporal_shift: Dict,
                                   momentum_change: Dict) -> str:
        """Determine the direction of sentiment shift."""
        temporal_direction = temporal_shift.get("direction", "stable")
        momentum_direction = momentum_change.get("direction", "neutral")
        
        # Combine signals
        if temporal_direction == "increasing" and momentum_direction in ["bullish", "positive"]:
            return "increasingly_bullish"
        elif temporal_direction == "increasing" and momentum_direction in ["bearish", "negative"]:
            return "increasingly_bearish"
        elif current_sentiment == "bullish":
            return "bullish"
        elif current_sentiment == "bearish":
            return "bearish"
        else:
            return "neutral"
    
    def _classify_sentiment_shift(self, magnitude: float, volatility: float) -> str:
        """Classify the type of sentiment shift."""
        if magnitude > 0.7:
            return "major_shift"
        elif magnitude > 0.4:
            return "moderate_shift"
        elif volatility > 0.6:
            return "volatile_shift"
        else:
            return "stable"
    
    def _calculate_trend_strength(self, market_sentiments: List[Dict[str, Any]],
                                  sentiment_distribution: Dict[str, float]) -> float:
        """Calculate the strength of the current sentiment trend."""
        if not market_sentiments:
            return 0.0
        
        # Factor 1: Dominant category ratio
        max_category_ratio = max(sentiment_distribution.values()) if sentiment_distribution else 0.0
        
        # Factor 2: Score consistency
        scores = [s.get("sentiment_score", 0.0) for s in market_sentiments]
        if len(scores) > 1:
            score_std = statistics.stdev(scores)
            consistency = max(0.0, 1.0 - score_std)
        else:
            consistency = 0.5
        
        # Factor 3: Confidence levels
        confidences = [s.get("confidence", 0.5) for s in market_sentiments]
        avg_confidence = statistics.mean(confidences) if confidences else 0.5
        
        # Composite trend strength
        strength = (
            max_category_ratio * 0.4 +
            consistency * 0.3 +
            avg_confidence * 0.3
        )
        
        return min(1.0, strength)
    
    def _calculate_trend_persistence(self, sentiment_analysis: Dict[str, Any]) -> float:
        """Calculate how persistent the current trend is likely to be."""
        # In production, this would analyze historical trend duration
        # For now, estimate based on current strength and volatility
        
        trends = sentiment_analysis.get("sentiment_trends", {})
        trend_strength = trends.get("trend_strength", 0.0)
        
        volatility = sentiment_analysis.get("sentiment_shifts", {}).get("volatility_index", 0.5) if "sentiment_shifts" in sentiment_analysis else 0.5
        
        # Higher strength and lower volatility = higher persistence
        persistence = trend_strength * (1.0 - volatility * 0.5)
        
        return min(1.0, persistence)
    
    def _calculate_shift_confidence(self, magnitude: float, volatility: float,
                                    trend_strength: float) -> float:
        """Calculate confidence in the detected shift."""
        # Higher magnitude and trend strength, lower volatility = higher confidence
        confidence = (magnitude * 0.4 + trend_strength * 0.4 + (1.0 - volatility) * 0.2)
        return min(1.0, confidence)
    
    def _assess_shift_risk_level(self, magnitude: float, volatility: float) -> str:
        """Assess the risk level associated with the sentiment shift."""
        risk_score = magnitude * 0.6 + volatility * 0.4
        
        if risk_score > 0.7:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"
    
    # ==================== Correlation Analysis Methods ====================
    
    def _calculate_comprehensive_correlation(self, sentiment_analysis: Dict[str, Any],
                                            trader: Dict[str, Any],
                                            trader_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive multi-dimensional correlation."""
        overall_sentiment = sentiment_analysis.get("overall_sentiment", "neutral")
        sentiment_score = self._sentiment_to_score(overall_sentiment)
        
        # Extract trader metrics
        trader_roi = trader.get("roi", 0.0)
        trader_win_rate = trader.get("win_rate", 0.5)
        trader_sharpe = trader.get("sharpe_ratio", 0.0)
        trader_risk = trader_data.get("risk", {}).get("risk_score", 0.5)
        
        # Calculate correlation components
        correlations = {}
        
        # ROI correlation
        if sentiment_score > 0 and trader_roi > 0:
            correlations["roi"] = min(1.0, 0.5 + abs(sentiment_score) * trader_roi)
        elif sentiment_score < 0 and trader_roi < 0:
            correlations["roi"] = min(1.0, 0.5 + abs(sentiment_score) * abs(trader_roi))
        else:
            correlations["roi"] = max(0.0, 0.5 - abs(sentiment_score - trader_roi) * 0.5)
        
        # Win rate correlation
        if trader_win_rate > 0.6:
            correlations["win_rate"] = 0.7 + (trader_win_rate - 0.6) * 0.5
        else:
            correlations["win_rate"] = 0.5
        
        # Risk correlation (inverse - lower risk with positive sentiment is good)
        if sentiment_score > 0:
            correlations["risk"] = 1.0 - trader_risk
        else:
            correlations["risk"] = trader_risk
        
        # Sharpe ratio correlation
        correlations["sharpe"] = min(1.0, 0.5 + trader_sharpe * 0.2)
        
        # Calculate weighted overall correlation
        overall_correlation = (
            correlations["roi"] * 0.35 +
            correlations["win_rate"] * 0.25 +
            correlations["risk"] * 0.20 +
            correlations["sharpe"] * 0.20
        )
        
        # Determine alignment
        if overall_correlation > 0.7:
            alignment = "high_alignment"
        elif overall_correlation > 0.5:
            alignment = "moderate_alignment"
        else:
            alignment = "low_alignment"
        
        # Calculate performance sensitivity to sentiment
        sensitivity = abs(sentiment_score) * trader_roi if trader_roi != 0 else 0.0
        
        return {
            "overall_correlation": overall_correlation,
            "breakdown": correlations,
            "alignment": alignment,
            "sensitivity": min(1.0, abs(sensitivity)),
            "risk_correlation": correlations["risk"],
            "confidence": 0.7  # Base confidence
        }
    
    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment category to numeric score."""
        sentiment_map = {
            "bullish": 1.0,
            "positive": 0.8,
            "neutral": 0.0,
            "negative": -0.8,
            "bearish": -1.0,
            "volatile": 0.0
        }
        return sentiment_map.get(sentiment, 0.0)
    
    def _identify_sentiment_aligned_opportunities(self, correlations: List[Dict[str, Any]],
                                                  sentiment_analysis: Dict[str, Any],
                                                  trader_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify trading opportunities aligned with current sentiment."""
        opportunities = []
        
        overall_sentiment = sentiment_analysis.get("overall_sentiment", "neutral")
        sentiment_strength = sentiment_analysis.get("sentiment_strength", 0.0)
        
        for correlation in correlations:
            if correlation.get("sentiment_correlation", 0) > self.high_correlation_threshold:
                trader_id = correlation.get("trader_id", "unknown")
                
                # Find corresponding trader data
                trader_data = next((t for t in trader_scores if t.get("trader", {}).get("trader_id") == trader_id), None)
                
                if trader_data:
                    opportunity_score = (
                        correlation["sentiment_correlation"] * 0.4 +
                        trader_data.get("score", 0.5) * 0.4 +
                        sentiment_strength * 0.2
                    )
                    
                    opportunities.append({
                        "trader_id": trader_id,
                        "platform": correlation.get("platform", "unknown"),
                        "opportunity_score": opportunity_score,
                        "sentiment_alignment": correlation.get("sentiment_alignment", "unknown"),
                        "current_sentiment": overall_sentiment,
                        "recommendation": "strong_buy" if opportunity_score > 0.8 else "buy"
                    })
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return opportunities[:10]  # Top 10 opportunities
    
    def _calculate_predictive_power(self, correlations: List[Dict[str, Any]],
                                    sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the predictive power of sentiment for trader performance."""
        if not correlations:
            return {"power": 0.0, "confidence": 0.0}
        
        # Average correlation as base predictive power
        valid_correlations = [c.get("sentiment_correlation", 0) for c in correlations if "error" not in c]
        
        if not valid_correlations:
            return {"power": 0.0, "confidence": 0.0}
        
        avg_correlation = statistics.mean(valid_correlations)
        correlation_consistency = 1.0 - statistics.stdev(valid_correlations) if len(valid_correlations) > 1 else 0.5
        
        sentiment_confidence = sentiment_analysis.get("confidence_level", 0.5)
        
        # Composite predictive power
        predictive_power = (
            avg_correlation * 0.5 +
            correlation_consistency * 0.3 +
            sentiment_confidence * 0.2
        )
        
        return {
            "power": predictive_power,
            "confidence": correlation_consistency,
            "sample_size": len(valid_correlations),
            "reliability": "high" if predictive_power > 0.7 else "moderate" if predictive_power > 0.5 else "low"
        }
    
    def _calculate_correlation_confidence(self, correlations: List[float], sample_size: int) -> float:
        """Calculate confidence in correlation analysis."""
        if not correlations or sample_size == 0:
            return 0.0
        
        # Factor 1: Sample size (more traders = higher confidence)
        sample_factor = min(1.0, sample_size / 20.0)
        
        # Factor 2: Correlation consistency
        if len(correlations) > 1:
            consistency = 1.0 - statistics.stdev(correlations)
        else:
            consistency = 0.5
        
        # Factor 3: Average correlation strength
        avg_correlation = statistics.mean(correlations)
        
        confidence = (
            sample_factor * 0.3 +
            consistency * 0.4 +
            avg_correlation * 0.3
        )
        
        return min(1.0, confidence)
