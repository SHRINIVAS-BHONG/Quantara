"""
Trader Analysis Agent for the LangGraph trading system.

This agent specializes in analyzing and scoring individual traders,
ranking them based on performance metrics, and generating recommendations.

Requirements implemented:
- 2.4: Trader_Analysis_Agent for trader scoring and ranking
- 8.1: Calculate comprehensive trader scores using performance metrics
- 8.2: Incorporate risk-adjusted returns in scoring algorithms
- 8.3: Rank traders based on weighted scoring criteria
- 8.4: Validate trader data quality and completeness
- 8.5: Generate confidence scores for recommendations
"""

import asyncio
import logging
import statistics
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

from ..core.state import TradingState, TraderProfile


class TraderAnalysisAgent:
    """
    Agent responsible for analyzing and scoring individual traders.
    
    This agent calculates comprehensive trader scores using performance metrics,
    incorporates risk-adjusted returns, ranks traders based on weighted criteria,
    and generates confidence scores for recommendations.
    
    Requirements implemented:
    - 8.1: Comprehensive trader scoring with multiple performance metrics
    - 8.2: Risk-adjusted return calculations in scoring algorithms
    - 8.3: Weighted ranking based on configurable criteria
    - 8.4: Data quality validation and completeness checks
    - 8.5: Confidence score generation for recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Trader Analysis Agent.
        
        Args:
            config: Optional configuration for agent behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Scoring weights for different factors (Requirement 8.1, 8.2, 8.3)
        self.scoring_weights = self.config.get("scoring_weights", {
            "performance": 0.35,    # ROI, win rate, Sharpe ratio
            "risk_adjusted": 0.25,  # Risk-adjusted metrics
            "consistency": 0.20,    # Volatility, drawdown consistency
            "sentiment": 0.10,      # Sentiment alignment
            "liquidity": 0.10       # Market liquidity and volume
        })
        
        # Performance categorization thresholds
        self.performance_thresholds = self.config.get("performance_thresholds", {
            "excellent": 0.8,
            "good": 0.6,
            "average": 0.4,
            "poor": 0.2
        })
        
        # Data quality thresholds (Requirement 8.4)
        self.min_quality_score = self.config.get("min_quality_score", 0.6)
        self.min_trades_threshold = self.config.get("min_trades_threshold", 10)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.5)
        
        # Risk-adjusted return parameters (Requirement 8.2)
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)  # 2% annual
        self.max_acceptable_drawdown = self.config.get("max_acceptable_drawdown", 0.3)
        self.volatility_penalty_factor = self.config.get("volatility_penalty_factor", 0.5)
    
    async def calculate_trader_scores(self, state: TradingState) -> TradingState:
        """
        Calculate comprehensive trader scores using performance metrics.
        
        Implements Requirement 8.1:
        - Calculate comprehensive trader scores using performance metrics
        
        Implements Requirement 8.2:
        - Incorporate risk-adjusted returns in scoring algorithms
        
        Args:
            state: Current TradingState with market and risk data
            
        Returns:
            Updated TradingState with trader scores
        """
        self.logger.info("Starting comprehensive trader scoring analysis")
        updated_state = state.copy()
        
        # Get data from previous analysis steps
        risk_assessment = state.get("risk_assessment", {})
        sentiment_analysis = state.get("sentiment_analysis", {})
        trader_risk_profiles = risk_assessment.get("trader_risk_profiles", [])
        user_preferences = state.get("user_preferences", {})
        
        # Initialize scoring metrics
        scoring_metrics = {
            "total_traders_scored": 0,
            "scoring_timestamp": datetime.now().isoformat(),
            "scoring_method": "multi_factor_weighted",
            "weights_used": self.scoring_weights
        }
        
        # Calculate scores for each trader
        trader_scores = []
        
        for trader_profile in trader_risk_profiles:
            try:
                score_data = await self._calculate_individual_trader_score(
                    trader_profile, sentiment_analysis, user_preferences
                )
                trader_scores.append(score_data)
                scoring_metrics["total_traders_scored"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to score trader {trader_profile.get('trader_id', 'unknown')}: {e}")
                # Add error to state
                updated_state["error_log"].append({
                    "step": "trader_scoring",
                    "trader_id": trader_profile.get("trader_id", "unknown"),
                    "error": str(e),
                    "severity": "medium",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Calculate aggregate scoring statistics
        if trader_scores:
            scores = [t["score"] for t in trader_scores]
            scoring_metrics["aggregate_statistics"] = {
                "avg_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "score_std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min_score": min(scores),
                "max_score": max(scores),
                "score_distribution": self._calculate_score_distribution(trader_scores)
            }
        
        # Store trader scores
        updated_state["trader_scores"] = trader_scores
        
        # Store scoring metrics
        if "trader_analysis" not in updated_state:
            updated_state["trader_analysis"] = {}
        updated_state["trader_analysis"]["scoring_metrics"] = scoring_metrics
        
        self.logger.info(
            f"Trader scoring completed: {len(trader_scores)} traders scored"
        )
        
        return updated_state
    
    async def rank_traders(self, state: TradingState) -> TradingState:
        """
        Rank traders based on weighted scoring criteria.
        
        Implements Requirement 8.3:
        - Rank traders based on weighted scoring criteria
        
        Args:
            state: Current TradingState with trader scores
            
        Returns:
            Updated TradingState with ranked traders
        """
        self.logger.info("Starting trader ranking analysis")
        updated_state = state.copy()
        
        trader_scores = updated_state.get("trader_scores", [])
        
        if not trader_scores:
            self.logger.warning("No trader scores available for ranking")
            return updated_state
        
        # Sort traders by overall score (descending)
        ranked_traders = sorted(
            trader_scores,
            key=lambda x: x.get("score", 0.0),
            reverse=True
        )
        
        # Add ranking information with comprehensive metrics
        for i, trader_data in enumerate(ranked_traders):
            trader_data["rank"] = i + 1
            trader_data["percentile"] = (len(ranked_traders) - i) / len(ranked_traders) if ranked_traders else 0
            trader_data["rank_category"] = self._categorize_rank(i + 1, len(ranked_traders))
            
            # Add relative performance metrics
            if i > 0:
                trader_data["score_gap_to_leader"] = ranked_traders[0]["score"] - trader_data["score"]
            else:
                trader_data["score_gap_to_leader"] = 0.0
        
        # Update state with ranked traders
        updated_state["trader_scores"] = ranked_traders
        
        # Generate comprehensive ranking summary
        ranking_summary = self._generate_ranking_summary(ranked_traders)
        
        # Store ranking metadata
        if "trader_analysis" not in updated_state:
            updated_state["trader_analysis"] = {}
        
        updated_state["trader_analysis"]["ranking_summary"] = ranking_summary
        updated_state["trader_analysis"]["total_traders"] = len(ranked_traders)
        updated_state["trader_analysis"]["ranking_timestamp"] = datetime.now().isoformat()
        updated_state["trader_analysis"]["ranking_method"] = "weighted_score_descending"
        
        self.logger.info(
            f"Trader ranking completed: {len(ranked_traders)} traders ranked"
        )
        
        return updated_state
    
    async def validate_trader_data(self, state: TradingState) -> TradingState:
        """
        Validate trader data quality and completeness.
        
        Args:
            state: Current TradingState with trader data
            
        Returns:
            Updated TradingState with validation results
        """
        updated_state = state.copy()
        
        trader_scores = updated_state.get("trader_scores", [])
        validation_results = []
        
        for trader_data in trader_scores:
            validation = await self._validate_individual_trader_data(trader_data)
            validation_results.append(validation)
        
        # Filter out traders with poor data quality
        quality_threshold = 0.6  # Minimum data quality score
        
        validated_traders = []
        for i, trader_data in enumerate(trader_scores):
            if validation_results[i]["quality_score"] >= quality_threshold:
                trader_data["data_validation"] = validation_results[i]
                validated_traders.append(trader_data)
        
        updated_state["trader_scores"] = validated_traders
        
        # Store validation summary
        if "trader_analysis" not in updated_state:
            updated_state["trader_analysis"] = {}
        
        updated_state["trader_analysis"]["validation_summary"] = {
            "total_traders": len(trader_scores),
            "validated_traders": len(validated_traders),
            "filtered_traders": len(trader_scores) - len(validated_traders),
            "avg_quality_score": sum(v["quality_score"] for v in validation_results) / max(1, len(validation_results)),
            "quality_threshold": quality_threshold
        }
        
        return updated_state
    
    async def generate_confidence_scores(self, state: TradingState) -> TradingState:
        """
        Generate confidence scores for trader recommendations.
        
        Args:
            state: Current TradingState with validated trader data
            
        Returns:
            Updated TradingState with confidence scores
        """
        updated_state = state.copy()
        
        trader_scores = updated_state.get("trader_scores", [])
        
        for trader_data in trader_scores:
            confidence_score = self._calculate_recommendation_confidence(
                trader_data, state.get("sentiment_analysis", {}), state.get("risk_assessment", {})
            )
            trader_data["recommendation_confidence"] = confidence_score
        
        # Calculate overall confidence for the recommendation set
        if trader_scores:
            overall_confidence = sum(t.get("recommendation_confidence", 0) for t in trader_scores) / len(trader_scores)
        else:
            overall_confidence = 0.0
        
        updated_state["confidence_score"] = overall_confidence
        
        return updated_state
    
    async def _calculate_individual_trader_score(
        self,
        trader_profile: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive score for an individual trader.
        
        Args:
            trader_profile: Trader risk profile and metrics
            sentiment_analysis: Market sentiment data
            user_preferences: User configuration and preferences
            
        Returns:
            Trader score data with detailed breakdown
        """
        await asyncio.sleep(0.01)  # Simulate processing time
        
        metrics = trader_profile.get("metrics", {})
        risk_factors = trader_profile.get("risk_factors", {})
        
        # Calculate component scores
        performance_score = self._calculate_performance_score(metrics)
        risk_adjusted_score = self._calculate_risk_adjusted_score(metrics, risk_factors)
        consistency_score = self._calculate_consistency_score(metrics)
        sentiment_score = self._calculate_sentiment_alignment_score(
            trader_profile, sentiment_analysis
        )
        liquidity_score = self._calculate_liquidity_score(trader_profile)
        
        # Calculate weighted overall score
        overall_score = (
            performance_score * self.scoring_weights["performance"] +
            risk_adjusted_score * self.scoring_weights["risk_adjusted"] +
            consistency_score * self.scoring_weights["consistency"] +
            sentiment_score * self.scoring_weights["sentiment"] +
            liquidity_score * self.scoring_weights["liquidity"]
        )
        
        # Create trader profile object
        trader = TraderProfile(
            trader_id=trader_profile["trader_id"],
            platform=trader_profile["platform"],
            niche="",  # Will be filled from market data
            win_rate=metrics.get("win_rate", 0.5),
            roi=metrics.get("roi", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 1.0),
            max_drawdown=metrics.get("max_drawdown", 0.1),
            risk_score=trader_profile["risk_score"],
            volatility=metrics.get("volatility", 0.2),
            correlation_market=0.5,  # Mock value
            total_trades=100,  # Mock value
            avg_position_size=1000.0,  # Mock value
            trading_frequency="daily",  # Mock value
            sentiment_score=sentiment_score,
            news_mentions=5,  # Mock value
            social_sentiment={},  # Mock value
            last_updated="2024-01-15T10:00:00Z",
            confidence_level=trader_profile.get("risk_confidence", 0.7),
            data_quality_score=0.8  # Will be calculated in validation
        )
        
        return {
            "trader": trader,
            "score": overall_score,
            "score_breakdown": {
                "performance": performance_score,
                "risk_adjusted": risk_adjusted_score,
                "consistency": consistency_score,
                "sentiment_alignment": sentiment_score,
                "liquidity": liquidity_score
            },
            "risk": trader_profile,
            "performance_category": self._categorize_performance(overall_score)
        }
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on ROI, win rate, and Sharpe ratio."""
        roi = metrics.get("roi", 0.0)
        win_rate = metrics.get("win_rate", 0.5)
        sharpe_ratio = metrics.get("sharpe_ratio", 1.0)
        
        # Normalize and weight performance metrics
        roi_score = min(1.0, max(0.0, (roi + 0.2) / 0.6))  # -20% to 40% ROI range
        win_rate_score = win_rate  # Already 0-1
        sharpe_score = min(1.0, max(0.0, sharpe_ratio / 3.0))  # 0-3 Sharpe range
        
        # Weighted average
        performance_score = (roi_score * 0.5 + win_rate_score * 0.3 + sharpe_score * 0.2)
        
        return performance_score
    
    def _calculate_risk_adjusted_score(self, metrics: Dict[str, Any], risk_factors: Dict[str, Any]) -> float:
        """Calculate risk-adjusted performance score."""
        roi = metrics.get("roi", 0.0)
        volatility = metrics.get("volatility", 0.2)
        max_drawdown = metrics.get("max_drawdown", 0.1)
        
        # Risk-adjusted return (ROI per unit of risk)
        risk_adjusted_roi = roi / max(0.05, volatility)
        
        # Drawdown penalty
        drawdown_penalty = 1.0 - min(0.5, max_drawdown * 2)  # Penalty for high drawdowns
        
        # Normalize risk-adjusted ROI
        risk_score = min(1.0, max(0.0, (risk_adjusted_roi + 1.0) / 3.0))  # -100% to 200% range
        
        return risk_score * drawdown_penalty
    
    def _calculate_consistency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate consistency score based on volatility and drawdown patterns."""
        volatility = metrics.get("volatility", 0.2)
        max_drawdown = metrics.get("max_drawdown", 0.1)
        win_rate = metrics.get("win_rate", 0.5)
        
        # Lower volatility = higher consistency
        volatility_score = 1.0 - min(1.0, volatility / 0.5)  # Normalize to 50% max volatility
        
        # Lower drawdown = higher consistency
        drawdown_score = 1.0 - min(1.0, max_drawdown / 0.3)  # Normalize to 30% max drawdown
        
        # Higher win rate = higher consistency
        win_rate_score = win_rate
        
        # Weighted average
        consistency_score = (volatility_score * 0.4 + drawdown_score * 0.4 + win_rate_score * 0.2)
        
        return consistency_score
    
    def _calculate_sentiment_alignment_score(
        self, 
        trader_profile: Dict[str, Any], 
        sentiment_analysis: Dict[str, Any]
    ) -> float:
        """Calculate how well trader performance aligns with market sentiment."""
        # Get sentiment correlation from sentiment analysis
        correlations = sentiment_analysis.get("performance_correlations", {})
        trader_correlations = correlations.get("trader_correlations", [])
        
        # Find correlation for this trader
        trader_id = trader_profile["trader_id"]
        trader_correlation = 0.5  # Default neutral correlation
        
        for correlation_data in trader_correlations:
            if correlation_data.get("trader_id") == trader_id:
                trader_correlation = correlation_data.get("sentiment_correlation", 0.5)
                break
        
        return trader_correlation
    
    def _calculate_liquidity_score(self, trader_profile: Dict[str, Any]) -> float:
        """Calculate score based on market liquidity and trading volume."""
        # This would typically use market data associated with the trader
        # For now, return a mock score based on platform
        
        platform = trader_profile.get("platform", "")
        
        # Different platforms might have different liquidity characteristics
        platform_liquidity = {
            "polymarket": 0.8,
            "kalshi": 0.7,
            "other": 0.5
        }
        
        return platform_liquidity.get(platform, 0.5)
    
    def _categorize_performance(self, score: float) -> str:
        """Categorize trader performance based on overall score."""
        if score >= self.performance_thresholds["excellent"]:
            return "excellent"
        elif score >= self.performance_thresholds["good"]:
            return "good"
        elif score >= self.performance_thresholds["average"]:
            return "average"
        else:
            return "poor"
    
    def _generate_ranking_summary(self, ranked_traders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for trader rankings."""
        if not ranked_traders:
            return {
                "total_traders": 0,
                "performance_distribution": {},
                "avg_score": 0.0,
                "top_performer_score": 0.0,
                "score_range": 0.0
            }
        
        scores = [t.get("score", 0.0) for t in ranked_traders]
        categories = [t.get("performance_category", "poor") for t in ranked_traders]
        
        performance_distribution = {
            "excellent": categories.count("excellent"),
            "good": categories.count("good"),
            "average": categories.count("average"),
            "poor": categories.count("poor")
        }
        
        return {
            "total_traders": len(ranked_traders),
            "performance_distribution": performance_distribution,
            "avg_score": sum(scores) / len(scores),
            "top_performer_score": max(scores),
            "bottom_performer_score": min(scores),
            "score_range": max(scores) - min(scores),
            "median_score": sorted(scores)[len(scores) // 2]
        }
    
    async def _validate_individual_trader_data(self, trader_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data quality for an individual trader.
        
        Args:
            trader_data: Trader score data to validate
            
        Returns:
            Validation results with quality score
        """
        await asyncio.sleep(0.005)  # Simulate validation processing
        
        validation_checks = {
            "has_trader_profile": "trader" in trader_data,
            "has_score": "score" in trader_data and isinstance(trader_data["score"], (int, float)),
            "has_risk_data": "risk" in trader_data,
            "score_in_range": 0.0 <= trader_data.get("score", -1) <= 1.0,
            "has_breakdown": "score_breakdown" in trader_data
        }
        
        # Calculate quality score based on validation checks
        passed_checks = sum(1 for check in validation_checks.values() if check)
        quality_score = passed_checks / len(validation_checks)
        
        # Additional quality factors
        trader = trader_data.get("trader")
        if trader:
            # Check for reasonable metric values
            if hasattr(trader, "win_rate") and 0.0 <= trader.win_rate <= 1.0:
                quality_score += 0.1
            if hasattr(trader, "roi") and -1.0 <= trader.roi <= 2.0:  # Reasonable ROI range
                quality_score += 0.1
            if hasattr(trader, "confidence_level") and trader.confidence_level > 0.5:
                quality_score += 0.1
        
        # Normalize quality score to 0-1 range
        quality_score = min(1.0, quality_score)
        
        return {
            "quality_score": quality_score,
            "validation_checks": validation_checks,
            "passed_checks": passed_checks,
            "total_checks": len(validation_checks),
            "data_completeness": passed_checks / len(validation_checks)
        }
    
    def _calculate_recommendation_confidence(
        self,
        trader_data: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for trader recommendation.
        
        Implements Requirement 8.5:
        - Generate confidence scores for recommendations
        
        Args:
            trader_data: Trader score and profile data
            sentiment_analysis: Market sentiment analysis
            risk_assessment: Risk assessment results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []
        
        # Data quality confidence (Requirement 8.4)
        validation = trader_data.get("data_validation", {})
        data_confidence = validation.get("quality_score", 0.5)
        confidence_factors.append(data_confidence * 0.3)
        
        # Score confidence (higher scores = higher confidence)
        score_confidence = trader_data.get("score", 0.0)
        confidence_factors.append(score_confidence * 0.25)
        
        # Risk confidence
        risk_data = trader_data.get("risk", {})
        risk_confidence = risk_data.get("risk_confidence", 0.5)
        confidence_factors.append(risk_confidence * 0.2)
        
        # Sentiment alignment confidence
        sentiment_score = trader_data.get("score_breakdown", {}).get("sentiment_alignment", 0.5)
        confidence_factors.append(sentiment_score * 0.15)
        
        # Consistency confidence
        consistency_score = trader_data.get("score_breakdown", {}).get("consistency", 0.5)
        confidence_factors.append(consistency_score * 0.1)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors)
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _calculate_score_distribution(self, trader_scores: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of scores across performance categories."""
        distribution = {
            "excellent": 0,
            "good": 0,
            "average": 0,
            "poor": 0
        }
        
        for trader_data in trader_scores:
            category = trader_data.get("performance_category", "poor")
            if category in distribution:
                distribution[category] += 1
        
        return distribution
    
    def _categorize_rank(self, rank: int, total_traders: int) -> str:
        """Categorize trader rank into tiers."""
        percentile = (total_traders - rank + 1) / total_traders
        
        if percentile >= 0.9:
            return "top_tier"
        elif percentile >= 0.75:
            return "high_tier"
        elif percentile >= 0.5:
            return "mid_tier"
        elif percentile >= 0.25:
            return "low_tier"
        else:
            return "bottom_tier"