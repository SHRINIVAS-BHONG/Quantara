"""
Risk Assessment Agent for the LangGraph trading system.

This agent specializes in risk analysis, portfolio optimization,
and risk-adjusted recommendation generation.

Requirements implemented:
- 2.3: Risk_Assessment_Agent for risk analysis and portfolio optimization
- 7.1: Calculate individual trader risk profiles using multiple metrics
- 7.2: Assess overall portfolio risk and correlation analysis
- 7.3: Recommend optimal position sizing based on risk parameters
- 7.4: Monitor risk limits and generate alerts for violations
- 7.5: Validate all recommendations against user risk tolerance
"""

import asyncio
import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import math

from ..core.state import TradingState


class RiskAssessmentAgent:
    """
    Agent focused on risk analysis and portfolio optimization.
    
    This agent assesses individual and portfolio risk, recommends position
    sizing strategies, monitors risk limits, and provides risk-adjusted
    recommendations aligned with user preferences.
    
    Requirements implemented:
    - 7.1: Individual trader risk profiling with multiple metrics
    - 7.2: Portfolio risk assessment and correlation analysis
    - 7.3: Position sizing recommendations based on risk parameters
    - 7.4: Risk limit monitoring and alert generation
    - 7.5: Recommendation validation against user risk tolerance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Risk Assessment Agent.
        
        Args:
            config: Optional configuration for agent behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Risk configuration
        self.risk_categories = ["low", "medium", "high", "extreme"]
        self.risk_factors = [
            "volatility", "liquidity", "market_depth", "correlation",
            "drawdown", "concentration", "leverage", "win_rate_consistency",
            "roi_stability", "sharpe_ratio"
        ]
        
        # Risk thresholds
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "extreme": 1.0
        }
        
        # Portfolio optimization parameters
        self.max_portfolio_risk = self.config.get("max_portfolio_risk", 0.7)
        self.max_individual_risk = self.config.get("max_individual_risk", 0.8)
        self.max_correlation = self.config.get("max_correlation", 0.6)
        self.min_diversification = self.config.get("min_diversification", 0.3)
        self.target_sharpe_ratio = self.config.get("target_sharpe_ratio", 1.5)
        
        # Position sizing parameters
        self.kelly_fraction = self.config.get("kelly_fraction", 0.25)  # Conservative Kelly
        self.max_position_size = self.config.get("max_position_size", 0.25)  # 25% max
        self.min_position_size = self.config.get("min_position_size", 0.01)  # 1% min
        
        # Risk metrics weights for scoring
        self.risk_metric_weights = {
            "volatility": 0.25,
            "max_drawdown": 0.20,
            "win_rate": 0.15,
            "sharpe_ratio": 0.15,
            "roi_consistency": 0.10,
            "liquidity_risk": 0.10,
            "correlation_risk": 0.05
        }
    
    async def assess_trader_risk(self, state: TradingState) -> TradingState:
        """
        Assess individual trader risk profiles using multiple metrics.
        
        Implements Requirement 7.1:
        - Calculate individual trader risk profiles using multiple metrics
        
        Args:
            state: Current TradingState with trader data
            
        Returns:
            Updated TradingState with comprehensive trader risk assessments
        """
        self.logger.info("Starting comprehensive individual trader risk assessment")
        updated_state = state.copy()
        
        # Get trader data from market discovery and sentiment analysis
        all_markets = state.get("polymarket_data", []) + state.get("kalshi_data", [])
        trader_scores = state.get("trader_scores", [])
        sentiment_analysis = state.get("sentiment_analysis", {})
        user_preferences = state.get("user_preferences", {})
        risk_tolerance = user_preferences.get("risk_tolerance", 0.5)
        
        # Initialize risk assessment metrics
        assessment_metrics = {
            "total_traders_assessed": 0,
            "assessment_timestamp": datetime.now().isoformat(),
            "risk_factors_analyzed": self.risk_factors,
            "assessment_method": "multi_factor_comprehensive"
        }
        
        # Assess risk for each trader with comprehensive metrics
        trader_risk_profiles = []
        
        # Process traders from trader_scores (primary source)
        for trader_data in trader_scores:
            trader = trader_data.get("trader", {})
            trader_id = trader.get("trader_id", "unknown")
            
            try:
                # Find associated market data
                associated_market = self._find_trader_market(trader_id, all_markets)
                
                # Calculate comprehensive risk profile
                risk_profile = await self._assess_individual_trader_risk_comprehensive(
                    trader, trader_data, associated_market, sentiment_analysis, risk_tolerance
                )
                
                trader_risk_profiles.append(risk_profile)
                assessment_metrics["total_traders_assessed"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to assess risk for trader {trader_id}: {e}")
                # Add error to state
                updated_state["error_log"].append({
                    "step": "risk_assessment",
                    "trader_id": trader_id,
                    "error": str(e),
                    "severity": "medium",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Also process traders from market data if not in trader_scores
        for market in all_markets:
            top_traders = market.get("top_traders", [])
            
            for trader_id in top_traders:
                # Check if already assessed
                if any(t["trader_id"] == trader_id for t in trader_risk_profiles):
                    continue
                
                try:
                    # Create minimal trader data structure
                    trader_data = {
                        "trader": {"trader_id": trader_id, "platform": market.get("platform", "unknown")},
                        "score": 0.5  # Default score
                    }
                    
                    risk_profile = await self._assess_individual_trader_risk_comprehensive(
                        trader_data["trader"], trader_data, market, sentiment_analysis, risk_tolerance
                    )
                    
                    trader_risk_profiles.append(risk_profile)
                    assessment_metrics["total_traders_assessed"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to assess risk for market trader {trader_id}: {e}")
        
        # Calculate aggregate risk statistics
        if trader_risk_profiles:
            risk_scores = [t["risk_score"] for t in trader_risk_profiles]
            assessment_metrics["aggregate_statistics"] = {
                "avg_risk_score": statistics.mean(risk_scores),
                "median_risk_score": statistics.median(risk_scores),
                "risk_score_std_dev": statistics.stdev(risk_scores) if len(risk_scores) > 1 else 0.0,
                "min_risk_score": min(risk_scores),
                "max_risk_score": max(risk_scores),
                "risk_distribution": self._calculate_risk_distribution(trader_risk_profiles)
            }
        
        # Store comprehensive risk assessments
        risk_assessment = updated_state.get("risk_assessment", {})
        risk_assessment["trader_risk_profiles"] = trader_risk_profiles
        risk_assessment["assessment_metrics"] = assessment_metrics
        risk_assessment["assessment_timestamp"] = datetime.now().isoformat()
        
        updated_state["risk_assessment"] = risk_assessment
        
        self.logger.info(
            f"Trader risk assessment completed: {len(trader_risk_profiles)} traders assessed"
        )
        
        return updated_state
    
    async def calculate_portfolio_risk(self, state: TradingState) -> TradingState:
        """
        Calculate overall portfolio risk and correlation analysis.
        
        Args:
            state: Current TradingState with individual risk assessments
            
        Returns:
            Updated TradingState with portfolio risk analysis
        """
        updated_state = state.copy()
        
        risk_assessment = updated_state.get("risk_assessment", {})
        trader_risk_profiles = risk_assessment.get("trader_risk_profiles", [])
        user_preferences = state.get("user_preferences", {})
        
        # Calculate portfolio-level risk metrics
        portfolio_risk = await self._calculate_portfolio_metrics(
            trader_risk_profiles, user_preferences
        )
        
        # Assess correlation risks
        correlation_analysis = self._analyze_trader_correlations(trader_risk_profiles)
        
        # Calculate diversification metrics
        diversification_metrics = self._calculate_diversification_metrics(
            trader_risk_profiles, state.get("niche", "")
        )
        
        # Update risk assessment with portfolio analysis
        risk_assessment["portfolio_risk"] = portfolio_risk
        risk_assessment["correlation_analysis"] = correlation_analysis
        risk_assessment["diversification_metrics"] = diversification_metrics
        
        updated_state["risk_assessment"] = risk_assessment
        
        return updated_state
    
    async def recommend_position_sizing(self, state: TradingState) -> TradingState:
        """
        Recommend optimal position sizes based on risk analysis.
        
        Args:
            state: Current TradingState with risk assessments
            
        Returns:
            Updated TradingState with position sizing recommendations
        """
        updated_state = state.copy()
        
        risk_assessment = updated_state.get("risk_assessment", {})
        trader_risk_profiles = risk_assessment.get("trader_risk_profiles", [])
        user_preferences = state.get("user_preferences", {})
        risk_tolerance = user_preferences.get("risk_tolerance", 0.5)
        
        # Calculate position sizing for each trader
        position_recommendations = []
        
        for trader_profile in trader_risk_profiles:
            position_size = self._calculate_optimal_position_size(
                trader_profile, risk_tolerance
            )
            
            position_recommendations.append({
                "trader_id": trader_profile["trader_id"],
                "recommended_position_size": position_size,
                "risk_category": trader_profile["risk_category"],
                "confidence": trader_profile["risk_confidence"],
                "max_position_size": position_size * 1.5,  # Maximum allowed
                "min_position_size": position_size * 0.5   # Minimum recommended
            })
        
        # Update risk assessment with position sizing
        risk_assessment["position_recommendations"] = position_recommendations
        updated_state["risk_assessment"] = risk_assessment
        
        return updated_state
    
    async def monitor_risk_limits(self, state: TradingState) -> TradingState:
        """
        Monitor risk limits and generate alerts for violations.
        
        Args:
            state: Current TradingState with risk assessments
            
        Returns:
            Updated TradingState with risk monitoring results
        """
        updated_state = state.copy()
        
        risk_assessment = updated_state.get("risk_assessment", {})
        portfolio_risk = risk_assessment.get("portfolio_risk", {})
        user_preferences = state.get("user_preferences", {})
        
        # Define risk limits based on user preferences
        risk_limits = {
            "max_portfolio_risk": user_preferences.get("max_portfolio_risk", 0.7),
            "max_individual_risk": user_preferences.get("max_individual_risk", 0.8),
            "max_correlation": user_preferences.get("max_correlation", 0.6),
            "min_diversification": user_preferences.get("min_diversification", 0.3)
        }
        
        # Check for risk limit violations
        violations = []
        
        # Portfolio risk check
        current_portfolio_risk = portfolio_risk.get("overall_risk_score", 0.0)
        if current_portfolio_risk > risk_limits["max_portfolio_risk"]:
            violations.append({
                "type": "portfolio_risk",
                "current": current_portfolio_risk,
                "limit": risk_limits["max_portfolio_risk"],
                "severity": "high" if current_portfolio_risk > 0.8 else "medium"
            })
        
        # Individual trader risk checks
        trader_risk_profiles = risk_assessment.get("trader_risk_profiles", [])
        for trader in trader_risk_profiles:
            if trader["risk_score"] > risk_limits["max_individual_risk"]:
                violations.append({
                    "type": "individual_risk",
                    "trader_id": trader["trader_id"],
                    "current": trader["risk_score"],
                    "limit": risk_limits["max_individual_risk"],
                    "severity": "medium"
                })
        
        # Correlation risk checks
        correlation_analysis = risk_assessment.get("correlation_analysis", {})
        max_correlation = correlation_analysis.get("max_correlation", 0.0)
        if max_correlation > risk_limits["max_correlation"]:
            violations.append({
                "type": "correlation_risk",
                "current": max_correlation,
                "limit": risk_limits["max_correlation"],
                "severity": "medium"
            })
        
        # Diversification checks
        diversification_metrics = risk_assessment.get("diversification_metrics", {})
        diversification_score = diversification_metrics.get("diversification_score", 1.0)
        if diversification_score < risk_limits["min_diversification"]:
            violations.append({
                "type": "diversification",
                "current": diversification_score,
                "limit": risk_limits["min_diversification"],
                "severity": "low"
            })
        
        # Update risk assessment with monitoring results
        risk_assessment["risk_monitoring"] = {
            "violations": violations,
            "risk_limits": risk_limits,
            "overall_compliance": len(violations) == 0,
            "monitoring_timestamp": asyncio.get_event_loop().time()
        }
        
        updated_state["risk_assessment"] = risk_assessment
        
        return updated_state
    
    async def _assess_individual_trader_risk(
        self, 
        trader_id: str, 
        market: Dict[str, Any], 
        risk_tolerance: float
    ) -> Dict[str, Any]:
        """
        Assess risk profile for an individual trader.
        
        Args:
            trader_id: Unique trader identifier
            market: Market data containing trader information
            risk_tolerance: User's risk tolerance (0.0 to 1.0)
            
        Returns:
            Trader risk profile dictionary
        """
        # Mock trader data - in real system, would fetch from APIs
        await asyncio.sleep(0.02)  # Simulate processing time
        
        # Generate mock trader metrics for risk assessment
        import random
        random.seed(hash(trader_id) % 2**32)  # Deterministic randomness
        
        # Mock performance metrics
        win_rate = random.uniform(0.4, 0.8)
        roi = random.uniform(-0.2, 0.4)
        sharpe_ratio = random.uniform(0.5, 2.5)
        max_drawdown = random.uniform(0.05, 0.3)
        volatility = random.uniform(0.1, 0.5)
        
        # Calculate risk score based on multiple factors
        risk_score = 0.0
        
        # Volatility factor (higher volatility = higher risk)
        risk_score += volatility * 0.3
        
        # Drawdown factor (higher drawdown = higher risk)
        risk_score += max_drawdown * 0.25
        
        # Win rate factor (lower win rate = higher risk)
        risk_score += (1.0 - win_rate) * 0.2
        
        # ROI consistency factor
        if roi < 0:
            risk_score += 0.15
        
        # Sharpe ratio factor (lower Sharpe = higher risk)
        if sharpe_ratio < 1.0:
            risk_score += 0.1
        
        # Normalize risk score to 0-1 range
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Determine risk category
        if risk_score < 0.3:
            risk_category = "low"
        elif risk_score < 0.6:
            risk_category = "medium"
        elif risk_score < 0.8:
            risk_category = "high"
        else:
            risk_category = "extreme"
        
        # Calculate risk-adjusted metrics
        risk_adjusted_roi = roi / max(0.1, volatility)  # ROI per unit of volatility
        risk_confidence = 1.0 - (volatility * 0.5 + max_drawdown * 0.3)
        
        return {
            "trader_id": trader_id,
            "platform": market.get("platform", ""),
            "market_id": market.get("event_id", ""),
            "risk_score": risk_score,
            "risk_category": risk_category,
            "risk_confidence": max(0.0, min(1.0, risk_confidence)),
            "metrics": {
                "win_rate": win_rate,
                "roi": roi,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "risk_adjusted_roi": risk_adjusted_roi
            },
            "risk_factors": {
                "volatility_risk": volatility,
                "drawdown_risk": max_drawdown,
                "performance_risk": 1.0 - win_rate,
                "consistency_risk": 0.5 if roi >= 0 else 0.8
            }
        }
    
    async def _calculate_portfolio_metrics(
        self, 
        trader_risk_profiles: List[Dict[str, Any]], 
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            trader_risk_profiles: List of individual trader risk profiles
            user_preferences: User configuration and preferences
            
        Returns:
            Portfolio risk metrics dictionary
        """
        if not trader_risk_profiles:
            return {
                "overall_risk_score": 0.0,
                "risk_distribution": {},
                "expected_return": 0.0,
                "portfolio_volatility": 0.0,
                "portfolio_sharpe": 0.0
            }
        
        # Calculate weighted portfolio metrics
        total_traders = len(trader_risk_profiles)
        
        # Overall risk score (average of individual risks)
        overall_risk_score = sum(t["risk_score"] for t in trader_risk_profiles) / total_traders
        
        # Risk distribution
        risk_categories = [t["risk_category"] for t in trader_risk_profiles]
        risk_distribution = {
            "low": risk_categories.count("low") / total_traders,
            "medium": risk_categories.count("medium") / total_traders,
            "high": risk_categories.count("high") / total_traders,
            "extreme": risk_categories.count("extreme") / total_traders
        }
        
        # Expected portfolio return (weighted average ROI)
        expected_return = sum(t["metrics"]["roi"] for t in trader_risk_profiles) / total_traders
        
        # Portfolio volatility (average volatility with correlation adjustment)
        avg_volatility = sum(t["metrics"]["volatility"] for t in trader_risk_profiles) / total_traders
        # Assume some correlation reduction benefit
        portfolio_volatility = avg_volatility * 0.8  # Diversification benefit
        
        # Portfolio Sharpe ratio
        portfolio_sharpe = expected_return / max(0.01, portfolio_volatility)
        
        return {
            "overall_risk_score": overall_risk_score,
            "risk_distribution": risk_distribution,
            "expected_return": expected_return,
            "portfolio_volatility": portfolio_volatility,
            "portfolio_sharpe": portfolio_sharpe,
            "trader_count": total_traders,
            "risk_adjusted_return": expected_return / max(0.1, overall_risk_score)
        }
    
    def _analyze_trader_correlations(self, trader_risk_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between traders."""
        if len(trader_risk_profiles) < 2:
            return {
                "max_correlation": 0.0,
                "avg_correlation": 0.0,
                "correlation_matrix": {},
                "high_correlation_pairs": []
            }
        
        # Mock correlation analysis - in real system, would use statistical methods
        correlations = []
        high_correlation_pairs = []
        
        for i, trader1 in enumerate(trader_risk_profiles):
            for j, trader2 in enumerate(trader_risk_profiles[i+1:], i+1):
                # Simple correlation based on risk similarity
                risk_diff = abs(trader1["risk_score"] - trader2["risk_score"])
                correlation = max(0.0, 1.0 - risk_diff * 2)  # Inverse relationship
                
                correlations.append(correlation)
                
                if correlation > 0.6:  # High correlation threshold
                    high_correlation_pairs.append({
                        "trader1": trader1["trader_id"],
                        "trader2": trader2["trader_id"],
                        "correlation": correlation
                    })
        
        return {
            "max_correlation": max(correlations) if correlations else 0.0,
            "avg_correlation": sum(correlations) / len(correlations) if correlations else 0.0,
            "correlation_count": len(correlations),
            "high_correlation_pairs": high_correlation_pairs
        }
    
    def _calculate_diversification_metrics(
        self, 
        trader_risk_profiles: List[Dict[str, Any]], 
        niche: str
    ) -> Dict[str, Any]:
        """Calculate portfolio diversification metrics."""
        if not trader_risk_profiles:
            return {
                "diversification_score": 0.0,
                "platform_diversification": 0.0,
                "risk_diversification": 0.0,
                "concentration_risk": 1.0
            }
        
        total_traders = len(trader_risk_profiles)
        
        # Platform diversification
        platforms = [t["platform"] for t in trader_risk_profiles]
        unique_platforms = len(set(platforms))
        platform_diversification = unique_platforms / max(1, len(platforms))
        
        # Risk category diversification
        risk_categories = [t["risk_category"] for t in trader_risk_profiles]
        unique_risk_categories = len(set(risk_categories))
        risk_diversification = unique_risk_categories / 4  # 4 risk categories
        
        # Concentration risk (inverse of diversification)
        # Higher concentration = higher risk
        concentration_risk = 1.0 / max(1, total_traders * 0.2)  # Decreases with more traders
        
        # Overall diversification score
        diversification_score = (platform_diversification + risk_diversification) / 2
        
        return {
            "diversification_score": diversification_score,
            "platform_diversification": platform_diversification,
            "risk_diversification": risk_diversification,
            "concentration_risk": min(1.0, concentration_risk),
            "trader_count": total_traders,
            "unique_platforms": unique_platforms,
            "unique_risk_categories": unique_risk_categories
        }
    
    def _calculate_optimal_position_size(
        self, 
        trader_profile: Dict[str, Any], 
        risk_tolerance: float
    ) -> float:
        """
        Calculate optimal position size for a trader based on risk profile.
        
        Args:
            trader_profile: Individual trader risk assessment
            risk_tolerance: User's risk tolerance (0.0 to 1.0)
            
        Returns:
            Recommended position size as percentage of portfolio (0.0 to 1.0)
        """
        trader_risk = trader_profile["risk_score"]
        trader_confidence = trader_profile["risk_confidence"]
        
        # Base position size inversely related to risk
        base_position = (1.0 - trader_risk) * 0.3  # Max 30% base allocation
        
        # Adjust for user risk tolerance
        risk_adjustment = risk_tolerance * 0.5  # Risk tolerance can increase allocation
        
        # Adjust for confidence in the trader
        confidence_adjustment = trader_confidence * 0.2
        
        # Calculate final position size
        position_size = base_position + risk_adjustment + confidence_adjustment
        
        # Apply constraints
        min_position = 0.01  # Minimum 1%
        max_position = 0.25  # Maximum 25% per trader
        
        return max(min_position, min(max_position, position_size))

    
    # ==================== Enhanced Helper Methods ====================
    
    def _find_trader_market(self, trader_id: str, markets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the market associated with a trader."""
        for market in markets:
            top_traders = market.get("top_traders", [])
            if trader_id in top_traders:
                return market
        return None
    
    async def _assess_individual_trader_risk_comprehensive(
        self,
        trader: Dict[str, Any],
        trader_data: Dict[str, Any],
        market: Optional[Dict[str, Any]],
        sentiment_analysis: Dict[str, Any],
        risk_tolerance: float
    ) -> Dict[str, Any]:
        """
        Comprehensive individual trader risk assessment with multiple metrics.
        
        Implements Requirement 7.1:
        - Calculate individual trader risk profiles using multiple metrics
        
        Args:
            trader: Trader information dictionary
            trader_data: Trader scoring data
            market: Associated market data (optional)
            sentiment_analysis: Sentiment analysis results
            risk_tolerance: User's risk tolerance
            
        Returns:
            Comprehensive trader risk profile
        """
        await asyncio.sleep(0.01)  # Simulate processing time
        
        trader_id = trader.get("trader_id", "unknown")
        platform = trader.get("platform", "unknown")
        
        # Generate deterministic mock metrics based on trader_id
        import random
        random.seed(hash(trader_id) % 2**32)
        
        # Performance metrics
        win_rate = random.uniform(0.4, 0.85)
        roi = random.uniform(-0.15, 0.45)
        sharpe_ratio = random.uniform(0.5, 2.8)
        max_drawdown = random.uniform(0.05, 0.35)
        volatility = random.uniform(0.08, 0.55)
        total_trades = random.randint(10, 500)
        avg_position_size = random.uniform(100, 10000)
        
        # Calculate win rate consistency (lower std dev = more consistent)
        win_rate_consistency = 1.0 - random.uniform(0.1, 0.4)
        
        # Calculate ROI stability
        roi_stability = 1.0 - (volatility * 0.5)
        
        # Liquidity risk based on market data
        liquidity_risk = 0.3  # Default
        if market:
            market_liquidity = market.get("liquidity", 0)
            liquidity_risk = max(0.1, min(0.9, 1.0 - (market_liquidity / 50000.0)))
        
        # Sentiment-based risk adjustment
        sentiment_risk_adjustment = 0.0
        if sentiment_analysis:
            overall_sentiment = sentiment_analysis.get("overall_sentiment", "neutral")
            if overall_sentiment == "bearish":
                sentiment_risk_adjustment = 0.1
            elif overall_sentiment == "volatile":
                sentiment_risk_adjustment = 0.15
        
        # Calculate comprehensive risk score using weighted factors
        risk_components = {
            "volatility": volatility * self.risk_metric_weights["volatility"],
            "max_drawdown": max_drawdown * self.risk_metric_weights["max_drawdown"],
            "win_rate": (1.0 - win_rate) * self.risk_metric_weights["win_rate"],
            "sharpe_ratio": max(0, (1.5 - sharpe_ratio) / 1.5) * self.risk_metric_weights["sharpe_ratio"],
            "roi_consistency": (1.0 - win_rate_consistency) * self.risk_metric_weights["roi_consistency"],
            "liquidity_risk": liquidity_risk * self.risk_metric_weights["liquidity_risk"],
            "sentiment_adjustment": sentiment_risk_adjustment
        }
        
        # Calculate overall risk score
        risk_score = sum(risk_components.values())
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Determine risk category
        risk_category = self._categorize_risk(risk_score)
        
        # Calculate risk-adjusted metrics
        risk_adjusted_roi = roi / max(0.1, volatility)
        risk_adjusted_sharpe = sharpe_ratio * (1.0 - risk_score * 0.3)
        
        # Calculate risk confidence based on data quality
        risk_confidence = self._calculate_risk_confidence(
            total_trades, win_rate_consistency, roi_stability, market
        )
        
        # Calculate Value at Risk (VaR) estimate
        var_95 = self._calculate_var(roi, volatility, 0.95)
        var_99 = self._calculate_var(roi, volatility, 0.99)
        
        # Calculate Expected Shortfall (CVaR)
        cvar_95 = var_95 * 1.3  # Simplified CVaR estimation
        
        return {
            "trader_id": trader_id,
            "platform": platform,
            "market_id": market.get("event_id", "unknown") if market else "unknown",
            "risk_score": risk_score,
            "risk_category": risk_category,
            "risk_confidence": risk_confidence,
            "risk_components": risk_components,
            "metrics": {
                "win_rate": win_rate,
                "roi": roi,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "total_trades": total_trades,
                "avg_position_size": avg_position_size,
                "win_rate_consistency": win_rate_consistency,
                "roi_stability": roi_stability,
                "risk_adjusted_roi": risk_adjusted_roi,
                "risk_adjusted_sharpe": risk_adjusted_sharpe
            },
            "risk_factors": {
                "volatility_risk": volatility,
                "drawdown_risk": max_drawdown,
                "performance_risk": 1.0 - win_rate,
                "consistency_risk": 1.0 - win_rate_consistency,
                "liquidity_risk": liquidity_risk,
                "sentiment_risk": sentiment_risk_adjustment
            },
            "var_metrics": {
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95
            },
            "assessment_timestamp": datetime.now().isoformat()
        }
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into risk category."""
        if risk_score < self.risk_thresholds["low"]:
            return "low"
        elif risk_score < self.risk_thresholds["medium"]:
            return "medium"
        elif risk_score < self.risk_thresholds["high"]:
            return "high"
        else:
            return "extreme"
    
    def _calculate_risk_confidence(
        self,
        total_trades: int,
        win_rate_consistency: float,
        roi_stability: float,
        market: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in risk assessment based on data quality."""
        confidence = 0.5  # Base confidence
        
        # More trades = higher confidence
        if total_trades > 100:
            confidence += 0.2
        elif total_trades > 50:
            confidence += 0.1
        
        # Higher consistency = higher confidence
        confidence += win_rate_consistency * 0.15
        
        # Higher stability = higher confidence
        confidence += roi_stability * 0.1
        
        # Market data availability increases confidence
        if market:
            confidence += 0.05
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_var(self, roi: float, volatility: float, confidence_level: float) -> float:
        """
        Calculate Value at Risk (VaR) using parametric method.
        
        Args:
            roi: Expected return
            volatility: Return volatility
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR estimate
        """
        # Z-scores for common confidence levels
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.99: 2.33
        }
        
        z_score = z_scores.get(confidence_level, 1.65)
        var = roi - (z_score * volatility)
        
        return var
    
    def _calculate_risk_distribution(self, trader_risk_profiles: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of traders across risk categories."""
        if not trader_risk_profiles:
            return {"low": 0.0, "medium": 0.0, "high": 0.0, "extreme": 0.0}
        
        total = len(trader_risk_profiles)
        categories = [t["risk_category"] for t in trader_risk_profiles]
        
        return {
            "low": categories.count("low") / total,
            "medium": categories.count("medium") / total,
            "high": categories.count("high") / total,
            "extreme": categories.count("extreme") / total
        }
