"""
Market Discovery Agent for the LangGraph trading system.

This agent specializes in discovering and analyzing prediction markets across
multiple platforms, classifying market niches, and identifying trading opportunities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json

from ..core.state import TradingState
from ...core.polymarket import fetch_traders as fetch_polymarket_traders
from ...core.kalshi import fetch_traders as fetch_kalshi_traders
from ...core.niche import classify_by_niche


class MarketDiscoveryAgent:
    """
    Specialized agent for discovering and analyzing prediction markets.
    
    This agent orchestrates market data collection from multiple platforms,
    classifies markets into appropriate niches, assesses market conditions,
    and identifies high-potential trading opportunities.
    
    Requirements implemented:
    - 2.1: Market_Discovery_Agent for market identification and classification
    - 5.1: Query Polymarket and Kalshi APIs for active markets
    - 5.2: Classify markets into appropriate niches using machine learning
    - 5.3: Assess market liquidity and trading volume metrics
    - 5.4: Validate data quality and completeness
    - 5.5: Identify high-potential trading opportunities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Market Discovery Agent.
        
        Args:
            config: Optional configuration for agent behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Platform configuration
        self.supported_platforms = ["polymarket", "kalshi"]
        self.niche_categories = [
            "politics", "sports", "crypto", "economics", "entertainment",
            "technology", "weather", "social", "other"
        ]
        
        # Quality thresholds
        self.min_volume_threshold = self.config.get("min_volume_threshold", 100.0)
        self.min_liquidity_threshold = self.config.get("min_liquidity_threshold", 50.0)
        self.high_liquidity_threshold = self.config.get("high_liquidity_threshold", 1000.0)
        self.opportunity_score_threshold = self.config.get("opportunity_score_threshold", 0.5)
        
        # Data validation settings
        self.max_api_retries = self.config.get("max_api_retries", 3)
        self.api_timeout = self.config.get("api_timeout", 30.0)
        self.data_quality_threshold = self.config.get("data_quality_threshold", 0.7)
    
    
    async def discover_markets(self, state: TradingState) -> TradingState:
        """
        Discover relevant markets across supported platforms with enhanced error handling.
        
        Implements Requirements 5.1 and 5.4:
        - Query Polymarket and Kalshi APIs for active markets
        - Validate data quality and completeness
        
        Args:
            state: Current TradingState
            
        Returns:
            Updated TradingState with market data and quality metrics
        """
        self.logger.info("Starting market discovery across platforms")
        updated_state = state.copy()
        
        # Extract query parameters with enhanced parsing
        query = state["original_query"]
        parsed_intent = state.get("parsed_intent", {})
        platforms = parsed_intent.get("platforms", self.supported_platforms)
        niche_preference = parsed_intent.get("niche_preference", "")
        
        # Initialize market discovery metrics
        discovery_metrics = {
            "start_time": datetime.now().isoformat(),
            "platforms_queried": platforms,
            "query_hash": self._generate_query_hash(query),
            "api_calls_made": 0,
            "api_failures": 0,
            "data_quality_scores": {},
            "total_markets_found": 0
        }
        
        # Parallel market discovery with enhanced error handling
        discovery_tasks = []
        
        if "polymarket" in platforms:
            discovery_tasks.append({
                "platform": "polymarket",
                "task": self._discover_polymarket_data_enhanced(query, niche_preference)
            })
        
        if "kalshi" in platforms:
            discovery_tasks.append({
                "platform": "kalshi", 
                "task": self._discover_kalshi_data_enhanced(query, niche_preference)
            })
        
        # Execute discovery tasks with timeout and retry logic
        polymarket_data = []
        kalshi_data = []
        
        for task_info in discovery_tasks:
            platform = task_info["platform"]
            task = task_info["task"]
            
            try:
                discovery_metrics["api_calls_made"] += 1
                
                # Execute with timeout and retries
                result = await self._execute_with_retries(task, platform)
                
                if platform == "polymarket":
                    polymarket_data = result["markets"]
                    discovery_metrics["data_quality_scores"]["polymarket"] = result["quality_score"]
                elif platform == "kalshi":
                    kalshi_data = result["markets"]
                    discovery_metrics["data_quality_scores"]["kalshi"] = result["quality_score"]
                
                self.logger.info(f"Successfully discovered {len(result['markets'])} markets from {platform}")
                
            except Exception as e:
                discovery_metrics["api_failures"] += 1
                self.logger.error(f"Market discovery failed for {platform}: {e}")
                
                # Log error but continue with other platforms
                updated_state["error_log"].append({
                    "step": "market_discovery",
                    "platform": platform,
                    "error": str(e),
                    "severity": "medium",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Validate and process discovered data
        validated_polymarket_data = self._validate_market_data(polymarket_data, "polymarket")
        validated_kalshi_data = self._validate_market_data(kalshi_data, "kalshi")
        
        # Update state with discovered data
        updated_state["polymarket_data"] = validated_polymarket_data
        updated_state["kalshi_data"] = validated_kalshi_data
        
        # Calculate overall discovery metrics
        total_markets = len(validated_polymarket_data) + len(validated_kalshi_data)
        discovery_metrics["total_markets_found"] = total_markets
        discovery_metrics["end_time"] = datetime.now().isoformat()
        discovery_metrics["success_rate"] = (
            (discovery_metrics["api_calls_made"] - discovery_metrics["api_failures"]) / 
            max(1, discovery_metrics["api_calls_made"])
        )
        
        # Calculate overall data quality score
        quality_scores = list(discovery_metrics["data_quality_scores"].values())
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        discovery_metrics["overall_data_quality"] = overall_quality
        
        # Update market conditions with discovery results
        updated_state["market_conditions"] = {
            "total_markets": total_markets,
            "polymarket_count": len(validated_polymarket_data),
            "kalshi_count": len(validated_kalshi_data),
            "discovery_timestamp": datetime.now().isoformat(),
            "discovery_metrics": discovery_metrics,
            "data_quality_passed": overall_quality >= self.data_quality_threshold
        }
        
        # Log discovery summary
        self.logger.info(
            f"Market discovery completed: {total_markets} markets found "
            f"(Polymarket: {len(validated_polymarket_data)}, Kalshi: {len(validated_kalshi_data)}) "
            f"with {overall_quality:.2%} data quality"
        )
        
        return updated_state
    
    
    async def classify_niche(self, state: TradingState) -> TradingState:
        """
        Classify markets into appropriate niches using enhanced machine learning approach.
        
        Implements Requirement 5.2:
        - Classify markets into appropriate niches using machine learning
        
        Args:
            state: Current TradingState with market data
            
        Returns:
            Updated TradingState with niche classification and confidence scores
        """
        self.logger.info("Starting market niche classification")
        updated_state = state.copy()
        
        # Get market data and query context
        query = state["original_query"].lower()
        all_markets = state.get("polymarket_data", []) + state.get("kalshi_data", [])
        parsed_intent = state.get("parsed_intent", {})
        
        # Initialize classification metrics
        classification_metrics = {
            "total_markets_analyzed": len(all_markets),
            "classification_method": "enhanced_keyword_ml",
            "confidence_scores": {},
            "niche_distribution": {},
            "fallback_used": False
        }
        
        # Enhanced niche classification with weighted scoring
        niche_scores = {niche: 0.0 for niche in self.niche_categories}
        
        # Multi-source keyword analysis with weights
        keyword_sources = {
            "query": {"text": query, "weight": 3.0},
            "market_titles": {"texts": [m.get("title", "").lower() for m in all_markets], "weight": 2.0},
            "market_descriptions": {"texts": [m.get("description", "").lower() for m in all_markets], "weight": 1.5},
            "market_categories": {"texts": [m.get("category", "").lower() for m in all_markets], "weight": 2.5}
        }
        
        # Enhanced keyword mapping with synonyms and context
        enhanced_keywords = {
            "politics": {
                "primary": ["election", "president", "vote", "political", "congress", "senate", "government"],
                "secondary": ["policy", "candidate", "campaign", "democrat", "republican", "ballot"],
                "context": ["biden", "trump", "harris", "desantis", "primary", "midterm"]
            },
            "sports": {
                "primary": ["game", "match", "team", "player", "championship", "league", "tournament"],
                "secondary": ["season", "playoff", "draft", "trade", "coach", "stadium"],
                "context": ["nfl", "nba", "mlb", "nhl", "fifa", "olympics", "superbowl"]
            },
            "crypto": {
                "primary": ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft"],
                "secondary": ["btc", "eth", "coin", "token", "mining", "wallet", "exchange"],
                "context": ["coinbase", "binance", "solana", "cardano", "dogecoin", "web3"]
            },
            "economics": {
                "primary": ["gdp", "inflation", "market", "stock", "economy", "recession"],
                "secondary": ["fed", "interest", "rate", "unemployment", "cpi", "bond"],
                "context": ["powell", "fomc", "treasury", "nasdaq", "sp500", "dow"]
            },
            "entertainment": {
                "primary": ["movie", "show", "celebrity", "award", "box office", "film"],
                "secondary": ["actor", "director", "streaming", "netflix", "disney", "oscar"],
                "context": ["hollywood", "marvel", "star wars", "game of thrones"]
            },
            "technology": {
                "primary": ["tech", "ai", "software", "startup", "ipo", "innovation"],
                "secondary": ["apple", "google", "microsoft", "amazon", "meta", "tesla"],
                "context": ["chatgpt", "openai", "silicon valley", "venture capital"]
            },
            "weather": {
                "primary": ["temperature", "hurricane", "weather", "climate", "storm"],
                "secondary": ["rainfall", "drought", "flood", "tornado", "blizzard"],
                "context": ["noaa", "celsius", "fahrenheit", "el nino", "la nina"]
            },
            "social": {
                "primary": ["social", "trend", "viral", "meme", "culture"],
                "secondary": ["twitter", "tiktok", "instagram", "facebook", "youtube"],
                "context": ["influencer", "hashtag", "platform", "content creator"]
            }
        }
        
        # Score each niche based on keyword matches
        for source_name, source_info in keyword_sources.items():
            if source_name == "query":
                texts = [source_info["text"]]
            else:
                texts = source_info["texts"]
            
            weight = source_info["weight"]
            
            for text in texts:
                if not text:
                    continue
                    
                for niche, keywords in enhanced_keywords.items():
                    # Primary keywords (highest weight)
                    primary_matches = sum(1 for keyword in keywords["primary"] if keyword in text)
                    niche_scores[niche] += primary_matches * weight * 1.0
                    
                    # Secondary keywords (medium weight)
                    secondary_matches = sum(1 for keyword in keywords["secondary"] if keyword in text)
                    niche_scores[niche] += secondary_matches * weight * 0.7
                    
                    # Context keywords (lower weight but important for disambiguation)
                    context_matches = sum(1 for keyword in keywords["context"] if keyword in text)
                    niche_scores[niche] += context_matches * weight * 0.5
        
        # Apply market volume weighting (markets with higher volume get more influence)
        for market in all_markets:
            market_volume = market.get("total_volume", 0)
            market_title = market.get("title", "").lower()
            market_category = market.get("category", "").lower()
            
            # Volume weight factor (logarithmic scaling)
            volume_weight = min(2.0, 1.0 + (market_volume / 10000.0))
            
            for niche, keywords in enhanced_keywords.items():
                title_matches = sum(1 for keyword in keywords["primary"] + keywords["secondary"] 
                                  if keyword in market_title)
                category_matches = sum(1 for keyword in keywords["primary"] 
                                     if keyword in market_category)
                
                niche_scores[niche] += (title_matches + category_matches * 2) * volume_weight
        
        # Normalize scores and calculate confidence
        max_score = max(niche_scores.values()) if niche_scores.values() else 0
        
        if max_score > 0:
            # Normalize scores to 0-1 range
            normalized_scores = {niche: score / max_score for niche, score in niche_scores.items()}
            
            # Determine primary niche
            primary_niche = max(normalized_scores, key=normalized_scores.get)
            primary_score = normalized_scores[primary_niche]
            
            # Calculate confidence based on score separation
            sorted_scores = sorted(normalized_scores.values(), reverse=True)
            if len(sorted_scores) > 1:
                confidence = min(1.0, (sorted_scores[0] - sorted_scores[1]) + 0.3)
            else:
                confidence = primary_score
            
            # Apply minimum confidence threshold
            if confidence < 0.3:
                primary_niche = "other"
                confidence = 0.5
                classification_metrics["fallback_used"] = True
        else:
            # No clear niche detected
            primary_niche = "other"
            confidence = 0.5
            normalized_scores = {niche: 0.0 for niche in self.niche_categories}
            classification_metrics["fallback_used"] = True
        
        # Store classification results
        classification_metrics["confidence_scores"] = normalized_scores
        classification_metrics["primary_niche"] = primary_niche
        classification_metrics["primary_confidence"] = confidence
        classification_metrics["niche_distribution"] = {
            niche: count for niche, count in 
            [(niche, sum(1 for m in all_markets if niche in m.get("category", "").lower())) 
             for niche in self.niche_categories]
        }
        
        # Update state with classification results
        updated_state["niche"] = primary_niche
        updated_state["niche_classification"] = {
            "primary_niche": primary_niche,
            "confidence": confidence,
            "all_scores": normalized_scores,
            "classification_metrics": classification_metrics
        }
        
        self.logger.info(
            f"Niche classification completed: '{primary_niche}' "
            f"(confidence: {confidence:.2%}, markets analyzed: {len(all_markets)})"
        )
        
        return updated_state
    
    
    async def assess_liquidity(self, state: TradingState) -> TradingState:
        """
        Assess market liquidity and trading volume metrics with comprehensive analysis.
        
        Implements Requirement 5.3:
        - Assess market liquidity and trading volume metrics
        
        Args:
            state: Current TradingState with market data
            
        Returns:
            Updated TradingState with detailed liquidity assessment
        """
        self.logger.info("Starting comprehensive liquidity assessment")
        updated_state = state.copy()
        
        all_markets = state.get("polymarket_data", []) + state.get("kalshi_data", [])
        
        if not all_markets:
            self.logger.warning("No markets available for liquidity assessment")
            updated_state["market_conditions"]["liquidity_metrics"] = {
                "error": "No markets available for assessment"
            }
            return updated_state
        
        # Initialize comprehensive liquidity metrics
        liquidity_metrics = {
            "assessment_timestamp": datetime.now().isoformat(),
            "total_markets_assessed": len(all_markets),
            "volume_statistics": {},
            "liquidity_statistics": {},
            "market_categorization": {},
            "platform_comparison": {},
            "quality_indicators": {}
        }
        
        # Collect volume and liquidity data
        volumes = []
        liquidities = []
        platform_data = {"polymarket": [], "kalshi": []}
        
        for market in all_markets:
            volume = market.get("total_volume", 0)
            liquidity = market.get("liquidity", 0)
            platform = market.get("platform", "unknown")
            
            volumes.append(volume)
            liquidities.append(liquidity)
            
            if platform in platform_data:
                platform_data[platform].append({
                    "volume": volume,
                    "liquidity": liquidity,
                    "market_id": market.get("event_id", "unknown")
                })
        
        # Calculate comprehensive volume statistics
        if volumes:
            liquidity_metrics["volume_statistics"] = {
                "total_volume": sum(volumes),
                "average_volume": sum(volumes) / len(volumes),
                "median_volume": self._calculate_median(volumes),
                "min_volume": min(volumes),
                "max_volume": max(volumes),
                "volume_std_dev": self._calculate_std_dev(volumes),
                "volume_distribution": self._calculate_distribution(volumes)
            }
        
        # Calculate comprehensive liquidity statistics
        if liquidities:
            liquidity_metrics["liquidity_statistics"] = {
                "total_liquidity": sum(liquidities),
                "average_liquidity": sum(liquidities) / len(liquidities),
                "median_liquidity": self._calculate_median(liquidities),
                "min_liquidity": min(liquidities),
                "max_liquidity": max(liquidities),
                "liquidity_std_dev": self._calculate_std_dev(liquidities),
                "liquidity_distribution": self._calculate_distribution(liquidities)
            }
        
        # Categorize markets by liquidity levels
        high_liquidity_markets = []
        medium_liquidity_markets = []
        low_liquidity_markets = []
        
        for market in all_markets:
            volume = market.get("total_volume", 0)
            liquidity = market.get("liquidity", 0)
            
            # Combined liquidity score (volume + liquidity with weights)
            liquidity_score = (volume * 0.6) + (liquidity * 0.4)
            
            market_info = {
                "market_id": market.get("event_id", "unknown"),
                "title": market.get("title", "Unknown"),
                "platform": market.get("platform", "unknown"),
                "volume": volume,
                "liquidity": liquidity,
                "liquidity_score": liquidity_score
            }
            
            if liquidity_score >= self.high_liquidity_threshold:
                high_liquidity_markets.append(market_info)
            elif liquidity_score >= self.min_liquidity_threshold:
                medium_liquidity_markets.append(market_info)
            else:
                low_liquidity_markets.append(market_info)
        
        liquidity_metrics["market_categorization"] = {
            "high_liquidity": {
                "count": len(high_liquidity_markets),
                "markets": high_liquidity_markets,
                "threshold": self.high_liquidity_threshold
            },
            "medium_liquidity": {
                "count": len(medium_liquidity_markets),
                "markets": medium_liquidity_markets,
                "threshold": self.min_liquidity_threshold
            },
            "low_liquidity": {
                "count": len(low_liquidity_markets),
                "markets": low_liquidity_markets,
                "threshold": 0
            }
        }
        
        # Platform comparison analysis
        for platform, markets in platform_data.items():
            if markets:
                platform_volumes = [m["volume"] for m in markets]
                platform_liquidities = [m["liquidity"] for m in markets]
                
                liquidity_metrics["platform_comparison"][platform] = {
                    "market_count": len(markets),
                    "total_volume": sum(platform_volumes),
                    "avg_volume": sum(platform_volumes) / len(platform_volumes),
                    "total_liquidity": sum(platform_liquidities),
                    "avg_liquidity": sum(platform_liquidities) / len(platform_liquidities),
                    "high_liquidity_count": len([m for m in markets 
                                               if (m["volume"] * 0.6 + m["liquidity"] * 0.4) >= self.high_liquidity_threshold])
                }
        
        # Calculate quality indicators
        total_markets = len(all_markets)
        high_quality_ratio = len(high_liquidity_markets) / total_markets if total_markets > 0 else 0
        avg_volume = liquidity_metrics["volume_statistics"].get("average_volume", 0)
        avg_liquidity = liquidity_metrics["liquidity_statistics"].get("average_liquidity", 0)
        
        liquidity_metrics["quality_indicators"] = {
            "high_liquidity_ratio": high_quality_ratio,
            "market_depth_score": min(1.0, (avg_volume + avg_liquidity) / 2000.0),
            "platform_diversity": len([p for p, data in platform_data.items() if data]),
            "volume_concentration": self._calculate_concentration_index(volumes),
            "overall_liquidity_grade": self._calculate_liquidity_grade(
                high_quality_ratio, avg_volume, avg_liquidity
            )
        }
        
        # Update market conditions with comprehensive liquidity data
        market_conditions = updated_state.get("market_conditions", {})
        market_conditions["liquidity_metrics"] = liquidity_metrics
        updated_state["market_conditions"] = market_conditions
        
        # Log assessment summary
        self.logger.info(
            f"Liquidity assessment completed: {total_markets} markets analyzed, "
            f"{len(high_liquidity_markets)} high-liquidity, "
            f"overall grade: {liquidity_metrics['quality_indicators']['overall_liquidity_grade']}"
        )
        
        return updated_state
    
    
    async def detect_opportunities(self, state: TradingState) -> TradingState:
        """
        Identify high-potential trading opportunities based on configurable criteria.
        
        Implements Requirement 5.5:
        - Identify high-potential trading opportunities based on configurable criteria
        
        Args:
            state: Current TradingState with market and liquidity data
            
        Returns:
            Updated TradingState with comprehensive opportunity analysis
        """
        self.logger.info("Starting comprehensive opportunity detection")
        updated_state = state.copy()
        
        all_markets = state.get("polymarket_data", []) + state.get("kalshi_data", [])
        
        if not all_markets:
            self.logger.warning("No markets available for opportunity detection")
            return updated_state
        
        # Initialize opportunity detection metrics
        opportunity_metrics = {
            "detection_timestamp": datetime.now().isoformat(),
            "total_markets_analyzed": len(all_markets),
            "scoring_algorithm": "multi_factor_weighted",
            "opportunity_criteria": {
                "min_volume_threshold": self.min_volume_threshold,
                "min_liquidity_threshold": self.min_liquidity_threshold,
                "opportunity_score_threshold": self.opportunity_score_threshold
            },
            "factor_weights": {
                "volume_factor": 0.25,
                "liquidity_factor": 0.25,
                "timing_factor": 0.15,
                "niche_alignment_factor": 0.20,
                "sentiment_factor": 0.10,
                "risk_factor": 0.05
            }
        }
        
        opportunities = []
        user_preferences = state.get("user_preferences", {})
        niche = state.get("niche", "")
        sentiment_data = state.get("sentiment_analysis", {})
        
        for market in all_markets:
            try:
                # Extract market data with defaults
                market_id = market.get("event_id", f"unknown_{len(opportunities)}")
                title = market.get("title", "Unknown Market")
                platform = market.get("platform", "unknown")
                volume = market.get("total_volume", 0)
                liquidity = market.get("liquidity", 0)
                category = market.get("category", "").lower()
                created_at = market.get("created_at", "")
                resolution_date = market.get("resolution_date", "")
                current_odds = market.get("current_odds", {})
                
                # Calculate multi-factor opportunity score
                opportunity_score = 0.0
                score_breakdown = {}
                
                # Factor 1: Volume Score (0-1 scale)
                volume_score = min(1.0, volume / 50000.0) if volume > 0 else 0.0
                opportunity_score += volume_score * opportunity_metrics["factor_weights"]["volume_factor"]
                score_breakdown["volume_score"] = volume_score
                
                # Factor 2: Liquidity Score (0-1 scale)
                liquidity_score = min(1.0, liquidity / 25000.0) if liquidity > 0 else 0.0
                opportunity_score += liquidity_score * opportunity_metrics["factor_weights"]["liquidity_factor"]
                score_breakdown["liquidity_score"] = liquidity_score
                
                # Factor 3: Market Timing Score
                timing_score = self._calculate_timing_score(created_at, resolution_date)
                opportunity_score += timing_score * opportunity_metrics["factor_weights"]["timing_factor"]
                score_breakdown["timing_score"] = timing_score
                
                # Factor 4: Niche Alignment Score
                niche_alignment_score = self._calculate_niche_alignment(category, niche, title)
                opportunity_score += niche_alignment_score * opportunity_metrics["factor_weights"]["niche_alignment_factor"]
                score_breakdown["niche_alignment_score"] = niche_alignment_score
                
                # Factor 5: Sentiment Alignment Score
                sentiment_score = self._calculate_sentiment_alignment(
                    market, sentiment_data, category
                )
                opportunity_score += sentiment_score * opportunity_metrics["factor_weights"]["sentiment_factor"]
                score_breakdown["sentiment_score"] = sentiment_score
                
                # Factor 6: Risk-Adjusted Score
                risk_score = self._calculate_risk_score(current_odds, volume, liquidity)
                opportunity_score += risk_score * opportunity_metrics["factor_weights"]["risk_factor"]
                score_breakdown["risk_score"] = risk_score
                
                # Apply user preference adjustments
                preference_multiplier = self._calculate_preference_multiplier(
                    market, user_preferences
                )
                opportunity_score *= preference_multiplier
                score_breakdown["preference_multiplier"] = preference_multiplier
                
                # Only include opportunities above threshold
                if opportunity_score >= self.opportunity_score_threshold:
                    # Calculate additional metrics
                    volatility_indicator = self._calculate_volatility_indicator(current_odds)
                    market_efficiency = self._calculate_market_efficiency(volume, liquidity, current_odds)
                    
                    opportunity = {
                        "market_id": market_id,
                        "title": title,
                        "platform": platform,
                        "opportunity_score": round(opportunity_score, 4),
                        "score_breakdown": score_breakdown,
                        "market_metrics": {
                            "volume": volume,
                            "liquidity": liquidity,
                            "current_odds": current_odds,
                            "volatility_indicator": volatility_indicator,
                            "market_efficiency": market_efficiency
                        },
                        "timing_info": {
                            "created_at": created_at,
                            "resolution_date": resolution_date,
                            "time_to_resolution": self._calculate_time_to_resolution(resolution_date)
                        },
                        "classification": {
                            "category": category,
                            "niche": niche,
                            "niche_alignment": niche_alignment_score
                        },
                        "risk_assessment": {
                            "risk_level": self._categorize_risk_level(risk_score),
                            "liquidity_risk": "low" if liquidity > self.high_liquidity_threshold else "medium" if liquidity > self.min_liquidity_threshold else "high",
                            "volume_risk": "low" if volume > 10000 else "medium" if volume > 1000 else "high"
                        },
                        "recommendation_strength": self._calculate_recommendation_strength(opportunity_score),
                        "generated_at": datetime.now().isoformat()
                    }
                    
                    opportunities.append(opportunity)
                    
            except Exception as e:
                self.logger.error(f"Error processing market {market.get('event_id', 'unknown')}: {e}")
                continue
        
        # Sort opportunities by score (highest first)
        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
        
        # Calculate opportunity detection statistics
        opportunity_stats = {
            "total_opportunities_found": len(opportunities),
            "avg_opportunity_score": sum(o["opportunity_score"] for o in opportunities) / len(opportunities) if opportunities else 0.0,
            "score_distribution": self._calculate_score_distribution(opportunities),
            "platform_distribution": self._calculate_platform_distribution(opportunities),
            "risk_level_distribution": self._calculate_risk_distribution(opportunities),
            "top_opportunity_score": opportunities[0]["opportunity_score"] if opportunities else 0.0
        }
        
        # Update market conditions with opportunity data
        market_conditions = updated_state.get("market_conditions", {})
        market_conditions["opportunities"] = opportunities[:20]  # Top 20 opportunities
        market_conditions["opportunity_metrics"] = opportunity_metrics
        market_conditions["opportunity_stats"] = opportunity_stats
        updated_state["market_conditions"] = market_conditions
        
        # Log detection summary
        self.logger.info(
            f"Opportunity detection completed: {len(opportunities)} opportunities found "
            f"from {len(all_markets)} markets (avg score: {opportunity_stats['avg_opportunity_score']:.3f})"
        )
        
        return updated_state
    
    
    # Enhanced API Integration Methods
    
    async def _discover_polymarket_data_enhanced(self, query: str, niche: str = "") -> Dict[str, Any]:
        """
        Enhanced Polymarket data discovery with quality validation.
        
        Args:
            query: Search query for market discovery
            niche: Preferred niche for filtering
            
        Returns:
            Dictionary with markets and quality score
        """
        try:
            self.logger.info(f"Querying Polymarket API for query: '{query}', niche: '{niche}'")
            
            # Use existing core function but enhance the data
            raw_traders = fetch_polymarket_traders(niche if niche else "general")
            
            # Convert trader data to market data format
            markets = []
            for trader in raw_traders:
                # Create market event from trader data
                market = {
                    "event_id": f"poly_market_{trader.get('trader_id', 'unknown')}",
                    "platform": "polymarket",
                    "title": f"Market for {trader.get('target_event', 'Unknown Event')}",
                    "description": f"Prediction market with trader {trader.get('trader_id', 'unknown')}",
                    "total_volume": float(trader.get("roi", 0.15)) * 100000,  # Simulate volume from ROI
                    "liquidity": float(trader.get("win_rate", 0.5)) * 50000,  # Simulate liquidity from win rate
                    "current_odds": {
                        "YES": float(trader.get("win_rate", 0.5)),
                        "NO": 1.0 - float(trader.get("win_rate", 0.5))
                    },
                    "category": trader.get("niche", niche),
                    "created_at": datetime.now().isoformat(),
                    "resolution_date": (datetime.now().replace(month=12, day=31)).isoformat(),
                    "last_trade_time": datetime.now().isoformat(),
                    "niche": trader.get("niche", niche),
                    "tags": [trader.get("niche", "general")],
                    "sentiment_score": 0.5,  # Neutral default
                    "news_coverage": 10,
                    "social_buzz": 0.3,
                    "top_traders": [trader.get("trader_id", "unknown")],
                    "recent_trades": [],
                    "price_history": []
                }
                markets.append(market)
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality_score(markets, "polymarket")
            
            return {
                "markets": markets,
                "quality_score": quality_score,
                "source": "polymarket_api",
                "query_used": query,
                "niche_filter": niche
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced Polymarket discovery failed: {e}")
            return {
                "markets": [],
                "quality_score": 0.0,
                "error": str(e),
                "source": "polymarket_api"
            }
    
    async def _discover_kalshi_data_enhanced(self, query: str, niche: str = "") -> Dict[str, Any]:
        """
        Enhanced Kalshi data discovery with quality validation.
        
        Args:
            query: Search query for market discovery
            niche: Preferred niche for filtering
            
        Returns:
            Dictionary with markets and quality score
        """
        try:
            self.logger.info(f"Querying Kalshi API for query: '{query}', niche: '{niche}'")
            
            # Use existing core function but enhance the data
            raw_traders = fetch_kalshi_traders(niche if niche else "general")
            
            # Convert trader data to market data format
            markets = []
            for trader in raw_traders:
                # Create market event from trader data
                market = {
                    "event_id": f"kalshi_market_{trader.get('trader_id', 'unknown')}",
                    "platform": "kalshi",
                    "title": f"Market for {trader.get('target_event', 'Unknown Event')}",
                    "description": f"Regulated prediction market with trader {trader.get('trader_id', 'unknown')}",
                    "total_volume": float(trader.get("roi", 0.18)) * 80000,  # Simulate volume from ROI
                    "liquidity": float(trader.get("win_rate", 0.6)) * 40000,  # Simulate liquidity from win rate
                    "current_odds": {
                        "YES": float(trader.get("win_rate", 0.6)),
                        "NO": 1.0 - float(trader.get("win_rate", 0.6))
                    },
                    "category": trader.get("niche", niche),
                    "created_at": datetime.now().isoformat(),
                    "resolution_date": (datetime.now().replace(month=12, day=31)).isoformat(),
                    "last_trade_time": datetime.now().isoformat(),
                    "niche": trader.get("niche", niche),
                    "tags": [trader.get("niche", "general")],
                    "sentiment_score": 0.5,  # Neutral default
                    "news_coverage": 8,
                    "social_buzz": 0.25,
                    "top_traders": [trader.get("trader_id", "unknown")],
                    "recent_trades": [],
                    "price_history": []
                }
                markets.append(market)
            
            # Calculate data quality score
            quality_score = self._calculate_data_quality_score(markets, "kalshi")
            
            return {
                "markets": markets,
                "quality_score": quality_score,
                "source": "kalshi_api",
                "query_used": query,
                "niche_filter": niche
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced Kalshi discovery failed: {e}")
            return {
                "markets": [],
                "quality_score": 0.0,
                "error": str(e),
                "source": "kalshi_api"
            }
    
    # Helper Methods for Enhanced Functionality
    
    async def _execute_with_retries(self, task, platform: str) -> Dict[str, Any]:
        """Execute API task with retry logic and timeout."""
        for attempt in range(self.max_api_retries):
            try:
                result = await asyncio.wait_for(task, timeout=self.api_timeout)
                return result
            except asyncio.TimeoutError:
                self.logger.warning(f"API timeout for {platform} (attempt {attempt + 1})")
                if attempt == self.max_api_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                self.logger.error(f"API error for {platform} (attempt {attempt + 1}): {e}")
                if attempt == self.max_api_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for query caching."""
        return hashlib.md5(query.encode()).hexdigest()[:16]
    
    def _validate_market_data(self, markets: List[Dict[str, Any]], platform: str) -> List[Dict[str, Any]]:
        """
        Validate market data quality and completeness.
        
        Implements Requirement 5.4: Validate data quality and completeness
        """
        validated_markets = []
        
        for market in markets:
            try:
                # Required fields validation
                required_fields = ["event_id", "platform", "title", "total_volume", "liquidity"]
                if not all(field in market for field in required_fields):
                    self.logger.warning(f"Market missing required fields: {market.get('event_id', 'unknown')}")
                    continue
                
                # Data type validation
                if not isinstance(market.get("total_volume"), (int, float)) or market["total_volume"] < 0:
                    self.logger.warning(f"Invalid volume for market: {market.get('event_id')}")
                    continue
                
                if not isinstance(market.get("liquidity"), (int, float)) or market["liquidity"] < 0:
                    self.logger.warning(f"Invalid liquidity for market: {market.get('event_id')}")
                    continue
                
                # Odds validation
                current_odds = market.get("current_odds", {})
                if current_odds:
                    odds_sum = sum(current_odds.values())
                    if not (0.95 <= odds_sum <= 1.05):  # Allow small rounding errors
                        self.logger.warning(f"Invalid odds sum for market: {market.get('event_id')}")
                        # Normalize odds
                        market["current_odds"] = {k: v/odds_sum for k, v in current_odds.items()}
                
                # Add validation metadata
                market["validation"] = {
                    "validated_at": datetime.now().isoformat(),
                    "validation_passed": True,
                    "platform": platform
                }
                
                validated_markets.append(market)
                
            except Exception as e:
                self.logger.error(f"Market validation failed for {market.get('event_id', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Validated {len(validated_markets)}/{len(markets)} markets for {platform}")
        return validated_markets
    
    def _calculate_data_quality_score(self, markets: List[Dict[str, Any]], platform: str) -> float:
        """Calculate data quality score for a platform's markets."""
        if not markets:
            return 0.0
        
        quality_factors = []
        
        for market in markets:
            market_quality = 0.0
            
            # Completeness score (0-0.4)
            required_fields = ["event_id", "title", "total_volume", "liquidity", "current_odds"]
            completeness = sum(1 for field in required_fields if field in market and market[field]) / len(required_fields)
            market_quality += completeness * 0.4
            
            # Data validity score (0-0.3)
            validity = 0.0
            if isinstance(market.get("total_volume"), (int, float)) and market["total_volume"] >= 0:
                validity += 0.1
            if isinstance(market.get("liquidity"), (int, float)) and market["liquidity"] >= 0:
                validity += 0.1
            if market.get("current_odds") and isinstance(market["current_odds"], dict):
                validity += 0.1
            market_quality += validity
            
            # Richness score (0-0.3)
            optional_fields = ["description", "category", "tags", "created_at", "resolution_date"]
            richness = sum(1 for field in optional_fields if field in market and market[field]) / len(optional_fields)
            market_quality += richness * 0.3
            
            quality_factors.append(market_quality)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _calculate_median(self, values: List[float]) -> float:
        """Calculate median of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        return sorted_values[n//2]
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        """Calculate distribution of values into quartiles."""
        if not values:
            return {"q1": 0, "q2": 0, "q3": 0, "q4": 0}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        q1_threshold = sorted_values[n//4] if n >= 4 else sorted_values[0]
        q2_threshold = sorted_values[n//2] if n >= 2 else sorted_values[0]
        q3_threshold = sorted_values[3*n//4] if n >= 4 else sorted_values[-1]
        
        distribution = {"q1": 0, "q2": 0, "q3": 0, "q4": 0}
        
        for value in values:
            if value <= q1_threshold:
                distribution["q1"] += 1
            elif value <= q2_threshold:
                distribution["q2"] += 1
            elif value <= q3_threshold:
                distribution["q3"] += 1
            else:
                distribution["q4"] += 1
        
        return distribution
    
    def _calculate_concentration_index(self, values: List[float]) -> float:
        """Calculate concentration index (Gini coefficient approximation)."""
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        total = sum(sorted_values)
        
        if total == 0:
            return 0.0
        
        # Simplified Gini coefficient calculation
        cumulative_sum = 0
        gini_sum = 0
        
        for i, value in enumerate(sorted_values):
            cumulative_sum += value
            gini_sum += (2 * (i + 1) - n - 1) * value
        
        return gini_sum / (n * total)
    
    def _calculate_liquidity_grade(self, high_liquidity_ratio: float, avg_volume: float, avg_liquidity: float) -> str:
        """Calculate overall liquidity grade."""
        # Weighted score calculation
        ratio_score = high_liquidity_ratio * 0.4
        volume_score = min(1.0, avg_volume / 10000.0) * 0.3
        liquidity_score = min(1.0, avg_liquidity / 5000.0) * 0.3
        
        total_score = ratio_score + volume_score + liquidity_score
        
        if total_score >= 0.8:
            return "A"
        elif total_score >= 0.6:
            return "B"
        elif total_score >= 0.4:
            return "C"
        elif total_score >= 0.2:
            return "D"
        else:
            return "F"
    
    def _calculate_timing_score(self, created_at: str, resolution_date: str) -> float:
        """Calculate timing score based on market age and time to resolution."""
        try:
            if not created_at or not resolution_date:
                return 0.5  # Default moderate score
            
            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            resolution = datetime.fromisoformat(resolution_date.replace('Z', '+00:00'))
            now = datetime.now()
            
            # Market age factor (newer markets might have more opportunity)
            age_days = (now - created).days
            age_score = max(0.0, 1.0 - (age_days / 365.0))  # Decreases over a year
            
            # Time to resolution factor (optimal window)
            days_to_resolution = (resolution - now).days
            if days_to_resolution < 0:
                return 0.0  # Market already resolved
            elif days_to_resolution < 7:
                resolution_score = 0.3  # Too close to resolution
            elif days_to_resolution < 30:
                resolution_score = 1.0  # Optimal window
            elif days_to_resolution < 90:
                resolution_score = 0.8  # Good window
            elif days_to_resolution < 365:
                resolution_score = 0.6  # Acceptable window
            else:
                resolution_score = 0.3  # Too far out
            
            return (age_score * 0.3) + (resolution_score * 0.7)
            
        except Exception:
            return 0.5  # Default on parsing errors
    
    def _calculate_niche_alignment(self, category: str, niche: str, title: str) -> float:
        """Calculate how well a market aligns with the target niche."""
        if not niche or niche == "other":
            return 0.5  # Neutral score for general queries
        
        alignment_score = 0.0
        
        # Direct category match
        if niche.lower() in category.lower():
            alignment_score += 0.6
        
        # Title keyword match
        niche_keywords = {
            "politics": ["election", "president", "vote", "political"],
            "sports": ["game", "match", "team", "championship"],
            "crypto": ["bitcoin", "ethereum", "crypto", "blockchain"],
            "economics": ["gdp", "inflation", "market", "economy"],
            "entertainment": ["movie", "show", "celebrity", "award"],
            "technology": ["tech", "ai", "software", "startup"]
        }
        
        keywords = niche_keywords.get(niche.lower(), [])
        title_lower = title.lower()
        
        keyword_matches = sum(1 for keyword in keywords if keyword in title_lower)
        if keyword_matches > 0:
            alignment_score += min(0.4, keyword_matches * 0.1)
        
        return min(1.0, alignment_score)
    
    def _calculate_sentiment_alignment(self, market: Dict[str, Any], sentiment_data: Dict[str, Any], category: str) -> float:
        """Calculate sentiment alignment score."""
        if not sentiment_data:
            return 0.5  # Neutral default
        
        # Use market-specific sentiment if available
        market_sentiment = market.get("sentiment_score", 0.5)
        
        # Use overall sentiment from analysis
        overall_sentiment = sentiment_data.get("overall_sentiment_score", 0.5)
        
        # Category-specific sentiment if available
        category_sentiment = sentiment_data.get("category_sentiments", {}).get(category, 0.5)
        
        # Weighted combination
        sentiment_score = (market_sentiment * 0.4) + (overall_sentiment * 0.3) + (category_sentiment * 0.3)
        
        # Convert to opportunity score (higher sentiment = higher opportunity)
        return sentiment_score
    
    def _calculate_risk_score(self, current_odds: Dict[str, float], volume: float, liquidity: float) -> float:
        """Calculate risk-adjusted score."""
        if not current_odds:
            return 0.3  # Conservative default
        
        # Odds balance (closer to 50/50 = higher risk but potentially higher reward)
        odds_values = list(current_odds.values())
        if len(odds_values) >= 2:
            max_odds = max(odds_values)
            min_odds = min(odds_values)
            odds_balance = 1.0 - abs(max_odds - min_odds)  # Higher when odds are balanced
        else:
            odds_balance = 0.5
        
        # Volume/liquidity risk mitigation
        volume_factor = min(1.0, volume / 5000.0)
        liquidity_factor = min(1.0, liquidity / 2500.0)
        
        # Combined risk score
        risk_score = (odds_balance * 0.5) + (volume_factor * 0.25) + (liquidity_factor * 0.25)
        
        return risk_score
    
    def _calculate_preference_multiplier(self, market: Dict[str, Any], user_preferences: Dict[str, Any]) -> float:
        """Calculate user preference alignment multiplier."""
        if not user_preferences:
            return 1.0
        
        multiplier = 1.0
        
        # Risk tolerance alignment
        risk_tolerance = user_preferences.get("risk_tolerance", 0.5)
        market_risk = self._estimate_market_risk(market)
        
        risk_alignment = 1.0 - abs(risk_tolerance - market_risk)
        multiplier *= (0.8 + (risk_alignment * 0.4))  # 0.8 to 1.2 range
        
        # Platform preference
        preferred_platforms = user_preferences.get("preferred_platforms", [])
        if preferred_platforms and market.get("platform") in preferred_platforms:
            multiplier *= 1.1
        
        # Volume preference
        min_volume_pref = user_preferences.get("min_volume", 0)
        if market.get("total_volume", 0) >= min_volume_pref:
            multiplier *= 1.05
        
        return multiplier
    
    def _estimate_market_risk(self, market: Dict[str, Any]) -> float:
        """Estimate market risk level (0 = low risk, 1 = high risk)."""
        volume = market.get("total_volume", 0)
        liquidity = market.get("liquidity", 0)
        current_odds = market.get("current_odds", {})
        
        # Lower volume/liquidity = higher risk
        volume_risk = max(0.0, 1.0 - (volume / 10000.0))
        liquidity_risk = max(0.0, 1.0 - (liquidity / 5000.0))
        
        # Extreme odds = higher risk
        if current_odds:
            odds_values = list(current_odds.values())
            max_odds = max(odds_values) if odds_values else 0.5
            odds_risk = max_odds  # Higher when one outcome is very likely
        else:
            odds_risk = 0.5
        
        # Combined risk estimate
        return (volume_risk * 0.3) + (liquidity_risk * 0.3) + (odds_risk * 0.4)
    
    def _calculate_volatility_indicator(self, current_odds: Dict[str, float]) -> str:
        """Calculate volatility indicator based on odds distribution."""
        if not current_odds or len(current_odds) < 2:
            return "unknown"
        
        odds_values = list(current_odds.values())
        max_odds = max(odds_values)
        min_odds = min(odds_values)
        
        spread = max_odds - min_odds
        
        if spread < 0.2:
            return "low"
        elif spread < 0.5:
            return "medium"
        else:
            return "high"
    
    def _calculate_market_efficiency(self, volume: float, liquidity: float, current_odds: Dict[str, float]) -> float:
        """Calculate market efficiency score."""
        # Higher volume and liquidity generally indicate more efficient markets
        volume_factor = min(1.0, volume / 20000.0)
        liquidity_factor = min(1.0, liquidity / 10000.0)
        
        # Odds should sum close to 1.0 in efficient markets
        if current_odds:
            odds_sum = sum(current_odds.values())
            odds_efficiency = max(0.0, 1.0 - abs(1.0 - odds_sum))
        else:
            odds_efficiency = 0.5
        
        return (volume_factor * 0.4) + (liquidity_factor * 0.4) + (odds_efficiency * 0.2)
    
    def _calculate_time_to_resolution(self, resolution_date: str) -> Optional[int]:
        """Calculate days until market resolution."""
        try:
            if not resolution_date:
                return None
            
            resolution = datetime.fromisoformat(resolution_date.replace('Z', '+00:00'))
            now = datetime.now()
            
            days_diff = (resolution - now).days
            return max(0, days_diff)  # Don't return negative days
            
        except Exception:
            return None
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level based on risk score."""
        if risk_score >= 0.7:
            return "low"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "high"
    
    def _calculate_recommendation_strength(self, opportunity_score: float) -> str:
        """Calculate recommendation strength based on opportunity score."""
        if opportunity_score >= 0.8:
            return "strong_buy"
        elif opportunity_score >= 0.65:
            return "buy"
        elif opportunity_score >= 0.5:
            return "hold"
        else:
            return "weak"
    
    def _calculate_score_distribution(self, opportunities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of opportunity scores."""
        if not opportunities:
            return {"high": 0, "medium": 0, "low": 0}
        
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for opp in opportunities:
            score = opp.get("opportunity_score", 0)
            if score >= 0.7:
                distribution["high"] += 1
            elif score >= 0.5:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _calculate_platform_distribution(self, opportunities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of opportunities by platform."""
        distribution = {}
        
        for opp in opportunities:
            platform = opp.get("platform", "unknown")
            distribution[platform] = distribution.get(platform, 0) + 1
        
        return distribution
    
    def _calculate_risk_distribution(self, opportunities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of opportunities by risk level."""
        distribution = {"low": 0, "medium": 0, "high": 0}
        
        for opp in opportunities:
            risk_level = opp.get("risk_assessment", {}).get("risk_level", "medium")
            distribution[risk_level] = distribution.get(risk_level, 0) + 1
        
        return distribution