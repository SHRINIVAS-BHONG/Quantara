"""
Central state management for the LangGraph trading system.

This module defines the TradingState TypedDict that serves as the central
state object for all graph operations, maintaining workflow context and data
consistency across multi-agent interactions.

Enhanced with comprehensive validation, consistency checks, and debugging support
for robust multi-agent operations with checkpointing capabilities.
"""

from typing import TypedDict, List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import hashlib
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor


class WorkflowStep(Enum):
    """Enumeration of valid workflow steps for state management."""
    INITIALIZATION = "initialization"
    MARKET_DISCOVERY = "market_discovery"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TRADER_ANALYSIS = "trader_analysis"
    RECOMMENDATION_GENERATION = "recommendation_generation"
    MONITORING_SETUP = "monitoring_setup"
    COMPLETED = "completed"
    ERROR = "error"


class StateValidationError(Exception):
    """Exception raised when state validation fails."""
    pass


class StateConsistencyError(Exception):
    """Exception raised when state consistency checks fail."""
    pass


@dataclass
class StateTransition:
    """Represents a state transition with metadata for debugging."""
    from_step: str
    to_step: str
    timestamp: datetime
    agent_id: str
    operation: str
    data_hash: str
    validation_passed: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "from_step": self.from_step,
            "to_step": self.to_step,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "operation": self.operation,
            "data_hash": self.data_hash,
            "validation_passed": self.validation_passed
        }


@dataclass
class StateDebugInfo:
    """Debug information for state management."""
    state_version: int = 0
    last_modified: datetime = field(default_factory=datetime.now)
    modification_count: int = 0
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    transition_log: List[StateTransition] = field(default_factory=list)
    consistency_checks: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_validation_result(self, validator: str, passed: bool, details: str = ""):
        """Add validation result to history."""
        self.validation_history.append({
            "validator": validator,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_transition(self, transition: StateTransition):
        """Add state transition to log."""
        self.transition_log.append(transition)
        self.modification_count += 1
        self.last_modified = datetime.now()
        self.state_version += 1
    
    def get_recent_transitions(self, count: int = 10) -> List[StateTransition]:
        """Get recent state transitions."""
        return self.transition_log[-count:] if self.transition_log else []


class TradingState(TypedDict):
    """
    Central state object for the LangGraph trading system.
    
    This TypedDict maintains all workflow context and data across graph operations,
    enabling stateful execution and multi-agent coordination with comprehensive
    validation and consistency checking.
    """
    # Query Context
    original_query: str
    parsed_intent: Dict[str, Any]
    niche: str
    platform_preference: str
    
    # Market Data
    polymarket_data: List[Dict[str, Any]]
    kalshi_data: List[Dict[str, Any]]
    market_conditions: Dict[str, Any]
    
    # Analysis Results
    trader_scores: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    
    # Workflow Control
    current_step: str
    completed_steps: List[str]
    error_log: List[Dict[str, Any]]
    state_checkpoint_id: Optional[str]  # Renamed to avoid reserved name conflict
    
    # Memory & Learning
    rag_context: List[Dict[str, Any]]
    historical_performance: Dict[str, Any]
    user_preferences: Dict[str, Any]
    
    # Output
    recommendations: List[Dict[str, Any]]
    confidence_score: float
    explanation: str
    
    # Enhanced State Management (Optional fields for backward compatibility)
    _debug_info: Optional[StateDebugInfo]
    _state_lock: Optional[str]  # For concurrent access control
    _validation_rules: Optional[Dict[str, Callable]]
    _consistency_validators: Optional[List[Callable]]


@dataclass
class StateManager:
    """
    Comprehensive state manager for TradingState with validation and consistency checks.
    
    Provides thread-safe state operations, validation, consistency checking,
    and debugging support for multi-agent operations.
    """
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=4))
    _active_locks: Dict[str, asyncio.Lock] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize state manager with default validators."""
        self._setup_default_validators()
    
    def _setup_default_validators(self):
        """Setup default validation rules and consistency validators."""
        # This would be expanded with actual validation logic
        pass
    
    async def validate_state(self, state: TradingState, strict: bool = True) -> bool:
        """
        Comprehensive state validation with configurable strictness.
        
        Args:
            state: TradingState to validate
            strict: If True, applies strict validation rules
            
        Returns:
            True if state is valid, False otherwise
            
        Raises:
            StateValidationError: If strict=True and validation fails
        """
        validation_results = []
        
        try:
            # Basic structure validation
            basic_valid = self._validate_basic_structure(state)
            validation_results.append(("basic_structure", basic_valid))
            
            # Data type validation
            types_valid = self._validate_data_types(state)
            validation_results.append(("data_types", types_valid))
            
            # Business logic validation
            business_valid = self._validate_business_logic(state)
            validation_results.append(("business_logic", business_valid))
            
            # Workflow consistency validation
            workflow_valid = self._validate_workflow_consistency(state)
            validation_results.append(("workflow_consistency", workflow_valid))
            
            # Update debug info if available
            if state.get("_debug_info"):
                for validator_name, result in validation_results:
                    state["_debug_info"].add_validation_result(validator_name, result)
            
            all_valid = all(result for _, result in validation_results)
            
            if strict and not all_valid:
                failed_validators = [name for name, result in validation_results if not result]
                raise StateValidationError(f"State validation failed for: {failed_validators}")
            
            return all_valid
            
        except Exception as e:
            self._logger.error(f"State validation error: {e}")
            if strict:
                raise
            return False
    
    def _validate_basic_structure(self, state: TradingState) -> bool:
        """Validate basic state structure and required fields."""
        required_fields = [
            "original_query", "current_step", "completed_steps",
            "trader_scores", "recommendations", "confidence_score"
        ]
        
        try:
            for field in required_fields:
                if field not in state:
                    self._logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate field types
            if not isinstance(state["original_query"], str):
                return False
            if not isinstance(state["completed_steps"], list):
                return False
            if not isinstance(state["trader_scores"], list):
                return False
            if not isinstance(state["recommendations"], list):
                return False
            if not isinstance(state["confidence_score"], (int, float)):
                return False
                
            return True
            
        except (KeyError, TypeError) as e:
            self._logger.error(f"Basic structure validation failed: {e}")
            return False
    
    def _validate_data_types(self, state: TradingState) -> bool:
        """Validate data types and value ranges."""
        try:
            # Confidence score bounds
            if not (0.0 <= state["confidence_score"] <= 1.0):
                self._logger.warning(f"Confidence score out of bounds: {state['confidence_score']}")
                return False
            
            # Validate trader scores structure
            for i, trader_score in enumerate(state["trader_scores"]):
                if not isinstance(trader_score, dict):
                    self._logger.warning(f"Trader score {i} is not a dictionary")
                    return False
                
                if "trader" not in trader_score or "score" not in trader_score:
                    self._logger.warning(f"Trader score {i} missing required fields")
                    return False
                
                score = trader_score["score"]
                if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                    self._logger.warning(f"Invalid trader score: {score}")
                    return False
            
            # Validate workflow step
            if state["current_step"] not in [step.value for step in WorkflowStep]:
                self._logger.warning(f"Invalid workflow step: {state['current_step']}")
                return False
            
            return True
            
        except (KeyError, TypeError, ValueError) as e:
            self._logger.error(f"Data type validation failed: {e}")
            return False
    
    def _validate_business_logic(self, state: TradingState) -> bool:
        """Validate business logic constraints."""
        try:
            # Market data consistency
            if state.get("polymarket_data") and state.get("kalshi_data"):
                # Both platforms have data - validate consistency
                if not self._validate_cross_platform_consistency(
                    state["polymarket_data"], state["kalshi_data"]
                ):
                    return False
            
            # Recommendation consistency with trader scores
            if state["recommendations"] and state["trader_scores"]:
                if not self._validate_recommendation_consistency(
                    state["recommendations"], state["trader_scores"]
                ):
                    return False
            
            # Risk assessment alignment
            if state.get("risk_assessment") and state["recommendations"]:
                if not self._validate_risk_alignment(
                    state["risk_assessment"], state["recommendations"]
                ):
                    return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Business logic validation failed: {e}")
            return False
    
    def _validate_workflow_consistency(self, state: TradingState) -> bool:
        """Validate workflow progression consistency."""
        try:
            current_step = state["current_step"]
            completed_steps = state["completed_steps"]
            
            # Current step should be valid
            if current_step not in [step.value for step in WorkflowStep]:
                return False
            
            # Completed steps should be valid
            for step in completed_steps:
                if step not in [step.value for step in WorkflowStep]:
                    return False
            
            # Allow flexible workflow progression - don't enforce strict ordering
            # This allows for parallel execution and dynamic workflows
            return True
            
        except (ValueError, KeyError) as e:
            self._logger.error(f"Workflow consistency validation failed: {e}")
            return False
    
    def _validate_cross_platform_consistency(self, polymarket_data: List[Dict], kalshi_data: List[Dict]) -> bool:
        """Validate consistency between platform data."""
        # Placeholder for cross-platform validation logic
        return True
    
    def _validate_recommendation_consistency(self, recommendations: List[Dict], trader_scores: List[Dict]) -> bool:
        """Validate that recommendations align with trader scores."""
        # Placeholder for recommendation consistency logic
        return True
    
    def _validate_risk_alignment(self, risk_assessment: Dict, recommendations: List[Dict]) -> bool:
        """Validate that recommendations align with risk assessment."""
        # Placeholder for risk alignment validation
        return True
    
    async def ensure_state_consistency(self, state: TradingState) -> TradingState:
        """
        Ensure state consistency across all components.
        
        Args:
            state: TradingState to check and fix
            
        Returns:
            Consistent TradingState
            
        Raises:
            StateConsistencyError: If consistency cannot be ensured
        """
        try:
            # Create a copy to avoid modifying original
            consistent_state = state.copy()
            
            # Ensure debug info exists
            if "_debug_info" not in consistent_state or not consistent_state["_debug_info"]:
                consistent_state["_debug_info"] = StateDebugInfo()
            
            # Consistency checks
            consistency_results = {}
            
            # Check 1: Workflow step consistency
            consistency_results["workflow_steps"] = self._ensure_workflow_consistency(consistent_state)
            
            # Check 2: Data referential integrity
            consistency_results["referential_integrity"] = self._ensure_referential_integrity(consistent_state)
            
            # Check 3: Score consistency
            consistency_results["score_consistency"] = self._ensure_score_consistency(consistent_state)
            
            # Check 4: Timestamp consistency
            consistency_results["timestamp_consistency"] = self._ensure_timestamp_consistency(consistent_state)
            
            # Update debug info
            consistent_state["_debug_info"].consistency_checks = consistency_results
            
            # Validate final state
            if not await self.validate_state(consistent_state, strict=False):
                raise StateConsistencyError("Could not achieve consistent state")
            
            return consistent_state
            
        except Exception as e:
            self._logger.error(f"State consistency error: {e}")
            raise StateConsistencyError(f"Failed to ensure state consistency: {e}")
    
    def _ensure_workflow_consistency(self, state: TradingState) -> bool:
        """Ensure workflow step consistency."""
        try:
            current_step = state["current_step"]
            completed_steps = state["completed_steps"]
            
            # Ensure initialization is always in completed steps if we've moved past it
            if current_step != WorkflowStep.INITIALIZATION.value:
                if WorkflowStep.INITIALIZATION.value not in completed_steps:
                    completed_steps.append(WorkflowStep.INITIALIZATION.value)
            
            # For non-initialization steps, ensure they're added to completed steps
            if current_step != WorkflowStep.INITIALIZATION.value and current_step not in completed_steps:
                completed_steps.append(current_step)
            
            return True
            
        except Exception as e:
            self._logger.error(f"Workflow consistency error: {e}")
            return False
    
    def _ensure_referential_integrity(self, state: TradingState) -> bool:
        """Ensure referential integrity between state components."""
        try:
            # Clean up invalid trader scores
            valid_trader_scores = []
            for score in state["trader_scores"]:
                if isinstance(score, dict) and "trader" in score and "score" in score:
                    # Valid structure
                    valid_trader_scores.append(score)
                elif isinstance(score, dict) and "score" in score:
                    # Missing trader info but has score - try to fix
                    if isinstance(score["score"], (int, float)) and 0.0 <= score["score"] <= 1.0:
                        score["trader"] = {"trader_id": f"unknown_{len(valid_trader_scores)}", "platform": "unknown"}
                        valid_trader_scores.append(score)
                # Skip invalid entries
            
            state["trader_scores"] = valid_trader_scores
            
            # Ensure trader IDs in recommendations exist in trader_scores
            recommendation_trader_ids = set()
            for rec in state["recommendations"]:
                if "trader_id" in rec:
                    recommendation_trader_ids.add(rec["trader_id"])
            
            trader_score_ids = set()
            for score in state["trader_scores"]:
                if "trader" in score and isinstance(score["trader"], dict) and "trader_id" in score["trader"]:
                    trader_score_ids.add(score["trader"]["trader_id"])
                elif "trader_id" in score:
                    trader_score_ids.add(score["trader_id"])
            
            # Remove recommendations for traders not in scores
            valid_recommendations = []
            for rec in state["recommendations"]:
                trader_id = rec.get("trader_id")
                if not trader_id or trader_id in trader_score_ids:
                    valid_recommendations.append(rec)
            
            state["recommendations"] = valid_recommendations
            return True
            
        except Exception as e:
            self._logger.error(f"Referential integrity error: {e}")
            return False
    
    def _ensure_score_consistency(self, state: TradingState) -> bool:
        """Ensure score consistency across components."""
        try:
            # Ensure confidence score is within bounds
            if state["confidence_score"] < 0.0:
                state["confidence_score"] = 0.0
            elif state["confidence_score"] > 1.0:
                state["confidence_score"] = 1.0
            
            # Ensure trader scores are within bounds
            for trader_score in state["trader_scores"]:
                if "score" in trader_score:
                    score = trader_score["score"]
                    if score < 0.0:
                        trader_score["score"] = 0.0
                    elif score > 1.0:
                        trader_score["score"] = 1.0
            
            return True
            
        except Exception as e:
            self._logger.error(f"Score consistency error: {e}")
            return False
    
    def _ensure_timestamp_consistency(self, state: TradingState) -> bool:
        """Ensure timestamp consistency across components."""
        try:
            current_time = datetime.now().isoformat()
            
            # Add timestamps to components that need them
            for error in state["error_log"]:
                if "timestamp" not in error:
                    error["timestamp"] = current_time
            
            for rec in state["recommendations"]:
                if "generated_at" not in rec:
                    rec["generated_at"] = current_time
            
            return True
            
        except Exception as e:
            self._logger.error(f"Timestamp consistency error: {e}")
            return False
    
    async def acquire_state_lock(self, state: TradingState, agent_id: str) -> bool:
        """
        Acquire exclusive lock on state for concurrent access control.
        
        Args:
            state: TradingState to lock
            agent_id: ID of the agent requesting the lock
            
        Returns:
            True if lock acquired, False otherwise
        """
        try:
            state_id = self._get_state_id(state)
            
            if state_id not in self._active_locks:
                self._active_locks[state_id] = asyncio.Lock()
            
            lock = self._active_locks[state_id]
            acquired = lock.locked() == False
            
            if acquired:
                await lock.acquire()
                state["_state_lock"] = agent_id
                self._logger.debug(f"State lock acquired by agent {agent_id}")
            
            return acquired
            
        except Exception as e:
            self._logger.error(f"Failed to acquire state lock: {e}")
            return False
    
    async def release_state_lock(self, state: TradingState, agent_id: str) -> bool:
        """
        Release state lock.
        
        Args:
            state: TradingState to unlock
            agent_id: ID of the agent releasing the lock
            
        Returns:
            True if lock released, False otherwise
        """
        try:
            state_id = self._get_state_id(state)
            
            if state_id in self._active_locks:
                lock = self._active_locks[state_id]
                if state.get("_state_lock") == agent_id:
                    lock.release()
                    state["_state_lock"] = None
                    self._logger.debug(f"State lock released by agent {agent_id}")
                    return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to release state lock: {e}")
            return False
    
    def _get_state_id(self, state: TradingState) -> str:
        """Generate unique ID for state based on content hash."""
        state_content = {
            "original_query": state.get("original_query", ""),
            "checkpoint_id": state.get("state_checkpoint_id", "")
        }
        content_str = json.dumps(state_content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def log_state_transition(self, state: TradingState, from_step: str, to_step: str, 
                           agent_id: str, operation: str) -> None:
        """
        Log state transition for debugging and audit purposes.
        
        Args:
            state: TradingState being transitioned
            from_step: Previous workflow step
            to_step: New workflow step
            agent_id: ID of agent performing transition
            operation: Description of operation
        """
        try:
            if "_debug_info" not in state or not state["_debug_info"]:
                state["_debug_info"] = StateDebugInfo()
            
            # Calculate data hash for integrity checking
            state_data = {k: v for k, v in state.items() if not k.startswith("_")}
            data_hash = hashlib.sha256(json.dumps(state_data, sort_keys=True).encode()).hexdigest()[:16]
            
            transition = StateTransition(
                from_step=from_step,
                to_step=to_step,
                timestamp=datetime.now(),
                agent_id=agent_id,
                operation=operation,
                data_hash=data_hash,
                validation_passed=True  # Will be updated by validation
            )
            
            state["_debug_info"].add_transition(transition)
            
            self._logger.info(f"State transition logged: {from_step} -> {to_step} by {agent_id}")
            
        except Exception as e:
            self._logger.error(f"Failed to log state transition: {e}")
    
    def get_state_debug_summary(self, state: TradingState) -> Dict[str, Any]:
        """
        Get comprehensive debug summary of state.
        
        Args:
            state: TradingState to analyze
            
        Returns:
            Dictionary containing debug information
        """
        try:
            debug_info = state.get("_debug_info")
            if not debug_info:
                return {"error": "No debug information available"}
            
            return {
                "state_version": debug_info.state_version,
                "last_modified": debug_info.last_modified.isoformat(),
                "modification_count": debug_info.modification_count,
                "recent_transitions": [t.to_dict() for t in debug_info.get_recent_transitions(5)],
                "validation_summary": {
                    "total_validations": len(debug_info.validation_history),
                    "recent_failures": [
                        v for v in debug_info.validation_history[-10:] 
                        if not v["passed"]
                    ]
                },
                "consistency_status": debug_info.consistency_checks,
                "performance_metrics": debug_info.performance_metrics,
                "current_step": state.get("current_step"),
                "completed_steps_count": len(state.get("completed_steps", [])),
                "error_count": len(state.get("error_log", [])),
                "recommendations_count": len(state.get("recommendations", [])),
                "trader_scores_count": len(state.get("trader_scores", []))
            }
            
        except Exception as e:
            self._logger.error(f"Failed to generate debug summary: {e}")
            return {"error": f"Debug summary generation failed: {e}"}


# Global state manager instance
_state_manager = StateManager()


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    return _state_manager
    """
    Comprehensive trader profile with performance and risk metrics.
    """
    trader_id: str
    platform: str
    niche: str
    
    # Performance Metrics
    win_rate: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk Metrics
    risk_score: float
    volatility: float
    correlation_market: float
    
    # Activity Metrics
    total_trades: int
    avg_position_size: float
    trading_frequency: str
    
    # Enrichment Data
    sentiment_score: float
    news_mentions: int
    social_sentiment: Dict[str, Any]
    
    # Metadata
    last_updated: str
    confidence_level: float
    data_quality_score: float
    
    def __post_init__(self):
        """Validate trader profile data after initialization."""
        if not (0.0 <= self.win_rate <= 1.0):
            raise ValueError("win_rate must be between 0.0 and 1.0")
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError("risk_score must be between 0.0 and 1.0")
        if self.total_trades < 0:
            raise ValueError("total_trades must be non-negative")


@dataclass
class MarketEvent:
    """
    Market event data model with comprehensive market information.
    """
    event_id: str
    platform: str
    title: str
    description: str
    
    # Market Data
    total_volume: float
    liquidity: float
    current_odds: Dict[str, float]
    
    # Timing
    created_at: str
    resolution_date: str
    last_trade_time: str
    
    # Classification
    niche: str
    tags: List[str]
    category: str
    
    # Sentiment
    sentiment_score: float
    news_coverage: int
    social_buzz: float
    
    # Trading Data
    top_traders: List[str]
    recent_trades: List[Dict[str, Any]]
    price_history: List[Dict[str, Any]]
    
    def __post_init__(self):
        """Validate market event data after initialization."""
        if self.total_volume < 0:
            raise ValueError("total_volume must be non-negative")
        
        # Validate odds sum to approximately 1.0 (allowing for small rounding errors)
        odds_sum = sum(self.current_odds.values())
        if not (0.95 <= odds_sum <= 1.05):
            raise ValueError("current_odds values must sum to approximately 1.0")


@dataclass
class TraderProfile:
    """
    Comprehensive trader profile with performance and risk metrics.
    """
    trader_id: str
    platform: str
    niche: str
    
    # Performance Metrics
    win_rate: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk Metrics
    risk_score: float
    volatility: float
    correlation_market: float
    
    # Activity Metrics
    total_trades: int
    avg_position_size: float
    trading_frequency: str
    
    # Enrichment Data
    sentiment_score: float
    news_mentions: int
    social_sentiment: Dict[str, Any]
    
    # Metadata
    last_updated: str
    confidence_level: float
    data_quality_score: float
    
    def __post_init__(self):
        """Validate trader profile data after initialization."""
        if not (0.0 <= self.win_rate <= 1.0):
            raise ValueError("win_rate must be between 0.0 and 1.0")
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError("risk_score must be between 0.0 and 1.0")
        if self.total_trades < 0:
            raise ValueError("total_trades must be non-negative")


@dataclass
class MarketEvent:
    """
    Market event data model with comprehensive market information.
    """
    event_id: str
    platform: str
    title: str
    description: str
    
    # Market Data
    total_volume: float
    liquidity: float
    current_odds: Dict[str, float]
    
    # Timing
    created_at: str
    resolution_date: str
    last_trade_time: str
    
    # Classification
    niche: str
    tags: List[str]
    category: str
    
    # Sentiment
    sentiment_score: float
    news_coverage: int
    social_buzz: float
    
    # Trading Data
    top_traders: List[str]
    recent_trades: List[Dict[str, Any]]
    price_history: List[Dict[str, Any]]
    
    def __post_init__(self):
        """Validate market event data after initialization."""
        if self.total_volume < 0:
            raise ValueError("total_volume must be non-negative")
        
        # Validate odds sum to approximately 1.0 (allowing for small rounding errors)
        odds_sum = sum(self.current_odds.values())
        if not (0.95 <= odds_sum <= 1.05):
            raise ValueError("current_odds values must sum to approximately 1.0")


def create_initial_state(query: str, user_preferences: Dict[str, Any]) -> TradingState:
    """
    Create an initial TradingState for a new analysis workflow.
    
    Args:
        query: The user's trading query
        user_preferences: User configuration and preferences
        
    Returns:
        Initialized TradingState ready for graph execution with enhanced debugging
    """
    initial_state = TradingState(
        # Query Context
        original_query=query,
        parsed_intent={},
        niche="",
        platform_preference=user_preferences.get("preferred_platforms", ["polymarket", "kalshi"])[0],
        
        # Market Data
        polymarket_data=[],
        kalshi_data=[],
        market_conditions={},
        
        # Analysis Results
        trader_scores=[],
        sentiment_analysis={},
        risk_assessment={},
        
        # Workflow Control
        current_step=WorkflowStep.INITIALIZATION.value,
        completed_steps=[],
        error_log=[],
        state_checkpoint_id=None,
        
        # Memory & Learning
        rag_context=[],
        historical_performance={},
        user_preferences=user_preferences,
        
        # Output
        recommendations=[],
        confidence_score=0.0,
        explanation="",
        
        # Enhanced State Management
        _debug_info=StateDebugInfo(),
        _state_lock=None,
        _validation_rules=None,
        _consistency_validators=None
    )
    
    # Log initial state creation
    state_manager = get_state_manager()
    state_manager.log_state_transition(
        initial_state, 
        "none", 
        WorkflowStep.INITIALIZATION.value,
        "system",
        "initial_state_creation"
    )
    
    return initial_state


async def validate_state_consistency(state: TradingState) -> bool:
    """
    Validate that the TradingState maintains consistency invariants.
    
    Args:
        state: The TradingState to validate
        
    Returns:
        True if state is consistent, False otherwise
    """
    state_manager = get_state_manager()
    return await state_manager.validate_state(state, strict=False)


async def update_workflow_progress(state: TradingState, step: str, agent_id: str = "system") -> TradingState:
    """
    Update workflow progress in the state with enhanced logging and validation.
    
    Args:
        state: Current TradingState
        step: The workflow step being completed
        agent_id: ID of the agent performing the update
        
    Returns:
        Updated TradingState with progress information
        
    Raises:
        StateValidationError: If the step transition is invalid
    """
    state_manager = get_state_manager()
    
    # Validate step transition
    if step not in [workflow_step.value for workflow_step in WorkflowStep]:
        raise StateValidationError(f"Invalid workflow step: {step}")
    
    # Log the transition
    previous_step = state["current_step"]
    state_manager.log_state_transition(state, previous_step, step, agent_id, "workflow_progress_update")
    
    # Update state
    updated_state = state.copy()
    updated_state["current_step"] = step
    
    if step not in updated_state["completed_steps"]:
        updated_state["completed_steps"].append(step)
    
    # Ensure state consistency
    try:
        consistent_state = await state_manager.ensure_state_consistency(updated_state)
        
        # Validate the updated state
        if not await state_manager.validate_state(consistent_state, strict=False):
            logging.warning(f"State validation failed after workflow update to {step}")
        
        return consistent_state
        
    except StateConsistencyError as e:
        logging.error(f"Failed to maintain state consistency during workflow update: {e}")
        # Return original state if consistency cannot be maintained
        return state


async def safe_state_update(state: TradingState, updates: Dict[str, Any], 
                          agent_id: str, operation: str) -> TradingState:
    """
    Safely update state with validation and consistency checks.
    
    Args:
        state: Current TradingState
        updates: Dictionary of updates to apply
        agent_id: ID of the agent performing the update
        operation: Description of the operation
        
    Returns:
        Updated and validated TradingState
        
    Raises:
        StateValidationError: If updates result in invalid state
        StateConsistencyError: If consistency cannot be maintained
    """
    state_manager = get_state_manager()
    
    # Acquire lock for thread safety
    lock_acquired = await state_manager.acquire_state_lock(state, agent_id)
    
    try:
        # Apply updates
        updated_state = state.copy()
        for key, value in updates.items():
            if key in updated_state:
                updated_state[key] = value
        
        # Log the update
        state_manager.log_state_transition(
            updated_state, 
            state["current_step"], 
            updated_state["current_step"],
            agent_id, 
            operation
        )
        
        # Ensure consistency
        consistent_state = await state_manager.ensure_state_consistency(updated_state)
        
        # Validate final state
        if not await state_manager.validate_state(consistent_state, strict=True):
            raise StateValidationError("State validation failed after update")
        
        return consistent_state
        
    finally:
        # Always release lock
        if lock_acquired:
            await state_manager.release_state_lock(state, agent_id)


def get_state_debug_info(state: TradingState) -> Dict[str, Any]:
    """
    Get debug information for the current state.
    
    Args:
        state: TradingState to analyze
        
    Returns:
        Dictionary containing debug information
    """
    state_manager = get_state_manager()
    return state_manager.get_state_debug_summary(state)


# Legacy function maintained for backward compatibility
def validate_state_consistency_legacy(state: TradingState) -> bool:
    """
    Legacy synchronous state validation function.
    
    Args:
        state: The TradingState to validate
        
    Returns:
        True if state is consistent, False otherwise
    """
    try:
        # Check required fields are present
        required_fields = [
            "original_query", "current_step", "completed_steps",
            "trader_scores", "recommendations", "confidence_score"
        ]
        
        for field in required_fields:
            if field not in state:
                return False
        
        # Validate confidence score bounds
        if not (0.0 <= state["confidence_score"] <= 1.0):
            return False
        
        # Validate workflow progression
        if state["current_step"] not in state["completed_steps"] and state["current_step"] != "initialization":
            # Current step should either be completed or be the next step
            pass
        
        # Validate trader scores structure
        for trader_score in state["trader_scores"]:
            if "trader" not in trader_score or "score" not in trader_score:
                return False
            if not (0.0 <= trader_score["score"] <= 1.0):
                return False
        
        return True
        
    except (KeyError, TypeError, ValueError):
        return False


def update_workflow_progress_legacy(state: TradingState, step: str) -> TradingState:
    """
    Legacy synchronous workflow progress update function.
    
    Args:
        state: Current TradingState
        step: The workflow step being completed
        
    Returns:
        Updated TradingState with progress information
    """
    updated_state = state.copy()
    updated_state["current_step"] = step
    
    if step not in updated_state["completed_steps"]:
        updated_state["completed_steps"].append(step)
    
    return updated_state