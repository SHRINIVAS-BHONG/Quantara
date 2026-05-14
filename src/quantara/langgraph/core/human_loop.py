"""
Human-in-the-loop integration for the Quantara LangGraph trading system.

This module provides interfaces and utilities for incorporating human decision-making
at critical workflow points with timeout handling, fallback mechanisms, and context
presentation for informed decision-making.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class DecisionType(Enum):
    """Types of decisions that can be requested from humans."""
    APPROVAL = "approval"
    SELECTION = "selection"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    RISK_OVERRIDE = "risk_override"
    STRATEGY_CHOICE = "strategy_choice"
    ABORT_CONTINUE = "abort_continue"


class DecisionPriority(Enum):
    """Priority levels for human decisions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DecisionOption:
    """
    Represents a single option in a human decision point.
    
    Attributes:
        id: Unique identifier for the option
        label: Human-readable label for the option
        description: Detailed description of what this option means
        recommended: Whether this option is recommended by the system
        metadata: Additional metadata about the option
    """
    id: str
    label: str
    description: str
    recommended: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext:
    """
    Context information for a human decision point.
    
    Attributes:
        decision_id: Unique identifier for this decision
        decision_type: Type of decision being requested
        priority: Priority level of the decision
        title: Brief title describing the decision
        description: Detailed description of the situation
        options: Available options for the decision
        current_state: Relevant state information for context
        recommendations: System recommendations with reasoning
        timeout_seconds: Maximum time to wait for human input
        default_option_id: Option to use if timeout occurs
        metadata: Additional context metadata
    """
    decision_id: str
    decision_type: DecisionType
    priority: DecisionPriority
    title: str
    description: str
    options: List[DecisionOption]
    current_state: Dict[str, Any]
    recommendations: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300  # 5 minutes default
    default_option_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DecisionResult:
    """
    Result of a human decision.
    
    Attributes:
        decision_id: ID of the decision this result is for
        selected_option_id: ID of the selected option
        selected_option: The full option that was selected
        decision_time: When the decision was made
        timed_out: Whether the decision timed out
        user_notes: Optional notes from the user
        metadata: Additional result metadata
    """
    decision_id: str
    selected_option_id: str
    selected_option: DecisionOption
    decision_time: datetime
    timed_out: bool = False
    user_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HumanLoopInterface:
    """
    Interface for human-in-the-loop decision making in the trading workflow.
    
    This class manages the presentation of decision points to human operators,
    handles timeout scenarios with fallback actions, validates decisions, and
    integrates them back into the workflow.
    
    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        decision_callback: Optional[Callable] = None
    ):
        """
        Initialize the HumanLoopInterface.
        
        Args:
            config: Optional configuration for human loop behavior
            decision_callback: Optional callback function for presenting decisions
                              Should accept DecisionContext and return DecisionResult
        
        Requirements: 10.1
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Decision callback for presenting decisions to humans
        self.decision_callback = decision_callback
        
        # Configuration
        self.default_timeout = self.config.get("default_timeout_seconds", 300)
        self.enable_timeout = self.config.get("enable_timeout", True)
        self.auto_select_recommended = self.config.get("auto_select_recommended", True)
        self.require_validation = self.config.get("require_validation", True)
        
        # Decision tracking
        self.pending_decisions: Dict[str, DecisionContext] = {}
        self.completed_decisions: Dict[str, DecisionResult] = {}
        self.decision_history: List[DecisionResult] = []
        
        # Statistics
        self.stats = {
            "total_decisions": 0,
            "timed_out_decisions": 0,
            "user_decisions": 0,
            "fallback_decisions": 0
        }
    
    async def request_decision(
        self,
        decision_type: DecisionType,
        title: str,
        description: str,
        options: List[DecisionOption],
        current_state: Dict[str, Any],
        priority: DecisionPriority = DecisionPriority.MEDIUM,
        timeout_seconds: Optional[int] = None,
        default_option_id: Optional[str] = None,
        recommendations: Optional[Dict[str, Any]] = None
    ) -> DecisionResult:
        """
        Request a decision from a human operator with timeout handling.
        
        Args:
            decision_type: Type of decision being requested
            title: Brief title for the decision
            description: Detailed description of the situation
            options: List of available options
            current_state: Current workflow state for context
            priority: Priority level of the decision
            timeout_seconds: Maximum time to wait (uses default if None)
            default_option_id: Option to use on timeout (uses recommended if None)
            recommendations: System recommendations with reasoning
        
        Returns:
            DecisionResult containing the selected option
        
        Requirements: 10.1, 10.2, 10.3
        """
        try:
            # Generate unique decision ID
            decision_id = self._generate_decision_id(decision_type)
            
            # Use default timeout if not specified
            if timeout_seconds is None:
                timeout_seconds = self.default_timeout
            
            # Determine default option if not specified
            if default_option_id is None and self.auto_select_recommended:
                recommended_options = [opt for opt in options if opt.recommended]
                if recommended_options:
                    default_option_id = recommended_options[0].id
                elif options:
                    default_option_id = options[0].id
            
            # Create decision context
            context = DecisionContext(
                decision_id=decision_id,
                decision_type=decision_type,
                priority=priority,
                title=title,
                description=description,
                options=options,
                current_state=self._sanitize_state_for_context(current_state),
                recommendations=recommendations or {},
                timeout_seconds=timeout_seconds,
                default_option_id=default_option_id
            )
            
            # Track pending decision
            self.pending_decisions[decision_id] = context
            self.stats["total_decisions"] += 1
            
            self.logger.info(
                f"Requesting human decision: {title} "
                f"(type={decision_type.value}, priority={priority.value}, "
                f"timeout={timeout_seconds}s)"
            )
            
            # Present decision and wait for response with timeout
            result = await self._present_decision_with_timeout(context)
            
            # Validate decision
            if self.require_validation:
                validation_result = self._validate_decision(result, context)
                if not validation_result["valid"]:
                    self.logger.warning(
                        f"Decision validation failed: {validation_result['reason']}"
                    )
                    # Use fallback if validation fails
                    result = self._create_fallback_result(context)
            
            # Track completed decision
            self.completed_decisions[decision_id] = result
            self.decision_history.append(result)
            del self.pending_decisions[decision_id]
            
            # Update statistics
            if result.timed_out:
                self.stats["timed_out_decisions"] += 1
                self.stats["fallback_decisions"] += 1
            elif result.user_notes and ("fallback" in result.user_notes.lower() or "emergency" in result.user_notes.lower()):
                # Also count non-timeout fallbacks
                self.stats["fallback_decisions"] += 1
            else:
                self.stats["user_decisions"] += 1
            
            self.logger.info(
                f"Decision completed: {decision_id} "
                f"(selected={result.selected_option_id}, timed_out={result.timed_out})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error requesting decision: {e}")
            # Create emergency fallback result
            return self._create_emergency_fallback(decision_type, options)
    
    async def _present_decision_with_timeout(
        self,
        context: DecisionContext
    ) -> DecisionResult:
        """
        Present decision to human with timeout handling.
        
        Args:
            context: Decision context to present
        
        Returns:
            DecisionResult from human or fallback
        
        Requirements: 10.2, 10.3
        """
        try:
            if self.decision_callback is None:
                # No callback configured, use default fallback immediately
                self.logger.warning(
                    "No decision callback configured, using default fallback"
                )
                return self._create_fallback_result(context)
            
            # Create task for getting human input
            decision_task = asyncio.create_task(
                self._get_human_input(context)
            )
            
            if self.enable_timeout and context.timeout_seconds > 0:
                # Wait for decision with timeout
                try:
                    result = await asyncio.wait_for(
                        decision_task,
                        timeout=context.timeout_seconds
                    )
                    return result
                    
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Decision {context.decision_id} timed out after "
                        f"{context.timeout_seconds} seconds, using fallback"
                    )
                    # Cancel the pending task
                    decision_task.cancel()
                    
                    # Use fallback option
                    return self._create_fallback_result(context, timed_out=True)
            else:
                # No timeout, wait indefinitely
                result = await decision_task
                return result
                
        except Exception as e:
            self.logger.error(f"Error presenting decision: {e}")
            return self._create_fallback_result(context)
    
    async def _get_human_input(self, context: DecisionContext) -> DecisionResult:
        """
        Get input from human operator via callback.
        
        Args:
            context: Decision context
        
        Returns:
            DecisionResult from human
        
        Requirements: 10.1, 10.2
        """
        try:
            # Call the decision callback
            result = await self.decision_callback(context)
            
            if isinstance(result, DecisionResult):
                return result
            else:
                # Callback returned unexpected type, use fallback
                self.logger.warning(
                    f"Decision callback returned unexpected type: {type(result)}"
                )
                return self._create_fallback_result(context)
                
        except Exception as e:
            self.logger.error(f"Error getting human input: {e}")
            return self._create_fallback_result(context)
    
    def _create_fallback_result(
        self,
        context: DecisionContext,
        timed_out: bool = False
    ) -> DecisionResult:
        """
        Create a fallback decision result using default option.
        
        Args:
            context: Decision context
            timed_out: Whether this is due to timeout
        
        Returns:
            DecisionResult with fallback option
        
        Requirements: 10.3
        """
        # Determine which option to use as fallback
        fallback_option_id = context.default_option_id
        
        if fallback_option_id is None:
            # No default specified, use first option
            if context.options:
                fallback_option_id = context.options[0].id
            else:
                # No options available, create emergency option
                self.logger.error("No options available for fallback")
                raise ValueError("Cannot create fallback result without options")
        
        # Find the option
        fallback_option = next(
            (opt for opt in context.options if opt.id == fallback_option_id),
            None
        )
        
        if fallback_option is None:
            # Default option not found, use first option
            fallback_option = context.options[0]
            fallback_option_id = fallback_option.id
        
        return DecisionResult(
            decision_id=context.decision_id,
            selected_option_id=fallback_option_id,
            selected_option=fallback_option,
            decision_time=datetime.now(),
            timed_out=timed_out,
            user_notes="Automatic fallback decision" if timed_out else "Default fallback",
            metadata={
                "fallback_reason": "timeout" if timed_out else "no_callback",
                "original_timeout": context.timeout_seconds
            }
        )
    
    def _create_emergency_fallback(
        self,
        decision_type: DecisionType,
        options: List[DecisionOption]
    ) -> DecisionResult:
        """
        Create an emergency fallback result when normal fallback fails.
        
        Args:
            decision_type: Type of decision
            options: Available options
        
        Returns:
            Emergency DecisionResult
        """
        if not options:
            # Create a safe default option
            options = [DecisionOption(
                id="emergency_continue",
                label="Continue",
                description="Emergency fallback: continue with default behavior",
                recommended=True
            )]
        
        return DecisionResult(
            decision_id=f"emergency_{decision_type.value}_{datetime.now().timestamp()}",
            selected_option_id=options[0].id,
            selected_option=options[0],
            decision_time=datetime.now(),
            timed_out=False,
            user_notes="Emergency fallback due to error",
            metadata={"emergency": True}
        )
    
    def _validate_decision(
        self,
        result: DecisionResult,
        context: DecisionContext
    ) -> Dict[str, Any]:
        """
        Validate a decision result.
        
        Args:
            result: Decision result to validate
            context: Original decision context
        
        Returns:
            Validation result with 'valid' boolean and 'reason' if invalid
        
        Requirements: 10.4
        """
        try:
            # Check that decision ID matches
            if result.decision_id != context.decision_id:
                return {
                    "valid": False,
                    "reason": "Decision ID mismatch"
                }
            
            # Check that selected option exists in context
            valid_option_ids = [opt.id for opt in context.options]
            if result.selected_option_id not in valid_option_ids:
                return {
                    "valid": False,
                    "reason": f"Invalid option ID: {result.selected_option_id}"
                }
            
            # Check that selected option matches the option object
            if result.selected_option.id != result.selected_option_id:
                return {
                    "valid": False,
                    "reason": "Selected option ID does not match option object"
                }
            
            # Validate decision time is reasonable
            time_since_creation = (result.decision_time - context.created_at).total_seconds()
            if time_since_creation < 0:
                return {
                    "valid": False,
                    "reason": "Decision time is before creation time"
                }
            
            if time_since_creation > context.timeout_seconds + 60:  # Allow 60s grace period
                return {
                    "valid": False,
                    "reason": "Decision time exceeds timeout by too much"
                }
            
            # All validation checks passed
            return {
                "valid": True,
                "reason": None
            }
            
        except Exception as e:
            self.logger.error(f"Error validating decision: {e}")
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            }
    
    def _sanitize_state_for_context(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sanitize state data for presentation to humans.
        
        Removes sensitive information and simplifies complex data structures.
        
        Args:
            state: Raw state data
        
        Returns:
            Sanitized state data suitable for human presentation
        
        Requirements: 10.2
        """
        try:
            # Fields to include in context (whitelist approach)
            relevant_fields = [
                "current_step",
                "completed_steps",
                "niche",
                "platform_preference",
                "confidence_score",
                "error_log"
            ]
            
            sanitized = {}
            
            for field in relevant_fields:
                if field in state:
                    value = state[field]
                    
                    # Simplify error log
                    if field == "error_log" and isinstance(value, list):
                        sanitized[field] = [
                            {
                                "step": e.get("step", "unknown"),
                                "severity": e.get("severity", "unknown"),
                                "error": str(e.get("error", ""))[:100]  # Truncate long errors
                            }
                            for e in value[-5:]  # Only last 5 errors
                        ]
                    else:
                        sanitized[field] = value
            
            # Add summary statistics
            sanitized["summary"] = {
                "market_count": len(state.get("polymarket_data", [])) + len(state.get("kalshi_data", [])),
                "trader_count": len(state.get("trader_scores", [])),
                "has_sentiment": bool(state.get("sentiment_analysis")),
                "has_risk": bool(state.get("risk_assessment")),
                "error_count": len(state.get("error_log", []))
            }
            
            return sanitized
            
        except Exception as e:
            self.logger.error(f"Error sanitizing state: {e}")
            return {"error": "Failed to sanitize state"}
    
    def _generate_decision_id(self, decision_type: DecisionType) -> str:
        """Generate a unique decision ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{decision_type.value}_{timestamp}"
    
    def get_decision_history(
        self,
        limit: Optional[int] = None,
        decision_type: Optional[DecisionType] = None
    ) -> List[DecisionResult]:
        """
        Get decision history with optional filtering.
        
        Args:
            limit: Maximum number of decisions to return
            decision_type: Filter by decision type
        
        Returns:
            List of DecisionResult objects
        """
        history = self.decision_history.copy()
        
        # Filter by type if specified
        if decision_type is not None:
            # Note: We'd need to store decision_type in DecisionResult for this
            # For now, return all
            pass
        
        # Apply limit
        if limit is not None:
            history = history[-limit:]
        
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about human loop decisions.
        
        Returns:
            Dictionary with decision statistics
        """
        stats = self.stats.copy()
        
        # Calculate additional metrics
        if stats["total_decisions"] > 0:
            stats["timeout_rate"] = stats["timed_out_decisions"] / stats["total_decisions"]
            stats["user_decision_rate"] = stats["user_decisions"] / stats["total_decisions"]
        else:
            stats["timeout_rate"] = 0.0
            stats["user_decision_rate"] = 0.0
        
        stats["pending_decisions"] = len(self.pending_decisions)
        stats["completed_decisions"] = len(self.completed_decisions)
        
        return stats
    
    def clear_history(self):
        """Clear decision history (for testing or cleanup)."""
        self.decision_history.clear()
        self.completed_decisions.clear()
        self.logger.info("Decision history cleared")


# Convenience functions for common decision types

async def request_approval_decision(
    human_loop: HumanLoopInterface,
    title: str,
    description: str,
    current_state: Dict[str, Any],
    priority: DecisionPriority = DecisionPriority.MEDIUM,
    timeout_seconds: Optional[int] = None
) -> bool:
    """
    Request a simple approval/rejection decision.
    
    Args:
        human_loop: HumanLoopInterface instance
        title: Decision title
        description: Decision description
        current_state: Current workflow state
        priority: Decision priority
        timeout_seconds: Timeout in seconds
    
    Returns:
        True if approved, False if rejected
    
    Requirements: 10.1, 10.5
    """
    options = [
        DecisionOption(
            id="approve",
            label="Approve",
            description="Approve and continue with the proposed action",
            recommended=True
        ),
        DecisionOption(
            id="reject",
            label="Reject",
            description="Reject and abort the proposed action",
            recommended=False
        )
    ]
    
    result = await human_loop.request_decision(
        decision_type=DecisionType.APPROVAL,
        title=title,
        description=description,
        options=options,
        current_state=current_state,
        priority=priority,
        timeout_seconds=timeout_seconds,
        default_option_id="approve"  # Default to approve on timeout
    )
    
    return result.selected_option_id == "approve"


async def request_selection_decision(
    human_loop: HumanLoopInterface,
    title: str,
    description: str,
    options: List[DecisionOption],
    current_state: Dict[str, Any],
    priority: DecisionPriority = DecisionPriority.MEDIUM,
    timeout_seconds: Optional[int] = None,
    default_option_id: Optional[str] = None
) -> str:
    """
    Request a selection from multiple options.
    
    Args:
        human_loop: HumanLoopInterface instance
        title: Decision title
        description: Decision description
        options: List of options to choose from
        current_state: Current workflow state
        priority: Decision priority
        timeout_seconds: Timeout in seconds
        default_option_id: Default option on timeout
    
    Returns:
        ID of the selected option
    
    Requirements: 10.1, 10.5
    """
    result = await human_loop.request_decision(
        decision_type=DecisionType.SELECTION,
        title=title,
        description=description,
        options=options,
        current_state=current_state,
        priority=priority,
        timeout_seconds=timeout_seconds,
        default_option_id=default_option_id
    )
    
    return result.selected_option_id


async def request_risk_override_decision(
    human_loop: HumanLoopInterface,
    title: str,
    description: str,
    risk_details: Dict[str, Any],
    current_state: Dict[str, Any],
    timeout_seconds: Optional[int] = None
) -> bool:
    """
    Request a decision to override risk limits.
    
    Args:
        human_loop: HumanLoopInterface instance
        title: Decision title
        description: Decision description
        risk_details: Details about the risk being overridden
        current_state: Current workflow state
        timeout_seconds: Timeout in seconds
    
    Returns:
        True if override approved, False otherwise
    
    Requirements: 10.1, 10.5
    """
    options = [
        DecisionOption(
            id="override",
            label="Override Risk Limits",
            description="Proceed despite risk limit violations",
            recommended=False,
            metadata=risk_details
        ),
        DecisionOption(
            id="respect_limits",
            label="Respect Risk Limits",
            description="Abort action and respect configured risk limits",
            recommended=True,
            metadata=risk_details
        )
    ]
    
    result = await human_loop.request_decision(
        decision_type=DecisionType.RISK_OVERRIDE,
        title=title,
        description=description,
        options=options,
        current_state=current_state,
        priority=DecisionPriority.HIGH,
        timeout_seconds=timeout_seconds,
        default_option_id="respect_limits",  # Default to safe option
        recommendations={
            "system_recommendation": "respect_limits",
            "reasoning": "Risk limits are configured for safety",
            "risk_details": risk_details
        }
    )
    
    return result.selected_option_id == "override"
