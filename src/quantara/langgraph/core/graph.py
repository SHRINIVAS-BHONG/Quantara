"""
Main LangGraph implementation for the Quantara trading system.

This module defines the TradingGraph class that orchestrates the multi-agent
trading analysis workflow using LangGraph's StateGraph architecture with
comprehensive workflow control, error handling, and execution framework.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncIterator, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from .state import (
    TradingState, 
    create_initial_state, 
    validate_state_consistency,
    update_workflow_progress,
    safe_state_update,
    get_state_manager,
    WorkflowStep
)
from .checkpoint import (
    CheckpointManager,
    CheckpointStorageType,
    get_checkpoint_manager
)
from .human_loop import (
    HumanLoopInterface,
    DecisionType,
    DecisionPriority,
    DecisionOption,
    request_approval_decision,
    request_selection_decision,
    request_risk_override_decision
)
from ..agents.market_discovery import MarketDiscoveryAgent
from ..agents.sentiment_analysis import SentimentAnalysisAgent
from ..agents.risk_assessment import RiskAssessmentAgent
from ..agents.trader_analysis import TraderAnalysisAgent


class WorkflowController:
    """
    Advanced workflow controller for orchestrating multi-agent trading analysis.
    
    This controller manages the execution flow, handles conditional branching,
    coordinates parallel operations, and ensures proper state transitions
    throughout the trading analysis workflow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow controller.
        
        Args:
            config: Optional configuration for workflow behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.state_manager = get_state_manager()
        
        # Workflow configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.timeout_seconds = self.config.get("timeout_seconds", 300)
        self.enable_parallel_execution = self.config.get("enable_parallel_execution", True)
        self.checkpoint_frequency = self.config.get("checkpoint_frequency", "major_steps")
    
    async def should_continue_analysis(self, state: TradingState) -> Literal["continue", "error", "end"]:
        """
        Determine if analysis should continue based on current state.
        
        Args:
            state: Current TradingState
            
        Returns:
            Decision on workflow continuation
        """
        try:
            # Check for critical errors
            error_log = state.get("error_log", [])
            critical_errors = [e for e in error_log if e.get("severity") == "critical"]
            
            if critical_errors:
                self.logger.error(f"Critical errors detected: {len(critical_errors)}")
                return "error"
            
            # Check if we have sufficient market data to continue
            market_data_count = len(state.get("polymarket_data", [])) + len(state.get("kalshi_data", []))
            
            if market_data_count == 0:
                self.logger.warning("No market data found, ending analysis")
                return "end"
            
            # Check for too many non-critical errors
            if len(error_log) > 5:
                self.logger.warning(f"Too many errors ({len(error_log)}), may need intervention")
                # Continue but with caution
            
            return "continue"
            
        except Exception as e:
            self.logger.error(f"Error in workflow decision: {e}")
            return "error"
    
    async def should_generate_report(self, state: TradingState) -> Literal["generate", "error", "insufficient_data"]:
        """
        Determine if report generation should proceed.
        
        Args:
            state: Current TradingState
            
        Returns:
            Decision on report generation
        """
        try:
            # Check for errors that would prevent report generation
            if state.get("error_log"):
                critical_errors = [e for e in state.get("error_log", []) if e.get("severity") == "critical"]
                if critical_errors:
                    return "error"
            
            # Check if we have sufficient data for meaningful recommendations
            trader_scores = state.get("trader_scores", [])
            
            if len(trader_scores) == 0:
                self.logger.warning("No trader scores available for report generation")
                return "insufficient_data"
            
            # Check data quality
            valid_scores = [t for t in trader_scores if t.get("score", 0) > 0]
            
            if len(valid_scores) < len(trader_scores) * 0.5:  # Less than 50% valid scores
                self.logger.warning("Insufficient quality trader data for reliable recommendations")
                return "insufficient_data"
            
            return "generate"
            
        except Exception as e:
            self.logger.error(f"Error in report generation decision: {e}")
            return "error"
    
    async def coordinate_parallel_execution(
        self, 
        state: TradingState, 
        tasks: List[Dict[str, Any]]
    ) -> TradingState:
        """
        Coordinate parallel execution of multiple analysis tasks.
        
        Args:
            state: Current TradingState
            tasks: List of task configurations for parallel execution
            
        Returns:
            Updated TradingState with results from parallel tasks
        """
        if not self.enable_parallel_execution or len(tasks) <= 1:
            # Execute sequentially if parallel execution is disabled
            return await self._execute_sequential_tasks(state, tasks)
        
        try:
            self.logger.info(f"Executing {len(tasks)} tasks in parallel")
            
            # Create tasks for parallel execution
            async_tasks = []
            for task_config in tasks:
                task_func = task_config["function"]
                task_args = task_config.get("args", [])
                task_kwargs = task_config.get("kwargs", {})
                
                async_task = asyncio.create_task(
                    task_func(state, *task_args, **task_kwargs),
                    name=task_config.get("name", "unnamed_task")
                )
                async_tasks.append(async_task)
            
            # Execute tasks with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*async_tasks, return_exceptions=True),
                timeout=self.timeout_seconds
            )
            
            # Process results and merge state updates
            updated_state = state.copy()
            
            for i, result in enumerate(results):
                task_name = tasks[i].get("name", f"task_{i}")
                
                if isinstance(result, Exception):
                    self.logger.error(f"Parallel task {task_name} failed: {result}")
                    updated_state["error_log"].append({
                        "step": "parallel_execution",
                        "task": task_name,
                        "error": str(result),
                        "severity": "medium",
                        "timestamp": asyncio.get_event_loop().time()
                    })
                else:
                    # Merge successful results
                    updated_state = self._merge_state_updates(updated_state, result, task_name)
            
            return updated_state
            
        except asyncio.TimeoutError:
            self.logger.error(f"Parallel execution timed out after {self.timeout_seconds} seconds")
            updated_state = state.copy()
            updated_state["error_log"].append({
                "step": "parallel_execution",
                "error": "Execution timeout",
                "severity": "high",
                "timestamp": asyncio.get_event_loop().time()
            })
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error in parallel execution coordination: {e}")
            updated_state = state.copy()
            updated_state["error_log"].append({
                "step": "parallel_execution",
                "error": str(e),
                "severity": "high",
                "timestamp": asyncio.get_event_loop().time()
            })
            return updated_state
    
    async def _execute_sequential_tasks(
        self, 
        state: TradingState, 
        tasks: List[Dict[str, Any]]
    ) -> TradingState:
        """Execute tasks sequentially as fallback."""
        updated_state = state.copy()
        
        for task_config in tasks:
            try:
                task_func = task_config["function"]
                task_args = task_config.get("args", [])
                task_kwargs = task_config.get("kwargs", {})
                
                result = await task_func(updated_state, *task_args, **task_kwargs)
                updated_state = result
                
            except Exception as e:
                task_name = task_config.get("name", "unnamed_task")
                self.logger.error(f"Sequential task {task_name} failed: {e}")
                updated_state["error_log"].append({
                    "step": "sequential_execution",
                    "task": task_name,
                    "error": str(e),
                    "severity": "medium",
                    "timestamp": asyncio.get_event_loop().time()
                })
        
        return updated_state
    
    def _merge_state_updates(
        self, 
        base_state: TradingState, 
        update_state: TradingState, 
        task_name: str
    ) -> TradingState:
        """
        Merge state updates from parallel tasks.
        
        Args:
            base_state: Base state to merge into
            update_state: State updates from parallel task
            task_name: Name of the task for logging
            
        Returns:
            Merged TradingState
        """
        try:
            merged_state = base_state.copy()
            
            # Define merge strategies for different state fields
            merge_strategies = {
                # Append to lists
                "error_log": "append",
                "completed_steps": "append_unique",
                "rag_context": "append",
                
                # Update dictionaries
                "market_conditions": "update_dict",
                "sentiment_analysis": "update_dict",
                "risk_assessment": "update_dict",
                "user_preferences": "update_dict",
                
                # Replace with latest
                "current_step": "replace",
                "confidence_score": "replace",
                "explanation": "replace",
                
                # Merge lists with deduplication
                "polymarket_data": "merge_list",
                "kalshi_data": "merge_list",
                "trader_scores": "merge_list",
                "recommendations": "merge_list"
            }
            
            # Merge all fields from update_state
            for field, update_value in update_state.items():
                if field in merge_strategies:
                    strategy = merge_strategies[field]
                    base_value = merged_state.get(field)
                    
                    if strategy == "append" and isinstance(base_value, list):
                        if isinstance(update_value, list):
                            merged_state[field] = base_value + update_value
                        else:
                            merged_state[field] = base_value + [update_value]
                    
                    elif strategy == "append_unique" and isinstance(base_value, list):
                        if isinstance(update_value, list):
                            for item in update_value:
                                if item not in base_value:
                                    base_value.append(item)
                        else:
                            if update_value not in base_value:
                                base_value.append(update_value)
                        merged_state[field] = base_value
                    
                    elif strategy == "update_dict" and isinstance(base_value, dict):
                        if isinstance(update_value, dict):
                            merged_dict = base_value.copy()
                            merged_dict.update(update_value)
                            merged_state[field] = merged_dict
                    
                    elif strategy == "replace":
                        merged_state[field] = update_value
                    
                    elif strategy == "merge_list" and isinstance(base_value, list):
                        if isinstance(update_value, list):
                            # Merge lists, avoiding duplicates based on ID fields
                            merged_list = base_value.copy()
                            for item in update_value:
                                if not self._is_duplicate_item(item, merged_list):
                                    merged_list.append(item)
                            merged_state[field] = merged_list
                else:
                    # For fields not in merge strategies, use simple replacement
                    # This handles custom fields added by tasks
                    merged_state[field] = update_value
            
            return merged_state
            
        except Exception as e:
            self.logger.error(f"Error merging state updates from {task_name}: {e}")
            return base_state
    
    def _is_duplicate_item(self, item: Any, item_list: List[Any]) -> bool:
        """Check if an item is a duplicate in a list based on common ID fields."""
        if not isinstance(item, dict):
            return item in item_list
        
        # Check for common ID fields
        id_fields = ["trader_id", "event_id", "market_id", "id"]
        
        for existing_item in item_list:
            if isinstance(existing_item, dict):
                for id_field in id_fields:
                    if (id_field in item and id_field in existing_item and 
                        item[id_field] == existing_item[id_field]):
                        return True
        
        return False


class TradingGraph:
    """
    Main LangGraph orchestrator for the Quantara trading system.
    
    This class implements a sophisticated multi-agent system using LangGraph's
    StateGraph to coordinate market analysis, sentiment analysis, risk assessment,
    and trader evaluation in a stateful workflow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TradingGraph with agents and workflow configuration.
        
        Args:
            config: Optional configuration dictionary for graph behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize workflow controller
        self.workflow_controller = WorkflowController(self.config.get("workflow", {}))
        
        # Initialize specialized agents
        self.market_discovery_agent = MarketDiscoveryAgent()
        self.sentiment_analysis_agent = SentimentAnalysisAgent()
        self.risk_assessment_agent = RiskAssessmentAgent()
        self.trader_analysis_agent = TraderAnalysisAgent()
        
        # Initialize checkpoint manager for state persistence
        self.checkpointer = MemorySaver()
        
        # Initialize custom checkpoint manager with persistent storage
        checkpoint_config = self.config.get("checkpoint", {})
        storage_type_str = checkpoint_config.get("storage_type", "file")
        storage_type = CheckpointStorageType(storage_type_str) if storage_type_str in [e.value for e in CheckpointStorageType] else CheckpointStorageType.FILE
        
        self.checkpoint_manager = get_checkpoint_manager(
            storage_type=storage_type,
            storage_config=checkpoint_config.get("storage_config", {}),
            retention_hours=checkpoint_config.get("retention_hours", 24),
            compression_enabled=checkpoint_config.get("compression_enabled", True),
            auto_cleanup=checkpoint_config.get("auto_cleanup", True)
        )
        
        # Initialize human-in-the-loop interface
        human_loop_config = self.config.get("human_loop", {})
        decision_callback = human_loop_config.get("decision_callback")
        self.human_loop = HumanLoopInterface(
            config=human_loop_config,
            decision_callback=decision_callback
        )
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Execution statistics
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0
        }
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph with comprehensive workflow control.
        
        Returns:
            Configured StateGraph ready for execution with enhanced orchestration
        """
        # Create the graph with TradingState
        graph = StateGraph(TradingState)
        
        # Add nodes for each workflow step
        graph.add_node("initialize_workflow", self._initialize_workflow_node)
        graph.add_node("query_router", self._route_query)
        graph.add_node("market_discovery", self._market_discovery_node)
        graph.add_node("parallel_analysis", self._parallel_analysis_node)
        graph.add_node("trader_analysis", self._trader_analysis_node)
        graph.add_node("human_decision_point", self._human_decision_point_node)
        graph.add_node("validation_checkpoint", self._validation_checkpoint_node)
        graph.add_node("report_generator", self._generate_report)
        graph.add_node("error_handler", self._handle_errors)
        graph.add_node("insufficient_data_handler", self._handle_insufficient_data)
        
        # Define the workflow edges with enhanced control flow
        graph.set_entry_point("initialize_workflow")
        
        # Initialization flow
        graph.add_edge("initialize_workflow", "query_router")
        graph.add_edge("query_router", "market_discovery")
        
        # Market discovery with conditional branching
        graph.add_conditional_edges(
            "market_discovery",
            self.workflow_controller.should_continue_analysis,
            {
                "continue": "parallel_analysis",
                "error": "error_handler",
                "end": END
            }
        )
        
        # Parallel analysis (sentiment + risk assessment)
        graph.add_edge("parallel_analysis", "trader_analysis")
        
        # Trader analysis with human decision point
        graph.add_edge("trader_analysis", "human_decision_point")
        
        # Human decision point with conditional routing
        graph.add_conditional_edges(
            "human_decision_point",
            self._route_after_human_decision,
            {
                "continue": "validation_checkpoint",
                "abort": END,
                "retry_analysis": "trader_analysis"
            }
        )
        
        # Validation checkpoint with multiple outcomes
        graph.add_conditional_edges(
            "validation_checkpoint",
            self.workflow_controller.should_generate_report,
            {
                "generate": "report_generator",
                "error": "error_handler",
                "insufficient_data": "insufficient_data_handler"
            }
        )
        
        # Terminal nodes
        graph.add_edge("report_generator", END)
        graph.add_edge("error_handler", END)
        graph.add_edge("insufficient_data_handler", END)
        
        # Compile graph with checkpointing
        compiled_graph = graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=self.config.get("interrupt_before", []),
            interrupt_after=self.config.get("interrupt_after", [])
        )
        
        self.logger.info("TradingGraph compiled successfully with enhanced workflow control")
        return compiled_graph
    
    async def analyze_trading_opportunity(
        self,
        query: str,
        preferences: Dict[str, Any],
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete trading analysis workflow with enhanced error handling.
        
        Args:
            query: User's trading query
            preferences: User preferences and configuration
            config: Optional LangGraph execution configuration
            
        Returns:
            Complete analysis results with recommendations
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.execution_stats["total_executions"] += 1
            
            # Create initial state with enhanced initialization
            initial_state = create_initial_state(query, preferences)
            
            # Execute the graph with timeout protection
            execution_config = config or {
                "configurable": {
                    "thread_id": f"trading_analysis_{start_time}",
                    "checkpoint_ns": "quantara_trading"
                }
            }
            
            result = await asyncio.wait_for(
                self.graph.ainvoke(initial_state, config=execution_config),
                timeout=self.workflow_controller.timeout_seconds
            )
            
            # Update execution statistics
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_execution_stats(True, execution_time)
            
            # Format and return results
            return self._format_analysis_results(result, execution_time)
            
        except asyncio.TimeoutError:
            self.logger.error(f"Analysis timed out after {self.workflow_controller.timeout_seconds} seconds")
            self._update_execution_stats(False, self.workflow_controller.timeout_seconds)
            
            return {
                "error": "Analysis timeout",
                "recommendations": [],
                "confidence_score": 0.0,
                "explanation": f"Analysis timed out after {self.workflow_controller.timeout_seconds} seconds",
                "execution_time": self.workflow_controller.timeout_seconds
            }
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Analysis failed: {e}")
            self._update_execution_stats(False, execution_time)
            
            return {
                "error": str(e),
                "recommendations": [],
                "confidence_score": 0.0,
                "explanation": f"Analysis failed: {str(e)}",
                "execution_time": execution_time
            }
    
    async def stream_analysis(
        self,
        query: str,
        preferences: Dict[str, Any],
        config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream analysis results in real-time with enhanced progress tracking.
        
        Args:
            query: User's trading query
            preferences: User preferences and configuration
            config: Optional LangGraph execution configuration
            
        Yields:
            Partial analysis results as they are computed
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            initial_state = create_initial_state(query, preferences)
            
            execution_config = config or {
                "configurable": {
                    "thread_id": f"streaming_analysis_{start_time}",
                    "checkpoint_ns": "quantara_streaming"
                }
            }
            
            step_count = 0
            total_steps = 7  # Estimated total workflow steps
            
            async for chunk in self.graph.astream(initial_state, config=execution_config):
                if chunk and isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        if node_output and isinstance(node_output, dict):
                            step_count += 1
                            
                            # Extract meaningful updates from the chunk
                            streaming_data = self._extract_streaming_data(node_output)
                            
                            yield {
                                "step": node_output.get("current_step", "unknown"),
                                "node": node_name,
                                "data": streaming_data,
                                "progress": min(1.0, step_count / total_steps),
                                "timestamp": asyncio.get_event_loop().time(),
                                "elapsed_time": asyncio.get_event_loop().time() - start_time,
                                "status": "processing"
                            }
            
            # Final completion update
            yield {
                "step": "completed",
                "node": "workflow_complete",
                "data": {"status": "Analysis completed successfully"},
                "progress": 1.0,
                "timestamp": asyncio.get_event_loop().time(),
                "elapsed_time": asyncio.get_event_loop().time() - start_time,
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Streaming analysis failed: {e}")
            yield {
                "step": "error",
                "node": "error_handler",
                "data": {"error": str(e)},
                "progress": 0.0,
                "timestamp": asyncio.get_event_loop().time(),
                "elapsed_time": asyncio.get_event_loop().time() - start_time,
                "status": "error"
            }
    
    def _extract_streaming_data(self, state: TradingState) -> Dict[str, Any]:
        """Extract relevant data for streaming updates with enhanced metrics."""
        try:
            return {
                "market_count": len(state.get("polymarket_data", [])) + len(state.get("kalshi_data", [])),
                "trader_count": len(state.get("trader_scores", [])),
                "confidence": state.get("confidence_score", 0.0),
                "current_step": state.get("current_step", ""),
                "completed_steps": len(state.get("completed_steps", [])),
                "error_count": len(state.get("error_log", [])),
                "niche": state.get("niche", ""),
                "platform_preference": state.get("platform_preference", ""),
                "has_sentiment_data": bool(state.get("sentiment_analysis")),
                "has_risk_data": bool(state.get("risk_assessment")),
                "recommendations_count": len(state.get("recommendations", []))
            }
        except Exception as e:
            self.logger.error(f"Error extracting streaming data: {e}")
            return {"error": "Data extraction failed"}
    
    def _format_analysis_results(self, result: TradingState, execution_time: float) -> Dict[str, Any]:
        """Format analysis results with comprehensive metadata."""
        try:
            return {
                "recommendations": result.get("recommendations", []),
                "confidence_score": result.get("confidence_score", 0.0),
                "explanation": result.get("explanation", ""),
                "market_data": {
                    "polymarket": result.get("polymarket_data", []),
                    "kalshi": result.get("kalshi_data", []),
                    "market_conditions": result.get("market_conditions", {})
                },
                "analysis": {
                    "sentiment": result.get("sentiment_analysis", {}),
                    "risk": result.get("risk_assessment", {}),
                    "trader_scores": result.get("trader_scores", []),
                    "trader_analysis": result.get("trader_analysis", {})
                },
                "workflow_metadata": {
                    "completed_steps": result.get("completed_steps", []),
                    "current_step": result.get("current_step", ""),
                    "error_count": len(result.get("error_log", [])),
                    "execution_time": execution_time,
                    "niche": result.get("niche", ""),
                    "checkpoint_id": result.get("state_checkpoint_id")
                },
                "execution_stats": self.execution_stats.copy()
            }
        except Exception as e:
            self.logger.error(f"Error formatting results: {e}")
            return {
                "error": "Result formatting failed",
                "recommendations": [],
                "confidence_score": 0.0,
                "explanation": f"Result formatting error: {str(e)}",
                "execution_time": execution_time
            }
    
    def _update_execution_stats(self, success: bool, execution_time: float):
        """Update execution statistics."""
        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
        
        # Update average execution time
        total_executions = self.execution_stats["total_executions"]
        current_avg = self.execution_stats["avg_execution_time"]
        
        self.execution_stats["avg_execution_time"] = (
            (current_avg * (total_executions - 1) + execution_time) / total_executions
        )
    
    async def _create_checkpoint_if_enabled(
        self,
        state: TradingState,
        workflow_step: str,
        tags: Optional[List[str]] = None
    ) -> TradingState:
        """
        Create a checkpoint at major workflow phases if checkpointing is enabled.
        
        Args:
            state: Current TradingState
            workflow_step: Current workflow step identifier
            tags: Optional tags for checkpoint categorization
            
        Returns:
            Updated TradingState with checkpoint ID
            
        Requirements: 4.1, 4.2
        """
        try:
            # Check if checkpointing is enabled
            checkpoint_enabled = self.config.get("checkpoint", {}).get("enabled", True)
            checkpoint_frequency = self.workflow_controller.checkpoint_frequency
            
            # Determine if we should create a checkpoint at this step
            major_steps = [
                WorkflowStep.INITIALIZATION.value,
                WorkflowStep.MARKET_DISCOVERY.value,
                WorkflowStep.SENTIMENT_ANALYSIS.value,
                WorkflowStep.TRADER_ANALYSIS.value,
                WorkflowStep.RECOMMENDATION_GENERATION.value,
                WorkflowStep.COMPLETED.value
            ]
            
            should_checkpoint = (
                checkpoint_enabled and
                (checkpoint_frequency == "all_steps" or 
                 (checkpoint_frequency == "major_steps" and workflow_step in major_steps))
            )
            
            if should_checkpoint:
                self.logger.debug(f"Creating checkpoint at workflow step: {workflow_step}")
                
                checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    state=state,
                    workflow_step=workflow_step,
                    tags=tags or []
                )
                
                if checkpoint_id:
                    self.logger.info(f"Checkpoint created successfully: {checkpoint_id}")
                    # State is already updated by create_checkpoint
                else:
                    self.logger.warning(f"Failed to create checkpoint at step: {workflow_step}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error creating checkpoint: {e}")
            # Don't fail the workflow if checkpointing fails
            return state
    
    async def _attempt_recovery_from_checkpoint(
        self,
        checkpoint_id: Optional[str] = None
    ) -> Optional[TradingState]:
        """
        Attempt to recover workflow state from a checkpoint.
        
        Args:
            checkpoint_id: Optional specific checkpoint ID to restore from
            
        Returns:
            Recovered TradingState if successful, None otherwise
            
        Requirements: 4.4, 4.5
        """
        try:
            self.logger.info("Attempting to recover from checkpoint")
            
            if checkpoint_id:
                # Restore specific checkpoint
                state = await self.checkpoint_manager.restore_checkpoint(checkpoint_id, validate=True)
            else:
                # Restore latest valid checkpoint
                state = await self.checkpoint_manager.get_latest_checkpoint()
            
            if state:
                self.logger.info("Successfully recovered state from checkpoint")
                return state
            else:
                self.logger.warning("No valid checkpoint found for recovery")
                return None
                
        except Exception as e:
            self.logger.error(f"Checkpoint recovery failed: {e}")
            return None
    
    # Enhanced Node implementations
    async def _initialize_workflow_node(self, state: TradingState) -> TradingState:
        """Initialize the workflow with enhanced state setup."""
        try:
            self.logger.info("Initializing trading analysis workflow")
            
            updated_state = await update_workflow_progress(
                state, 
                WorkflowStep.INITIALIZATION.value, 
                "workflow_controller"
            )
            
            # Validate initial state
            if not await validate_state_consistency(updated_state):
                self.logger.warning("Initial state validation failed, attempting correction")
                updated_state = await self.workflow_controller.state_manager.ensure_state_consistency(updated_state)
            
            # Set workflow metadata
            updated_state["workflow_metadata"] = {
                "start_time": asyncio.get_event_loop().time(),
                "workflow_version": "1.4.0",
                "graph_config": self.config
            }
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Workflow initialization failed: {e}")
            error_state = state.copy()
            error_state["error_log"].append({
                "step": "initialization",
                "error": str(e),
                "severity": "critical",
                "timestamp": asyncio.get_event_loop().time()
            })
            return error_state
    
    # Node implementations
    async def _route_query(self, state: TradingState) -> TradingState:
        """Route and parse the initial query with enhanced analysis."""
        try:
            self.logger.info(f"Routing query: {state['original_query'][:100]}...")
            
            # Enhanced query parsing
            query = state["original_query"].lower()
            user_preferences = state.get("user_preferences", {})
            
            # Determine query intent with more sophisticated analysis
            query_intent = {
                "query_type": self._determine_query_type(query),
                "platforms": user_preferences.get("preferred_platforms", ["polymarket", "kalshi"]),
                "risk_tolerance": user_preferences.get("risk_tolerance", 0.5),
                "niche_preference": self._extract_niche_preference(query),
                "analysis_depth": self._determine_analysis_depth(query, user_preferences),
                "time_horizon": self._extract_time_horizon(query),
                "focus_areas": self._extract_focus_areas(query)
            }
            
            updated_state = await safe_state_update(
                state,
                {
                    "parsed_intent": query_intent,
                    "current_step": WorkflowStep.MARKET_DISCOVERY.value
                },
                "query_router",
                "query_parsing_and_routing"
            )
            
            if WorkflowStep.MARKET_DISCOVERY.value not in updated_state["completed_steps"]:
                updated_state["completed_steps"].append("query_routing")
            
            self.logger.info(f"Query routed successfully. Intent: {query_intent['query_type']}, Niche: {query_intent['niche_preference']}")
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Query routing failed: {e}")
            error_state = state.copy()
            error_state["error_log"].append({
                "step": "query_routing",
                "error": str(e),
                "severity": "high",
                "timestamp": asyncio.get_event_loop().time()
            })
            return error_state
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of analysis requested."""
        if any(word in query for word in ["trader", "follow", "copy", "best"]):
            return "trader_analysis"
        elif any(word in query for word in ["market", "opportunity", "predict"]):
            return "market_analysis"
        elif any(word in query for word in ["risk", "safe", "conservative"]):
            return "risk_focused"
        else:
            return "general_analysis"
    
    def _extract_niche_preference(self, query: str) -> str:
        """Extract niche preference from query."""
        niche_keywords = {
            "politics": ["election", "president", "vote", "political", "congress"],
            "sports": ["game", "match", "team", "player", "championship"],
            "crypto": ["bitcoin", "ethereum", "crypto", "blockchain", "defi"],
            "economics": ["gdp", "inflation", "market", "stock", "economy"],
            "entertainment": ["movie", "show", "celebrity", "award"],
            "technology": ["tech", "ai", "software", "startup", "ipo"]
        }
        
        for niche, keywords in niche_keywords.items():
            if any(keyword in query for keyword in keywords):
                return niche
        
        return ""
    
    def _determine_analysis_depth(self, query: str, preferences: Dict[str, Any]) -> str:
        """Determine the depth of analysis required."""
        if any(word in query for word in ["detailed", "comprehensive", "thorough"]):
            return "comprehensive"
        elif any(word in query for word in ["quick", "fast", "simple"]):
            return "basic"
        else:
            return preferences.get("default_analysis_depth", "standard")
    
    def _extract_time_horizon(self, query: str) -> str:
        """Extract time horizon from query."""
        if any(word in query for word in ["short", "day", "week"]):
            return "short_term"
        elif any(word in query for word in ["long", "month", "year"]):
            return "long_term"
        else:
            return "medium_term"
    
    def _extract_focus_areas(self, query: str) -> List[str]:
        """Extract specific focus areas from query."""
        focus_areas = []
        
        focus_keywords = {
            "performance": ["roi", "return", "profit", "performance"],
            "risk": ["risk", "safe", "conservative", "volatility"],
            "sentiment": ["sentiment", "news", "social", "buzz"],
            "liquidity": ["liquid", "volume", "trading"]
        }
        
        for area, keywords in focus_keywords.items():
            if any(keyword in query for keyword in keywords):
                focus_areas.append(area)
        
        return focus_areas if focus_areas else ["performance", "risk"]
    
    async def _market_discovery_node(self, state: TradingState) -> TradingState:
        """Execute market discovery with enhanced error handling and validation."""
        try:
            self.logger.info("Starting market discovery phase")
            
            # Update workflow progress
            updated_state = await update_workflow_progress(
                state, 
                WorkflowStep.MARKET_DISCOVERY.value, 
                "market_discovery_agent"
            )
            
            # Execute market discovery steps sequentially with error handling
            try:
                updated_state = await self.market_discovery_agent.discover_markets(updated_state)
                self.logger.info(f"Markets discovered: {len(updated_state.get('polymarket_data', []))} Polymarket, {len(updated_state.get('kalshi_data', []))} Kalshi")
            except Exception as e:
                self.logger.error(f"Market discovery failed: {e}")
                updated_state["error_log"].append({
                    "step": "market_discovery",
                    "substep": "discover_markets",
                    "error": str(e),
                    "severity": "high",
                    "timestamp": asyncio.get_event_loop().time()
                })
            
            try:
                updated_state = await self.market_discovery_agent.classify_niche(updated_state)
                self.logger.info(f"Market niche classified as: {updated_state.get('niche', 'unknown')}")
            except Exception as e:
                self.logger.error(f"Niche classification failed: {e}")
                updated_state["error_log"].append({
                    "step": "market_discovery",
                    "substep": "classify_niche",
                    "error": str(e),
                    "severity": "medium",
                    "timestamp": asyncio.get_event_loop().time()
                })
            
            try:
                updated_state = await self.market_discovery_agent.assess_liquidity(updated_state)
                updated_state = await self.market_discovery_agent.detect_opportunities(updated_state)
                
                opportunities = updated_state.get("market_conditions", {}).get("opportunities", [])
                self.logger.info(f"Opportunities detected: {len(opportunities)}")
                
            except Exception as e:
                self.logger.error(f"Liquidity/opportunity analysis failed: {e}")
                updated_state["error_log"].append({
                    "step": "market_discovery",
                    "substep": "liquidity_analysis",
                    "error": str(e),
                    "severity": "medium",
                    "timestamp": asyncio.get_event_loop().time()
                })
            
            # Validate market discovery results
            market_count = len(updated_state.get("polymarket_data", [])) + len(updated_state.get("kalshi_data", []))
            
            if market_count == 0:
                self.logger.warning("No markets discovered - this may indicate API issues or query mismatch")
                updated_state["error_log"].append({
                    "step": "market_discovery",
                    "error": "No markets found",
                    "severity": "high",
                    "timestamp": asyncio.get_event_loop().time()
                })
            
            # Create checkpoint after market discovery (major workflow phase)
            updated_state = await self._create_checkpoint_if_enabled(
                updated_state,
                WorkflowStep.MARKET_DISCOVERY.value,
                tags=["market_discovery", "post_discovery"]
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Market discovery node failed: {e}")
            error_state = state.copy()
            error_state["error_log"].append({
                "step": "market_discovery",
                "error": str(e),
                "severity": "critical",
                "timestamp": asyncio.get_event_loop().time()
            })
            return error_state
    
    async def _parallel_analysis_node(self, state: TradingState) -> TradingState:
        """Execute sentiment and risk analysis in parallel."""
        try:
            self.logger.info("Starting parallel analysis phase (sentiment + risk)")
            
            # Define parallel tasks
            parallel_tasks = [
                {
                    "name": "sentiment_analysis",
                    "function": self._execute_sentiment_analysis,
                },
                {
                    "name": "risk_assessment", 
                    "function": self._execute_risk_assessment,
                }
            ]
            
            # Execute tasks in parallel using workflow controller
            updated_state = await self.workflow_controller.coordinate_parallel_execution(
                state, parallel_tasks
            )
            
            # Update workflow progress
            updated_state = await update_workflow_progress(
                updated_state,
                WorkflowStep.SENTIMENT_ANALYSIS.value,
                "parallel_analysis_coordinator"
            )
            
            if WorkflowStep.RISK_ASSESSMENT.value not in updated_state["completed_steps"]:
                updated_state["completed_steps"].append(WorkflowStep.RISK_ASSESSMENT.value)
            
            # Create checkpoint after parallel analysis (major workflow phase)
            updated_state = await self._create_checkpoint_if_enabled(
                updated_state,
                WorkflowStep.SENTIMENT_ANALYSIS.value,
                tags=["parallel_analysis", "sentiment", "risk"]
            )
            
            self.logger.info("Parallel analysis completed successfully")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Parallel analysis failed: {e}")
            error_state = state.copy()
            error_state["error_log"].append({
                "step": "parallel_analysis",
                "error": str(e),
                "severity": "high",
                "timestamp": asyncio.get_event_loop().time()
            })
            return error_state
    
    async def _execute_sentiment_analysis(self, state: TradingState) -> TradingState:
        """Execute sentiment analysis steps."""
        updated_state = state.copy()
        
        try:
            updated_state = await self.sentiment_analysis_agent.analyze_sentiment(updated_state)
            updated_state = await self.sentiment_analysis_agent.enrich_with_news(updated_state)
            updated_state = await self.sentiment_analysis_agent.detect_sentiment_shifts(updated_state)
            updated_state = await self.sentiment_analysis_agent.correlate_sentiment_performance(updated_state)
            
            sentiment_data = updated_state.get("sentiment_analysis", {})
            overall_sentiment = sentiment_data.get("overall_sentiment", "neutral")
            self.logger.info(f"Sentiment analysis completed. Overall sentiment: {overall_sentiment}")
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis execution failed: {e}")
            updated_state["error_log"].append({
                "step": "sentiment_analysis",
                "error": str(e),
                "severity": "medium",
                "timestamp": asyncio.get_event_loop().time()
            })
        
        return updated_state
    
    async def _execute_risk_assessment(self, state: TradingState) -> TradingState:
        """Execute risk assessment steps."""
        updated_state = state.copy()
        
        try:
            updated_state = await self.risk_assessment_agent.assess_trader_risk(updated_state)
            updated_state = await self.risk_assessment_agent.calculate_portfolio_risk(updated_state)
            updated_state = await self.risk_assessment_agent.recommend_position_sizing(updated_state)
            updated_state = await self.risk_assessment_agent.monitor_risk_limits(updated_state)
            
            risk_data = updated_state.get("risk_assessment", {})
            portfolio_risk = risk_data.get("portfolio_risk", {})
            overall_risk = portfolio_risk.get("overall_risk_score", 0.0)
            self.logger.info(f"Risk assessment completed. Portfolio risk score: {overall_risk:.2f}")
            
        except Exception as e:
            self.logger.error(f"Risk assessment execution failed: {e}")
            updated_state["error_log"].append({
                "step": "risk_assessment",
                "error": str(e),
                "severity": "medium",
                "timestamp": asyncio.get_event_loop().time()
            })
        
        return updated_state
    
    async def _trader_analysis_node(self, state: TradingState) -> TradingState:
        """Execute comprehensive trader analysis with validation."""
        try:
            self.logger.info("Starting trader analysis phase")
            
            # Update workflow progress
            updated_state = await update_workflow_progress(
                state,
                WorkflowStep.TRADER_ANALYSIS.value,
                "trader_analysis_agent"
            )
            
            # Execute trader analysis steps
            try:
                updated_state = await self.trader_analysis_agent.calculate_trader_scores(updated_state)
                trader_count = len(updated_state.get("trader_scores", []))
                self.logger.info(f"Trader scores calculated for {trader_count} traders")
            except Exception as e:
                self.logger.error(f"Trader score calculation failed: {e}")
                updated_state["error_log"].append({
                    "step": "trader_analysis",
                    "substep": "calculate_scores",
                    "error": str(e),
                    "severity": "high",
                    "timestamp": asyncio.get_event_loop().time()
                })
            
            try:
                updated_state = await self.trader_analysis_agent.rank_traders(updated_state)
                self.logger.info("Trader ranking completed")
            except Exception as e:
                self.logger.error(f"Trader ranking failed: {e}")
                updated_state["error_log"].append({
                    "step": "trader_analysis",
                    "substep": "rank_traders",
                    "error": str(e),
                    "severity": "medium",
                    "timestamp": asyncio.get_event_loop().time()
                })
            
            try:
                updated_state = await self.trader_analysis_agent.validate_trader_data(updated_state)
                updated_state = await self.trader_analysis_agent.generate_confidence_scores(updated_state)
                
                confidence = updated_state.get("confidence_score", 0.0)
                self.logger.info(f"Trader analysis completed with confidence: {confidence:.2f}")
                
            except Exception as e:
                self.logger.error(f"Trader validation/confidence generation failed: {e}")
                updated_state["error_log"].append({
                    "step": "trader_analysis",
                    "substep": "validation_confidence",
                    "error": str(e),
                    "severity": "medium",
                    "timestamp": asyncio.get_event_loop().time()
                })
            
            # Create checkpoint after trader analysis (major workflow phase)
            updated_state = await self._create_checkpoint_if_enabled(
                updated_state,
                WorkflowStep.TRADER_ANALYSIS.value,
                tags=["trader_analysis", "post_analysis"]
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Trader analysis node failed: {e}")
            error_state = state.copy()
            error_state["error_log"].append({
                "step": "trader_analysis",
                "error": str(e),
                "severity": "critical",
                "timestamp": asyncio.get_event_loop().time()
            })
            return error_state
    
    async def _human_decision_point_node(self, state: TradingState) -> TradingState:
        """
        Human-in-the-loop decision point for critical workflow decisions.
        
        This node presents analysis results to human operators and requests
        decisions on how to proceed with the workflow.
        
        Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
        """
        try:
            self.logger.info("Entering human decision point")
            
            # Check if human loop is enabled
            human_loop_enabled = self.config.get("human_loop", {}).get("enabled", False)
            
            if not human_loop_enabled:
                self.logger.info("Human loop disabled, auto-continuing")
                updated_state = state.copy()
                updated_state["human_decision"] = {
                    "decision_made": False,
                    "auto_continue": True,
                    "reason": "Human loop disabled in configuration"
                }
                return updated_state
            
            # Determine if human input is required based on state
            requires_human_input = self._should_request_human_input(state)
            
            if not requires_human_input:
                self.logger.info("Human input not required, auto-continuing")
                updated_state = state.copy()
                updated_state["human_decision"] = {
                    "decision_made": False,
                    "auto_continue": True,
                    "reason": "No critical issues requiring human input"
                }
                return updated_state
            
            # Prepare decision context
            decision_title = "Review Trading Analysis Results"
            decision_description = self._prepare_decision_description(state)
            
            # Create decision options
            options = [
                DecisionOption(
                    id="continue",
                    label="Continue to Report Generation",
                    description="Proceed with generating recommendations based on current analysis",
                    recommended=True,
                    metadata={"action": "continue"}
                ),
                DecisionOption(
                    id="retry_analysis",
                    label="Retry Trader Analysis",
                    description="Re-run trader analysis with adjusted parameters",
                    recommended=False,
                    metadata={"action": "retry"}
                ),
                DecisionOption(
                    id="abort",
                    label="Abort Analysis",
                    description="Stop the analysis and return current results",
                    recommended=False,
                    metadata={"action": "abort"}
                )
            ]
            
            # Determine priority based on state
            priority = self._determine_decision_priority(state)
            
            # Prepare recommendations
            recommendations = {
                "system_recommendation": "continue",
                "reasoning": self._generate_recommendation_reasoning(state),
                "confidence": state.get("confidence_score", 0.0),
                "data_quality": self._assess_data_quality(state)
            }
            
            # Request human decision
            self.logger.info(f"Requesting human decision (priority={priority.value})")
            
            decision_result = await self.human_loop.request_decision(
                decision_type=DecisionType.STRATEGY_CHOICE,
                title=decision_title,
                description=decision_description,
                options=options,
                current_state=state,
                priority=priority,
                timeout_seconds=self.config.get("human_loop", {}).get("decision_timeout", 300),
                default_option_id="continue",
                recommendations=recommendations
            )
            
            # Update state with decision result
            updated_state = state.copy()
            updated_state["human_decision"] = {
                "decision_made": True,
                "selected_option": decision_result.selected_option_id,
                "timed_out": decision_result.timed_out,
                "decision_time": decision_result.decision_time.isoformat(),
                "user_notes": decision_result.user_notes,
                "decision_id": decision_result.decision_id
            }
            
            # Log decision
            self.logger.info(
                f"Human decision received: {decision_result.selected_option_id} "
                f"(timed_out={decision_result.timed_out})"
            )
            
            # Add to completed steps
            if "human_decision_point" not in updated_state["completed_steps"]:
                updated_state["completed_steps"].append("human_decision_point")
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Human decision point failed: {e}")
            error_state = state.copy()
            error_state["error_log"].append({
                "step": "human_decision_point",
                "error": str(e),
                "severity": "medium",
                "timestamp": asyncio.get_event_loop().time()
            })
            # Default to continue on error
            error_state["human_decision"] = {
                "decision_made": False,
                "auto_continue": True,
                "reason": f"Error in human decision point: {str(e)}"
            }
            return error_state
    
    def _should_request_human_input(self, state: TradingState) -> bool:
        """
        Determine if human input should be requested based on state.
        
        Args:
            state: Current TradingState
        
        Returns:
            True if human input is required
        
        Requirements: 10.1
        """
        # Request human input if there are high-severity errors
        error_log = state.get("error_log", [])
        high_severity_errors = [
            e for e in error_log 
            if e.get("severity") in ["high", "critical"]
        ]
        
        if high_severity_errors:
            return True
        
        # Request human input if confidence is low
        confidence = state.get("confidence_score", 1.0)
        if confidence < 0.5:
            return True
        
        # Request human input if risk assessment indicates high risk
        risk_data = state.get("risk_assessment", {})
        portfolio_risk = risk_data.get("portfolio_risk", {})
        overall_risk = portfolio_risk.get("overall_risk_score", 0.0)
        
        if overall_risk > 0.7:  # High risk threshold
            return True
        
        # Check if any risk limits are violated
        risk_violations = risk_data.get("risk_violations", [])
        if risk_violations:
            return True
        
        # Request human input if configured to always ask
        always_ask = self.config.get("human_loop", {}).get("always_ask", False)
        if always_ask:
            return True
        
        return False
    
    def _prepare_decision_description(self, state: TradingState) -> str:
        """
        Prepare a detailed description for the human decision.
        
        Args:
            state: Current TradingState
        
        Returns:
            Formatted description string
        
        Requirements: 10.2
        """
        description_parts = []
        
        # Analysis summary
        market_count = len(state.get("polymarket_data", [])) + len(state.get("kalshi_data", []))
        trader_count = len(state.get("trader_scores", []))
        
        description_parts.append(
            f"Analysis Summary: Analyzed {market_count} markets and scored {trader_count} traders."
        )
        
        # Confidence and quality
        confidence = state.get("confidence_score", 0.0)
        description_parts.append(f"Overall Confidence: {confidence:.1%}")
        
        # Error summary
        error_log = state.get("error_log", [])
        if error_log:
            error_summary = {}
            for error in error_log:
                severity = error.get("severity", "unknown")
                error_summary[severity] = error_summary.get(severity, 0) + 1
            
            error_str = ", ".join([f"{count} {severity}" for severity, count in error_summary.items()])
            description_parts.append(f"Errors Encountered: {error_str}")
        
        # Risk assessment
        risk_data = state.get("risk_assessment", {})
        if risk_data:
            portfolio_risk = risk_data.get("portfolio_risk", {})
            overall_risk = portfolio_risk.get("overall_risk_score", 0.0)
            description_parts.append(f"Portfolio Risk Score: {overall_risk:.1%}")
            
            risk_violations = risk_data.get("risk_violations", [])
            if risk_violations:
                description_parts.append(
                    f"Risk Violations: {len(risk_violations)} limit(s) exceeded"
                )
        
        # Sentiment
        sentiment_data = state.get("sentiment_analysis", {})
        if sentiment_data:
            overall_sentiment = sentiment_data.get("overall_sentiment", "neutral")
            description_parts.append(f"Market Sentiment: {overall_sentiment}")
        
        return "\n".join(description_parts)
    
    def _determine_decision_priority(self, state: TradingState) -> DecisionPriority:
        """
        Determine the priority level for a human decision.
        
        Args:
            state: Current TradingState
        
        Returns:
            DecisionPriority level
        
        Requirements: 10.1
        """
        # Check for critical errors
        error_log = state.get("error_log", [])
        critical_errors = [e for e in error_log if e.get("severity") == "critical"]
        
        if critical_errors:
            return DecisionPriority.CRITICAL
        
        # Check for high-severity errors
        high_errors = [e for e in error_log if e.get("severity") == "high"]
        if high_errors:
            return DecisionPriority.HIGH
        
        # Check for risk violations
        risk_data = state.get("risk_assessment", {})
        risk_violations = risk_data.get("risk_violations", [])
        
        if risk_violations:
            return DecisionPriority.HIGH
        
        # Check confidence level
        confidence = state.get("confidence_score", 1.0)
        if confidence < 0.3:
            return DecisionPriority.HIGH
        elif confidence < 0.5:
            return DecisionPriority.MEDIUM
        
        return DecisionPriority.LOW
    
    def _generate_recommendation_reasoning(self, state: TradingState) -> str:
        """
        Generate reasoning for the system recommendation.
        
        Args:
            state: Current TradingState
        
        Returns:
            Reasoning string
        
        Requirements: 10.2
        """
        reasoning_parts = []
        
        # Analyze confidence
        confidence = state.get("confidence_score", 0.0)
        if confidence >= 0.7:
            reasoning_parts.append("High confidence in analysis results.")
        elif confidence >= 0.5:
            reasoning_parts.append("Moderate confidence in analysis results.")
        else:
            reasoning_parts.append("Low confidence in analysis results.")
        
        # Analyze errors
        error_log = state.get("error_log", [])
        critical_errors = [e for e in error_log if e.get("severity") == "critical"]
        
        if critical_errors:
            reasoning_parts.append("Critical errors detected that may affect reliability.")
        elif len(error_log) > 5:
            reasoning_parts.append("Multiple errors encountered during analysis.")
        elif not error_log:
            reasoning_parts.append("No errors encountered during analysis.")
        
        # Analyze data quality
        trader_count = len(state.get("trader_scores", []))
        if trader_count >= 10:
            reasoning_parts.append(f"Sufficient trader data available ({trader_count} traders).")
        elif trader_count > 0:
            reasoning_parts.append(f"Limited trader data available ({trader_count} traders).")
        else:
            reasoning_parts.append("No trader data available.")
        
        return " ".join(reasoning_parts)
    
    def _assess_data_quality(self, state: TradingState) -> Dict[str, Any]:
        """
        Assess the quality of data in the current state.
        
        Args:
            state: Current TradingState
        
        Returns:
            Data quality assessment
        
        Requirements: 10.2
        """
        quality = {
            "market_data_available": False,
            "trader_data_available": False,
            "sentiment_data_available": False,
            "risk_data_available": False,
            "overall_quality": "unknown"
        }
        
        # Check market data
        market_count = len(state.get("polymarket_data", [])) + len(state.get("kalshi_data", []))
        quality["market_data_available"] = market_count > 0
        quality["market_count"] = market_count
        
        # Check trader data
        trader_count = len(state.get("trader_scores", []))
        quality["trader_data_available"] = trader_count > 0
        quality["trader_count"] = trader_count
        
        # Check sentiment data
        quality["sentiment_data_available"] = bool(state.get("sentiment_analysis"))
        
        # Check risk data
        quality["risk_data_available"] = bool(state.get("risk_assessment"))
        
        # Calculate overall quality
        available_count = sum([
            quality["market_data_available"],
            quality["trader_data_available"],
            quality["sentiment_data_available"],
            quality["risk_data_available"]
        ])
        
        if available_count == 4:
            quality["overall_quality"] = "excellent"
        elif available_count == 3:
            quality["overall_quality"] = "good"
        elif available_count == 2:
            quality["overall_quality"] = "fair"
        elif available_count == 1:
            quality["overall_quality"] = "poor"
        else:
            quality["overall_quality"] = "insufficient"
        
        return quality
    
    def _route_after_human_decision(self, state: TradingState) -> Literal["continue", "abort", "retry_analysis"]:
        """
        Route workflow based on human decision.
        
        Args:
            state: Current TradingState
        
        Returns:
            Routing decision
        
        Requirements: 10.4, 10.5
        """
        human_decision = state.get("human_decision", {})
        
        # If no decision was made (auto-continue), continue
        if not human_decision.get("decision_made", False):
            if human_decision.get("auto_continue", False):
                return "continue"
            else:
                # Default to continue if unclear
                return "continue"
        
        # Route based on selected option
        selected_option = human_decision.get("selected_option", "continue")
        
        if selected_option == "continue":
            return "continue"
        elif selected_option == "abort":
            return "abort"
        elif selected_option == "retry_analysis":
            return "retry_analysis"
        else:
            # Unknown option, default to continue
            self.logger.warning(f"Unknown human decision option: {selected_option}, defaulting to continue")
            return "continue"
    
    async def _validation_checkpoint_node(self, state: TradingState) -> TradingState:
        """Validation checkpoint to ensure data quality before report generation."""
        try:
            self.logger.info("Executing validation checkpoint")
            
            # Comprehensive state validation
            validation_passed = await validate_state_consistency(state)
            
            if not validation_passed:
                self.logger.warning("State consistency validation failed, attempting correction")
                try:
                    corrected_state = await self.workflow_controller.state_manager.ensure_state_consistency(state)
                    validation_passed = await validate_state_consistency(corrected_state)
                    
                    if validation_passed:
                        self.logger.info("State consistency corrected successfully")
                        state = corrected_state
                    else:
                        self.logger.error("State consistency could not be corrected")
                        
                except Exception as e:
                    self.logger.error(f"State correction failed: {e}")
            
            # Data quality checks
            quality_checks = {
                "has_trader_scores": len(state.get("trader_scores", [])) > 0,
                "has_market_data": (len(state.get("polymarket_data", [])) + len(state.get("kalshi_data", []))) > 0,
                "has_sentiment_data": bool(state.get("sentiment_analysis")),
                "has_risk_data": bool(state.get("risk_assessment")),
                "confidence_in_range": 0.0 <= state.get("confidence_score", 0.0) <= 1.0
            }
            
            passed_checks = sum(1 for check in quality_checks.values() if check)
            quality_score = passed_checks / len(quality_checks)
            
            # Update state with validation results
            updated_state = state.copy()
            updated_state["validation_checkpoint"] = {
                "validation_passed": validation_passed,
                "quality_checks": quality_checks,
                "quality_score": quality_score,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Update workflow progress
            updated_state = await update_workflow_progress(
                updated_state,
                WorkflowStep.RECOMMENDATION_GENERATION.value,
                "validation_checkpoint"
            )
            
            self.logger.info(f"Validation checkpoint completed. Quality score: {quality_score:.2f}")
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Validation checkpoint failed: {e}")
            error_state = state.copy()
            error_state["error_log"].append({
                "step": "validation_checkpoint",
                "error": str(e),
                "severity": "high",
                "timestamp": asyncio.get_event_loop().time()
            })
            return error_state
    
    async def _generate_report(self, state: TradingState) -> TradingState:
        """Generate comprehensive analysis report with enhanced formatting."""
        try:
            self.logger.info("Generating comprehensive analysis report")
            
            # Update workflow progress
            updated_state = await update_workflow_progress(
                state,
                WorkflowStep.COMPLETED.value,
                "report_generator"
            )
            
            # Generate enhanced recommendations
            recommendations = []
            
            # Get top traders from analysis with enhanced data
            top_traders = sorted(
                updated_state.get("trader_scores", []),
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:10]  # Top 10 traders
            
            for i, trader_data in enumerate(top_traders):
                trader = trader_data.get("trader", {})
                
                recommendation = {
                    "rank": i + 1,
                    "trader_id": getattr(trader, "trader_id", "unknown"),
                    "platform": getattr(trader, "platform", "unknown"),
                    "score": trader_data.get("score", 0.0),
                    "risk_level": trader_data.get("risk", {}).get("risk_score", 0.0),
                    "roi": getattr(trader, "roi", 0.0),
                    "win_rate": getattr(trader, "win_rate", 0.5),
                    "confidence": trader_data.get("recommendation_confidence", 0.0),
                    "performance_category": trader_data.get("performance_category", "unknown"),
                    "score_breakdown": trader_data.get("score_breakdown", {}),
                    "risk_category": trader_data.get("risk", {}).get("risk_category", "unknown"),
                    "niche_alignment": updated_state.get("niche", ""),
                    "sentiment_alignment": trader_data.get("score_breakdown", {}).get("sentiment_alignment", 0.5)
                }
                
                recommendations.append(recommendation)
            
            # Calculate enhanced confidence score
            if recommendations:
                confidence_scores = [r["confidence"] for r in recommendations if r["confidence"] > 0]
                if confidence_scores:
                    confidence_score = sum(confidence_scores) / len(confidence_scores)
                else:
                    confidence_score = 0.5  # Default moderate confidence
            else:
                confidence_score = 0.0
            
            # Generate comprehensive explanation
            market_count = len(updated_state.get("polymarket_data", [])) + len(updated_state.get("kalshi_data", []))
            sentiment_data = updated_state.get("sentiment_analysis", {})
            risk_data = updated_state.get("risk_assessment", {})
            
            explanation_parts = [
                f"Analyzed {market_count} markets across {len(set(r['platform'] for r in recommendations))} platforms.",
                f"Identified {len(recommendations)} high-quality traders in the {updated_state.get('niche', 'general')} niche."
            ]
            
            if sentiment_data:
                overall_sentiment = sentiment_data.get("overall_sentiment", "neutral")
                explanation_parts.append(f"Market sentiment: {overall_sentiment}.")
            
            if risk_data:
                portfolio_risk = risk_data.get("portfolio_risk", {})
                overall_risk = portfolio_risk.get("overall_risk_score", 0.0)
                explanation_parts.append(f"Portfolio risk assessment: {overall_risk:.1%}.")
            
            explanation_parts.append(f"Recommendation confidence: {confidence_score:.1%}.")
            
            explanation = " ".join(explanation_parts)
            
            # Update state with final results
            updated_state["recommendations"] = recommendations
            updated_state["confidence_score"] = confidence_score
            updated_state["explanation"] = explanation
            
            # Add report metadata
            updated_state["report_metadata"] = {
                "generation_timestamp": asyncio.get_event_loop().time(),
                "total_traders_analyzed": len(updated_state.get("trader_scores", [])),
                "top_recommendations": len(recommendations),
                "analysis_niche": updated_state.get("niche", "general"),
                "market_coverage": {
                    "polymarket": len(updated_state.get("polymarket_data", [])),
                    "kalshi": len(updated_state.get("kalshi_data", []))
                },
                "quality_metrics": {
                    "confidence_score": confidence_score,
                    "data_completeness": len([r for r in recommendations if r["confidence"] > 0.5]) / max(1, len(recommendations)),
                    "error_count": len(updated_state.get("error_log", []))
                }
            }
            
            self.logger.info(f"Report generated successfully with {len(recommendations)} recommendations")
            
            # Create final checkpoint after report generation (workflow completion)
            updated_state = await self._create_checkpoint_if_enabled(
                updated_state,
                WorkflowStep.COMPLETED.value,
                tags=["report_generation", "completed", "final"]
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            error_state = state.copy()
            error_state["error_log"].append({
                "step": "report_generation",
                "error": str(e),
                "severity": "high",
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Provide fallback minimal report
            error_state["recommendations"] = []
            error_state["confidence_score"] = 0.0
            error_state["explanation"] = f"Report generation failed: {str(e)}"
            
            return error_state
    
    async def _handle_errors(self, state: TradingState) -> TradingState:
        """Enhanced error handling with recovery strategies."""
        try:
            self.logger.info("Handling workflow errors")
            
            error_log = state.get("error_log", [])
            critical_errors = [e for e in error_log if e.get("severity") == "critical"]
            high_errors = [e for e in error_log if e.get("severity") == "high"]
            
            # Categorize errors
            error_summary = {
                "total_errors": len(error_log),
                "critical_errors": len(critical_errors),
                "high_errors": len(high_errors),
                "medium_errors": len([e for e in error_log if e.get("severity") == "medium"]),
                "low_errors": len([e for e in error_log if e.get("severity") == "low"])
            }
            
            # Generate error-based explanation
            if critical_errors:
                explanation = f"Analysis failed due to {len(critical_errors)} critical errors. System recovery required."
                confidence_score = 0.0
            elif high_errors:
                explanation = f"Analysis completed with {len(high_errors)} significant issues. Results may be incomplete."
                confidence_score = 0.2
            else:
                explanation = f"Analysis completed with {len(error_log)} minor issues. Results are generally reliable."
                confidence_score = 0.6
            
            # Update state with error handling results
            error_state = state.copy()
            error_state["current_step"] = WorkflowStep.ERROR.value
            error_state["recommendations"] = []
            error_state["confidence_score"] = confidence_score
            error_state["explanation"] = explanation
            error_state["error_summary"] = error_summary
            
            self.logger.warning(f"Error handling completed. {error_summary}")
            
            return error_state
            
        except Exception as e:
            self.logger.error(f"Error handler itself failed: {e}")
            
            # Ultimate fallback
            fallback_state = state.copy()
            fallback_state["current_step"] = "critical_failure"
            fallback_state["recommendations"] = []
            fallback_state["confidence_score"] = 0.0
            fallback_state["explanation"] = f"Critical system failure: {str(e)}"
            
            return fallback_state
    
    async def _handle_insufficient_data(self, state: TradingState) -> TradingState:
        """Handle cases where insufficient data is available for analysis."""
        try:
            self.logger.info("Handling insufficient data scenario")
            
            # Analyze what data is missing
            data_availability = {
                "market_data": len(state.get("polymarket_data", [])) + len(state.get("kalshi_data", [])),
                "trader_scores": len(state.get("trader_scores", [])),
                "sentiment_data": bool(state.get("sentiment_analysis")),
                "risk_data": bool(state.get("risk_assessment"))
            }
            
            missing_data = [key for key, value in data_availability.items() if not value]
            
            explanation = f"Insufficient data for reliable analysis. Missing: {', '.join(missing_data)}. "
            
            # Provide guidance based on what's missing
            if not data_availability["market_data"]:
                explanation += "No markets found matching your criteria. Try broadening your search or checking different platforms."
            elif not data_availability["trader_scores"]:
                explanation += "No trader data available for analysis. This may indicate API connectivity issues."
            else:
                explanation += "Partial data available but insufficient for confident recommendations."
            
            # Update state
            insufficient_data_state = state.copy()
            insufficient_data_state["current_step"] = "insufficient_data"
            insufficient_data_state["recommendations"] = []
            insufficient_data_state["confidence_score"] = 0.0
            insufficient_data_state["explanation"] = explanation
            insufficient_data_state["data_availability"] = data_availability
            
            self.logger.warning(f"Insufficient data handling completed. Available data: {data_availability}")
            
            return insufficient_data_state
            
        except Exception as e:
            self.logger.error(f"Insufficient data handler failed: {e}")
            
            # Fallback to error handler
            return await self._handle_errors(state)
    
    # Remove old conditional edge functions as they're now in WorkflowController
    # Keep for backward compatibility but delegate to controller
    def _should_continue_analysis(self, state: TradingState) -> str:
        """Legacy method - delegates to workflow controller."""
        import asyncio
        return asyncio.run(self.workflow_controller.should_continue_analysis(state))
    
    def _should_generate_report(self, state: TradingState) -> str:
        """Legacy method - delegates to workflow controller.""" 
        import asyncio
        return asyncio.run(self.workflow_controller.should_generate_report(state))