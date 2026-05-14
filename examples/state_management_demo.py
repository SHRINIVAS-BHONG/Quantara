#!/usr/bin/env python3
"""
Demonstration of the enhanced TradingState management system.

This script shows how the state management system works with validation,
consistency checks, concurrent access, and debugging support.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantara.langgraph.core.state import (
    TradingState,
    WorkflowStep,
    create_initial_state,
    update_workflow_progress,
    safe_state_update,
    get_state_debug_info,
    get_state_manager
)


async def demonstrate_basic_workflow():
    """Demonstrate basic workflow with state management."""
    print("=== Basic Workflow Demonstration ===")
    
    # Create initial state
    query = "Find high-performing crypto prediction market traders"
    preferences = {
        "preferred_platforms": ["polymarket", "kalshi"],
        "risk_tolerance": 0.6,
        "min_roi": 0.15
    }
    
    state = create_initial_state(query, preferences)
    print(f"Initial state created: {state['current_step']}")
    
    # Progress through workflow steps
    steps = [
        WorkflowStep.MARKET_DISCOVERY,
        WorkflowStep.SENTIMENT_ANALYSIS,
        WorkflowStep.RISK_ASSESSMENT,
        WorkflowStep.TRADER_ANALYSIS,
        WorkflowStep.RECOMMENDATION_GENERATION
    ]
    
    for step in steps:
        state = await update_workflow_progress(state, step.value, f"{step.value}_agent")
        print(f"Progressed to: {state['current_step']}")
        
        # Add some sample data for each step
        if step == WorkflowStep.MARKET_DISCOVERY:
            updates = {
                "niche": "crypto",
                "polymarket_data": [{"market_id": "pm_123", "volume": 50000}],
                "kalshi_data": [{"market_id": "k_456", "volume": 30000}]
            }
        elif step == WorkflowStep.SENTIMENT_ANALYSIS:
            updates = {
                "sentiment_analysis": {
                    "overall_sentiment": 0.7,
                    "confidence": 0.8
                }
            }
        elif step == WorkflowStep.RISK_ASSESSMENT:
            updates = {
                "risk_assessment": {
                    "overall_risk_score": 0.3,
                    "risk_category": "moderate"
                }
            }
        elif step == WorkflowStep.TRADER_ANALYSIS:
            updates = {
                "trader_scores": [
                    {
                        "trader": {"trader_id": "trader_001", "platform": "polymarket"},
                        "score": 0.85,
                        "roi": 0.22,
                        "win_rate": 0.78
                    }
                ]
            }
        elif step == WorkflowStep.RECOMMENDATION_GENERATION:
            updates = {
                "recommendations": [
                    {
                        "trader_id": "trader_001",
                        "confidence": 0.85,
                        "recommendation_type": "follow"
                    }
                ],
                "confidence_score": 0.82
            }
        
        state = await safe_state_update(state, updates, f"{step.value}_agent", f"{step.value}_data_update")
        print(f"  Added data for {step.value}")
    
    # Show final state summary
    debug_info = get_state_debug_info(state)
    print(f"\nWorkflow completed!")
    print(f"Final confidence score: {state['confidence_score']}")
    print(f"Recommendations: {len(state['recommendations'])}")
    print(f"State transitions: {debug_info['modification_count']}")
    print(f"Completed steps: {len(state['completed_steps'])}")
    
    return state


async def demonstrate_concurrent_operations():
    """Demonstrate concurrent state operations."""
    print("\n=== Concurrent Operations Demonstration ===")
    
    state = create_initial_state("Concurrent test query", {"risk_tolerance": 0.5})
    state = await update_workflow_progress(state, WorkflowStep.MARKET_DISCOVERY.value, "system")
    
    async def agent_work(agent_id: str, data_key: str, data_value: Any):
        """Simulate agent work with state updates."""
        updates = {data_key: data_value}
        return await safe_state_update(state, updates, agent_id, f"{agent_id}_operation")
    
    # Run multiple agents concurrently
    print("Running concurrent agent operations...")
    tasks = [
        agent_work("polymarket_agent", "polymarket_data", [{"market": "crypto_1"}]),
        agent_work("kalshi_agent", "kalshi_data", [{"market": "crypto_2"}]),
        agent_work("sentiment_agent", "sentiment_analysis", {"score": 0.6}),
        agent_work("risk_agent", "risk_assessment", {"score": 0.4})
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check results
    successful_operations = sum(1 for r in results if not isinstance(r, Exception))
    print(f"Successful concurrent operations: {successful_operations}/{len(tasks)}")
    
    # Get final state from any successful result
    final_state = next(r for r in results if not isinstance(r, Exception))
    debug_info = get_state_debug_info(final_state)
    print(f"Total state modifications: {debug_info['modification_count']}")
    
    return final_state


async def demonstrate_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n=== Error Handling Demonstration ===")
    
    state = create_initial_state("Error test query", {"risk_tolerance": 0.5})
    
    # Try invalid updates
    print("Testing invalid data handling...")
    
    try:
        invalid_updates = {
            "confidence_score": 2.0,  # Out of bounds
            "trader_scores": [{"invalid": "structure"}]  # Invalid structure
        }
        
        state = await safe_state_update(state, invalid_updates, "error_agent", "invalid_update")
        
        # Check if invalid data was corrected
        print(f"Confidence score corrected to: {state['confidence_score']}")
        print(f"Invalid trader scores removed: {len(state['trader_scores'])} remaining")
        
    except Exception as e:
        print(f"Error handling failed: {e}")
    
    # Verify state is still valid
    state_manager = get_state_manager()
    is_valid = await state_manager.validate_state(state, strict=False)
    print(f"State is valid after error handling: {is_valid}")
    
    return state


async def demonstrate_debugging_features():
    """Demonstrate debugging and monitoring features."""
    print("\n=== Debugging Features Demonstration ===")
    
    state = create_initial_state("Debug test query", {"risk_tolerance": 0.5})
    
    # Perform several operations to generate debug data
    operations = [
        (WorkflowStep.MARKET_DISCOVERY.value, {"niche": "crypto"}),
        (WorkflowStep.SENTIMENT_ANALYSIS.value, {"sentiment_analysis": {"score": 0.7}}),
        (WorkflowStep.RISK_ASSESSMENT.value, {"risk_assessment": {"score": 0.3}})
    ]
    
    for step, updates in operations:
        state = await update_workflow_progress(state, step, f"{step}_agent")
        state = await safe_state_update(state, updates, f"{step}_agent", f"{step}_operation")
    
    # Get comprehensive debug information
    debug_info = get_state_debug_info(state)
    
    print("Debug Information:")
    print(f"  State version: {debug_info['state_version']}")
    print(f"  Last modified: {debug_info['last_modified']}")
    print(f"  Total modifications: {debug_info['modification_count']}")
    print(f"  Recent transitions: {len(debug_info['recent_transitions'])}")
    print(f"  Validation summary: {debug_info['validation_summary']['total_validations']} validations")
    print(f"  Consistency status: {debug_info['consistency_status']}")
    
    # Show recent transitions
    print("\nRecent State Transitions:")
    for i, transition in enumerate(debug_info['recent_transitions'][-3:], 1):
        print(f"  {i}. {transition['from_step']} -> {transition['to_step']} by {transition['agent_id']}")
        print(f"     Operation: {transition['operation']}")
        print(f"     Timestamp: {transition['timestamp']}")
    
    return state


async def demonstrate_state_validation():
    """Demonstrate comprehensive state validation."""
    print("\n=== State Validation Demonstration ===")
    
    state_manager = get_state_manager()
    
    # Create a valid state
    state = create_initial_state("Validation test", {"risk_tolerance": 0.5})
    
    # Test validation on valid state
    is_valid = await state_manager.validate_state(state, strict=True)
    print(f"Valid state passes validation: {is_valid}")
    
    # Test validation on invalid state
    invalid_state = state.copy()
    invalid_state["confidence_score"] = 1.5  # Invalid
    del invalid_state["original_query"]  # Missing required field
    
    is_valid = await state_manager.validate_state(invalid_state, strict=False)
    print(f"Invalid state fails validation: {not is_valid}")
    
    # Test consistency enforcement
    print("Testing consistency enforcement...")
    try:
        consistent_state = await state_manager.ensure_state_consistency(invalid_state)
        print("Consistency enforcement succeeded")
        
        # Check if issues were fixed
        final_valid = await state_manager.validate_state(consistent_state, strict=False)
        print(f"State is valid after consistency enforcement: {final_valid}")
        
    except Exception as e:
        print(f"Consistency enforcement failed: {e}")
    
    return state


async def main():
    """Run all demonstrations."""
    print("Enhanced TradingState Management System Demonstration")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        await demonstrate_basic_workflow()
        await demonstrate_concurrent_operations()
        await demonstrate_error_handling()
        await demonstrate_debugging_features()
        await demonstrate_state_validation()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())