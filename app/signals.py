"""
Unified signals-based architecture for coach window communication.

This module provides a centralized signals emitter that coordinates
real-time updates between the stats panel, coach managers, and the
unified optimization coach window.
"""

from PySide6.QtCore import QObject, Signal
from typing import Any, Dict, Optional


class CoachSignals(QObject):
    """Centralized signals for coach window updates."""
    
    # Coach analysis update signals
    coach_analysis_started = Signal()  # Coach analysis started
    coach_analysis_complete = Signal(dict)  # Coach analysis complete with data
    coach_message = Signal(str, str)  # (message, message_type: 'info'|'warning'|'error'|'blue')
    
    # Population state signals  
    population_state_updated = Signal(dict)  # Population metrics updated
    fitness_history_updated = Signal(list)  # Fitness history updated
    
    # Recommendation signals
    recommendation_updated = Signal(dict)  # Current recommendation data
    recommendation_history_updated = Signal(list)  # Recommendation history
    
    # Decision reasoning signals (Classic Coach detailed info)
    decision_reasoning_updated = Signal(dict)  # (reasoning, parameters_adjusted)
    phase_info_updated = Signal(dict)  # Phase boundaries, current phase, prediction
    crisis_detection_updated = Signal(dict)  # Crisis info (active_crises, generation)
    learning_insights_updated = Signal(dict)  # Learning insights and patterns
    
    # Tool call signals (for OpenAI agents)
    tool_call_started = Signal(str, dict)  # (tool_name, parameters)
    tool_call_complete = Signal(str, dict, dict)  # (tool_name, parameters, response)
    
    # Progress signals
    progress_updated = Signal(int, int, str)  # (current, total, message)
    progress_hidden = Signal()  # Hide progress bar
    
    # Coach mode signals
    coach_mode_changed = Signal(str)  # Coach mode changed (classic|openai|disabled)
    coach_enabled_changed = Signal(bool)  # Coach enabled/disabled
    
    # Coach window lifecycle signals
    coach_window_requested = Signal(str)  # Open coach window with mode
    coach_window_closed = Signal()  # Coach window closed
    
    # Real-time status signals for main UI
    coach_activity_started = Signal(str)  # Activity type (e.g., "analyzing", "analyzing_population")
    coach_activity_completed = Signal()  # Activity completed


class OptimizationSignals(QObject):
    """Signals for optimization process."""
    
    # Generation signals
    generation_started = Signal(int)  # Generation number
    generation_complete = Signal(int, dict)  # (generation, metrics)
    
    # Best individual signals
    best_individual_updated = Signal(object, dict)  # (individual, backtest_result)
    
    # Optimization state signals
    optimization_started = Signal(str)  # Optimization type (ga|adam|multi_step)
    optimization_paused = Signal()
    optimization_resumed = Signal()
    optimization_completed = Signal(object, dict)  # (best_individual, final_metrics)
    optimization_error = Signal(str)  # Error message


# Global signal instances
_coach_signals = None
_optimization_signals = None


def get_coach_signals() -> CoachSignals:
    """Get the global coach signals instance."""
    global _coach_signals
    if _coach_signals is None:
        _coach_signals = CoachSignals()
    return _coach_signals


def get_optimization_signals() -> OptimizationSignals:
    """Get the global optimization signals instance."""
    global _optimization_signals
    if _optimization_signals is None:
        _optimization_signals = OptimizationSignals()
    return _optimization_signals
