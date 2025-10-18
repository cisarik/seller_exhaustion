"""
Integration layer between GA and Evolution Coach LLM.

Converts GA state to Coach requests, applies Coach recommendations to GA configuration.
"""

import asyncio
from typing import Optional, List, Tuple
from dataclasses import asdict
import numpy as np

from backtest.coach_protocol import EvolutionState, CoachAnalysis, CoachRecommendation, RecommendationCategory
from backtest.llm_coach import GemmaCoachClient
from backtest.optimizer import Population
from core.models import FitnessConfig, OptimizationConfig, Timeframe


def build_evolution_state(
    population: Population,
    fitness_config: FitnessConfig,
    ga_config: OptimizationConfig,
    raw_logs: Optional[List[str]] = None
) -> EvolutionState:
    """
    Build EvolutionState from current GA population and configuration.
    
    This converts GA state into format understood by Coach.
    
    Args:
        population: Current population
        fitness_config: Fitness configuration
        ga_config: GA configuration
        raw_logs: Optional raw log lines for context
    
    Returns:
        EvolutionState ready to send to Coach
    """
    # Get population statistics
    stats = population.get_stats()
    trade_counts = [ind.metrics.get('n', 0) for ind in population.individuals]
    
    # Calculate effective min_trades for this generation
    effective_min_trades = fitness_config.get_effective_min_trades(population.generation)
    
    # Count individuals below minimum requirement
    below_min_trades_count = sum(1 for tc in trade_counts if tc < effective_min_trades)
    below_min_trades_percent = 100 * below_min_trades_count / len(population.individuals)
    
    # Get best individual metrics
    best = population.best_ever or population.get_best()
    best_metrics = best.metrics if best else {'n': 0, 'win_rate': 0.0, 'avg_R': 0.0, 'total_pnl': 0.0}
    
    # Calculate recent improvement
    recent_improvement = None
    if len(population.history) >= 2:
        recent_bests = [h['best_fitness'] for h in population.history[-5:]]
        recent_improvement = max(recent_bests) - min(recent_bests) if recent_bests else 0.0
    
    # Check stagnation
    is_stagnant = False
    if len(population.history) >= ga_config.stagnation_threshold:
        recent_bests = [h['best_fitness'] for h in population.history[-ga_config.stagnation_threshold:]]
        best_improvement = max(recent_bests) - min(recent_bests)
        is_stagnant = best_improvement < ga_config.stagnation_fitness_tolerance
    
    # Get diversity
    diversity = population.get_diversity_metric() if ga_config.track_diversity else 1.0
    
    # Create state object
    state = EvolutionState(
        generation=population.generation,
        population_size=population.size,
        mean_fitness=float(stats['mean_fitness']),
        std_fitness=float(stats['std_fitness']),
        best_fitness=float(stats['max_fitness']),
        best_trades=int(best_metrics.get('n', 0)),
        best_win_rate=float(best_metrics.get('win_rate', 0.0)),
        best_avg_r=float(best_metrics.get('avg_R', 0.0)),
        best_pnl=float(best_metrics.get('total_pnl', 0.0)),
        below_min_trades_percent=below_min_trades_percent,
        mean_trade_count=float(np.mean(trade_counts)),
        diversity_metric=float(diversity),
        recent_improvement=recent_improvement,
        is_stagnant=is_stagnant,
        fitness_config_dict=fitness_config.dict(),
        ga_config_dict=ga_config.dict()
    )
    
    return state


async def get_coach_analysis(
    population: Population,
    fitness_config: FitnessConfig,
    ga_config: OptimizationConfig,
    raw_logs: Optional[List[str]] = None,
    base_url: str = "http://localhost:1234",
    model: str = "gemma-2-9b-it"
) -> Optional[CoachAnalysis]:
    """
    Get Coach analysis for current evolution state.
    
    Args:
        population: Current population
        fitness_config: Fitness configuration
        ga_config: GA configuration
        raw_logs: Optional raw log lines
        base_url: LM Studio URL
        model: Model name
    
    Returns:
        CoachAnalysis or None if error/no connection
    """
    # Build evolution state
    state = build_evolution_state(population, fitness_config, ga_config, raw_logs)
    
    # Call coach
    coach = GemmaCoachClient(base_url=base_url, model=model, verbose=True)
    analysis = await coach.analyze_evolution(state, raw_logs)
    
    return analysis


def apply_coach_recommendations(
    analysis: CoachAnalysis,
    fitness_config: FitnessConfig,
    ga_config: OptimizationConfig
) -> Tuple[FitnessConfig, OptimizationConfig]:
    """
    Apply Coach recommendations to GA configuration.
    
    Returns new configs with Coach's changes applied.
    
    Args:
        analysis: Coach analysis with recommendations
        fitness_config: Current fitness configuration
        ga_config: Current GA configuration
    
    Returns:
        (new_fitness_config, new_ga_config) with recommendations applied
    """
    # Deep copy to avoid mutation
    new_fitness = fitness_config.copy(deep=True)
    new_ga = ga_config.copy(deep=True)
    
    applied_count = 0
    
    for rec in analysis.recommendations:
        try:
            if rec.category == RecommendationCategory.FITNESS_WEIGHTS:
                # Apply to fitness weights
                if hasattr(new_fitness, rec.parameter):
                    old_val = getattr(new_fitness, rec.parameter)
                    setattr(new_fitness, rec.parameter, rec.suggested_value)
                    print(f"✅ Applied: fitness.{rec.parameter} = {rec.suggested_value} (was {old_val})")
                    applied_count += 1
            
            elif rec.category == RecommendationCategory.FITNESS_PENALTIES:
                # Apply penalty strength parameters
                if hasattr(new_fitness, rec.parameter):
                    old_val = getattr(new_fitness, rec.parameter)
                    setattr(new_fitness, rec.parameter, rec.suggested_value)
                    print(f"✅ Applied: fitness.{rec.parameter} = {rec.suggested_value:.2f} (was {old_val:.2f})")
                    applied_count += 1
            
            elif rec.category == RecommendationCategory.FITNESS_FUNCTION_TYPE:
                # Switch fitness function type
                if rec.parameter == "fitness_function_type":
                    old_val = new_fitness.fitness_function_type
                    new_fitness.fitness_function_type = rec.suggested_value
                    print(f"✅ Applied: fitness.fitness_function_type = '{rec.suggested_value}' (was '{old_val}')")
                    applied_count += 1
            
            elif rec.category == RecommendationCategory.FITNESS_GATES:
                # Adjust minimum requirements
                if hasattr(new_fitness, rec.parameter):
                    old_val = getattr(new_fitness, rec.parameter)
                    setattr(new_fitness, rec.parameter, rec.suggested_value)
                    print(f"✅ Applied: fitness.{rec.parameter} = {rec.suggested_value} (was {old_val})")
                    applied_count += 1
            
            elif rec.category == RecommendationCategory.CURRICULUM:
                # Enable/configure curriculum learning
                if hasattr(new_fitness, rec.parameter):
                    old_val = getattr(new_fitness, rec.parameter)
                    setattr(new_fitness, rec.parameter, rec.suggested_value)
                    print(f"✅ Applied: fitness.{rec.parameter} = {rec.suggested_value} (was {old_val})")
                    applied_count += 1
            
            elif rec.category == RecommendationCategory.GA_HYPERPARAMS:
                # Apply to GA hyperparameters
                if hasattr(new_ga, rec.parameter):
                    old_val = getattr(new_ga, rec.parameter)
                    setattr(new_ga, rec.parameter, rec.suggested_value)
                    print(f"✅ Applied: ga.{rec.parameter} = {rec.suggested_value} (was {old_val})")
                    applied_count += 1
            
            elif rec.category == RecommendationCategory.DIVERSITY:
                # Apply diversity settings
                if hasattr(new_ga, rec.parameter):
                    old_val = getattr(new_ga, rec.parameter)
                    setattr(new_ga, rec.parameter, rec.suggested_value)
                    print(f"✅ Applied: ga.{rec.parameter} = {rec.suggested_value} (was {old_val})")
                    applied_count += 1
            
            elif rec.category == RecommendationCategory.BOUNDS:
                # Apply bounds override
                if rec.parameter == "bounds" and isinstance(rec.suggested_value, dict):
                    new_ga.override_bounds = rec.suggested_value
                    print(f"✅ Applied: bounds override for {len(rec.suggested_value)} parameters")
                    applied_count += 1
        
        except Exception as e:
            print(f"⚠️  Failed to apply recommendation {rec.parameter}: {e}")
    
    print(f"\n📊 Applied {applied_count}/{len(analysis.recommendations)} recommendations")
    
    return new_fitness, new_ga


def format_coach_output(analysis: CoachAnalysis) -> str:
    """Format Coach analysis for console display."""
    output = f"""
╔════════════════════════════════════════════════════════════╗
║              EVOLUTION COACH ANALYSIS (Gen {analysis.generation:03d})              ║
╚════════════════════════════════════════════════════════════╝

📋 SUMMARY:
{analysis.summary}

⚠️  FLAGS:
  • Stagnation: {'Yes' if analysis.stagnation_detected else 'No'}
  • Diversity: {'Concern' if analysis.diversity_concern else 'Good'}
  • Overall: {analysis.overall_assessment.upper()}

📌 RECOMMENDATIONS ({len(analysis.recommendations)}):
"""
    
    for i, rec in enumerate(analysis.recommendations, 1):
        output += f"\n  {i}. {rec.parameter.upper()}"
        output += f"\n     Category: {rec.category.value}"
        output += f"\n     Current: {rec.current_value}"
        output += f"\n     Suggested: {rec.suggested_value}"
        output += f"\n     Confidence: {rec.confidence:.0%}"
        output += f"\n     Reason: {rec.reasoning[:60]}..."
    
    output += f"\n\n🎯 NEXT STEPS:\n"
    for step in analysis.next_steps:
        output += f"  • {step}\n"
    
    return output


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("Coach Integration Module")
    print("Provides bidirectional Coach ↔ GA communication")
    print("\nAvailable functions:")
    print("  - build_evolution_state()")
    print("  - get_coach_analysis()")
    print("  - apply_coach_recommendations()")
    print("  - format_coach_output()")
