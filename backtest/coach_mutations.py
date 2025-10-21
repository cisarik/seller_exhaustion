"""
Coach Mutation Operations

Tools for manual and coach-recommended mutations:
- Mutate individual parameters
- Drop individuals from population
- Insert new individuals from coach recommendations
- Apply coach recommendations to population

Maintains mutation history and logging.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from backtest.optimizer import Population, Individual
from backtest.coach_session import CoachAnalysisSession
from backtest.coach_protocol import CoachAnalysis, CoachRecommendation
from core.models import Timeframe
import logging

logger = logging.getLogger(__name__)


@dataclass
class MutationRecord:
    """Record of a mutation operation."""
    timestamp: datetime
    session_id: str
    operation: str  # "mutate", "drop", "insert"
    individual_id: int
    parameter_name: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    reason: Optional[str] = None
    
    def __repr__(self) -> str:
        if self.operation == "mutate":
            return (
                f"[{self.session_id}] Mutate Ind#{self.individual_id:02d} "
                f"{self.parameter_name}: {self.old_value} → {self.new_value}"
            )
        elif self.operation == "drop":
            return f"[{self.session_id}] Drop Ind#{self.individual_id:02d}"
        elif self.operation == "insert":
            return f"[{self.session_id}] Insert new individual"
        return f"[{self.session_id}] {self.operation}"


class CoachMutationManager:
    """
    Manages mutations to population from coach recommendations.
    
    Features:
    - Apply coach recommendations to population
    - Manual parameter mutations
    - Individual dropping/insertion
    - Mutation history tracking
    """
    
    def __init__(self, verbose: bool = True):
        self.mutation_history: List[MutationRecord] = []
        self.verbose = verbose
    
    def mutate_individual_parameter(
        self,
        population: Population,
        session: CoachAnalysisSession,
        individual_id: int,
        parameter_name: str,
        new_value: Any,
        reason: str = ""
    ) -> bool:
        """
        Mutate a single parameter of an individual.
        
        Args:
            population: Population to mutate
            session: Analysis session for tracking
            individual_id: Individual ID (0-indexed)
            parameter_name: Parameter to mutate (e.g., "ema_fast")
            new_value: New value
            reason: Reason for mutation (for logging)
        
        Returns:
            True if mutation successful
        """
        if individual_id < 0 or individual_id >= len(population.individuals):
            logger.warning(f"Invalid individual ID: {individual_id}")
            return False
        
        individual = population.individuals[individual_id]
        
        # Find and update parameter
        old_value = None
        parameter_found = False
        
        # Try seller_params first
        if hasattr(individual, 'seller_params'):
            if hasattr(individual.seller_params, parameter_name):
                old_value = getattr(individual.seller_params, parameter_name)
                setattr(individual.seller_params, parameter_name, new_value)
                parameter_found = True
        
        # Try backtest_params
        if not parameter_found and hasattr(individual, 'backtest_params'):
            if hasattr(individual.backtest_params, parameter_name):
                old_value = getattr(individual.backtest_params, parameter_name)
                setattr(individual.backtest_params, parameter_name, new_value)
                parameter_found = True
        
        if not parameter_found:
            logger.warning(f"Parameter not found: {parameter_name}")
            return False
        
        # Reset fitness to force re-evaluation
        individual.fitness = 0.0
        individual.metrics = {}
        
        # Record mutation
        record = MutationRecord(
            timestamp=datetime.utcnow(),
            session_id=session.session_id,
            operation="mutate",
            individual_id=individual_id,
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason
        )
        self.mutation_history.append(record)
        
        # Log
        log_msg = f"[MUTATE ] Ind#{individual_id:02d} {parameter_name}: {old_value} → {new_value}"
        if reason:
            log_msg += f" ({reason})"
        print(log_msg)
        
        if self.verbose:
            print(f"✏️  {log_msg}")
        
        return True
    
    def drop_individual(
        self,
        population: Population,
        session: CoachAnalysisSession,
        individual_id: int,
        reason: str = "Coach recommendation"
    ) -> bool:
        """
        Drop individual from population (remove and re-index).
        
        Args:
            population: Population to mutate
            session: Analysis session for tracking
            individual_id: Individual ID to drop
            reason: Reason for dropping
        
        Returns:
            True if drop successful
        """
        if individual_id < 0 or individual_id >= len(population.individuals):
            logger.warning(f"Invalid individual ID: {individual_id}")
            return False
        
        # Record mutation BEFORE removal
        record = MutationRecord(
            timestamp=datetime.utcnow(),
            session_id=session.session_id,
            operation="drop",
            individual_id=individual_id,
            reason=reason
        )
        self.mutation_history.append(record)
        
        # Remove individual
        population.individuals.pop(individual_id)
        
        # Log
        log_msg = f"[DROP   ] Removed Ind#{individual_id:02d} ({reason})"
        print(log_msg)
        
        if self.verbose:
            print(f"🗑️  {log_msg}")
        
        return True
    
    def insert_individual(
        self,
        population: Population,
        session: CoachAnalysisSession,
        new_individual: Individual,
        reason: str = "Coach recommendation",
        position: Optional[int] = None
    ) -> int:
        """
        Insert new individual into population.
        
        Args:
            population: Population to mutate
            session: Analysis session for tracking
            new_individual: New individual to insert
            reason: Reason for insertion
            position: Where to insert (None = append)
        
        Returns:
            New individual ID (position in population)
        """
        if position is None:
            position = len(population.individuals)
        
        # Insert individual
        population.individuals.insert(position, new_individual)
        
        # Record mutation
        record = MutationRecord(
            timestamp=datetime.utcnow(),
            session_id=session.session_id,
            operation="insert",
            individual_id=position,
            reason=reason
        )
        self.mutation_history.append(record)
        
        # Log
        log_msg = (
            f"[INSERT ] New individual at Ind#{position:02d} "
            f"(fitness={new_individual.fitness:.4f}, {reason})"
        )
        print(log_msg)
        
        if self.verbose:
            print(f"➕ {log_msg}")
        
        return position
    
    def create_coach_designed_individual(
        self,
        population: Population,
        session: CoachAnalysisSession,
        individual_params: "IndividualParameters",
        reason: str = "Coach-designed individual",
        position: Optional[int] = None
    ) -> int:
        """
        Create new individual with coach-designed parameters.
        
        Args:
            population: Population to add to
            session: Analysis session for tracking
            individual_params: Complete parameter set from coach
            reason: Reason for creation
            position: Where to insert (None = append)
        
        Returns:
            New individual ID (position in population)
        """
        from backtest.optimizer import Individual
        from core.models import BacktestParams
        from strategy.seller_exhaustion import SellerParams
        
        # Create seller params
        seller_params = SellerParams(
            ema_fast=individual_params.ema_fast,
            ema_slow=individual_params.ema_slow,
            z_window=individual_params.z_window,
            atr_window=individual_params.atr_window,
            vol_z=individual_params.vol_z,
            tr_z=individual_params.tr_z,
            cloc_min=individual_params.cloc_min
        )
        
        # Create backtest params
        backtest_params = BacktestParams(
            fib_swing_lookback=individual_params.fib_swing_lookback,
            fib_swing_lookahead=individual_params.fib_swing_lookahead,
            fib_target_level=individual_params.fib_target_level,
            use_fib_exits=individual_params.use_fib_exits,
            use_stop_loss=individual_params.use_stop_loss,
            use_traditional_tp=individual_params.use_traditional_tp,
            use_time_exit=individual_params.use_time_exit,
            atr_stop_mult=individual_params.atr_stop_mult,
            reward_r=individual_params.reward_r,
            max_hold=individual_params.max_hold,
            fee_bp=individual_params.fee_bp,
            slippage_bp=individual_params.slippage_bp
        )
        
        # Create new individual
        new_individual = Individual(
            seller_params=seller_params,
            backtest_params=backtest_params,
            fitness=0.0,  # Will be evaluated
            generation=population.generation
        )
        
        return self.insert_individual(
            population, session, new_individual, reason, position
        )
    
    def create_random_individual(
        self,
        population: Population,
        session: CoachAnalysisSession,
        timeframe: "Timeframe",
        reason: str = "Random individual injection",
        position: Optional[int] = None
    ) -> int:
        """
        Create new individual with random parameters within bounds.
        
        Args:
            population: Population to add to
            session: Analysis session for tracking
            timeframe: Timeframe for parameter bounds
            reason: Reason for creation
            position: Where to insert (None = append)
        
        Returns:
            New individual ID (position in population)
        """
        from backtest.optimizer import Individual, get_param_bounds_for_timeframe
        from core.models import BacktestParams
        from strategy.seller_exhaustion import SellerParams
        import random
        
        # Get parameter bounds for timeframe
        bounds = get_param_bounds_for_timeframe(timeframe)
        
        # Create random seller params
        seller_params = SellerParams(
            ema_fast=int(random.uniform(*bounds['ema_fast'])),
            ema_slow=int(random.uniform(*bounds['ema_slow'])),
            z_window=int(random.uniform(*bounds['z_window'])),
            atr_window=int(random.uniform(*bounds['atr_window'])),
            vol_z=random.uniform(*bounds['vol_z']),
            tr_z=random.uniform(*bounds['tr_z']),
            cloc_min=random.uniform(*bounds['cloc_min'])
        )
        
        # Create random backtest params
        # Import VALID_FIB_LEVELS for fib_target_level
        from backtest.optimizer import VALID_FIB_LEVELS
        
        backtest_params = BacktestParams(
            fib_swing_lookback=int(random.uniform(*bounds['fib_swing_lookback'])),
            fib_swing_lookahead=int(random.uniform(*bounds['fib_swing_lookahead'])),
            fib_target_level=random.choice(VALID_FIB_LEVELS),  # Special case: discrete choice
            use_fib_exits=random.choice([True, False]),
            use_stop_loss=random.choice([True, False]),
            use_traditional_tp=random.choice([True, False]),
            use_time_exit=random.choice([True, False]),
            atr_stop_mult=random.uniform(*bounds['atr_stop_mult']),
            reward_r=random.uniform(*bounds['reward_r']),
            max_hold=int(random.uniform(*bounds['max_hold'])),
            fee_bp=random.uniform(*bounds['fee_bp']),
            slippage_bp=random.uniform(*bounds['slippage_bp'])
        )
        
        # Create new individual
        new_individual = Individual(
            seller_params=seller_params,
            backtest_params=backtest_params,
            fitness=0.0,  # Will be evaluated
            generation=population.generation
        )
        
        return self.insert_individual(
            population, session, new_individual, reason, position
        )
    
    def apply_coach_recommendations(
        self,
        population: Population,
        session: CoachAnalysisSession,
        analysis: CoachAnalysis
    ) -> Dict[str, Any]:
        """
        Apply coach recommendations to population.
        
        Supports mutation recommendations (via category MUTATIONS or INDIVIDUAL_MUTATION):
        - mutate_<id>_<param>: Mutate specific individual parameter
        - drop_<id>: Drop individual
        - insert_coach_<params>: Insert coach-designed individual
        - insert_random: Insert random individual
        
        Args:
            population: Population to mutate
            session: Analysis session (frozen population state)
            analysis: Coach analysis with recommendations
        
        Returns:
            Summary of mutations applied
        """
        summary = {
            "total_mutations": 0,
            "mutations_by_type": {"mutate": 0, "drop": 0, "insert": 0},
            "mutations_applied": [],
            "mutations_failed": []
        }
        
        if not analysis or not analysis.recommendations:
            print("[APPLY  ] ⚠️  No recommendations to apply")
            return summary
        
        print(f"[APPLY  ] Processing {len(analysis.recommendations)} recommendations")
        
        # Parse recommendations for mutations
        for recommendation in analysis.recommendations:
            param = recommendation.parameter
            
            # Parse mutation commands
            if param.startswith("mutate_"):
                # Format: mutate_<individual_id>_<param_name>
                # Value: new_value
                parts = param.split("_")
                if len(parts) >= 3:
                    try:
                        ind_id = int(parts[1])
                        param_name = "_".join(parts[2:])
                        
                        success = self.mutate_individual_parameter(
                            population,
                            session,
                            ind_id,
                            param_name,
                            recommendation.suggested_value,
                            reason=recommendation.reasoning
                        )
                        
                        if success:
                            summary["mutations_by_type"]["mutate"] += 1
                            summary["total_mutations"] += 1
                            summary["mutations_applied"].append(param)
                            session.mutations_applied.append(param)
                        else:
                            summary["mutations_failed"].append(param)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse mutation: {param} - {e}")
                        summary["mutations_failed"].append(param)
            
            elif param.startswith("drop_"):
                # Format: drop_<individual_id>
                parts = param.split("_")
                if len(parts) >= 2:
                    try:
                        ind_id = int(parts[1])
                        
                        success = self.drop_individual(
                            population,
                            session,
                            ind_id,
                            reason=recommendation.reasoning
                        )
                        
                        if success:
                            summary["mutations_by_type"]["drop"] += 1
                            summary["total_mutations"] += 1
                            summary["mutations_applied"].append(param)
                            session.mutations_applied.append(param)
                        else:
                            summary["mutations_failed"].append(param)
                    except ValueError as e:
                        logger.warning(f"Failed to parse drop: {param} - {e}")
                        summary["mutations_failed"].append(param)
            
            elif param.startswith("insert_coach_"):
                # Coach recommends inserting a new individual with specific parameters
                if recommendation.individual_params:
                    success = self.create_coach_designed_individual(
                        population,
                        session,
                        recommendation.individual_params,
                        reason=recommendation.reasoning
                    )
                    if success is not None:
                        summary["mutations_by_type"]["insert"] += 1
                        summary["total_mutations"] += 1
                        summary["mutations_applied"].append(param)
                        session.mutations_applied.append(param)
                    else:
                        summary["mutations_failed"].append(param)
                else:
                    logger.warning(f"Coach insert recommendation missing individual_params: {param}")
                    summary["mutations_failed"].append(param)
            
            elif param == "insert_random":
                # Coach recommends inserting a random individual
                # Need to get timeframe from somewhere - use default for now
                from core.models import Timeframe
                success = self.create_random_individual(
                    population,
                    session,
                    Timeframe.m15,  # TODO: Get actual timeframe from context
                    reason=recommendation.reasoning
                )
                if success is not None:
                    summary["mutations_by_type"]["insert"] += 1
                    summary["total_mutations"] += 1
                    summary["mutations_applied"].append(param)
                    session.mutations_applied.append(param)
                else:
                    summary["mutations_failed"].append(param)
        
        # Log summary
        if summary["total_mutations"] > 0:
            print(
                f"[APPLY  ] Applied {summary['total_mutations']} mutations from coach: "
                f"{summary['mutations_by_type']['mutate']} mutate, "
                f"{summary['mutations_by_type']['drop']} drop, "
                f"{summary['mutations_by_type']['insert']} insert"
            )
        
        return summary
    
    def get_mutation_history(self, session_id: Optional[str] = None) -> List[MutationRecord]:
        """Get mutation history, optionally filtered by session."""
        if session_id:
            return [m for m in self.mutation_history if m.session_id == session_id]
        return self.mutation_history
    
    def clear_history(self):
        """Clear mutation history."""
        self.mutation_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mutation statistics."""
        return {
            "total_mutations": len(self.mutation_history),
            "by_operation": {
                "mutate": sum(1 for m in self.mutation_history if m.operation == "mutate"),
                "drop": sum(1 for m in self.mutation_history if m.operation == "drop"),
                "insert": sum(1 for m in self.mutation_history if m.operation == "insert"),
            }
        }
