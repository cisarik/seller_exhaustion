"""
Evolution Coach Analysis Session

Captures population state at a specific generation for coach analysis.
Provides blocking semantics: population frozen during coach analysis,
recommendations applied to exact same population state.
"""

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
from copy import deepcopy

from backtest.optimizer import Population, Individual
from core.models import FitnessConfig, OptimizationConfig
from backtest.coach_protocol import CoachAnalysis, CoachRecommendation


@dataclass
class IndividualSnapshot:
    """Snapshot of an individual at analysis time."""
    
    id: int  # Position in population
    fitness: float
    parameters: Dict[str, Any]  # seller_params + backtest_params
    metrics: Dict[str, Any]  # Backtest metrics
    
    def __repr__(self) -> str:
        return (
            f"Ind#{self.id:02d} fit={self.fitness:.4f} "
            f"trades={self.metrics.get('n', 0)} wr={self.metrics.get('win_rate', 0):.1%}"
        )


@dataclass
class CoachAnalysisSession:
    """
    Frozen population state for coach analysis.
    
    CRITICAL: Population is FROZEN at session creation time.
    Coach analyzes this exact state.
    Recommendations apply ONLY to this exact state.
    Evolution can resume after recommendations applied.
    """
    
    session_id: str  # Unique session identifier
    generation: int
    timestamp: datetime
    
    # Population snapshot (FROZEN)
    population_size: int
    individuals_snapshot: List[IndividualSnapshot] = field(default_factory=list)
    
    # Configuration snapshots (FROZEN)
    fitness_config_dict: Dict[str, Any] = field(default_factory=dict)
    ga_config_dict: Dict[str, Any] = field(default_factory=dict)
    
    # Coach analysis results (populated after analysis)
    analysis: Optional[CoachAnalysis] = None
    analysis_timestamp: Optional[datetime] = None
    
    # Tracking mutations applied
    mutations_applied: List[str] = field(default_factory=list)  # ["mutate_0", "drop_5", "insert_coach_0"]
    
    def __post_init__(self):
        """Generate session ID if not provided."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())[:8]
    
    @property
    def session_label(self) -> str:
        """Human-readable session label."""
        return f"Session {self.session_id} Gen {self.generation}"
    
    def get_individual_snapshot(self, individual_id: int) -> Optional[IndividualSnapshot]:
        """Get snapshot for specific individual."""
        for snap in self.individuals_snapshot:
            if snap.id == individual_id:
                return snap
        return None
    
    def get_best_individual_snapshot(self) -> Optional[IndividualSnapshot]:
        """Get best individual in this session."""
        if not self.individuals_snapshot:
            return None
        return max(self.individuals_snapshot, key=lambda x: x.fitness)
    
    def get_worst_individual_snapshot(self) -> Optional[IndividualSnapshot]:
        """Get worst individual in this session."""
        if not self.individuals_snapshot:
            return None
        return min(self.individuals_snapshot, key=lambda x: x.fitness)
    
    def get_population_metrics(self) -> Dict[str, Any]:
        """Calculate population-level metrics."""
        if not self.individuals_snapshot:
            return {}
        
        fitnesses = [ind.fitness for ind in self.individuals_snapshot]
        trade_counts = [ind.metrics.get('n', 0) for ind in self.individuals_snapshot]
        win_rates = [ind.metrics.get('win_rate', 0.0) for ind in self.individuals_snapshot]
        
        import numpy as np
        
        return {
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "best_fitness": float(np.max(fitnesses)),
            "worst_fitness": float(np.min(fitnesses)),
            "mean_trades": float(np.mean(trade_counts)),
            "std_trades": float(np.std(trade_counts)),
            "mean_win_rate": float(np.mean(win_rates)),
            "diversity": calculate_diversity(self.individuals_snapshot)
        }
    
    def to_dict_for_coach(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for coach analysis.
        Includes full individual data for context.
        """
        return {
            "session_id": self.session_id,
            "generation": self.generation,
            "timestamp": self.timestamp.isoformat(),
            "population_size": self.population_size,
            "individuals": [
                {
                    "id": snap.id,
                    "fitness": snap.fitness,
                    "parameters": snap.parameters,
                    "metrics": snap.metrics
                }
                for snap in self.individuals_snapshot
            ],
            "population_metrics": self.get_population_metrics(),
            "fitness_config": self.fitness_config_dict,
            "ga_config": self.ga_config_dict,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "session_id": self.session_id,
            "generation": self.generation,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "population_size": self.population_size,
            "individuals_snapshot_count": len(self.individuals_snapshot),
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "mutations_applied": self.mutations_applied,
        }
    
    @staticmethod
    def from_population(
        population: Population,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig
    ) -> "CoachAnalysisSession":
        """
        Create session from current population state (FREEZES population).
        
        Args:
            population: Population to snapshot
            fitness_config: Current fitness configuration
            ga_config: Current GA configuration
        
        Returns:
            New CoachAnalysisSession with frozen population snapshot
        """
        # Snapshot all individuals
        individual_snapshots = []
        for idx, individual in enumerate(population.individuals):
            # Merge seller_params and backtest_params
            params_dict = {}
            if hasattr(individual, 'seller_params'):
                seller_params = individual.seller_params
                if hasattr(seller_params, 'model_dump'):
                    params_dict.update(seller_params.model_dump())
                elif is_dataclass(seller_params):
                    params_dict.update(asdict(seller_params))
                else:
                    params_dict.update(getattr(seller_params, '__dict__', {}))
            if hasattr(individual, 'backtest_params'):
                backtest_params = individual.backtest_params
                if hasattr(backtest_params, 'model_dump'):
                    params_dict.update(backtest_params.model_dump())
                elif is_dataclass(backtest_params):
                    params_dict.update(asdict(backtest_params))
                else:
                    params_dict.update(getattr(backtest_params, '__dict__', {}))
            
            snap = IndividualSnapshot(
                id=idx,
                fitness=individual.fitness,
                parameters=params_dict,
                metrics=individual.metrics.copy() if individual.metrics else {}
            )
            individual_snapshots.append(snap)
        
        # Convert configs to dicts
        fitness_dict = (
            fitness_config.model_dump()
            if hasattr(fitness_config, 'model_dump')
            else asdict(fitness_config) if is_dataclass(fitness_config)
            else getattr(fitness_config, '__dict__', {})
        )
        ga_dict = (
            ga_config.model_dump()
            if hasattr(ga_config, 'model_dump')
            else asdict(ga_config) if is_dataclass(ga_config)
            else getattr(ga_config, '__dict__', {})
        )
        
        return CoachAnalysisSession(
            session_id="",  # Will be generated in __post_init__
            generation=population.generation,
            timestamp=datetime.utcnow(),
            population_size=len(population.individuals),
            individuals_snapshot=individual_snapshots,
            fitness_config_dict=fitness_dict,
            ga_config_dict=ga_dict,
        )


def calculate_diversity(individuals: List[IndividualSnapshot]) -> float:
    """
    Calculate population diversity (0.0 = homogeneous, 1.0 = diverse).
    
    Uses parameter space distance as diversity metric.
    """
    if len(individuals) < 2:
        return 1.0
    
    import numpy as np
    
    # Extract numeric parameter values
    param_vectors = []
    for ind in individuals:
        # Normalize parameters to 0-1 range (simplified)
        vector = []
        for param_name, param_val in ind.parameters.items():
            if isinstance(param_val, (int, float)):
                # Simple normalization (may need improvement)
                if isinstance(param_val, float):
                    vector.append(np.clip(param_val, 0.0, 1.0))
                else:
                    vector.append(min(1.0, param_val / 1000.0))  # Heuristic for int params
        param_vectors.append(vector)
    
    if not param_vectors or not param_vectors[0]:
        return 0.5  # Default if can't calculate
    
    # Calculate mean pairwise distance
    param_array = np.array(param_vectors)
    distances = []
    for i in range(len(param_array)):
        for j in range(i + 1, len(param_array)):
            dist = np.linalg.norm(param_array[i] - param_array[j])
            distances.append(dist)
    
    # Normalize to 0-1 (max possible distance ~sqrt(n_params))
    if distances:
        mean_distance = np.mean(distances)
        max_distance = np.sqrt(len(param_vectors[0]))
        diversity = min(1.0, mean_distance / max_distance)
        return float(diversity)
    
    return 0.5


def from_dataclass(obj):
    """Convert dataclass to dict recursively."""
    if hasattr(obj, '__dataclass_fields__'):
        return asdict(obj)
    return obj


def format_population_snapshot_for_coach(session: CoachAnalysisSession) -> str:
    """
    Format complete population snapshot for coach analysis.
    
    Args:
        session: Coach analysis session with population snapshot
    
    Returns:
        Formatted string with complete individual data for coach
    """
    lines = []
    lines.append(f"📸 Population Snapshot (Gen {session.generation}):")
    lines.append(f"Session ID: {session.session_id}")
    lines.append(f"Population Size: {session.population_size}")
    lines.append("")
    
    for snap in session.individuals_snapshot:
        # Individual summary line
        metrics = snap.metrics
        trades = metrics.get('n', '--')
        win_rate = metrics.get('win_rate', 0.0)
        avg_r = metrics.get('avg_R', 0.0)
        pnl = metrics.get('total_pnl', 0.0)
        
        lines.append(f"[IND {snap.id:2d}] fit={snap.fitness:.4f} | n={trades:3} | wr={win_rate:5.1%} | avgR={avg_r:5.2f} | pnl={pnl:6.3f}")
        
        # Complete parameter dump
        params = snap.parameters
        
        # Seller parameters (strategy entry logic)
        lines.append(f"    SELLER: ema_f={params.get('ema_fast', 0):3d} ema_s={params.get('ema_slow', 0):3d} z_win={params.get('z_window', 0):3d} atr_win={params.get('atr_window', 0):3d}")
        lines.append(f"    THRESH: vol_z={params.get('vol_z', 0.0):.2f} tr_z={params.get('tr_z', 0.0):.2f} cloc_min={params.get('cloc_min', 0.0):.2f}")
        
        # Backtest parameters (exit logic and costs)
        lines.append(f"    EXIT: fib_lookback={params.get('fib_swing_lookback', 0):3d} fib_lookahead={params.get('fib_swing_lookahead', 0):2d} fib_target={params.get('fib_target_level', 0.0):.3f}")
        lines.append(f"    COSTS: fee_bp={params.get('fee_bp', 0.0):.1f} slippage_bp={params.get('slippage_bp', 0.0):.1f} max_hold={params.get('max_hold', 0):3d}")
        
        # Suppress exit toggles line to reduce noise
        lines.append("")  # Empty line for readability
    
    # Population summary
    if session.individuals_snapshot:
        avg_fitness = sum(snap.fitness for snap in session.individuals_snapshot) / len(session.individuals_snapshot)
        best_fitness = max(snap.fitness for snap in session.individuals_snapshot)
        avg_trades = sum(snap.metrics.get('n', 0) for snap in session.individuals_snapshot) / len(session.individuals_snapshot)
        
        lines.append(f"📊 Population Summary:")
        lines.append(f"    Average Fitness: {avg_fitness:.4f}")
        lines.append(f"    Best Fitness: {best_fitness:.4f}")
        lines.append(f"    Average Trades: {avg_trades:.1f}")
        
        # Calculate diversity if possible
        try:
            diversity = calculate_diversity(session.individuals_snapshot)
            lines.append(f"    Diversity: {diversity:.3f}")
        except:
            lines.append(f"    Diversity: N/A")
    
    return "\n".join(lines)
