"""
Evolutionary algorithm for strategy parameter optimization.

Uses genetic algorithm with:
- Tournament selection
- Arithmetic crossover
- Gaussian mutation (small, controlled changes)
- Elitism to preserve best solutions

Console output cleanliness:
- Replaces per-individual print spam with a compact tqdm progress bar
- Emits concise, human-friendly summaries via the shared logger
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import json
from pathlib import Path
from tqdm import tqdm

from strategy.seller_exhaustion import SellerParams
from core.models import BacktestParams, Timeframe, minutes_to_bars
from backtest.engine import run_backtest
from strategy.seller_exhaustion import build_features
from core.logging_utils import get_logger
from config.settings import settings

logger = get_logger(__name__)


# Valid Fibonacci target levels (discrete choices)
VALID_FIB_LEVELS = [0.382, 0.500, 0.618, 0.786, 1.000]

# Time-based bounds (in minutes) - universal across timeframes
TIME_BOUNDS = {
    # Entry signal parameters (time-based)
    'ema_fast_minutes': (720, 2880),      # 12h - 48h
    'ema_slow_minutes': (5040, 20160),    # 3.5d - 14d
    'z_window_minutes': (5040, 20160),    # 3.5d - 14d
    'atr_window_minutes': (720, 2880),    # 12h - 48h
    
    # Fibonacci exit parameters (time-based)
    'fib_swing_lookback_minutes': (720, 2880),    # 12h - 48h
    'fib_swing_lookahead_minutes': (15, 120),     # 15min - 2h
    
    # Universal thresholds (not time-dependent)
    'vol_z': (1.0, 3.5),
    'tr_z': (0.8, 2.0),
    'cloc_min': (0.4, 0.8),
    
    # Transaction costs
    'fee_bp': (2.0, 10.0),
    'slippage_bp': (2.0, 10.0),
}


def get_param_bounds_for_timeframe(tf: Timeframe) -> Dict[str, Tuple[float, float]]:
    """
    Get optimization bounds appropriate for a specific timeframe.
    
    Converts time-based bounds to bar-based bounds for the given timeframe.
    This ensures the optimizer explores the same TIME RANGES regardless of timeframe.
    
    Args:
        tf: Timeframe to optimize for
    
    Returns:
        Dict mapping parameter name to (min, max) bounds in bars
    
    Example:
        For 15m timeframe:
            ema_fast: (720min, 2880min) → (48 bars, 192 bars)
        For 1m timeframe:
            ema_fast: (720min, 2880min) → (720 bars, 2880 bars)
        
        Same time range, different bar counts!
    """
    bounds = {}
    
    # Convert time-based bounds to bars
    bounds['ema_fast'] = (
        minutes_to_bars(TIME_BOUNDS['ema_fast_minutes'][0], tf),
        minutes_to_bars(TIME_BOUNDS['ema_fast_minutes'][1], tf)
    )
    bounds['ema_slow'] = (
        minutes_to_bars(TIME_BOUNDS['ema_slow_minutes'][0], tf),
        minutes_to_bars(TIME_BOUNDS['ema_slow_minutes'][1], tf)
    )
    bounds['z_window'] = (
        minutes_to_bars(TIME_BOUNDS['z_window_minutes'][0], tf),
        minutes_to_bars(TIME_BOUNDS['z_window_minutes'][1], tf)
    )
    bounds['atr_window'] = (
        minutes_to_bars(TIME_BOUNDS['atr_window_minutes'][0], tf),
        minutes_to_bars(TIME_BOUNDS['atr_window_minutes'][1], tf)
    )
    # Fibonacci parameters (time-based)
    bounds['fib_swing_lookback'] = (
        minutes_to_bars(TIME_BOUNDS['fib_swing_lookback_minutes'][0], tf),
        minutes_to_bars(TIME_BOUNDS['fib_swing_lookback_minutes'][1], tf)
    )
    bounds['fib_swing_lookahead'] = (
        minutes_to_bars(TIME_BOUNDS['fib_swing_lookahead_minutes'][0], tf),
        minutes_to_bars(TIME_BOUNDS['fib_swing_lookahead_minutes'][1], tf)
    )
    
    # Copy universal thresholds as-is
    bounds['vol_z'] = TIME_BOUNDS['vol_z']
    bounds['tr_z'] = TIME_BOUNDS['tr_z']
    bounds['cloc_min'] = TIME_BOUNDS['cloc_min']
    bounds['fee_bp'] = TIME_BOUNDS['fee_bp']
    bounds['slippage_bp'] = TIME_BOUNDS['slippage_bp']
    
    # Add missing backtest parameters (not time-based)
    bounds['atr_stop_mult'] = (0.3, 1.5)      # Stop loss multiplier
    bounds['reward_r'] = (1.0, 5.0)           # Risk:Reward ratio
    bounds['max_hold'] = (48, 288)            # Max hold time in bars (12h-72h on 15m)
    
    return bounds


# Default bounds for backward compatibility (15m timeframe)
PARAM_BOUNDS = get_param_bounds_for_timeframe(Timeframe.m15)


@dataclass
class Individual:
    """Represents one solution (parameter combination) in the population."""
    seller_params: SellerParams
    backtest_params: BacktestParams
    fitness: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, gen={self.generation})"


class Population:
    """Manages a population of individuals for evolutionary optimization."""
    
    def __init__(self, size: int = 24, seed_individual: Individual = None, timeframe: Timeframe = Timeframe.m15):
        """
        Initialize population.
        
        Args:
            size: Population size
            seed_individual: Optional starting point (e.g., current params)
            timeframe: Timeframe to optimize for (affects parameter bounds)
        """
        self.size = size
        self.individuals: List[Individual] = []
        self.generation = 0
        self.best_ever: Individual = None
        self.history: List[Dict[str, Any]] = []
        self.timeframe = timeframe
        
        # Get timeframe-specific bounds
        self.bounds = get_param_bounds_for_timeframe(timeframe)
        
        # Initialize population
        if seed_individual:
            # Add seed as first individual
            self.individuals.append(deepcopy(seed_individual))
            # Generate rest randomly
            for _ in range(size - 1):
                self.individuals.append(self._create_random_individual())
        else:
            # All random
            for _ in range(size):
                self.individuals.append(self._create_random_individual())

    # -------- Population I/O (Export / Import) --------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize population to a JSON-serializable dict."""
        return {
            "version": 1,
            "timeframe": str(self.timeframe.value if isinstance(self.timeframe, Timeframe) else self.timeframe),
            "generation": int(self.generation),
            "size": int(self.size),
            "individuals": [
                {
                    "seller_params": deepcopy(ind.seller_params.__dict__),
                    "backtest_params": deepcopy(
                        ind.backtest_params.model_dump() if hasattr(ind.backtest_params, "model_dump") else ind.backtest_params.__dict__
                    ),
                    "fitness": float(ind.fitness),
                    "metrics": deepcopy(ind.metrics),
                    "generation": int(ind.generation),
                }
                for ind in self.individuals
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save population to a JSON file.

        Args:
            path: Output file path ('.json')
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _individual_from_dict(d: Dict[str, Any]) -> Individual:
        sp = SellerParams(**d.get("seller_params", {}))
        bp = BacktestParams(**d.get("backtest_params", {}))
        ind = Individual(seller_params=sp, backtest_params=bp)
        ind.fitness = float(d.get("fitness", 0.0))
        ind.metrics = d.get("metrics", {}) or {}
        ind.generation = int(d.get("generation", 0))
        return ind

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        timeframe: Optional[Timeframe] = None,
        limit: Optional[int] = None,
    ) -> "Population":
        """Construct a population from a JSON file previously saved via save().

        Args:
            path: JSON file path
            timeframe: Optional override timeframe; if None, uses value from file when available
            limit: Optional maximum number of individuals to load (truncates if provided)

        Returns:
            Population instance initialized with loaded individuals (no random fill)
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Population file not found: {path}")

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        inds_raw = data.get("individuals", [])
        if limit is not None:
            inds_raw = inds_raw[: int(limit)]

        inds = [cls._individual_from_dict(d) for d in inds_raw]

        # Determine timeframe
        tf_str = data.get("timeframe")
        tf_val = timeframe
        try:
            if tf_val is None and tf_str is not None:
                # Map string like '15m' or '15' to enum when possible
                tf_val = Timeframe(tf_str) if tf_str in [t.value for t in Timeframe] else timeframe
        except Exception:
            tf_val = timeframe or Timeframe.m15

        pop = cls(size=len(inds) or data.get("size", 0) or 0, timeframe=tf_val or Timeframe.m15)
        pop.individuals = inds
        pop.size = len(inds)
        pop.generation = int(data.get("generation", 0))
        pop.best_ever = deepcopy(max(inds, key=lambda x: x.fitness)) if inds else None
        # Keep timeframe-aware bounds consistent
        pop.bounds = get_param_bounds_for_timeframe(pop.timeframe)
        return pop
    
    def _create_random_individual(self) -> Individual:
        """Create an individual with random parameters within timeframe-appropriate bounds."""
        # Random SellerParams
        seller_params = SellerParams(
            ema_fast=random.randint(*self.bounds['ema_fast']),
            ema_slow=random.randint(*self.bounds['ema_slow']),
            z_window=random.randint(*self.bounds['z_window']),
            vol_z=random.uniform(*self.bounds['vol_z']),
            tr_z=random.uniform(*self.bounds['tr_z']),
            cloc_min=random.uniform(*self.bounds['cloc_min']),
            atr_window=random.randint(*self.bounds['atr_window']),
        )
        
        # Random BacktestParams (ONLY Fibonacci + costs)
        backtest_params = BacktestParams(
            # Fibonacci parameters (ONLY exit mechanism)
            fib_swing_lookback=random.randint(*self.bounds['fib_swing_lookback']),
            fib_swing_lookahead=random.randint(*self.bounds['fib_swing_lookahead']),
            fib_target_level=random.choice(VALID_FIB_LEVELS),
            # Transaction costs
            fee_bp=random.uniform(*self.bounds['fee_bp']),
            slippage_bp=random.uniform(*self.bounds['slippage_bp']),
        )
        
        return Individual(
            seller_params=seller_params,
            backtest_params=backtest_params,
            generation=self.generation
        )
    
    def get_best(self) -> Individual:
        """Return the best individual in current population."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda x: x.fitness)
    
    def get_worst(self) -> Individual:
        """Return the worst individual in current population."""
        if not self.individuals:
            return None
        return min(self.individuals, key=lambda x: x.fitness)
    
    def get_stats(self) -> Dict[str, float]:
        """Get population statistics."""
        if not self.individuals:
            return {'generation': self.generation}
        
        fitnesses = [ind.fitness for ind in self.individuals]
        return {
            'generation': self.generation,
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'best_ever_fitness': self.best_ever.fitness if self.best_ever else 0.0,
        }
    
    def add_immigrants(
        self,
        fraction: float = 0.15,
        strategy: str = "worst_replacement",
        generation: int = 0
    ) -> int:
        """
        Add random immigrants to maintain population diversity.
        
        Helps prevent premature convergence when population becomes homogeneous.
        
        Args:
            fraction: Fraction of population to add/replace (e.g., 0.15 = 15%)
            strategy: "worst_replacement" = replace worst individuals (default)
                     "random" = random insertion
            generation: Current generation (for tracking)
        
        Returns:
            Number of immigrants added
        """
        n_immigrants = max(1, int(self.size * fraction))
        
        # Create new random individuals
        new_immigrants = [
            self._create_random_individual()
            for _ in range(n_immigrants)
        ]
        
        if strategy == "worst_replacement":
            # Sort by fitness and replace worst N
            sorted_inds = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
            self.individuals = sorted_inds[:self.size - n_immigrants] + new_immigrants
        else:  # random strategy
            # Keep best individuals, add immigrants at end, trim to size
            self.individuals = self.individuals[:self.size - n_immigrants] + new_immigrants
        
        return n_immigrants
    
    def get_diversity_metric(self) -> float:
        """
        Calculate population diversity (0=homogeneous, 1=highly diverse).
        
        Based on normalized parameter variance across population.
        """
        if len(self.individuals) < 2:
            return 1.0
        
        # Collect normalized parameter vectors
        param_vectors = []
        for ind in self.individuals:
            # Normalize each parameter to 0-1 range based on bounds
            vals = []
            
            # SellerParams
            for param_name in ['ema_fast', 'ema_slow', 'z_window', 'atr_window']:
                if param_name in self.bounds:
                    min_v, max_v = self.bounds[param_name]
                    raw_val = getattr(ind.seller_params, param_name)
                    normalized = (raw_val - min_v) / max(max_v - min_v, 1)
                    vals.append(np.clip(normalized, 0.0, 1.0))
            
            # BacktestParams (Fibonacci + costs)
            for param_name in ['fib_swing_lookback', 'fib_swing_lookahead']:
                if param_name in self.bounds:
                    min_v, max_v = self.bounds[param_name]
                    raw_val = getattr(ind.backtest_params, param_name)
                    normalized = (raw_val - min_v) / max(max_v - min_v, 1)
                    vals.append(np.clip(normalized, 0.0, 1.0))
            
            # Statistical thresholds
            for param_name in ['vol_z', 'tr_z', 'cloc_min', 'fee_bp', 'slippage_bp']:
                if param_name in self.bounds:
                    min_v, max_v = self.bounds[param_name]
                    raw_val = getattr(ind.seller_params if param_name in ['vol_z', 'tr_z', 'cloc_min'] 
                                      else ind.backtest_params, param_name)
                    normalized = (raw_val - min_v) / max(max_v - min_v, 1)
                    vals.append(np.clip(normalized, 0.0, 1.0))
            
            param_vectors.append(np.array(vals))
        
        if not param_vectors or len(param_vectors[0]) == 0:
            return 1.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(param_vectors)):
            for j in range(i + 1, len(param_vectors)):
                dist = np.linalg.norm(param_vectors[i] - param_vectors[j])
                distances.append(dist)
        
        if not distances:
            return 1.0
        
        # Normalize: max possible distance = sqrt(n_params)
        mean_dist = np.mean(distances)
        n_params = len(param_vectors[0])
        max_possible_dist = np.sqrt(n_params)
        
        diversity = min(mean_dist / max(max_possible_dist, 1.0), 1.0)
        return float(diversity)
    
    def apply_bounds_override(self, override_bounds: Dict[str, Tuple[float, float]]) -> None:
        """
        Apply Coach-recommended bounds overrides to modify search space.
        
        Allows Coach to expand or narrow search space based on analysis.
        
        Args:
            override_bounds: Dict mapping param names to new (min, max) bounds
        """
        for param_name, (min_val, max_val) in override_bounds.items():
            if param_name in self.bounds:
                old_bounds = self.bounds[param_name]
                self.bounds[param_name] = (min_val, max_val)
                print(f"  📌 Bounds {param_name}: {old_bounds} → ({min_val}, {max_val})")


def calculate_fitness(
    metrics: Dict[str, Any],
    config: "FitnessConfig" = None,
    generation: int = 0
) -> float:
    """
    Calculate composite fitness from backtest metrics using configurable weights.
    
    Supports:
    - Multiple optimization strategies (Balanced, HF, Conservative, Profit-focused)
    - Soft penalties (continuous) vs hard gates (discrete clipping)
    - Curriculum learning (gradually increase min_trades requirement)
    
    Args:
        metrics: Backtest metrics dictionary
        config: FitnessConfig with weights and requirements (uses balanced if None)
        generation: Current generation (for curriculum learning)
    
    Returns:
        Fitness score (higher is better)
    """
    # Import here to avoid circular dependency
    from core.models import FitnessConfig
    import numpy as np
    
    if config is None:
        config = FitnessConfig()  # Use balanced defaults
    
    # Handle case with no trades
    if metrics['n'] == 0:
        return -1000.0
    
    # Extract metrics
    n_trades = metrics['n']
    win_rate = metrics.get('win_rate', 0.0)
    avg_r = metrics.get('avg_R', 0.0)
    total_pnl = metrics.get('total_pnl', 0.0)
    max_dd = metrics.get('max_dd', 0.0)
    
    # Get effective min_trades (supports curriculum learning)
    effective_min_trades = config.get_effective_min_trades(generation)
    effective_min_wr = config.min_win_rate
    
    # Normalize components (0-1 scale)
    trade_count_normalized = min(n_trades / 100.0, 1.0)
    pnl_normalized = np.tanh(total_pnl / 0.5)  # Range: -1 to 1
    avg_r_normalized = np.clip((avg_r + 2) / 7.0, 0.0, 1.0)
    dd_normalized = max(max_dd / 0.5, -1.0)
    
    # Calculate base fitness (weighted components)
    fitness_base = (
        config.trade_count_weight * trade_count_normalized +
        config.win_rate_weight * win_rate +
        config.avg_r_weight * avg_r_normalized +
        config.total_pnl_weight * pnl_normalized +
        config.max_drawdown_penalty * dd_normalized
    )
    
    # Apply selected fitness function type
    if config.fitness_function_type == "hard_gates":
        # Original: hard fail if below minimums
        if n_trades < effective_min_trades:
            return -100.0
        if win_rate < effective_min_wr:
            return -50.0
        return fitness_base
    
    else:  # "soft_penalties" (default, recommended)
        # Continuous penalty for shortfalls - maintains gradient for GA
        penalty_trades = 0.0
        if n_trades < effective_min_trades:
            shortfall_ratio = (effective_min_trades - n_trades) / max(effective_min_trades, 1)
            penalty_trades = config.penalty_trades_strength * shortfall_ratio
        
        penalty_wr = 0.0
        if win_rate < effective_min_wr:
            shortfall_ratio = (effective_min_wr - win_rate) / max(effective_min_wr, 0.01)
            penalty_wr = config.penalty_wr_strength * shortfall_ratio
        
        return fitness_base - penalty_trades - penalty_wr


def evaluate_individual(
    individual: Individual,
    data: pd.DataFrame,
    tf: Timeframe = Timeframe.m15,
    fitness_config: "FitnessConfig" = None,
    generation: int = 0
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate an individual by running backtest and calculating fitness.
    
    Args:
        individual: Individual to evaluate
        data: Historical OHLCV data
        tf: Timeframe
        fitness_config: FitnessConfig for fitness calculation (uses balanced if None)
        generation: Current generation (for curriculum learning)
    
    Returns:
        (fitness, metrics) tuple
    """
    try:
        feats = build_features(
            data,
            individual.seller_params,
            tf,
        )
        result = run_backtest(feats, individual.backtest_params)
        
        # Calculate fitness with configurable weights and curriculum learning
        metrics = result['metrics']
        fitness = calculate_fitness(metrics, fitness_config, generation=generation)
        
        return fitness, metrics
        
    except Exception as e:
        logger.exception("Error evaluating individual: %s", e)
        return -1000.0, {'n': 0, 'error': str(e)}


INTEGER_PARAMS = {
    'ema_fast',
    'ema_slow',
    'z_window',
    'atr_window',
    'fib_swing_lookback',
    'fib_swing_lookahead',
}


def mutate_parameter(
    value: float,
    param_name: str,
    bounds: Dict[str, Tuple[float, float]],
    sigma: float = 0.1
) -> float:
    """
    Mutate a single parameter with Gaussian noise.
    
    Special handling for fib_target_level (discrete choice from valid levels).
    
    Args:
        value: Current parameter value
        param_name: Name of parameter (for bounds lookup)
        bounds: Parameter bounds (timeframe-aware)
        sigma: Standard deviation as fraction of parameter range
    
    Returns:
        Mutated value clamped to bounds
    """
    # Special case: fib_target_level is discrete
    if param_name == 'fib_target_level':
        return random.choice(VALID_FIB_LEVELS)
    
    param_bounds = bounds.get(param_name) or PARAM_BOUNDS.get(param_name)
    if param_bounds is None:
        return value
    
    min_val, max_val = param_bounds
    param_range = max_val - min_val
    if param_range <= 0:
        return int(round(min_val)) if param_name in INTEGER_PARAMS else float(min_val)
    
    # Gaussian noise scaled to parameter range
    noise = np.random.normal(0, sigma * param_range)
    
    # Add noise
    new_value = value + noise
    
    # Clamp to bounds
    new_value = np.clip(new_value, min_val, max_val)
    
    # Round if original was integer
    if param_name in INTEGER_PARAMS:
        new_value = int(round(new_value))
    
    return new_value


def mutate_individual(
    individual: Individual,
    bounds: Dict[str, Tuple[float, float]],
    mutation_rate: float = 0.3,
    sigma: float = 0.1,
    generation: int = 0
) -> Individual:
    """
    Mutate an individual with probability mutation_rate per parameter.
    
    Uses Gaussian mutation with small sigma for gradual changes.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each parameter
        sigma: Standard deviation for Gaussian noise (as fraction of range)
        generation: Current generation (for tracking)
    
    Returns:
        New mutated individual
    """
    # Create new params (don't modify original)
    seller_dict = individual.seller_params.__dict__.copy()
    backtest_dict = individual.backtest_params.__dict__.copy()
    
    # Mutate SellerParams
    for param_name in ['ema_fast', 'ema_slow', 'z_window', 'vol_z', 'tr_z', 'cloc_min', 'atr_window']:
        if random.random() < mutation_rate:
            seller_dict[param_name] = mutate_parameter(
                seller_dict[param_name],
                param_name,
                bounds,
                sigma
            )
    
    # Mutate BacktestParams (ONLY Fibonacci + costs)
    for param_name in ['fib_swing_lookback', 'fib_swing_lookahead', 'fib_target_level',
                       'fee_bp', 'slippage_bp']:
        if random.random() < mutation_rate:
            backtest_dict[param_name] = mutate_parameter(
                backtest_dict[param_name],
                param_name,
                bounds,
                sigma
            )
    
    return Individual(
        seller_params=SellerParams(**seller_dict),
        backtest_params=BacktestParams(**backtest_dict),
        generation=generation
    )


def crossover(
    parent1: Individual,
    parent2: Individual,
    alpha: float = None,
    generation: int = 0
) -> Tuple[Individual, Individual]:
    """
    Create two offspring via arithmetic crossover.
    
    Args:
        parent1: First parent
        parent2: Second parent
        alpha: Blend factor (random if None). child1 = alpha*p1 + (1-alpha)*p2
        generation: Current generation (for tracking)
    
    Returns:
        (child1, child2) tuple
    """
    if alpha is None:
        alpha = random.uniform(0.3, 0.7)  # Favor balanced blend
    
    # Crossover SellerParams
    def blend_seller_params(sp1: SellerParams, sp2: SellerParams, a: float) -> SellerParams:
        return SellerParams(
            ema_fast=int(round(a * sp1.ema_fast + (1-a) * sp2.ema_fast)),
            ema_slow=int(round(a * sp1.ema_slow + (1-a) * sp2.ema_slow)),
            z_window=int(round(a * sp1.z_window + (1-a) * sp2.z_window)),
            vol_z=a * sp1.vol_z + (1-a) * sp2.vol_z,
            tr_z=a * sp1.tr_z + (1-a) * sp2.tr_z,
            cloc_min=a * sp1.cloc_min + (1-a) * sp2.cloc_min,
            atr_window=int(round(a * sp1.atr_window + (1-a) * sp2.atr_window)),
        )
    
    # Crossover BacktestParams (ONLY Fibonacci + costs)
    def blend_backtest_params(bp1: BacktestParams, bp2: BacktestParams, a: float) -> BacktestParams:
        # For discrete fib_target_level, randomly pick from one parent
        fib_target = bp1.fib_target_level if random.random() < a else bp2.fib_target_level
        
        return BacktestParams(
            # Fibonacci parameters (ONLY exit mechanism)
            fib_swing_lookback=int(round(a * bp1.fib_swing_lookback + (1-a) * bp2.fib_swing_lookback)),
            fib_swing_lookahead=int(round(a * bp1.fib_swing_lookahead + (1-a) * bp2.fib_swing_lookahead)),
            fib_target_level=fib_target,  # Discrete choice
            # Transaction costs
            fee_bp=a * bp1.fee_bp + (1-a) * bp2.fee_bp,
            slippage_bp=a * bp1.slippage_bp + (1-a) * bp2.slippage_bp,
        )
    
    # Create children with complementary blend factors
    child1 = Individual(
        seller_params=blend_seller_params(parent1.seller_params, parent2.seller_params, alpha),
        backtest_params=blend_backtest_params(parent1.backtest_params, parent2.backtest_params, alpha),
        generation=generation
    )
    
    child2 = Individual(
        seller_params=blend_seller_params(parent1.seller_params, parent2.seller_params, 1-alpha),
        backtest_params=blend_backtest_params(parent1.backtest_params, parent2.backtest_params, 1-alpha),
        generation=generation
    )
    
    return child1, child2


def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
    """
    Select an individual via tournament selection.
    
    Args:
        population: Population to select from
        tournament_size: Number of individuals in tournament
    
    Returns:
        Selected individual
    """
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness)


def evolution_step(
    population: Population,
    data: pd.DataFrame,
    tf: Timeframe = Timeframe.m15,
    fitness_config: "FitnessConfig" = None,
    ga_config: "OptimizationConfig" = None,  # NEW: structured GA config
    mutation_rate: float = None,  # Deprecated: use ga_config instead
    sigma: float = None,
    elite_fraction: float = None,
    tournament_size: int = None,
    mutation_probability: float = None
) -> Population:
    """
    Perform one generation of evolution with Coach-compatible configuration.
    
    Supports:
    - Soft penalties + curriculum learning (via fitness_config)
    - Dynamic GA hyperparameters (via ga_config)
    - Random immigrants for diversity (via ga_config)
    - Bounds override (via ga_config)
    
    Args:
        population: Current population
        data: Historical OHLCV data
        tf: Timeframe
        fitness_config: Fitness function configuration
        ga_config: Optimization configuration (Coach applies changes here)
        
        [Deprecated parameters - use ga_config instead]:
        mutation_rate, sigma, elite_fraction, tournament_size, mutation_probability
    
    Returns:
        New population for next generation
    """
    from core.models import OptimizationConfig
    
    # Handle backward compatibility: if ga_config not provided, build from old params
    if ga_config is None:
        ga_config = OptimizationConfig(
            mutation_rate=mutation_rate or 0.3,
            sigma=sigma or 0.1,
            elite_fraction=elite_fraction or 0.1,
            tournament_size=tournament_size or 3,
            mutation_probability=mutation_probability or 0.9
        )
    
    pop_size = population.size
    current_gen = population.generation
    
    logger.info("[Gen %s] Starting evolution step", current_gen)
    
    # Apply bounds override if Coach recommended changes
    if ga_config.override_bounds:
        population.apply_bounds_override(ga_config.override_bounds)
    
    # Step 1: Evaluate fitness for unevaluated individuals
    unevaluated = [ind for ind in population.individuals if ind.fitness == 0.0]
    
    if unevaluated:
        total = len(unevaluated)
        logger.info("[Gen %s] Evaluating %s individuals", current_gen, total)
        with tqdm(total=total, desc=f"Gen {current_gen} eval", unit="ind", leave=False, disable=not getattr(settings, 'log_progress_bars', True)) as pbar:
            for ind in unevaluated:
                fitness, metrics = evaluate_individual(
                    ind, data, tf, fitness_config,
                    generation=current_gen  # Pass generation for curriculum
                )
                ind.fitness = fitness
                ind.metrics = metrics
                pbar.update(1)
    
    # Update best ever
    current_best = population.get_best()
    if population.best_ever is None or current_best.fitness > population.best_ever.fitness:
        population.best_ever = deepcopy(current_best)
        logger.info(
            "[Gen %s] New best: fitness=%.4f trades=%s win_rate=%.2f%%",
            current_gen,
            current_best.fitness,
            current_best.metrics.get('n', 0),
            100.0 * (current_best.metrics.get('win_rate', 0.0) or 0.0),
        )
    
    # Population statistics
    stats = population.get_stats()
    trade_counts = [ind.metrics.get('n', 0) for ind in population.individuals]
    below_min_trades = sum(1 for tc in trade_counts if tc < (fitness_config.get_effective_min_trades(current_gen) if fitness_config else 10))
    diversity = population.get_diversity_metric() if ga_config.track_diversity else 1.0
    
    logger.info(
        "[Gen %s] Pop: mean=%.4f std=%.4f best=%.4f | below_min=%d/%.0f (%.1f%%) | mean_trades=%.1f | diversity=%.2f",
        current_gen,
        stats['mean_fitness'], stats['std_fitness'], stats['max_fitness'],
        below_min_trades, pop_size, 100 * below_min_trades / pop_size,
        np.mean(trade_counts), diversity,
    )
    
    # Detect stagnation (Coach cares about this)
    is_stagnant = False
    if ga_config.track_stagnation and len(population.history) >= ga_config.stagnation_threshold:
        recent_bests = [h['best_fitness'] for h in population.history[-ga_config.stagnation_threshold:]]
        best_improvement = max(recent_bests) - min(recent_bests)
        is_stagnant = best_improvement < ga_config.stagnation_fitness_tolerance
        
        if is_stagnant:
            print(f"⚠️  STAGNATION: Fitness flat for {ga_config.stagnation_threshold} gens (improvement: {best_improvement:.4f})")
    
    # Record history with new metrics
    population.history.append({
        'generation': current_gen,
        'best_fitness': stats['max_fitness'],
        'mean_fitness': stats['mean_fitness'],
        'std_fitness': stats['std_fitness'],
        'diversity': diversity,
        'below_min_trades': below_min_trades,
        'mean_trades': np.mean(trade_counts),
        'stagnant': is_stagnant,
    })
    
    # Step 2: Selection
    parents = [
        tournament_selection(population.individuals, ga_config.tournament_size)
        for _ in range(pop_size)
    ]
    
    # Step 3: Crossover
    offspring = []
    for i in range(0, len(parents) - 1, 2):
        child1, child2 = crossover(parents[i], parents[i+1], generation=current_gen + 1)
        offspring.extend([child1, child2])
    
    # Handle odd number
    if len(offspring) < pop_size:
        offspring.append(deepcopy(parents[-1]))
        offspring[-1].generation = current_gen + 1
    
    # Step 4: Mutation
    bounds = population.bounds
    for child in offspring:
        if random.random() < ga_config.mutation_probability:
            mutated = mutate_individual(
                child,
                bounds,
                ga_config.mutation_rate,
                ga_config.sigma,
                current_gen + 1
            )
            child.seller_params = mutated.seller_params
            child.backtest_params = mutated.backtest_params
            child.fitness = 0.0  # Reset fitness (needs re-evaluation)
    
    # Step 5: Elitism
    elite_size = max(1, int(pop_size * ga_config.elite_fraction))
    elite = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)[:elite_size]
    
    # Create new generation
    new_individuals = elite + offspring[:pop_size - elite_size]
    
    # NEW: Add random immigrants for diversity (Coach controls this)
    diversity_trigger = (
        is_stagnant or
        (ga_config.track_diversity and diversity < 0.3)
    )
    
    if ga_config.immigrant_fraction > 0 and diversity_trigger:
        n_added = Population(size=pop_size, timeframe=population.timeframe).add_immigrants(
            ga_config.immigrant_fraction,
            ga_config.immigrant_strategy,
            current_gen + 1
        )
        # Replace worst individuals with immigrants
        new_individuals = sorted(new_individuals, key=lambda x: x.fitness, reverse=True)
        n_replace = int(pop_size * ga_config.immigrant_fraction)
        new_individuals = new_individuals[:-n_replace] if n_replace > 0 else new_individuals
        # Add new immigrants
        for _ in range(min(n_replace, n_added)):
            new_immigrants = Population(size=n_replace, timeframe=population.timeframe)
            new_individuals.extend(new_immigrants.individuals[:n_replace - len(new_individuals) + elite_size])
    
    # Ensure population size is correct
    new_individuals = new_individuals[:pop_size]
    
    # Create new population
    new_population = Population(size=pop_size, timeframe=population.timeframe)
    new_population.individuals = new_individuals
    new_population.generation = current_gen + 1
    new_population.best_ever = population.best_ever
    new_population.history = population.history
    new_population.bounds = population.bounds
    new_population.timeframe = population.timeframe
    
    return new_population


# -------- Convenience helpers (explicit export/import API) --------
def export_population(population: Population, path: str | Path) -> None:
    """Export a population to JSON file."""
    population.save(path)


def import_population(path: str | Path, timeframe: Optional[Timeframe] = None, limit: Optional[int] = None) -> Population:
    """Import a population from JSON file.

    Args:
        path: JSON file path
        timeframe: Optional override timeframe
        limit: Optional max number of individuals
    """
    return Population.from_file(path, timeframe=timeframe, limit=limit)
