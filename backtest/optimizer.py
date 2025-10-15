"""
Evolutionary algorithm for strategy parameter optimization.

Uses genetic algorithm with:
- Tournament selection
- Arithmetic crossover
- Gaussian mutation (small, controlled changes)
- Elitism to preserve best solutions
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import random
from copy import deepcopy

from strategy.seller_exhaustion import SellerParams
from core.models import BacktestParams, Timeframe, minutes_to_bars
from backtest.engine import run_backtest
from strategy.seller_exhaustion import build_features


# Time-based bounds (in minutes) - universal across timeframes
TIME_BOUNDS = {
    'ema_fast_minutes': (720, 2880),      # 12h - 48h
    'ema_slow_minutes': (5040, 20160),    # 3.5d - 14d
    'z_window_minutes': (5040, 20160),    # 3.5d - 14d
    'atr_window_minutes': (720, 2880),    # 12h - 48h
    'max_hold_minutes': (720, 2880),      # 12h - 48h
    
    # Universal thresholds (not time-dependent)
    'vol_z': (1.0, 3.5),
    'tr_z': (0.8, 2.0),
    'cloc_min': (0.4, 0.8),
    'atr_stop_mult': (0.3, 1.5),
    'reward_r': (1.5, 4.0),
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
    bounds['max_hold'] = (
        minutes_to_bars(TIME_BOUNDS['max_hold_minutes'][0], tf),
        minutes_to_bars(TIME_BOUNDS['max_hold_minutes'][1], tf)
    )
    
    # Copy universal thresholds as-is
    bounds['vol_z'] = TIME_BOUNDS['vol_z']
    bounds['tr_z'] = TIME_BOUNDS['tr_z']
    bounds['cloc_min'] = TIME_BOUNDS['cloc_min']
    bounds['atr_stop_mult'] = TIME_BOUNDS['atr_stop_mult']
    bounds['reward_r'] = TIME_BOUNDS['reward_r']
    bounds['fee_bp'] = TIME_BOUNDS['fee_bp']
    bounds['slippage_bp'] = TIME_BOUNDS['slippage_bp']
    
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
        
        # Random BacktestParams
        backtest_params = BacktestParams(
            atr_stop_mult=random.uniform(*self.bounds['atr_stop_mult']),
            reward_r=random.uniform(*self.bounds['reward_r']),
            max_hold=random.randint(*self.bounds['max_hold']),
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
            return {}
        
        fitnesses = [ind.fitness for ind in self.individuals]
        return {
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'best_ever_fitness': self.best_ever.fitness if self.best_ever else 0.0,
        }


def calculate_fitness(metrics: Dict[str, Any], config: "FitnessConfig" = None) -> float:
    """
    Calculate composite fitness from backtest metrics using configurable weights.
    
    Supports multiple optimization strategies:
    - Balanced: Standard multi-objective (default)
    - High Frequency: Maximize trade count (scalping/day trading)
    - Conservative: Prioritize win rate and drawdown control
    - Profit Focused: Maximize total PnL
    - Custom: User-defined weights
    
    Args:
        metrics: Backtest metrics dictionary
        config: FitnessConfig with weights and requirements (uses balanced if None)
    
    Returns:
        Fitness score (higher is better)
    """
    # Import here to avoid circular dependency
    from core.models import FitnessConfig
    
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
    
    # Apply minimum requirements (hard filters)
    if n_trades < config.min_trades:
        return -100.0  # Fail: not enough trades
    
    if win_rate < config.min_win_rate:
        return -50.0  # Fail: win rate too low
    
    # Normalize trade count (0-1 scale, higher is better)
    # Scale based on reasonable expectations: 100 trades = 1.0
    trade_count_normalized = min(n_trades / 100.0, 1.0)
    
    # Normalize PnL (sigmoid-like, centered at 0)
    # Positive PnL → positive contribution, negative → penalty
    import numpy as np
    pnl_normalized = np.tanh(total_pnl / 0.5)  # Range: -1 to 1
    
    # Normalize avg R-multiple (typical range: -2 to 5)
    # Scale to 0-1 range
    avg_r_normalized = np.clip((avg_r + 2) / 7.0, 0.0, 1.0)
    
    # Normalize drawdown penalty (0 = no DD, -1 = severe DD)
    # More negative DD → more negative contribution
    dd_normalized = max(max_dd / 0.5, -1.0)
    
    # Calculate weighted fitness
    fitness = (
        config.trade_count_weight * trade_count_normalized +
        config.win_rate_weight * win_rate +  # Already 0-1
        config.avg_r_weight * avg_r_normalized +
        config.total_pnl_weight * pnl_normalized +
        config.max_drawdown_penalty * dd_normalized  # Negative contribution
    )
    
    return fitness


def evaluate_individual(
    individual: Individual,
    data: pd.DataFrame,
    tf: Timeframe = Timeframe.m15,
    fitness_config: "FitnessConfig" = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate an individual by running backtest and calculating fitness.
    
    Args:
        individual: Individual to evaluate
        data: Historical OHLCV data
        tf: Timeframe
        fitness_config: FitnessConfig for fitness calculation (uses balanced if None)
    
    Returns:
        (fitness, metrics) tuple
    """
    try:
        # Build features with individual's seller params
        feats = build_features(data, individual.seller_params, tf)
        
        # Run backtest with individual's backtest params
        result = run_backtest(feats, individual.backtest_params)
        
        # Calculate fitness with configurable weights
        metrics = result['metrics']
        fitness = calculate_fitness(metrics, fitness_config)
        
        return fitness, metrics
        
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        return -1000.0, {'n': 0, 'error': str(e)}


def mutate_parameter(value: float, param_name: str, sigma: float = 0.1) -> float:
    """
    Mutate a single parameter with Gaussian noise.
    
    Args:
        value: Current parameter value
        param_name: Name of parameter (for bounds lookup)
        sigma: Standard deviation as fraction of parameter range
    
    Returns:
        Mutated value clamped to bounds
    """
    if param_name not in PARAM_BOUNDS:
        return value
    
    min_val, max_val = PARAM_BOUNDS[param_name]
    param_range = max_val - min_val
    
    # Gaussian noise scaled to parameter range
    noise = np.random.normal(0, sigma * param_range)
    
    # Add noise
    new_value = value + noise
    
    # Clamp to bounds
    new_value = np.clip(new_value, min_val, max_val)
    
    # Round if original was integer
    if isinstance(value, int):
        new_value = int(round(new_value))
    
    return new_value


def mutate_individual(
    individual: Individual,
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
        if param_name in PARAM_BOUNDS and random.random() < mutation_rate:
            seller_dict[param_name] = mutate_parameter(
                seller_dict[param_name],
                param_name,
                sigma
            )
    
    # Mutate BacktestParams
    for param_name in ['atr_stop_mult', 'reward_r', 'max_hold', 'fee_bp', 'slippage_bp']:
        if param_name in PARAM_BOUNDS and random.random() < mutation_rate:
            backtest_dict[param_name] = mutate_parameter(
                backtest_dict[param_name],
                param_name,
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
    
    # Crossover BacktestParams
    def blend_backtest_params(bp1: BacktestParams, bp2: BacktestParams, a: float) -> BacktestParams:
        return BacktestParams(
            atr_stop_mult=a * bp1.atr_stop_mult + (1-a) * bp2.atr_stop_mult,
            reward_r=a * bp1.reward_r + (1-a) * bp2.reward_r,
            max_hold=int(round(a * bp1.max_hold + (1-a) * bp2.max_hold)),
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
    mutation_rate: float = 0.3,
    sigma: float = 0.1,
    elite_fraction: float = 0.1,
    tournament_size: int = 3,
    mutation_probability: float = 0.9
) -> Population:
    """
    Perform one generation of evolution.
    
    Steps:
    1. Evaluate fitness for all unevaluated individuals
    2. Selection (tournament)
    3. Crossover
    4. Mutation
    5. Elitism (preserve best individuals)
    
    Args:
        population: Current population
        data: Historical OHLCV data
        tf: Timeframe
        mutation_rate: Probability of mutating each parameter
        sigma: Mutation strength (std dev as fraction of range)
        elite_fraction: Fraction of population to preserve as elite
        tournament_size: Size of tournament for selection
        mutation_probability: Probability that an offspring undergoes mutation
    
    Returns:
        New population for next generation
    """
    pop_size = population.size
    
    # Step 1: Evaluate fitness for unevaluated individuals
    print(f"\n=== Generation {population.generation} ===")
    unevaluated = [ind for ind in population.individuals if ind.fitness == 0.0]
    
    if unevaluated:
        print(f"Evaluating {len(unevaluated)} individuals...")
        for i, ind in enumerate(unevaluated):
            fitness, metrics = evaluate_individual(ind, data, tf, fitness_config)
            ind.fitness = fitness
            ind.metrics = metrics
            print(f"  [{i+1}/{len(unevaluated)}] Fitness: {fitness:.4f} | Trades: {metrics.get('n', 0)} | Win Rate: {metrics.get('win_rate', 0):.2%}")
    
    # Update best ever
    current_best = population.get_best()
    if population.best_ever is None or current_best.fitness > population.best_ever.fitness:
        population.best_ever = deepcopy(current_best)
        print(f"🌟 NEW BEST: Fitness={current_best.fitness:.4f}")
    
    # Population statistics
    stats = population.get_stats()
    print(f"Population stats: mean={stats['mean_fitness']:.4f}, std={stats['std_fitness']:.4f}, best={stats['max_fitness']:.4f}")
    
    # Record history
    population.history.append({
        'generation': population.generation,
        'best_fitness': stats['max_fitness'],
        'mean_fitness': stats['mean_fitness'],
        'std_fitness': stats['std_fitness'],
    })
    
    # Step 2: Selection
    parents = [tournament_selection(population.individuals, tournament_size) for _ in range(pop_size)]
    
    # Step 3: Crossover
    offspring = []
    for i in range(0, len(parents) - 1, 2):
        child1, child2 = crossover(parents[i], parents[i+1], generation=population.generation + 1)
        offspring.extend([child1, child2])
    
    # Handle odd number
    if len(offspring) < pop_size:
        offspring.append(deepcopy(parents[-1]))
        offspring[-1].generation = population.generation + 1
    
    # Step 4: Mutation
    for child in offspring:
        if random.random() < mutation_probability:
            mutated = mutate_individual(child, mutation_rate, sigma, population.generation + 1)
            child.seller_params = mutated.seller_params
            child.backtest_params = mutated.backtest_params
            child.fitness = 0.0  # Reset fitness (needs re-evaluation)
    
    # Step 5: Elitism
    elite_size = max(1, int(pop_size * elite_fraction))
    elite = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)[:elite_size]
    
    # Create new generation
    new_individuals = elite + offspring[:pop_size - elite_size]
    
    # Create new population
    new_population = Population(size=pop_size)
    new_population.individuals = new_individuals
    new_population.generation = population.generation + 1
    new_population.best_ever = population.best_ever
    new_population.history = population.history
    
    return new_population
