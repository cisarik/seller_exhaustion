"""
Evolutionary Algorithm optimizer (wraps existing GA implementation).

Supports multiple acceleration modes:
- CPU: Single-core genetic algorithm
- Multi-Core: Parallel evaluation across CPU cores
- GPU: CUDA-accelerated batch evaluation
"""

import multiprocessing
from typing import Optional
from copy import deepcopy

from backtest.optimizer_base import BaseOptimizer, OptimizationResult
from backtest.optimizer import (
    Population, Individual, evolution_step,
    get_param_bounds_for_timeframe
)
from strategy.seller_exhaustion import SellerParams
from core.models import BacktestParams, Timeframe, FitnessConfig


class EvolutionaryOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer with flexible acceleration.
    
    Uses tournament selection, arithmetic crossover, and Gaussian mutation
    with configurable elitism.
    """
    
    def __init__(
        self,
        population_size: int = 24,
        mutation_rate: float = 0.3,
        sigma: float = 0.1,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        mutation_probability: float = 0.9,
        acceleration: str = "multicore",  # cpu, multicore, or gpu
        n_workers: Optional[int] = None
    ):
        """
        Initialize evolutionary optimizer.
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutating each parameter
            sigma: Mutation strength (std dev as fraction of range)
            elite_fraction: Fraction of population preserved as elite
            tournament_size: Tournament selection size
            mutation_probability: Probability that offspring undergoes mutation
            acceleration: Acceleration mode (cpu/multicore/gpu)
            n_workers: Number of workers for multicore (defaults to CPU count)
        """
        self.config = {
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'sigma': sigma,
            'elite_fraction': elite_fraction,
            'tournament_size': tournament_size,
            'mutation_probability': mutation_probability,
        }
        self.acceleration = acceleration.lower()
        self.n_workers = n_workers or multiprocessing.cpu_count()
        self.population = None
        self.timeframe = None
        
        # Validate acceleration mode
        if self.acceleration not in ['cpu', 'multicore', 'gpu']:
            raise ValueError(f"Invalid acceleration mode: {acceleration}. Must be cpu/multicore/gpu")
        
        # Check GPU availability
        if self.acceleration == 'gpu':
            try:
                from backtest.optimizer_gpu import has_gpu
                if not has_gpu():
                    print("⚠ GPU not available, falling back to multi-core CPU")
                    self.acceleration = 'multicore'
            except ImportError:
                print("⚠ GPU acceleration not available (PyTorch not installed), falling back to multi-core CPU")
                self.acceleration = 'multicore'
        
        print(f"✓ Evolutionary optimizer initialized: {self.get_acceleration_mode()}")
    
    def initialize(
        self,
        seed_seller_params: Optional[SellerParams] = None,
        seed_backtest_params: Optional[BacktestParams] = None,
        timeframe: Timeframe = Timeframe.m15
    ) -> None:
        """Initialize population with optional seed."""
        self.timeframe = timeframe
        
        # Create seed individual if params provided
        seed_individual = None
        if seed_seller_params and seed_backtest_params:
            seed_individual = Individual(
                seller_params=seed_seller_params,
                backtest_params=seed_backtest_params
            )
        
        # Initialize population
        self.population = Population(
            size=self.config['population_size'],
            seed_individual=seed_individual,
            timeframe=timeframe
        )
        
        print(
            f"✓ Population initialized: {self.population.size} individuals | "
            f"mutation_rate={self.config['mutation_rate']:.3f} | "
            f"sigma={self.config['sigma']:.3f}"
        )
    
    def step(
        self,
        data,
        timeframe: Timeframe,
        fitness_config: FitnessConfig,
        progress_callback: Optional[callable] = None,
        stop_flag: Optional[callable] = None
    ) -> OptimizationResult:
        """Run one generation of evolution."""
        if self.population is None:
            raise RuntimeError("Optimizer not initialized. Call initialize() first.")
        
        print(f"\n{'='*60}")
        print(f"Evolution Step [{ self.get_acceleration_mode()}]")
        print(f"Generation: {self.population.generation}")
        print(f"Fitness Preset: {fitness_config.preset}")
        print(f"{'='*60}")
        
        # Choose evolution method based on acceleration
        if self.acceleration == 'multicore':
            from backtest.optimizer_multicore import evolution_step_multicore
            self.population = evolution_step_multicore(
                self.population,
                data,
                timeframe,
                fitness_config=fitness_config,
                n_workers=self.n_workers,
                mutation_rate=self.config['mutation_rate'],
                sigma=self.config['sigma'],
                elite_fraction=self.config['elite_fraction'],
                tournament_size=self.config['tournament_size'],
                mutation_probability=self.config['mutation_probability']
            )
        elif self.acceleration == 'gpu':
            from backtest.optimizer_gpu import evolution_step_gpu
            self.population = evolution_step_gpu(
                self.population,
                data,
                timeframe,
                fitness_config=fitness_config,
                mutation_rate=self.config['mutation_rate'],
                sigma=self.config['sigma'],
                elite_fraction=self.config['elite_fraction'],
                tournament_size=self.config['tournament_size'],
                mutation_probability=self.config['mutation_probability']
            )
        else:  # cpu
            self.population = evolution_step(
                self.population,
                data,
                timeframe,
                fitness_config=fitness_config,
                mutation_rate=self.config['mutation_rate'],
                sigma=self.config['sigma'],
                elite_fraction=self.config['elite_fraction'],
                tournament_size=self.config['tournament_size'],
                mutation_probability=self.config['mutation_probability']
            )
        
        # Extract best individual
        best = self.population.best_ever
        if best is None:
            raise RuntimeError("No best individual found after evolution step")
        
        # Show best metrics
        print(f"\n🏆 Best Individual:")
        print(f"   Fitness: {best.fitness:.4f}")
        print(f"   Trades: {best.metrics.get('n', 0)}")
        print(f"   Win Rate: {best.metrics.get('win_rate', 0):.2%}")
        print(f"   Avg R: {best.metrics.get('avg_R', 0):.2f}")
        print(f"   Total PnL: ${best.metrics.get('total_pnl', 0):.4f}")
        
        return OptimizationResult(
            best_seller_params=deepcopy(best.seller_params),
            best_backtest_params=deepcopy(best.backtest_params),
            fitness=best.fitness,
            metrics=best.metrics.copy(),
            iteration=self.population.generation,
            additional_info={'population_stats': self.population.get_stats()}
        )
    
    def get_best_params(self):
        """Return best parameters found so far."""
        if self.population is None or self.population.best_ever is None:
            return None, None, 0.0
        
        best = self.population.best_ever
        return (
            deepcopy(best.seller_params),
            deepcopy(best.backtest_params),
            best.fitness
        )
    
    def get_stats(self) -> dict:
        """Return population statistics."""
        if self.population is None:
            return {}
        return self.population.get_stats()
    
    def get_history(self) -> list:
        """Return optimization history."""
        if self.population is None:
            return []
        
        # Convert generation to iteration for consistency
        history = []
        for h in self.population.history:
            history.append({
                'iteration': h['generation'],
                'best_fitness': h['best_fitness'],
                'mean_fitness': h['mean_fitness'],
                'std_fitness': h.get('std_fitness', 0.0)
            })
        return history
    
    def get_optimizer_name(self) -> str:
        """Return optimizer name."""
        return "Evolutionary Algorithm"
    
    def get_acceleration_mode(self) -> str:
        """Return acceleration mode."""
        if self.acceleration == 'multicore':
            return f"Multi-Core CPU ({self.n_workers} workers)"
        elif self.acceleration == 'gpu':
            return "GPU (CUDA)"
        else:
            return "Single-Core CPU"
