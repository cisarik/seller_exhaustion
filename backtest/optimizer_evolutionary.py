"""
Evolutionary Algorithm optimizer (wraps existing GA implementation).

Runs on CPU with optional multiprocessing for parallel evaluation.
"""

import multiprocessing
from typing import Optional
from copy import deepcopy
from tqdm import tqdm

from backtest.optimizer_base import BaseOptimizer, OptimizationResult
from backtest.optimizer import (
    Population, Individual, evolution_step,
    get_param_bounds_for_timeframe,
    export_population as _export_population,
)
from strategy.seller_exhaustion import SellerParams, build_features
from core.models import BacktestParams, Timeframe, FitnessConfig
from core.logging_utils import get_logger
from config.settings import settings

logger = get_logger(__name__)


class EvolutionaryOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer with configurable CPU worker count.

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
        n_workers: Optional[int] = None,
        initial_population_file: Optional[str] = None,
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
            n_workers: Number of worker processes (defaults to CPU count)
        """
        self.config = {
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'sigma': sigma,
            'elite_fraction': elite_fraction,
            'tournament_size': tournament_size,
            'mutation_probability': mutation_probability,
        }
        # Parallel evaluation (>=1 worker)
        self.n_workers = max(1, n_workers or multiprocessing.cpu_count())
        self.config['n_workers'] = self.n_workers
        self.population = None
        self.timeframe = None
        self.initial_population_file = initial_population_file
        
        # Initialization note
        logger.info("✓ Evolutionary optimizer initialized")
    
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
        
        # Initialize population (optionally from saved file)
        if self.initial_population_file:
            try:
                self.population = Population.from_file(
                    self.initial_population_file,
                    timeframe=timeframe,
                    limit=self.config['population_size']
                )
                # If user provided seed_individual as well, ensure it's present as the first individual
                if seed_individual is not None:
                    self.population.individuals.insert(0, seed_individual)
                    # Truncate to pop size
                    self.population.individuals = self.population.individuals[: self.config['population_size']]
                    self.population.size = len(self.population.individuals)
                logger.info("✓ Population loaded from file: %s (%s individuals)", self.initial_population_file, self.population.size)
            except Exception as e:
                logger.warning("⚠ Failed to load initial population from file: %s. Falling back to random initialization.", e)
                self.population = Population(
                    size=self.config['population_size'],
                    seed_individual=seed_individual,
                    timeframe=timeframe
                )
        else:
            self.population = Population(
                size=self.config['population_size'],
                seed_individual=seed_individual,
                timeframe=timeframe
            )
        
        logger.info(
            "✓ Population initialized: %s individuals | mutation_rate=%.3f | sigma=%.3f",
            self.population.size, self.config['mutation_rate'], self.config['sigma']
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
        
        logger.info(
            "[Gen %s] Evolution step | preset=%s", self.population.generation, getattr(fitness_config, 'preset', 'custom')
        )
        
        # Choose evolution method based on worker count
        if self.n_workers > 1:
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
        else:
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
        logger.info(
            "[Gen %s] Best: fit=%.4f | n=%s | wr=%.1f%% | avgR=%.2f | pnl=$%.4f",
            self.population.generation,
            best.fitness,
            best.metrics.get('n', 0),
            100.0 * (best.metrics.get('win_rate', 0.0) or 0.0),
            best.metrics.get('avg_R', 0.0) or 0.0,
            best.metrics.get('total_pnl', 0.0) or 0.0,
        )
        
        return OptimizationResult(
            best_seller_params=deepcopy(best.seller_params),
            best_backtest_params=deepcopy(best.backtest_params),
            fitness=best.fitness,
            metrics=best.metrics.copy(),
            generation=self.population.generation,
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
        
        # Return history with generation (consistent naming)
        history = []
        for h in self.population.history:
            history.append({
                'generation': h['generation'],
                'best_fitness': h['best_fitness'],
                'mean_fitness': h['mean_fitness'],
                'std_fitness': h.get('std_fitness', 0.0)
            })
        return history
    
    def get_optimizer_name(self) -> str:
        """Return optimizer name."""
        return "Evolutionary Algorithm"

    def get_worker_count(self) -> int:
        """Return configured worker count."""
        return int(self.n_workers)

    # Convenience: export current population
    def export_population(self, path: str) -> None:
        if self.population is None:
            raise RuntimeError("Population not initialized")
        _export_population(self.population, path)
