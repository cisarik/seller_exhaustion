"""
GPU-aware optimizer - Uses single GPU with serial evaluation.

For environments where GPU evaluation should be done in a single process
rather than distributing across CPU workers.

This is more efficient than multicore CPU when GPU is available.
"""

from copy import deepcopy
from backtest.optimizer import (
    Individual, Population, tournament_selection, 
    crossover, mutate_individual, calculate_fitness
)
from strategy.seller_exhaustion import build_features
import config.settings as cfg_settings
from core.models import Timeframe, FitnessConfig
import numpy as np


def evolution_step_gpu(
    population: Population,
    data,
    tf: Timeframe = Timeframe.m15,
    fitness_config: FitnessConfig = None,
    mutation_rate: float = 0.3,
    sigma: float = 0.1,
    elite_fraction: float = 0.1,
    tournament_size: int = 3,
    mutation_probability: float = 0.9
):
    """
    GPU-optimized evolution step.
    
    Evaluates individuals serially on GPU (not multicore).
    More efficient than multicore CPU when GPU is available.
    """
    from backtest.spectre_trading import run_spectre_trading, _SPECTRE_AVAILABLE
    from backtest.engine import run_backtest
    
    pop_size = population.size
    print(f"\n=== Generation {population.generation} (GPU Accelerated Mode) ===")
    
    # Check if GPU should be used
    use_gpu = _SPECTRE_AVAILABLE and bool(getattr(cfg_settings.settings, 'use_spectre_cuda', False))
    print(f"GPU Available: {_SPECTRE_AVAILABLE}, Using GPU: {use_gpu}")
    
    # Evaluate unevaluated individuals
    unevaluated = [ind for ind in population.individuals if ind.fitness == 0.0]
    
    if unevaluated:
        print(f"Evaluating {len(unevaluated)} individuals on GPU...")
        for i, ind in enumerate(unevaluated):
            try:
                if use_gpu:
                    # GPU path
                    result = run_spectre_trading(
                        data,
                        ind.seller_params,
                        ind.backtest_params,
                        tf,
                        use_cuda=True
                    )
                else:
                    # CPU fallback
                    feats = build_features(
                        data,
                        ind.seller_params,
                        tf,
                        use_spectre=bool(getattr(cfg_settings.settings, 'use_spectre', True))
                    )
                    from backtest.engine import run_backtest
                    result = run_backtest(feats, ind.backtest_params)
                
                metrics = result['metrics']
                fitness = calculate_fitness(metrics, fitness_config)
                
            except Exception as e:
                print(f"    Error evaluating individual {i}: {e}")
                fitness = -100.0
                metrics = {'n': 0, 'win_rate': 0.0, 'avg_R': 0.0, 'total_pnl': 0.0, 'max_dd': 0.0}
            
            ind.fitness = fitness
            ind.metrics = metrics
            
            if i % 5 == 0 or i == len(unevaluated) - 1:
                print(f"  [{i+1}/{len(unevaluated)}] Fitness: {fitness:.4f} | Trades: {metrics.get('n', 0)}")
    
    # Update best ever
    current_best = population.get_best()
    if population.best_ever is None or current_best.fitness > population.best_ever.fitness:
        population.best_ever = deepcopy(current_best)
        print(f"🌟 NEW BEST: Fitness={current_best.fitness:.4f}, Trades={current_best.metrics.get('n', 0)}")
    
    # Selection and reproduction
    n_elite = max(1, int(pop_size * elite_fraction))
    elite_individuals = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)[:n_elite]
    
    # Create offspring
    offspring = []
    while len(offspring) < pop_size - n_elite:
        parent1 = tournament_selection(population.individuals, tournament_size)
        parent2 = tournament_selection(population.individuals, tournament_size)
        
        child_seller, child_backtest = crossover(
            parent1.seller_params, parent1.backtest_params,
            parent2.seller_params, parent2.backtest_params,
            population.bounds
        )
        
        if np.random.rand() < mutation_probability:
            child_seller, child_backtest = mutate_individual(
                child_seller, child_backtest,
                population.bounds,
                mutation_rate,
                sigma
            )
        
        offspring.append(Individual(
            seller_params=child_seller,
            backtest_params=child_backtest,
            generation=population.generation + 1
        ))
    
    # Assemble new population
    population.individuals = elite_individuals + offspring[:pop_size - n_elite]
    population.generation += 1
    
    return population
