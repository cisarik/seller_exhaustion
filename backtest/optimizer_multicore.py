"""
Multi-core CPU optimizer - Correct and Fast.

Uses multiprocessing to evaluate population in parallel.
GUARANTEES correctness by using exact same code as single-threaded CPU.
"""

import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from copy import deepcopy

from backtest.optimizer import (
    Individual, Population, get_param_bounds_for_timeframe,
    tournament_selection, crossover, mutate_individual,
    calculate_fitness
)
from strategy.seller_exhaustion import SellerParams, build_features
from backtest.engine import run_backtest
from core.models import BacktestParams, Timeframe, FitnessConfig


def evaluate_individual_worker(args: Tuple) -> Tuple[float, Dict[str, Any]]:
    """
    Worker function for multiprocessing.
    
    Uses EXACT same code as CPU optimizer (guaranteed correct).
    """
    seller_params, backtest_params, data_dict, tf, fitness_config = args
    
    # Reconstruct DataFrame from dict (passed through pickle)
    data = pd.DataFrame(data_dict['values'], index=data_dict['index'], columns=data_dict['columns'])
    
    try:
        # Use EXACT CPU functions
        feats = build_features(data, seller_params, tf)
        result = run_backtest(feats, backtest_params)
        fitness = calculate_fitness(result['metrics'], fitness_config)
        
        return fitness, result['metrics']
    
    except Exception as e:
        # Return penalty fitness on error
        return -100.0, {
            'n': 0,
            'win_rate': 0.0,
            'avg_R': 0.0,
            'total_pnl': 0.0,
            'max_dd': 0.0
        }


def evolution_step_multicore(
    population: Population,
    data: pd.DataFrame,
    tf: Timeframe = Timeframe.m15,
    fitness_config: FitnessConfig = None,
    mutation_rate: float = 0.3,
    sigma: float = 0.1,
    elite_fraction: float = 0.1,
    tournament_size: int = 3,
    mutation_probability: float = 0.9,
    n_workers: int = None
) -> Population:
    """
    Multi-core evolution step.
    
    Same genetic algorithm as CPU, but evaluates population in parallel.
    GUARANTEES correctness by using exact CPU functions.
    
    Args:
        population: Current population
        data: Historical OHLCV data
        tf: Timeframe
        fitness_config: Fitness configuration
        mutation_rate: Mutation rate
        sigma: Mutation strength
        elite_fraction: Elite fraction
        tournament_size: Tournament size
        mutation_probability: Mutation probability
        n_workers: Number of worker processes (default: CPU count)
    
    Returns:
        New population for next generation
    """
    pop_size = population.size
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Step 1: Evaluate unevaluated individuals in parallel
    print(f"\n=== Generation {population.generation} (Multi-Core CPU Mode - {n_workers} workers) ===")
    unevaluated = [ind for ind in population.individuals if ind.fitness == 0.0]
    
    if unevaluated:
        print(f"Evaluating {len(unevaluated)} individuals on {n_workers} CPU cores...")
        
        # Convert DataFrame to dict for pickling
        data_dict = {
            'values': data.values,
            'index': data.index,
            'columns': data.columns.tolist()
        }
        
        # Prepare arguments for workers
        args_list = [
            (ind.seller_params, ind.backtest_params, data_dict, tf, fitness_config)
            for ind in unevaluated
        ]
        
        # Parallel evaluation
        with mp.Pool(n_workers) as pool:
            results = pool.map(evaluate_individual_worker, args_list)
        
        # Update individuals with results
        for i, (ind, (fitness, metrics)) in enumerate(zip(unevaluated, results)):
            ind.fitness = float(fitness)
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
    
    # Steps 2-5: Same genetic operations as single-threaded CPU
    # (Selection, crossover, mutation don't need parallelization)
    
    # Step 2: Selection
    parents = [tournament_selection(population.individuals, tournament_size) for _ in range(pop_size)]
    
    # Step 3: Crossover
    offspring = []
    for i in range(0, pop_size, 2):
        if i + 1 < pop_size:
            child1, child2 = crossover(parents[i], parents[i + 1])
            offspring.extend([child1, child2])
        else:
            offspring.append(deepcopy(parents[i]))
    
    offspring = offspring[:pop_size]
    
    # Step 4: Mutation
    import random
    for child in offspring:
        if random.random() < mutation_probability:
            mutate_individual(child, mutation_rate, sigma, get_param_bounds_for_timeframe(tf))
    
    # Step 5: Elitism
    n_elite = max(1, int(pop_size * elite_fraction))
    sorted_pop = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)
    elite = sorted_pop[:n_elite]
    
    offspring[-n_elite:] = [deepcopy(ind) for ind in elite]
    
    # Create new population
    new_population = Population(size=pop_size)
    new_population.individuals = offspring
    new_population.generation = population.generation + 1
    new_population.best_ever = population.best_ever
    new_population.history = population.history
    
    return new_population
