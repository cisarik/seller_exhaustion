"""
GPU-accelerated evolutionary optimizer.

Uses PyTorch + CUDA for massive speedup in population evaluation.
"""

import numpy as np
import pandas as pd
from typing import Optional
from copy import deepcopy

from backtest.optimizer import (
    Individual, Population, PARAM_BOUNDS,
    tournament_selection, crossover, mutate_individual
)
from backtest.engine_gpu_full import (
    batch_backtest_full_gpu,
    has_gpu,
    calculate_fitness_gpu_batch
)
from strategy.seller_exhaustion import SellerParams
from core.models import BacktestParams, Timeframe, FitnessConfig


def evolution_step_gpu(
    population: Population,
    data: pd.DataFrame,
    tf: Timeframe = Timeframe.m15,
    fitness_config: FitnessConfig = None,
    mutation_rate: float = 0.3,
    sigma: float = 0.1,
    elite_fraction: float = 0.1,
    tournament_size: int = 3,
    mutation_probability: float = 0.9
) -> Population:
    """
    GPU-accelerated evolution step.
    
    Key optimization: Batch evaluate all unevaluated individuals on GPU at once!
    
    Args:
        population: Current population
        data: Historical OHLCV data
        tf: Timeframe
        fitness_config: Fitness configuration for scoring individuals
        mutation_rate: Probability of mutating each parameter
        sigma: Mutation strength (std dev as fraction of range)
        elite_fraction: Fraction of population to preserve as elite
        tournament_size: Tournament selection size
        mutation_probability: Probability that an offspring undergoes mutation
    
    Returns:
        New population for next generation
    """
    pop_size = population.size
    
    # Step 1: Batch evaluate all unevaluated individuals on GPU
    print(f"\n=== Generation {population.generation} (GPU Mode - Batch Processing) ===")
    unevaluated = [ind for ind in population.individuals if ind.fitness == 0.0]
    
    if unevaluated:
        print(f"Evaluating {len(unevaluated)} individuals on GPU (FULL PIPELINE - High Utilization)...")
        
        # Prepare parameter lists for batch processing
        seller_params_list = [ind.seller_params for ind in unevaluated]
        backtest_params_list = [ind.backtest_params for ind in unevaluated]
        
        # Use FULL GPU PIPELINE for 95-100% GPU utilization
        try:
            import torch
            
            # Run full GPU pipeline (features + backtest all on GPU)
            results_list, stats = batch_backtest_full_gpu(
                data,
                seller_params_list,
                backtest_params_list,
                tf,
                fitness_config=fitness_config,
                device=None,  # Auto-detect CUDA
                verbose=True
            )
            
            # Calculate fitness from metrics
            metrics_list = [r['metrics'] for r in results_list]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            fitness_tensor = calculate_fitness_gpu_batch(
                metrics_list,
                fitness_config=fitness_config,
                device=device
            )
            fitness_scores = fitness_tensor.cpu().numpy()
            
            # Update individuals with results
            for ind, fitness, metrics in zip(unevaluated, fitness_scores, metrics_list):
                ind.fitness = float(fitness)
                ind.metrics = metrics
                if metrics.get('n', 0) > 0:
                    print(f"  Fitness: {fitness:.4f} | Trades: {metrics.get('n', 0)} | Win Rate: {metrics.get('win_rate', 0):.2%}")
            
            # Show GPU stats
            print(f"\n  📊 GPU Pipeline Stats:")
            print(f"     Total time: {stats.total_time:.3f}s ({stats.total_time/stats.n_individuals*1000:.1f}ms per individual)")
            print(f"     Feature build: {stats.feature_build_time:.3f}s")
            print(f"     Backtest: {stats.backtest_time:.3f}s")
            print(f"     Total trades: {stats.total_trades}")
            
        except Exception as e:
            print(f"⚠ Full GPU pipeline failed, falling back to batch engine: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to CPU sequential
            print(f"⚠ Falling back to CPU sequential evaluation")
            from backtest.optimizer import evolution_step as cpu_evolution_step
            # Use regular CPU evolution step
            return cpu_evolution_step(
                population,
                data,
                tf,
                fitness_config,
                mutation_rate,
                sigma,
                elite_fraction,
                tournament_size,
                mutation_probability
            )
    
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
    
    # Step 2-5: Same genetic operations as CPU version
    # (Selection, crossover, mutation don't need GPU acceleration)
    
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
    bounds = population.bounds if hasattr(population, "bounds") else PARAM_BOUNDS
    for child in offspring:
        if np.random.random() < mutation_probability:
            mutated = mutate_individual(
                child,
                bounds,
                mutation_rate,
                sigma,
                population.generation + 1
            )
            child.seller_params = mutated.seller_params
            child.backtest_params = mutated.backtest_params
            child.fitness = 0.0  # Reset fitness (needs re-evaluation)
    
    # Step 5: Elitism
    elite_size = max(1, int(pop_size * elite_fraction))
    elite = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)[:elite_size]
    
    # Create new generation
    new_individuals = elite + offspring[:pop_size - elite_size]
    
    # Create new population
    new_population = Population(size=pop_size, timeframe=population.timeframe)
    new_population.individuals = new_individuals
    new_population.generation = population.generation + 1
    new_population.best_ever = population.best_ever
    new_population.history = population.history
    new_population.bounds = population.bounds
    new_population.timeframe = population.timeframe
    
    return new_population


class GPUOptimizer:
    """
    High-level GPU optimizer interface.
    
    Automatically detects GPU and falls back to CPU if unavailable.
    """
    
    def __init__(self):
        self.has_gpu = has_gpu()
        
        if self.has_gpu:
            print("✓ GPU Optimizer initialized")
        else:
            print("⚠ GPU not available, will use CPU optimizer")
    
    def evolution_step(
        self,
        population: Population,
        data: pd.DataFrame,
        tf: Timeframe = Timeframe.m15,
        **kwargs
    ) -> Population:
        """
        Run one evolution step (GPU if available, else CPU).
        
        Args:
            population: Current population
            data: Historical data
            tf: Timeframe
            **kwargs: Additional arguments for evolution_step
        
        Returns:
            New population
        """
        if self.has_gpu:
            return evolution_step_gpu(
                population,
                data,
                tf,
                **kwargs
            )
        else:
            # Fallback to CPU
            from backtest.optimizer import evolution_step
            return evolution_step(population, data, tf, **kwargs)
    
    def get_speedup_info(self, population_size: int, data_size: int) -> dict:
        """
        Get information about expected speedup.
        
        Args:
            population_size: Size of population
            data_size: Number of data points
        
        Returns:
            Dictionary with speedup information
        """
        if not self.has_gpu:
            return {
                'available': False,
                'device': 'CPU',
                'speedup': 1.0,
                'message': 'GPU not available'
            }
        
        import torch
        return {
            'available': True,
            'device': torch.cuda.get_device_name(0),
            'speedup': 10.0,  # Estimated from full GPU pipeline
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'message': f"Full GPU pipeline available on {torch.cuda.get_device_name(0)}"
        }


def benchmark_gpu_vs_cpu(
    data: pd.DataFrame,
    population_size: int = 10,
    tf: Timeframe = Timeframe.m15
):
    """
    Benchmark GPU vs CPU performance.
    
    Args:
        data: Historical data
        population_size: Number of individuals to test
        tf: Timeframe
    """
    import time
    from backtest.optimizer import Population, evolution_step
    
    print(f"\n{'='*60}")
    print("Benchmarking GPU vs CPU")
    print(f"{'='*60}")
    print(f"Population size: {population_size}")
    print(f"Data points: {len(data)}")
    
    # Create test population
    seed_ind = Individual(
        seller_params=SellerParams(),
        backtest_params=BacktestParams()
    )
    pop = Population(size=population_size, seed_individual=seed_ind)
    
    # Test GPU
    if has_gpu():
        print(f"\nTesting GPU...")
        pop_gpu = deepcopy(pop)
        
        start = time.time()
        evolution_step_gpu(pop_gpu, data, tf)
        gpu_time = time.time() - start
        
        print(f"GPU Time: {gpu_time:.2f}s")
    else:
        print("\nGPU not available")
        gpu_time = None
    
    # Test CPU
    print(f"\nTesting CPU...")
    pop_cpu = deepcopy(pop)
    
    start = time.time()
    evolution_step(pop_cpu, data, tf)
    cpu_time = time.time() - start
    
    print(f"CPU Time: {cpu_time:.2f}s")
    
    # Compare
    if gpu_time:
        speedup = cpu_time / gpu_time
        print(f"\n{'='*60}")
        print(f"Speedup: {speedup:.2f}x faster on GPU")
        print(f"{'='*60}")
    
    return {'gpu_time': gpu_time, 'cpu_time': cpu_time}
