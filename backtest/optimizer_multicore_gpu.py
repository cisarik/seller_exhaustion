"""
Multi-core GPU optimizer with proper CUDA context management.

Uses torch.multiprocessing with 'spawn' start method for proper CUDA handling.
Each worker process gets its own CUDA context.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy
import warnings

# Use torch.multiprocessing for proper CUDA handling
try:
    import torch
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
    
    # Try to set spawn method for CUDA compatibility
    # Note: This only works if called before any multiprocessing happens
    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            mp.set_start_method('spawn')
        elif current_method != 'spawn':
            # Can't change after it's been set, will use existing method
            warnings.warn(f"Multiprocessing start method already set to '{current_method}'. "
                        f"'spawn' is recommended for GPU acceleration.")
    except RuntimeError as e:
        # Already set, continue with existing method
        warnings.warn(f"Could not set spawn method: {e}. Will use existing start method.")
        
except ImportError:
    import multiprocessing as mp
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU acceleration will not work.")

from backtest.optimizer import (
    Individual, Population, get_param_bounds_for_timeframe,
    tournament_selection, crossover, mutate_individual,
    calculate_fitness
)
from strategy.seller_exhaustion import SellerParams, build_features
import config.settings as cfg_settings
from backtest.engine import run_backtest
from core.models import BacktestParams, Timeframe, FitnessConfig


def evaluate_individual_worker_gpu(args: Tuple) -> Tuple[float, Dict[str, Any]]:
    """
    Worker function for GPU-accelerated multiprocessing.
    
    Properly initializes CUDA context in each worker process.
    Uses 'spawn' start method to avoid CUDA fork issues.
    
    Args:
        args: Tuple of (seller_params, backtest_params, data_dict, tf, fitness_config, gpu_id, use_gpu)
    
    Returns:
        (fitness, metrics) tuple
    """
    seller_params, backtest_params, data_dict, tf, fitness_config, gpu_id, use_gpu = args
    
    # Reconstruct DataFrame from dict (passed through pickle)
    # CRITICAL: Restore DatetimeIndex with proper timezone
    index = pd.to_datetime(data_dict['index'])
    if data_dict.get('index_tz'):
        index = index.tz_localize('UTC') if index.tz is None else index.tz_convert('UTC')
    data = pd.DataFrame(data_dict['values'], index=index, columns=data_dict['columns'])
    
    try:
        # Initialize CUDA in this worker process if requested
        if use_gpu and TORCH_AVAILABLE:
            try:
                import torch
                # Check if CUDA is available in this process
                if torch.cuda.is_available():
                    # Set the GPU device for this worker
                    if gpu_id is not None and gpu_id >= 0:
                        torch.cuda.set_device(gpu_id)
                    
                    # Initialize CUDA context
                    torch.cuda.init()
                    
                    # Clear any existing GPU memory
                    torch.cuda.empty_cache()
                    
                    # Run GPU-accelerated backtest
                    from backtest.spectre_trading import run_spectre_trading, _SPECTRE_AVAILABLE
                    if _SPECTRE_AVAILABLE:
                        result = run_spectre_trading(
                            data, 
                            seller_params, 
                            backtest_params, 
                            tf, 
                            use_cuda=True
                        )
                        
                        # Clean up GPU memory after evaluation
                        torch.cuda.empty_cache()
                    else:
                        # Fallback to GPU feature computation
                        use_spectre = bool(getattr(cfg_settings.settings, 'use_spectre', True))
                        feats = build_features(data, seller_params, tf, use_spectre=use_spectre)
                        result = run_backtest(feats, backtest_params)
                else:
                    # CUDA not available, use CPU
                    use_spectre = bool(getattr(cfg_settings.settings, 'use_spectre', True))
                    feats = build_features(data, seller_params, tf, use_spectre=use_spectre)
                    result = run_backtest(feats, backtest_params)
                    
            except Exception as gpu_error:
                # GPU failed, fallback to CPU
                print(f"    GPU error in worker (falling back to CPU): {gpu_error}")
                use_spectre = bool(getattr(cfg_settings.settings, 'use_spectre', True))
                feats = build_features(data, seller_params, tf, use_spectre=use_spectre)
                result = run_backtest(feats, backtest_params)
        else:
            # CPU path
            use_spectre = bool(getattr(cfg_settings.settings, 'use_spectre', True))
            feats = build_features(data, seller_params, tf, use_spectre=use_spectre)
            result = run_backtest(feats, backtest_params)
        
        fitness = calculate_fitness(result['metrics'], fitness_config)
        return fitness, result['metrics']
    
    except Exception as e:
        # Return penalty fitness on error
        print(f"    Worker error: {e}")
        return -100.0, {
            'n': 0,
            'win_rate': 0.0,
            'avg_R': 0.0,
            'total_pnl': 0.0,
            'max_dd': 0.0
        }


def evolution_step_multicore_gpu(
    population: Population,
    data: pd.DataFrame,
    tf: Timeframe = Timeframe.m15,
    fitness_config: FitnessConfig = None,
    mutation_rate: float = 0.3,
    sigma: float = 0.1,
    elite_fraction: float = 0.1,
    tournament_size: int = 3,
    mutation_probability: float = 0.9,
    n_workers: int = None,
    use_gpu: bool = True,
    gpu_id: Optional[int] = 0
) -> Population:
    """
    Multi-core GPU evolution step with proper CUDA context management.
    
    Uses torch.multiprocessing with 'spawn' start method to properly handle CUDA contexts.
    Each worker process initializes its own CUDA context.
    
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
        use_gpu: Whether to use GPU acceleration in workers
        gpu_id: GPU device ID (0 for single GPU, None to let PyTorch choose)
    
    Returns:
        New population for next generation
    """
    pop_size = population.size
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Check GPU availability
    if use_gpu and not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available, falling back to CPU")
        use_gpu = False
    
    if use_gpu and TORCH_AVAILABLE:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            use_gpu = False
    
    mode_str = "GPU-Accelerated" if use_gpu else "Multi-Core CPU"
    
    # Limit workers for GPU to avoid OOM (each worker needs ~300-900MB VRAM)
    if use_gpu and TORCH_AVAILABLE:
        import torch
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Conservative: 2-3GB per worker, limit to 3 workers max for 10GB GPU
            max_gpu_workers = max(2, min(3, int(gpu_mem_gb / 3)))
            if n_workers > max_gpu_workers:
                print(f"⚠️  Limiting workers from {n_workers} to {max_gpu_workers} to avoid GPU OOM")
                n_workers = max_gpu_workers
    
    print(f"\n=== Generation {population.generation} ({mode_str} Mode - {n_workers} workers) ===")
    
    # Step 1: Evaluate unevaluated individuals in parallel
    unevaluated = [ind for ind in population.individuals if ind.fitness == 0.0]
    
    if unevaluated:
        print(f"Evaluating {len(unevaluated)} individuals on {n_workers} workers (GPU: {use_gpu})...")
        
        # Convert DataFrame to dict for pickling
        # CRITICAL: Preserve DatetimeIndex timezone information
        data_dict = {
            'values': data.values,
            'index': data.index.tolist() if hasattr(data.index, 'tolist') else list(data.index),
            'index_tz': str(data.index.tz) if hasattr(data.index, 'tz') else None,
            'columns': data.columns.tolist()
        }
        
        # Prepare arguments for workers
        args_list = [
            (ind.seller_params, ind.backtest_params, data_dict, tf, fitness_config, gpu_id, use_gpu)
            for ind in unevaluated
        ]
        
        # Parallel evaluation with proper CUDA context handling
        try:
            if TORCH_AVAILABLE and use_gpu:
                # Use torch.multiprocessing for proper CUDA handling
                with mp.Pool(n_workers) as pool:
                    results = pool.map(evaluate_individual_worker_gpu, args_list)
            else:
                # Standard multiprocessing for CPU
                import multiprocessing as standard_mp
                with standard_mp.Pool(n_workers) as pool:
                    results = pool.map(evaluate_individual_worker_gpu, args_list)
        
        except Exception as e:
            print(f"⚠️  Multiprocessing error: {e}")
            # Fallback to serial evaluation
            print("Falling back to serial evaluation...")
            results = [evaluate_individual_worker_gpu(args) for args in args_list]
        
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
    
    # Steps 2-5: Same genetic operations as single-threaded
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
    bounds = population.bounds if hasattr(population, "bounds") else get_param_bounds_for_timeframe(tf)
    for child in offspring:
        if random.random() < mutation_probability:
            mutated = mutate_individual(
                child,
                bounds,
                mutation_rate,
                sigma,
                population.generation + 1
            )
            child.seller_params = mutated.seller_params
            child.backtest_params = mutated.backtest_params
            child.fitness = 0.0
    
    # Step 5: Elitism
    n_elite = max(1, int(pop_size * elite_fraction))
    sorted_pop = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)
    elite = sorted_pop[:n_elite]
    
    offspring[-n_elite:] = [deepcopy(ind) for ind in elite]
    
    # Create new population
    new_population = Population(size=pop_size, timeframe=population.timeframe)
    new_population.individuals = offspring
    new_population.generation = population.generation + 1
    new_population.best_ever = population.best_ever
    new_population.history = population.history
    new_population.bounds = population.bounds
    new_population.timeframe = population.timeframe
    
    return new_population
