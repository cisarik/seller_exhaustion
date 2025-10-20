"""
Factory for creating optimizer instances.

Provides unified interface for instantiating different optimizer types
with appropriate acceleration settings.
"""

import multiprocessing
from typing import Optional

from backtest.optimizer_base import BaseOptimizer
from backtest.optimizer_evolutionary import EvolutionaryOptimizer
from backtest.optimizer_adam import AdamOptimizer
import config.settings as config_settings


def create_optimizer(
    optimizer_type: str,
    acceleration: str,
    **kwargs
) -> BaseOptimizer:
    """
    Factory function to create optimizers.
    
    Args:
        optimizer_type: "evolutionary" or "adam"
        acceleration: "cpu" or "multicore"
        **kwargs: Optimizer-specific parameters (overrides defaults from settings)
    
    Returns:
        BaseOptimizer instance
    
    Raises:
        ValueError: If optimizer_type or acceleration is invalid
    
    Example:
        # Create evolutionary optimizer with multi-core acceleration
        opt = create_optimizer("evolutionary", "multicore", population_size=48)
        
        # Create ADAM optimizer (CPU or multicore)
        opt = create_optimizer("adam", "multicore", learning_rate=0.01)
    """
    optimizer_type = optimizer_type.lower()
    # Normalize and validate acceleration
    acceleration = (acceleration or 'cpu').lower()
    available = get_available_accelerations(optimizer_type)
    if acceleration not in available:
        # Fallback safely to CPU if unsupported
        acceleration = 'cpu'
    
    if optimizer_type == "evolutionary":
        return _create_evolutionary_optimizer(acceleration, **kwargs)
    elif optimizer_type == "adam":
        return _create_adam_optimizer(acceleration, **kwargs)
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Must be 'evolutionary' or 'adam'"
        )


def _create_evolutionary_optimizer(acceleration: str, **kwargs) -> EvolutionaryOptimizer:
    """
    Create evolutionary algorithm optimizer.
    
    Defaults are loaded from settings, can be overridden via kwargs.
    """
    # Load defaults from settings
    s = config_settings.settings
    defaults = {
        'population_size': int(s.ga_population_size),
        'mutation_rate': float(s.ga_mutation_rate),
        'sigma': float(s.ga_sigma),
        'elite_fraction': float(s.ga_elite_fraction),
        'tournament_size': int(s.ga_tournament_size),
        'mutation_probability': float(s.ga_mutation_probability),
    }
    
    # Override with kwargs
    defaults.update(kwargs)
    
    # Add acceleration (pass-through)
    defaults['acceleration'] = acceleration
    
    # Add n_workers for multicore (from settings or default to cpu_count)
    if acceleration == 'multicore':
        try:
            n_workers = int(getattr(s, 'cpu_workers', multiprocessing.cpu_count()))
        except Exception:
            n_workers = multiprocessing.cpu_count()
        defaults['n_workers'] = max(1, n_workers)
    
    return EvolutionaryOptimizer(**defaults)


def _create_adam_optimizer(acceleration: str, **kwargs) -> AdamOptimizer:
    """
    Create ADAM optimizer.
    
    Defaults are loaded from settings, can be overridden via kwargs.
    """
    # Load defaults from settings
    s = config_settings.settings
    defaults = {
        'learning_rate': float(s.adam_learning_rate),
        'epsilon': float(s.adam_epsilon),
        'max_grad_norm': float(s.adam_max_grad_norm),
        'adam_beta1': 0.9,  # Momentum (not configurable in settings)
        'adam_beta2': 0.999,  # RMSprop-like (not configurable in settings)
        'adam_epsilon': 1e-8,  # Numerical stability (not configurable in settings)
    }
    
    # Override with kwargs
    defaults.update(kwargs)
    
    # Add acceleration (pass-through)
    defaults['acceleration'] = acceleration
    
    # Add n_workers for multicore (from settings or default to cpu_count)
    if acceleration == 'multicore':
        try:
            n_workers = int(getattr(s, 'cpu_workers', multiprocessing.cpu_count()))
        except Exception:
            n_workers = multiprocessing.cpu_count()
        defaults['n_workers'] = max(1, n_workers)
    
    return AdamOptimizer(**defaults)


def get_available_optimizers() -> list[str]:
    """
    Get list of available optimizer types.
    
    Returns:
        List of optimizer names
    """
    return ['evolutionary', 'adam']


def get_available_accelerations(optimizer_type: str) -> list[str]:
    """
    Get list of available acceleration modes for an optimizer.
    
    Args:
        optimizer_type: Optimizer type
    
    Returns:
        List of acceleration modes
    """
    # CPU is always available; enable multi-core universally
    modes = ['cpu', 'multicore']
    return modes


def get_optimizer_display_name(optimizer_type: str) -> str:
    """
    Get human-readable display name for optimizer.
    
    Args:
        optimizer_type: Optimizer type
    
    Returns:
        Display name
    """
    names = {
        'evolutionary': 'Evolutionary Algorithm',
        'adam': 'ADAM',
    }
    return names.get(optimizer_type.lower(), optimizer_type)


def get_acceleration_display_name(acceleration: str, optimizer_type: str = None) -> str:
    """
    Get human-readable display name for acceleration mode.
    
    Args:
        acceleration: Acceleration mode
        optimizer_type: Optional optimizer type (for context)
    
    Returns:
        Display name
    """
    if acceleration == 'multicore':
        try:
            # Prefer configured workers if available
            import config.settings as _cfg
            n_workers = int(getattr(_cfg.settings, 'cpu_workers', multiprocessing.cpu_count()))
        except Exception:
            n_workers = multiprocessing.cpu_count()
        return f'Multi-Core CPU ({n_workers} workers)'
    else:  # cpu
        return 'Single-Core CPU'
