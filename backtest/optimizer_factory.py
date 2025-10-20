"""
Factory for creating optimizer instances.

Provides unified interface for instantiating different optimizer types
with consistent configuration (notably CPU worker count).
"""

import multiprocessing
from typing import Optional

from backtest.optimizer_base import BaseOptimizer
from backtest.optimizer_evolutionary import EvolutionaryOptimizer
from backtest.optimizer_adam import AdamOptimizer
import config.settings as config_settings


def create_optimizer(
    optimizer_type: str,
    n_workers: Optional[int] = None,
    **kwargs
) -> BaseOptimizer:
    """
    Factory function to create optimizers.
    
    Args:
        optimizer_type: "evolutionary" or "adam"
        n_workers: Optional worker-process count (defaults to settings/CPU count)
        **kwargs: Optimizer-specific parameters (overrides defaults from settings)
    
    Returns:
        BaseOptimizer instance
    
    Raises:
        ValueError: If optimizer_type is invalid
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "evolutionary":
        return _create_evolutionary_optimizer(n_workers=n_workers, **kwargs)
    elif optimizer_type == "adam":
        return _create_adam_optimizer(n_workers=n_workers, **kwargs)
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Must be 'evolutionary' or 'adam'"
        )


def _create_evolutionary_optimizer(n_workers: Optional[int], **kwargs) -> EvolutionaryOptimizer:
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
    
    # Worker count
    if n_workers is None:
        try:
            n_workers = int(getattr(s, 'optimizer_workers', multiprocessing.cpu_count()))
        except Exception:
            n_workers = multiprocessing.cpu_count()
    defaults['n_workers'] = max(1, n_workers)
    
    return EvolutionaryOptimizer(**defaults)


def _create_adam_optimizer(n_workers: Optional[int], **kwargs) -> AdamOptimizer:
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
    
    # Worker count
    if n_workers is None:
        try:
            n_workers = int(getattr(s, 'optimizer_workers', multiprocessing.cpu_count()))
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
