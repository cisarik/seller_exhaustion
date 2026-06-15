"""
Factory for creating optimizer instances.
"""

from __future__ import annotations

import multiprocessing
from typing import Optional

from backtest.optimizer_base import BaseOptimizer
from backtest.optimizer_evolutionary import EvolutionaryOptimizer
import config.settings as config_settings


def create_optimizer(
    optimizer_type: str = "evolutionary",
    n_workers: Optional[int] = None,
    **kwargs,
) -> BaseOptimizer:
    """
    Create an optimizer instance.

    Only the evolutionary (genetic algorithm) optimizer is supported.
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type != "evolutionary":
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type!r}. "
            "Only 'evolutionary' is supported."
        )
    return _create_evolutionary_optimizer(n_workers=n_workers, **kwargs)


def _create_evolutionary_optimizer(n_workers: Optional[int], **kwargs) -> EvolutionaryOptimizer:
    s = config_settings.settings
    defaults = {
        "population_size": int(s.ga_population_size),
        "mutation_rate": float(s.ga_mutation_rate),
        "sigma": float(s.ga_sigma),
        "elite_fraction": float(s.ga_elite_fraction),
        "tournament_size": int(s.ga_tournament_size),
        "mutation_probability": float(s.ga_mutation_probability),
    }
    defaults.update(kwargs)

    if n_workers is None:
        try:
            n_workers = int(getattr(s, "optimizer_workers", multiprocessing.cpu_count()))
        except Exception:
            n_workers = multiprocessing.cpu_count()
    defaults["n_workers"] = max(1, n_workers)

    return EvolutionaryOptimizer(**defaults)


def get_available_optimizers() -> list[str]:
    return ["evolutionary"]


def get_optimizer_display_name(optimizer_type: str) -> str:
    names = {
        "evolutionary": "Evolutionary Algorithm",
    }
    return names.get(optimizer_type.lower(), optimizer_type)
