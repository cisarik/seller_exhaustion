"""
Multi-core evolution step — thin wrapper around parallel evaluation + GA logic.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd

from backtest.optimizer import (
    Population,
    crossover,
    get_param_bounds_for_timeframe,
    mutate_individual,
    tournament_selection,
)
from backtest.parallel import evaluate_population_parallel
from core.logging_utils import get_logger
from core.models import FitnessConfig, OptimizationConfig, Timeframe
from config.settings import settings

logger = get_logger(__name__)


def evolution_step_multicore(
    population: Population,
    data: pd.DataFrame,
    tf: Timeframe = Timeframe.m15,
    fitness_config: FitnessConfig = None,
    ga_config: OptimizationConfig = None,
    mutation_rate: float = None,
    sigma: float = None,
    elite_fraction: float = None,
    tournament_size: int = None,
    mutation_probability: float = None,
    n_workers: int = None,
) -> Population:
    """One GA generation with parallel fitness evaluation."""
    if ga_config is None:
        ga_config = OptimizationConfig(
            mutation_rate=mutation_rate or 0.3,
            sigma=sigma or 0.1,
            elite_fraction=elite_fraction or 0.1,
            tournament_size=tournament_size or 3,
            mutation_probability=mutation_probability or 0.9,
        )

    pop_size = population.size
    current_gen = population.generation

    if ga_config.override_bounds:
        population.apply_bounds_override(ga_config.override_bounds)

    evaluate_population_parallel(
        population.individuals,
        data,
        tf,
        fitness_config=fitness_config,
        generation=current_gen,
        n_workers=n_workers,
    )

    current_best = population.get_best()
    if population.best_ever is None or current_best.fitness > population.best_ever.fitness:
        population.best_ever = deepcopy(current_best)
        logger.info("[Gen %s] New best fitness = %.4f", current_gen, current_best.fitness)

    stats = population.get_stats()
    logger.info(
        "[Gen %s] Pop: mean=%.4f std=%.4f best=%.4f",
        current_gen,
        stats["mean_fitness"],
        stats["std_fitness"],
        stats["max_fitness"],
    )

    population.history.append({
        "generation": current_gen,
        "best_fitness": stats["max_fitness"],
        "mean_fitness": stats["mean_fitness"],
        "std_fitness": stats["std_fitness"],
    })

    import random

    parents = [tournament_selection(population.individuals, ga_config.tournament_size) for _ in range(pop_size)]

    offspring = []
    for i in range(0, pop_size, 2):
        if i + 1 < pop_size:
            child1, child2 = crossover(parents[i], parents[i + 1], generation=current_gen + 1)
            offspring.extend([child1, child2])
        else:
            offspring.append(deepcopy(parents[i]))

    offspring = offspring[:pop_size]
    bounds = population.bounds if hasattr(population, "bounds") else get_param_bounds_for_timeframe(tf)

    for child in offspring:
        if random.random() < ga_config.mutation_probability:
            mutated = mutate_individual(
                child,
                bounds,
                ga_config.mutation_rate,
                ga_config.sigma,
                current_gen + 1,
            )
            child.seller_params = mutated.seller_params
            child.backtest_params = mutated.backtest_params
            child.fitness = 0.0

    n_elite = max(1, int(pop_size * ga_config.elite_fraction))
    sorted_pop = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)
    elite = sorted_pop[:n_elite]
    offspring[-n_elite:] = [deepcopy(ind) for ind in elite]

    new_population = Population(size=pop_size, timeframe=population.timeframe)
    new_population.individuals = offspring
    new_population.generation = current_gen + 1
    new_population.best_ever = population.best_ever
    new_population.history = population.history
    new_population.bounds = population.bounds
    new_population.timeframe = population.timeframe

    return new_population
