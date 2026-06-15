"""Tests for optimizer factory and fitness."""

import pytest

from backtest.optimizer_factory import create_optimizer, get_available_optimizers
from backtest.optimizer import calculate_fitness, Population, Individual
from core.models import FitnessConfig, BacktestParams, Timeframe
from strategy.seller_exhaustion import SellerParams


def test_only_evolutionary_available():
    assert get_available_optimizers() == ["evolutionary"]


def test_create_evolutionary_optimizer():
    opt = create_optimizer("evolutionary", n_workers=2, population_size=10)
    assert opt.get_optimizer_name().lower().startswith("evolution")
    assert opt.get_worker_count() == 2


def test_adam_optimizer_removed():
    with pytest.raises(ValueError, match="Only 'evolutionary'"):
        create_optimizer("adam")


def test_fitness_presets():
    metrics = {"n": 25, "win_rate": 0.55, "avg_R": 0.5, "total_pnl": 0.05, "max_dd": -0.02}
    balanced = calculate_fitness(metrics, FitnessConfig())
    hf = calculate_fitness(metrics, FitnessConfig.get_preset_config("high_frequency"))
    assert balanced > -50
    assert hf > -50


def test_population_seed_individual():
    seed = Individual(seller_params=SellerParams(vol_z=2.5), backtest_params=BacktestParams())
    pop = Population(size=5, seed_individual=seed, timeframe=Timeframe.m15)
    assert pop.individuals[0].seller_params.vol_z == 2.5
    assert len(pop.individuals) == 5
