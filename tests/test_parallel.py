"""Tests for parallel CPU evaluation."""

import pytest
import pandas as pd
import numpy as np

from backtest.optimizer import Population, Individual, evolution_step
from backtest.optimizer_multicore import evolution_step_multicore
from backtest.parallel import evaluate_population_parallel, dataframe_to_payload, payload_to_dataframe
from strategy.seller_exhaustion import SellerParams
from core.models import BacktestParams, Timeframe, FitnessConfig


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=500, freq="15min", tz="UTC")
    np.random.seed(42)
    prices = 0.5 + np.cumsum(np.random.randn(500) * 0.001)
    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(500) * 0.002),
        "low": prices - np.abs(np.random.randn(500) * 0.002),
        "close": prices + np.random.randn(500) * 0.001,
        "volume": 1000 + np.abs(np.random.randn(500) * 100),
    }, index=dates)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def test_dataframe_payload_roundtrip(sample_data):
    payload = dataframe_to_payload(sample_data)
    restored = payload_to_dataframe(payload)
    pd.testing.assert_frame_equal(restored, sample_data)


def test_parallel_evaluates_unevaluated_only(sample_data):
    seed = Individual(seller_params=SellerParams(), backtest_params=BacktestParams(), fitness=1.0)
    unevaluated = Individual(seller_params=SellerParams(), backtest_params=BacktestParams(), fitness=0.0)
    individuals = [seed, unevaluated]

    evaluate_population_parallel(individuals, sample_data, Timeframe.m15, n_workers=2)

    assert seed.fitness == 1.0
    assert unevaluated.fitness != 0.0
    assert unevaluated.metrics.get("n", 0) >= 0


def test_multicore_produces_valid_population(sample_data):
    seed = Individual(seller_params=SellerParams(), backtest_params=BacktestParams())
    pop = Population(size=6, seed_individual=seed)

    pop = evolution_step_multicore(pop, sample_data, Timeframe.m15, n_workers=2)

    assert pop.generation == 1
    assert pop.best_ever is not None
    assert pop.best_ever.fitness != 0.0


def test_single_and_multicore_same_best_ever(sample_data):
    """Best-ever fitness after gen 0 must match between single- and multi-core."""
    seed = Individual(seller_params=SellerParams(), backtest_params=BacktestParams())

    import random

    pop_single = Population(size=8, seed_individual=seed)
    random.seed(99)
    np.random.seed(99)
    pop_single = evolution_step(pop_single, sample_data, Timeframe.m15)

    pop_multi = Population(size=8, seed_individual=seed)
    random.seed(99)
    np.random.seed(99)
    pop_multi = evolution_step_multicore(pop_multi, sample_data, Timeframe.m15, n_workers=2)

    assert pop_single.best_ever is not None
    assert pop_multi.best_ever is not None
    assert abs(pop_single.best_ever.fitness - pop_multi.best_ever.fitness) < 1e-9
