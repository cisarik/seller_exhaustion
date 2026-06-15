"""Tests for strategy GA pipeline and paper trader."""

import json
import numpy as np
import pandas as pd
import pytest

from backtest.strategy_ga import (
    OPTIMIZABLE_STRATEGIES,
    split_train_oos,
    evaluate_strategy_individual,
    run_strategy_ga,
)
from backtest.optimizer import Individual, Population
from core.models import BacktestParams, FitnessConfig, Timeframe
from strategy.seller_exhaustion import SellerParams
from exec.paper_trader import save_optimized_config, config_to_individual, OptimizedStrategyConfig
from strategy.regime_weekly import update_weekly_regime, get_current_regime_gate, CACHE_PATH


@pytest.fixture
def ohlcv():
    n = 1200
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    np.random.seed(11)
    prices = 0.5 + np.cumsum(np.random.randn(n) * 0.0008)
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 0.002,
        "low": prices - 0.002,
        "close": prices,
        "volume": 900 + np.abs(np.random.randn(n) * 100),
    }, index=dates)
    return df


def test_split_train_oos(ohlcv):
    train, oos = split_train_oos(ohlcv, 0.7)
    assert len(train) + len(oos) == len(ohlcv)
    assert len(train) > len(oos)


def test_evaluate_all_strategies(ohlcv):
    ind = Individual(seller_params=SellerParams(ema_fast=48, ema_slow=288), backtest_params=BacktestParams(use_fib_exits=False))
    for sid in OPTIMIZABLE_STRATEGIES:
        fit, m = evaluate_strategy_individual(sid, ind, ohlcv, Timeframe.m15)
        assert fit > -1000 or m["n"] == 0


def test_run_strategy_ga_short(ohlcv):
    train, _ = split_train_oos(ohlcv, 0.8)
    pop = run_strategy_ga("seller_aggressive", train, Timeframe.m15, generations=2, population_size=6)
    assert pop.best_ever is not None
    assert pop.best_ever.fitness != 0.0


def test_save_load_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ind = Individual(seller_params=SellerParams(), backtest_params=BacktestParams())
    path = save_optimized_config("seller_aggressive", ind, Timeframe.m15, {"n": 5}, {"n": 2})
    assert path.exists()
    with path.open() as f:
        data = json.load(f)
    cfg = OptimizedStrategyConfig(**data)
    assert config_to_individual(cfg).seller_params.ema_fast == ind.seller_params.ema_fast


def test_weekly_regime_cache(ohlcv, tmp_path, monkeypatch):
    monkeypatch.setattr("strategy.regime_weekly.CACHE_PATH", tmp_path / "regime.json")
    entry = update_weekly_regime(ohlcv, Timeframe.m15, force=True)
    assert "score" in entry
    current = get_current_regime_gate()
    assert current["label"] in ("avoid", "neutral", "oversold_bounce", None) or current.get("week")
