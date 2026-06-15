"""Tests for fusion v2 and panic floor."""

import numpy as np
import pandas as pd
import pytest

from core.models import BacktestParams, Timeframe
from strategy.panic_floor import build_features as build_panic, PanicFloorParams
from backtest.engine import run_backtest
from backtest.fusion_tune import tune_panic_floor
from backtest.strategy_ga import split_train_oos


@pytest.fixture
def ohlcv():
    n = 2000
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    np.random.seed(42)
    prices = 0.45 + np.cumsum(np.random.randn(n) * 0.0006)
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 0.003,
        "low": prices - 0.003,
        "close": prices,
        "volume": 800 + np.abs(np.random.randn(n) * 200),
    }, index=dates)
    return df


def test_panic_floor_builds(ohlcv):
    feats = build_panic(ohlcv, PanicFloorParams(), Timeframe.m15, BacktestParams(use_fib_exits=False))
    assert "signal" in feats.columns
    result = run_backtest(feats, BacktestParams(use_fib_exits=False, use_stop_loss=True))
    assert "metrics" in result


def test_panic_floor_tune(ohlcv):
    train, oos = split_train_oos(ohlcv, 0.75)
    best = tune_panic_floor(train, oos, Timeframe.m15)
    assert "params" in best
    assert "oos" in best


@pytest.mark.skipif(
    not __import__("pathlib").Path("strategies_optimized/mean_reversion_15m.json").exists(),
    reason="needs optimized configs from optimize-strategies",
)
def test_fusion_v2_builds(ohlcv):
    from strategy.fusion_v2 import build_features, FusionV2Params
    feats = build_features(ohlcv, FusionV2Params(require_oversold=False), Timeframe.m15)
    assert "signal" in feats.columns
    assert "confidence" in feats.columns
