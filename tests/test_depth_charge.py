"""Tests for Depth Charge tri-channel engine."""

import numpy as np
import pandas as pd
import pytest

from core.models import BacktestParams, Timeframe
from strategy.depth_charge import (
    DepthChargeParams,
    build_features,
    score_channels,
    META,
)
from backtest.engine import run_backtest
from backtest.strategy_ga import split_train_oos
from backtest.depth_tune import tune_depth_charge


@pytest.fixture
def ohlcv():
    n = 2500
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    np.random.seed(7)
    prices = 0.48 + np.cumsum(np.random.randn(n) * 0.0007)
    vol = 700 + np.abs(np.random.randn(n) * 150)
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 0.0025,
        "low": prices - 0.0025,
        "close": prices,
        "volume": vol,
    }, index=dates)
    return df


def test_meta_registered():
    from strategy.registry import get_strategy
    spec = get_strategy("depth_charge")
    assert spec["meta"].id == "depth_charge"


def test_score_channels(ohlcv):
    scored = score_channels(ohlcv, DepthChargeParams(), Timeframe.m15)
    for col in ("depth_score", "absorption_score", "deceleration_score", "conviction"):
        assert col in scored.columns
        assert scored[col].max() <= 1.01


def test_build_and_backtest(ohlcv):
    feats = build_features(ohlcv, DepthChargeParams(min_conviction=0.5), Timeframe.m15,
                           BacktestParams(use_fib_exits=False))
    assert "signal" in feats.columns
    result = run_backtest(feats, BacktestParams(use_fib_exits=False, use_stop_loss=True))
    assert "metrics" in result


def test_echo_includes_seller_aggressive():
    p = DepthChargeParams()
    assert "seller_aggressive" in p.echo_strategies


@pytest.mark.skip(reason="full grid tune is slow; covered by CLI integration")
def test_tune_depth_charge(ohlcv):
    train, oos = split_train_oos(ohlcv, 0.75)
    best = tune_depth_charge(train, oos, Timeframe.m15)
    assert best["oos"]["n"] >= 0
    assert "params" in best
