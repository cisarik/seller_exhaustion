"""Tests for multi-strategy framework."""

import numpy as np
import pandas as pd
import pytest

from core.models import BacktestParams, Timeframe
from backtest.engine import run_backtest
from backtest.compare import compare_all, run_single
from strategy.registry import list_strategies, build
from strategy.base import SIGNAL_COL
from indicators.local import bollinger_bands


@pytest.fixture
def ohlcv():
    n = 800
    dates = pd.date_range("2024-06-01", periods=n, freq="1h", tz="UTC")
    np.random.seed(7)
    prices = 0.55 + np.cumsum(np.random.randn(n) * 0.002)
    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.003),
        "low": prices - np.abs(np.random.randn(n) * 0.003),
        "close": prices + np.random.randn(n) * 0.001,
        "volume": 800 + np.abs(np.random.randn(n) * 200),
    }, index=dates)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def test_registry_lists_five_strategies():
    ids = {m.id for m in list_strategies()}
    assert "seller_classic" in ids
    assert "seller_aggressive" in ids
    assert "mean_reversion" in ids
    assert "vol_squeeze" in ids
    assert "fusion" in ids


def test_each_strategy_builds_signal_column(ohlcv):
    bt = BacktestParams()
    for meta in list_strategies():
        feats = build(meta.id, ohlcv, Timeframe.m60, bt)
        assert SIGNAL_COL in feats.columns
        assert "atr" in feats.columns


def test_aggressive_has_more_signals_than_classic(ohlcv):
    bt = BacktestParams(use_fib_exits=False)
    classic = build("seller_classic", ohlcv, Timeframe.m60, bt)
    aggressive = build("seller_aggressive", ohlcv, Timeframe.m60, bt)
    assert aggressive[SIGNAL_COL].sum() >= classic[SIGNAL_COL].sum()


def test_fusion_runs_backtest(ohlcv):
    bt = BacktestParams(use_fib_exits=True, use_stop_loss=True, use_time_exit=True, max_hold=48)
    feats = build("fusion", ohlcv, Timeframe.m60, bt)
    result = run_backtest(feats, bt)
    assert "metrics" in result
    assert result["metrics"]["n"] >= 0


def test_compare_all_returns_sorted(ohlcv):
    df = compare_all(ohlcv, Timeframe.m60)
    assert len(df) == 11
    assert "profit_score" in df.columns
    assert df["profit_score"].is_monotonic_decreasing


def test_bollinger_bands():
    s = pd.Series(np.linspace(1, 2, 50))
    bb = bollinger_bands(s, 20, 2.0).dropna()
    assert (bb["upper"] >= bb["mid"]).all()
    assert (bb["mid"] >= bb["lower"]).all()
