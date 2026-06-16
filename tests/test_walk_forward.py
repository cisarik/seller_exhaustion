"""Tests for rolling walk-forward."""

import numpy as np
import pandas as pd

from core.models import Timeframe
from backtest.walk_forward import iter_rolling_folds, walk_forward_report, slice_last_days


def _ohlcv(n: int = 5000, freq: str = "15min") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    np.random.seed(1)
    p = 0.5 + np.cumsum(np.random.randn(n) * 0.0005)
    return pd.DataFrame({
        "open": p,
        "high": p + 0.002,
        "low": p - 0.002,
        "close": p,
        "volume": 800 + np.abs(np.random.randn(n) * 100),
    }, index=dates)


def test_iter_rolling_folds_count():
    df = _ohlcv(12000)
    folds = list(iter_rolling_folds(df, test_days=30, step_days=30, tf=Timeframe.m15, warmup_days=14))
    assert len(folds) >= 3
    fold, ctx, test = folds[0]
    assert len(test) == 30 * 96
    assert len(ctx) == len(test) + 14 * 96


def test_slice_last_days():
    df = _ohlcv(5000)
    sliced = slice_last_days(df, 30, Timeframe.m15)
    assert len(sliced) == 30 * 96


def test_walk_forward_report_runs():
    df = _ohlcv(8000)
    report = walk_forward_report(
        df,
        ["depth_charge"],
        Timeframe.m15,
        test_days=45,
        step_days=45,
        warmup_days=14,
    )
    assert "folds" in report
    assert "depth_charge" in report["summaries"]
