"""Tests for fractal / Hurst indicators."""

import numpy as np
import pandas as pd

from indicators.fractal import (
    bill_williams_fractal_high,
    hurst_exponent,
    mean_reversion_hurst_score,
)


def test_hurst_returns_series_same_length():
    n = 200
    close = pd.Series(np.cumsum(np.random.randn(n)) + 100, index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"))
    h = hurst_exponent(close, window=48)
    assert len(h) == n
    assert h.iloc[:48].isna().all()
    assert h.iloc[-1] == h.iloc[-1]  # not nan at end (usually)


def test_mean_reversion_hurst_score_bounds():
    h = pd.Series([0.3, 0.5, 0.7])
    s = mean_reversion_hurst_score(h)
    assert s.min() >= 0.0
    assert s.max() <= 1.0
    assert s.iloc[0] > s.iloc[2]  # lower Hurst → higher MR score


def test_fractal_high_detects_peak():
    high = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    peaks = bill_williams_fractal_high(high, left=2, right=2)
    assert peaks.iloc[2]
