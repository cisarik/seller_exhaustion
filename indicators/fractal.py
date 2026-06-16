"""
Fractal / self-similarity indicators for regime detection.

Used to boost mean-reversion entries when market shows anti-persistent structure
(Hurst < 0.5) and to mark Bill Williams fractal pivot highs/lows for Fib context.

See BRAINSTORMING.md § Fractal Market Hypothesis for research notes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hurst_exponent(series: pd.Series, window: int = 96) -> pd.Series:
    """
    Rolling Hurst exponent via simplified R/S (Rescaled Range) analysis.

    Interpretation (Mandelbrot / Peters):
    - H < 0.45: mean-reverting — favorable for bounce / exhaustion longs
    - H ≈ 0.50: random walk
    - H > 0.55: trending — unfavorable for counter-trend entries

    Args:
        series: Price series (typically close)
        window: Rolling window in bars (96 ≈ 24h on 15m)
    """
    values = series.astype(float).values
    n = len(values)
    out = np.full(n, np.nan)

    if window < 20:
        return pd.Series(out, index=series.index)

    for i in range(window, n):
        chunk = values[i - window : i]
        if np.any(np.isnan(chunk)):
            continue
        mean = chunk.mean()
        dev = chunk - mean
        cum = np.cumsum(dev)
        r = cum.max() - cum.min()
        s = chunk.std(ddof=1)
        if s <= 0 or r <= 0:
            out[i] = 0.5
            continue
        rs = r / s
        # H ≈ log(R/S) / log(n) for window n
        out[i] = float(np.clip(np.log(rs) / np.log(window), 0.0, 1.0))

    return pd.Series(out, index=series.index)


def mean_reversion_hurst_score(hurst: pd.Series) -> pd.Series:
    """Map Hurst to 0..1 score (1 = strongly mean-reverting)."""
    return ((0.55 - hurst) / 0.15).clip(0.0, 1.0)


def bill_williams_fractal_high(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """True where bar is a local maximum (fractal pivot high)."""
    roll = high.rolling(left + right + 1, center=True).max()
    return (high >= roll) & (high == roll)


def bill_williams_fractal_low(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """True where bar is a local minimum (fractal pivot low)."""
    roll = low.rolling(left + right + 1, center=True).min()
    return (low <= roll) & (low == roll)
