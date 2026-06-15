"""
Volatility Squeeze Release — expansion after compression.

Detects ATR compression (squeeze) then entries on range/volume expansion
with close near highs — classic post-squeeze mean-reversion / bounce setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.local import ema, atr, zscore, sma
from indicators.fibonacci import add_fib_levels_to_df
from core.models import BacktestParams, Timeframe, minutes_to_bars
from strategy.base import StrategyMeta, SIGNAL_COL, finalize_signals

META = StrategyMeta(
    name="Volatility Squeeze Release",
    id="vol_squeeze",
    description="ATR squeeze then expansion candle with volume spike",
    style="deterministic",
)


@dataclass
class VolSqueezeParams:
    ema_fast: int = 24
    ema_slow: int = 168
    atr_window: int = 24
    squeeze_lookback: int = 48
    squeeze_percentile: float = 0.25
    expansion_mult: float = 1.35
    vol_z_window: int = 168
    vol_z_min: float = 1.2
    cloc_min: float = 0.58
    max_ema_gap_pct: float = 0.04

    ema_fast_minutes: int | None = None
    ema_slow_minutes: int | None = None
    atr_window_minutes: int | None = None
    vol_z_window_minutes: int | None = None


def _resolve(p: VolSqueezeParams, tf: Timeframe) -> VolSqueezeParams:
    if p.ema_fast_minutes is not None:
        return VolSqueezeParams(
            ema_fast=minutes_to_bars(p.ema_fast_minutes, tf),
            ema_slow=minutes_to_bars(p.ema_slow_minutes or 10080, tf),
            atr_window=minutes_to_bars(p.atr_window_minutes or 1440, tf),
            vol_z_window=minutes_to_bars(p.vol_z_window_minutes or 10080, tf),
            squeeze_lookback=p.squeeze_lookback,
            squeeze_percentile=p.squeeze_percentile,
            expansion_mult=p.expansion_mult,
            vol_z_min=p.vol_z_min,
            cloc_min=p.cloc_min,
            max_ema_gap_pct=p.max_ema_gap_pct,
        )
    return p


def build_features(
    df: pd.DataFrame,
    p: VolSqueezeParams | None = None,
    tf: Timeframe = Timeframe.m60,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    p = _resolve(p or VolSqueezeParams(), tf)
    bt = bt_params or BacktestParams()
    out = df.copy()

    out["ema_f"] = ema(out["close"], p.ema_fast)
    out["ema_s"] = ema(out["close"], p.ema_slow)
    out["atr"] = atr(out["high"], out["low"], out["close"], p.atr_window)

    tr = out["high"] - out["low"]
    atr_pct = out["atr"].rolling(p.squeeze_lookback, min_periods=p.squeeze_lookback).apply(
        lambda x: float(np.nanpercentile(x, p.squeeze_percentile * 100)),
        raw=True,
    )
    was_squeezed = out["atr"].shift(1) <= atr_pct.shift(1)

    range_expansion = tr > out["atr"] * p.expansion_mult
    out["vol_z"] = zscore(out["volume"], p.vol_z_window)

    span = tr.replace(0, np.nan)
    out["cloc"] = (out["close"] - out["low"]) / span

    ema_gap = (out["ema_s"] - out["ema_f"]) / out["ema_s"]
    trend_ok = (out["ema_f"] < out["ema_s"]) | (ema_gap < p.max_ema_gap_pct)

    out["raw_signal"] = (
        was_squeezed.fillna(False)
        & range_expansion
        & (out["vol_z"] > p.vol_z_min)
        & (out["cloc"] > p.cloc_min)
        & trend_ok
    )

    out = finalize_signals(out)

    if bt.use_fib_exits:
        out = add_fib_levels_to_df(
            out,
            signal_col=SIGNAL_COL,
            lookback=bt.fib_swing_lookback,
            lookahead=bt.fib_swing_lookahead,
        )

    return out
