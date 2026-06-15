"""
Mean Reversion Snapback — deterministic, higher-frequency entries.

Buys capitulation bounces in downtrends: oversold RSI + lower Bollinger touch
+ volume confirmation + bullish close location.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.local import ema, atr, rsi, sma, bollinger_bands
from indicators.fibonacci import add_fib_levels_to_df
from core.models import BacktestParams, Timeframe, minutes_to_bars
from strategy.base import StrategyMeta, SIGNAL_COL, finalize_signals

META = StrategyMeta(
    name="Mean Reversion Snapback",
    id="mean_reversion",
    description="RSI oversold + Bollinger lower band touch in downtrend",
    style="deterministic",
)


@dataclass
class MeanReversionParams:
    ema_fast: int = 24
    ema_slow: int = 168
    rsi_window: int = 14
    rsi_max: float = 28.0
    bb_window: int = 20
    bb_std: float = 2.0
    vol_sma_window: int = 20
    vol_mult: float = 1.35
    cloc_min: float = 0.55
    atr_window: int = 24

    ema_fast_minutes: int | None = None
    ema_slow_minutes: int | None = None
    atr_window_minutes: int | None = None


def _bars(p: MeanReversionParams, tf: Timeframe) -> MeanReversionParams:
    if p.ema_fast_minutes is not None:
        return MeanReversionParams(
            ema_fast=minutes_to_bars(p.ema_fast_minutes, tf),
            ema_slow=minutes_to_bars(p.ema_slow_minutes or 10080, tf),
            atr_window=minutes_to_bars(p.atr_window_minutes or 1440, tf),
            rsi_window=p.rsi_window,
            rsi_max=p.rsi_max,
            bb_window=p.bb_window,
            bb_std=p.bb_std,
            vol_sma_window=p.vol_sma_window,
            vol_mult=p.vol_mult,
            cloc_min=p.cloc_min,
        )
    return p


def build_features(
    df: pd.DataFrame,
    p: MeanReversionParams | None = None,
    tf: Timeframe = Timeframe.m60,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    p = _bars(p or MeanReversionParams(), tf)
    bt = bt_params or BacktestParams()
    out = df.copy()

    out["ema_f"] = ema(out["close"], p.ema_fast)
    out["ema_s"] = ema(out["close"], p.ema_slow)
    out["downtrend"] = out["ema_f"] < out["ema_s"]
    out["atr"] = atr(out["high"], out["low"], out["close"], p.atr_window)
    out["rsi"] = rsi(out["close"], p.rsi_window)

    bb = bollinger_bands(out["close"], p.bb_window, p.bb_std)
    out["bb_lower"] = bb["lower"]
    out["bb_mid"] = bb["mid"]

    vol_sma = sma(out["volume"], p.vol_sma_window)
    span = (out["high"] - out["low"]).replace(0, np.nan)
    out["cloc"] = (out["close"] - out["low"]) / span

    touched_lower = out["low"] <= out["bb_lower"]
    vol_ok = out["volume"] > vol_sma * p.vol_mult

    out["raw_signal"] = (
        out["downtrend"]
        & (out["rsi"] < p.rsi_max)
        & touched_lower
        & vol_ok
        & (out["cloc"] > p.cloc_min)
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
