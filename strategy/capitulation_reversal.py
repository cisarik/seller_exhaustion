"""
Capitulation Reversal — hammer wick + volume climax in downtrend.

Round-1 insight: seller_aggressive had best R but too few trades; mean_reversion
had volume but weak structure. This requires BOTH a rejection wick (long lower
shadow) AND volume climax — institutional absorption pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.local import ema, atr, rsi, zscore, bollinger_bands
from indicators.fibonacci import add_fib_levels_to_df
from core.models import BacktestParams, Timeframe, minutes_to_bars
from strategy.base import StrategyMeta, SIGNAL_COL, finalize_signals

META = StrategyMeta(
    name="Capitulation Reversal",
    id="capitulation_reversal",
    description="Hammer wick + volume climax + RSI oversold in downtrend",
    style="deterministic",
)


@dataclass
class CapitulationParams:
    ema_fast: int = 96
    ema_slow: int = 672
    z_window: int = 672
    atr_window: int = 96
    vol_z_min: float = 2.0
    wick_min: float = 0.45
    body_max: float = 0.45
    rsi_window: int = 14
    rsi_max: float = 36.0
    cloc_min: float = 0.58
    min_range_atr: float = 0.85

    ema_fast_minutes: int | None = None
    ema_slow_minutes: int | None = None
    z_window_minutes: int | None = None
    atr_window_minutes: int | None = None


def _resolve(p: CapitulationParams, tf: Timeframe) -> CapitulationParams:
    if p.ema_fast_minutes is not None:
        return CapitulationParams(
            ema_fast=minutes_to_bars(p.ema_fast_minutes, tf),
            ema_slow=minutes_to_bars(p.ema_slow_minutes or 10080, tf),
            z_window=minutes_to_bars(p.z_window_minutes or 10080, tf),
            atr_window=minutes_to_bars(p.atr_window_minutes or 1440, tf),
            vol_z_min=p.vol_z_min,
            wick_min=p.wick_min,
            body_max=p.body_max,
            rsi_window=p.rsi_window,
            rsi_max=p.rsi_max,
            cloc_min=p.cloc_min,
            min_range_atr=p.min_range_atr,
        )
    return p


def build_features(
    df: pd.DataFrame,
    p: CapitulationParams | None = None,
    tf: Timeframe = Timeframe.m15,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    p = _resolve(p or CapitulationParams(), tf)
    bt = bt_params or BacktestParams()
    out = df.copy()

    out["ema_f"] = ema(out["close"], p.ema_fast)
    out["ema_s"] = ema(out["close"], p.ema_slow)
    out["downtrend"] = out["ema_f"] < out["ema_s"]
    out["atr"] = atr(out["high"], out["low"], out["close"], p.atr_window)
    out["rsi"] = rsi(out["close"], p.rsi_window)
    out["vol_z"] = zscore(out["volume"], p.z_window)

    span = (out["high"] - out["low"]).replace(0, np.nan)
    lower_wick = (out[["open", "close"]].min(axis=1) - out["low"]) / span
    body = (out["close"] - out["open"]).abs() / span
    out["cloc"] = (out["close"] - out["low"]) / span
    range_ok = span >= out["atr"] * p.min_range_atr

    hammer = (
        (lower_wick >= p.wick_min)
        & (body <= p.body_max)
        & (out["close"] >= out["open"])
    )

    bb = bollinger_bands(out["close"], 20, 2.0)
    touch_bb = out["low"] <= bb["lower"]
    structure = hammer.fillna(False) | touch_bb.fillna(False)

    out["raw_signal"] = (
        out["downtrend"]
        & structure
        & (out["vol_z"] > p.vol_z_min)
        & (out["rsi"] < p.rsi_max)
        & (out["cloc"] > p.cloc_min)
        & range_ok.fillna(False)
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
