"""
Liquidity Sweep & Reclaim — stop-hunt below structure then close back inside.

Round-1 insight: vol_squeeze fired on random expansions. Sweeps target a
specific microstructure event: liquidity grab below swing low + bullish reclaim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.local import ema, atr, sma, rsi
from indicators.fibonacci import add_fib_levels_to_df
from core.models import BacktestParams, Timeframe, minutes_to_bars
from strategy.base import StrategyMeta, SIGNAL_COL, finalize_signals

META = StrategyMeta(
    name="Liquidity Sweep Reclaim",
    id="liquidity_sweep",
    description="Sweep below swing low then close back above — stop hunt reversal",
    style="deterministic",
)


@dataclass
class LiquiditySweepParams:
    ema_fast: int = 96
    ema_slow: int = 672
    swing_lookback: int = 48
    sweep_pct: float = 0.0015
    vol_sma_window: int = 20
    vol_mult: float = 1.65
    rsi_window: int = 14
    rsi_max: float = 42.0
    cloc_min: float = 0.55
    atr_window: int = 96
    max_trend_gap: float = 0.06

    ema_fast_minutes: int | None = None
    ema_slow_minutes: int | None = None
    atr_window_minutes: int | None = None


def _resolve(p: LiquiditySweepParams, tf: Timeframe) -> LiquiditySweepParams:
    if p.ema_fast_minutes is not None:
        return LiquiditySweepParams(
            ema_fast=minutes_to_bars(p.ema_fast_minutes, tf),
            ema_slow=minutes_to_bars(p.ema_slow_minutes or 10080, tf),
            atr_window=minutes_to_bars(p.atr_window_minutes or 1440, tf),
            swing_lookback=p.swing_lookback,
            sweep_pct=p.sweep_pct,
            vol_sma_window=p.vol_sma_window,
            vol_mult=p.vol_mult,
            rsi_window=p.rsi_window,
            rsi_max=p.rsi_max,
            cloc_min=p.cloc_min,
            max_trend_gap=p.max_trend_gap,
        )
    return p


def build_features(
    df: pd.DataFrame,
    p: LiquiditySweepParams | None = None,
    tf: Timeframe = Timeframe.m15,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    p = _resolve(p or LiquiditySweepParams(), tf)
    bt = bt_params or BacktestParams()
    out = df.copy()

    out["ema_f"] = ema(out["close"], p.ema_fast)
    out["ema_s"] = ema(out["close"], p.ema_slow)
    out["atr"] = atr(out["high"], out["low"], out["close"], p.atr_window)
    out["rsi"] = rsi(out["close"], p.rsi_window)

    swing_low = out["low"].shift(1).rolling(p.swing_lookback, min_periods=p.swing_lookback).min()
    swept = out["low"] < swing_low * (1 - p.sweep_pct)
    reclaimed = out["close"] > swing_low
    bullish = out["close"] > out["open"]
    sweep_depth = (swing_low - out["low"]) / out["atr"].replace(0, np.nan)
    deep_enough = sweep_depth > 0.12

    vol_sma = sma(out["volume"], p.vol_sma_window)
    span = (out["high"] - out["low"]).replace(0, np.nan)
    out["cloc"] = (out["close"] - out["low"]) / span

    ema_gap = (out["ema_s"] - out["ema_f"]) / out["ema_s"]
    trend_ok = (out["ema_f"] < out["ema_s"]) | (ema_gap < p.max_trend_gap)

    out["raw_signal"] = (
        swept.fillna(False)
        & reclaimed.fillna(False)
        & bullish
        & deep_enough.fillna(False)
        & (out["volume"] > vol_sma * p.vol_mult)
        & (out["rsi"] < p.rsi_max)
        & (out["cloc"] > p.cloc_min)
        & trend_ok.fillna(False)
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
