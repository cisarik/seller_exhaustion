"""
RSI Bullish Divergence Bounce — price lower low, RSI higher low + green candle.

Round-1 insight: mean_reversion entered too late (extreme RSI + BB). Divergence
catches momentum shift earlier while still in macro downtrend — more trades
with structural confirmation than pure exhaustion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.local import ema, atr, rsi, sma, zscore
from indicators.fibonacci import add_fib_levels_to_df
from core.models import BacktestParams, Timeframe, minutes_to_bars
from strategy.base import StrategyMeta, SIGNAL_COL, finalize_signals

META = StrategyMeta(
    name="RSI Divergence Bounce",
    id="divergence_bounce",
    description="Bullish RSI divergence in downtrend with volume confirmation",
    style="deterministic",
)


@dataclass
class DivergenceParams:
    ema_fast: int = 96
    ema_slow: int = 672
    div_lookback: int = 32
    rsi_window: int = 14
    rsi_max: float = 40.0
    min_rsi_delta: float = 3.0
    vol_z_window: int = 672
    vol_z_min: float = 0.8
    cloc_min: float = 0.52
    atr_window: int = 96
    min_price_drop_pct: float = 0.003

    ema_fast_minutes: int | None = None
    ema_slow_minutes: int | None = None
    z_window_minutes: int | None = None
    atr_window_minutes: int | None = None


def _resolve(p: DivergenceParams, tf: Timeframe) -> DivergenceParams:
    if p.ema_fast_minutes is not None:
        return DivergenceParams(
            ema_fast=minutes_to_bars(p.ema_fast_minutes, tf),
            ema_slow=minutes_to_bars(p.ema_slow_minutes or 10080, tf),
            vol_z_window=minutes_to_bars(p.z_window_minutes or 10080, tf),
            atr_window=minutes_to_bars(p.atr_window_minutes or 1440, tf),
            div_lookback=p.div_lookback,
            rsi_window=p.rsi_window,
            rsi_max=p.rsi_max,
            min_rsi_delta=p.min_rsi_delta,
            vol_z_min=p.vol_z_min,
            cloc_min=p.cloc_min,
            min_price_drop_pct=p.min_price_drop_pct,
        )
    return p


def build_features(
    df: pd.DataFrame,
    p: DivergenceParams | None = None,
    tf: Timeframe = Timeframe.m15,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    p = _resolve(p or DivergenceParams(), tf)
    bt = bt_params or BacktestParams()
    out = df.copy()

    out["ema_f"] = ema(out["close"], p.ema_fast)
    out["ema_s"] = ema(out["close"], p.ema_slow)
    out["downtrend"] = out["ema_f"] < out["ema_s"]
    out["atr"] = atr(out["high"], out["low"], out["close"], p.atr_window)
    out["rsi"] = rsi(out["close"], p.rsi_window)
    out["vol_z"] = zscore(out["volume"], p.vol_z_window)

    lb = p.div_lookback
    prev_price_low = out["low"].shift(1).rolling(lb, min_periods=lb).min()
    prev_rsi_low = out["rsi"].shift(1).rolling(lb, min_periods=lb).min()

    price_lower_low = out["low"] < prev_price_low
    rsi_higher_low = out["rsi"] > prev_rsi_low + p.min_rsi_delta
    bullish_candle = out["close"] > out["open"]

    span = (out["high"] - out["low"]).replace(0, np.nan)
    out["cloc"] = (out["close"] - out["low"]) / span

    out["raw_signal"] = (
        out["downtrend"]
        & price_lower_low.fillna(False)
        & rsi_higher_low.fillna(False)
        & bullish_candle
        & (out["rsi"] < p.rsi_max)
        & (out["vol_z"] > p.vol_z_min)
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
