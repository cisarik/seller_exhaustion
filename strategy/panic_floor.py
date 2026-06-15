"""
Panic Floor Hunter — ground-up capitulation strategy (minimal params, hard to overfit).

Intuition: ADA bounces happen at STRUCTURAL lows after panic volume, not mid-range.
Only enter when price is in the bottom 2% of a 7-day range AND volume is extreme
AND buyers show up same bar (green + close high in range).

No GA required for core logic — tune only 2-3 thresholds via grid search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.local import ema, atr, rsi
from indicators.fibonacci import add_fib_levels_to_df
from core.models import BacktestParams, Timeframe, minutes_to_bars
from strategy.base import StrategyMeta, SIGNAL_COL, finalize_signals
from strategy.regime import compute_regime, apply_regime_gate

META = StrategyMeta(
    name="Panic Floor Hunter",
    id="panic_floor",
    description="7d range floor + panic volume + same-bar buyer response",
    style="deterministic",
)


@dataclass
class PanicFloorParams:
    range_days: int = 7
    floor_pct: float = 0.025
    vol_pctile_min: float = 0.82
    vol_lookback_days: int = 14
    rsi_max: float = 38.0
    cloc_min: float = 0.58
    ema_slow_days: int = 7
    min_regime_score: float = 0.42
    dump_guard_pct: float = 0.04

    range_minutes: int | None = None
    vol_lookback_minutes: int | None = None
    ema_slow_minutes: int | None = None


def _resolve(p: PanicFloorParams, tf: Timeframe) -> PanicFloorParams:
    if p.range_minutes is not None:
        return p
    return PanicFloorParams(
        range_days=p.range_days,
        floor_pct=p.floor_pct,
        vol_pctile_min=p.vol_pctile_min,
        vol_lookback_days=p.vol_lookback_days,
        rsi_max=p.rsi_max,
        cloc_min=p.cloc_min,
        ema_slow_days=p.ema_slow_days,
        min_regime_score=p.min_regime_score,
        dump_guard_pct=p.dump_guard_pct,
        range_minutes=p.range_days * 1440,
        vol_lookback_minutes=p.vol_lookback_days * 1440,
        ema_slow_minutes=p.ema_slow_days * 1440,
    )


def build_features(
    df: pd.DataFrame,
    p: PanicFloorParams | None = None,
    tf: Timeframe = Timeframe.m15,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    p = _resolve(p or PanicFloorParams(), tf)
    bt = bt_params or BacktestParams(
        use_fib_exits=True,
        use_stop_loss=True,
        use_time_exit=True,
        fib_target_level=0.618,
        max_hold=96,
        atr_stop_mult=0.75,
    )

    range_lb = minutes_to_bars(p.range_minutes or 10080, tf)
    vol_lb = minutes_to_bars(p.vol_lookback_minutes or 20160, tf)
    ema_slow = minutes_to_bars(p.ema_slow_minutes or 10080, tf)
    atr_w = max(24, range_lb // 28)

    out = df.copy()
    out["ema_s"] = ema(out["close"], ema_slow)
    out["atr"] = atr(out["high"], out["low"], out["close"], atr_w)
    out["rsi"] = rsi(out["close"], 14)

    range_low = out["low"].rolling(range_lb, min_periods=range_lb // 2).min()
    range_high = out["high"].rolling(range_lb, min_periods=range_lb // 2).max()
    span = (range_high - range_low).replace(0, np.nan)
    position_in_range = (out["close"] - range_low) / span

    at_floor = position_in_range <= p.floor_pct
    vol_pctile = out["volume"].rolling(vol_lb, min_periods=vol_lb // 2).rank(pct=True)
    panic_vol = vol_pctile >= p.vol_pctile_min

    candle_span = (out["high"] - out["low"]).replace(0, np.nan)
    out["cloc"] = (out["close"] - out["low"]) / candle_span
    buyer_response = (out["close"] > out["open"]) & (out["cloc"] > p.cloc_min)

    dump_guard = out["close"] >= out["close"].shift(3) * (1 - p.dump_guard_pct)

    out = compute_regime(out, tf)
    regime_ok = out["regime_score"] >= p.min_regime_score

    out["raw_signal"] = (
        at_floor.fillna(False)
        & panic_vol.fillna(False)
        & buyer_response.fillna(False)
        & (out["rsi"] < p.rsi_max)
        & dump_guard.fillna(False)
        & regime_ok.fillna(False)
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
