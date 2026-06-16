"""
Depth Charge — tri-channel conviction engine (standalone, no fusion dependency).

Trading desk mental model (prompt-engineered rules):

  CHANNEL A — STRUCTURAL DEPTH
    "Is price actually cheap?" Bottom of rolling range, not mid-chop.

  CHANNEL B — FLOW ABSORPTION
    "Is someone buying the panic?" CVD proxy rising while price tags lows,
    volume exceptional, close in upper candle half.

  CHANNEL C — MOMENTUM DECELERATION
    "Is the dump exhausting?" Prior bars fell hard; current bar stabilizes;
    RSI slope turns up without needing extreme oversold.

  HARD REJECTS (desk veto):
    - Waterfall: 3+ consecutive heavy red bars (falling knife)
    - Mid-range: depth score too low
    - Regime avoid zone

Entry when conviction >= threshold AND all channel floors met.
Designed to be uncorrelated with seller-exhaustion / mean-reversion booleans.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from indicators.local import ema, atr, rsi
from indicators.fibonacci import add_fib_levels_to_df
from core.models import BacktestParams, Timeframe, minutes_to_bars
from strategy.base import StrategyMeta, SIGNAL_COL, finalize_signals
from strategy.regime import compute_regime, apply_regime_gate

META = StrategyMeta(
    name="Depth Charge",
    id="depth_charge",
    description="Tri-channel conviction + CVD absorption + proven-strategy echo confirm",
    style="deterministic",
)

PARAMS_PATH = Path("strategies_optimized/depth_charge_params.json")


@dataclass
class DepthChargeParams:
    range_days: int = 10
    floor_pct: float = 0.065
    vol_lookback_days: int = 14
    vol_pctile_min: float = 0.82
    cvd_lookback: int = 24
    rsi_max: float = 40.0
    min_depth: float = 0.75
    min_absorption: float = 0.55
    min_deceleration: float = 0.48
    min_conviction: float = 0.66
    min_regime_score: float = 0.44
    require_echo: bool = True
    echo_strategies: tuple[str, ...] = (
        "mean_reversion",
        "liquidity_sweep",
        "seller_aggressive",
    )
    waterfall_bars: int = 3
    waterfall_drop_pct: float = 0.018
    atr_window_days: int = 1

    range_minutes: int | None = None
    vol_lookback_minutes: int | None = None
    atr_window_minutes: int | None = None


def _resolve(p: DepthChargeParams, tf: Timeframe) -> DepthChargeParams:
    if p.range_minutes is not None:
        return p
    return DepthChargeParams(
        range_days=p.range_days,
        floor_pct=p.floor_pct,
        vol_lookback_days=p.vol_lookback_days,
        vol_pctile_min=p.vol_pctile_min,
        cvd_lookback=p.cvd_lookback,
        rsi_max=p.rsi_max,
        min_depth=p.min_depth,
        min_absorption=p.min_absorption,
        min_deceleration=p.min_deceleration,
        min_conviction=p.min_conviction,
        min_regime_score=p.min_regime_score,
        require_echo=p.require_echo,
        echo_strategies=p.echo_strategies,
        waterfall_bars=p.waterfall_bars,
        waterfall_drop_pct=p.waterfall_drop_pct,
        atr_window_days=p.atr_window_days,
        range_minutes=p.range_days * 1440,
        vol_lookback_minutes=p.vol_lookback_days * 1440,
        atr_window_minutes=p.atr_window_days * 1440,
    )


def _signed_volume(df: pd.DataFrame) -> pd.Series:
    body = df["close"] - df["open"]
    sign = np.sign(body.to_numpy(dtype=float))
    fallback = np.sign(df["close"].diff().to_numpy(dtype=float))
    sign = np.where(sign == 0, fallback, sign)
    return df["volume"] * pd.Series(sign, index=df.index).fillna(0)


def _waterfall_mask(close: pd.Series, bars: int, drop_pct: float) -> pd.Series:
    red = close < close.shift(1)
    streak = red.rolling(bars).sum() >= bars
    drop = close.pct_change(bars) < -drop_pct * bars
    return (streak & drop).fillna(False)


def score_channels(df: pd.DataFrame, p: DepthChargeParams, tf: Timeframe) -> pd.DataFrame:
    """Compute channel scores and conviction (no entry mask yet)."""
    p = _resolve(p, tf)
    out = df.copy()

    range_lb = minutes_to_bars(p.range_minutes or 14400, tf)
    vol_lb = minutes_to_bars(p.vol_lookback_minutes or 20160, tf)
    atr_w = max(16, minutes_to_bars(p.atr_window_minutes or 1440, tf))

    out["atr"] = atr(out["high"], out["low"], out["close"], atr_w)
    out["rsi"] = rsi(out["close"], 14)

    r_hi = out["high"].rolling(range_lb, min_periods=range_lb // 2).max()
    r_lo = out["low"].rolling(range_lb, min_periods=range_lb // 2).min()
    span = (r_hi - r_lo).replace(0, np.nan)
    range_pos = (out["close"] - r_lo) / span

    # Channel A: structural depth (1.0 = at floor)
    out["depth_score"] = np.clip(1.0 - range_pos / p.floor_pct, 0, 1)
    at_floor = range_pos <= p.floor_pct

    # Channel B: flow absorption
    vol_pct = out["volume"].rolling(vol_lb, min_periods=vol_lb // 2).rank(pct=True)
    candle = (out["high"] - out["low"]).replace(0, np.nan)
    out["cloc"] = (out["close"] - out["low"]) / candle
    lower_wick = (out[["open", "close"]].min(axis=1) - out["low"]) / candle
    bullish = out["close"] > out["open"]

    cvd = _signed_volume(out).cumsum()
    lb = p.cvd_lookback
    price_ll = out["low"] <= out["low"].shift(1).rolling(lb, min_periods=lb // 2).min()
    cvd_ll = cvd.shift(1).rolling(lb, min_periods=lb // 2).min()
    cvd_divergence = price_ll & (cvd > cvd_ll + cvd.diff(lb).abs().rolling(lb).median().fillna(0) * 0.05)
    cvd_rising = cvd.diff(max(5, lb // 4)) > 0

    vol_ok = vol_pct >= p.vol_pctile_min
    buyer_ok = bullish & (out["cloc"] > 0.52)
    wick_ok = lower_wick > 0.22
    absorb_raw = vol_ok.astype(float) * 0.30 + buyer_ok.astype(float) * 0.28
    absorb_raw += wick_ok.astype(float) * 0.17
    absorb_raw += cvd_divergence.astype(float) * 0.15
    absorb_raw += cvd_rising.astype(float) * 0.10
    out["absorption_score"] = absorb_raw.clip(0, 1)

    # Channel C: momentum deceleration
    roc3 = out["close"].pct_change(3)
    roc1 = out["close"].pct_change(1)
    was_falling = roc3 < -0.008
    stabilizing = roc1 > roc3 / 3
    rsi_slope = out["rsi"] - out["rsi"].shift(3)
    rsi_turn = (out["rsi"] < p.rsi_max) & (rsi_slope > 0.5)
    decel_raw = was_falling.astype(float) * 0.35 + stabilizing.astype(float) * 0.30
    decel_raw += rsi_turn.astype(float) * 0.35
    out["deceleration_score"] = decel_raw.clip(0, 1)

    out["conviction"] = (
        0.38 * out["depth_score"]
        + 0.37 * out["absorption_score"]
        + 0.25 * out["deceleration_score"]
    )

    out["at_floor"] = at_floor.fillna(False)
    out["cvd"] = cvd
    out["waterfall"] = _waterfall_mask(out["close"], p.waterfall_bars, p.waterfall_drop_pct)
    return out


def build_features(
    df: pd.DataFrame,
    p: DepthChargeParams | None = None,
    tf: Timeframe = Timeframe.m15,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    if PARAMS_PATH.exists() and p is None:
        p, _ = load_depth_params()

    p = _resolve(p or DepthChargeParams(), tf)
    bt = bt_params or default_backtest_params()

    out = score_channels(df, p, tf)
    out = compute_regime(out, tf)

    raw = (
        (out["conviction"] >= p.min_conviction)
        & (out["depth_score"] >= p.min_depth)
        & (out["absorption_score"] >= p.min_absorption)
        & (out["deceleration_score"] >= p.min_deceleration)
        & out["at_floor"]
        & ~out["waterfall"]
    )
    raw = apply_regime_gate(raw, out["regime_score"], p.min_regime_score)

    if p.require_echo:
        from strategy.optimized_build import build_optimized_signals
        echo = pd.Series(False, index=out.index)
        for sid in p.echo_strategies:
            try:
                sig, _, _ = build_optimized_signals(sid, df, tf)
                echo = echo | sig
            except FileNotFoundError:
                continue
        raw = raw & echo

    out["raw_signal"] = raw
    out = finalize_signals(out)

    if bt.use_fib_exits:
        out = add_fib_levels_to_df(
            out,
            signal_col=SIGNAL_COL,
            lookback=bt.fib_swing_lookback,
            lookahead=bt.fib_swing_lookahead,
        )
    return out


def default_backtest_params() -> BacktestParams:
    if PARAMS_PATH.exists():
        _, bt = load_depth_params()
        return bt
    return BacktestParams(
        use_fib_exits=True,
        use_stop_loss=True,
        use_time_exit=True,
        fib_target_level=0.618,
        fib_swing_lookback=72,
        fib_swing_lookahead=5,
        atr_stop_mult=0.78,
        max_hold=80,
        fee_bp=6.0,
        slippage_bp=5.5,
    )


def save_depth_params(p: DepthChargeParams, bt: BacktestParams, metrics: dict) -> Path:
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "depth_params": asdict(p),
        "backtest_params": bt.model_dump(),
        "metrics": metrics,
    }
    with PARAMS_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    return PARAMS_PATH


def load_depth_params() -> tuple[DepthChargeParams, BacktestParams]:
    with PARAMS_PATH.open() as f:
        data = json.load(f)
    dp_raw = dict(data["depth_params"])
    if "echo_strategies" in dp_raw and isinstance(dp_raw["echo_strategies"], list):
        dp_raw["echo_strategies"] = tuple(dp_raw["echo_strategies"])
    return DepthChargeParams(**dp_raw), BacktestParams(**data["backtest_params"])
