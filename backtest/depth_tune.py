"""Grid tuner for Depth Charge — OOS-first, two-phase (fast scan + fib validate)."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.profit import profit_score, simulate_account
from core.models import BacktestParams, Timeframe, minutes_to_bars
from strategy.depth_charge import (
    DepthChargeParams,
    score_channels,
    save_depth_params,
)
from strategy.regime import compute_regime, apply_regime_gate
from strategy.base import SIGNAL_COL, finalize_signals


def _fast_backtest(base: pd.DataFrame, p: DepthChargeParams, bt: BacktestParams, df: pd.DataFrame, tf: Timeframe) -> dict:
    raw = (
        (base["conviction"] >= p.min_conviction)
        & (base["depth_score"] >= p.min_depth)
        & (base["absorption_score"] >= p.min_absorption)
        & (base["deceleration_score"] >= p.min_deceleration)
        & base["at_floor"]
        & ~base["waterfall"]
    )
    raw = apply_regime_gate(raw, base["regime_score"], p.min_regime_score)
    if p.require_echo:
        from strategy.optimized_build import build_optimized_signals
        echo = pd.Series(False, index=base.index)
        for sid in p.echo_strategies:
            try:
                sig, _, _ = build_optimized_signals(sid, df, tf)
                echo = echo | sig
            except FileNotFoundError:
                continue
        raw = raw & echo
    out = base.copy()
    out["raw_signal"] = raw
    out = finalize_signals(out)
    out["atr"] = base["atr"]
    return run_backtest(out, bt)


def tune_depth_charge(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    tf: Timeframe,
) -> dict[str, Any]:
    grid = {
        "min_conviction": [0.62, 0.68, 0.74],
        "floor_pct": [0.05, 0.065, 0.08],
        "vol_pctile_min": [0.80, 0.86, 0.90],
        "min_depth": [0.70, 0.78],
        "min_absorption": [0.50, 0.58],
        "min_deceleration": [0.44, 0.50],
        "min_regime_score": [0.38, 0.44, 0.48],
        "require_echo": [True, False],
    }
    fib_levels = [0.5, 0.618, 0.786]

    base_fp = DepthChargeParams()
    train_base = compute_regime(score_channels(train, base_fp, tf), tf)
    oos_base = compute_regime(score_channels(oos, base_fp, tf), tf)

    fast_bt = BacktestParams(
        use_fib_exits=False,
        use_stop_loss=True,
        use_time_exit=True,
        atr_stop_mult=0.78,
        max_hold=80,
        reward_r=2.0,
        fee_bp=6.0,
        slippage_bp=5.5,
    )

    candidates: list[dict[str, Any]] = []
    keys = list(grid.keys())

    for combo in itertools.product(*grid.values()):
        dp = DepthChargeParams(**dict(zip(keys, combo)))
        range_lb = minutes_to_bars(dp.range_days * 1440, tf)
        for base in (train_base, oos_base):
            r_hi = base["high"].rolling(range_lb, min_periods=range_lb // 2).max()
            r_lo = base["low"].rolling(range_lb, min_periods=range_lb // 2).min()
            span = (r_hi - r_lo).replace(0, np.nan)
            range_pos = (base["close"] - r_lo) / span
            base["depth_score"] = np.clip(1.0 - range_pos / dp.floor_pct, 0, 1)
            base["at_floor"] = (range_pos <= dp.floor_pct).fillna(False)

        try:
            train_res = _fast_backtest(train_base, dp, fast_bt, train, tf)
            train_acc = simulate_account(train_res["trades"])
            train_score = profit_score(train_res["metrics"], train_acc)
            oos_res = _fast_backtest(oos_base, dp, fast_bt, oos, tf)
            oos_acc = simulate_account(oos_res["trades"])
            oos_score = profit_score(oos_res["metrics"], oos_acc)
            oos_pnl = oos_res["metrics"].get("total_pnl", 0.0)
            oos_n = oos_res["metrics"].get("n", 0)
            combined = 0.25 * train_score + 0.75 * oos_score
            if oos_n < 3:
                combined -= 2.0
            if oos_pnl <= 0:
                combined -= 2.5
            else:
                combined += min(oos_pnl * 3.0, 1.0)
            candidates.append({
                "params": dp,
                "combined": combined,
                "train": train_res["metrics"],
                "oos": oos_res["metrics"],
                "oos_score": oos_score,
                "oos_pnl": oos_pnl,
            })
        except Exception:
            continue

    if not candidates:
        raise RuntimeError("Depth Charge tune: no valid config")

    candidates.sort(key=lambda x: x["combined"], reverse=True)
    positive = [c for c in candidates if c["oos_pnl"] > 0 and c["oos"].get("n", 0) >= 2]
    top = (positive or candidates)[:8]

    from indicators.fibonacci import add_fib_levels_to_df

    best: dict[str, Any] | None = None
    for cand in top:
        dp = cand["params"]
        for fib in fib_levels:
            bt = BacktestParams(
                use_fib_exits=True,
                use_stop_loss=True,
                use_time_exit=True,
                fib_target_level=fib,
                fib_swing_lookback=72,
                fib_swing_lookahead=5,
                atr_stop_mult=0.78,
                max_hold=80,
                fee_bp=6.0,
                slippage_bp=5.5,
            )
            for base in (train_base, oos_base):
                range_lb = minutes_to_bars(dp.range_days * 1440, tf)
                r_hi = base["high"].rolling(range_lb, min_periods=range_lb // 2).max()
                r_lo = base["low"].rolling(range_lb, min_periods=range_lb // 2).min()
                span = (r_hi - r_lo).replace(0, np.nan)
                range_pos = (base["close"] - r_lo) / span
                base["depth_score"] = np.clip(1.0 - range_pos / dp.floor_pct, 0, 1)
                base["at_floor"] = (range_pos <= dp.floor_pct).fillna(False)

            raw = (
                (oos_base["conviction"] >= dp.min_conviction)
                & (oos_base["depth_score"] >= dp.min_depth)
                & (oos_base["absorption_score"] >= dp.min_absorption)
                & (oos_base["deceleration_score"] >= dp.min_deceleration)
                & oos_base["at_floor"]
                & ~oos_base["waterfall"]
            )
            raw = apply_regime_gate(raw, oos_base["regime_score"], dp.min_regime_score)
            out = oos_base.copy()
            out["raw_signal"] = raw
            out = finalize_signals(out)
            out = add_fib_levels_to_df(out, signal_col=SIGNAL_COL, lookback=72, lookahead=5)
            oos_res = run_backtest(out, bt)
            oos_acc = simulate_account(oos_res["trades"])
            oos_score = profit_score(oos_res["metrics"], oos_acc)
            row = {**cand, "bt": bt, "oos_fib": oos_res["metrics"], "oos_fib_score": oos_score}
            if best is None or oos_score > best.get("oos_fib_score", -999):
                train_raw = (
                    (train_base["conviction"] >= dp.min_conviction)
                    & (train_base["depth_score"] >= dp.min_depth)
                    & (train_base["absorption_score"] >= dp.min_absorption)
                    & (train_base["deceleration_score"] >= dp.min_deceleration)
                    & train_base["at_floor"]
                    & ~train_base["waterfall"]
                )
                train_raw = apply_regime_gate(train_raw, train_base["regime_score"], dp.min_regime_score)
                tout = train_base.copy()
                tout["raw_signal"] = train_raw
                tout = finalize_signals(tout)
                tout = add_fib_levels_to_df(tout, signal_col=SIGNAL_COL, lookback=72, lookahead=5)
                train_res = run_backtest(tout, bt)
                best = {**row, "train": train_res["metrics"], "oos": oos_res["metrics"], "oos_score": oos_score}

    if best is None:
        best = {**top[0], "bt": fast_bt}

    save_depth_params(best["params"], best["bt"], {"train": best["train"], "oos": best["oos"]})
    return best
