"""Fast grid tuner for fusion_v2 and panic_floor — no per-strategy GA."""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from backtest.engine import run_backtest
from backtest.profit import profit_score, simulate_account
from core.models import BacktestParams, Timeframe
from strategy.fusion_v2 import (
    FusionV2Params,
    save_fusion_params,
    default_backtest_params,
)
from strategy.panic_floor import PanicFloorParams, build_features as build_panic
from strategy.optimized_build import build_optimized_signals, oos_weight, FUSION_V2_MEMBERS as MEMBERS
from strategy.regime import compute_regime, apply_regime_gate
from strategy.base import SIGNAL_COL, finalize_signals
from indicators.local import atr
from indicators.fibonacci import add_fib_levels_to_df
from core.models import minutes_to_bars


def _precompute_fusion_base(df: pd.DataFrame, tf: Timeframe, fp: FusionV2Params) -> pd.DataFrame:
    """Build fusion columns once; grid only recomputes final mask."""
    base = df.copy()
    weights = {sid: oos_weight(sid, tf) for sid in MEMBERS}
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}

    confidence = pd.Series(0.0, index=df.index)
    for sid in MEMBERS:
        sig, _, _ = build_optimized_signals(sid, df, tf)
        base[f"sig_{sid}"] = sig
        confidence += sig.astype(float) * weights.get(sid, 1 / 3)

    base["signal_count"] = sum(base[f"sig_{sid}"].astype(int) for sid in MEMBERS)
    base["confidence"] = confidence

    lookback = minutes_to_bars(fp.oversold_lookback_days * 1440, tf)
    rolling_low = base["low"].rolling(lookback, min_periods=lookback // 2).min()
    base["near_floor"] = base["close"] <= rolling_low * (1 + fp.oversold_pct)

    vol_lb = minutes_to_bars(fp.volume_lookback_days * 1440, tf)
    base["vol_pctile"] = base["volume"].rolling(vol_lb, min_periods=vol_lb // 2).rank(pct=True)

    base = compute_regime(base, tf)
    base["atr"] = atr(base["high"], base["low"], base["close"], max(24, lookback // 28))
    return base


def _apply_fusion_mask(
    base: pd.DataFrame,
    fp: FusionV2Params,
    bt: BacktestParams,
) -> pd.DataFrame:
    vol_ok = base["vol_pctile"] >= fp.min_volume_pctile
    raw = (
        (base["signal_count"] >= fp.min_agreement)
        & (base["confidence"] >= fp.min_confidence)
        & vol_ok.fillna(False)
    )
    if fp.require_oversold:
        raw = raw & base["near_floor"].fillna(False)
    if fp.use_regime_gate:
        raw = apply_regime_gate(raw, base["regime_score"], fp.min_regime_score)

    out = base.copy()
    out["raw_signal"] = raw
    out = finalize_signals(out)
    if bt.use_fib_exits:
        out = add_fib_levels_to_df(
            out, signal_col=SIGNAL_COL,
            lookback=bt.fib_swing_lookback, lookahead=bt.fib_swing_lookahead,
        )
    return out


def tune_fusion_v2(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    tf: Timeframe,
) -> dict[str, Any]:
    grid = {
        "min_agreement": [2, 3],
        "min_confidence": [0.48, 0.55, 0.62],
        "min_regime_score": [0.42, 0.48, 0.52],
        "require_oversold": [True, False],
        "oversold_pct": [0.018, 0.025, 0.032],
        "min_volume_pctile": [0.68, 0.78],
    }

    bt = default_backtest_params(tf)
    base_fp = FusionV2Params()
    train_base = _precompute_fusion_base(train, tf, base_fp)
    oos_base = _precompute_fusion_base(oos, tf, base_fp)

    # Recompute near_floor per oversold_pct during search
    lookback = minutes_to_bars(base_fp.oversold_lookback_days * 1440, tf)
    for label, base in (("train", train_base), ("oos", oos_base)):
        rolling_low = base["low"].rolling(lookback, min_periods=lookback // 2).min()

    best: dict[str, Any] | None = None
    keys = list(grid.keys())

    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        fp = FusionV2Params(**params)

        for base in (train_base, oos_base):
            rl = base["low"].rolling(lookback, min_periods=lookback // 2).min()
            base["near_floor"] = base["close"] <= rl * (1 + fp.oversold_pct)

        try:
            train_feats = _apply_fusion_mask(train_base, fp, bt)
            train_res = run_backtest(train_feats, bt)
            train_acc = simulate_account(train_res["trades"])
            train_score = profit_score(train_res["metrics"], train_acc)

            oos_feats = _apply_fusion_mask(oos_base, fp, bt)
            oos_res = run_backtest(oos_feats, bt)
            oos_acc = simulate_account(oos_res["trades"])
            oos_score = profit_score(oos_res["metrics"], oos_acc)

            combined = 0.30 * train_score + 0.70 * oos_score
            if oos_res["metrics"].get("n", 0) < 2:
                combined -= 3.0
            if oos_res["metrics"].get("total_pnl", 0) <= 0:
                combined -= 0.5

            row = {
                "params": fp,
                "combined": combined,
                "train": train_res["metrics"],
                "oos": oos_res["metrics"],
                "train_score": train_score,
                "oos_score": oos_score,
                "oos_pnl": oos_res["metrics"].get("total_pnl", 0.0),
            }
            if best is None or combined > best["combined"]:
                best = row
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Fusion V2 tune found no valid configuration")

    save_fusion_params(best["params"], bt, {"train": best["train"], "oos": best["oos"]})
    return best


def tune_panic_floor(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    tf: Timeframe,
) -> dict[str, Any]:
    grid = {
        "floor_pct": [0.018, 0.025, 0.035],
        "vol_pctile_min": [0.78, 0.85, 0.90],
        "rsi_max": [34.0, 38.0, 42.0],
        "min_regime_score": [0.38, 0.42, 0.48],
    }
    bt = BacktestParams(
        use_fib_exits=True,
        use_stop_loss=True,
        use_time_exit=True,
        fib_target_level=0.618,
        max_hold=96,
        atr_stop_mult=0.75,
    )

    best: dict[str, Any] | None = None
    keys = list(grid.keys())
    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        pp = PanicFloorParams(**params)
        try:
            oos_feats = build_panic(oos, pp, tf, bt)
            oos_res = run_backtest(oos_feats, bt)
            oos_acc = simulate_account(oos_res["trades"])
            oos_score = profit_score(oos_res["metrics"], oos_acc)
            train_feats = build_panic(train, pp, tf, bt)
            train_res = run_backtest(train_feats, bt)
            train_acc = simulate_account(train_res["trades"])
            train_score = profit_score(train_res["metrics"], train_acc)
            combined = 0.3 * train_score + 0.7 * oos_score
            if oos_res["metrics"].get("n", 0) < 2:
                combined -= 3.0
            if oos_res["metrics"].get("total_pnl", 0) <= 0:
                combined -= 0.5
            row = {"params": pp, "combined": combined, "train": train_res["metrics"], "oos": oos_res["metrics"]}
            if best is None or combined > best["combined"]:
                best = row
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Panic floor tune found no valid configuration")
    return best
