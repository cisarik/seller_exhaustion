"""
Fusion V2 — confluence of proven OOS winners with structural floor filter.

Round 1+2 lesson: single strategies overfit or trade too rarely.
Fusion V1 used vol_squeeze (loser). V2 uses only OOS-positive configs:
  mean_reversion + seller_aggressive + liquidity_sweep

Extra filters (intuition):
  - Oversold zone: only buy near rolling N-day low (capitulation floor)
  - Weighted confidence from OOS performance (not equal vote)
  - Regime gate (weekly LLM + deterministic)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.models import BacktestParams, Timeframe, minutes_to_bars
from strategy.base import StrategyMeta, SIGNAL_COL, finalize_signals
from strategy.regime import apply_regime_gate, compute_regime
from strategy.regime_weekly import get_current_regime_gate
from strategy.optimized_build import FUSION_V2_MEMBERS, build_optimized_signals, oos_weight
from indicators.fibonacci import add_fib_levels_to_df
from indicators.local import atr

META = StrategyMeta(
    name="Fusion V2 Elite Confluence",
    id="fusion_v2",
    description="2-of-3 OOS winners + floor zone + weighted confidence + regime",
    style="hybrid",
)

CONFIG_PATH = Path("strategies_optimized/fusion_v2_params.json")


@dataclass
class FusionV2Params:
    min_agreement: int = 2
    min_confidence: float = 0.55
    min_regime_score: float = 0.48
    use_regime_gate: bool = True
    require_oversold: bool = True
    oversold_pct: float = 0.022
    oversold_lookback_days: int = 7
    min_volume_pctile: float = 0.70
    volume_lookback_days: int = 14


def default_backtest_params(tf: Timeframe = Timeframe.m15) -> BacktestParams:
    """Exit params from best OOS single strategy (mean_reversion)."""
    try:
        from exec.paper_trader import load_optimized_config, config_to_individual
        cfg = load_optimized_config("mean_reversion", tf)
        return config_to_individual(cfg).backtest_params
    except FileNotFoundError:
        return BacktestParams(
            use_fib_exits=True,
            use_stop_loss=True,
            use_time_exit=True,
            fib_target_level=0.618,
            max_hold=96,
        )


def member_weights(tf: Timeframe) -> dict[str, float]:
    raw = {sid: oos_weight(sid, tf) for sid in FUSION_V2_MEMBERS}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def build_features(
    df: pd.DataFrame,
    p: FusionV2Params | None = None,
    tf: Timeframe = Timeframe.m15,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    p = p or FusionV2Params()
    bt = bt_params
    if CONFIG_PATH.exists() and bt_params is None:
        try:
            tuned_p, tuned_bt = load_fusion_params()
            p = tuned_p
            bt = tuned_bt
        except Exception:
            pass
    bt = bt or default_backtest_params(tf)
    base = df.copy()
    weights = member_weights(tf)

    confidence = pd.Series(0.0, index=df.index)
    signals: dict[str, pd.Series] = {}

    for sid in FUSION_V2_MEMBERS:
        sig, _, _ = build_optimized_signals(sid, df, tf)
        signals[sid] = sig
        base[f"sig_{sid}"] = sig
        confidence += sig.astype(float) * weights.get(sid, 1 / 3)

    base["signal_count"] = sum(s.astype(int) for s in signals.values())
    base["confidence"] = confidence

    lookback = minutes_to_bars(p.oversold_lookback_days * 1440, tf)
    rolling_low = base["low"].rolling(lookback, min_periods=lookback // 2).min()
    base["rolling_low"] = rolling_low
    base["near_floor"] = base["close"] <= rolling_low * (1 + p.oversold_pct)

    vol_lb = minutes_to_bars(p.volume_lookback_days * 1440, tf)
    vol_rank = base["volume"].rolling(vol_lb, min_periods=vol_lb // 2).rank(pct=True)
    base["vol_pctile"] = vol_rank
    vol_ok = vol_rank >= p.min_volume_pctile

    raw = (
        (base["signal_count"] >= p.min_agreement)
        & (base["confidence"] >= p.min_confidence)
        & vol_ok.fillna(False)
    )
    if p.require_oversold:
        raw = raw & base["near_floor"].fillna(False)

    base = compute_regime(base, tf)
    if p.use_regime_gate:
        weekly = get_current_regime_gate()
        min_score = max(p.min_regime_score, weekly.get("min_regime_score", p.min_regime_score))
        raw = apply_regime_gate(raw, base["regime_score"], min_score)

    base["raw_signal"] = raw
    base["atr"] = atr(base["high"], base["low"], base["close"], max(24, lookback // 28))

    out = finalize_signals(base)
    if bt.use_fib_exits:
        out = add_fib_levels_to_df(
            out,
            signal_col=SIGNAL_COL,
            lookback=bt.fib_swing_lookback,
            lookahead=bt.fib_swing_lookahead,
        )
    return out


def save_fusion_params(p: FusionV2Params, bt: BacktestParams, metrics: dict) -> Path:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fusion_params": asdict(p),
        "backtest_params": bt.model_dump(),
        "metrics": metrics,
    }
    with CONFIG_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    return CONFIG_PATH


def load_fusion_params() -> tuple[FusionV2Params, BacktestParams]:
    if not CONFIG_PATH.exists():
        return FusionV2Params(), default_backtest_params()
    with CONFIG_PATH.open() as f:
        data = json.load(f)
    return FusionV2Params(**data["fusion_params"]), BacktestParams(**data["backtest_params"])
