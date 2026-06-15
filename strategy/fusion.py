"""
Multi-strategy fusion — OR-combine signals from three engines.

Optional regime gate filters entries to favorable bounce conditions
(deterministic stand-in for LLM regime classification).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.models import BacktestParams, Timeframe
from strategy.base import StrategyMeta, SIGNAL_COL, finalize_signals
from strategy import seller_aggressive, mean_reversion, vol_squeeze
from strategy.regime import apply_regime_gate, compute_regime
from strategy.regime_weekly import get_current_regime_gate

META = StrategyMeta(
    name="Triple Fusion + Regime Gate",
    id="fusion",
    description="OR of 3 strategies, filtered by rule-based regime (LLM-ready hook)",
    style="hybrid",
)


@dataclass
class FusionParams:
    use_regime_gate: bool = True
    min_regime_score: float = 0.50
    min_agreement: int = 2  # require N strategies to agree (reduces noise)


def build_features(
    df: pd.DataFrame,
    p: FusionParams | None = None,
    tf: Timeframe = Timeframe.m60,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    p = p or FusionParams()
    bt = bt_params or BacktestParams()

    base = df.copy()
    s1 = seller_aggressive.build_features(df, tf=tf, bt_params=bt)
    s2 = mean_reversion.build_features(df, tf=tf, bt_params=bt)
    s3 = vol_squeeze.build_features(df, tf=tf, bt_params=bt)

    base["sig_seller"] = s1[SIGNAL_COL]
    base["sig_mr"] = s2[SIGNAL_COL]
    base["sig_squeeze"] = s3[SIGNAL_COL]
    base["signal_count"] = base["sig_seller"].astype(int) + base["sig_mr"].astype(int) + base["sig_squeeze"].astype(int)

    raw = base["signal_count"] >= p.min_agreement

    base = compute_regime(base, tf)
    if p.use_regime_gate:
        weekly = get_current_regime_gate()
        min_score = max(p.min_regime_score, weekly.get("min_regime_score", p.min_regime_score))
        raw = apply_regime_gate(raw, base["regime_score"], min_score)

    base["raw_signal"] = raw

    # Carry indicators needed for exits from seller path (atr, fib)
    for col in ("atr", "ema_f", "ema_s", "fib_0382", "fib_0500", "fib_0618", "fib_0786", "fib_1000", "fib_swing_high"):
        if col in s1.columns:
            base[col] = s1[col]

    if bt.use_fib_exits and "fib_0618" not in base.columns:
        from indicators.fibonacci import add_fib_levels_to_df
        base = finalize_signals(base)
        base = add_fib_levels_to_df(base, signal_col=SIGNAL_COL, lookback=bt.fib_swing_lookback, lookahead=bt.fib_swing_lookahead)
        return base

    return finalize_signals(base)
