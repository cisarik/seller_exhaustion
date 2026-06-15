"""
Seller Exhaustion Aggressive — more signals via relaxed thresholds.

Same logic as classic seller exhaustion but tuned for higher trade frequency.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.models import BacktestParams, Timeframe
from strategy.base import StrategyMeta, finalize_signals
from strategy.seller_exhaustion import SellerParams, _build_features_pandas

META = StrategyMeta(
    name="Seller Exhaustion Aggressive",
    id="seller_aggressive",
    description="Relaxed vol/tr/cloc thresholds for more exhaustion signals",
    style="deterministic",
)


def default_params(tf: Timeframe = Timeframe.m60) -> SellerParams:
    """Time-scaled aggressive thresholds."""
    if tf == Timeframe.m60:
        return SellerParams(
            ema_fast=24,
            ema_slow=168,
            z_window=168,
            atr_window=24,
            vol_z=1.4,
            tr_z=0.95,
            cloc_min=0.48,
            rsi_max=45.0,
        )
    return SellerParams(
        vol_z=1.5,
        tr_z=1.0,
        cloc_min=0.5,
        rsi_max=42.0,
    )


def build_features(
    df: pd.DataFrame,
    p: SellerParams | None = None,
    tf: Timeframe = Timeframe.m60,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    bt = bt_params or BacktestParams()
    params = p or default_params(tf)
    out = _build_features_pandas(
        df,
        params,
        tf,
        add_fib=bt.use_fib_exits,
        fib_lookback=bt.fib_swing_lookback,
        fib_lookahead=bt.fib_swing_lookahead,
    )
    out["raw_signal"] = out.get("exhaustion", False)
    return finalize_signals(out)
