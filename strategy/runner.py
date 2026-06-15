"""Unified strategy build for CLI, GUI, and backtests."""

from __future__ import annotations

import pandas as pd

from core.models import BacktestParams, Timeframe
from strategy.registry import build, list_strategies, get_strategy
from strategy.seller_exhaustion import SellerParams, build_features as build_seller
from strategy.base import StrategyMeta

DEFAULT_STRATEGY = "depth_charge"


def available_strategies() -> list[StrategyMeta]:
    return list_strategies()


def build_strategy(
    strategy_id: str,
    df: pd.DataFrame,
    tf: Timeframe,
    seller_params: SellerParams | None = None,
    bt_params: BacktestParams | None = None,
) -> pd.DataFrame:
    """Build features for any registered strategy."""
    if strategy_id in ("seller_classic", "seller_exhaustion"):
        sp = seller_params or SellerParams()
        bt = bt_params or BacktestParams()
        return build_seller(
            df, sp, tf,
            add_fib=bt.use_fib_exits,
            fib_lookback=bt.fib_swing_lookback,
            fib_lookahead=bt.fib_swing_lookahead,
        )
    return build(strategy_id, df, tf, bt_params or BacktestParams())
