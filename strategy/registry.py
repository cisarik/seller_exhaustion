"""Strategy registry — load strategies by id."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from core.models import BacktestParams, Timeframe
from strategy.base import StrategyMeta

BuildFn = Callable[..., pd.DataFrame]


def _entry(meta: StrategyMeta, build_fn: BuildFn) -> dict:
    return {"meta": meta, "build": build_fn}


STRATEGIES: dict[str, dict] = {}


def _register():
    from strategy import seller_exhaustion
    from strategy import seller_aggressive
    from strategy import mean_reversion
    from strategy import vol_squeeze
    from strategy import fusion
    from strategy import capitulation_reversal
    from strategy import liquidity_sweep
    from strategy import divergence_bounce
    from strategy import fusion_v2
    from strategy import panic_floor
    from strategy import depth_charge

    STRATEGIES["seller_classic"] = _entry(
        StrategyMeta(
            name="Seller Exhaustion Classic",
            id="seller_classic",
            description="Original seller exhaustion (default thresholds)",
            style="deterministic",
        ),
        lambda df, tf, bt_params: seller_exhaustion.build_features(
            df,
            seller_exhaustion.SellerParams(),
            tf,
            add_fib=bt_params.use_fib_exits,
            fib_lookback=bt_params.fib_swing_lookback,
            fib_lookahead=bt_params.fib_swing_lookahead,
        ),
    )
    STRATEGIES["seller_aggressive"] = _entry(seller_aggressive.META, seller_aggressive.build_features)
    STRATEGIES["mean_reversion"] = _entry(mean_reversion.META, mean_reversion.build_features)
    STRATEGIES["vol_squeeze"] = _entry(vol_squeeze.META, vol_squeeze.build_features)
    STRATEGIES["fusion"] = _entry(fusion.META, fusion.build_features)
    STRATEGIES["capitulation_reversal"] = _entry(capitulation_reversal.META, capitulation_reversal.build_features)
    STRATEGIES["liquidity_sweep"] = _entry(liquidity_sweep.META, liquidity_sweep.build_features)
    STRATEGIES["divergence_bounce"] = _entry(divergence_bounce.META, divergence_bounce.build_features)
    STRATEGIES["fusion_v2"] = _entry(fusion_v2.META, fusion_v2.build_features)
    STRATEGIES["panic_floor"] = _entry(panic_floor.META, panic_floor.build_features)
    STRATEGIES["depth_charge"] = _entry(depth_charge.META, depth_charge.build_features)


_register()


def list_strategies() -> list[StrategyMeta]:
    return [STRATEGIES[k]["meta"] for k in sorted(STRATEGIES)]


def get_strategy(strategy_id: str) -> dict:
    if strategy_id not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {strategy_id}. Available: {list(STRATEGIES)}")
    return STRATEGIES[strategy_id]


def build(strategy_id: str, df: pd.DataFrame, tf: Timeframe, bt: BacktestParams | None = None) -> pd.DataFrame:
    spec = get_strategy(strategy_id)
    bt = bt or BacktestParams()
    return spec["build"](df, tf=tf, bt_params=bt)