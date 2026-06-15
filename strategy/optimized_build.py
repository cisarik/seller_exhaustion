"""Build strategy features using frozen GA-optimized configs."""

from __future__ import annotations

import pandas as pd

from backtest.optimizer import Individual
from core.models import BacktestParams, Timeframe
from strategy.base import SIGNAL_COL

# Proven OOS-positive members from rounds 1+2
FUSION_V2_MEMBERS = ("mean_reversion", "seller_aggressive", "liquidity_sweep")


def build_optimized_signals(
    strategy_id: str,
    df: pd.DataFrame,
    tf: Timeframe,
) -> tuple[pd.Series, BacktestParams, Individual]:
    """Return signal series + backtest params from strategies_optimized/."""
    from backtest.strategy_ga import individual_to_features
    from exec.paper_trader import load_optimized_config, config_to_individual
    from strategy.registry import build as registry_build

    try:
        cfg = load_optimized_config(strategy_id, tf)
        ind = config_to_individual(cfg)
        feats = individual_to_features(strategy_id, ind, df, tf)
        return feats[SIGNAL_COL].fillna(False).astype(bool), ind.backtest_params, ind
    except FileNotFoundError:
        from strategy.seller_exhaustion import SellerParams
        feats = registry_build(strategy_id, df, tf, BacktestParams(use_fib_exits=False))
        bt = BacktestParams()
        return feats[SIGNAL_COL].fillna(False).astype(bool), bt, Individual(SellerParams(), bt)


def oos_weight(strategy_id: str, tf: Timeframe) -> float:
    """Weight by OOS edge × sqrt(trades) — rewards consistent profit."""
    from exec.paper_trader import load_optimized_config

    try:
        cfg = load_optimized_config(strategy_id, tf)
    except FileNotFoundError:
        return 1 / 3
    m = cfg.oos_metrics
    n = max(m.get("n", 0), 0)
    pnl = max(m.get("total_pnl", 0.0), 0.0)
    pf = max(m.get("profit_factor", 0.0), 0.0)
    if n == 0:
        return 0.05
    return pnl * (n ** 0.5) + 0.02 * pf
