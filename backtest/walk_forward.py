"""
Rolling walk-forward evaluation for frozen strategies (no re-tune).

Each fold: build features on context (warmup + test), backtest only the test window.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterator

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.profit import simulate_account
from core.models import BacktestParams, Timeframe
from exec.paper_trader import FROZEN_STRATEGIES


WARMUP_DAYS_DEFAULT = 14


@dataclass
class FoldResult:
    fold: int
    strategy_id: str
    test_start: str
    test_end: str
    bars: int
    trades: int
    total_pnl: float
    win_rate: float
    expectancy_r: float
    max_dd: float
    profit_factor: float


@dataclass
class StrategySummary:
    strategy_id: str
    folds: int
    total_trades: int
    sum_pnl: float
    median_pnl: float
    positive_folds: int
    median_win_rate: float
    median_expectancy_r: float


def _bars_per_day(tf: Timeframe) -> int:
    bar_minutes = tf.value if isinstance(tf.value, int) else 15
    return max(1, 1440 // bar_minutes)


def slice_last_days(df: pd.DataFrame, days: int, tf: Timeframe) -> pd.DataFrame:
    """Keep trailing `days` calendar days of bars."""
    need = days * _bars_per_day(tf)
    if len(df) <= need:
        return df.copy()
    return df.iloc[-need:].copy()


def iter_rolling_folds(
    df: pd.DataFrame,
    test_days: int,
    step_days: int,
    tf: Timeframe,
    warmup_days: int = WARMUP_DAYS_DEFAULT,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """
    Yield (fold_index, context_df, test_df).

    context = warmup + test (for indicator lookback)
    test = OOS window only (passed to backtest)
    """
    bars_per_day = _bars_per_day(tf)
    test_bars = test_days * bars_per_day
    step_bars = max(1, step_days * bars_per_day)
    warmup_bars = warmup_days * bars_per_day
    min_start = warmup_bars

    fold = 0
    start = min_start
    while start + test_bars <= len(df):
        fold += 1
        ctx_start = start - warmup_bars
        ctx_end = start + test_bars
        context = df.iloc[ctx_start:ctx_end].copy()
        test = df.iloc[start:ctx_end].copy()
        yield fold, context, test
        start += step_bars


def build_frozen_features(
    strategy_id: str,
    df: pd.DataFrame,
    tf: Timeframe,
) -> tuple[pd.DataFrame, BacktestParams]:
    """Build features using frozen configs from strategies_optimized/."""
    if strategy_id == "depth_charge":
        from strategy.depth_charge import build_features, load_depth_params

        dp, bt = load_depth_params()
        return build_features(df, dp, tf, bt), bt

    if strategy_id == "fusion_v2":
        from strategy.fusion_v2 import build_features, load_fusion_params

        fp, bt = load_fusion_params()
        return build_features(df, fp, tf, bt), bt

    if strategy_id in FROZEN_STRATEGIES:
        raise ValueError(f"Frozen builder missing for {strategy_id}")

    from backtest.strategy_ga import individual_to_features
    from exec.paper_trader import load_optimized_config, config_to_individual

    try:
        cfg = load_optimized_config(strategy_id, tf)
        ind = config_to_individual(cfg)
        feats = individual_to_features(strategy_id, ind, df, tf)
        return feats, ind.backtest_params
    except FileNotFoundError:
        from strategy.registry import build

        bt = BacktestParams()
        feats = build(strategy_id, df, tf, bt)
        return feats, bt


def evaluate_fold(
    strategy_id: str,
    context: pd.DataFrame,
    test: pd.DataFrame,
    tf: Timeframe,
    fold: int,
) -> FoldResult:
    feats, bt = build_frozen_features(strategy_id, context, tf)
    feats = feats.loc[test.index.intersection(feats.index)]
    result = run_backtest(feats, bt)
    m = result["metrics"]
    account = simulate_account(result["trades"])

    return FoldResult(
        fold=fold,
        strategy_id=strategy_id,
        test_start=str(test.index[0]),
        test_end=str(test.index[-1]),
        bars=len(test),
        trades=int(m.get("n", 0)),
        total_pnl=float(m.get("total_pnl", 0.0)),
        win_rate=float(m.get("win_rate", 0.0)),
        expectancy_r=float(m.get("expectancy_r", account.get("expectancy_r", 0.0))),
        max_dd=float(m.get("max_dd", 0.0)),
        profit_factor=float(m.get("profit_factor", 0.0)),
    )


def walk_forward_report(
    df: pd.DataFrame,
    strategy_ids: list[str],
    tf: Timeframe,
    test_days: int = 60,
    step_days: int = 60,
    warmup_days: int = WARMUP_DAYS_DEFAULT,
) -> dict[str, Any]:
    """Run rolling walk-forward for each strategy; return fold rows + summaries."""
    all_folds: list[FoldResult] = []

    for sid in strategy_ids:
        for fold, context, test in iter_rolling_folds(df, test_days, step_days, tf, warmup_days):
            all_folds.append(evaluate_fold(sid, context, test, tf, fold))

    summaries: dict[str, StrategySummary] = {}
    for sid in strategy_ids:
        rows = [f for f in all_folds if f.strategy_id == sid and f.trades >= 0]
        if not rows:
            continue
        pnls = [r.total_pnl for r in rows]
        summaries[sid] = StrategySummary(
            strategy_id=sid,
            folds=len(rows),
            total_trades=sum(r.trades for r in rows),
            sum_pnl=float(sum(pnls)),
            median_pnl=float(np.median(pnls)),
            positive_folds=sum(1 for p in pnls if p > 0),
            median_win_rate=float(np.median([r.win_rate for r in rows])),
            median_expectancy_r=float(np.median([r.expectancy_r for r in rows])),
        )

    return {
        "folds": [asdict(f) for f in all_folds],
        "summaries": {k: asdict(v) for k, v in summaries.items()},
        "params": {
            "test_days": test_days,
            "step_days": step_days,
            "warmup_days": warmup_days,
            "timeframe": tf.value,
            "bars": len(df),
            "range": f"{df.index[0]} → {df.index[-1]}",
        },
    }
