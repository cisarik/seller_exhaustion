"""Compare multiple strategies on the same dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtest.engine import run_backtest
from backtest.profit import simulate_account, profit_score, walk_forward_splits
from core.models import BacktestParams, Timeframe
from strategy.base import SIGNAL_COL
from strategy.registry import list_strategies, build


@dataclass
class StrategyResult:
    strategy_id: str
    name: str
    signals: int
    trades: int
    win_rate: float
    total_pnl: float
    expectancy_r: float
    cagr_pct: float
    profit_score: float
    max_dd: float


def run_single(
    df: pd.DataFrame,
    strategy_id: str,
    tf: Timeframe,
    bt: BacktestParams | None = None,
) -> StrategyResult:
    bt = bt or BacktestParams()
    feats = build(strategy_id, df, tf, bt)
    result = run_backtest(feats, bt)
    m = result["metrics"]
    account = simulate_account(result["trades"])
    signals = int(feats[SIGNAL_COL].sum()) if SIGNAL_COL in feats else 0

    meta = next(s for s in list_strategies() if s.id == strategy_id)
    return StrategyResult(
        strategy_id=strategy_id,
        name=meta.name,
        signals=signals,
        trades=m.get("n", 0),
        win_rate=m.get("win_rate", 0.0),
        total_pnl=m.get("total_pnl", 0.0),
        expectancy_r=m.get("expectancy_r", account["expectancy_r"]),
        cagr_pct=m.get("cagr_pct", account["cagr_pct"]),
        profit_score=profit_score(m, account),
        max_dd=m.get("max_dd", 0.0),
    )


def compare_all(
    df: pd.DataFrame,
    tf: Timeframe,
    bt: BacktestParams | None = None,
    strategy_ids: list[str] | None = None,
) -> pd.DataFrame:
    ids = strategy_ids or [m.id for m in list_strategies()]
    rows = [run_single(df, sid, tf, bt).__dict__ for sid in ids]
    out = pd.DataFrame(rows)
    return out.sort_values("profit_score", ascending=False).reset_index(drop=True)


def walk_forward_compare(
    df: pd.DataFrame,
    strategy_id: str,
    tf: Timeframe,
    bt: BacktestParams | None = None,
    n_folds: int = 3,
) -> list[dict[str, Any]]:
    bt = bt or BacktestParams()
    results = []
    for i, fold in enumerate(walk_forward_splits(df, n_folds)):
        r = run_single(fold, strategy_id, tf, bt)
        results.append({"fold": i + 1, **r.__dict__})
    return results
