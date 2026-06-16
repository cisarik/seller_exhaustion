"""Aggregate stats and go/no-go verdict for rolling paper-forward runs."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaperStatsThresholds:
    min_positive_pct: float = 0.45
    min_expectancy: float = 0.05
    min_trades_per_loop: float = 0.5


@dataclass
class PaperStatsResult:
    strategy_id: str
    n_loops: int
    n_active: int
    n_positive: int
    positive_pct: float
    sum_pnl: float
    median_pnl: float
    median_exp: float
    median_exp_active: float
    avg_trades: float
    loop_sharpe: float
    active_sum_pnl: float
    checks: dict[str, bool] = field(default_factory=dict)
    passed: int = 0
    total_checks: int = 0
    verdict_level: str = "NO-GO"  # GO | MARGINAL | NO-GO


def record_expectancy(record: dict[str, Any]) -> float:
    m = record.get("metrics", {})
    a = record.get("account", {})
    return float(m.get("expectancy_r", a.get("expectancy_r", 0.0)))


def record_pnl(record: dict[str, Any]) -> float:
    return float(record.get("metrics", {}).get("total_pnl", 0.0))


def analyze_paper_runs(
    records: list[dict[str, Any]],
    thresholds: PaperStatsThresholds | None = None,
) -> PaperStatsResult:
    """Compute stability metrics and go/no-go checks from loop/monitor records."""
    if not records:
        raise ValueError("No records to analyze")

    th = thresholds or PaperStatsThresholds()
    strategy_id = str(records[0].get("strategy_id", "?"))
    n_loops = len(records)

    pnls = [record_pnl(r) for r in records]
    exps = [record_expectancy(r) for r in records]
    trades = [int(r.get("n_trades", 0)) for r in records]

    active = [r for r in records if int(r.get("n_trades", 0)) > 0]
    n_active = len(active)
    n_positive = sum(1 for p in pnls if p > 0)
    positive_pct = n_positive / n_loops

    active_exp = [record_expectancy(r) for r in active]
    median_exp_active = statistics.median(active_exp) if active_exp else 0.0
    # Verdict uses active-loop expectancy when enough samples exist
    verdict_exp = median_exp_active if n_active >= 3 else statistics.median(exps)

    pnl_std = statistics.stdev(pnls) if n_loops > 1 else 0.0
    loop_sharpe = (statistics.mean(pnls) / pnl_std) if pnl_std > 0 else 0.0

    checks = {
        f"Positive loops ≥ {th.min_positive_pct:.0%}": positive_pct >= th.min_positive_pct,
        f"Median expectancy ≥ {th.min_expectancy:.2f}": verdict_exp >= th.min_expectancy,
        f"Avg trades/loop ≥ {th.min_trades_per_loop:.1f}": statistics.mean(trades) >= th.min_trades_per_loop,
        "Sum PnL > 0": sum(pnls) > 0,
        "Loop Sharpe > 0": loop_sharpe > 0,
    }
    passed = sum(checks.values())
    total_checks = len(checks)

    if passed == total_checks:
        verdict_level = "GO"
    elif passed >= math.ceil(total_checks * 0.6):
        verdict_level = "MARGINAL"
    else:
        verdict_level = "NO-GO"

    return PaperStatsResult(
        strategy_id=strategy_id,
        n_loops=n_loops,
        n_active=n_active,
        n_positive=n_positive,
        positive_pct=positive_pct,
        sum_pnl=sum(pnls),
        median_pnl=statistics.median(pnls),
        median_exp=statistics.median(exps),
        median_exp_active=median_exp_active,
        avg_trades=statistics.mean(trades),
        loop_sharpe=loop_sharpe,
        active_sum_pnl=sum(record_pnl(r) for r in active),
        checks=checks,
        passed=passed,
        total_checks=total_checks,
        verdict_level=verdict_level,
    )


def verdict_message(result: PaperStatsResult) -> str:
    if result.verdict_level == "GO":
        return "✅  GO — candidate is stable, consider paper trading"
    if result.verdict_level == "MARGINAL":
        return (
            f"⚠  MARGINAL ({result.passed}/{result.total_checks}) — "
            f"paper trading with caution, monitor closely"
        )
    return (
        f"🛑  NO-GO ({result.passed}/{result.total_checks}) — "
        f"candidate lacks edge stability"
    )
