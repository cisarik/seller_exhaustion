"""
Execution-aware validation — fee stress, streaks, bootstrap, kill-switch rules.

Complements paper-top-stats (loop stability) with questions that matter for live:
  - Does edge survive 2× fees/slippage?
  - How bad are losing streaks?
  - Is recent performance degrading?
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backtest.engine import run_backtest
from backtest.walk_forward import build_frozen_features, slice_last_days
from backtest.paper_stats import record_pnl
from core.models import BacktestParams, Timeframe


@dataclass
class StreakAnalysis:
    max_consecutive_losses: int
    max_consecutive_wins: int
    recent_n_sum_pnl: float
    recent_n: int
    degrading: bool  # recent window worse than median


@dataclass
class CostStressRow:
    multiplier: float
    fee_bp: float
    slippage_bp: float
    total_pnl: float
    n_trades: int
    win_rate: float
    profitable: bool


@dataclass
class BootstrapResult:
    n_samples: int
    pnl_mean: float
    pnl_p5: float
    pnl_p95: float
    prob_positive: float


@dataclass
class ValidationReport:
    strategy_id: str
    timeframe: str
    loop_stats: dict[str, Any]
    streaks: StreakAnalysis
    cost_stress: list[CostStressRow]
    bootstrap: BootstrapResult | None
    kill_switch: str  # OK | CAUTION | PAUSE | RETUNE
    kill_reasons: list[str] = field(default_factory=list)
    execution_verdict: str = "UNKNOWN"  # READY | CAUTION | BLOCKED

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "timeframe": self.timeframe,
            "loop_stats": self.loop_stats,
            "streaks": {
                "max_consecutive_losses": self.streaks.max_consecutive_losses,
                "max_consecutive_wins": self.streaks.max_consecutive_wins,
                "recent_n_sum_pnl": self.streaks.recent_n_sum_pnl,
                "recent_n": self.streaks.recent_n,
                "degrading": self.streaks.degrading,
            },
            "cost_stress": [
                {
                    "multiplier": r.multiplier,
                    "fee_bp": r.fee_bp,
                    "slippage_bp": r.slippage_bp,
                    "total_pnl": r.total_pnl,
                    "n_trades": r.n_trades,
                    "win_rate": r.win_rate,
                    "profitable": r.profitable,
                }
                for r in self.cost_stress
            ],
            "bootstrap": (
                {
                    "n_samples": self.bootstrap.n_samples,
                    "pnl_mean": self.bootstrap.pnl_mean,
                    "pnl_p5": self.bootstrap.pnl_p5,
                    "pnl_p95": self.bootstrap.pnl_p95,
                    "prob_positive": self.bootstrap.prob_positive,
                }
                if self.bootstrap
                else None
            ),
            "kill_switch": self.kill_switch,
            "kill_reasons": self.kill_reasons,
            "execution_verdict": self.execution_verdict,
        }


def _max_streak(values: list[bool], target: bool) -> int:
    best = cur = 0
    for v in values:
        if v == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def analyze_streaks(records: list[dict[str, Any]], recent_n: int = 3) -> StreakAnalysis:
    """Streak analysis on ordered loop records (oldest → newest if sorted by loop_index)."""
    ordered = sorted(records, key=lambda r: r.get("loop_index", r.get("offset_days", 0)))
    pnls = [record_pnl(r) for r in ordered]
    active_pnls = [record_pnl(r) for r in ordered if int(r.get("n_trades", 0)) > 0]

    losses = [p < 0 for p in active_pnls] if active_pnls else [p < 0 for p in pnls]
    wins = [not x for x in losses]

    recent = pnls[-recent_n:] if pnls else []
    recent_sum = sum(recent)
    median_all = statistics.median(pnls) if pnls else 0.0
    degrading = recent_sum < median_all * recent_n if pnls else False

    return StreakAnalysis(
        max_consecutive_losses=_max_streak(losses, True),
        max_consecutive_wins=_max_streak(wins, True),
        recent_n_sum_pnl=recent_sum,
        recent_n=recent_n,
        degrading=degrading,
    )


def bootstrap_loop_pnl(
    pnls: list[float],
    n_samples: int = 2000,
    seed: int = 42,
) -> BootstrapResult:
    """Resample loop PnLs with replacement to estimate uncertainty."""
    if not pnls:
        return BootstrapResult(0, 0.0, 0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(pnls)
    totals = [sum(rng.choice(pnls) for _ in range(n)) for _ in range(n_samples)]
    totals.sort()
    p5 = totals[int(0.05 * n_samples)]
    p95 = totals[int(0.95 * n_samples)]
    return BootstrapResult(
        n_samples=n_samples,
        pnl_mean=statistics.mean(totals),
        pnl_p5=p5,
        pnl_p95=p95,
        prob_positive=sum(1 for t in totals if t > 0) / n_samples,
    )


def run_cost_stress(
    strategy_id: str,
    df: pd.DataFrame,
    tf: Timeframe,
    forward_days: int = 60,
    multipliers: list[float] | None = None,
) -> list[CostStressRow]:
    """
    Re-run forward window with scaled fee_bp + slippage_bp.

    Uses same slice logic as paper_trader (warmup + forward).
    """
    from exec.paper_trader import _forward_slice

    multipliers = multipliers or [1.0, 1.5, 2.0, 3.0]
    context, forward = _forward_slice(df, forward_days, tf)
    feats, bt = build_frozen_features(strategy_id, context, tf)
    feats = feats.loc[forward.index.intersection(feats.index)]

    rows: list[CostStressRow] = []
    for mult in multipliers:
        stressed = bt.model_copy(
            update={
                "fee_bp": bt.fee_bp * mult,
                "slippage_bp": bt.slippage_bp * mult,
            }
        )
        result = run_backtest(feats, stressed)
        m = result["metrics"]
        rows.append(
            CostStressRow(
                multiplier=mult,
                fee_bp=stressed.fee_bp,
                slippage_bp=stressed.slippage_bp,
                total_pnl=float(m.get("total_pnl", 0.0)),
                n_trades=int(m.get("n", 0)),
                win_rate=float(m.get("win_rate", 0.0)),
                profitable=float(m.get("total_pnl", 0.0)) > 0,
            )
        )
    return rows


def evaluate_kill_switch(
    streaks: StreakAnalysis,
    cost_stress: list[CostStressRow],
    loop_positive_pct: float,
    max_loss_streak: int = 3,
) -> tuple[str, list[str]]:
    """
    Execution kill-switch state from validation metrics.

    Returns (state, reasons).
    """
    reasons: list[str] = []
    state = "OK"

    stress_2x = next((r for r in cost_stress if r.multiplier == 2.0), None)
    if stress_2x and not stress_2x.profitable:
        state = "PAUSE"
        reasons.append(f"2× costs unprofitable (PnL {stress_2x.total_pnl:+.4f})")

    if streaks.max_consecutive_losses >= max_loss_streak:
        if state != "PAUSE":
            state = "CAUTION"
        reasons.append(f"{streaks.max_consecutive_losses} consecutive losing loops")

    if streaks.recent_n_sum_pnl < 0:
        state = "RETUNE" if state == "OK" else state
        reasons.append(f"Last {streaks.recent_n} loops sum PnL {streaks.recent_n_sum_pnl:+.4f}")

    if streaks.degrading and loop_positive_pct < 0.55:
        if state == "OK":
            state = "CAUTION"
        reasons.append("Recent loops below historical median")

    if not reasons:
        reasons.append("All execution checks nominal")

    return state, reasons


def execution_verdict(kill_switch: str, stress_2x_profitable: bool) -> str:
    if kill_switch in ("PAUSE", "RETUNE"):
        return "BLOCKED"
    if kill_switch == "CAUTION" or not stress_2x_profitable:
        return "CAUTION"
    return "READY"


def build_validation_report(
    strategy_id: str,
    tf: Timeframe,
    records: list[dict[str, Any]],
    df: pd.DataFrame | None,
    loop_stats: dict[str, Any],
    forward_days: int = 60,
) -> ValidationReport:
    from backtest.paper_stats import analyze_paper_runs

    stats = analyze_paper_runs(records)
    streaks = analyze_streaks(records)
    pnls = [record_pnl(r) for r in records]
    bootstrap = bootstrap_loop_pnl(pnls) if len(pnls) >= 4 else None

    cost_stress: list[CostStressRow] = []
    if df is not None and len(df) > 100:
        try:
            cost_stress = run_cost_stress(strategy_id, df, tf, forward_days=forward_days)
        except Exception:
            cost_stress = []

    kill, reasons = evaluate_kill_switch(streaks, cost_stress, stats.positive_pct)
    stress_2x = next((r for r in cost_stress if r.multiplier == 2.0), None)
    exec_verdict = execution_verdict(kill, stress_2x.profitable if stress_2x else True)

    return ValidationReport(
        strategy_id=strategy_id,
        timeframe=tf.value,
        loop_stats={
            "verdict_level": stats.verdict_level,
            "positive_pct": stats.positive_pct,
            "sum_pnl": stats.sum_pnl,
            "median_exp_active": stats.median_exp_active,
            "passed_checks": stats.passed,
            "total_checks": stats.total_checks,
            **loop_stats,
        },
        streaks=streaks,
        cost_stress=cost_stress,
        bootstrap=bootstrap,
        kill_switch=kill,
        kill_reasons=reasons,
        execution_verdict=exec_verdict,
    )
