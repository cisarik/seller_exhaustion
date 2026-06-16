"""Tests for execution validation layer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.validation import (
    analyze_streaks,
    bootstrap_loop_pnl,
    build_validation_report,
    evaluate_kill_switch,
    run_cost_stress,
    StreakAnalysis,
    CostStressRow,
)
from core.models import Timeframe


def _rec(pnl: float, trades: int = 2, idx: int = 1) -> dict:
    return {
        "loop_index": idx,
        "strategy_id": "fusion_v2",
        "n_trades": trades,
        "metrics": {"total_pnl": pnl, "expectancy_r": 0.1},
    }


def test_analyze_streaks_consecutive_losses():
    records = [_rec(0.01, idx=1), _rec(-0.02, idx=2), _rec(-0.01, idx=3), _rec(-0.03, idx=4)]
    s = analyze_streaks(records)
    assert s.max_consecutive_losses >= 3


def test_bootstrap_prob_positive():
    pnls = [0.02, 0.01, -0.005, 0.03, 0.01]
    b = bootstrap_loop_pnl(pnls, n_samples=500, seed=1)
    assert b.prob_positive > 0.5


def test_kill_switch_pause_on_2x_costs():
    streaks = StreakAnalysis(1, 2, 0.05, 3, False)
    stress = [
        CostStressRow(1.0, 5, 5, 0.02, 5, 0.6, True),
        CostStressRow(2.0, 10, 10, -0.01, 5, 0.4, False),
    ]
    state, reasons = evaluate_kill_switch(streaks, stress, 0.6)
    assert state == "PAUSE"
    assert any("2×" in r for r in reasons)


def test_build_validation_report_minimal():
    records = [_rec(0.01, idx=i) for i in range(1, 6)]
    report = build_validation_report("fusion_v2", Timeframe.m15, records, df=None, loop_stats={})
    assert report.execution_verdict in ("READY", "CAUTION", "BLOCKED")
    assert "verdict_level" in report.loop_stats
