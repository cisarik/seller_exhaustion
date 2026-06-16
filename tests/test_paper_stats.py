"""Tests for paper-forward stability stats."""

from backtest.paper_stats import (
    PaperStatsThresholds,
    analyze_paper_runs,
    verdict_message,
)


def _rec(pnl: float, exp: float = 0.1, trades: int = 2) -> dict:
    return {
        "strategy_id": "depth_charge",
        "n_trades": trades,
        "metrics": {"total_pnl": pnl, "expectancy_r": exp, "win_rate": 0.5},
        "account": {},
    }


def test_analyze_go_verdict():
    records = [_rec(0.02, 0.2), _rec(0.01, 0.15), _rec(-0.005, 0.08), _rec(0.03, 0.25)]
    result = analyze_paper_runs(records, PaperStatsThresholds(min_positive_pct=0.5))
    assert result.n_positive == 3
    assert result.sum_pnl > 0
    assert result.verdict_level in ("GO", "MARGINAL")


def test_analyze_no_go_verdict():
    records = [_rec(-0.02, -0.3), _rec(-0.01, -0.2), _rec(0.0, 0.0, trades=0)]
    result = analyze_paper_runs(records)
    assert result.verdict_level == "NO-GO"
    assert "NO-GO" in verdict_message(result)
