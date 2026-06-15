"""
Account-level profit simulation from R-multiple trades.

Converts per-trade R results into compounded equity curve using fixed fractional risk.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def simulate_account(
    trades: pd.DataFrame,
    start_capital: float = 10_000.0,
    risk_per_trade_pct: float = 1.0,
) -> dict[str, Any]:
    """
    Simulate account growth using fixed fractional position sizing.

    Each trade risks `risk_per_trade_pct` percent of current equity.
    Dollar PnL = equity * (risk_pct/100) * R_multiple.

    Returns equity curve stats including CAGR and max drawdown in dollars.
    """
    if trades is None or len(trades) == 0:
        return {
            "start_capital": start_capital,
            "end_capital": start_capital,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "max_account_dd_pct": 0.0,
            "expectancy_r": 0.0,
            "equity_curve": [start_capital],
        }

    equity = start_capital
    curve = [equity]
    peak = equity
    max_dd_pct = 0.0

    for r_mult in trades["R"].astype(float):
        risk_dollars = equity * (risk_per_trade_pct / 100.0)
        equity += risk_dollars * r_mult
        equity = max(equity, 0.01)  # floor to avoid negative equity blow-up in sim
        curve.append(equity)
        peak = max(peak, equity)
        dd_pct = (equity - peak) / peak * 100.0 if peak > 0 else 0.0
        max_dd_pct = min(max_dd_pct, dd_pct)

    n_days = _trade_span_days(trades)
    years = max(n_days / 365.25, 1 / 365.25)
    total_return_pct = (equity / start_capital - 1.0) * 100.0
    cagr_pct = ((equity / start_capital) ** (1.0 / years) - 1.0) * 100.0 if equity > 0 else -100.0

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / len(trades)
    avg_win_r = float(wins["R"].mean()) if len(wins) else 0.0
    avg_loss_r = float(losses["R"].mean()) if len(losses) else 0.0
    expectancy_r = win_rate * avg_win_r + (1 - win_rate) * avg_loss_r

    return {
        "start_capital": start_capital,
        "end_capital": round(equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "max_account_dd_pct": round(max_dd_pct, 2),
        "expectancy_r": round(expectancy_r, 4),
        "equity_curve": curve,
    }


def _trade_span_days(trades: pd.DataFrame) -> int:
    if "entry_ts" not in trades.columns or "exit_ts" not in trades.columns:
        return 365
    try:
        entry = pd.to_datetime(trades["entry_ts"], utc=True)
        exit_ = pd.to_datetime(trades["exit_ts"], utc=True)
        return max(int((exit_.max() - entry.min()).total_seconds() // 86400), 1)
    except Exception:
        return 365


def walk_forward_splits(df: pd.DataFrame, n_folds: int = 3) -> list[pd.DataFrame]:
    """Split time series into contiguous walk-forward folds."""
    n_folds = max(2, n_folds)
    size = len(df) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * size
        end = (i + 1) * size if i < n_folds - 1 else len(df)
        folds.append(df.iloc[start:end].copy())
    return folds


def profit_score(metrics: dict[str, Any], account: dict[str, Any]) -> float:
    """
    Single profit-oriented score for comparing configurations.

    Combines expectancy, CAGR, trade count significance, and drawdown penalty.
    """
    n = metrics.get("n", 0)
    if n < 3:
        return -1000.0

    expectancy = account.get("expectancy_r", 0.0)
    cagr = account.get("cagr_pct", 0.0)
    dd = abs(account.get("max_account_dd_pct", 0.0))
    significance = min(math.sqrt(n / 30.0), 1.0)  # ~30 trades for full confidence

    raw = (
        0.40 * np.tanh(expectancy / 0.5) +
        0.35 * np.tanh(cagr / 15.0) +
        0.15 * significance +
        0.10 * np.tanh(metrics.get("profit_factor", 0.0) / 2.0)
    )
    dd_penalty = min(dd / 25.0, 1.0) * 0.25
    return float(raw - dd_penalty)
