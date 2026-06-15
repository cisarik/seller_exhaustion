"""Common strategy interface for multi-strategy backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from core.models import BacktestParams, Timeframe

# Canonical entry signal column (engine reads this, falls back to exhaustion)
SIGNAL_COL = "signal"


@dataclass
class StrategyMeta:
    name: str
    id: str
    description: str
    style: str  # deterministic | stochastic | hybrid


class Strategy(Protocol):
    meta: StrategyMeta

    def build_features(
        self,
        df: pd.DataFrame,
        tf: Timeframe,
        bt_params: BacktestParams | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV + indicators + signal column."""
        ...


def finalize_signals(out: pd.DataFrame, raw_col: str = "raw_signal") -> pd.DataFrame:
    """Copy raw signal into canonical columns for the backtest engine."""
    if raw_col in out.columns:
        out[SIGNAL_COL] = out[raw_col].fillna(False).astype(bool)
        out["exhaustion"] = out[SIGNAL_COL]  # backward compatibility
    return out
