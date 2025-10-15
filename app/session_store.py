"""
Session persistence helpers for restoring the last backtest on application restart.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.models import BacktestParams, Timeframe
from strategy.seller_exhaustion import SellerParams
from config.settings import settings

SESSION_FILE = Path(settings.data_dir) / "last_session.json"


def _to_builtin(value: Any) -> Any:
    """Convert numpy/pandas scalars to native Python types for JSON serialization."""
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


def save_session_snapshot(
    *,
    ticker: str,
    date_from: str,
    date_to: str,
    timeframe: Timeframe,
    multiplier: int,
    timespan: str,
    seller_params: SellerParams,
    backtest_params: BacktestParams,
    backtest_result: Optional[dict],
) -> None:
    """
    Persist the latest backtest snapshot so it can be restored on restart.
    """
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)

    snapshot: dict[str, Any] = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "date_from": date_from,
        "date_to": date_to,
        "timeframe": timeframe.value,
        "multiplier": multiplier,
        "timespan": timespan,
        "seller_params": _to_builtin(asdict(seller_params)),
        "backtest_params": _to_builtin(backtest_params.model_dump()),
    }

    if backtest_result:
        metrics = backtest_result.get("metrics", {})
        trades_df = backtest_result.get("trades")
        trades_records: list[dict[str, Any]] = []

        if isinstance(trades_df, pd.DataFrame) and len(trades_df) > 0:
            trades_records = [
                _to_builtin({k: v for k, v in row.items()})
                for row in trades_df.to_dict(orient="records")
            ]

        snapshot["backtest_result"] = {
            "metrics": _to_builtin(metrics),
            "trades": trades_records,
        }

    with SESSION_FILE.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def load_session_snapshot() -> Optional[dict[str, Any]]:
    """Load previously saved snapshot if it exists."""
    if not SESSION_FILE.exists():
        return None

    with SESSION_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        tf = Timeframe(data["timeframe"])
    except Exception:
        tf = Timeframe.m15

    seller_params = SellerParams(**data.get("seller_params", {}))
    backtest_params = BacktestParams(**data.get("backtest_params", {}))

    backtest_result = None
    raw_result = data.get("backtest_result")
    if isinstance(raw_result, dict):
        trades_records = raw_result.get("trades") or []
        trades_df = pd.DataFrame(trades_records)
        if not trades_df.empty:
            numeric_cols = ["entry", "exit", "stop", "tp", "pnl", "R"]
            for col in numeric_cols:
                if col in trades_df.columns:
                    trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")
            if "bars_held" in trades_df.columns:
                trades_df["bars_held"] = pd.to_numeric(trades_df["bars_held"], errors="coerce").astype("Int64")

        metrics = raw_result.get("metrics") or {}
        metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}

        backtest_result = {
            "trades": trades_df,
            "metrics": metrics,
        }

    return {
        "ticker": data.get("ticker", "X:ADAUSD"),
        "date_from": data.get("date_from", "2024-01-01"),
        "date_to": data.get("date_to", "2024-12-31"),
        "timeframe": tf,
        "multiplier": int(data.get("multiplier", 15)),
        "timespan": data.get("timespan", "minute"),
        "seller_params": seller_params,
        "backtest_params": backtest_params,
        "backtest_result": backtest_result,
    }


def clear_session_snapshot() -> None:
    """Remove persisted snapshot."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink(missing_ok=True)
