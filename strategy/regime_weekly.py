"""
Weekly LLM regime filter — classifies market once per week, cached on disk.

Used only as a gate (not signal generation). Falls back to deterministic
regime scoring when OPENAI_API_KEY is missing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core.models import Timeframe
from strategy.regime import classify_regime_llm, compute_regime

CACHE_PATH = Path(".data/regime_weekly.json")
REGIME_MODEL = os.getenv("REGIME_LLM_MODEL", "gpt-4o-mini")


def _week_key(ts: pd.Timestamp) -> str:
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def summarize_week(df: pd.DataFrame) -> str:
    """Compact market summary for LLM regime classification."""
    if len(df) < 10:
        return "Insufficient data"
    ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    vol = df["volume"].mean()
    hi, lo = df["high"].max(), df["low"].min()
    return (
        f"Asset ADAUSD weekly summary | bars={len(df)} | "
        f"return={ret:+.2f}% | range={lo:.4f}-{hi:.4f} | avg_volume={vol:.0f} | "
        f"last_close={df['close'].iloc[-1]:.4f}"
    )


def update_weekly_regime(
    df: pd.DataFrame,
    tf: Timeframe = Timeframe.m15,
    force: bool = False,
) -> dict[str, Any]:
    """
    Update weekly regime for the latest complete week in df.

    Returns cache entry: {week, score, label, reason, min_regime_score, updated_at}
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()

    week_end = df.index[-1]
    week_start = week_end - timedelta(days=7)
    week_df = df[df.index >= week_start]
    key = _week_key(week_end)

    if not force and key in cache:
        return cache[key]

    summary = summarize_week(week_df)
    feats = compute_regime(week_df, tf)
    det_score = float(feats["regime_score"].iloc[-1]) if len(feats) else 0.5
    det_label = str(feats["regime_label"].iloc[-1]) if len(feats) else "neutral"

    llm = classify_regime_llm(summary)
    llm_score = float(llm.get("score", det_score))
    llm_label = str(llm.get("label", det_label))
    reason = str(llm.get("reason", "deterministic fallback"))

    # Blend LLM + deterministic when API available
    has_llm = "LLM unavailable" not in reason and "LLM error" not in reason
    score = 0.6 * llm_score + 0.4 * det_score if has_llm else det_score
    label = llm_label if has_llm else det_label

    min_regime = 0.55 if label == "oversold_bounce" else 0.50 if label == "neutral" else 0.65

    entry = {
        "week": key,
        "score": round(score, 4),
        "label": label,
        "reason": reason,
        "min_regime_score": min_regime,
        "deterministic_score": round(det_score, 4),
        "llm_score": round(llm_score, 4) if has_llm else None,
        "model": REGIME_MODEL if has_llm else "deterministic",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }
    cache[key] = entry
    _save_cache(cache)
    return entry


def get_current_regime_gate() -> dict[str, Any]:
    """Return latest cached weekly regime or neutral default."""
    cache = _load_cache()
    if not cache:
        return {
            "week": None,
            "score": 0.5,
            "label": "neutral",
            "min_regime_score": 0.50,
            "reason": "no cache — run regime-update",
        }
    latest = sorted(cache.keys())[-1]
    return cache[latest]


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open() as f:
        return json.load(f)


def _save_cache(cache: dict) -> None:
    with CACHE_PATH.open("w") as f:
        json.dump(cache, f, indent=2)
