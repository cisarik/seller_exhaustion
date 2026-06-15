"""
Market regime scoring — deterministic proxy for LLM regime classification.

Scores 0..1 how favorable conditions are for mean-reversion longs in crypto.
Optional LLM enhancement via classify_regime_llm() when OPENAI_API_KEY is set.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from indicators.local import ema, atr, rsi
from core.models import Timeframe, minutes_to_bars


def compute_regime(df: pd.DataFrame, tf: Timeframe = Timeframe.m60) -> pd.DataFrame:
    """
    Add regime columns:
    - regime_score: 0..1 favorability for bounce longs
    - regime_label: oversold_bounce | neutral | avoid
    """
    out = df.copy()
    fast = 24 if tf == Timeframe.m60 else max(12, minutes_to_bars(1440, tf))
    slow = 168 if tf == Timeframe.m60 else max(48, minutes_to_bars(10080, tf))

    if "ema_f" not in out.columns:
        out["ema_f"] = ema(out["close"], fast)
    if "ema_s" not in out.columns:
        out["ema_s"] = ema(out["close"], slow)
    if "atr" not in out.columns:
        out["atr"] = atr(out["high"], out["low"], out["close"], fast)
    if "rsi" not in out.columns:
        out["rsi"] = rsi(out["close"], 14)

    downtrend = (out["ema_f"] < out["ema_s"]).astype(float)
    rsi_score = np.clip((45 - out["rsi"]) / 25.0, 0, 1)
    atr_rank = out["atr"] / out["atr"].rolling(168, min_periods=48).median()
    vol_score = np.clip((atr_rank - 0.8) / 1.2, 0, 1)

    trend_depth = (out["ema_s"] - out["ema_f"]) / out["ema_s"]
    depth_score = np.clip(trend_depth / 0.08, 0, 1)

    out["regime_score"] = (
        0.35 * downtrend +
        0.30 * rsi_score +
        0.20 * vol_score +
        0.15 * depth_score
    ).clip(0, 1)

    out["regime_label"] = pd.cut(
        out["regime_score"],
        bins=[-0.01, 0.35, 0.55, 1.01],
        labels=["avoid", "neutral", "oversold_bounce"],
    ).astype(str)

    return out


def apply_regime_gate(signals: pd.Series, regime_score: pd.Series, min_score: float = 0.45) -> pd.Series:
    return signals & (regime_score >= min_score)


def classify_regime_llm(summary: str, api_key: Optional[str] = None) -> dict:
    """
    Optional LLM regime classifier (uses OpenAI when key available).

    Returns {"score": float, "label": str, "reason": str}.
    Falls back to rule parsing when no API key.
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return {"score": 0.5, "label": "neutral", "reason": "LLM unavailable — using deterministic gate"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=os.getenv("REGIME_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify crypto market regime for mean-reversion long entries. "
                        "Reply JSON only: {\"score\":0-1,\"label\":\"oversold_bounce|neutral|avoid\",\"reason\":\"...\"}"
                    ),
                },
                {"role": "user", "content": summary},
            ],
            temperature=0.1,
            max_tokens=120,
        )
        import json
        text = resp.choices[0].message.content or "{}"
        return json.loads(text.strip())
    except Exception as e:
        return {"score": 0.5, "label": "neutral", "reason": f"LLM error: {e}"}
