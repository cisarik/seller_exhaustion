"""Paper trading execution layer (simulation + forward test)."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engine import run_backtest
from backtest.profit import simulate_account
from backtest.strategy_ga import individual_to_features
from backtest.optimizer import Individual
from core.models import BacktestParams, Timeframe
from strategy.seller_exhaustion import SellerParams
from strategy.regime_weekly import get_current_regime_gate

PAPER_LOG = Path(".data/paper_trades.jsonl")
CONFIG_DIR = Path("strategies_optimized")
WARMUP_DAYS = 14

FROZEN_STRATEGIES = frozenset({"fusion_v2", "depth_charge"})


@dataclass
class OptimizedStrategyConfig:
    strategy_id: str
    timeframe: str
    seller_params: dict
    backtest_params: dict
    train_metrics: dict
    oos_metrics: dict
    optimized_at: str


def save_optimized_config(
    strategy_id: str,
    individual: Individual,
    tf: Timeframe,
    train_metrics: dict,
    oos_metrics: dict,
) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = OptimizedStrategyConfig(
        strategy_id=strategy_id,
        timeframe=tf.value,
        seller_params=individual.seller_params.__dict__,
        backtest_params=individual.backtest_params.model_dump(),
        train_metrics=train_metrics,
        oos_metrics=oos_metrics,
        optimized_at=datetime.now(timezone.utc).isoformat(),
    )
    path = CONFIG_DIR / f"{strategy_id}_{tf.value}.json"
    with path.open("w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return path


def load_optimized_config(strategy_id: str, tf: Timeframe) -> OptimizedStrategyConfig:
    path = CONFIG_DIR / f"{strategy_id}_{tf.value}.json"
    if not path.exists():
        raise FileNotFoundError(f"No optimized config at {path}. Run optimize-strategies first.")
    with path.open() as f:
        data = json.load(f)
    return OptimizedStrategyConfig(**data)


def config_to_individual(cfg: OptimizedStrategyConfig) -> Individual:
    return Individual(
        seller_params=SellerParams(**cfg.seller_params),
        backtest_params=BacktestParams(**cfg.backtest_params),
    )


def _bars_per_day(tf: Timeframe) -> int:
    bar_minutes = tf.value if isinstance(tf.value, int) else 15
    return max(1, 1440 // bar_minutes)


def _forward_slice(df: pd.DataFrame, min_days: int, tf: Timeframe) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (context_with_warmup, forward_window) for indicator lookback."""
    bars_per_day = _bars_per_day(tf)
    min_bars = min_days * bars_per_day
    warmup_bars = WARMUP_DAYS * bars_per_day
    need = min_bars + warmup_bars
    if len(df) < need:
        raise ValueError(
            f"Need {need} bars ({min_days}d forward + {WARMUP_DAYS}d warmup on {tf.value}), got {len(df)}"
        )
    context = df.iloc[-need:].copy()
    forward = df.iloc[-min_bars:].copy()
    return context, forward


def _log_paper_record(record: dict[str, Any]) -> None:
    PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PAPER_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run_paper_forward(
    df: pd.DataFrame,
    strategy_id: str,
    tf: Timeframe,
    min_days: int = 30,
    use_regime_gate: bool = True,
) -> dict[str, Any]:
    """
    Forward paper test on holdout data (no re-optimization).

    Uses pre-optimized / frozen configs from strategies_optimized/.
    """
    if strategy_id in FROZEN_STRATEGIES:
        return run_paper_forward_frozen(df, strategy_id, tf, min_days)

    cfg = load_optimized_config(strategy_id, tf)
    ind = config_to_individual(cfg)
    context, forward = _forward_slice(df, min_days, tf)

    feats = individual_to_features(strategy_id, ind, context, tf)
    feats = feats.loc[forward.index.intersection(feats.index)]

    if use_regime_gate:
        gate = get_current_regime_gate()
        min_score = gate.get("min_regime_score", 0.5)
        if "regime_score" in feats.columns:
            from strategy.regime import apply_regime_gate
            feats["signal"] = apply_regime_gate(feats["signal"], feats["regime_score"], min_score)
            feats["exhaustion"] = feats["signal"]

    result = run_backtest(feats, ind.backtest_params)
    account = simulate_account(result["trades"])

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy_id,
        "timeframe": tf.value,
        "forward_days": min_days,
        "warmup_days": WARMUP_DAYS,
        "bars": len(forward),
        "regime_gate": get_current_regime_gate() if use_regime_gate else None,
        "metrics": result["metrics"],
        "account": {k: v for k, v in account.items() if k != "equity_curve"},
        "n_trades": len(result["trades"]),
    }
    _log_paper_record(record)
    return record


def run_paper_forward_frozen(
    df: pd.DataFrame,
    strategy_id: str,
    tf: Timeframe,
    min_days: int = 30,
) -> dict[str, Any]:
    """Paper forward for fusion_v2 / depth_charge using frozen param files."""
    context, forward = _forward_slice(df, min_days, tf)

    if strategy_id == "fusion_v2":
        from strategy.fusion_v2 import build_features, load_fusion_params
        fp, bt = load_fusion_params()
        feats = build_features(context, fp, tf, bt)
    elif strategy_id == "depth_charge":
        from strategy.depth_charge import build_features, load_depth_params
        dp, bt = load_depth_params()
        feats = build_features(context, dp, tf, bt)
    else:
        raise ValueError(f"Unknown frozen strategy: {strategy_id}")

    feats = feats.loc[forward.index.intersection(feats.index)]
    result = run_backtest(feats, bt)
    account = simulate_account(result["trades"])

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy_id,
        "timeframe": tf.value,
        "forward_days": min_days,
        "warmup_days": WARMUP_DAYS,
        "bars": len(forward),
        "regime_gate": get_current_regime_gate(),
        "metrics": result["metrics"],
        "account": {k: v for k, v in account.items() if k != "equity_curve"},
        "n_trades": len(result["trades"]),
    }
    _log_paper_record(record)
    return record
