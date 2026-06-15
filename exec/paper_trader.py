"""Paper trading execution layer (simulation + forward test)."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from backtest.engine import run_backtest
from backtest.profit import simulate_account
from backtest.strategy_ga import individual_to_features, evaluate_strategy_individual
from backtest.optimizer import Individual
from core.models import BacktestParams, Timeframe, FitnessConfig
from strategy.seller_exhaustion import SellerParams
from strategy.regime_weekly import get_current_regime_gate

PAPER_LOG = Path(".data/paper_trades.jsonl")
CONFIG_DIR = Path("strategies_optimized")


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


def run_paper_forward(
    df: pd.DataFrame,
    strategy_id: str,
    tf: Timeframe,
    min_days: int = 30,
    use_regime_gate: bool = True,
) -> dict[str, Any]:
    """
    Forward paper test on holdout data (no re-optimization).

    Uses pre-optimized config from strategies_optimized/.
    """
    cfg = load_optimized_config(strategy_id, tf)
    ind = config_to_individual(cfg)

    bar_minutes = tf.value if isinstance(tf.value, int) else 15
    bars_per_day = max(1, 1440 // bar_minutes)
    min_bars = min_days * bars_per_day
    if len(df) < min_bars:
        raise ValueError(f"Need at least {min_bars} bars ({min_days}d on {tf.value}), got {len(df)}")

    forward_df = df.iloc[-min_bars:].copy()
    feats = individual_to_features(strategy_id, ind, forward_df, tf)

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
        "bars": len(forward_df),
        "regime_gate": get_current_regime_gate() if use_regime_gate else None,
        "metrics": result["metrics"],
        "account": {k: v for k, v in account.items() if k != "equity_curve"},
        "n_trades": len(result["trades"]),
    }

    PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PAPER_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")

    return record
