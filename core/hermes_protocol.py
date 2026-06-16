"""
HERMES agent integration protocol (v1 draft).

This backtesting tool validates strategies; HERMES executes them live.
Communication is file/HTTP JSON — no direct exchange coupling here.

Lifecycle:
  1. walk-forward + paper-forward-top --loop  → robust candidate
  2. paper-top-stats / paper-monitor          → GO | MARGINAL | NO-GO
  3. validate-candidate                         → READY | CAUTION | BLOCKED
  4. hermes-export (blocked if BLOCKED)         → .data/hermes_bundle_<tf>.json
  5. HERMES agent imports bundle, paper → testnet → live

See docs/HERMES_PROTOCOL.md and docs/VALIDATION.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.models import Timeframe


PROTOCOL_VERSION = "1.0.0"


@dataclass
class HermesVerdict:
    level: str  # GO | MARGINAL | NO-GO
    passed_checks: int
    total_checks: int
    positive_loop_pct: float
    sum_pnl: float
    median_expectancy_r: float
    message: str


@dataclass
class HermesStrategyBundle:
    """Deployable package for HERMES live agent."""

    protocol_version: str = PROTOCOL_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_app: str = "seller_exhaustion"
    strategy_id: str = ""
    timeframe: str = "15m"
    verdict: dict[str, Any] = field(default_factory=dict)
    config_paths: dict[str, str] = field(default_factory=dict)
    candidate_meta: dict[str, Any] = field(default_factory=dict)
    monitoring: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    deploy_allowed: bool = False
    risk_defaults: dict[str, Any] = field(default_factory=lambda: {
        "paper_trading": True,
        "testnet": True,
        "max_daily_loss_pct": 3.0,
        "max_open_positions": 1,
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_hermes_bundle(
    strategy_id: str,
    tf: Timeframe,
    verdict_level: str,
    stats: dict[str, Any],
    candidate_path: Path,
    runs_path: Path | None = None,
    validation: dict[str, Any] | None = None,
    deploy_allowed: bool = True,
) -> HermesStrategyBundle:
    """Assemble bundle from top candidate + stability + execution validation."""
    config_map: dict[str, str] = {}
    if strategy_id in ("fusion_v2", "depth_charge"):
        p = Path(f"strategies_optimized/{strategy_id}_params.json")
        if p.exists():
            config_map["params"] = str(p)
    else:
        p = Path(f"strategies_optimized/{strategy_id}_{tf.value}.json")
        if p.exists():
            config_map["optimized"] = str(p)

    candidate_meta: dict[str, Any] = {}
    if candidate_path.exists():
        candidate_meta = json.loads(candidate_path.read_text())

    return HermesStrategyBundle(
        strategy_id=strategy_id,
        timeframe=tf.value,
        verdict={
            "level": verdict_level,
            "stats": stats,
        },
        config_paths=config_map,
        candidate_meta=candidate_meta,
        monitoring={
            "runs_log": str(runs_path) if runs_path else "",
            "recommended_cli": f"poetry run python cli.py paper-monitor --tf {tf.value}",
            "stats_cli": f"poetry run python cli.py paper-top-stats --tf {tf.value}",
            "validate_cli": f"poetry run python cli.py validate-candidate --tf {tf.value}",
        },
        validation=validation or {},
        deploy_allowed=deploy_allowed,
    )


def export_hermes_bundle(bundle: HermesStrategyBundle, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(bundle.to_dict(), f, indent=2)
    return output
