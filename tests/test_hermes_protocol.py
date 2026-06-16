"""Tests for HERMES protocol bundle."""

import json
from pathlib import Path

from core.hermes_protocol import build_hermes_bundle, export_hermes_bundle
from core.models import Timeframe


def test_build_and_export_bundle(tmp_path):
    cand = tmp_path / "top.json"
    cand.write_text(json.dumps({"strategy_id": "fusion_v2", "robust_profit_score": 0.07}))

    bundle = build_hermes_bundle(
        strategy_id="fusion_v2",
        tf=Timeframe.m15,
        verdict_level="GO",
        stats={"sum_pnl": 0.1},
        candidate_path=cand,
        runs_path=None,
    )
    assert bundle.strategy_id == "fusion_v2"
    assert bundle.protocol_version == "1.0.0"

    out = tmp_path / "hermes.json"
    export_hermes_bundle(bundle, out)
    data = json.loads(out.read_text())
    assert data["verdict"]["level"] == "GO"
    assert data["risk_defaults"]["paper_trading"] is True
