"""Execution telemetry for HERMES handoff and audit trail."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TELEMETRY_LOG = Path(".data/hermes_telemetry.jsonl")


def log_event(event_type: str, payload: dict[str, Any]) -> None:
    """Append structured execution/validation event."""
    TELEMETRY_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **payload,
    }
    with TELEMETRY_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
