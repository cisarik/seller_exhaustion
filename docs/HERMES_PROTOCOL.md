# HERMES Protocol v1.0

JSON handoff from **seller_exhaustion** (research) to **HERMES** (live execution agent).

## Lifecycle

```
walk-forward → top_candidate.json
paper-forward-top --loop → top_candidate_runs.jsonl
paper-top-stats → GO | MARGINAL | NO-GO
validate-candidate → READY | CAUTION | BLOCKED   ← required before export
hermes-export → hermes_bundle_<tf>.json (blocked if BLOCKED)
HERMES import → paper → testnet → live
```

See also **docs/VALIDATION.md** for execution verdict rules.

## Bundle schema (`hermes_bundle_<tf>.json`)

```json
{
  "protocol_version": "1.0.0",
  "created_at": "2026-06-16T12:00:00+00:00",
  "source_app": "seller_exhaustion",
  "strategy_id": "fusion_v2",
  "timeframe": "15m",
  "verdict": {
    "level": "GO",
    "stats": {
      "verdict_level": "GO",
      "n_loops": 11,
      "positive_pct": 0.64,
      "sum_pnl": 0.1112,
      "median_expectancy_r": 0.229,
      "passed_checks": 5,
      "total_checks": 5
    }
  },
  "config_paths": {
    "params": "strategies_optimized/fusion_v2_params.json"
  },
  "candidate_meta": { "...": "contents of top_candidate_15m.json" },
  "monitoring": {
    "runs_log": ".data/top_candidate_runs_15m.jsonl",
    "validate_cli": "poetry run python cli.py validate-candidate --tf 15m"
  },
  "validation": {
    "execution_verdict": "READY",
    "kill_switch": "OK"
  },
  "deploy_allowed": true,
  "risk_defaults": {
    "paper_trading": true,
    "testnet": true,
    "max_daily_loss_pct": 3.0,
    "max_open_positions": 1
  }
}
```

## Verdict rules (mirrors `paper-top-stats`)

| Level | Meaning | HERMES action |
|-------|---------|---------------|
| **GO** | 5/5 checks pass | Requires execution READY for deploy |
| **MARGINAL** | 3–4/5 checks | Paper only; validate-candidate weekly |
| **NO-GO** | ≤2/5 checks | Do not deploy |

**Execution gate** (`validate-candidate`): `hermes-export` fails if `execution_verdict=BLOCKED` (use `--force` for infra tests only).

Checks: positive loop %, median expectancy R (active loops), avg trades/loop, sum PnL, loop Sharpe.

## Strategy config files

| strategy_id | Config path |
|-------------|-------------|
| fusion_v2 | `strategies_optimized/fusion_v2_params.json` |
| depth_charge | `strategies_optimized/depth_charge_params.json` |
| others | `strategies_optimized/{id}_{tf}.json` |

## HERMES agent responsibilities (out of scope for this repo)

1. Load bundle + resolve config paths
2. Stream OHLCV (exchange WebSocket preferred)
3. Build features identically to backtester (pandas pipeline)
4. Execute entries at next bar open; exits per BacktestParams
5. Enforce `risk_defaults` + kill switch
6. Append telemetry to `hermes_telemetry.jsonl`
7. On daily loss breach → flat + alert

## Future v2 (planned)

- WebSocket command channel (pause/resume/kill)
- Signed bundles (Ed25519)
- Rolling verdict refresh pushed from scheduler
- Multi-strategy portfolio weights

## CLI

```bash
poetry run python cli.py hermes-export --tf 15m
poetry run python cli.py paper-scheduler --tf 15m --refresh \
  --data .data/X_ADAUSD_2024-01-01_2026-06-15_15minute.parquet
```

Implementation: `core/hermes_protocol.py`
