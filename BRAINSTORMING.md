# BRAINSTORMING.md — Profit Roadmap & Research Notes

**Project**: ADA Seller-Exhaustion backtesting → HERMES live agent  
**North star**: **ZISK** (consistent out-of-sample profit on ADA intraday)  
**Last updated**: 2026-06-16 (session: walk-forward, paper-monitor, fusion_v2 GO)

---

## 1. Session outcomes (2026-06-16)

### What we built
| Component | Purpose |
|-----------|---------|
| `walk-forward` | Rolling OOS folds, `robust_profit_score` ranking |
| `params-sanity` / `bootstrap-configs` | Time-consistent params across 5m/15m |
| `paper-forward-top --loop` | Rolling stability log → `.data/top_candidate_runs_<tf>.jsonl` |
| `paper-top-stats` | Go / Marginal / No-Go verdict (5 checks) |
| `paper-monitor` | Cron-friendly snapshot + cumulative verdict |
| `paper-scheduler` | Full cycle: refresh → monitor → HERMES export |
| `hermes-export` | JSON bundle for live agent handoff |
| `indicators/fractal.py` | Hurst exponent + Bill Williams fractals |
| Hurst in `regime.py` | 15% weight in regime_score for MR bias |
| Parallel walk-forward `-j` | ProcessPoolExecutor on folds |
| `core/env_bootstrap.py` | Silence py2app/pkg_resources warnings |

### Key findings
- **Polygon ADAUSD 15m** only from ~mid-2024 (not 2018) — walk-forward limited to ~11 folds.
- **Parameter drift**: optimized JSONs had wrong bar counts vs time intent — fixed via `param_sanity`.
- **Top candidate shifted**: `depth_charge` MARGINAL → **`fusion_v2` GO (5/5)** on 15m loops.
  - 64% positive loops, sum PnL +0.111, median exp R 0.229, ~4 trades/loop.
  - Weak spot: latest window (Jun 2026) — monitor required.
- **`depth_charge` echo** extended with `seller_aggressive` (high R, low frequency confirmation).
- **Vision LLM on charts**: useful for weekly regime narrative, **not** for entry signals (cost, non-determinism).

---

## 2. Profit hypothesis stack (current best)

```
Polygon OHLCV
    → frozen configs (strategies_optimized/)
    → feature build (pandas, ~0.16s)
    → regime gate (RSI + trend + vol + Hurst MR score)
    → fusion_v2: 2-of-3 member vote + floor zone + volume pctile
    → Fib 61.8% exit (default from mean_reversion OOS)
    → walk-forward + paper loops → GO verdict
    → HERMES bundle → live paper → testnet → live
```

**Why fusion_v2 wins today**: diversifies single-strategy overfit; trades more often than depth_charge alone; OOS-weighted confidence.

---

## 3. Fractal market hypothesis (research)

### Implemented (v4.1)
- **Hurst exponent** (rolling R/S): H < 0.45 → mean-reversion favorable for our long-bounce edge.
- **Bill Williams fractals**: pivot highs/lows — future use for swing Fib anchors.

### Next experiments (priority order)
1. **Gate entries on Hurst < 0.48** explicitly in fusion_v2 (not only blended in regime_score).
2. **Fibonacci + fractal pivots**: use fractal low as swing anchor instead of rolling min.
3. **Multi-timeframe Hurst**: 15m signal only when 1h Hurst also MR-biased.
4. **Golden ratio grid**: optimize exit at 0.618 vs 0.786 per regime bucket.
5. **Elliott wave proxy**: 3-wave down + volume climax (heuristic, no full EW engine).

### References (web research themes)
- Mandelbrot — fractal markets, scaling laws
- Peters — Hurst exponent in finance
- Bill Williams — fractals + Alligator (trend filter)
- Fibonacci retracements as self-similar structure across TFs

---

## 4. Ideas ranked by expected profit impact

### Tier A — do next (high ROI, low risk)
| Idea | Rationale |
|------|-----------|
| Daily `paper-scheduler` cron | Catch regime drift before live |
| Re-tune `fusion_v2` grid (like depth_tune) | Member weights + floor_pct OOS-first |
| Hurst hard gate A/B test | Walk-forward with/without |
| More data (Binance ADAUSDT 15m 2018+) | Polygon gap; second source in `data/provider` |
| `profit_focused` GA on fusion members only | Exit params per member |

### Tier B — medium effort
| Idea | Rationale |
|------|-----------|
| Ensemble: fusion_v2 + depth_charge echo OR | Rare high-R depth_charge + frequent fusion |
| Weekly LLM regime (text) + Hurst (numeric) | LLM sets min_regime_score dynamically |
| 5m paper after bootstrap + sanity | Higher frequency if spreads OK |
| Slippage stress test | +10bp in backtest before GO |
| Walk-forward on fees ×2 | Robustness filter |

### Tier C — strategic / HERMES
| Idea | Rationale |
|------|-----------|
| HERMES protocol v1 → v2 (WebSocket heartbeat) | Live agent health + kill switch |
| Position sizing by expectancy_r rolling | Kelly fraction capped |
| Multi-asset (BTC, ETH) same pipeline | Diversification |
| Vision LLM weekly 4H chart | Regime label only, not entries |

### Tier D — performance engineering (not primary profit driver)
| Idea | Notes |
|------|-------|
| **Numba** on Hurst inner loop | Easier than Cython; 10–50× on rolling Hurst |
| **Cython** on backtest inner bar loop | engine.py hot path if profiling proves bottleneck |
| **Parallel depth_tune grid** | Same pattern as walk-forward `-j` |
| **Polars** for feature pipeline | Only if pandas becomes proven bottleneck |
| **Rust extension** | Overkill until GA + walk-forward saturate CPU |

**Decision**: Profile first. Current GA already uses `multiprocessing`. Walk-forward now parallel. **Numba on Hurst** is the next perf win if needed.

---

## 5. HERMES agent architecture (draft)

```
┌─────────────────────────┐     hermes_bundle.json      ┌──────────────────┐
│ seller_exhaustion       │ ──────────────────────────► │ HERMES agent     │
│ (this repo)             │     + config JSON paths     │ (live execution) │
│ research + validation   │                             │ paper→testnet→live│
└─────────────────────────┘ ◄──── telemetry.jsonl ───── └──────────────────┘
         │                                                          │
         │ paper_trades.jsonl                                       │ exchange API
         ▼                                                          ▼
    .data/ monitoring                                      Binance / etc.
```

**Milestone 1**: GO verdict on 15m fusion_v2 + 30 days paper on HERMES  
**Milestone 2**: Testnet with real latency  
**Milestone 3**: Live with 0.5% risk/trade cap  

See `docs/HERMES_PROTOCOL.md` for JSON schema.

---

## 6. JSON artifact map (for agents)

| File | Producer | Consumer |
|------|----------|----------|
| `.data/top_candidate_<tf>.json` | walk-forward | paper-forward-top |
| `.data/top_candidate_runs_<tf>.jsonl` | paper-forward-top --loop, paper-monitor | paper-top-stats |
| `.data/hermes_bundle_<tf>.json` | hermes-export, paper-scheduler | HERMES agent |
| `.data/paper_trades.jsonl` | run_paper_forward | audit / UI |
| `strategies_optimized/*.json` | tune-*, optimize-strategies | frozen strategies |
| `brainstorming/backlog.json` | manual / agent | priority queue |

---

## 7. Open questions (user + agent brainstorming)

**User**: Can Vision LLM replace indicator pipeline?  
**Answer**: No for entries. Yes for weekly regime narrative layered on Hurst/RSI.

**User**: Fractals + golden ratio strategy?  
**Answer**: Hurst gate + fractal pivots for Fib anchors — implement incrementally, validate walk-forward.

**User**: 2018–2026 data for AI patterns?  
**Answer**: Need alternate data source; Polygon ADAUSD 15m starts ~2024.

**Agent**: Why MARGINAL → GO after strategy switch?  
**Answer**: fusion_v2 trades more loops with positive expectancy; depth_charge too sparse.

---

## 8. Suggested next 3 agent sessions (automatic)

1. **Data**: Binance/yfinance fallback + merge into provider; re-run walk-forward on longer history.
2. **Tune**: `fusion_tune.py` grid (mirror depth_tune) with parallel workers.
3. **HERMES**: Minimal Python agent stub that reads `hermes_bundle_15m.json` and runs paper loop.

---

## 9. Risk register

| Risk | Mitigation |
|------|------------|
| Overfit to 2024–2026 bull/side | Walk-forward + fee stress |
| Low trade count | Prefer fusion over depth_charge alone |
| Param timeframe drift | Always `--auto-sanity` before WF |
| Latest window negative | paper-monitor alerts |
| Live slippage | paper on HERMES 30d minimum |

---

*This file is living documentation. Update after every major session.*
