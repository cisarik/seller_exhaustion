# PROMPTS.md — Prompt Engineering for Agents & LLM Features

Templates for **continuing development**, **regime LLM**, and **user ↔ agent** handoff.  
Copy-paste and fill `{placeholders}`. See `BRAINSTORMING.md` for strategy context.

---

## 1. Agent continuation (new Cursor session)

```
You are continuing the ADA seller_exhaustion backtesting project.

Read first (in order):
1. AGENTS.md — architecture
2. BRAINSTORMING.md — profit roadmap & session notes
3. brainstorming/backlog.json — priority queue
4. .data/top_candidate_15m.json — current best strategy

North star: ZISK. Current top: fusion_v2 15m GO (5/5 paper-top-stats).

Do NOT re-tune during walk-forward validation.
Always run params-sanity --normalize before comparing timeframes.

Next task: {TASK from backlog.json}
```

---

## 2. Walk-forward analysis prompt

```
Analyze walk-forward JSON at {path}.

For each strategy report:
- robust_profit_score vs sum_pnl (which is more stable?)
- folds with zero trades (sparse strategy risk)
- last 3 folds trend (degrading edge?)

Recommend: keep | re-tune | drop strategy.
Output: markdown table + single GO/MARGINAL/NO-GO per strategy.
```

---

## 3. Paper stability review

```
Read .data/top_candidate_runs_15m.jsonl

Compute:
- rolling 4-loop sum PnL trend
- worst consecutive losing loops
- trade frequency vs expectancy tradeoff

Verdict: safe for HERMES paper? What monitoring threshold triggers re-tune?
```

---

## 4. Regime LLM (weekly, text-only)

**System**:
```
You classify crypto market regime for mean-reversion LONG entries on ADA.
Reply JSON only: {"score":0-1,"label":"oversold_bounce|neutral|avoid","reason":"..."}
Score 0.7+ = favorable for bounce longs after seller exhaustion.
Do NOT predict price targets.
```

**User** (fill from weekly stats):
```
Asset: ADA/USDT
Timeframe: 15m + 4H context
Last 7d: return {pct}%, avg ATR rank {atr_rank}, RSI(14) median {rsi}
Trend: EMA24h vs EMA7d = {downtrend|up|flat}
Recent strategy paper PnL 30d: {pnl}

Classify regime for opening new long bounce trades this week.
```

---

## 5. Vision LLM (weekly chart — regime only)

**Not for entry signals.** Use 4H candlestick screenshot.

```
Analyze this 4H ADA chart image for REGIME only (not entries).

Return JSON:
{
  "structure": "downtrend|range|uptrend",
  "capitulation_signs": true|false,
  "regime_score": 0.0-1.0,
  "label": "oversold_bounce|neutral|avoid",
  "notes": "max 2 sentences"
}

Mean-reversion long bounces favor: downtrend + capitulation wick + volume spike.
```

---

## 6. Parameter sanity audit

```
Run mental audit on strategies_optimized/{file}.json for timeframe {tf}.

Check bar counts vs time intent:
- ema_fast ≈ 1440min / bar_minutes
- ema_slow ≈ 10080min / bar_minutes
- max_hold, fib_swing_lookback same scaling

List mismatches with expected bar count. Suggest normalized values.
```

---

## 7. HERMES handoff prompt

```
Run validate-candidate before any HERMES export.

If execution_verdict=BLOCKED: do NOT deploy. Report kill_switch reasons.
If READY: hermes-export → verify deploy_allowed=true in bundle.

Steps:
1. validate-candidate --tf 15m --data <parquet>
2. Read .data/validation_<tf>.json
3. Only if not BLOCKED: hermes-export
4. Kill switch rules: daily loss %, max trades, validate BLOCKED → flat

Output deployment checklist for human review.
```

---

## 8. Profit-focused optimization (GA)

```
Optimize {strategy_id} with profit_focused fitness on TRAIN only.

Constraints:
- min 15 trades on train
- max_drawdown_penalty ≥ 0.15
- validate on OOS holdout — reject if OOS n < 5 or OOS pnl < 0

Report train vs OOS delta (overfit indicator).
Save only if OOS pnl > 0 and profit_factor > 1.2.
```

---

## 9. User "I don't know how to ask" meta-prompt

```
I'm building a profitable ADA intraday bot. I don't know technical terms.

My goal: {e.g. "more stable wins, fewer big losses"}

Please:
1. Translate my goal into 2-3 measurable metrics (win rate, expectancy R, drawdown)
2. Pick ONE command from cli.py to run next
3. Explain result in plain Slovak

Project context: BRAINSTORMING.md
```

---

## 10. Anti-patterns (do not prompt)

- ❌ "Optimize until backtest shows 100% win rate" → guaranteed overfit
- ❌ "Use LLM for every bar entry" → non-deterministic, expensive
- ❌ "Skip walk-forward because train PnL is good"
- ❌ "Copy 15m bar params to 5m without bootstrap-configs"
- ❌ "Deploy live on MARGINAL without 30d HERMES paper"

---

## 12. Execution validation prompt

```
Run validate-candidate for {tf} and interpret for HERMES readiness.

Compare:
- loop verdict (GO/MARGINAL/NO-GO) vs execution_verdict (READY/CAUTION/BLOCKED)
- cost_stress 1× vs 2× PnL
- recent_3_loops vs historical median

If BLOCKED: explain whether to wait (regime) or re-tune. No HERMES deploy.
Reference: docs/VALIDATION.md
```

---

## 13. Slovak quick prompts (pre užívateľa)

**Stabilita vs execution:**
```
Spusti validate-candidate na 15m. Môžem ísť na HERMES alebo sme BLOCKED?
```

**Porovnanie stratégií:**
```
Walk-forward mean_reversion vs fusion_v2 vs depth_charge na 15m s auto-sanity.
Ktorá má najlepší robust_score?
```

**Ďalší krok k zisku:**
```
Prečítaj BRAINSTORMING.md tier A a urob prvú položku z backlogu.
```

---

*Update PROMPTS.md when new CLI commands or HERMES protocol versions ship.*
