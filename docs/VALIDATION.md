# Validation & Execution Guide (v4.2)

**Purpose**: Rozlišovať *historickú stabilitu* (loop GO) od *pripravenosti na execution* (READY/BLOCKED).

---

## Dva verdikty — prečo oba

| Nástroj | Otázka | Verdikt |
|---------|--------|---------|
| `paper-top-stats` | Boli historické rolling loopy ziskové? | GO / MARGINAL / NO-GO |
| `validate-candidate` | Prežije edge **teraz** + pri **2× nákladoch**? | READY / CAUTION / **BLOCKED** |

**GO ≠ READY.** Môžeš mať GO na 11 loopoch, ale BLOCKED na poslednom 60d forward okne.

---

## Pipeline (povinné pred HERMES)

```bash
# 1. Rank + export top candidate
poetry run python cli.py walk-forward --data .data/X_ADAUSD_*_15minute.parquet \
  --strategies fusion_v2,depth_charge,mean_reversion --tf 15m --auto-sanity -j 4

# 2. Rolling paper stability
poetry run python cli.py paper-forward-top --tf 15m --loop --loops 11 --step-days 60 --clear

# 3. Loop verdict
poetry run python cli.py paper-top-stats --tf 15m

# 4. Execution validation (POVINNÉ pred HERMES)
poetry run python cli.py validate-candidate --tf 15m \
  --data .data/X_ADAUSD_2024-01-01_2026-06-15_15minute.parquet

# 5. HERMES export (blokuje pri BLOCKED, --force len pre paper test infra)
poetry run python cli.py hermes-export --tf 15m
```

Skratka: `make validate TF=15m DATA=.data/...parquet`

---

## Čo meria `validate-candidate`

### 1. Loop stability
Rovnaké metriky ako `paper-top-stats` (positive %, sum PnL, expectancy).

### 2. Streaks
- `max_consecutive_losses` — koľko loopov za sebou bolo v mínuse
- `recent_n_sum_pnl` — sum posledných N loopov (default 3)
- `degrading` — recent horší ako historický medián

### 3. Cost stress
Re-run **posledných 60d forward** s násobenými `fee_bp` + `slippage_bp`:

| Multiplier | Význam |
|------------|--------|
| 1.0× | Baseline config |
| 1.5× | Mierne horšie execution |
| 2.0× | **Kill-switch prah** — musí byť ziskové pre READY |
| 3.0× | Stress test |

### 4. Bootstrap
Resampling loop PnL → `P(profit)`, 5.–95. percentil. Potrebuje ≥4 loopy.

### 5. Kill-switch

| Stav | Podmienka |
|------|-----------|
| **OK** | Všetko nominálne |
| **CAUTION** | Loss streak / degradácia |
| **PAUSE** | 2× costs neziskové |
| **RETUNE** | Posledné 3 loopy sum PnL < 0 |

### 6. Execution verdict

| Verdict | HERMES |
|---------|--------|
| **READY** | Paper trading OK |
| **CAUTION** | Paper s prísnym monitoringom |
| **BLOCKED** | **Nepustiť** — re-tune alebo čakať na regime |

---

## Aktuálne zistenia (2026-06-16, fusion_v2 15m)

Report: `.data/validation_15m.json`

| Metrika | Hodnota |
|---------|---------|
| Loop stats | GO (5/5) |
| Posledných 60d forward | **−0.046 PnL**, 0 % WR |
| Posledné 3 loopy | **−0.079 sum** |
| 2× costs | **−0.050 PnL** |
| Bootstrap P(profit) | 65 % (5. pct −0.21) |
| **Execution verdict** | **BLOCKED** |

### Záver pre agentov

1. **Nepustiť HERMES live** — edge v aktuálnom režime nefunguje.
2. **Historický GO je validný** — stratégia mala edge v minulosti; nie je to náhodný backtest.
3. **Ďalší krok**: týždenný `validate-candidate`; ak 2–3 týždne READY → HERMES paper 30d.
4. **Ak BLOCKED persistuje**: fusion_tune, regime filter, alebo pause do oversold režimu.

---

## Súbory

| Súbor | Obsah |
|-------|-------|
| `.data/validation_<tf>.json` | Posledný validation report |
| `.data/hermes_telemetry.jsonl` | Audit udalosti (validation, monitor) |
| `.data/top_candidate_runs_<tf>.jsonl` | Vstup pre validate |

---

## Kód

| Modul | Úloha |
|-------|-------|
| `backtest/validation.py` | Cost stress, streaks, bootstrap, kill-switch |
| `backtest/paper_stats.py` | Loop GO/MARGINAL/NO-GO |
| `exec/telemetry.py` | JSONL audit log |

---

## HERMES integrácia

`hermes-export` načíta `.data/validation_<tf>.json` ak existuje:
- `execution_verdict: BLOCKED` → export zlyhá (exit 2)
- `--force` → export s `"deploy_allowed": false` pre infra test

Bundle obsahuje pole `validation` s kill-switch stavom.

---

*Aktualizovať po každom validate-candidate behu.*
