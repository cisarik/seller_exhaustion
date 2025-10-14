# PRD — ADA Seller‑Exhaustion Agent (15‑minute)

**Product:** Intraday (15m) research → backtest → live trading agent for Cardano (ADAUSD)  
**Tech:** Python 3.10, async I/O, PySide6 UI, PyQtGraph candles, Polygon.io crypto REST, optional Yahoo/yfinance fallback  
**Theme:** Dark Forest UI (QSS)  
**Owner:** Michal  
**Goal date (MVP):** T+7 dní

---

## 1) Cieľ a kontext

Chcem backtestovať a potom spúšťať agenta, ktorý na **15‑min timeframe** deteguje **„seller exhaustion“** (vyčerpanie predajcov) a **otvára long** po potvrdení sviečky.

**Why now:** ADA je volatilná, 15m timeframe dáva dostatok signálov a rýchlosť iterácie výskumu.

**Key result (MVP):**
- Získať 15m historické OHLCV pre `X:ADAUSD` (Polygon) a vybudovať **deterministický backtest** s parametrami.
- V UI zobraziť sviece, overlay indikátory (EMA/SMA/MACD/RSI) a zvýrazniť signály.
- „Paper trade“ mód s jednoduchým exekútorom (bez reálneho brokera) a plánovačom na **bar‑close** každej 15m sviečky (UTC :00/:15/:30/:45).

---

## 2) Scope

### In‑scope (MVP)
- **Datasety**: Polygon.io crypto aggregates (15m pre `X:ADAUSD`), voliteľne fallback **yfinance** (`ADA-USD`) pre backtest len na validáciu tvaru dát.
- **Indikátory**: EMA/SMA/MACD/RSI –
  - primárne **počítať lokálne** (pandas),
  - voliteľne **fetch**ovať cez Polygon „Technical Indicators“ (SMA/EMA/MACD/RSI) pre porovnanie.
- **Signál seller‑exhaustion**: downtrend filter (EMA_f < EMA_s), volume z‑score spike, range z‑score spike, close v top časti sviečky.
- **Backtest**: event‑driven, vstup t+1 open, ATR stop, 2R TP, `max_hold` safety; voliteľná provízia a slippage.
- **UI**: PySide6 okno, PyQtGraph sviece + overlay indikátorov, panel s parametrami, log panel.
- **Async runtime**: `httpx.AsyncClient`, `qasync`/`QtAsyncio` integrácia s Qt event loop.

### Out‑of‑scope (MVP)
- Priamy broker execution (Binance/Kraken). Papierový exekútor stačí.
- Portfóliová alokácia viacerých párov.

---

## 3) Požiadavky

### Funkčné
1. **Backtest 15m** na zvolenom období s parametrizáciou prahov (z‑window, EMA dĺžky, ATR stop, R:R, max_hold, fees, slippage).
2. **UI candles** s overlay EMA/SMA/RSI/MACD; zvýraznenie barov, kde platí seller‑exhaustion.
3. **Paper‑trading** tick (15m): po uzatvorení baru skontrolovať signál a podľa pravidiel otvoriť/riadiť pozíciu.
4. **Export**: `trades.csv`, `features.csv`, JSON snapshot parametrov a metrík.

### Nefunkčné
- **Deterministickosť** backtestu (rovnaké vstupy → rovnaké výsledky).
- **Modularita**: čisté hranice medzi DataProvider, Indicators, Strategy, Backtest, UI, Execution.
- **Async I/O**: žiadne blokovanie UI počas fetchu.

---

## 4) Dátové zdroje a časovanie

- **Polygon.io**: Crypto aggregates `/v2/aggs/ticker/X:ADAUSD/range/15/minute/{from}/{to}`; indikátory voliteľne `/v1/indicators/{sma|ema|macd|rsi}/X:ADAUSD`.
- **Timezone**: všetko v **UTC**. Bar close target: každých 15 min na :00/:15/:30/:45.
- **Fallback**: Yahoo Finance `ADA-USD` (yfinance) – len denné/minútové podľa dostupnosti; použije sa iba pre sanity checks.
- **Poznámka o objemoch**: Polygon **nekonsoliduje Binance**; objem sa môže líšiť od grafov Binance. Thresholdy ladiť **na rovnakom datasete**.

---

## 5) Architektúra

```
ada-agent/
  app/
    main.py              # PySide6 + qasync bootstrap, routing tabov
    theme.py             # Dark Forest UI QSS ako Python reťazec
    widgets/
      candle_view.py     # PyQtGraph Canvas + overlays (EMA/SMA/RSI/MACD) + signal markers
      settings_dialog.py # Nastavenia (API kľúč, prahy, fees, slippage)
      log_panel.py
  core/
    models.py            # Pydantic dataclass-y: Bar, IndicatorBundle, Trade, Params
    timeutils.py         # Align na 15m boundary, UTC helpers
  data/
    polygon_client.py    # Async fetch aggregates + indicators (httpx)
    yahoo_client.py      # yfinance fallback (sync → async wrapper)
    cache.py             # Disk/SQLite cache (serde JSON/Parquet)
    provider.py          # Interface + orchestrácia
  indicators/
    local.py             # Pandas výpočty EMA/SMA/RSI/MACD, ATR
  strategy/
    seller_exhaustion.py # signál + parametre
  backtest/
    engine.py            # event-driven backtester
    metrics.py           # výpočty metrík
  exec/
    paper.py             # paper trade execution + positions state
    scheduler.py         # asyncio plánovanie bar-close ticku
  cli.py                 # Typer príkazy: fetch, backtest, ui, live
  config/
    settings.py          # pydantic Settings (.env)
  tests/
    test_strategy.py     # unit test signálu
    test_backtest.py     # základné testy
  .env.example
  pyproject.toml
  README.md
```

Kľúčové voľby:
- **PyQtGraph** (výkon, interaktivita) pre sviece a overlaye.
- **httpx** pre async HTTP.
- **qasync / QtAsyncio**: integrácia asyncio slučky s Qt.
- **pydantic** pre typovo‑bezpečné konfigurácie a DAOs.
- **Typer** (alebo argparse) pre ergonomické CLI.

---

## 6) Dark Forest UI (QSS)

Bude aplikovaný globálne pri štarte app. (Zhrnutie farieb a QSS je súčasťou tohto PRD pre rýchle copy‑paste.)

```python
# app/theme.py
DARK_FOREST_QSS = r"""
QDialog, QWidget { background-color: #0f1a12; color: #e8f5e9; }
QLabel { color: #e8f5e9; }
.form-label, QLabel[variant="secondary"] { color: #b6e0bd; font-weight: bold; }
QTabWidget::pane { border: 1px solid #2f5c39; background: #0f1a12; }
QTabBar::tab { background:#1a2f1f; color:#b6e0bd; padding:8px 16px; border:1px solid #2f5c39; margin-right:2px; }
QTabBar::tab:selected { background:#295c33; color:#e8f5e9; font-weight:bold; border-bottom:2px solid #4caf50; }
QTabBar::tab:hover { background:#213f29; }
QPushButton { padding:8px 16px; font-size:12px; border-radius:4px; background:#182c1d; color:#b6e0bd; border:1px solid #2f5c39; }
QPushButton:hover { background:#213f29; border-color:#4caf50; }
QPushButton:pressed { background:#152820; }
QPushButton#primaryButton { padding:10px 20px; font-size:13px; font-weight:bold; border-radius:6px; background:#2e7d32; color:#e8f5e9; border:2px solid #4caf50; }
QPushButton#primaryButton:hover { background:#388e3c; }
QPushButton#primaryButton:pressed { background:#1b5e20; }
QPushButton[danger="true"] { background:#2c1d1d; color:#e0b6b6; border:1px solid #5c2f2f; }
QPushButton[danger="true"]:hover { background:#3f2929; border-color:#af4c4c; }
QLineEdit, QTextEdit, QComboBox { background:#000; color:#e8f5e9; border:1px solid #2f5c39; border-radius:4px; padding:6px 8px; }
QLineEdit:hover, QTextEdit:hover, QComboBox:hover { border-color:#4caf50; background:#0a0a0a; }
QLineEdit:focus, QTextEdit:focus, QComboBox:focus { border-color:#4caf50; background:#0a0a0a; }
QComboBox QAbstractItemView { background:#000; color:#e8f5e9; selection-background-color:#295c33; }
QLabel[role="title"] { font-size:18px; font-weight:bold; color:#e8f5e9; padding:8px 0px; }
QLabel[role="statusbar"] { color:#4caf50; font-size:14px; font-weight:bold; padding:12px 16px; background:#000; border-top:1px solid #2f5c39; }
QLabel[variant="warn"] { background:#3d2a0f; color:#ffd54f; padding:10px; border:2px solid #ff9800; border-radius:6px; font-weight:bold; }
"""
```

---

## 7) Konfigurácia a .env

```
# .env.example
POLYGON_API_KEY=your_key_here
DATA_DIR=.data
TZ=UTC
```

```python
# config/settings.py
from pydantic import BaseSettings
class Settings(BaseSettings):
    polygon_api_key: str
    data_dir: str = ".data"
    tz: str = "UTC"
    class Config:
        env_prefix = ""
        env_file = ".env"
settings = Settings()
```

---

## 8) Data provider (async, Polygon + fallback)

```python
# data/polygon_client.py
from __future__ import annotations
import asyncio, httpx
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from config.settings import settings

BASE = "https://api.polygon.io"

class AggBar(BaseModel):
    ts: int
    open: float; high: float; low: float; close: float; volume: float

class PolygonClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 20.0):
        self.api_key = api_key or settings.polygon_api_key
        self.client = httpx.AsyncClient(timeout=timeout)

    async def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {**params, "apiKey": self.api_key}
        r = await self.client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    async def aggregates_15m(self, ticker: str, date_from: str, date_to: str) -> List[AggBar]:
        url = f"{BASE}/v2/aggs/ticker/{ticker}/range/15/minute/{date_from}/{date_to}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        items: List[Dict[str, Any]] = []
        while True:
            data = await self._get(url, params)
            items.extend(data.get("results", []))
            next_url = data.get("next_url")
            if not next_url:
                break
            url, params = next_url, {"apiKey": self.api_key}
            await asyncio.sleep(0.15)  # be nice
        return [AggBar(ts=i["t"], open=i["o"], high=i["h"], low=i["l"], close=i["c"], volume=i["v"]) for i in items]

    async def indicator_sma(self, ticker: str, window: int, timespan: str = "minute", multiplier: int = 15,
                            date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, Any]:
        url = f"{BASE}/v1/indicators/sma/{ticker}"
        params = {"timespan": timespan, "multiplier": multiplier, "window": window,
                  "series_type": "close", "expand_underlying": "false"}
        if date_from: params["from"] = date_from
        if date_to: params["to"] = date_to
        return await self._get(url, params)

    async def close(self):
        await self.client.aclose()
```

Fallback (yfinance) – načítanie do DataFrame, mapovanie na `AggBar`.

---

## 9) Indikátory (lokálne výpočty)

```python
# indicators/local.py
import pandas as pd, numpy as np

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))

def atr(high, low, close, window: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()
```

---

## 10) Seller‑Exhaustion signál & parametre

```python
# strategy/seller_exhaustion.py
from dataclasses import dataclass
import pandas as pd, numpy as np
from indicators.local import ema, atr

@dataclass
class SellerParams:
    ema_fast: int = 96
    ema_slow: int = 672
    z_window: int = 672
    vol_z: float = 2.0
    tr_z: float = 1.2
    cloc_min: float = 0.6
    atr_window: int = 96

def zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - m) / sd

def build_features(df: pd.DataFrame, p: SellerParams) -> pd.DataFrame:
    out = df.copy()
    out["ema_f"], out["ema_s"] = ema(out.close, p.ema_fast), ema(out.close, p.ema_slow)
    out["downtrend"] = out.ema_f < out.ema_s
    out["atr"] = atr(out.high, out.low, out.close, p.atr_window)
    out["vol_z"] = zscore(out.volume, p.z_window)
    tr = out.atr * p.atr_window
    out["tr_z"] = zscore(tr, p.z_window)
    span = (out.high - out.low).replace(0, np.nan)
    out["cloc"] = (out.close - out.low) / span
    out["exhaustion"] = out.downtrend & (out.vol_z > p.vol_z) & (out.tr_z > p.tr_z) & (out.cloc > p.cloc_min)
    return out
```

---

## 11) Backtest engine (event‑driven)

```python
# backtest/engine.py
from dataclasses import dataclass
import pandas as pd, numpy as np

@dataclass
class btParams:
    atr_stop_mult: float = 0.7
    reward_r: float = 2.0
    max_hold: int = 96
    fee_bp: float = 5.0   # 5 bps per trade
    slippage_bp: float = 5.0

def run_backtest(df: pd.DataFrame, p: btParams) -> dict:
    d = df.dropna(subset=["atr"]).copy()
    trades = []
    in_pos = False
    for i in range(len(d.index) - 1):
        t, nxt = d.index[i], d.index[i+1]
        row, nxt_row = d.loc[t], d.loc[nxt]
        if not in_pos and bool(row.exhaustion):
            entry = float(nxt_row.open)
            stop = float(row.low - p.atr_stop_mult * row.atr)
            risk = max(1e-8, entry - stop)
            tp = entry + p.reward_r * risk
            bars = 0
            in_pos = True
        elif in_pos:
            bars += 1
            op, lo, hi, cl = map(float, [nxt_row.open, nxt_row.low, nxt_row.high, nxt_row.close])
            exit_price, reason = None, None
            if op <= stop: exit_price, reason = op, "stop_gap"
            elif lo <= stop: exit_price, reason = stop, "stop"
            elif hi >= tp: exit_price, reason = tp, "tp"
            elif bars >= p.max_hold: exit_price, reason = cl, "time"
            if exit_price is not None:
                fee = (entry + exit_price) * (p.fee_bp + p.slippage_bp) / 10000.0
                pnl = exit_price - entry - fee
                R = pnl / risk
                trades.append({"entry_ts": t, "exit_ts": nxt, "entry": entry, "exit": exit_price, "pnl": pnl, "R": R, "reason": reason})
                in_pos = False
    tr = pd.DataFrame(trades)
    return {"trades": tr, "metrics": {
        "n": int(len(tr)),
        "win_rate": float((tr.pnl>0).mean()) if len(tr) else 0.0,
        "avg_R": float(tr.R.mean()) if len(tr) else 0.0,
        "total_pnl": float(tr.pnl.sum()) if len(tr) else 0.0,
    }}
```

---

## 12) PySide6 UI + qasync integrácia

```python
# app/main.py
import sys, asyncio
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
try:
    import qasync  # preferované
    HAS_QASYNC = True
except Exception:
    HAS_QASYNC = False
from app.theme import DARK_FOREST_QSS
from app.widgets.candle_view import CandleChartWidget

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADA 15m — Seller Exhaustion")
        self.view = CandleChartWidget()
        self.setCentralWidget(self.view)

async def async_boot():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_FOREST_QSS)
    win = Main(); win.show()
    # prvotné načítanie dát (await fetch + render)
    await win.view.load_initial()
    # plánovač refreshu na bar‑close
    await win.view.schedule_bar_closes()
    if HAS_QASYNC:
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        with loop: loop.run_forever()
    else:
        # Bez qasync: fallback — beží Qt loop + spúšťame asyncio cez QTimer
        timer = QTimer(); timer.start(50); timer.timeout.connect(lambda: None)
        sys.exit(app.exec())

if __name__ == "__main__":
    asyncio.run(async_boot())
```

```python
# app/widgets/candle_view.py
import asyncio, pandas as pd
from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features

class CandleChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0f1a12")
        lay = QVBoxLayout(self)
        lay.addWidget(self.plot)
        self.dp = DataProvider()
        self.params = SellerParams()

    async def load_initial(self):
        df = await self.dp.fetch_15m("X:ADAUSD", "2024-01-01", "2025-10-13")
        feats = build_features(df, self.params)
        self.render_candles(feats)

    async def schedule_bar_closes(self):
        while True:
            # počkaj do najbližšieho 15m boundary (UTC)
            await asyncio.sleep(60)
            # TODO: align na boundary; potom refetch posledných N barov a redraw

    def render_candles(self, df: pd.DataFrame):
        # TODO: vykresliť sviece (custom CandlestickItem) + overlay EMA
        pass
```

---

## 13) CLI (Typer)

```python
# cli.py
import asyncio, typer, pandas as pd
from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from backtest.engine import run_backtest, btParams

app = typer.Typer()

@app.command()
def fetch(from_: str = "2023-01-01", to: str = "2025-10-13"):
    async def _run():
        dp = DataProvider(); df = await dp.fetch_15m("X:ADAUSD", from_, to)
        print(df.tail())
    asyncio.run(_run())

@app.command()
def backtest(from_: str = "2023-01-01", to: str = "2025-10-13"):
    async def _run():
        dp = DataProvider(); df = await dp.fetch_15m("X:ADAUSD", from_, to)
        feats = build_features(df, SellerParams())
        res = run_backtest(feats, btParams())
        print(res["metrics"])
        res["trades"].to_csv("trades.csv", index=False)
    asyncio.run(_run())

if __name__ == "__main__":
    app()
```

---

## 14) DataProvider orchestrácia

```python
# data/provider.py
import pandas as pd
from data.polygon_client import PolygonClient

class DataProvider:
    def __init__(self):
        self.poly = PolygonClient()

    async def fetch_15m(self, ticker: str, from_: str, to: str) -> pd.DataFrame:
        bars = await self.poly.aggregates_15m(ticker, from_, to)
        df = pd.DataFrame([b.dict() for b in bars])
        df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        return df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
```

---

## 15) Parametre stratégie (default pre 15m)

- `ema_fast=96` (~1 deň na 15m)
- `ema_slow=672` (~7 dní)
- `z_window=672` (~7 dní)
- `vol_z=2.0`, `tr_z=1.2`, `cloc_min=0.6`
- `atr_window=96`, `atr_stop_mult=0.7`, `reward_r=2.0`, `max_hold=96`

---

## 16) Metodika backtestu

- Vstup **t+1 open** po signálovej sviečke.
- Stop = `low(signal) - atr_stop_mult*ATR(signal)`.
- TP = `entry + reward_r*(entry - stop)`.
- **Fees/slippage**: v bps (oboma smermi).
- Výstupy: `trades.csv`, metriky (N, win%, avg R, total pnl), equity curve (voliteľne).

---

## 17) Roadmap

**MVP (Týždeň 1)**
1. DataLayer (Polygon async client + provider, UTC handling, cache).
2. Strategy + Backtest + CLI.
3. UI skeleton (PySide6 + PyQtGraph, vykreslenie sviec + EMA/SMA, zvýraznenie signálov).

**Týždeň 2**
4. Paper‑trading plánovač (bar‑close), živé logy v UI.
5. Indikátory z Polygon Technical Indicators (porovnanie s lokálnymi výpočtami).
6. Param‑panel a export reportu (CSV/PNG/SVG z grafu).

**Nice‑to‑have**
- Walk‑forward (train/test split), grid‑search prahov.
- Monte‑Carlo re‑sample PnL.
- „Regime filter“ (napr. 15m vs 1h konfluencia).

---

## 18) Test plán

- **Unit**: z‑score, ATR, RSI, seller‑exhaustion maska (syntetické dáta).
- **Integration**: fetch posledných 3 dní 15m agregátov → non‑empty DataFrame; deterministické indexy (UTC).
- **Backtest determinism**: fix seed (ak sa použije), rovnaké výsledky.
- **UI smoke**: okno sa otvorí, graf sa vykreslí, žiadny block UI pri fetchi.

---

## 19) Inštalácia a spustenie

**Poetry**
```
poetry init -n
poetry add httpx pydantic pandas numpy pyqtgraph PySide6 qasync typer python-dotenv
poetry add -D pytest
```

**pyproject.toml** (výňatok)
```toml
[tool.poetry.dependencies]
python = "^3.10"
httpx = "^0.27"
pydantic = "^2.7"
pandas = "^2.2"
numpy = "^1.26"
pyqtgraph = "^0.13"
PySide6 = "^6.7"
qasync = "^0.27"
typer = "^0.12"
python-dotenv = "^1.0"
```

**Beh**
```
cp .env.example .env  # doplň POLYGON_API_KEY
poetry run python cli.py fetch --from 2023-01-01 --to 2025-10-13
poetry run python cli.py backtest --from 2023-01-01 --to 2025-10-13
poetry run python -m app.main
```

---

## 20) Kreatívne nápady (extras)
- **Signal Heatmap**: pre sliding windows ukázať pravdepodobnosť úspechu signálu (per‑hour, per‑weekday).
- **Volume regime**: adaptívne prahy vol_z podľa percentilu objemu pre danú dennú hodinu.
- **Alerty**: system tray notifikácia pri novom signále.
- **Screenshot export**: PNG/SVG grafu s vyznačenými vstupmi/výstupmi.
- **Notebook export**: auto‑generovaný Jupyter s parametrami a výsledkami backtestu.

---

## 21) Akceptačné kritériá
- Backtest zbehne na min. 1 roku 15m dát bez chýb, uloží `trades.csv` a vypíše metriky.
- UI zobrazí minimálne 5k sviec plynulo a bez výrazného lagovania (PyQtGraph).
- Paper‑trading odchytí **aspoň 1 reálny signál** počas behu (dá sa simulovať posunom času / načítaním historického „bar‑close“).

---

## 22) Poznámky k rizikám
- Polygon indikátory môžu mať odlišné definície (source price, rounding); pri porovnaní s lokálnymi výpočtami očakávaj malé rozdiely.
- Bez Binance zdroja môžu byť iné volumá; thresholdy kalibrovať na použitý feed.
- UI výkon: Vyhýbaj sa redraw celej scény – aktualizuj len najnovšie segmenty.

---

## 24) Prílohy
- Starter skript (pôvodná verzia) je k dispozícii v canvase tohto chatu, adaptovaný na 15m (pozri „starter.py“ obsah).
- Theme: Dark Forest UI (QSS) vyššie.

