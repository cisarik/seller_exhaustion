# AGENTS.md - AI Agent Guide for ADA Seller-Exhaustion Trading Agent

**Last Updated**: 2025-01-14  
**Project**: ADA 15-minute Seller-Exhaustion Trading Agent  
**Owner**: Michal  
**Python Version**: 3.10+ (tested on 3.13)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Module-by-Module Breakdown](#module-by-module-breakdown)
4. [Development Workflows](#development-workflows)
5. [Testing Guidelines](#testing-guidelines)
6. [Common Tasks & Examples](#common-tasks--examples)
7. [Code Patterns & Conventions](#code-patterns--conventions)
8. [Data Flow & Dependencies](#data-flow--dependencies)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Future Enhancements](#future-enhancements)

---

## Project Overview

### Purpose
Intraday trading research and backtesting system for Cardano (ADAUSD) on 15-minute timeframe. Detects "seller exhaustion" patterns (high volume, high volatility bottoms in downtrends) and provides:
- Historical data fetching from Polygon.io
- Technical indicator calculations (local, pandas-based)
- Event-driven backtesting with configurable parameters
- Interactive GUI with candlestick charts and overlays
- Paper trading framework (skeleton implemented)

### Tech Stack
- **Language**: Python 3.10+ (Poetry managed)
- **Async**: httpx for HTTP, qasync for Qt integration
- **Data**: pandas + numpy for time series
- **UI**: PySide6 (Qt) + PyQtGraph for charts
- **CLI**: Typer + Rich for terminal output
- **Testing**: pytest
- **API**: Polygon.io crypto aggregates (15-minute bars)

### Key Design Principles
1. **Deterministic**: Same inputs → same backtest results
2. **Async-first**: Non-blocking I/O for data fetching and UI
3. **Modular**: Clear separation between data/indicators/strategy/backtest/UI
4. **UTC everywhere**: All timestamps in UTC timezone
5. **Type-safe**: Pydantic models for configuration and data validation

---

## Architecture Deep Dive

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI / UI Layer                       │
│  cli.py (Typer commands)    app/main.py (PySide6 window)   │
└────────────────┬────────────────────────┬───────────────────┘
                 │                        │
                 ▼                        ▼
┌────────────────────────────┐  ┌──────────────────────────┐
│   Data Provider Layer      │  │   UI Widgets Layer       │
│   data/provider.py         │  │   app/widgets/           │
│   data/polygon_client.py   │  │   - candle_view.py       │
└────────────┬───────────────┘  └──────────────────────────┘
             │
             ▼
┌────────────────────────────┐
│   Strategy Layer           │
│   strategy/                │
│   - seller_exhaustion.py   │
│   ├─ SellerParams          │
│   └─ build_features()      │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐  ┌──────────────────────────┐
│   Indicators Layer         │  │   Backtest Engine        │
│   indicators/local.py      │  │   backtest/engine.py     │
│   - EMA, SMA, RSI, ATR     │  │   - Event-driven logic   │
│   - MACD, Z-score          │  │   - Entry/exit rules     │
└────────────────────────────┘  └──────────────────────────┘
```

### Directory Structure Explained

```
ada-agent/
├── app/                      # PySide6 UI application
│   ├── __init__.py
│   ├── main.py              # Main window, qasync event loop setup
│   ├── theme.py             # Dark Forest QSS stylesheet
│   └── widgets/
│       ├── __init__.py
│       └── candle_view.py   # PyQtGraph chart widget
│
├── backtest/                # Backtesting engine
│   ├── __init__.py
│   ├── engine.py           # Event-driven backtest logic
│   └── metrics.py          # Performance metrics calculations
│
├── config/                  # Configuration and settings
│   ├── __init__.py
│   └── settings.py         # Pydantic Settings (reads .env)
│
├── core/                    # Core utilities and models
│   ├── __init__.py
│   ├── models.py           # Pydantic models (Bar, Trade, Params)
│   └── timeutils.py        # UTC time utilities, 15m alignment
│
├── data/                    # Data fetching layer
│   ├── __init__.py
│   ├── polygon_client.py   # Async Polygon.io API client
│   └── provider.py         # High-level data provider interface
│
├── exec/                    # Paper trading execution (placeholder)
│   └── __init__.py         # Future: position management, scheduler
│
├── indicators/              # Technical indicators
│   ├── __init__.py
│   └── local.py            # Pandas-based TA calculations
│
├── strategy/                # Trading strategies
│   ├── __init__.py
│   └── seller_exhaustion.py # Seller exhaustion signal logic
│
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_backtest.py
│   ├── test_indicators.py
│   └── test_strategy.py
│
├── cli.py                   # Typer CLI with fetch/backtest/ui commands
├── pyproject.toml           # Poetry dependencies
├── Makefile                 # Convenience commands
├── README.md                # User documentation
├── QUICKSTART.md            # Quick reference
├── PRD.md                   # Product requirements document
├── .env.example             # Environment variables template
└── .gitignore
```

---

## Module-by-Module Breakdown

### 1. `config/settings.py`

**Purpose**: Centralized configuration management using pydantic-settings.

**Key Components**:
```python
class Settings(BaseSettings):
    polygon_api_key: str = ""  # From .env
    data_dir: str = ".data"    # Cache directory
    tz: str = "UTC"            # Timezone

settings = Settings()  # Global singleton
```

**Usage Pattern**:
```python
from config.settings import settings
api_key = settings.polygon_api_key
```

**Important**: 
- Reads from `.env` file automatically
- Falls back to defaults if .env missing
- Required: `POLYGON_API_KEY` must be set for data fetching

---

### 2. `core/models.py`

**Purpose**: Pydantic data models for type safety and validation.

**Key Models**:

1. **Bar**: Raw OHLCV data point
   ```python
   Bar(ts=1234567890000, open=0.5, high=0.52, low=0.48, close=0.51, volume=1000)
   ```

2. **BacktestParams**: Backtest configuration
   ```python
   BacktestParams(
       atr_stop_mult=0.7,  # Stop distance as multiple of ATR
       reward_r=2.0,        # Risk:Reward ratio (2R = 2:1)
       max_hold=96,         # Max bars to hold (96 = 24 hours on 15m)
       fee_bp=5.0,          # Fees in basis points
       slippage_bp=5.0      # Slippage in basis points
   )
   ```

**When to Modify**:
- Adding new data structures: Create new Pydantic models here
- Adding validation: Use Pydantic validators
- Type checking: Import models for type hints

---

### 3. `core/timeutils.py`

**Purpose**: Time manipulation utilities for 15-minute bar alignment.

**Key Functions**:

```python
utc_now() -> datetime
    # Returns current UTC time

align_to_15m(dt: datetime) -> datetime
    # Floor to nearest 15-minute boundary
    # 10:37:23 → 10:30:00

next_15m_boundary(dt: datetime) -> datetime
    # Next 15-minute boundary
    # 10:37:23 → 10:45:00

seconds_until_next_15m() -> float
    # Time until next bar close
```

**Critical for**:
- Paper trading scheduler (exec/ module)
- Real-time data refresh
- Bar alignment validation

---

### 4. `data/polygon_client.py`

**Purpose**: Async HTTP client for Polygon.io API.

**Architecture**:
```python
class PolygonClient:
    def __init__(self, api_key, timeout=20.0):
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def aggregates_15m(ticker, date_from, date_to) -> List[Bar]:
        # Handles pagination automatically
        # Rate limiting: 0.15s delay between pages
        # Returns list of Bar objects
    
    async def indicator_sma/ema(ticker, window, ...) -> Dict:
        # Optional: fetch indicators from Polygon
        # For comparison with local calculations
```

**Important Details**:
- **Pagination**: Handles `next_url` automatically
- **Rate limiting**: Built-in 0.15s delay between requests
- **Error handling**: Raises httpx exceptions (handle in caller)
- **Timeout**: 20s default, configurable

**API Endpoints Used**:
```
GET /v2/aggs/ticker/X:ADAUSD/range/15/minute/{from}/{to}
  → Returns: { results: [{t, o, h, l, c, v}, ...], next_url? }

GET /v1/indicators/{sma|ema|macd|rsi}/X:ADAUSD
  → Returns: Polygon-calculated indicators (optional, for validation)
```

**Gotchas**:
- Timestamps in milliseconds (converted to datetime in provider)
- Free tier rate limits: 5 API calls/minute
- Volume data: Not Binance-consolidated, may differ from exchange UIs

---

### 5. `data/provider.py`

**Purpose**: High-level data orchestration layer.

**Key Method**:
```python
async def fetch_15m(ticker: str, from_: str, to: str) -> pd.DataFrame:
    # 1. Fetch bars from Polygon
    bars = await self.poly.aggregates_15m(ticker, from_, to)
    
    # 2. Convert to DataFrame
    df = pd.DataFrame([b.model_dump() for b in bars])
    
    # 3. Process timestamps
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    
    # 4. Return OHLCV columns
    return df[["open", "high", "low", "close", "volume"]]
```

**Returns**: DataFrame with DatetimeIndex (UTC), sorted ascending.

**When to Extend**:
- Add caching: Implement disk/SQLite cache here
- Add fallback: Integrate yfinance as secondary source
- Add multiple tickers: Modify to return dict of DataFrames

---

### 6. `indicators/local.py`

**Purpose**: Technical indicator calculations using pandas.

**Why Local Calculations**:
1. Deterministic (same every time)
2. No API rate limits
3. Faster (no network I/O)
4. Full control over parameters

**Implemented Indicators**:

```python
# Exponential Moving Average
ema(series, span) -> pd.Series
    # Uses pandas .ewm(span=span, adjust=False)

# Simple Moving Average  
sma(series, window) -> pd.Series
    # Uses pandas .rolling(window).mean()

# Relative Strength Index
rsi(close, window=14) -> pd.Series
    # Classic RSI: 100 - (100 / (1 + RS))
    # Where RS = avg_gain / avg_loss

# Average True Range
atr(high, low, close, window=14) -> pd.Series
    # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    # ATR = rolling average of TR

# Moving Average Convergence Divergence
macd(close, fast=12, slow=26, signal=9) -> pd.DataFrame
    # Returns: {macd, signal, histogram}

# Z-Score (rolling)
zscore(series, window) -> pd.Series
    # (value - rolling_mean) / rolling_std
```

**Performance Notes**:
- All use pandas vectorized operations (fast)
- First `window` values will be NaN (expected)
- Use `.dropna()` before backtesting

**Adding New Indicators**:
1. Add function to `indicators/local.py`
2. Follow pattern: take Series/DataFrame, return Series/DataFrame
3. Add unit test to `tests/test_indicators.py`
4. Document expected behavior

---

### 7. `strategy/seller_exhaustion.py`

**Purpose**: Core strategy logic for detecting seller exhaustion signals.

**Key Components**:

```python
@dataclass
class SellerParams:
    ema_fast: int = 96      # ~1 day (15m bars)
    ema_slow: int = 672     # ~7 days
    z_window: int = 672     # Lookback for z-score
    vol_z: float = 2.0      # Volume z-score threshold
    tr_z: float = 1.2       # True range z-score threshold
    cloc_min: float = 0.6   # Min close location (0-1)
    atr_window: int = 96    # ATR calculation period
```

**Signal Logic** (in `build_features`):

```python
def build_features(df: pd.DataFrame, p: SellerParams) -> pd.DataFrame:
    # 1. Calculate EMAs
    df["ema_f"] = ema(df["close"], p.ema_fast)
    df["ema_s"] = ema(df["close"], p.ema_slow)
    
    # 2. Downtrend filter
    df["downtrend"] = df["ema_f"] < df["ema_s"]
    
    # 3. Calculate ATR
    df["atr"] = atr(df["high"], df["low"], df["close"], p.atr_window)
    
    # 4. Volume spike detection (z-score)
    df["vol_z"] = zscore(df["volume"], p.z_window)
    
    # 5. Range expansion detection (z-score)
    tr = df["atr"] * p.atr_window
    df["tr_z"] = zscore(tr, p.z_window)
    
    # 6. Close location in candle
    # 0 = at low, 1 = at high
    span = (df["high"] - df["low"]).replace(0, np.nan)
    df["cloc"] = (df["close"] - df["low"]) / span
    
    # 7. Generate signal
    df["exhaustion"] = (
        df["downtrend"] &           # In downtrend
        (df["vol_z"] > p.vol_z) &   # High volume
        (df["tr_z"] > p.tr_z) &     # Large range
        (df["cloc"] > p.cloc_min)   # Close near high
    )
    
    return df
```

**Signal Interpretation**:
- **Downtrend**: Price in consolidation/decline (EMA_fast < EMA_slow)
- **Volume spike**: Unusual selling pressure (z-score > 2.0)
- **Range expansion**: High volatility candle (z-score > 1.2)
- **Close near high**: Buyers stepped in (close in top 60% of range)

**Theory**: Seller exhaustion = heavy selling met with buying, potential bottom.

**Customization**:
- Adjust thresholds via SellerParams
- Add more filters (time of day, RSI < 30, etc.)
- Combine with other signals (MACD cross, etc.)

---

### 8. `backtest/engine.py`

**Purpose**: Event-driven backtest simulation.

**Architecture**:

```python
def run_backtest(df: pd.DataFrame, p: BacktestParams) -> dict:
    # State machine:
    # - in_pos = False: looking for signals
    # - in_pos = True: managing open position
    
    for each bar in df:
        if not in_pos and signal_detected:
            # Entry at next bar open
            entry = next_bar.open
            stop = signal_bar.low - atr_stop_mult * atr
            tp = entry + reward_r * (entry - stop)
            in_pos = True
        
        elif in_pos:
            # Check exit conditions (order matters!):
            if open <= stop:  exit_price = open, reason = "stop_gap"
            elif low <= stop: exit_price = stop, reason = "stop"
            elif high >= tp:  exit_price = tp, reason = "tp"
            elif bars >= max_hold: exit_price = close, reason = "time"
            
            if exited:
                calculate_pnl_with_fees()
                log_trade()
                in_pos = False
```

**Key Features**:
1. **Entry**: t+1 open (no lookahead bias)
2. **Stop**: Signal low - ATR_mult × ATR
3. **TP**: Risk-adjusted (entry + R × risk)
4. **Max Hold**: Time-based exit (safety)
5. **Fees/Slippage**: Deducted from both sides

**Exit Priority**:
1. Gap down through stop (open)
2. Stop hit (intrabar)
3. TP hit (intrabar)
4. Max hold exceeded (close)

**Output Structure**:
```python
{
    "trades": pd.DataFrame([
        {
            "entry_ts": "2024-01-15 10:30:00+00:00",
            "exit_ts": "2024-01-15 14:45:00+00:00",
            "entry": 0.5123,
            "exit": 0.5234,
            "stop": 0.5050,
            "tp": 0.5269,
            "pnl": 0.0091,
            "R": 1.52,
            "reason": "tp",
            "bars_held": 17
        },
        ...
    ]),
    "metrics": {
        "n": 45,
        "win_rate": 0.56,
        "avg_R": 0.42,
        "total_pnl": 0.1234,
        "max_dd": -0.0456,
        "sharpe": 0.89
    }
}
```

**Important Notes**:
- Assumes single contract (no position sizing)
- No compounding (fixed capital)
- No partial exits
- Deterministic (given same inputs)

---

### 9. `backtest/metrics.py`

**Purpose**: Calculate performance metrics from trades DataFrame.

**Key Metrics**:

```python
calculate_metrics(trades: pd.DataFrame) -> dict:
    return {
        "total_trades": int,
        "win_rate": float,        # % trades with pnl > 0
        "avg_R": float,           # Average R-multiple
        "avg_win": float,         # Average winning trade $
        "avg_loss": float,        # Average losing trade $
        "profit_factor": float,   # Total wins / Total losses
        "total_pnl": float,       # Sum of all PnL
        "max_drawdown": float,    # Worst equity drop
        "sharpe": float           # Returns / StdDev (approx)
    }
```

**Usage**:
```python
from backtest.metrics import calculate_metrics, print_metrics

result = run_backtest(feats, params)
metrics = calculate_metrics(result["trades"])
print_metrics(metrics)  # Pretty print to console
```

---

### 10. `app/main.py`

**Purpose**: PySide6 main window with qasync event loop.

**Key Architecture**:

```python
class MainWindow(QMainWindow):
    def __init__(self):
        # Create tabs
        self.tabs = QTabWidget()
        self.chart_view = CandleChartWidget()
        self.tabs.addTab(self.chart_view, "Chart")
    
    async def initialize(self):
        # Called after window shown
        await self.chart_view.load_initial()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_FOREST_QSS)  # Apply theme
    
    window = MainWindow()
    window.show()
    
    if HAS_QASYNC:
        # Use qasync for proper async integration
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        with loop:
            loop.create_task(window.initialize())
            loop.run_forever()
    else:
        # Fallback (limited async support)
        sys.exit(app.exec())
```

**Important**:
- **qasync is required** for async data fetching in UI
- Without it, UI will block during network calls
- Event loop runs in main thread (Qt requirement)

---

### 11. `app/widgets/candle_view.py`

**Purpose**: PyQtGraph-based candlestick chart with overlays.

**Key Components**:

```python
class CandlestickItem(pg.GraphicsObject):
    # Custom item for drawing OHLC candles
    # Green = close > open
    # Red = close < open
    
class CandleChartWidget(QWidget):
    async def load_initial(self):
        # 1. Fetch data
        df = await self.dp.fetch_15m(...)
        
        # 2. Build features
        feats = build_features(df, self.params)
        
        # 3. Render
        self.render_candles(feats)
    
    def render_candles(self, df: pd.DataFrame):
        # 1. Sample if > 5000 candles (performance)
        # 2. Create CandlestickItem
        # 3. Add EMA overlays
        # 4. Mark signal points
        # 5. Add legend
```

**Performance Optimizations**:
- Sample to last 5000 candles if dataset is large
- Use PyQtGraph's fast rendering (OpenGL backend)
- Update incrementally (don't redraw all on refresh)

**Visual Elements**:
- Candlesticks: Green/red based on direction
- EMA Fast (96): Cyan line
- EMA Slow (672): Orange line
- Signals: Yellow triangles at exhaustion bars

---

### 12. `cli.py`

**Purpose**: Typer-based CLI with three commands.

**Commands**:

1. **fetch**: Download and display data
   ```bash
   poetry run python cli.py fetch --from 2024-01-01 --to 2025-01-13
   ```

2. **backtest**: Run strategy backtest
   ```bash
   poetry run python cli.py backtest \
     --from 2024-01-01 \
     --to 2025-01-13 \
     --ema-fast 96 \
     --vol-z 2.0 \
     --reward-r 2.0 \
     --output trades.csv
   ```

3. **ui**: Launch PySide6 GUI
   ```bash
   poetry run python cli.py ui
   ```

**Error Handling**:
- All async operations wrapped in try/except
- Errors printed with Rich formatting
- Stack traces shown in verbose mode

---

## Development Workflows

### Setting Up Development Environment

```bash
# 1. Clone/navigate to project
cd /home/agile/a0

# 2. Install dependencies
poetry install

# 3. Configure environment
cp .env.example .env
nano .env  # Add POLYGON_API_KEY

# 4. Run tests
poetry run pytest tests/ -v

# 5. Verify CLI
poetry run python cli.py fetch --from 2024-12-01 --to 2024-12-31
```

### Adding a New Indicator

```bash
# 1. Implement in indicators/local.py
def my_indicator(close: pd.Series, param: int) -> pd.Series:
    """Calculate custom indicator."""
    return close.rolling(param).apply(lambda x: custom_logic(x))

# 2. Add test in tests/test_indicators.py
def test_my_indicator():
    data = pd.Series([1, 2, 3, 4, 5])
    result = my_indicator(data, param=3)
    assert result.iloc[-1] == expected_value

# 3. Use in strategy
from indicators.local import my_indicator
df["my_ind"] = my_indicator(df["close"], param=10)

# 4. Run tests
poetry run pytest tests/test_indicators.py::test_my_indicator -v
```

### Modifying Strategy Logic

```bash
# 1. Edit strategy/seller_exhaustion.py
# Add new condition to build_features():
df["new_filter"] = some_condition
df["exhaustion"] = df["exhaustion"] & df["new_filter"]

# 2. Update SellerParams if needed
@dataclass
class SellerParams:
    # ... existing params
    new_threshold: float = 1.5

# 3. Add test
def test_new_filter():
    params = SellerParams(new_threshold=2.0)
    df = build_features(test_data, params)
    assert df["new_filter"].sum() > 0

# 4. Run backtest to validate
poetry run python cli.py backtest --from 2024-01-01 --to 2024-12-31
```

### Adding a UI Widget

```bash
# 1. Create widget file
# app/widgets/metrics_panel.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

class MetricsPanel(QWidget):
    def __init__(self):
        super().__init__()
        # ... widget code

# 2. Add to main window
# app/main.py
from app.widgets.metrics_panel import MetricsPanel

self.metrics = MetricsPanel()
self.tabs.addTab(self.metrics, "Metrics")

# 3. Test
poetry run python cli.py ui
```

### Debugging Backtest Issues

```bash
# 1. Add print statements in engine.py
print(f"Signal at {t}: entry={entry}, stop={stop}, tp={tp}")

# 2. Export features DataFrame
feats.to_csv("debug_features.csv")

# 3. Check specific trade
trades[trades["reason"] == "stop"].head()

# 4. Visualize in UI
# Signals will be marked on chart
poetry run python cli.py ui
```

---

## Testing Guidelines

### Test Structure

```
tests/
├── test_indicators.py    # Unit tests for each indicator
├── test_strategy.py      # Strategy logic and feature building
└── test_backtest.py      # Backtest engine and determinism
```

### Running Tests

```bash
# All tests
poetry run pytest tests/ -v

# Specific file
poetry run pytest tests/test_strategy.py -v

# Specific test
poetry run pytest tests/test_strategy.py::test_seller_params_defaults -v

# With coverage
poetry run pytest tests/ --cov=. --cov-report=html
```

### Writing Tests

**Pattern for Indicators**:
```python
def test_my_indicator():
    # 1. Create synthetic data
    s = pd.Series([1, 2, 3, 4, 5])
    
    # 2. Calculate indicator
    result = my_indicator(s, window=3)
    
    # 3. Assert properties
    assert len(result) == len(s)
    assert result.iloc[2] == expected_value
    assert result.iloc[:2].isna().all()  # First window should be NaN
```

**Pattern for Strategy**:
```python
def test_signal_conditions():
    # 1. Create scenario data
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    df = pd.DataFrame({
        'open': [...],
        'high': [...],
        'low': [...],
        'close': [...],
        'volume': [...]
    }, index=dates)
    
    # 2. Build features
    params = SellerParams(...)
    result = build_features(df, params)
    
    # 3. Assert signals
    assert 'exhaustion' in result.columns
    assert result['exhaustion'].sum() > 0
```

**Pattern for Backtest**:
```python
def test_backtest_scenario():
    # 1. Create controlled scenario
    df = create_winning_trade_scenario()
    
    # 2. Run backtest
    params = BacktestParams()
    result = run_backtest(df, params)
    
    # 3. Assert outcomes
    assert result['metrics']['n'] == 1
    assert result['trades'].iloc[0]['reason'] == 'tp'
    assert result['trades'].iloc[0]['pnl'] > 0
```

### Test Data Helpers

```python
# tests/conftest.py (create this for shared fixtures)
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    np.random.seed(42)
    return pd.DataFrame({
        'open': 0.5 + np.random.randn(1000) * 0.01,
        'high': 0.52 + np.random.randn(1000) * 0.01,
        'low': 0.48 + np.random.randn(1000) * 0.01,
        'close': 0.5 + np.random.randn(1000) * 0.01,
        'volume': 1000 + np.random.randn(1000) * 100,
    }, index=dates)
```

---

## Common Tasks & Examples

### Task 1: Fetch Recent Data

```python
import asyncio
from data.provider import DataProvider

async def fetch_recent():
    dp = DataProvider()
    df = await dp.fetch_15m("X:ADAUSD", "2024-12-01", "2024-12-31")
    print(f"Fetched {len(df)} bars")
    print(df.head())
    await dp.close()

asyncio.run(fetch_recent())
```

### Task 2: Calculate Indicators on Custom Data

```python
import pandas as pd
from indicators.local import ema, rsi, atr

# Load your data
df = pd.read_csv("my_data.csv", index_col=0, parse_dates=True)

# Calculate indicators
df["ema_20"] = ema(df["close"], 20)
df["rsi_14"] = rsi(df["close"], 14)
df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)

# Save with indicators
df.to_csv("data_with_indicators.csv")
```

### Task 3: Run Parameter Sweep

```python
import asyncio
from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from backtest.engine import run_backtest, BacktestParams

async def parameter_sweep():
    dp = DataProvider()
    df = await dp.fetch_15m("X:ADAUSD", "2024-01-01", "2024-12-31")
    
    results = []
    for vol_z in [1.5, 2.0, 2.5]:
        for tr_z in [1.0, 1.2, 1.5]:
            params = SellerParams(vol_z=vol_z, tr_z=tr_z)
            feats = build_features(df, params)
            
            bt_params = BacktestParams()
            result = run_backtest(feats, bt_params)
            
            results.append({
                "vol_z": vol_z,
                "tr_z": tr_z,
                **result["metrics"]
            })
    
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv("parameter_sweep.csv", index=False)
    
    await dp.close()

asyncio.run(parameter_sweep())
```

### Task 4: Export Chart as Image

```python
# In app/widgets/candle_view.py, add method:
def export_chart(self, filename="chart.png"):
    """Export chart to PNG."""
    exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
    exporter.export(filename)
    print(f"Chart saved to {filename}")

# Usage in UI:
# Add button that calls: self.chart_view.export_chart()
```

### Task 5: Live Paper Trading (Skeleton)

```python
# exec/scheduler.py
import asyncio
from datetime import datetime
from core.timeutils import seconds_until_next_15m
from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features

async def live_trading_loop():
    dp = DataProvider()
    params = SellerParams()
    
    while True:
        # Wait until next bar close
        wait_time = seconds_until_next_15m()
        print(f"Waiting {wait_time:.0f}s until next bar close...")
        await asyncio.sleep(wait_time)
        
        # Fetch recent data (last 1000 bars for indicator calculation)
        now = datetime.utcnow()
        from_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")
        
        df = await dp.fetch_15m("X:ADAUSD", from_date, to_date)
        feats = build_features(df, params)
        
        # Check latest bar for signal
        latest = feats.iloc[-1]
        if latest["exhaustion"]:
            print(f"🚨 SIGNAL at {latest.name}")
            # TODO: Execute paper trade
            # - Record entry at next open (simulated)
            # - Set stop/TP levels
            # - Track position
        
        # Check existing positions
        # TODO: Check if stop/TP hit
        
        await asyncio.sleep(1)  # Small delay before next cycle

# Run in background task:
asyncio.create_task(live_trading_loop())
```

---

## Code Patterns & Conventions

### 1. Async Functions

**Pattern**:
```python
async def fetch_something(...) -> ReturnType:
    """Docstring explaining what it does."""
    # Use async/await for I/O operations
    result = await async_operation()
    return result
```

**When to use**:
- Network I/O (httpx, aiohttp)
- File I/O (aiofiles)
- Database operations
- UI operations with qasync

### 2. DataFrame Operations

**Pattern**:
```python
def process_dataframe(df: pd.DataFrame, ...) -> pd.DataFrame:
    """Process DataFrame and return modified copy."""
    out = df.copy()  # Don't modify input
    out["new_col"] = calculation(out["existing_col"])
    return out
```

**Best practices**:
- Always copy input DataFrames
- Use vectorized operations (avoid loops)
- Handle NaN values explicitly
- Use `.loc[]` for assignments to avoid warnings

### 3. Pydantic Models

**Pattern**:
```python
from pydantic import BaseModel, Field, validator

class MyModel(BaseModel):
    field1: str = Field(..., description="Required field")
    field2: int = Field(default=10, ge=0, le=100)
    
    @validator("field1")
    def validate_field1(cls, v):
        if not v.startswith("X:"):
            raise ValueError("Must start with X:")
        return v
```

### 4. Error Handling

**Pattern**:
```python
try:
    result = await risky_operation()
except httpx.HTTPError as e:
    print(f"HTTP error: {e}")
    raise
except Exception as e:
    print(f"Unexpected error: {e}")
    # Decide: re-raise, return default, or exit
```

**Guidelines**:
- Catch specific exceptions first
- Log errors with context
- Re-raise if caller should handle
- Use Rich for pretty error messages in CLI

### 5. Type Hints

**Use everywhere**:
```python
def calculate_something(
    data: pd.DataFrame,
    param1: float,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    ...
```

**Benefits**:
- Better IDE autocomplete
- Catches errors early
- Self-documenting code

### 6. Docstrings

**Format**:
```python
def function(arg1: type, arg2: type) -> return_type:
    """
    One-line summary.
    
    Detailed explanation if needed.
    Multiple paragraphs OK.
    
    Args:
        arg1: Description
        arg2: Description
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When it's raised
    """
```

---

## Data Flow & Dependencies

### Dependency Graph

```
settings.py (config)
    ↓
polygon_client.py (data layer)
    ↓
provider.py (data orchestration)
    ↓
local.py (indicators)
    ↓
seller_exhaustion.py (strategy)
    ↓
engine.py (backtest)
    ↓
cli.py / main.py (presentation)
```

### Data Flow: Backtest

```
1. CLI/UI triggers backtest
   ↓
2. provider.fetch_15m() → DataFrame
   ↓
3. build_features(df, params) → DataFrame with signals
   ↓
4. run_backtest(feats, bt_params) → {trades, metrics}
   ↓
5. Display/export results
```

### Data Flow: Live Trading (Future)

```
1. Scheduler waits for bar close
   ↓
2. Fetch recent bars (last 7 days)
   ↓
3. build_features() on full history
   ↓
4. Check latest bar for signal
   ↓
5. If signal: log paper trade (entry at next open)
   ↓
6. Check existing positions for stop/TP
   ↓
7. Sleep until next bar close
```

---

## Troubleshooting Guide

### Issue: ImportError for modules

**Symptom**:
```
ModuleNotFoundError: No module named 'app'
```

**Solution**:
```bash
# Use Poetry's virtual environment
poetry shell
python cli.py backtest

# Or prefix with poetry run
poetry run python cli.py backtest
```

### Issue: Polygon API returns empty results

**Symptoms**:
- `df` is empty
- 0 bars fetched

**Checks**:
1. Verify API key in `.env`: `echo $POLYGON_API_KEY`
2. Check date range: must be YYYY-MM-DD format
3. Check ticker: should be `X:ADAUSD` (not `ADA-USD`)
4. Check API quota: free tier has limits

**Debug**:
```python
# Add to polygon_client.py
print(f"URL: {url}")
print(f"Params: {params}")
print(f"Response: {data}")
```

### Issue: UI freezes during data fetch

**Symptom**: Window becomes unresponsive

**Cause**: Not using qasync or async/await properly

**Solution**:
```bash
# Install qasync
poetry add qasync

# Verify in app/main.py
if HAS_QASYNC:
    # Should enter this branch
```

### Issue: Tests fail with NaN errors

**Symptom**:
```
AssertionError: assert nan == expected_value
```

**Cause**: Indicators need warmup period

**Solution**:
```python
# In tests, drop NaN values
result = result.dropna()
assert result.iloc[0] == expected_value

# Or skip warmup period
assert result.iloc[window:].notna().all()
```

### Issue: Backtest returns 0 trades

**Checks**:
1. Are there exhaustion signals? `feats['exhaustion'].sum()`
2. Check NaN values: `feats['atr'].isna().sum()`
3. Verify date range has enough data (need > 672 bars for default params)

**Debug**:
```python
feats.to_csv("debug_features.csv")
# Inspect in Excel/pandas
# Look for exhaustion=True rows
```

### Issue: Different backtest results each run

**Symptom**: Metrics change between runs

**Cause**: Non-determinism (e.g., random seed, dict ordering)

**Check**:
1. No random operations in code
2. DataFrames sorted consistently: `df.sort_index()`
3. No dependency on current time
4. No parallel operations with race conditions

**Verify**:
```python
# Run twice, compare
result1 = run_backtest(feats, params)
result2 = run_backtest(feats, params)
assert result1["metrics"] == result2["metrics"]
```

---

## Future Enhancements

From PRD and project roadmap:

### Week 2 Enhancements

1. **Paper Trading Scheduler** (`exec/scheduler.py`)
   - Implement bar-close event loop
   - Real-time signal detection
   - Position tracking in memory
   - Live logs in UI

2. **Polygon Technical Indicators** (`data/polygon_client.py`)
   - Fetch SMA/EMA/RSI/MACD from API
   - Compare with local calculations
   - Validate accuracy

3. **Parameter Panel** (`app/widgets/params_panel.py`)
   - UI for adjusting strategy params
   - Re-run backtest button
   - Save/load presets

4. **Enhanced Reporting** (`backtest/report.py`)
   - Equity curve plot
   - Drawdown chart
   - Trade distribution histogram
   - Export to PDF/HTML

### Nice-to-Have

1. **Walk-Forward Analysis** (`backtest/walk_forward.py`)
   - Train/test split
   - Rolling window optimization
   - Out-of-sample validation

2. **Monte Carlo Simulation** (`backtest/monte_carlo.py`)
   - Resample trade sequences
   - Confidence intervals
   - Worst-case scenarios

3. **Multi-Timeframe** (`strategy/confluence.py`)
   - 15m + 1h signal confluence
   - Higher TF trend filter
   - Adaptive parameters

4. **Signal Heatmap** (`app/widgets/heatmap.py`)
   - Win rate by hour of day
   - Win rate by day of week
   - Seasonal patterns

5. **Real Broker Integration** (`exec/binance.py`, `exec/kraken.py`)
   - Live order execution
   - Portfolio management
   - Risk limits

### Code Quality

1. **Type Checking**
   ```bash
   poetry add --dev mypy
   poetry run mypy .
   ```

2. **Pre-commit Hooks**
   ```bash
   poetry add --dev pre-commit
   # .pre-commit-config.yaml
   ```

3. **Documentation**
   ```bash
   poetry add --dev sphinx
   # Auto-generate API docs
   ```

4. **CI/CD**
   ```yaml
   # .github/workflows/tests.yml
   - Run pytest on push
   - Check coverage
   - Lint with ruff
   ```

---

## Performance Optimization Tips

### 1. DataFrame Operations

**Slow**:
```python
for i in range(len(df)):
    df.loc[i, "new_col"] = calculation(df.loc[i, "value"])
```

**Fast**:
```python
df["new_col"] = df["value"].apply(calculation)  # Better
df["new_col"] = vectorized_calculation(df["value"])  # Best
```

### 2. Indicator Calculations

- Use pandas built-in methods (`.rolling()`, `.ewm()`)
- Avoid Python loops
- Calculate once, reuse results

### 3. UI Rendering

- Sample large datasets (> 5000 candles)
- Use PyQtGraph's OpenGL backend
- Update incrementally, not full redraw

### 4. Data Fetching

- Cache results to disk/SQLite
- Fetch only necessary date range
- Use pagination wisely (don't fetch all if not needed)

---

## Key Files Quick Reference

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `cli.py` | CLI entry point | `fetch()`, `backtest()`, `ui()` |
| `config/settings.py` | Configuration | `Settings`, `settings` |
| `data/polygon_client.py` | API client | `PolygonClient.aggregates_15m()` |
| `data/provider.py` | Data orchestration | `DataProvider.fetch_15m()` |
| `indicators/local.py` | TA indicators | `ema()`, `sma()`, `rsi()`, `atr()`, `macd()` |
| `strategy/seller_exhaustion.py` | Strategy logic | `SellerParams`, `build_features()` |
| `backtest/engine.py` | Backtest simulation | `run_backtest()` |
| `backtest/metrics.py` | Performance metrics | `calculate_metrics()` |
| `app/main.py` | UI main window | `MainWindow`, `main()` |
| `app/widgets/candle_view.py` | Chart widget | `CandleChartWidget` |
| `app/theme.py` | UI styling | `DARK_FOREST_QSS` |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POLYGON_API_KEY` | Yes | - | Polygon.io API key (get from polygon.io) |
| `DATA_DIR` | No | `.data` | Directory for cached data (future) |
| `TZ` | No | `UTC` | Timezone (always UTC, don't change) |

---

## Common Commands Cheat Sheet

```bash
# Development
poetry install                    # Install dependencies
poetry shell                      # Activate virtual environment
poetry add <package>              # Add new dependency
poetry add --dev <package>        # Add dev dependency

# Testing
make test                         # Run all tests
poetry run pytest tests/ -v       # Verbose test output
poetry run pytest tests/test_strategy.py::test_name -v  # Single test
poetry run pytest --cov=. --cov-report=html  # Coverage report

# Running
make fetch                        # Fetch sample data
make backtest                     # Run sample backtest
make ui                           # Launch GUI

# CLI
poetry run python cli.py fetch --from 2024-01-01 --to 2024-12-31
poetry run python cli.py backtest --ema-fast 96 --vol-z 2.0 --output trades.csv
poetry run python cli.py ui

# Linting
make lint                         # Run ruff
poetry run ruff check .           # Check code style
poetry run ruff format .          # Format code

# Cleanup
make clean                        # Remove generated files
```

---

## Contact & Support

**Project Owner**: Michal  
**Repository**: /home/agile/a0  
**Documentation**: README.md, QUICKSTART.md, PRD.md, AGENTS.md (this file)

**For Future Agents**:
- Read PRD.md first (requirements)
- Check QUICKSTART.md for usage
- Refer to this file for deep dive
- Run tests before making changes
- Keep this file updated with new patterns/decisions

**Philosophy**:
- Deterministic & testable
- Async-first for I/O
- Type-safe with Pydantic
- Clean separation of concerns
- Document as you go

---

**Last Updated**: 2025-01-14  
**Version**: 1.0 (MVP completed)  
**Next Milestone**: Week 2 enhancements (paper trading scheduler, parameter UI)
