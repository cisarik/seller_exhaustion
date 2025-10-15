# PRD — Live Trading Agent for Seller-Exhaustion Strategy

**Product:** Real-time trading execution agent for automated strategy deployment  
**Purpose:** Execute seller-exhaustion strategy on live markets with full risk management  
**Tech:** Python 3.10+, PySide6 UI, CCXT exchange integration, WebSocket real-time data  
**Deployment:** VPS (Linux recommended), 24/7 uptime, low-latency connection  
**Owner:** Michal  
**Version:** 1.0.0 (Trading Agent)  
**Goal:** Production-ready paper trading → Live trading capability

---

## 🎯 Executive Summary

### What This Application Does

**NOT** a backtesting tool (that's the separate application you already have).  
**IS** a live trading execution engine that:

1. **Loads** strategy parameters from JSON config file (exported from backtest app)
2. **Connects** to cryptocurrency exchange (Binance/Kraken/etc.) via API
3. **Monitors** real-time price data via WebSocket or REST API
4. **Calculates** indicators and strategy signals in real-time
5. **Executes** trades automatically when signals occur
6. **Manages** open positions with Fibonacci exit levels
7. **Tracks** balance, PnL, and performance metrics
8. **Displays** real-time candlestick chart and position status
9. **Logs** all activity for audit and debugging

### Core Principle

**SIMPLICITY + SAFETY + RELIABILITY**

- Simple UI (no complex backtesting tools)
- Safety-first (paper trading default, multiple fail-safes)
- Reliable (handles network failures, reconnections, edge cases)

### Use Cases

1. **Paper Trading** (Phase 1): Test strategy on live data without risking money
2. **Testnet Trading** (Phase 2): Test with testnet/sandbox exchange
3. **Live Trading** (Phase 3): Real money execution with full risk management

---

## 📋 System Requirements

### Hardware Requirements

**Minimum**:
- CPU: 2 cores, 2.0 GHz
- RAM: 2 GB
- Storage: 10 GB available
- Network: Stable internet (100 KB/s+)

**Recommended (VPS)**:
- CPU: 4 cores, 2.5 GHz+
- RAM: 4 GB
- Storage: 20 GB SSD
- Network: Low-latency (<100ms to exchange), stable
- Location: Close to exchange servers (AWS us-east-1 for Binance US, etc.)

### Software Requirements

- **OS**: Linux (Ubuntu 22.04+ recommended), Windows 10+, or macOS 12+
- **Python**: 3.10 or 3.11 (tested on 3.13)
- **Dependencies**: PySide6, CCXT, pandas, numpy, websockets
- **Optional**: systemd (for Linux service), supervisord

### Exchange Requirements

**Supported Exchanges** (via CCXT):
- Binance (primary)
- Kraken
- Coinbase Pro
- Bybit
- OKX
- Others (CCXT supports 100+ exchanges)

**API Requirements**:
- API key + secret with trading permissions
- Testnet/sandbox access recommended for testing
- Rate limits: Must support REST + WebSocket

---

## 🏗️ Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   TRADING AGENT APPLICATION                  │
│                      (Separate Project)                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         UI LAYER                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Main Window (PySide6)                                 │  │
│  │  ├─ Real-time Candlestick Chart (PyQtGraph)           │  │
│  │  ├─ Position Panel (entries, exits, PnL)              │  │
│  │  ├─ Stats Dashboard (balance, win rate, drawdown)     │  │
│  │  ├─ Control Panel (start/stop, emergency close)       │  │
│  │  └─ Log Viewer (live activity feed)                   │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRADING ENGINE LAYER                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Strategy Manager                                      │  │
│  │  ├─ Load config.json (parameters from backtest app)   │  │
│  │  ├─ Calculate indicators (EMA, ATR, z-score, etc.)    │  │
│  │  ├─ Detect seller exhaustion signals                  │  │
│  │  ├─ Calculate Fibonacci exit levels                   │  │
│  │  └─ Manage position state machine                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Risk Manager                                          │  │
│  │  ├─ Position sizing (% of account)                    │  │
│  │  ├─ Max daily loss enforcement                        │  │
│  │  ├─ Max open positions limit                          │  │
│  │  ├─ Slippage tolerance check                          │  │
│  │  └─ Emergency stop (kill switch)                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Order Manager                                         │  │
│  │  ├─ Market order execution                            │  │
│  │  ├─ Limit order execution                             │  │
│  │  ├─ Order status tracking                             │  │
│  │  ├─ Fill confirmation                                 │  │
│  │  └─ Order timeout/retry logic                         │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Data Feed Manager                                     │  │
│  │  ├─ WebSocket real-time candles (primary)            │  │
│  │  ├─ REST API polling (fallback)                       │  │
│  │  ├─ Candle completion detection                       │  │
│  │  ├─ Data validation (OHLCV integrity)                │  │
│  │  └─ Gap detection and alerts                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Exchange Connector (CCXT)                             │  │
│  │  ├─ Exchange API wrapper                              │  │
│  │  ├─ Rate limiting                                     │  │
│  │  ├─ Connection health monitoring                      │  │
│  │  ├─ Automatic reconnection                            │  │
│  │  └─ Error handling and retries                        │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   PERSISTENCE LAYER                          │
│  ├─ config.json (loaded from backtest app export)           │
│  ├─ trades.db (SQLite trade history)                        │
│  ├─ balance.db (balance snapshots)                          │
│  ├─ logs/ (timestamped log files)                           │
│  └─ state.json (resume after restart)                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   EXTERNAL SERVICES                          │
│  ├─ Binance/Kraken/etc. API (REST + WebSocket)             │
│  ├─ Optional: Telegram bot (alerts)                         │
│  └─ Optional: Polygon.io (data backup)                      │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
ada-trading-agent/          # NEW SEPARATE PROJECT
├── agent/
│   ├── __init__.py
│   ├── main.py            # Application entry point
│   ├── engine.py          # Core trading engine
│   ├── strategy.py        # Strategy signal detection (mirrors backtest)
│   ├── risk_manager.py    # Risk management and position sizing
│   ├── order_manager.py   # Order execution and tracking
│   └── state_machine.py   # Position state management
│
├── data/
│   ├── __init__.py
│   ├── exchange.py        # CCXT exchange wrapper
│   ├── websocket.py       # WebSocket data feed
│   ├── candle_builder.py  # Build OHLCV candles from ticks
│   └── validator.py       # Data integrity validation
│
├── indicators/
│   ├── __init__.py
│   └── realtime.py        # Indicator calculations (copy from backtest app)
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py     # Main PySide6 window
│   ├── chart_widget.py    # Real-time candlestick chart
│   ├── position_panel.py  # Open positions display
│   ├── stats_panel.py     # Performance stats
│   ├── control_panel.py   # Start/stop controls
│   ├── log_widget.py      # Live log viewer
│   └── theme.py           # Dark Forest theme (copy from backtest app)
│
├── config/
│   ├── __init__.py
│   ├── loader.py          # Load strategy config from JSON
│   └── validator.py       # Validate config before trading
│
├── persistence/
│   ├── __init__.py
│   ├── trades_db.py       # SQLite trade history
│   ├── balance_db.py      # Balance tracking
│   └── state_manager.py   # Save/resume state
│
├── monitoring/
│   ├── __init__.py
│   ├── logger.py          # Structured logging
│   ├── alerts.py          # Telegram/email alerts
│   └── health_check.py    # Connection health monitoring
│
├── tests/
│   ├── test_strategy.py
│   ├── test_risk_manager.py
│   ├── test_order_manager.py
│   └── test_exchange.py
│
├── config.json            # Strategy configuration (imported from backtest app)
├── .env                   # Exchange credentials (NEVER commit!)
├── pyproject.toml         # Poetry dependencies
├── README.md
├── LICENSE
└── .gitignore
```

---

## 📦 Parameter File Format (config.json)

### Complete Specification

The trading agent loads ALL configuration from a single `config.json` file exported by the backtesting application.

**File Location**: `./config.json` (same directory as agent executable)

**Format**: JSON (validated against TradingConfig schema)

**Example**:
```json
{
  "version": "2.1.0",
  "created_at": "2025-01-15T10:30:00Z",
  "description": "Optimized 15m seller exhaustion strategy with 61.8% Fibonacci exits",
  "strategy_name": "Seller Exhaustion",
  
  "timeframe": "15m",
  
  "seller_params": {
    "ema_fast": 96,
    "ema_slow": 672,
    "z_window": 672,
    "atr_window": 96,
    "ema_fast_minutes": null,
    "ema_slow_minutes": null,
    "z_window_minutes": null,
    "atr_window_minutes": null,
    "vol_z": 2.0,
    "tr_z": 1.2,
    "cloc_min": 0.6
  },
  
  "backtest_params": {
    "use_stop_loss": false,
    "use_time_exit": false,
    "use_fib_exits": true,
    "use_traditional_tp": false,
    "atr_stop_mult": 0.7,
    "reward_r": 2.0,
    "max_hold": 96,
    "fib_swing_lookback": 96,
    "fib_swing_lookahead": 5,
    "fib_target_level": 0.618,
    "fee_bp": 5.0,
    "slippage_bp": 5.0
  },
  
  "risk_management": {
    "risk_per_trade_percent": 1.0,
    "max_position_size_percent": 10.0,
    "max_daily_loss_percent": 5.0,
    "max_daily_trades": 10,
    "max_open_positions": 1,
    "slippage_tolerance_percent": 0.5,
    "order_timeout_seconds": 30
  },
  
  "exchange": {
    "exchange_name": "binance",
    "trading_pair": "ADA/USDT",
    "base_currency": "ADA",
    "quote_currency": "USDT",
    "api_key": "YOUR_API_KEY_HERE",
    "api_secret": "YOUR_API_SECRET_HERE",
    "api_passphrase": null,
    "testnet": true,
    "enable_rate_limit": true,
    "paper_trading": true,
    "paper_initial_balance": 10000.0
  },
  
  "data_feed": {
    "data_source": "exchange",
    "polygon_api_key": "YOUR_POLYGON_KEY_HERE",
    "use_websocket": true,
    "websocket_ping_interval": 20,
    "rest_api_interval_seconds": 60,
    "max_missing_bars": 3,
    "validate_ohlcv": true
  },
  
  "backtest_metrics": {
    "total_trades": 45,
    "win_rate": 0.56,
    "avg_R": 0.42,
    "total_pnl": 0.1234,
    "max_drawdown": -0.0456,
    "sharpe": 0.89
  }
}
```

### Parameter Definitions

#### **seller_params** (Strategy Entry Logic)

| Parameter | Type | Default | Description | Critical |
|-----------|------|---------|-------------|----------|
| `ema_fast` | int | 96 | Fast EMA period (bars) | ✅ YES |
| `ema_slow` | int | 672 | Slow EMA period (bars) | ✅ YES |
| `z_window` | int | 672 | Z-score lookback window (bars) | ✅ YES |
| `atr_window` | int | 96 | ATR calculation period (bars) | ✅ YES |
| `vol_z` | float | 2.0 | Volume z-score threshold | ✅ YES |
| `tr_z` | float | 1.2 | True Range z-score threshold | ✅ YES |
| `cloc_min` | float | 0.6 | Minimum close location (0-1) | ✅ YES |

**Critical**: Any change affects signal generation. Must match backtest.

#### **backtest_params** (Strategy Exit Logic)

| Parameter | Type | Default | Description | Critical |
|-----------|------|---------|-------------|----------|
| `use_fib_exits` | bool | true | Enable Fibonacci exits | ✅ YES |
| `fib_swing_lookback` | int | 96 | Swing high lookback (bars) | ✅ YES (if fib) |
| `fib_swing_lookahead` | int | 5 | Swing high lookahead (bars) | ✅ YES (if fib) |
| `fib_target_level` | float | 0.618 | Target Fib level (0.382-1.0) | ✅ YES (if fib) |
| `use_stop_loss` | bool | false | Enable stop-loss | ✅ YES |
| `atr_stop_mult` | float | 0.7 | Stop distance (ATR multiplier) | ✅ YES (if SL) |
| `use_time_exit` | bool | false | Enable time-based exit | ✅ YES |
| `max_hold` | int | 96 | Max hold time (bars) | ✅ YES (if time) |
| `use_traditional_tp` | bool | false | Enable fixed R:R target | ✅ YES |
| `reward_r` | float | 2.0 | Risk:Reward ratio | ✅ YES (if TP) |
| `fee_bp` | float | 5.0 | Trading fee (basis points) | ⚠️ Informational |
| `slippage_bp` | float | 5.0 | Expected slippage (bp) | ⚠️ Informational |

**Critical**: Determines when/how to exit positions. Must match backtest logic exactly.

#### **risk_management** (Position Sizing & Protection)

| Parameter | Type | Default | Description | Critical |
|-----------|------|---------|-------------|----------|
| `risk_per_trade_percent` | float | 1.0 | % of account risked per trade | ✅ CRITICAL |
| `max_position_size_percent` | float | 10.0 | Max position size (% account) | ✅ CRITICAL |
| `max_daily_loss_percent` | float | 5.0 | Stop trading if daily loss > X% | ✅ CRITICAL |
| `max_daily_trades` | int | 10 | Max trades per day | ✅ CRITICAL |
| `max_open_positions` | int | 1 | Max concurrent positions | ✅ CRITICAL |
| `slippage_tolerance_percent` | float | 0.5 | Reject orders if slippage > X% | ✅ CRITICAL |
| `order_timeout_seconds` | int | 30 | Cancel order after X seconds | ✅ CRITICAL |

**Critical**: These protect against catastrophic losses. Never disable.

#### **exchange** (Connection & Credentials)

| Parameter | Type | Default | Description | Critical |
|-----------|------|---------|-------------|----------|
| `exchange_name` | str | "binance" | Exchange identifier | ✅ CRITICAL |
| `trading_pair` | str | "ADA/USDT" | Trading pair symbol | ✅ CRITICAL |
| `base_currency` | str | "ADA" | Base currency | ✅ CRITICAL |
| `quote_currency` | str | "USDT" | Quote currency | ✅ CRITICAL |
| `api_key` | str | "YOUR_API_KEY_HERE" | Exchange API key | ✅ CRITICAL |
| `api_secret` | str | "YOUR_API_SECRET_HERE" | Exchange API secret | ✅ CRITICAL |
| `api_passphrase` | str | null | API passphrase (if required) | ✅ CRITICAL (some exchanges) |
| `testnet` | bool | true | Use testnet/sandbox | ✅ CRITICAL |
| `enable_rate_limit` | bool | true | Enable rate limiting | ✅ CRITICAL |
| `paper_trading` | bool | true | Paper trading mode | ✅ CRITICAL |
| `paper_initial_balance` | float | 10000.0 | Starting balance (paper) | ⚠️ Info |

**Critical**: Wrong exchange/pair = trades won't execute. Missing credentials = can't connect.

#### **data_feed** (Real-time Data)

| Parameter | Type | Default | Description | Critical |
|-----------|------|---------|-------------|----------|
| `data_source` | str | "exchange" | Data source (exchange/polygon/both) | ✅ YES |
| `use_websocket` | bool | true | Use WebSocket (faster) | ✅ YES |
| `websocket_ping_interval` | int | 20 | WS ping interval (seconds) | ⚠️ Performance |
| `rest_api_interval_seconds` | int | 60 | REST polling interval | ⚠️ Performance |
| `max_missing_bars` | int | 3 | Alert if > X bars missing | ⚠️ Safety |
| `validate_ohlcv` | bool | true | Validate data integrity | ✅ YES |

**Critical**: Data feed must be reliable and match timeframe.

---

## 🔒 Security & Credential Management

### **CRITICAL**: API Keys NEVER in config.json

**Problem**: config.json is exported from backtest app and may be shared/committed to git.

**Solution**: Agent app reads credentials from `.env` file (never committed).

### Credential Flow

```
1. Backtest App:
   ├─ Exports config.json with placeholder: "YOUR_API_KEY_HERE"
   └─ User NEVER enters real credentials in backtest app

2. Trading Agent App:
   ├─ Reads config.json (strategy parameters)
   ├─ Reads .env file (real credentials)
   ├─ On startup: Prompts user to enter credentials if missing
   ├─ Saves credentials to .env (encrypted, never committed)
   └─ config.json remains credential-free
```

### .env File Format (Trading Agent Only)

```bash
# Exchange API Credentials (NEVER commit this file!)
EXCHANGE_API_KEY=your_actual_api_key_here
EXCHANGE_API_SECRET=your_actual_api_secret_here
EXCHANGE_API_PASSPHRASE=your_passphrase_if_needed

# Optional: Data feed credentials
POLYGON_API_KEY=your_polygon_key_here

# Monitoring
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Security Checklist

- ✅ `.env` in `.gitignore`
- ✅ Credentials stored with OS-level encryption (keyring library)
- ✅ API keys use IP whitelist (exchange settings)
- ✅ API keys have withdrawal disabled
- ✅ Paper trading enabled by default
- ✅ Testnet enabled by default
- ✅ Two-factor confirmation before live trading

---

## 🎨 User Interface Design

### Design Principles

1. **Clarity**: Every UI element has ONE clear purpose
2. **Safety**: Critical actions require confirmation
3. **Visibility**: All important info visible without scrolling
4. **Responsiveness**: UI updates in <100ms
5. **Dark Theme**: Same Dark Forest theme as backtest app

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│ ADA Seller-Exhaustion Trading Agent          [−] [□] [×]    │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 🔴 STOPPED | Paper Trading | Testnet | Balance: $10,000 │ │
│ │ [▶ Start Trading]  [⏸ Pause]  [🛑 Emergency Stop All]   │ │
│ └─────────────────────────────────────────────────────────┘ │
├──────────────┬──────────────────────────────────────────────┤
│              │                                               │
│  CHART       │  Real-Time Candlestick Chart                  │
│  (50%)       │  ┌─────────────────────────────────────────┐ │
│              │  │                                           │ │
│              │  │   [Candlesticks with Fibonacci ladders]   │ │
│              │  │   [EMA Fast/Slow overlays]                │ │
│              │  │   [Entry/exit markers]                    │ │
│              │  │                                           │ │
│              │  │   Latest Price: $0.5234 ↗ +1.2%          │ │
│              │  │                                           │ │
│              │  └─────────────────────────────────────────┘ │
├──────────────┼──────────────────────────────────────────────┤
│              │                                               │
│  POSITIONS   │  Open Positions (1 active)                    │
│  (25%)       │  ┌─────────────────────────────────────────┐ │
│              │  │ Entry: $0.5100 | Current: $0.5234        │ │
│              │  │ Size: 10,000 ADA ($5,234)                │ │
│              │  │ PnL: +$134 (+2.63%)                      │ │
│              │  │ Fib Target: $0.5350 (61.8%)              │ │
│              │  │ Stop: $0.4980 (-2.35%)                   │ │
│              │  │ Duration: 3h 24m                         │ │
│              │  │ [Manual Close]                           │ │
│              │  └─────────────────────────────────────────┘ │
├──────────────┼──────────────────────────────────────────────┤
│              │                                               │
│  STATS       │  Performance Statistics                       │
│  (25%)       │  ┌─────────────────────────────────────────┐ │
│              │  │ Today: 3 trades | 2 wins | 67% WR        │ │
│              │  │ Daily PnL: +$234 (+2.34%)                │ │
│              │  │ Total PnL: +$1,234 (+12.34%)             │ │
│              │  │ Max Drawdown: -$45 (-0.45%)              │ │
│              │  │ Sharpe Ratio: 1.89                       │ │
│              │  │                                           │ │
│              │  │ Data Feed: ● CONNECTED (WS)              │ │
│              │  │ Exchange: ● CONNECTED (12ms ping)        │ │
│              │  │ Last Update: 2025-01-15 14:32:05 UTC     │ │
│              │  └─────────────────────────────────────────┘ │
└──────────────┴──────────────────────────────────────────────┘
```

### Control Panel

Located in top section, always visible.

**Components**:

1. **Status Indicator**
   - 🔴 STOPPED (red) - Not trading
   - 🟡 STARTING (yellow) - Initializing
   - 🟢 RUNNING (green) - Active trading
   - 🟠 PAUSED (orange) - Temporarily stopped
   - 🔴 ERROR (red, flashing) - Critical error

2. **Mode Indicators**
   - "Paper Trading" badge (green) if paper trading enabled
   - "LIVE TRADING" badge (red, bold) if real money
   - "Testnet" badge (blue) if testnet mode

3. **Balance Display**
   - Current account balance (updates in real-time)
   - Today's PnL (green if positive, red if negative)

4. **Action Buttons**
   - **▶ Start Trading**: Begin monitoring and trading
   - **⏸ Pause**: Stop taking new trades (keep existing positions)
   - **🛑 Emergency Stop All**: Close all positions immediately (market orders)

### Chart Widget

Real-time candlestick chart with strategy overlays.

**Features**:
- Live updating (new candle every timeframe period)
- Zoom and pan (mouse wheel / drag)
- Crosshair with price/time display
- Auto-scroll to latest candle

**Overlays**:
- EMA Fast (cyan line)
- EMA Slow (orange line)
- Entry points (green triangle up)
- Exit points (red triangle down for stop, green triangle down for TP)
- Fibonacci ladders (rainbow, same as backtest app)
- Current position (shaded region from entry to now)

### Position Panel

Detailed view of open position(s).

**Information Displayed**:
- Entry price and timestamp
- Current price (live)
- Position size (base currency amount + quote currency value)
- Unrealized PnL ($ and %)
- Fibonacci target level and price
- Stop-loss level and price (if enabled)
- Time in position (duration)
- Manual close button

**Features**:
- Updates every second
- Color-coded PnL (green/red)
- Manual close confirmation dialog

### Stats Panel

Performance metrics dashboard.

**Metrics Displayed**:

**Today**:
- Number of trades
- Win count / loss count
- Win rate (%)
- Daily PnL ($ and %)

**All-Time**:
- Total PnL
- Total trades
- Overall win rate
- Max drawdown
- Sharpe ratio
- Average trade duration

**System Health**:
- Data feed status (● CONNECTED / ○ DISCONNECTED)
- Exchange API status (● CONNECTED / ○ DISCONNECTED)
- Ping latency (ms)
- Last update timestamp

### Log Viewer

Live activity log at bottom of window (optional toggle).

**Log Levels**:
- 🔵 INFO - Normal activity (signals detected, orders placed, etc.)
- 🟡 WARNING - Non-critical issues (missed bars, slow response, etc.)
- 🔴 ERROR - Critical errors (connection lost, order failed, etc.)
- 🟢 DEBUG - Detailed debugging info (only in debug mode)

**Format**:
```
[2025-01-15 14:32:05 UTC] [INFO] New candle completed: $0.5234 (+1.2%)
[2025-01-15 14:32:05 UTC] [INFO] Seller exhaustion signal detected!
[2025-01-15 14:32:06 UTC] [INFO] Calculating Fibonacci levels...
[2025-01-15 14:32:06 UTC] [INFO] Fib target: $0.5350 (61.8%)
[2025-01-15 14:32:07 UTC] [INFO] Risk check passed: $512 risk (1.02% account)
[2025-01-15 14:32:08 UTC] [INFO] Order placed: BUY 10,000 ADA @ MARKET
[2025-01-15 14:32:09 UTC] [INFO] Order filled: 10,000 ADA @ $0.5100 avg
[2025-01-15 14:32:09 UTC] [INFO] Position opened: +$5,100 (ID: 12345)
```

---

## ⚙️ Trading Engine Implementation

### State Machine

The trading engine operates as a state machine:

```
┌──────────────┐
│  INITIALIZING│  (Load config, connect to exchange)
└──────┬───────┘
       ▼
┌──────────────┐
│   CONNECTED  │  (Subscribed to data feed, ready)
└──────┬───────┘
       ▼
┌──────────────┐
│   MONITORING │  (Watching for signals, no position)
└──────┬───────┘
       ▼
┌──────────────┐
│   SIGNAL     │  (Seller exhaustion detected)
└──────┬───────┘
       ▼
┌──────────────┐
│   RISK_CHECK │  (Validate position sizing, limits)
└──────┬───────┘
       ▼
┌──────────────┐
│   ENTERING   │  (Placing order, waiting for fill)
└──────┬───────┘
       ▼
┌──────────────┐
│   IN_POSITION│  (Monitoring for exit conditions)
└──────┬───────┘
       ▼
┌──────────────┐
│   EXITING    │  (Placing close order)
└──────┬───────┘
       ▼
┌──────────────┐
│   CLOSED     │  (Position closed, logging trade)
└──────┬───────┘
       │
       └───────► Back to MONITORING
```

### Core Loop (Pseudocode)

```python
async def trading_loop():
    """Main trading engine loop."""
    
    # 1. Load configuration
    config = load_config("config.json")
    validate_config(config)
    
    # 2. Initialize components
    exchange = connect_exchange(config.exchange)
    data_feed = DataFeed(exchange, config.timeframe)
    strategy = SellerExhaustionStrategy(config.seller_params)
    risk_manager = RiskManager(config.risk_management)
    order_manager = OrderManager(exchange)
    
    # 3. Start data feed
    await data_feed.start()
    
    # 4. Build indicator buffer (need history for EMA/ATR/z-score)
    historical_bars = await fetch_historical_data(
        lookback=max(config.seller_params.ema_slow, config.seller_params.z_window) + 100
    )
    df = pandas.DataFrame(historical_bars)
    
    # 5. Main loop
    while True:
        # Wait for next candle completion
        new_candle = await data_feed.wait_for_next_candle()
        
        # Append to dataframe
        df = df.append(new_candle, ignore_index=True)
        df = df.tail(max_lookback + 200)  # Keep sufficient history
        
        # Calculate indicators
        df = calculate_indicators(df, config.seller_params)
        
        # Check for signal
        latest_bar = df.iloc[-1]
        if strategy.check_signal(latest_bar):
            logger.info("🚨 Seller exhaustion signal detected!")
            
            # Calculate Fibonacci levels
            fib_levels = calculate_fibonacci_levels(
                df, 
                config.backtest_params.fib_swing_lookback,
                config.backtest_params.fib_swing_lookahead
            )
            
            if not fib_levels:
                logger.warning("⚠️ No Fibonacci levels found, skipping trade")
                continue
            
            # Get target level
            target_level = config.backtest_params.fib_target_level
            target_price = fib_levels[target_level]
            
            # Risk check
            entry_price = await get_current_market_price()
            stop_price = latest_bar['low'] - (latest_bar['atr'] * config.backtest_params.atr_stop_mult)
            risk_per_coin = entry_price - stop_price
            
            position_size = risk_manager.calculate_position_size(
                account_balance=await exchange.get_balance(config.exchange.quote_currency),
                risk_per_coin=risk_per_coin,
                entry_price=entry_price
            )
            
            if not risk_manager.validate_position(position_size, entry_price):
                logger.warning("⚠️ Risk check failed, skipping trade")
                continue
            
            # Place order
            logger.info(f"💼 Placing order: BUY {position_size} {config.exchange.base_currency} @ MARKET")
            order = await order_manager.place_market_order(
                side="buy",
                amount=position_size,
                timeout=config.risk_management.order_timeout_seconds
            )
            
            if not order or order['status'] != 'filled':
                logger.error("❌ Order failed or timed out")
                continue
            
            # Position opened
            position = Position(
                entry_time=datetime.utcnow(),
                entry_price=order['average'],
                size=order['filled'],
                stop_price=stop_price,
                target_price=target_price,
                fib_levels=fib_levels
            )
            
            logger.info(f"✅ Position opened: {position.size} {config.exchange.base_currency} @ ${position.entry_price:.4f}")
            logger.info(f"🎯 Target: ${position.target_price:.4f} (Fib {target_level})")
            logger.info(f"🛑 Stop: ${position.stop_price:.4f}")
            
            # Monitor position
            while position.is_open:
                await asyncio.sleep(1)  # Check every second
                
                current_price = await get_current_market_price()
                
                # Check exit conditions
                if config.backtest_params.use_fib_exits:
                    # Check if any Fib level hit
                    for fib_level, fib_price in fib_levels.items():
                        if current_price >= fib_price:
                            logger.info(f"🎯 Fibonacci {fib_level} hit @ ${current_price:.4f}")
                            await close_position(position, current_price, reason=f"fib_{fib_level}")
                            break
                
                if config.backtest_params.use_stop_loss:
                    if current_price <= position.stop_price:
                        logger.warning(f"🛑 Stop-loss hit @ ${current_price:.4f}")
                        await close_position(position, current_price, reason="stop")
                        break
                
                if config.backtest_params.use_time_exit:
                    bars_held = (datetime.utcnow() - position.entry_time).total_seconds() / (config.timeframe * 60)
                    if bars_held >= config.backtest_params.max_hold:
                        logger.info(f"⏰ Max hold time reached ({bars_held} bars)")
                        await close_position(position, current_price, reason="time")
                        break
        
        # Update UI
        ui.update_chart(df)
        ui.update_position(position if position.is_open else None)
        ui.update_stats(get_stats())
```

### Position Management

**Entry Logic**:
1. Wait for candle close with seller exhaustion signal
2. Calculate Fibonacci levels on closed candle
3. Validate Fibonacci levels exist (need swing high)
4. Calculate position size based on risk
5. Validate risk limits (max position size, daily loss, etc.)
6. Place market order
7. Wait for fill confirmation
8. Record position in database

**Exit Logic** (checked every second):
1. **Fibonacci Exit** (if enabled, default):
   - Check if current price >= any Fib level
   - Close at first level hit (38.2%, 50%, 61.8%, 78.6%, 100%)
   - Market order for immediate fill

2. **Stop-Loss** (if enabled):
   - Check if current price <= stop_price
   - Close immediately with market order

3. **Time Exit** (if enabled):
   - Check if bars_held >= max_hold
   - Close with market order

4. **Manual Close**:
   - User clicks "Manual Close" button
   - Confirmation dialog
   - Close with market order

**Exit Priority** (if multiple enabled):
- Stop-loss > Fibonacci > Traditional TP > Time exit

### Risk Management

**Position Sizing Algorithm**:

```python
def calculate_position_size(
    account_balance: float,
    risk_per_coin: float,
    entry_price: float,
    risk_per_trade_percent: float,
    max_position_size_percent: float
) -> float:
    """
    Calculate position size based on risk management rules.
    
    Returns:
        Position size in base currency (e.g., ADA)
    """
    # Risk-based sizing
    risk_amount = account_balance * (risk_per_trade_percent / 100.0)
    size_by_risk = risk_amount / risk_per_coin
    
    # Max position sizing
    max_position_value = account_balance * (max_position_size_percent / 100.0)
    size_by_max = max_position_value / entry_price
    
    # Take minimum
    position_size = min(size_by_risk, size_by_max)
    
    # Round to exchange precision
    position_size = round_to_exchange_precision(position_size)
    
    return position_size
```

**Risk Checks** (before every trade):
1. ✅ Is risk_per_trade within limits? (default 1% max)
2. ✅ Is position_size within max? (default 10% max)
3. ✅ Have we hit daily trade limit? (default 10 max)
4. ✅ Have we hit daily loss limit? (default 5% max)
5. ✅ Is max_open_positions limit OK? (default 1 max)
6. ✅ Is account balance sufficient?

If ANY check fails → Skip trade and log warning.

### Order Execution

**Market Orders** (default):
- Immediate execution
- Accepts slippage
- Used for: entries, Fib exits, stop-loss, time exits

**Slippage Protection**:
- Get current bid/ask spread before order
- Calculate expected slippage
- If slippage > slippage_tolerance_percent → reject order
- Log warning and wait for better conditions

**Order Timeout**:
- If order not filled within order_timeout_seconds → cancel
- Log error
- Do NOT retry automatically (may be gap/halt)

**Fill Confirmation**:
- Wait for exchange fill confirmation
- Record actual fill price (may differ from market price)
- Calculate actual slippage
- Log if slippage > expected

---

## 📡 Real-Time Data Integration

### WebSocket Data Feed (Primary)

**Advantages**:
- Real-time updates (< 1 second latency)
- No polling overhead
- Efficient bandwidth usage
- Exchange-native data

**Implementation**:

```python
class WebSocketDataFeed:
    """
    WebSocket data feed for real-time candles.
    """
    
    def __init__(self, exchange, trading_pair, timeframe):
        self.exchange = exchange
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.ws = None
        self.current_candle = None
        self.candle_queue = asyncio.Queue()
    
    async def start(self):
        """Connect to WebSocket and subscribe to candle updates."""
        # CCXT unified WebSocket API
        self.ws = self.exchange.watch_ohlcv(
            symbol=self.trading_pair,
            timeframe=self.timeframe
        )
        
        asyncio.create_task(self._process_updates())
    
    async def _process_updates(self):
        """Process incoming WebSocket messages."""
        while True:
            # Wait for new candle
            ohlcv = await self.ws
            
            timestamp, open, high, low, close, volume = ohlcv
            
            # Check if candle is complete
            current_time = datetime.utcnow()
            candle_close_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            
            # Add timeframe to get close time
            candle_close_time += timedelta(minutes=self.timeframe_minutes)
            
            if current_time >= candle_close_time:
                # Candle complete - add to queue
                complete_candle = {
                    'timestamp': timestamp,
                    'open': open,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'completed': True
                }
                await self.candle_queue.put(complete_candle)
                logger.info(f"📊 New candle completed: ${close:.4f} @ {candle_close_time}")
            else:
                # Candle in progress - update current
                self.current_candle = {
                    'timestamp': timestamp,
                    'open': open,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'completed': False
                }
    
    async def wait_for_next_candle(self):
        """Wait for next complete candle."""
        candle = await self.candle_queue.get()
        return candle
```

**Supported Exchanges** (via CCXT):
- Binance (websocket)
- Kraken (websocket)
- Coinbase Pro (websocket)
- Bybit (websocket)
- Most major exchanges

### REST API Data Feed (Fallback)

**Advantages**:
- Works on all exchanges
- More reliable (no connection drops)
- Simpler error handling

**Disadvantages**:
- Higher latency (poll every 15-60 seconds)
- More bandwidth usage
- May hit rate limits

**Implementation**:

```python
class RESTDataFeed:
    """
    REST API data feed (fallback if WebSocket unavailable).
    """
    
    def __init__(self, exchange, trading_pair, timeframe, poll_interval=60):
        self.exchange = exchange
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.poll_interval = poll_interval
        self.last_candle_timestamp = None
    
    async def start(self):
        """Start polling loop."""
        asyncio.create_task(self._poll_loop())
    
    async def _poll_loop(self):
        """Poll exchange REST API for new candles."""
        while True:
            try:
                # Fetch latest candles (last 2 for safety)
                ohlcv_list = await self.exchange.fetch_ohlcv(
                    symbol=self.trading_pair,
                    timeframe=self.timeframe,
                    limit=2
                )
                
                # Get latest complete candle
                latest_candle = ohlcv_list[-2]  # Second-to-last is complete
                timestamp = latest_candle[0]
                
                # Check if new candle
                if self.last_candle_timestamp is None or timestamp > self.last_candle_timestamp:
                    self.last_candle_timestamp = timestamp
                    
                    candle = {
                        'timestamp': timestamp,
                        'open': latest_candle[1],
                        'high': latest_candle[2],
                        'low': latest_candle[3],
                        'close': latest_candle[4],
                        'volume': latest_candle[5],
                        'completed': True
                    }
                    
                    await self.candle_queue.put(candle)
                    logger.info(f"📊 New candle fetched: ${candle['close']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error fetching candles: {e}")
            
            # Wait before next poll
            await asyncio.sleep(self.poll_interval)
```

### Data Validation

**Validation Checks** (every candle):
1. ✅ Timestamp is monotonically increasing
2. ✅ High >= Low
3. ✅ High >= Open, Close
4. ✅ Low <= Open, Close
5. ✅ Volume >= 0
6. ✅ No NaN values
7. ✅ Gap detection (> max_missing_bars)

If validation fails → Log error, skip candle, alert user.

### Connection Health Monitoring

**Health Checks** (every 30 seconds):
1. WebSocket connection alive? (ping/pong)
2. REST API reachable? (ping endpoint)
3. Last candle timestamp recent? (< 2× timeframe)
4. Exchange balance accessible?

If health check fails → Attempt reconnection (max 3 retries, exponential backoff).

If reconnection fails → Alert user, pause trading, log error.

---

## 🗄️ Persistence & State Management

### Trade Database (SQLite)

**Schema**:

```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_timestamp TEXT NOT NULL,
    exit_timestamp TEXT,
    entry_price REAL NOT NULL,
    exit_price REAL,
    position_size REAL NOT NULL,
    pnl REAL,
    pnl_percent REAL,
    exit_reason TEXT,
    fib_target_level REAL,
    fib_target_price REAL,
    stop_price REAL,
    bars_held INTEGER,
    paper_trading BOOLEAN NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_entry_timestamp ON trades(entry_timestamp);
CREATE INDEX idx_exit_timestamp ON trades(exit_timestamp);
CREATE INDEX idx_paper_trading ON trades(paper_trading);
```

**Operations**:
- Insert on position open
- Update on position close
- Query for stats/metrics
- Export to CSV

### Balance Database (SQLite)

**Schema**:

```sql
CREATE TABLE balance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    balance REAL NOT NULL,
    equity REAL NOT NULL,  -- balance + unrealized PnL
    paper_trading BOOLEAN NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_timestamp ON balance_snapshots(timestamp);
```

**Operations**:
- Snapshot every 1 minute
- Snapshot on every trade
- Query for equity curve
- Export to CSV

### State Persistence (JSON)

**Purpose**: Resume trading after restart without losing state.

**File**: `state.json`

**Contents**:
```json
{
  "last_update": "2025-01-15T14:32:05Z",
  "trading_active": true,
  "current_position": {
    "entry_time": "2025-01-15T11:00:00Z",
    "entry_price": 0.5100,
    "size": 10000.0,
    "stop_price": 0.4980,
    "target_price": 0.5350,
    "fib_levels": {
      "0.382": 0.5250,
      "0.5": 0.5300,
      "0.618": 0.5350,
      "0.786": 0.5420,
      "1.0": 0.5500
    }
  },
  "daily_stats": {
    "date": "2025-01-15",
    "trades_count": 3,
    "daily_pnl": 234.56,
    "daily_loss": 0.0
  }
}
```

**Operations**:
- Save on every state change
- Load on startup
- Atomic write (temp file → rename)

---

## 🚨 Error Handling & Alerts

### Error Categories

**Level 1: INFO** (normal operation)
- Signal detected
- Order placed
- Position opened/closed
- Candle completed

**Level 2: WARNING** (non-critical issues)
- Missed candle
- High latency (> 1 second)
- Slippage above expected
- Risk check prevented trade

**Level 3: ERROR** (critical issues)
- Exchange connection lost
- Order failed
- Data validation failed
- Balance fetch failed

**Level 4: CRITICAL** (catastrophic failures)
- Cannot connect to exchange (after retries)
- Daily loss limit exceeded
- Emergency stop triggered
- Configuration invalid

### Error Handling Strategy

**Transient Errors** (network, timeout):
- Retry with exponential backoff (1s, 2s, 4s, 8s, 16s)
- Max 5 retries
- Log each attempt
- If all retries fail → Escalate to ERROR

**Permanent Errors** (invalid API key, insufficient balance):
- No retry
- Log ERROR immediately
- Alert user
- Pause trading (do NOT exit positions)

**Critical Errors** (daily loss limit):
- Close all positions immediately (market orders)
- Stop trading engine
- Send critical alert
- Require manual restart

### Alert System

**Telegram Bot** (recommended):
- Instant notifications to your phone
- Critical alerts only (Level 3+)
- Position opened/closed notifications
- Daily summary

**Email Alerts** (backup):
- Same as Telegram
- Higher latency (1-5 minutes)

**UI Notifications**:
- All log levels
- Toast notifications for Level 2+
- Sound alerts for Level 3+

---

## 🧪 Testing Strategy

### Unit Tests

**Components to Test**:
- Indicator calculations (EMA, ATR, z-score, Fibonacci)
- Signal detection logic
- Position sizing calculations
- Risk management checks
- Order validation
- Data validation

**Test Framework**: pytest

**Coverage Target**: 80%+

### Integration Tests

**Scenarios to Test**:
1. Connect to testnet exchange
2. Fetch real-time data via WebSocket
3. Detect seller exhaustion signal (use historical replay)
4. Place paper trade order
5. Monitor position with Fibonacci exits
6. Close position when Fib level hit
7. Verify trade recorded in database

### End-to-End Tests

**Full Trading Cycle**:
1. Load config.json
2. Connect to testnet
3. Run for 24 hours (automated)
4. Verify at least 1 trade executed
5. Verify PnL calculated correctly
6. Verify state persisted correctly

### Manual Testing Checklist

Before going live:
- [ ] Testnet trading for 7+ days
- [ ] At least 10 completed trades
- [ ] Win rate matches backtest (±10%)
- [ ] No critical errors
- [ ] Connection health stable
- [ ] UI responsive under load
- [ ] Emergency stop works
- [ ] State persistence works (restart test)
- [ ] Daily loss limit works
- [ ] Manual close works

---

## 🚀 Deployment

### VPS Setup (Linux/Ubuntu)

**Step 1: Provision VPS**
- Provider: AWS EC2, DigitalOcean, Vultr, Linode
- Region: Close to exchange (us-east-1 for Binance US)
- Instance: 2 vCPU, 4 GB RAM, 20 GB SSD
- OS: Ubuntu 22.04 LTS

**Step 2: Install Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip -y

# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone agent repository
git clone https://github.com/yourusername/ada-trading-agent.git
cd ada-trading-agent

# Install dependencies
poetry install --no-dev
```

**Step 3: Configure Application**
```bash
# Copy config from backtest app
cp ~/Downloads/strategy_config.json ./config.json

# Create .env file
nano .env
# (Add exchange credentials)

# Validate configuration
poetry run python -m agent.cli validate-config config.json
```

**Step 4: Run as systemd Service**
```bash
# Create service file
sudo nano /etc/systemd/system/ada-trading-agent.service
```

```ini
[Unit]
Description=ADA Trading Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ada-trading-agent
ExecStart=/home/ubuntu/ada-trading-agent/.venv/bin/python -m agent.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable ada-trading-agent
sudo systemctl start ada-trading-agent

# Check status
sudo systemctl status ada-trading-agent

# View logs
sudo journalctl -u ada-trading-agent -f
```

### Monitoring & Maintenance

**Daily Checklist**:
- [ ] Check Telegram alerts
- [ ] Verify trades executed correctly
- [ ] Check connection health
- [ ] Review logs for warnings
- [ ] Verify balance matches expected

**Weekly Maintenance**:
- [ ] Backup database (`trades.db`, `balance.db`)
- [ ] Review performance metrics
- [ ] Update strategy config if needed
- [ ] Check for software updates
- [ ] Restart service (clean state)

**Monthly Review**:
- [ ] Compare live results vs backtest
- [ ] Analyze slippage patterns
- [ ] Review rejected trades
- [ ] Optimize parameters if needed
- [ ] Archive old logs

---

## 📊 Performance Expectations

### Comparison: Backtest vs Live

**Expected Differences**:
- **Win Rate**: Live may be 5-10% lower (slippage, missed signals)
- **Average R**: Live may be 10-20% lower (execution delays)
- **Trade Count**: Live may be 10-30% lower (risk checks, connection issues)
- **Drawdown**: Live may be 20-50% higher (adverse fills, gaps)

**Acceptable Degradation**: 20% performance drop is normal.

**Unacceptable Degradation**: > 50% drop indicates problem (investigate immediately).

### Monitoring Metrics

**Key Performance Indicators (KPIs)**:
1. **Trade Execution Rate**: % of signals that became trades
   - Target: > 80%
   - Alert if: < 60%

2. **Slippage**: Average slippage per trade
   - Target: < 0.2%
   - Alert if: > 0.5%

3. **Order Fill Time**: Seconds from order to fill
   - Target: < 5 seconds
   - Alert if: > 30 seconds

4. **Connection Uptime**: % of time connected to exchange
   - Target: > 99%
   - Alert if: < 95%

5. **Signal Detection Latency**: Seconds from candle close to signal detected
   - Target: < 1 second
   - Alert if: > 5 seconds

---

## 🔐 Security Best Practices

### API Key Security

**Never**:
- ❌ Commit API keys to git
- ❌ Share API keys in chat/email
- ❌ Store API keys in plain text
- ❌ Use same API keys across multiple bots

**Always**:
- ✅ Use `.env` file (in `.gitignore`)
- ✅ Enable API key IP whitelist
- ✅ Disable withdrawal permissions
- ✅ Use separate API keys per bot
- ✅ Rotate API keys monthly
- ✅ Use subaccounts if available

### VPS Security

**Firewall**:
- Allow SSH (port 22) from your IP only
- Allow HTTPS (port 443) outbound only
- Block all other inbound traffic

**SSH**:
- Use SSH keys (disable password auth)
- Change default port (22 → random high port)
- Install fail2ban (auto-ban brute force attempts)

**Updates**:
- Enable automatic security updates
- Review and apply patches weekly

---

## 📚 Appendix

### A. Exchange-Specific Notes

#### Binance
- **WebSocket**: Excellent, < 100ms latency
- **API Rate Limits**: 1200 req/min (weight-based)
- **Testnet**: Available at testnet.binance.vision
- **Recommended**: Yes (most liquidity)

#### Kraken
- **WebSocket**: Good, ~200ms latency
- **API Rate Limits**: Tiered (15-20 req/min base)
- **Testnet**: Not available (use paper trading)
- **Recommended**: Yes (regulated, secure)

#### Coinbase Pro
- **WebSocket**: Good, ~150ms latency
- **API Rate Limits**: 10 req/sec public, 5/sec private
- **Testnet**: Available (sandbox)
- **Recommended**: Yes (US-based, regulated)

### B. Timeframe Considerations

**1-minute** (Timeframe.m1):
- Very active (dozens of signals per day)
- Higher slippage (fast market)
- Requires low-latency connection
- Recommended: Paper trading only for testing

**15-minute** (Timeframe.m15):
- Moderate activity (2-5 signals per day)
- Balanced slippage
- Tested timeframe (default)
- Recommended: Start here

**1-hour** (Timeframe.m60):
- Lower activity (0-2 signals per day)
- Lower slippage
- More stable
- Recommended: Conservative traders

### C. Common Issues & Solutions

**Issue**: "Connection refused" on startup  
**Solution**: Check exchange API status, verify credentials, check firewall

**Issue**: "Insufficient balance" error  
**Solution**: Verify exchange balance, check quote currency (USDT vs USD)

**Issue**: No signals detected after hours  
**Solution**: Verify market conditions (need downtrend), check indicator calculations

**Issue**: Orders timing out  
**Solution**: Increase order_timeout_seconds, check exchange latency

**Issue**: High slippage (> 1%)  
**Solution**: Use limit orders instead of market, trade during high liquidity hours

---

## ✅ Pre-Launch Checklist

### Configuration
- [ ] config.json loaded and validated
- [ ] Exchange credentials in .env (not committed!)
- [ ] Paper trading enabled
- [ ] Testnet enabled
- [ ] Risk limits configured (1% risk, 10% max position)
- [ ] Daily loss limit set (5% max)

### Testing
- [ ] Unit tests passing (80%+ coverage)
- [ ] Integration tests passing (exchange connection)
- [ ] Manual testing completed (7+ days testnet)
- [ ] Emergency stop tested
- [ ] State persistence tested (restart)

### Monitoring
- [ ] Telegram bot configured
- [ ] Log rotation configured
- [ ] Database backups automated
- [ ] Alert thresholds configured

### Deployment
- [ ] VPS provisioned and secured
- [ ] systemd service configured
- [ ] Firewall rules applied
- [ ] SSH keys configured
- [ ] Automatic updates enabled

### Documentation
- [ ] README.md created
- [ ] Configuration guide written
- [ ] Troubleshooting guide written
- [ ] Emergency procedures documented

---

## 📄 License & Disclaimer

**License**: MIT (or proprietary, your choice)

**DISCLAIMER**:
```
THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.

TRADING CRYPTOCURRENCIES INVOLVES SUBSTANTIAL RISK OF LOSS. PAST PERFORMANCE 
IS NOT INDICATIVE OF FUTURE RESULTS. THE AUTHORS AND CONTRIBUTORS ARE NOT 
RESPONSIBLE FOR ANY FINANCIAL LOSSES INCURRED THROUGH USE OF THIS SOFTWARE.

USE AT YOUR OWN RISK. ALWAYS START WITH PAPER TRADING. NEVER RISK MORE THAN 
YOU CAN AFFORD TO LOSE.

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT.
```

---

## 📞 Support & Contact

**Issues**: GitHub Issues (after repository created)  
**Documentation**: README.md in repository  
**Questions**: [Your contact method]

---

**END OF TRADING AGENT PRD**

**Version**: 1.0.0  
**Last Updated**: 2025-01-15  
**Document Status**: ✅ COMPLETE AND PRODUCTION-READY

This PRD provides exhaustive specifications for building a production-grade live trading agent. All critical components are specified in detail. Implement phase by phase (paper trading → testnet → live) with comprehensive testing at each stage.
