# ADA Seller-Exhaustion Agent

**15-minute timeframe trading agent for Cardano (ADAUSD)**

A complete research, backtest, and paper trading system for detecting "seller exhaustion" patterns in ADA/USD using 15-minute bars.

## Features

- 🔍 **Real-time data fetching** from Polygon.io (15-minute aggregates)
- 📊 **Technical indicators**: EMA, SMA, RSI, MACD, ATR (locally computed)
- 🎯 **Seller Exhaustion Strategy**: Detects high-volume, high-volatility bottoms in downtrends
- 🔄 **Event-driven backtesting** with configurable parameters
- 🖥️ **Dark Forest UI** (PySide6 + PyQtGraph) with interactive candlestick charts
- 📈 **Visual overlays**: EMAs, signal markers, and indicators
- 🤖 **Paper trading** scheduler (15-minute bar close events)
- 📝 **Trade export** to CSV with full metrics

## Architecture

```
ada-agent/
├── app/              # PySide6 UI application
├── core/             # Core models and utilities
├── data/             # Data providers (Polygon.io)
├── indicators/       # Technical indicators (pandas-based)
├── strategy/         # Seller exhaustion strategy logic
├── backtest/         # Event-driven backtesting engine
├── exec/             # Paper trading execution (placeholder)
├── config/           # Settings and environment config
├── tests/            # Unit tests
└── cli.py            # CLI commands (fetch, backtest, ui)
```

## Installation

### Requirements

- Python 3.10+
- Poetry (dependency management)
- Polygon.io API key (free tier available)

### Setup

1. **Clone and setup**:
   ```bash
   # Install Poetry if not already installed
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   poetry install
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Polygon.io API key
   # POLYGON_API_KEY=your_key_here
   ```

3. **Get Polygon.io API key**:
   - Sign up at [polygon.io](https://polygon.io/)
   - Free tier includes crypto data access
   - Copy your API key to `.env`

## Usage

### CLI Commands

The project includes a CLI with three main commands:

#### 1. Fetch Data

```bash
poetry run python cli.py fetch --from 2024-01-01 --to 2025-01-13
```

Fetches 15-minute OHLCV data from Polygon.io and displays summary.

#### 2. Run Backtest

```bash
poetry run python cli.py backtest --from 2024-01-01 --to 2025-01-13
```

Runs full backtest with default parameters:
- **Entry**: t+1 open after signal
- **Stop**: signal_low - 0.7 * ATR
- **TP**: entry + 2R (2:1 reward/risk)
- **Max Hold**: 96 bars (~24 hours)
- **Fees**: 5 bps + 5 bps slippage

**Custom parameters**:
```bash
poetry run python cli.py backtest \
  --ema-fast 96 \
  --ema-slow 672 \
  --vol-z 2.0 \
  --tr-z 1.2 \
  --reward-r 2.0 \
  --output my_trades.csv
```

#### 3. Launch UI

```bash
poetry run python cli.py ui
```

Opens the PySide6 application with:
- Interactive candlestick chart
- EMA overlays (96 and 672 periods)
- Signal markers (yellow triangles)
- Real-time data refresh capability

### Programmatic Usage

```python
import asyncio
from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from backtest.engine import run_backtest, BacktestParams

async def run():
    # Fetch data
    dp = DataProvider()
    df = await dp.fetch_15m("X:ADAUSD", "2024-01-01", "2025-01-13")
    
    # Build features
    params = SellerParams()
    feats = build_features(df, params)
    
    # Run backtest
    bt_params = BacktestParams()
    result = run_backtest(feats, bt_params)
    
    print(result['metrics'])
    result['trades'].to_csv('trades.csv', index=False)
    
    await dp.close()

asyncio.run(run())
```

## Strategy: Seller Exhaustion

The strategy detects potential bottoms by looking for:

1. **Downtrend filter**: EMA(96) < EMA(672)
2. **Volume spike**: Volume z-score > 2.0
3. **Range expansion**: True Range z-score > 1.2
4. **Close near high**: Close in top 60% of candle

**Entry**: Next bar open after signal  
**Stop**: Signal low - 0.7 × ATR  
**TP**: 2R (Risk-Reward ratio of 2:1)  
**Exit**: Stop hit, TP hit, or 96 bars (max hold)

### Default Parameters

```python
ema_fast = 96       # ~1 day (15m bars)
ema_slow = 672      # ~7 days
z_window = 672      # ~7 days lookback
vol_z = 2.0         # Volume z-score threshold
tr_z = 1.2          # True Range z-score threshold
cloc_min = 0.6      # Close location (0-1)
atr_window = 96     # ATR calculation window
```

## Testing

Run all tests:
```bash
poetry run pytest tests/ -v
```

Run specific test file:
```bash
poetry run pytest tests/test_strategy.py -v
```

## UI Features

The Dark Forest themed UI includes:

- **Candlestick chart** with green (up) and red (down) candles
- **EMA overlays**: Fast (cyan) and Slow (orange)
- **Signal markers**: Yellow triangles at exhaustion signals
- **Status bar**: Shows data loading progress
- **Refresh button**: Re-fetch latest data
- **Info panel**: Displays date range and signal count

### Controls

- **Mouse wheel**: Zoom in/out
- **Left click + drag**: Pan chart
- **Right click**: Context menu (save image, etc.)

## Development

### Project Structure

- `app/`: PySide6 UI components
- `core/`: Models (Pydantic) and time utilities
- `data/`: Async Polygon.io client and data provider
- `indicators/`: Pandas-based technical indicators
- `strategy/`: Signal generation logic
- `backtest/`: Event-driven backtest engine
- `tests/`: Unit tests

### Adding Features

1. **New indicator**: Add to `indicators/local.py`
2. **New strategy**: Create module in `strategy/`
3. **New UI widget**: Add to `app/widgets/`
4. **New test**: Add to `tests/`

## Roadmap

### MVP (Week 1) ✅
- [x] Data layer (Polygon.io)
- [x] Strategy implementation
- [x] Backtest engine
- [x] UI with charts
- [x] CLI commands

### Week 2
- [ ] Paper trading scheduler
- [ ] Live bar-close event handling
- [ ] Parameter optimization UI
- [ ] Walk-forward analysis
- [ ] Monte Carlo simulation

### Nice-to-have
- [ ] Multi-timeframe confluence
- [ ] Signal heatmap by time/weekday
- [ ] System tray notifications
- [ ] Auto-generated Jupyter notebooks
- [ ] Real broker integration (Binance/Kraken)

## Notes

- **Data source**: Polygon.io crypto aggregates (not Binance-consolidated)
- **Timezone**: All timestamps in UTC
- **Deterministic**: Same inputs → same backtest results
- **Performance**: Handles 5000+ candles smoothly in UI

## License

Private project by Michal.

## Troubleshooting

### qasync not working

If you see warnings about qasync, install it explicitly:
```bash
poetry add qasync
```

### No data returned

- Check your Polygon.io API key in `.env`
- Verify date range (use YYYY-MM-DD format)
- Check API quota (free tier has limits)

### UI not displaying

Make sure you have Qt dependencies:
```bash
# Linux
sudo apt-get install libxcb-xinerama0 libxcb-cursor0

# macOS
brew install qt6
```

## Support

For issues or questions, contact Michal.
