# Seller-Exhaustion Entry - Fibonacci Exit Trading strategy Optimization Tool

**Multi-timeframe strategy research, backtesting, and optimization**

A complete strategy development platform with Fibonacci-based exits, parameter optimization, and **strategy export for live trading**.

**🚨 IMPORTANT**: This is a BACKTESTING tool, not a live trading application.  
For live trading, export your strategy using **💾 Export Strategy** and use the separate trading agent (see **PRD_TRADING_AGENT.md**).

---

## 🎯 Core Strategy

**Entry**: BUY at seller exhaustion bottoms only  
**Exit**: SELL at first Fibonacci retracement level hit (**default: 61.8% Golden Ratio**)

**Philosophy**: Let market structure guide exits. No arbitrary time limits or fixed R-multiples by default.

### Entry Signal (Seller Exhaustion)
All four conditions must be true:
1. **Downtrend**: EMA_fast (96) < EMA_slow (672)
2. **Volume Spike**: Volume z-score > 2.0 (unusual selling pressure)
3. **Range Expansion**: True Range z-score > 1.2 (high volatility)
4. **Close Near High**: Close location > 0.6 (buyers stepping in)

### Exit Strategy (**NEW in v2.0**)
**Default**: ✅ Fibonacci exits ENABLED, ❌ Stop-loss OFF, ❌ Time exit OFF

- **Fibonacci Levels**: Exit at first level hit (38.2%, 50%, 61.8%, 78.6%, 100%)
- **Stop-Loss**: Optional - Signal low - (ATR × 0.7)
- **Traditional TP**: Optional - Entry + (Risk × 2.0)
- **Time Exit**: Optional - After 96 bars (~24h on 15m)

---

## ✨ Key Features

### 🎯 Configurable Fitness Functions (**NEW**, **v2.1**)
- **Goal-oriented optimization** for different trading styles
- **4 presets**: Balanced, High Frequency, Conservative, Profit Focused
- **Custom weights**: Fine-tune optimization goals (trade count, win rate, avg R, PnL, drawdown)
- **Minimum requirements**: Filter strategies by min trades and win rate
- **Real-time switching**: Change optimization goals without restarting
- **Example**: High Frequency preset generates 50-100+ trades vs 10-20 with Balanced

### 📐 Reorganized Parameter Editor (**NEW**, **v2.1**)
- **4 logical sections**: Strategy Parameters, Exit Strategy, Transaction Costs, Fitness Function
- **Time-based display**: All time parameters shown in minutes (e.g., "1440 min" = 24h)
- **Automatic conversion**: Minutes ↔ bars based on active timeframe
- **No duplication**: Single source of truth for all parameters
- **Always visible**: Sidebar access without opening dialogs
- **Helpful tooltips**: Show both time periods and bar counts

### 💾 Data Caching System
- **Automatic caching** of downloaded data to `.data/` directory
- **Parquet format** for efficient storage
- **Auto-load on startup** - no more re-downloading
- Force refresh option when needed
- Cache management utilities

### 📏 Timeframe Scaling (**NEW**, **CRITICAL**)
- **Time-based parameters** (24h, 7d) that auto-convert to bars
- **Smart auto-adjustment** when changing timeframes
- Confirmation dialog showing parameter changes
- Prevents common mistake of using 15m parameters on 1m
- **Example**: EMA Fast = 96 bars on 15m = 1440 bars on 1m (both = 24 hours)

### 🌈 Fibonacci Ladder Visualization (**NEW**)
- **Beautiful rainbow ladder** on chart showing exit strategy
- Swing high marker (⭐ gold star)
- Color-coded levels: 38.2% blue → 50% cyan → **61.8% GOLD** → 78.6% orange → 100% red
- **Golden Ratio prominently highlighted**
- Exit line showing actual outcome
- Toggle in Settings → Chart Indicators

### 💾 Strategy Export System (**NEW**, **v2.1**, **CRITICAL FOR LIVE TRADING**)
- **Export complete strategy** to JSON for live trading agent
- **All parameters included**: Entry signals, exits, risk management, exchange settings
- **Validation on export**: Warns about configuration issues
- **Import strategies**: Load configurations from other users or previous sessions
- **Safe defaults**: Paper trading and testnet enabled by default
- **Complete specification**: See **PRD_TRADING_AGENT.md** for trading agent setup

**Export Format Includes**:
- ✅ Strategy parameters (EMA, z-score thresholds, etc.)
- ✅ Exit configuration (Fibonacci, stop-loss, time, TP toggles)
- ✅ Risk management (position sizing, daily limits)
- ✅ Exchange connection (API placeholders, testnet/paper settings)
- ✅ Data feed configuration (WebSocket/REST preferences)
- ✅ Backtest metrics (performance reference)

**Workflow**: Backtest → Optimize → **Export** → Deploy to VPS Trading Agent

### 🧬 Population Export/Import 
- Export the full GA population to JSON for later continuation or sharing
- Initialize optimization from a saved population instead of a random start
– Automatic export on finish: writes `populations/<pid>.json` after results render
- Process marker: creates an empty file `processes/<pid>` at launch (no contents)
- APIs:
  - `from backtest.optimizer import export_population, Population`
  - `export_population(pop, "population.json")`
  - `Population.from_file("population.json", timeframe=Timeframe.m15, limit=24)`
  - `EvolutionaryOptimizer(..., initial_population_file="population.json").initialize(...)`

### 🧰 CLI Commands (Population Management)

- `ui --ga-init-from PATH`
  - Launches UI, loads population from `PATH`, auto-starts optimization when the window and data are ready.
  - Parameters: `PATH` is a JSON exported by ga-export or via the API.

- `ga-export OUTPUT [--size N] [--timeframe TF]`
  - Exports a population seeded from current `.env` strategy/backtest settings.
  - `OUTPUT`: destination JSON path.
  - `--size`: population size; defaults to `GA_POPULATION_SIZE` in `.env`.
  - `--timeframe`: one of `1m,3m,5m,10m,15m,30m,60m`; if omitted uses `TIMEFRAME` from `.env`.

- `ga-init-from PATH`
  - Convenience alias for `ui --ga-init-from PATH`.

### 🤖 Auto Behavior

- On app launch: creates `processes/<pid>` (empty file) to mark the running process.
- On optimization finish: automatically exports `populations/<pid>.json` (after results render).
- Generations run by the auto-start flow are read from `.env` as `OPTIMIZER_ITERATIONS`.

### 🎨 Strategy Editor
- Comprehensive parameter management with detailed explanations
- **⭐ Golden Button**: One-click setup for optimal 61.8% Fibonacci target
- Exit toggles for stop-loss, time, and TP exits
- Save/load evolved parameters from genetic algorithm
- Export to YAML for documentation

### 📊 Fibonacci Exit System
- Market-driven exits at natural resistance levels
- Automatic swing high detection
- Configurable lookback/lookahead periods
- Exit at 38.2%, 50%, 61.8%, 78.6%, or 100% retracement

### 💾 Parameter Persistence
- Save configurations with metadata (generation, fitness, date)
- Load parameter sets with one click
- Browse saved configurations
- Export to JSON/YAML

### ⚠ Acceleration Note
Feature computation now runs on pandas vectorized operations only (Spectre/GPU path removed in v2.2 after benchmarking at 319× slower and producing incorrect signals). Backtesting and optimization are CPU-only with optional multicore evaluation, delivering ~0.16 s feature builds, ~0.18 s backtests, and ~4–5 s generations on a 12-core CPU.

### 📈 Multi-Timeframe Support
- 1m, 3m, 5m, 10m, 15m timeframes
- **Time-based and bar-based parameter conversion**
- **Consistent strategy behavior across all timeframes**
- Automatic parameter scaling with user confirmation

### 🎛️ Genetic Algorithm Optimizer
- Population-based parameter search
- **Timeframe-aware optimization bounds**
- Configurable mutation rate, sigma, elitism
- Fitness evolution tracking
- Apply best parameters to UI
- Worker process count configurable in Settings → Optimization (set to 1 for sequential runs)

### 🖥️ Dark Forest UI
- Interactive PyQtGraph candlestick charts
- Real-time optimization dashboard
- Settings dialog with data download
- Stats panel with performance metrics

---

## 🚀 Quick Start

### System Requirements
- Python 3.10+
- 8 GB+ RAM
- Multi-core CPU recommended (optimizer scales with cores)
- No GPU required (pandas CPU pipeline is fastest)

```bash
# 1. Install dependencies
poetry install

# 2. Configure API key
cp .env.example .env
# Edit .env: POLYGON_API_KEY=your_key_here

# 3. Launch GUI
poetry run python cli.py ui

# 4. In GUI:
#    - Click ⚙ Settings → Download data
#    - Click 📊 Strategy Editor → ⭐ Set Golden
#    - Click ▶ Run Backtest

# 5. Optimize (optional):
#    - Stats Panel → Initialize Population
#    - Click Step repeatedly (parameters auto-apply)
#    - Save in Strategy Editor if satisfied
```

---

## 🔄 Complete Workflow: From Backtest to Live Trading

This tool is **part 1** of a two-application system:

```
┌─────────────────────────────────────────────────────────────┐
│  PART 1: BACKTESTING TOOL (This Application)                │
│  Purpose: Research, optimize, and export strategies          │
├─────────────────────────────────────────────────────────────┤
│  1. Download historical data (Polygon.io)                    │
│  2. Configure strategy parameters                            │
│  3. Run backtests to validate strategy                       │
│  4. Optimize with genetic algorithm (optional)               │
│  5. 💾 EXPORT STRATEGY to config.json                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Transfer config.json
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PART 2: LIVE TRADING AGENT (Separate Application)          │
│  Purpose: Execute strategies on live markets                 │
│  Spec: See PRD_TRADING_AGENT.md (1,234 lines)               │
├─────────────────────────────────────────────────────────────┤
│  1. Import config.json (contains ALL strategy parameters)   │
│  2. Configure exchange credentials (.env file)               │
│  3. Start paper trading (7+ days testing)                    │
│  4. Graduate to testnet (7+ days testing)                    │
│  5. Deploy live on VPS (start small!)                        │
└─────────────────────────────────────────────────────────────┘
```

### Export System Security

**Critical**: API credentials are NEVER in exported config.json
- Backtesting tool exports with placeholders: `"YOUR_API_KEY_HERE"`
- Trading agent reads real keys from separate `.env` file
- Safe to commit/share config.json (no sensitive data)

### Step-by-Step Guide

1. **Research Phase** (this tool):
   ```bash
   poetry run python cli.py ui
   # → Download data, run backtests, optimize
   # → Click 💾 Export Strategy
   # → Save as strategy_15m_optimized.json
   ```

2. **Deployment Phase** (trading agent):
   ```bash
   # Clone trading agent repo (implement per PRD_TRADING_AGENT.md)
   cp strategy_15m_optimized.json trading-agent/config.json
   
   # Configure credentials (NEVER commit this!)
   echo "EXCHANGE_API_KEY=..." >> trading-agent/.env
   
   # Start with paper trading
   cd trading-agent && poetry run python -m agent.main
   ```

3. **Documentation Path**:
   - **docs/STRATEGY_EXPORT_GUIDE.md** - How to export/import strategies
   - **PRD_TRADING_AGENT.md** - Complete trading-agent specification

---

## 📖 Documentation

### Core Documentation

| File | Purpose | Size |
|------|---------|------|
| **README.md** | This file - overview and quick start | Overview |
| **AGENTS.md** | Comprehensive development guide for AI agents | 2,346 lines |
| **PRD.md** | Product requirements document (backtesting tool) | 850 lines |

### Strategy Export & Live Trading

| File | Purpose | Size |
|------|---------|------|
| **PRD_TRADING_AGENT.md** | **Complete specification for live trading agent** | **1,234 lines** |
| **docs/STRATEGY_EXPORT_GUIDE.md** | How to export/import strategies | 650 lines |

### Feature-Specific Guides

| File | Purpose | Size |
|------|---------|------|
| **docs/STRATEGY_DEFAULTS_GUIDE.md** | Default behavior and customization guide | Reference |

### Quick Navigation

- **Want to backtest?** → Start with this README
- **Want to develop?** → Read AGENTS.md
- **Want to deploy live?** → Read PRD_TRADING_AGENT.md
- **Need help with export?** → Read docs/STRATEGY_EXPORT_GUIDE.md

---

## 📂 Project Structure

```
seller_exhaustion-1/
├── app/
│   ├── main.py                    # Main window with auto-load
│   ├── theme.py                   # Dark Forest theme
│   └── widgets/
│       ├── candle_view.py         # Chart with Fib ladder viz 🌈
│       ├── settings_dialog.py     # Settings with TF auto-adjust
│       ├── stats_panel.py         # Optimization dashboard
│       └── strategy_editor.py     # Parameter editor ⭐
├── backtest/
│   ├── engine.py                  # CPU backtest with exit toggles
│   ├── optimizer.py               # GA helpers (population, evolution)
│   ├── optimizer_evolutionary.py  # Evolutionary optimizer
│   ├── optimizer_adam.py          # Gradient-based optimizer variant
│   ├── optimizer_factory.py       # Optimizer selection helpers
│   └── metrics.py                 # Performance calculations
├── indicators/
│   ├── local.py                   # Pandas indicators
│   └── fibonacci.py               # Fib retracement calculations
├── strategy/
│   ├── seller_exhaustion.py       # Strategy with Fib support
│   ├── params_store.py            # Parameter persistence
│   └── timeframe_defaults.py      # ⭐ Timeframe scaling 
├── data/
│   ├── polygon_client.py          # Polygon.io API client
│   ├── provider.py                # Data provider with cache
│   ├── cache.py                   # ⭐ Parquet caching 
│   └── cleaning.py                # Data cleaning utilities
├── core/
│   ├── models.py                  # Pydantic models (Bar, Trade, Params, FitnessConfig)
│   ├── timeutils.py               # UTC time utilities
│   └── strategy_export.py         # ⭐ Strategy export/import system 
├── config/                        # Settings management
├── tests/                         # 19 tests, all passing ✅
└── cli.py                         # CLI commands
```

---

## 🔥 New Feature Highlights

### 1. Data Caching - Never Re-Download Again!

**Problem Solved**: Data was lost after closing the app, requiring re-download every time.

**How It Works**:
- Downloaded data automatically saved to `.data/` directory in Parquet format
- On app startup, cached data loads automatically
- No API calls unless you explicitly download fresh data

```python
# Automatic behind the scenes:
# 1. Download data → Saved to .data/X_ADAUSD_2024-01-01_2024-12-31_1minute.parquet
# 2. Close app
# 3. Reopen app → Data loads instantly from cache!
```

### 2. Timeframe Scaling - Critical Architecture Fix! ⚠️

**Problem Solved**: Parameters hardcoded for 15m didn't work on other timeframes.

**Example of the Problem**:
```
Using 15m defaults on 1m:
- EMA Fast = 96 bars
  ✅ On 15m: 96 × 15min = 1440min = 24 hours (correct!)
  ❌ On 1m:  96 × 1min = 96min = 1.6 hours (WAY TOO SHORT!)
```

**How It Works**:
- When you change timeframe in Settings, a dialog appears:
  ```
  Adjust parameters for 1 minute timeframe?
  
  EMA Fast: 96 bars → 1440 bars (24 hours)
  EMA Slow: 672 bars → 10080 bars (7 days)
  
  [Yes] [No]
  ```
- Click Yes → Parameters automatically scaled!
- **All timeframes use same TIME PERIODS** (24h short-term, 7d long-term)
- Optimization bounds also scale automatically

**Timeframe Comparison**:
| Timeframe | EMA Fast (24h) | EMA Slow (7d) | Max Hold | Style |
|-----------|----------------|---------------|----------|-------|
| 1m | 1440 bars | 10080 bars | 240 bars (4h) | Scalping |
| 3m | 480 bars | 3360 bars | 160 bars (8h) | Scalping |
| 5m | 288 bars | 2016 bars | 144 bars (12h) | Scalping |
| 10m | 144 bars | 1008 bars | 144 bars (24h) | Intraday |
| 15m | 96 bars | 672 bars | 96 bars (24h) | Intraday |

### 3. Fibonacci Ladder Visualization - See Your Exit Strategy! 🌈

**Problem Solved**: Users couldn't see WHY exits happened at specific prices.

**How It Works**:
- Beautiful rainbow-colored Fibonacci levels displayed on chart
- Shows swing high (⭐ gold star) used for calculation
- Color gradient: 38.2% (blue) → 50% (cyan) → **61.8% GOLD** → 78.6% (orange) → 100% (red)
- Bold exit line shows actual outcome
- Toggle in Settings → Chart Indicators → "📊 Fibonacci Exit Ladders"

**Visual Example**:
```
⭐ Swing High (gold star showing Fib source)
│
├─ ─ ─ ─ ─ (dashed line to entry)
│
├────────────── 100% (RED) Full Retracement
├────────────── 78.6% (ORANGE) Aggressive
├══════════════ 61.8% (GOLD) ⭐ GOLDEN EXIT ← Actual exit
├────────────── 50.0% (CYAN) Balanced
├────────────── 38.2% (BLUE) Conservative
│
▲ ENTRY (green arrow)
```

**Benefits**:
- **Transparency**: See exactly where exits come from
- **Education**: Learn how Fibonacci retracements work
- **Trust**: Understanding builds confidence in the system

---

## 🎓 Strategy Parameters

### Entry (SellerParams)
```python
ema_fast = 96           # ~1 day on 15m
ema_slow = 672          # ~7 days on 15m  
z_window = 672          # Z-score lookback
atr_window = 96         # ATR calculation
vol_z = 2.0             # Volume threshold
tr_z = 1.2              # Range threshold
cloc_min = 0.6          # Close location (60%)
```

### Exit (BacktestParams v2.0)
```python
# Exit Toggles 
use_fib_exits = True         # ✅ ON by default
use_stop_loss = False        # ❌ OFF by default
use_time_exit = False        # ❌ OFF by default
use_traditional_tp = False   # ❌ OFF by default

# Fibonacci Parameters
fib_swing_lookback = 96      # Bars to search
fib_swing_lookahead = 5      # Confirmation
fib_target_level = 0.618     # Golden Ratio

# Optional Exit Parameters
atr_stop_mult = 0.7          # If stop enabled
reward_r = 2.0               # If TP enabled
max_hold = 96                # If time exit enabled

# Costs (always applied)
fee_bp = 5.0                 # 0.05%
slippage_bp = 5.0            # 0.05%
```

---

## 🧪 Testing

```bash
# Run all tests
poetry run pytest tests/ -v

# Test Fibonacci functionality
poetry run pytest tests/test_fibonacci.py -v

# With coverage
poetry run pytest tests/ --cov=. --cov-report=html

# (GPU not required; pandas handles feature computation extremely fast)
```



---

## 💻 Tech Stack

- **Python**: 3.10+ (tested on 3.13)
- **Package Manager**: Poetry
- **Async**: httpx, qasync
- **Data**: pandas, numpy
- **UI**: PySide6 (Qt6), PyQtGraph
- **Optimization**: NumPy GA; ADAM prototype
- **Config**: Pydantic Settings, python-dotenv
- **Persistence**: JSON, YAML (pyyaml)
- **CLI**: Typer, Rich
- **Testing**: pytest

### Performance
- **Feature computation**: 0.16 s for ~1,440 bars (pandas vectorized)
- **Backtest**: 0.18 s per evaluation
- **Optimization**: ~4–5 s per generation on a 12-core CPU (~30 s single-core)
- **100 generations**: ~8 minutes end-to-end on modern desktop hardware

---

## ⚙️ Configuration

Create `.env` file:
```bash
# Required
POLYGON_API_KEY=your_key_here

# Optional (defaults shown)
DATA_DIR=.data
TZ=UTC

# Genetic Algorithm
GA_POPULATION_SIZE=24
GA_MUTATION_RATE=0.3
GA_SIGMA=0.1
GA_ELITE_FRACTION=0.1
GA_TOURNAMENT_SIZE=3
GA_MUTATION_PROBABILITY=0.9
```

All settings editable via UI Settings dialog and auto-saved.

---

## 🎯 Usage Examples

### CLI

```bash
# Fetch data
poetry run python cli.py fetch --from 2024-01-01 --to 2024-12-31

# Run backtest (Fibonacci exits only, default)
poetry run python cli.py backtest --from 2024-01-01 --to 2024-12-31

# Launch GUI (recommended)
poetry run python cli.py ui
```

### Programmatic

```python
import asyncio
from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from backtest.engine import run_backtest, BacktestParams
from core.models import Timeframe

async def run():
    # Fetch data
    dp = DataProvider()
    df = await dp.fetch_15m("X:ADAUSD", "2024-01-01", "2024-12-31")
    
    # Build features with Fibonacci levels
    params = SellerParams()
    feats = build_features(df, params, Timeframe.m15, add_fib=True)
    
    # Run backtest (default: Fib exits only)
    bt_params = BacktestParams()  # use_fib_exits=True by default
    result = run_backtest(feats, bt_params)
    
    # Analyze results
    print(f"Trades: {result['metrics']['n']}")
    print(f"Win Rate: {result['metrics']['win_rate']:.1%}")
    print(f"Avg R: {result['metrics']['avg_R']:.2f}")
    
    # Export trades
    result['trades'].to_csv('trades.csv', index=False)
    
    await dp.close()

asyncio.run(run())
```

---

## 📊 Performance Notes

Pandas provides the feature pipeline; backtesting runs on CPU and the optimizer supports single‑core and multi-core evaluation.

---

## 🆕 What's New

### v2.1 (Latest)
✨ **Configurable Fitness Functions** - Optimize for HFT, conservative, profit-focused strategies  
✨ **Reorganized Parameter Editor** - 4 logical sections with time-based display  
✨ **Time-Based Parameters** - Intuitive minutes display, auto-converts to bars  
✨ **No Duplication** - Single source of truth for all parameters  

### v2.0
✨ **Fibonacci Exit System** - Market-driven exits at resistance levels  
✨ **Strategy Editor** - Comprehensive parameter UI with explanations  
✨ **Parameter Persistence** - Save/load evolved configurations  
✨ **Exit Toggles** - Clean defaults (Fib-only by default)  
✨ **Golden Button** - One-click optimal setup (61.8%)  
 
✨ **Multi-Timeframe** - 1m, 3m, 5m, 10m, 15m support  

### Breaking Changes
⚠️ **Default behavior changed**: Only Fibonacci exits enabled by default  
⚠️ **Stop-loss OFF by default**: Enable in Strategy Editor if needed  
⚠️ **Time exit OFF by default**: Enable for capital efficiency  

---

## ⚠ Note on Acceleration

Legacy GPU code has been removed entirely—the UI, feature builder, and optimizer are tuned for CPU workloads and already outperform the deprecated GPU path.

### Multi-Step Optimization UI

Features:
- **🚀 Optimize button**: Run 10-1000 generations automatically
- **Progress bar**: Real-time ETA and generation count
- **⏹ Cancel button**: Graceful interruption without data loss
- Thread-safe, non-blocking UI during optimization

**Workflow**:
```bash
# 1. Launch UI
poetry run python cli.py ui

# 2. In UI:
#    - Stats Panel → Initialize Population
#    - Set generations: 50
#    - Click "🚀 Optimize"
#    - Watch progress bar!
#    - Cancel anytime if needed

# Result: 50 generations (CPU-only focus)
```

### Notes
The entire pipeline is CPU-based. Parallelism comes from the configurable worker count; no GPU or external acceleration paths are required.

---

## 🐛 Troubleshooting

 

**No trades in backtest?**
- Check signals: `feats['exhaustion'].sum()`
- Check Fib levels: `feats['fib_swing_high'].notna().sum()`
- Try looser params: Lower vol_z, tr_z

**UI freezes during fetch?**
- Ensure `qasync` installed: `poetry add qasync`
- Check async/await used in data ops

**Strategy Editor not saving?**
- Click 💾 Save Params (not auto-saved)
- Check `.strategy_params/` directory created

See `AGENTS.md` Troubleshooting section for more.

---

## 🛣️ Roadmap

### v2.1 (Next)
- [ ] Multi-Fib targets (partial exits)
- [ ] Fibonacci visualization on chart
- [ ] Auto-save best GA parameters
- [ ] Performance comparison UI

### Future
- [ ] Live paper trading scheduler
- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation
- [ ] Real broker integration

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

- **Polygon.io** for crypto data
- **PySide6** for Qt bindings
- **PyQtGraph** for fast charting
 

---

## 📞 Support

For issues or questions:
- Check **AGENTS.md** for detailed guide
- Review **docs/STRATEGY_DEFAULTS_GUIDE.md** for behavior
- See **Troubleshooting** section above

---

## 🎯 Fitness Function Presets

### ⚖️ Balanced (Default)
Multi-objective optimization with balanced emphasis:
- Trade Count: 15% | Win Rate: 25% | Avg R: 30% | Total PnL: 20% | DD Penalty: 10%
- Min 10 trades, 40% win rate required
- **Use case**: General-purpose optimization

### 🚀 High Frequency (Scalping/Day Trading)
Maximizes trade count for active strategies:
- **Trade Count: 40%** | Win Rate: 15% | Avg R: 20% | Total PnL: 15% | DD Penalty: 10%
- Min 20 trades, 40% win rate required
- **Expected result**: 50-100+ trades (vs 10-20 with balanced)
- **Use case**: Scalpers and day traders wanting maximum activity

### 🛡️ Conservative (Quality over Quantity)
Prioritizes reliability and risk control:
- Trade Count: 5% | **Win Rate: 35%** | Avg R: 25% | Total PnL: 15% | **DD Penalty: 20%**
- Min 5 trades, **50% win rate required**
- **Expected result**: 60%+ win rate, minimal drawdowns
- **Use case**: Risk-averse traders prioritizing consistency

### 💰 Profit Focused (Maximum PnL)
Maximizes absolute returns:
- Trade Count: 10% | Win Rate: 20% | Avg R: 30% | **Total PnL: 30%** | DD Penalty: 10%
- Min 10 trades, 40% win rate required
- **Expected result**: 2-3x higher total PnL
- **Use case**: Aggressive profit maximization

### ✏️ Custom
User-defined weights for specific optimization goals.

---

**Version**: 2.2.0  
**Last Updated**: 2025-01-17  
**Status**: ✅ Production Ready ????????????????????????????????????
