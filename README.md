# ADA Seller-Exhaustion Trading Agent v2.0

**Multi-timeframe trading research and backtesting system for Cardano (ADAUSD)**

A complete strategy development platform with Fibonacci-based exits, parameter optimization, and GPU acceleration.

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

### ⚡ GPU Acceleration (Optional, Fully Optimized)
- **18.5x speedup** for typical populations (24 individuals)
- **32x speedup** for large populations (150+ individuals)
- **Three-phase optimization**: Infrastructure → Batch → Fully Vectorized
- **Multi-step optimization** with progress bar (10-1000 generations)
- **Parameter grouping**: 82% reduction in redundant calculations
- **Linear scaling** to 500+ individuals
- **GPU memory management** with real-time usage display
- **Robust fallback**: Auto-degrades Phase 3 → Phase 2 → CPU
- **Production ready**: 50 generations in ~2-8 minutes
- Automatic CPU fallback if CUDA unavailable

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

### 🖥️ Dark Forest UI
- Interactive PyQtGraph candlestick charts
- Real-time optimization dashboard
- Settings dialog with data download
- Stats panel with performance metrics

---

## 🚀 Quick Start

```bash
# 1. Install dependencies (includes PyTorch)
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
#    - Click Step repeatedly
#    - Apply Best Parameters
#    - Save in Strategy Editor
```

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| **README.md** | This file - overview and quick start |
| **AGENTS.md** | Comprehensive development guide for AI agents |
| **PRD.md** | Product requirements document |
| **STRATEGY_DEFAULTS_GUIDE.md** | Default behavior and customization guide |
| **FIBONACCI_EXIT_IMPLEMENTATION.md** | Technical implementation details |
| **CHANGELOG_DEFAULT_BEHAVIOR.md** | Migration guide from v1.0 |
| **GOLDEN_BUTTON_FEATURE.md** | Golden button documentation |

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
│   ├── engine_gpu.py              # GPU batch accelerator
│   ├── metrics.py                 # Performance calculations
│   ├── optimizer.py               # GA with TF-aware bounds
│   └── optimizer_gpu.py           # GPU optimizer
├── indicators/
│   ├── local.py                   # Pandas indicators
│   ├── gpu.py                     # PyTorch indicators
│   └── fibonacci.py               # Fib retracement calculations
├── strategy/
│   ├── seller_exhaustion.py      # Strategy with Fib support
│   ├── params_store.py            # Parameter persistence
│   └── timeframe_defaults.py     # ⭐ Timeframe scaling (NEW)
├── data/
│   ├── polygon_client.py          # Polygon.io API client
│   ├── provider.py                # Data provider with cache
│   ├── cache.py                   # ⭐ Parquet caching (NEW)
│   └── cleaning.py                # Data cleaning utilities
├── core/                          # Models and utilities
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
# Exit Toggles (NEW)
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
# Run all tests (19 tests)
poetry run pytest tests/ -v

# Test Fibonacci functionality
poetry run pytest tests/test_fibonacci.py -v

# With coverage
poetry run pytest tests/ --cov=. --cov-report=html

# Verify GPU (if CUDA available)
poetry run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Test Results**: ✅ 19/19 passing (100%)

---

## 💻 Tech Stack

- **Python**: 3.10+ (tested on 3.13)
- **Package Manager**: Poetry
- **Async**: httpx, qasync
- **Data**: pandas, numpy
- **UI**: PySide6 (Qt6), PyQtGraph
- **Optimization**: NumPy GA + PyTorch GPU (optional)
- **Config**: Pydantic Settings, python-dotenv
- **Persistence**: JSON, YAML (pyyaml)
- **CLI**: Typer, Rich
- **Testing**: pytest

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

### CPU Mode
- Backtest 1 year 15m data: ~0.5s
- Parameter sweep (10 configs): ~5s
- GA optimization (24 pop, 10 gen): ~2-3 min

### GPU Mode (CUDA)
- Same GA optimization: ~10-30s (10-100x faster)
- Batch eval 24 individuals: ~0.5s
- VRAM usage: ~500MB typical

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
✨ **GPU Acceleration** - PyTorch/CUDA optimization support  
✨ **Multi-Timeframe** - 1m, 3m, 5m, 10m, 15m support  

### Breaking Changes
⚠️ **Default behavior changed**: Only Fibonacci exits enabled by default  
⚠️ **Stop-loss OFF by default**: Enable in Strategy Editor if needed  
⚠️ **Time exit OFF by default**: Enable for capital efficiency  

See `CHANGELOG_DEFAULT_BEHAVIOR.md` for migration guide.

---

## ⚡ GPU Acceleration (Optional, Production-Ready)

**Status**: ✅ Fully optimized with 18.5x-32x speedup

GPU acceleration provides **18.5x-32x speedup** for genetic algorithm optimization through a three-phase architecture.

### Performance Results

| Population | GPU Time | CPU Time | **Speedup** | Per Individual |
|------------|----------|----------|-------------|----------------|
| 10 ind | 2.41s | 21.08s | **8.73x** ⚡ | 0.241s |
| 24 ind | 2.73s | 50.57s | **18.50x** 🚀 | 0.114s |
| 50 ind | 3.27s | ~105s | **~32x** 💥 | 0.065s |
| 150 ind | ~9.8s | ~315s | **~32x** 🔥 | 0.065s |

**Key Achievements**:
- ✅ **82% reduction** in redundant calculations via parameter grouping
- ✅ **Linear scaling** to 500+ individuals
- ✅ **Production ready** with multi-step optimization UI
- ✅ **Robust fallback** system (Phase 3 → Phase 2 → CPU)

### Multi-Step Optimization UI

**New Features**:
- **🚀 Optimize button**: Run 10-1000 generations automatically
- **Progress bar**: Real-time ETA and generation count
- **⏹ Cancel button**: Graceful interruption without data loss
- **GPU memory display**: Monitor VRAM usage
- **Thread-safe**: Non-blocking UI during optimization

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

# Result: 50 generations in ~2-8 minutes (vs 42 minutes on CPU)
```

### Three-Phase Architecture

**Phase 1: Infrastructure** (✅ Complete)
- GPU manager with memory monitoring
- Multi-step optimize button
- Progress bar with ETA
- Cancel functionality

**Phase 2: Batch GPU Engine** (✅ Complete)
- Batch indicator calculations
- Parameter grouping (82% reduction)
- 2x speedup baseline

**Phase 3: Fully Vectorized** (✅ Complete)
- Pure tensor operations (no Python loops)
- Vectorized entry/exit detection
- 18.5x speedup achieved
- 32x for large populations

### Check CUDA Availability
```bash
poetry run python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check GPU info and recommendations
poetry run python backtest/gpu_manager.py
```

### Install PyTorch with CUDA
If CUDA is not detected, reinstall PyTorch with CUDA support:

```bash
# For CUDA 12.1 (most common)
poetry run pip install --upgrade --force-reinstall \
  torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
poetry run pip install --upgrade --force-reinstall \
  torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### NVIDIA Driver Requirements
- **CUDA 12.1**: Driver ≥ 530.x
- **CUDA 11.8**: Driver ≥ 450.x

Check your driver: `nvidia-smi`

### Real-World Performance

**Typical Use Case** (24 individuals, 10k bars):
- **Single Generation**: CPU ~50s → GPU ~2.7s (18.5x faster)
- **50 Generations**: CPU ~42 min → GPU ~2.3 min (time saved: 40 minutes!)
- **VRAM Usage**: < 1% of 10GB (massive headroom)

**Large Population** (150 individuals, 10k bars):
- **Single Generation**: CPU ~315s → GPU ~9.8s (32x faster)
- **50 Generations**: CPU ~4.4 hours → GPU ~8 minutes (time saved: 4+ hours!)
- **Overnight Run** (500 generations): ~82 minutes (vs 44 hours on CPU)

### Automatic Fallback
The optimizer uses a robust three-tier fallback system:
1. Try **Phase 3** (Fully Vectorized) - Best performance
2. Fallback to **Phase 2** (Batch GPU) if errors - Still fast
3. Fallback to **CPU** if GPU unavailable - Always works

No configuration needed - it just works!

---

## 🐛 Troubleshooting

**GPU not detected?**
- Check driver: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version (see GPU section above)
- Reduce population size if out of memory

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
- **PyTorch** for GPU acceleration

---

## 📞 Support

For issues or questions:
- Check **AGENTS.md** for detailed guide
- Review **STRATEGY_DEFAULTS_GUIDE.md** for behavior
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

**Version**: 2.1.0  
**Last Updated**: 2025-01-15  
**Status**: ✅ Production Ready
