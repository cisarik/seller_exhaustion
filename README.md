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

### 🎨 Strategy Editor (**NEW**)
- Comprehensive parameter management with detailed explanations
- **⭐ Golden Button**: One-click setup for optimal 61.8% Fibonacci target
- Exit toggles for stop-loss, time, and TP exits
- Save/load evolved parameters from genetic algorithm
- Export to YAML for documentation

### 📊 Fibonacci Exit System (**NEW**)
- Market-driven exits at natural resistance levels
- Automatic swing high detection
- Configurable lookback/lookahead periods
- Exit at 38.2%, 50%, 61.8%, 78.6%, or 100% retracement

### 💾 Parameter Persistence (**NEW**)
- Save configurations with metadata (generation, fitness, date)
- Load parameter sets with one click
- Browse saved configurations
- Export to JSON/YAML

### ⚡ GPU Acceleration (**NEW**, Optional)
- PyTorch/CUDA support for genetic algorithm
- 10-100x speedup for optimization
- Automatic CPU fallback if CUDA unavailable
- Memory management utilities

### 📈 Multi-Timeframe Support (**NEW**)
- 1m, 3m, 5m, 10m, 15m timeframes
- Bar-based and time-based parameter conversion
- Consistent strategy across timeframes

### 🎛️ Genetic Algorithm Optimizer
- Population-based parameter search
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
│   ├── main.py                    # Main window
│   ├── theme.py                   # Dark Forest theme
│   └── widgets/
│       ├── candle_view.py         # Candlestick chart
│       ├── settings_dialog.py     # Settings & data download
│       ├── stats_panel.py         # Optimization dashboard
│       └── strategy_editor.py     # ⭐ Parameter editor (NEW)
├── backtest/
│   ├── engine.py                  # CPU backtest with exit toggles
│   ├── engine_gpu.py              # GPU batch accelerator (NEW)
│   ├── metrics.py                 # Performance calculations
│   ├── optimizer.py               # Genetic algorithm CPU (NEW)
│   └── optimizer_gpu.py           # GPU optimizer (NEW)
├── indicators/
│   ├── local.py                   # Pandas indicators
│   ├── gpu.py                     # PyTorch indicators (NEW)
│   └── fibonacci.py               # Fib calculations (NEW)
├── strategy/
│   ├── seller_exhaustion.py      # Strategy with Fib support
│   └── params_store.py            # Parameter persistence (NEW)
├── data/                          # Polygon.io data fetching
├── core/                          # Models and utilities
├── config/                        # Settings management
├── tests/                         # 19 tests, all passing ✅
└── cli.py                         # CLI commands
```

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

## 🆕 What's New in v2.0

### Major Features
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

## ⚡ GPU Acceleration (Optional)

GPU acceleration can provide 10-100x speedup for genetic algorithm optimization.

### Check CUDA Availability
```bash
poetry run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
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

### Performance Benchmarks
- **CPU**: GA optimization (24 pop, 10 gen) → 2-3 minutes
- **GPU**: Same optimization → 10-30 seconds
- **VRAM**: ~500MB typical usage

### Automatic Fallback
If CUDA is unavailable, the optimizer automatically falls back to CPU mode. No configuration needed.

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

**Version**: 2.0.0  
**Last Updated**: 2025-01-14  
**Status**: ✅ Production Ready
