# Spectre Trading Integration

## Overview

This document summarizes the implementation of an experimental Spectre trading runner for the Seller Exhaustion strategy. The integration provides an alternative to the event-driven CPU backtester while maintaining backward compatibility.

## What Was Implemented

### 1. Settings Toggle (`config/settings.py`)
- Added `use_spectre_trading: bool = False` to Settings model
- Persisted to `.env` file as `USE_SPECTRE_TRADING`
- Clearable from environment to prevent override

### 2. UI Checkbox (`app/widgets/settings_dialog.py`)
- Added checkbox in "⚡ Acceleration" tab under "Feature Engine" group
- Labeled: "Experimental: Use Spectre Trading API for backtests (CustomAlgorithm + blotter)"
- Tooltip explains the purpose and that optimization remains on CPU engine
- Loads/saves state from `.env` via SettingsManager
- Enabled only when Spectre is available

### 3. Spectre Trading Runner Module (`backtest/spectre_trading.py`)
New module providing:

#### Key Classes and Functions

- **`PositionState`**: Dataclass tracking open position metadata
  - entry_ts, entry_price, target_price, risk, bars_held, stop_price, tp_price

- **`df_to_memory_loader()`**: Converts simple DatetimeIndex DataFrame to Spectre format
  - Builds MultiIndex [date, asset]
  - Ensures UTC timezone
  - Creates MemoryLoader with explicit OHLCV specification

- **`inject_series_factor()`**: Injects external Series as SeriesDataFactor
  - Used for Fibonacci targets and other external data
  - Handles index alignment and format conversion

- **`build_seller_exhaustion_factors()`**: Constructs Spectre factors for entry signals
  - EMA (fast/slow), downtrend, ATR, volume/TR z-scores, close location
  - Returns dict of factor references for debugging

- **`SellerExhaustionAlg(trading.CustomAlgorithm)`**: Main trading algorithm
  - `initialize()`: Builds factors, configures blotter, schedules rebalance
  - `rebalance()`: Entry/exit logic per bar
  - `terminate()`: Cleanup logging
  - Supports optional GPU via `engine.to_cuda()`

- **`run_spectre_trading()`**: Main entry point
  - Accepts: df (raw OHLCV), seller_params, backtest_params, timeframe, use_cuda
  - Returns: dict with "trades" DataFrame and "metrics" dict (matches CPU engine)
  - Handles warmup calculation for Spectre factors
  - Computes Fibonacci targets preemptively
  - Records trades with entry_ts, exit_ts, entry, exit, pnl, R, reason, bars_held
  - Calculates: n, win_rate, avg_R, total_pnl, max_dd, sharpe

### 4. UI Integration (`app/main.py`)

#### Imports
- Added import for `run_spectre_trading` and `_SPECTRE_AVAILABLE`

#### run_backtest() Method
- Checks `settings.use_spectre_trading` flag
- Routes to appropriate engine:
  - If Spectre trading enabled AND available: passes raw OHLCV data
  - Otherwise: passes pre-computed features to CPU engine
- Applies same GPU flag from `settings.use_spectre_cuda`

#### _engine_label() Method
- Updated to reflect Spectre trading vs feature engine
- Shows "Spectre GPU" if trading + CUDA
- Shows "Spectre" if trading (CPU)
- Shows "Spectre (features)" for feature-only builds
- Falls back to "CPU" for standard mode

## Output Format Compatibility

The Spectre trading runner produces the same output structure as the CPU engine:

```python
{
    "trades": pd.DataFrame([
        {
            "entry_ts": "2024-01-15T10:30:00+00:00",  # ISO format string
            "exit_ts": "2024-01-15T14:45:00+00:00",
            "entry": 0.5123,
            "exit": 0.5234,
            "pnl": 0.0091,
            "R": 1.52,
            "reason": "FIB_61.8",
            "bars_held": 17
        },
        ...
    ]),
    "metrics": {
        "n": 45,                    # Total trades
        "win_rate": 0.56,          # % trades with pnl > 0
        "avg_R": 0.42,             # Average R-multiple
        "total_pnl": 0.1234,       # Sum of PnL
        "max_dd": -0.0456,         # Maximum drawdown
        "sharpe": 0.89             # Sharpe approximation
    }
}
```

This ensures seamless integration with existing UI (trade rendering, stats display, etc.).

## Design Decisions

### 1. Single-Asset Focus
- Implemented for X:ADAUSD by default
- Multi-asset extension straightforward via loop over asset IDs

### 2. Fibonacci Targets
- Precomputed on CPU using `add_fib_levels_to_df()`
- Strategy accesses via `row.get("fib_target", np.nan)`
- Avoids complexity of injecting time-series into Spectre mid-run

### 3. Commission/Slippage Mapping
- `fee_bp + slippage_bp` total for round trip
- Split per-side: `total_bp / 20000.0` as percentage
- Applied via `blotter.set_commission(percentage=side_pct)`

### 4. Entry/Exit Simulation
- **Entry**: At t+1 bar open after signal (simulates no lookahead)
- **Exit**: Intrabar high >= Fibonacci target (earliest possible price)
- **Other exits**: Can be extended (stop-loss, traditional TP, time exit)

### 5. GPU Optional
- CUDA enabled only if `settings.use_spectre_cuda = True` AND GPU available
- Fallback to CPU silently if CUDA requested but unavailable
- Controlled via `engine.to_cuda()` in algorithm.initialize()

## Known Limitations & Future Work

### Current Limitations
1. ~~**Fibonacci Injection**~~: **RESOLVED** - Uses pre-computed CPU lookup dict instead of Spectre factors
   - Avoided DatetimeIndex serialization issues
   - Fast O(1) lookup during trading simulation

2. ~~**tqdm dependency**~~: **RESOLVED** - Added to poetry.lock
   - Required by Spectre's progress bars during run_backtest

3. ~~**Volume type**~~: **RESOLVED** - Automatically converts to float for Spectre compatibility
   - Spectre tensor operations require float32, not int64

4. **Single-Direction Exits**: Currently exits only on Fibonacci targets
   - **Future**: Extend SellerExhaustionAlg.rebalance() to support stop-loss, traditional TP, time-based exits
   - Can mirror BacktestParams toggles: use_stop_loss, use_traditional_tp, use_time_exit

5. **No Position Sizing**: Fixed 100% position weight
   - **Future**: Add variable sizing via capital allocation logic

### Possible Extensions
- [ ] Support multi-timeframe backtests
- [ ] Extend to multiple assets simultaneously
- [ ] Add realistic market impact / partial fills
- [ ] Compare Spectre vs CPU performance on same data
- [ ] Parity testing framework for validation

## Testing

### Basic Test
```bash
cd /home/agile/seller_exhaustion
poetry run python3 << 'EOF'
from backtest.spectre_trading import run_spectre_trading, _SPECTRE_AVAILABLE
from strategy.seller_exhaustion import SellerParams
from core.models import BacktestParams, Timeframe
import pandas as pd
import numpy as np

# Create synthetic data...
df = pd.DataFrame({...})

# Run Spectre backtest
result = run_spectre_trading(df, SellerParams(), BacktestParams(), Timeframe.m15)
print(f"Trades: {result['metrics']['n']}")
EOF
```

### UI Test
1. Launch UI: `poetry run python cli.py ui`
2. Open Settings → ⚡ Acceleration
3. Check "Experimental: Use Spectre Trading API for backtests"
4. Save settings
5. Download data and run backtest
6. Verify trades render correctly

## Logging

All Spectre trading activity logged via `core.logging_utils.get_logger(__name__)`:

```python
logger.info("Spectre trading run | bars=1000 | asset=X:ADAUSD | tf=15m")
logger.info("Spectre engine moved to CUDA")
logger.debug("Entry at 2024-01-15T10:30 | price=0.5123 | target=0.5234")
logger.debug("Exit at 2024-01-15T14:45 | price=0.5234 | pnl=0.0091 | reason=FIB_61.8")
logger.info("Spectre trading complete | trades=45 | pnl=0.1234 | sharpe=0.89")
```

## Files Modified

- `config/settings.py`: Added use_spectre_trading setting
- `app/widgets/settings_dialog.py`: Added UI checkbox + load/save logic
- `app/main.py`: Added routing logic + _engine_label() update
- `backtest/spectre_trading.py`: NEW - Main implementation

## Backward Compatibility

✅ **Fully backward compatible:**
- Toggle is OFF by default (CPU engine remains default)
- No changes to existing APIs or data structures
- UI gracefully handles missing Spectre
- Optimization always uses CPU engine (proven, stable path)
