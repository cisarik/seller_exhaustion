# Strategy Default Behavior Guide

**Strategy Name**: Seller Exhaustion Trading Agent  
**Last Updated**: 2025-01-14  
**Default Version**: 2.0 (Fibonacci-only exits)

---

## 🎯 Default Strategy Behavior

### Entry Logic
**BUY Signals**: Only at seller exhaustion bottoms

Conditions that must ALL be true:
1. **Downtrend**: EMA_fast (96) < EMA_slow (672)
2. **Volume Spike**: Volume z-score > 2.0 (unusual selling pressure)
3. **Range Expansion**: True Range z-score > 1.2 (high volatility)
4. **Close Near High**: Close location > 0.6 (buyers stepping in, close in top 40% of candle)

**Entry Execution**: Next bar open after signal (t+1 open)

---

### Exit Logic  
**SELL Signals**: FIRST Fibonacci retracement level hit ONLY

**Default Exit Type**: Fibonacci retracement levels
- 38.2% Fib
- 50.0% Fib
- 61.8% Fib ← **Default target**
- 78.6% Fib
- 100% Fib (full retracement to swing high)

**How it works**:
1. When entry signal occurs, system finds most recent swing high
2. Calculates Fibonacci levels from signal low → swing high
3. Position exits at FIRST Fibonacci level price reaches
4. No stop-loss, no time limits

**Exit Reason Examples**:
- `fib_38.2` - Exited at 38.2% Fibonacci level
- `fib_61.8` - Exited at 61.8% Fibonacci level (most common)
- `fib_100.0` - Exited at 100% (full retracement)

---

## ⚙️ Optional Exits (DISABLED by default)

### Stop-Loss Exit
**Default**: ❌ DISABLED  
**Toggle**: `use_stop_loss = False`

When enabled:
- Stop placed at: `signal_low - (ATR × atr_stop_mult)`
- Default multiplier: 0.7
- Protects against major drawdowns
- Exit reason: `stop` or `stop_gap`

**When to enable**:
- High-risk markets
- Leveraged positions
- Want defined maximum loss per trade

---

### Traditional Take Profit Exit
**Default**: ❌ DISABLED  
**Toggle**: `use_traditional_tp = False`

When enabled:
- TP placed at: `entry + (reward_r × risk)`
- Default R-multiple: 2.0 (2:1 reward:risk)
- Fixed profit target regardless of market structure
- Exit reason: `tp`

**When to enable**:
- Want consistent R-multiple exits
- Fibonacci levels not available (insufficient history)
- Testing traditional strategies

---

### Time-Based Exit
**Default**: ❌ DISABLED  
**Toggle**: `use_time_exit = False`

When enabled:
- Maximum hold time: 96 bars (~24 hours on 15m timeframe)
- Forces exit if no other exit triggered
- Exit reason: `time`

**When to enable**:
- Mean-reversion strategy (expect quick moves)
- Want to free capital for new signals
- Avoid holding through low-probability zones

---

## 📊 Default Parameters Summary

### Strategy Parameters (SellerParams)
```python
ema_fast = 96           # ~1 day on 15m
ema_slow = 672          # ~7 days on 15m
z_window = 672          # Z-score lookback
atr_window = 96         # ATR calculation period
vol_z = 2.0             # Volume z-score threshold
tr_z = 1.2              # True Range z-score threshold
cloc_min = 0.6          # Close location minimum (60%)
```

### Backtest Parameters (BacktestParams)
```python
# Exit toggles
use_stop_loss = False          ❌ Stop-loss OFF
use_time_exit = False          ❌ Time exit OFF
use_fib_exits = True           ✅ Fibonacci ON (DEFAULT)
use_traditional_tp = False     ❌ Traditional TP OFF

# Fibonacci settings
fib_swing_lookback = 96        # Bars to search for swing high
fib_swing_lookahead = 5        # Confirmation period
fib_target_level = 0.618       # 61.8% (Golden Ratio)

# Optional exit settings (unused by default)
atr_stop_mult = 0.7           # If stop-loss enabled
reward_r = 2.0                 # If traditional TP enabled
max_hold = 96                  # If time exit enabled (~24h)

# Transaction costs (always applied)
fee_bp = 5.0                   # 0.05% fees
slippage_bp = 5.0              # 0.05% slippage
```

---

## 🔧 How to Change Defaults

### Via Strategy Editor UI

1. Launch application: `poetry run python cli.py ui`
2. Click **📊 Strategy Editor** in toolbar
3. Navigate to "Exit Strategy (toggles)" section
4. Check/uncheck desired exit types:
   - ✓ Use Fibonacci exits (DEFAULT) ← default ON
   - ☐ Use stop-loss (optional) ← default OFF
   - ☐ Use traditional TP (optional) ← default OFF
   - ☐ Use time-based exit (optional) ← default OFF
5. Adjust parameters for enabled exits
6. Click **💾 Save Params** to store configuration

### Via Code

```python
from backtest.engine import BacktestParams

# Default behavior (Fib-only exits)
params_default = BacktestParams()

# Enable stop-loss protection
params_with_stop = BacktestParams(
    use_stop_loss=True,        # Enable stop
    atr_stop_mult=0.7
)

# Enable time exit (for mean-reversion)
params_with_time = BacktestParams(
    use_time_exit=True,
    max_hold=48                # 12 hours on 15m
)

# Enable all exits (maximum protection)
params_all_exits = BacktestParams(
    use_stop_loss=True,
    use_time_exit=True,
    use_fib_exits=True,
    use_traditional_tp=False   # Usually don't need both Fib and TP
)

# Traditional strategy (no Fibonacci)
params_traditional = BacktestParams(
    use_fib_exits=False,       # Disable Fib
    use_stop_loss=True,
    use_traditional_tp=True,
    reward_r=2.0
)
```

---

## 📈 Expected Behavior

### With Default Settings (Fib-only)

**Typical trade flow**:
1. Entry signal detected at seller exhaustion bottom
2. Position opened at next bar open
3. Price rallies toward previous swing high
4. Position exits at first Fibonacci level hit (e.g., 61.8%)
5. **No stop-loss** → drawdowns possible if price continues down
6. **No time limit** → position can stay open indefinitely

**Advantages**:
- ✅ Exits at natural resistance (Fibonacci levels)
- ✅ Lets winners run (no premature time exit)
- ✅ Clean strategy logic (one entry type, one exit type)

**Risks**:
- ⚠️ No downside protection (no stop-loss)
- ⚠️ Capital tied up until Fib level hit
- ⚠️ If Fibonacci levels not available, position stays open

---

### When to Enable Optional Exits

#### Enable Stop-Loss If:
- Trading with leverage
- Want maximum drawdown control
- Risk management requirement
- Volatile market conditions

#### Enable Time Exit If:
- Trading mean-reversion strategies
- Want capital rotation for new signals
- Limited holding capacity
- Prefer shorter holding periods

#### Enable Traditional TP If:
- Want consistent R-multiples
- Fibonacci levels unreliable (choppy markets)
- Testing traditional strategy approaches
- Comparing against fixed-TP strategies

---

## 🧪 Testing Different Configurations

### Test 1: Default (Fib-only)
```python
params = BacktestParams()  # All defaults
result = run_backtest(feats, params)
```

### Test 2: Fib + Stop-Loss
```python
params = BacktestParams(use_stop_loss=True, atr_stop_mult=1.0)
result = run_backtest(feats, params)
```

### Test 3: Fib + Time Exit
```python
params = BacktestParams(use_time_exit=True, max_hold=48)
result = run_backtest(feats, params)
```

### Test 4: All Exits Enabled
```python
params = BacktestParams(
    use_stop_loss=True,
    use_time_exit=True,
    use_fib_exits=True,
    atr_stop_mult=0.7,
    max_hold=96
)
result = run_backtest(feats, params)
```

### Compare Results
```python
print(f"Default (Fib): {result_default['metrics']}")
print(f"With Stop: {result_stop['metrics']}")
print(f"With Time: {result_time['metrics']}")
print(f"All Exits: {result_all['metrics']}")
```

---

## 📝 Summary

**Default = Clean & Simple**:
- BUY at exhaustion bottoms
- SELL at Fibonacci resistance
- No stops, no time limits
- Let market structure guide exits

**Customizable = Your Choice**:
- Enable stop-loss for protection
- Enable time exit for capital efficiency
- Enable traditional TP for consistency
- Mix and match as needed

**Access via Strategy Editor**:
- Visual toggles for each exit type
- Detailed parameter explanations
- Save/load configurations
- Export to YAML for documentation

---

## 🔗 Related Documentation

- **FIBONACCI_EXIT_IMPLEMENTATION.md** - Technical details of Fibonacci implementation
- **AGENTS.md** - Full architecture and development guide
- **README.md** - Project overview and quick start
- **Strategy Editor UI** - In-app parameter explanations

---

**Questions?** All exit behavior is configurable via the Strategy Editor UI. Experiment with different combinations to find what works best for your trading style!
