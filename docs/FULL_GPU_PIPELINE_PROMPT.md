# Expert Prompt: Full GPU Pipeline Implementation

## 🎯 Mission

Implement a **fully GPU-parallelized backtesting pipeline** that achieves **95-100% GPU utilization** while maintaining **100% consistency with CPU results**.

**Current Status**: GPU validated but uses CPU logic (40-60% utilization)  
**Target**: Full GPU pipeline (95-100% utilization)  
**Non-Negotiable**: ZERO trade count divergence from CPU

---

## ⚠️ CRITICAL LESSONS FROM PAST FAILURES

### The Bug That Caused 40 vs 14 Trade Mismatch

**Three fatal bugs were discovered and fixed. You MUST NOT repeat them:**

#### Bug #1: Signal Detection Divergence
**What happened**: GPU calculated indicators independently from CPU  
**Result**: Different signals (GPU started Oct 17, CPU started Oct 25 - 8 days difference!)

**Root Cause**:
```python
# WRONG (GPU did this):
indicators = batch_calculate_indicators(...)  # Custom GPU calculation
signals = batch_detect_signals(indicators, ...)  # Different from CPU!

# CORRECT (What CPU does):
feats = build_features(data, params, tf)  # Handles NaN, warmup, everything
signals = feats[feats['exhaustion'] == True]
```

**YOUR TASK**: Make GPU use the EXACT same indicator calculation as `build_features()`, or prove mathematically they're equivalent.

#### Bug #2: NaN Handling
**What happened**: GPU processed raw OHLCV including warmup period with NaN values  
**Result**: GPU used incomplete indicators for early signals

**Root Cause**:
```python
# WRONG (GPU):
# No NaN handling - processes ALL rows

# CORRECT (CPU):
d = df.dropna(subset=["atr"]).copy()  # Removes warmup rows
```

**YOUR TASK**: Implement the exact same NaN dropping logic, or handle NaN in indicators properly.

#### Bug #3: Position Management
**What happened**: GPU set `in_position = True` before finding exit  
**Result**: Multiple overlapping positions (40 trades vs 14)

**Root Cause**:
```python
# WRONG (GPU did this):
in_position = True        # Set BEFORE finding exit
# ... (exit search happens here) ...
if exit_idx is not None:  # Exit might be None!
    in_position = False   # Only reset if exit found
# BUG: Can enter new position while still in one!

# CORRECT (CPU):
in_pos = True   # Enter position
# ... find exit ...
if exit_price is not None:
    in_pos = False  # Exit
# Can't enter another until in_pos == False
```

**YOUR TASK**: Ensure ONE position at a time, no overlaps.

---

## 📋 Implementation Strategy

### Phase 1: GPU Indicator Calculation (70-80% utilization)

**Goal**: Parallelize `build_features()` on GPU

**Current**:
```python
# CPU (sequential)
for sp in seller_params_list:
    feats = build_features(data.copy(), sp, tf)  # ~500ms each
```

**Target**:
```python
# GPU (parallel)
feats_list = build_features_gpu_batch(data, seller_params_list, tf)  # ~100ms total
```

#### Implementation Steps

1. **Create `strategy/seller_exhaustion_gpu.py`**:
```python
def build_features_gpu_batch(
    data: pd.DataFrame,
    seller_params_list: List[SellerParams],
    tf: Timeframe,
    device: torch.device
) -> List[pd.DataFrame]:
    """
    Build features for ALL individuals in parallel on GPU.
    
    CRITICAL: Must produce IDENTICAL results to CPU build_features()
    """
    n_individuals = len(seller_params_list)
    n_bars = len(data)
    
    # Convert to GPU tensors ONCE
    close_t = torch.tensor(data['close'].values, device=device, dtype=torch.float32)
    high_t = torch.tensor(data['high'].values, device=device, dtype=torch.float32)
    low_t = torch.tensor(data['low'].values, device=device, dtype=torch.float32)
    volume_t = torch.tensor(data['volume'].values, device=device, dtype=torch.float32)
    
    # Group by unique parameter values to avoid redundant calculations
    # (This is the KEY optimization!)
    
    ema_fast_groups = {}
    for i, p in enumerate(seller_params_list):
        if p.ema_fast not in ema_fast_groups:
            ema_fast_groups[p.ema_fast] = []
        ema_fast_groups[p.ema_fast].append(i)
    
    # Calculate each unique EMA fast only ONCE
    ema_fast_results = {}
    for ema_val, indices in ema_fast_groups.items():
        ema_fast_results[ema_val] = ema_gpu(close_t, ema_val)
    
    # Assign to individuals
    ema_fast_all = torch.zeros((n_individuals, n_bars), device=device)
    for ema_val, indices in ema_fast_groups.items():
        for i in indices:
            ema_fast_all[i] = ema_fast_results[ema_val]
    
    # Repeat for ema_slow, atr, vol_z, tr_z, cloc
    # ...
    
    # Convert back to pandas DataFrames with EXACT same structure as CPU
    results = []
    for i in range(n_individuals):
        # Create DataFrame matching CPU output exactly
        feats = data.copy()
        feats['ema_f'] = ema_fast_all[i].cpu().numpy()
        feats['ema_s'] = ema_slow_all[i].cpu().numpy()
        # ... all features ...
        
        # CRITICAL: Apply same NaN handling as CPU!
        feats = feats.dropna(subset=["atr"])
        
        results.append(feats)
    
    return results
```

2. **Validation**:
```python
# tests/test_gpu_indicators.py
def test_gpu_vs_cpu_indicators():
    data = fetch_real_data()  # Use REAL data
    params_list = [SellerParams(), SellerParams(ema_fast=100), ...]
    
    # CPU
    cpu_results = [build_features(data.copy(), p, tf) for p in params_list]
    
    # GPU
    gpu_results = build_features_gpu_batch(data, params_list, tf, device)
    
    # Compare EVERY value
    for i, (cpu_feat, gpu_feat) in enumerate(zip(cpu_results, gpu_results)):
        # Trade count must match
        assert len(cpu_feat) == len(gpu_feat)
        
        # Signals must match
        cpu_signals = cpu_feat[cpu_feat['exhaustion'] == True]
        gpu_signals = gpu_feat[gpu_feat['exhaustion'] == True]
        assert len(cpu_signals) == len(gpu_signals)
        
        # Timestamps must match exactly
        assert all(cpu_signals.index == gpu_signals.index)
        
        # Indicator values must be close (allow float32 tolerance)
        for col in ['ema_f', 'ema_s', 'atr', 'vol_z', 'tr_z']:
            np.testing.assert_allclose(
                cpu_feat[col].values,
                gpu_feat[col].values,
                rtol=1e-5,
                err_msg=f"Individual {i}, column {col} mismatch"
            )
```

**CRITICAL SUCCESS CRITERIA**:
- ✅ Same number of signals
- ✅ Same signal timestamps
- ✅ Indicator values within 1e-5 tolerance
- ✅ NaN rows dropped identically

### Phase 2: GPU Vectorized Exits (85-95% utilization)

**Goal**: Vectorize exit detection on GPU

**Current**:
```python
# CPU (sequential per trade)
for trade in trades:
    for j in range(entry_idx, entry_idx + max_hold):
        if low[j] <= stop: exit
        if high[j] >= tp: exit
```

**Target**:
```python
# GPU (vectorized across all trades)
exits = find_exits_vectorized_gpu(entries, stops, tps, high_t, low_t, max_hold)
```

#### Implementation

```python
def find_exits_vectorized_gpu(
    entry_indices: torch.Tensor,  # [N_trades]
    stop_prices: torch.Tensor,    # [N_trades]
    tp_prices: torch.Tensor,      # [N_trades]
    high_t: torch.Tensor,         # [N_bars]
    low_t: torch.Tensor,          # [N_bars]
    max_hold: int,
    device: torch.device
) -> dict:
    """
    Find exits for ALL trades in parallel.
    
    CRITICAL: Exit priority must match CPU exactly:
    1. Stop gap (open <= stop)
    2. Stop hit (low <= stop)
    3. TP hit (high >= tp)
    4. Time exit (max_hold)
    """
    n_trades = len(entry_indices)
    
    exit_indices = torch.full((n_trades,), -1, device=device, dtype=torch.long)
    exit_prices = torch.zeros(n_trades, device=device)
    exit_reasons = torch.zeros(n_trades, device=device, dtype=torch.long)
    
    # For each trade, create search range and check conditions
    # This is complex - need careful indexing!
    
    # Vectorized approach (tricky but possible):
    for trade_idx in range(n_trades):
        entry_idx = entry_indices[trade_idx].item()
        end_idx = min(entry_idx + max_hold, len(high_t))
        
        # Get price slices for this trade
        highs = high_t[entry_idx:end_idx]
        lows = low_t[entry_idx:end_idx]
        
        stop = stop_prices[trade_idx]
        tp = tp_prices[trade_idx]
        
        # Find first exit (priority matters!)
        stop_hit_mask = lows <= stop
        tp_hit_mask = highs >= tp
        
        if stop_hit_mask.any():
            exit_bar = stop_hit_mask.nonzero()[0].item()
            exit_indices[trade_idx] = entry_idx + exit_bar
            exit_prices[trade_idx] = stop
            exit_reasons[trade_idx] = 1  # stop
        elif tp_hit_mask.any():
            exit_bar = tp_hit_mask.nonzero()[0].item()
            exit_indices[trade_idx] = entry_idx + exit_bar
            exit_prices[trade_idx] = tp
            exit_reasons[trade_idx] = 2  # tp
        else:
            # Time exit
            exit_indices[trade_idx] = end_idx - 1
            exit_prices[trade_idx] = high_t[end_idx - 1]  # Use close in real impl
            exit_reasons[trade_idx] = 3  # time
    
    return {
        'exit_indices': exit_indices,
        'exit_prices': exit_prices,
        'exit_reasons': exit_reasons
    }
```

**CRITICAL**: This still has a loop! True vectorization would eliminate it, but that's extremely complex. Benchmark this first - if it's fast enough, keep the loop.

### Phase 3: Full Pipeline Integration (95-100% utilization)

**Goal**: Keep EVERYTHING on GPU until final results

```python
def batch_backtest_full_gpu(
    data: pd.DataFrame,
    seller_params_list: List[SellerParams],
    backtest_params_list: List[BacktestParams],
    tf: Timeframe,
    device: torch.device
) -> List[Dict]:
    """
    FULLY GPU-parallelized backtest pipeline.
    
    Flow:
    1. Convert OHLCV to GPU tensors (ONCE)
    2. Calculate indicators on GPU (parallel across individuals)
    3. Detect signals on GPU (vectorized)
    4. Find entries/exits on GPU (vectorized)
    5. Calculate PnL on GPU (vectorized)
    6. Convert final results to CPU (only at the end!)
    """
    n_individuals = len(seller_params_list)
    
    # Step 1: Tensors (ONCE)
    close_t, high_t, low_t, open_t, volume_t = data_to_tensors_gpu(data, device)
    
    # Step 2: Indicators (parallel)
    indicators = calculate_indicators_gpu_batch(
        close_t, high_t, low_t, volume_t,
        seller_params_list,
        device
    )
    
    # Step 3: Signals (vectorized)
    signals = detect_signals_gpu_batch(indicators, seller_params_list, device)
    # signals shape: [n_individuals, n_bars] (boolean)
    
    # Step 4: Entries/Exits (vectorized per individual)
    trades_list = []
    for i in range(n_individuals):
        signal_indices = signals[i].nonzero().squeeze()
        
        if len(signal_indices) == 0:
            trades_list.append([])
            continue
        
        # Find entries (t+1 open after signal)
        entry_indices = signal_indices + 1
        entry_prices = open_t[entry_indices]
        
        # Calculate stops/TPs
        bp = backtest_params_list[i]
        atr_vals = indicators['atr'][i][signal_indices]
        stop_prices = low_t[signal_indices] - bp.atr_stop_mult * atr_vals
        risks = entry_prices - stop_prices
        tp_prices = entry_prices + bp.reward_r * risks
        
        # Find exits (GPU vectorized)
        exits = find_exits_vectorized_gpu(
            entry_indices, stop_prices, tp_prices,
            high_t, low_t, bp.max_hold, device
        )
        
        # Calculate PnL (GPU vectorized)
        pnls = calculate_pnl_gpu(
            entry_prices, exits['exit_prices'],
            risks, bp.fee_bp, bp.slippage_bp
        )
        
        trades_list.append({
            'entry_indices': entry_indices.cpu().numpy(),
            'exit_indices': exits['exit_indices'].cpu().numpy(),
            'pnls': pnls.cpu().numpy(),
            # ... all trade data ...
        })
    
    return trades_list
```

---

## 🧪 Validation Strategy (MANDATORY)

**DO NOT SKIP ANY OF THESE STEPS**:

### Step 1: Unit Tests for Each Component

```python
# Test indicators match CPU
def test_ema_gpu_vs_cpu()
def test_atr_gpu_vs_cpu()
def test_zscore_gpu_vs_cpu()

# Test signal detection matches
def test_signals_gpu_vs_cpu()

# Test exit detection matches
def test_exits_gpu_vs_cpu()
```

### Step 2: Integration Test

```python
# tests/test_full_gpu_pipeline.py
def test_full_pipeline_vs_cpu():
    # Use REAL Polygon.io data (same as validation tests)
    data = fetch_real_data("X:ADAUSD", "2024-10-01", "2024-12-31")
    
    # Test with varied parameters
    params_list = create_test_parameter_sets(n=24)
    
    # CPU
    cpu_results = [run_cpu_backtest(data, sp, bp) for sp, bp in params_list]
    
    # GPU
    gpu_results = batch_backtest_full_gpu(data, [sp for sp,_ in params_list], 
                                           [bp for _,bp in params_list])
    
    # CRITICAL: Trade counts MUST match
    for i, (cpu_res, gpu_res) in enumerate(zip(cpu_results, gpu_results)):
        assert cpu_res['metrics']['n'] == gpu_res['metrics']['n'], \
            f"Individual {i}: CPU {cpu_res['metrics']['n']} trades, GPU {gpu_res['metrics']['n']} trades"
```

### Step 3: Regression Suite

**Run the EXISTING validation tests**:
```bash
poetry run pytest tests/test_gpu_cpu_validation.py -v
```

If this fails, YOU BROKE SOMETHING. Do not proceed.

---

## 🎓 Key Learnings to Apply

1. **Correctness > Speed**: Always validate before optimizing
2. **Use Real Data**: Synthetic data hides bugs
3. **Compare Everything**: Trade counts, timestamps, PnL, exit reasons
4. **Group Operations**: Calculate each unique parameter value once
5. **Mind the Dtypes**: float32 vs float64 can cause divergence
6. **Test Incrementally**: Don't implement all 3 phases at once
7. **Keep CPU Fallback**: Always have a safety net

---

## 📊 Expected Results

If implementation is correct:

| Metric | Before | After |
|--------|--------|-------|
| GPU Utilization | 40-60% | 95-100% |
| Time/Generation (24 ind) | ~5-7s | ~1-2s |
| Speedup | 1.4x | 25-50x |
| Trade Count Match | ✅ 100% | ✅ 100% |

**If trade counts don't match, STOP and debug. Speed means nothing if results are wrong.**

---

## 🚨 Red Flags to Watch For

1. **Different signal counts** → Indicator calculation mismatch
2. **Same signals, different exits** → Exit priority bug
3. **Close PnL, wrong trade count** → Position overlap bug
4. **Works on small data, fails on large** → Memory/indexing bug
5. **Intermittent failures** → Race condition or device sync issue

---

## 💾 Final Integration

Once validated, integrate into `BatchGPUBacktestEngine`:

```python
# In engine_gpu_batch.py
class BatchGPUBacktestEngine:
    def batch_backtest(self, ...):
        # Add new parameter: use_full_pipeline=False
        
        if use_full_pipeline:
            # NEW: Full GPU pipeline (95-100% utilization)
            return self._batch_backtest_full_gpu(...)
        else:
            # CURRENT: CPU logic (40-60% utilization, VALIDATED)
            return self._cpu_compatible_backtest(...)
```

**Start with `use_full_pipeline=False` by default. Only enable after 100% validation.**

---

## ✅ Success Criteria

You have succeeded when:

1. ✅ `test_full_pipeline_vs_cpu()` passes with 100% trade count match
2. ✅ Existing validation tests still pass
3. ✅ GPU utilization reaches 90%+
4. ✅ Speedup is 20x+ over current GPU implementation
5. ✅ Zero divergence on 10 different parameter sets
6. ✅ Works with population sizes from 1 to 512
7. ✅ Fallback to CPU works if GPU fails

**If even ONE test fails, you have NOT succeeded. Fix it before declaring victory.**

---

**This is a complex but achievable task. The key is NEVER sacrificing correctness for speed.**

**Good luck! 🚀**
