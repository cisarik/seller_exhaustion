"""
GPU-accelerated seller exhaustion feature building.

CRITICAL: Must produce IDENTICAL results to CPU build_features() to avoid
the 40 vs 14 trade mismatch bug.

Key Requirements:
1. Same indicator calculations as indicators/local.py
2. Same NaN handling as build_features()
3. Same signal detection logic
4. Float32 precision acceptable (validated < 1e-5 tolerance)

Performance Target: 70-80% GPU utilization in Phase 1
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from core.models import Timeframe
from strategy.seller_exhaustion import SellerParams
from indicators.gpu import (
    get_device, to_tensor, to_numpy,
    ema_gpu, atr_gpu, zscore_gpu, sma_gpu
)


@dataclass
class GPUFeatureBuildStats:
    """Statistics for GPU feature building performance."""
    n_individuals: int
    n_bars: int
    unique_ema_fast: int
    unique_ema_slow: int
    unique_z_window: int
    unique_atr_window: int
    total_time: float
    gpu_utilization: float = 0.0
    
    def __repr__(self):
        return (f"GPUFeatureBuildStats(individuals={self.n_individuals}, "
                f"bars={self.n_bars}, unique_params={self.unique_ema_fast}/"
                f"{self.unique_ema_slow}/{self.unique_z_window}/{self.unique_atr_window}, "
                f"time={self.total_time:.3f}s, gpu_util={self.gpu_utilization:.1%})")


def build_features_gpu_batch(
    data: pd.DataFrame,
    seller_params_list: List[SellerParams],
    tf: Timeframe = Timeframe.m15,
    device: Optional[torch.device] = None,
    add_fib: bool = False,  # Fibonacci calculation still on CPU for now
    fib_lookback: int = 96,
    fib_lookahead: int = 5,
    verbose: bool = False
) -> Tuple[List[pd.DataFrame], GPUFeatureBuildStats]:
    """
    Build features for ALL individuals in parallel on GPU.
    
    CRITICAL: Must produce IDENTICAL results to CPU build_features()
    
    Strategy:
    1. Convert OHLCV to GPU tensors ONCE
    2. Group by unique parameter values
    3. Calculate each unique indicator value ONCE
    4. Assign to individuals (indexing, no redundant computation)
    5. Convert back to pandas with EXACT same structure as CPU
    6. Apply EXACT same NaN handling as CPU
    
    Args:
        data: Raw OHLCV DataFrame
        seller_params_list: List of SellerParams (one per individual)
        tf: Timeframe
        device: PyTorch device (auto-detect if None)
        add_fib: Add Fibonacci levels (CPU fallback for now)
        fib_lookback: Fibonacci lookback window
        fib_lookahead: Fibonacci lookahead window
        verbose: Print detailed progress
    
    Returns:
        (results_list, stats) tuple:
        - results_list: List of DataFrames with features (one per individual)
        - stats: Performance statistics
    """
    import time
    start_time = time.time()
    
    if device is None:
        device = get_device()
    
    n_individuals = len(seller_params_list)
    n_bars = len(data)
    
    if verbose:
        print(f"\n🚀 GPU Feature Building: {n_individuals} individuals × {n_bars} bars")
        print(f"   Device: {device}")
    
    # ========================================================================
    # Step 1: Convert OHLCV to GPU tensors ONCE
    # ========================================================================
    
    if verbose:
        print(f"   [1/6] Converting OHLCV to GPU tensors...")
    
    close_t = to_tensor(data['close'], device)
    high_t = to_tensor(data['high'], device)
    low_t = to_tensor(data['low'], device)
    volume_t = to_tensor(data['volume'], device)
    
    # ========================================================================
    # Step 2: Group by unique parameter values
    # ========================================================================
    
    if verbose:
        print(f"   [2/6] Grouping by unique parameter values...")
    
    # Resolve time-based parameters to bar counts
    from core.models import minutes_to_bars
    
    resolved_params = []
    for p in seller_params_list:
        ema_fast_bars = minutes_to_bars(p.ema_fast_minutes, tf) if p.ema_fast_minutes is not None else p.ema_fast
        ema_slow_bars = minutes_to_bars(p.ema_slow_minutes, tf) if p.ema_slow_minutes is not None else p.ema_slow
        z_window_bars = minutes_to_bars(p.z_window_minutes, tf) if p.z_window_minutes is not None else p.z_window
        atr_window_bars = minutes_to_bars(p.atr_window_minutes, tf) if p.atr_window_minutes is not None else p.atr_window
        
        resolved_params.append({
            'ema_fast': ema_fast_bars,
            'ema_slow': ema_slow_bars,
            'z_window': z_window_bars,
            'atr_window': atr_window_bars,
            'vol_z': p.vol_z,
            'tr_z': p.tr_z,
            'cloc_min': p.cloc_min
        })
    
    # Group indices by unique parameter values
    ema_fast_groups: Dict[int, List[int]] = {}
    ema_slow_groups: Dict[int, List[int]] = {}
    z_window_groups: Dict[int, List[int]] = {}
    atr_window_groups: Dict[int, List[int]] = {}
    
    for i, rp in enumerate(resolved_params):
        # EMA Fast
        if rp['ema_fast'] not in ema_fast_groups:
            ema_fast_groups[rp['ema_fast']] = []
        ema_fast_groups[rp['ema_fast']].append(i)
        
        # EMA Slow
        if rp['ema_slow'] not in ema_slow_groups:
            ema_slow_groups[rp['ema_slow']] = []
        ema_slow_groups[rp['ema_slow']].append(i)
        
        # Z-Window
        if rp['z_window'] not in z_window_groups:
            z_window_groups[rp['z_window']] = []
        z_window_groups[rp['z_window']].append(i)
        
        # ATR Window
        if rp['atr_window'] not in atr_window_groups:
            atr_window_groups[rp['atr_window']] = []
        atr_window_groups[rp['atr_window']].append(i)
    
    if verbose:
        print(f"      Unique EMA Fast: {len(ema_fast_groups)}")
        print(f"      Unique EMA Slow: {len(ema_slow_groups)}")
        print(f"      Unique Z-Window: {len(z_window_groups)}")
        print(f"      Unique ATR Window: {len(atr_window_groups)}")
    
    # ========================================================================
    # Step 3: Calculate unique indicator values ONCE each
    # ========================================================================
    
    if verbose:
        print(f"   [3/6] Calculating unique indicator values on GPU...")
    
    # EMA Fast (calculate once per unique value)
    ema_fast_cache: Dict[int, torch.Tensor] = {}
    for span in ema_fast_groups.keys():
        ema_fast_cache[span] = ema_gpu(close_t, span)
    
    # EMA Slow
    ema_slow_cache: Dict[int, torch.Tensor] = {}
    for span in ema_slow_groups.keys():
        ema_slow_cache[span] = ema_gpu(close_t, span)
    
    # ATR (depends on window)
    atr_cache: Dict[int, torch.Tensor] = {}
    for window in atr_window_groups.keys():
        atr_cache[window] = atr_gpu(high_t, low_t, close_t, window)
    
    # Volume z-score (depends on z_window)
    vol_z_cache: Dict[int, torch.Tensor] = {}
    
    for window in z_window_groups.keys():
        vol_z_cache[window] = zscore_gpu(volume_t, window)
    
    # TR z-score calculation - CRITICAL: Must match CPU exactly!
    # CPU calculates: tr = out["atr"] * atr_window_bars
    # Then: out["tr_z"] = zscore(tr, z_window_bars)
    #
    # This means we need ATR * atr_window for each individual's atr_window,
    # then calculate z-score with their z_window
    #
    # Strategy: Build TR for each unique (atr_window, z_window) combination
    
    # Find unique (atr_window, z_window) combinations
    tr_z_combinations = {}
    for i, rp in enumerate(resolved_params):
        key = (rp['atr_window'], rp['z_window'])
        if key not in tr_z_combinations:
            tr_z_combinations[key] = []
        tr_z_combinations[key].append(i)
    
    # Calculate TR z-score for each unique combination
    tr_z_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    
    for (atr_win, z_win), indices in tr_z_combinations.items():
        # Get ATR for this window (already calculated)
        atr_vals = atr_cache[atr_win]
        
        # Calculate TR = ATR * atr_window (matching CPU exactly!)
        tr = atr_vals * float(atr_win)
        
        # Calculate z-score of TR
        tr_z_cache[(atr_win, z_win)] = zscore_gpu(tr, z_win)
    
    # Close location (universal, doesn't depend on params)
    span = (high_t - low_t)
    # Replace 0 with nan (same as CPU)
    span = torch.where(span == 0, torch.tensor(float('nan'), device=device), span)
    cloc = (close_t - low_t) / span
    
    if verbose:
        print(f"      ✓ Cached {len(ema_fast_cache)} EMA Fast values")
        print(f"      ✓ Cached {len(ema_slow_cache)} EMA Slow values")
        print(f"      ✓ Cached {len(atr_cache)} ATR values")
        print(f"      ✓ Cached {len(vol_z_cache)} Vol Z values")
        print(f"      ✓ Cached {len(tr_z_cache)} TR Z values")
    
    # ========================================================================
    # Step 4: Assign to individuals (no redundant computation!)
    # ========================================================================
    
    if verbose:
        print(f"   [4/6] Assigning indicators to individuals...")
    
    # Allocate output tensors (shape: [n_individuals, n_bars])
    ema_f_all = torch.zeros((n_individuals, n_bars), device=device, dtype=torch.float32)
    ema_s_all = torch.zeros((n_individuals, n_bars), device=device, dtype=torch.float32)
    atr_all = torch.zeros((n_individuals, n_bars), device=device, dtype=torch.float32)
    vol_z_all = torch.zeros((n_individuals, n_bars), device=device, dtype=torch.float32)
    tr_z_all = torch.zeros((n_individuals, n_bars), device=device, dtype=torch.float32)
    
    # Assign from cache (pure indexing, no computation)
    for i, rp in enumerate(resolved_params):
        ema_f_all[i] = ema_fast_cache[rp['ema_fast']]
        ema_s_all[i] = ema_slow_cache[rp['ema_slow']]
        atr_all[i] = atr_cache[rp['atr_window']]
        vol_z_all[i] = vol_z_cache[rp['z_window']]
        
        # TR z-score uses combination key
        tr_key = (rp['atr_window'], rp['z_window'])
        tr_z_all[i] = tr_z_cache[tr_key]
    
    # ========================================================================
    # Step 5: Generate signals on GPU (vectorized!)
    # ========================================================================
    
    if verbose:
        print(f"   [5/6] Generating signals on GPU...")
    
    # Downtrend filter (vectorized across all individuals)
    downtrend_all = ema_f_all < ema_s_all
    
    # Close location filter (broadcast cloc to all individuals)
    cloc_broadcast = cloc.unsqueeze(0).expand(n_individuals, -1)
    
    # Signal detection (vectorized!)
    exhaustion_all = torch.zeros((n_individuals, n_bars), device=device, dtype=torch.bool)
    
    for i, rp in enumerate(resolved_params):
        exhaustion_all[i] = (
            downtrend_all[i] &
            (vol_z_all[i] > rp['vol_z']) &
            (tr_z_all[i] > rp['tr_z']) &
            (cloc_broadcast[i] > rp['cloc_min'])
        )
    
    # ========================================================================
    # Step 6: Convert back to pandas DataFrames with EXACT CPU structure
    # ========================================================================
    
    if verbose:
        print(f"   [6/6] Converting results to pandas DataFrames...")
    
    results = []
    
    for i in range(n_individuals):
        # Start with copy of original data (same as CPU)
        out = data.copy()
        
        # Add features (convert tensors to numpy)
        out["ema_f"] = to_numpy(ema_f_all[i])
        out["ema_s"] = to_numpy(ema_s_all[i])
        out["downtrend"] = to_numpy(downtrend_all[i])
        out["atr"] = to_numpy(atr_all[i])
        out["vol_z"] = to_numpy(vol_z_all[i])
        out["tr_z"] = to_numpy(tr_z_all[i])
        out["cloc"] = to_numpy(cloc_broadcast[i])
        out["exhaustion"] = to_numpy(exhaustion_all[i])
        
        # CRITICAL: Apply same NaN handling as CPU!
        # CPU does: (implicitly handled by pandas rolling, first window values are NaN)
        # We need to ensure NaN propagation matches
        
        # The indicators already have NaN in first window positions from GPU calculations
        # But we need to explicitly dropna like CPU does in backtest engine
        # Actually, CPU build_features() doesn't drop NaN - that's done in run_backtest()
        # So we match CPU exactly by NOT dropping here
        
        # Add Fibonacci levels if requested (CPU fallback for now)
        if add_fib:
            from indicators.fibonacci import add_fib_levels_to_df
            out = add_fib_levels_to_df(
                out,
                signal_col="exhaustion",
                lookback=fib_lookback,
                lookahead=fib_lookahead
            )
        
        results.append(out)
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    elapsed = time.time() - start_time
    
    stats = GPUFeatureBuildStats(
        n_individuals=n_individuals,
        n_bars=n_bars,
        unique_ema_fast=len(ema_fast_groups),
        unique_ema_slow=len(ema_slow_groups),
        unique_z_window=len(z_window_groups),
        unique_atr_window=len(atr_window_groups),
        total_time=elapsed
    )
    
    if verbose:
        print(f"\n   ✓ GPU Feature Building Complete: {elapsed:.3f}s")
        print(f"      {elapsed/n_individuals*1000:.1f}ms per individual")
        print(f"      Efficiency: {(n_individuals / (len(ema_fast_groups) + len(ema_slow_groups) + len(atr_window_groups))):.1f}x reuse")
    
    return results, stats


def validate_gpu_vs_cpu(
    data: pd.DataFrame,
    seller_params: SellerParams,
    tf: Timeframe = Timeframe.m15,
    tolerance: float = 1e-5
) -> Dict[str, any]:
    """
    Validate that GPU feature building matches CPU exactly.
    
    Args:
        data: OHLCV DataFrame
        seller_params: Strategy parameters
        tf: Timeframe
        tolerance: Tolerance for floating point comparison
    
    Returns:
        Validation results dict with pass/fail and detailed diffs
    """
    from strategy.seller_exhaustion import build_features
    
    # CPU
    cpu_feats = build_features(data.copy(), seller_params, tf, add_fib=False)
    
    # GPU
    gpu_feats_list, stats = build_features_gpu_batch(
        data, [seller_params], tf, add_fib=False, verbose=False
    )
    gpu_feats = gpu_feats_list[0]
    
    # Compare
    issues = []
    
    # Check columns exist
    expected_cols = ['ema_f', 'ema_s', 'downtrend', 'atr', 'vol_z', 'tr_z', 'cloc', 'exhaustion']
    for col in expected_cols:
        if col not in gpu_feats.columns:
            issues.append(f"Missing column in GPU: {col}")
    
    if issues:
        return {'valid': False, 'issues': issues}
    
    # Compare values
    for col in ['ema_f', 'ema_s', 'atr', 'vol_z', 'tr_z', 'cloc']:
        cpu_vals = cpu_feats[col].values
        gpu_vals = gpu_feats[col].values
        
        # Ignore NaN positions
        valid_mask = ~(np.isnan(cpu_vals) | np.isnan(gpu_vals))
        
        if valid_mask.sum() == 0:
            continue
        
        diff = np.abs(cpu_vals[valid_mask] - gpu_vals[valid_mask])
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        if max_diff > tolerance:
            issues.append({
                'column': col,
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'tolerance': tolerance
            })
    
    # Compare signals (must match exactly!)
    cpu_signals = cpu_feats[cpu_feats['exhaustion'] == True]
    gpu_signals = gpu_feats[gpu_feats['exhaustion'] == True]
    
    signal_count_match = len(cpu_signals) == len(gpu_signals)
    
    if not signal_count_match:
        issues.append({
            'type': 'signal_count',
            'cpu_count': len(cpu_signals),
            'gpu_count': len(gpu_signals),
            'message': 'Signal counts do not match!'
        })
    else:
        # Check timestamps match
        if not all(cpu_signals.index == gpu_signals.index):
            issues.append({
                'type': 'signal_timestamps',
                'message': 'Signal timestamps do not match!'
            })
    
    return {
        'valid': len(issues) == 0,
        'signal_count_match': signal_count_match,
        'cpu_signals': len(cpu_signals),
        'gpu_signals': len(gpu_signals),
        'issues': issues,
        'stats': stats
    }
