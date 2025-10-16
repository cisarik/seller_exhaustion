"""
Phase 1 Validation: GPU Indicator Calculation vs CPU

Tests that GPU indicator calculations produce IDENTICAL results to CPU.

CRITICAL: These tests must ALL pass before proceeding to Phase 2.

Run with:
    poetry run pytest tests/test_gpu_indicator_validation.py -v -s
"""

import pytest
import asyncio
import torch
import numpy as np
import pandas as pd

from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from strategy.seller_exhaustion_gpu import (
    build_features_gpu_batch,
    validate_gpu_vs_cpu,
    GPUFeatureBuildStats
)
from core.models import Timeframe
from indicators.gpu import get_device


GPU_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope="module")
def sample_data():
    """Download real data for validation."""
    async def fetch():
        dp = DataProvider()
        try:
            print("\n📥 Downloading sample data from Polygon.io...")
            data = await dp.fetch_15m("X:ADAUSD", "2024-10-01", "2024-12-31")
            print(f"✓ Downloaded {len(data)} bars")
            return data
        finally:
            await dp.close()
    
    return asyncio.run(fetch())


def test_single_indicator_ema_fast(sample_data):
    """Test EMA Fast calculation matches CPU exactly."""
    print("\n" + "="*80)
    print("TEST: EMA Fast Indicator - GPU vs CPU")
    print("="*80)
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    params = SellerParams(ema_fast=96, ema_slow=672)
    
    # CPU
    cpu_feats = build_features(sample_data.copy(), params, Timeframe.m15, add_fib=False)
    
    # GPU
    gpu_feats_list, stats = build_features_gpu_batch(
        sample_data, [params], Timeframe.m15, add_fib=False, verbose=True
    )
    gpu_feats = gpu_feats_list[0]
    
    # Compare EMA Fast
    cpu_ema_f = cpu_feats['ema_f'].values
    gpu_ema_f = gpu_feats['ema_f'].values
    
    # Ignore NaN positions
    valid_mask = ~(np.isnan(cpu_ema_f) | np.isnan(gpu_ema_f))
    
    diff = np.abs(cpu_ema_f[valid_mask] - gpu_ema_f[valid_mask])
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\n📊 EMA Fast Comparison:")
    print(f"   Max Diff: {max_diff:.8e}")
    print(f"   Mean Diff: {mean_diff:.8e}")
    print(f"   Tolerance: 1e-5")
    
    assert max_diff < 1e-5, f"EMA Fast max diff {max_diff:.8e} exceeds tolerance"
    
    print("\n✅ EMA Fast matches CPU within tolerance")


def test_single_indicator_atr(sample_data):
    """Test ATR calculation matches CPU exactly."""
    print("\n" + "="*80)
    print("TEST: ATR Indicator - GPU vs CPU")
    print("="*80)
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    params = SellerParams(atr_window=96)
    
    # CPU
    cpu_feats = build_features(sample_data.copy(), params, Timeframe.m15, add_fib=False)
    
    # GPU
    gpu_feats_list, stats = build_features_gpu_batch(
        sample_data, [params], Timeframe.m15, add_fib=False, verbose=True
    )
    gpu_feats = gpu_feats_list[0]
    
    # Compare ATR
    cpu_atr = cpu_feats['atr'].values
    gpu_atr = gpu_feats['atr'].values
    
    valid_mask = ~(np.isnan(cpu_atr) | np.isnan(gpu_atr))
    
    diff = np.abs(cpu_atr[valid_mask] - gpu_atr[valid_mask])
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\n📊 ATR Comparison:")
    print(f"   Max Diff: {max_diff:.8e}")
    print(f"   Mean Diff: {mean_diff:.8e}")
    print(f"   Tolerance: 1e-5")
    
    assert max_diff < 1e-5, f"ATR max diff {max_diff:.8e} exceeds tolerance"
    
    print("\n✅ ATR matches CPU within tolerance")


def test_single_indicator_zscore(sample_data):
    """Test Z-Score calculation matches CPU exactly."""
    print("\n" + "="*80)
    print("TEST: Z-Score Indicators - GPU vs CPU")
    print("="*80)
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    params = SellerParams(z_window=672)
    
    # CPU
    cpu_feats = build_features(sample_data.copy(), params, Timeframe.m15, add_fib=False)
    
    # GPU
    gpu_feats_list, stats = build_features_gpu_batch(
        sample_data, [params], Timeframe.m15, add_fib=False, verbose=True
    )
    gpu_feats = gpu_feats_list[0]
    
    # Compare Vol Z-Score
    print(f"\n📊 Volume Z-Score Comparison:")
    cpu_vol_z = cpu_feats['vol_z'].values
    gpu_vol_z = gpu_feats['vol_z'].values
    
    valid_mask = ~(np.isnan(cpu_vol_z) | np.isnan(gpu_vol_z))
    
    diff = np.abs(cpu_vol_z[valid_mask] - gpu_vol_z[valid_mask])
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"   Max Diff: {max_diff:.8e}")
    print(f"   Mean Diff: {mean_diff:.8e}")
    
    assert max_diff < 1e-5, f"Vol Z max diff {max_diff:.8e} exceeds tolerance"
    
    # Compare TR Z-Score
    print(f"\n📊 True Range Z-Score Comparison:")
    cpu_tr_z = cpu_feats['tr_z'].values
    gpu_tr_z = gpu_feats['tr_z'].values
    
    valid_mask = ~(np.isnan(cpu_tr_z) | np.isnan(gpu_tr_z))
    
    diff = np.abs(cpu_tr_z[valid_mask] - gpu_tr_z[valid_mask])
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"   Max Diff: {max_diff:.8e}")
    print(f"   Mean Diff: {mean_diff:.8e}")
    
    assert max_diff < 1e-5, f"TR Z max diff {max_diff:.8e} exceeds tolerance"
    
    print("\n✅ Z-Score indicators match CPU within tolerance")


def test_signal_detection(sample_data):
    """
    CRITICAL TEST: Signal detection must match CPU exactly.
    
    This is where the 40 vs 14 trade bug originated!
    """
    print("\n" + "="*80)
    print("TEST: Signal Detection - GPU vs CPU (CRITICAL!)")
    print("="*80)
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    params = SellerParams()
    
    # CPU
    cpu_feats = build_features(sample_data.copy(), params, Timeframe.m15, add_fib=False)
    cpu_signals = cpu_feats[cpu_feats['exhaustion'] == True]
    
    # GPU
    gpu_feats_list, stats = build_features_gpu_batch(
        sample_data, [params], Timeframe.m15, add_fib=False, verbose=True
    )
    gpu_feats = gpu_feats_list[0]
    gpu_signals = gpu_feats[gpu_feats['exhaustion'] == True]
    
    print(f"\n📊 Signal Detection Comparison:")
    print(f"   CPU Signals: {len(cpu_signals)}")
    print(f"   GPU Signals: {len(gpu_signals)}")
    
    # CRITICAL: Signal counts MUST match exactly
    assert len(cpu_signals) == len(gpu_signals), \
        f"Signal count mismatch! CPU={len(cpu_signals)}, GPU={len(gpu_signals)}"
    
    # CRITICAL: Signal timestamps MUST match exactly
    if len(cpu_signals) > 0:
        cpu_ts = cpu_signals.index.tolist()
        gpu_ts = gpu_signals.index.tolist()
        
        matches = sum(1 for c, g in zip(cpu_ts, gpu_ts) if c == g)
        
        print(f"   Timestamp Matches: {matches}/{len(cpu_signals)}")
        
        if matches != len(cpu_signals):
            print("\n   ⚠ Timestamp Mismatches:")
            for i, (c, g) in enumerate(zip(cpu_ts, gpu_ts)):
                if c != g:
                    print(f"      [{i}] CPU: {c}, GPU: {g}")
                    if i >= 5:
                        print(f"      ... ({len(cpu_signals) - matches} more)")
                        break
        
        assert matches == len(cpu_signals), \
            f"Signal timestamps mismatch! {matches}/{len(cpu_signals)} match"
    
    print("\n✅ Signal detection matches CPU exactly!")


def test_batch_consistency(sample_data):
    """Test that batch processing maintains consistency."""
    print("\n" + "="*80)
    print("TEST: Batch Consistency - Multiple Individuals")
    print("="*80)
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    # Create varied parameter sets
    params_list = [
        SellerParams(ema_fast=90, vol_z=1.8),
        SellerParams(ema_fast=96, vol_z=2.0),
        SellerParams(ema_fast=100, vol_z=2.2),
        SellerParams(ema_fast=96, vol_z=2.0),  # Duplicate of #1 (should reuse)
    ]
    
    # GPU Batch
    gpu_feats_list, stats = build_features_gpu_batch(
        sample_data, params_list, Timeframe.m15, verbose=True
    )
    
    # CPU Sequential
    cpu_feats_list = []
    for params in params_list:
        cpu_feats = build_features(sample_data.copy(), params, Timeframe.m15, add_fib=False)
        cpu_feats_list.append(cpu_feats)
    
    print(f"\n📊 Batch Consistency Check:")
    
    # Compare each individual
    all_match = True
    for i, (cpu_feats, gpu_feats) in enumerate(zip(cpu_feats_list, gpu_feats_list)):
        cpu_signals = cpu_feats[cpu_feats['exhaustion'] == True]
        gpu_signals = gpu_feats[gpu_feats['exhaustion'] == True]
        
        match = len(cpu_signals) == len(gpu_signals)
        icon = "✓" if match else "✗"
        
        print(f"   [{i}] {icon} CPU: {len(cpu_signals)} signals, GPU: {len(gpu_signals)} signals")
        
        if not match:
            all_match = False
    
    assert all_match, "Some individuals have signal count mismatches!"
    
    # Check that duplicate parameters (#1 and #3) produce identical results
    print(f"\n📊 Duplicate Parameter Check (#1 vs #3):")
    
    dup1_signals = gpu_feats_list[1][gpu_feats_list[1]['exhaustion'] == True]
    dup2_signals = gpu_feats_list[3][gpu_feats_list[3]['exhaustion'] == True]
    
    print(f"   #1 Signals: {len(dup1_signals)}")
    print(f"   #3 Signals: {len(dup2_signals)}")
    
    assert len(dup1_signals) == len(dup2_signals), "Duplicate parameters produced different results!"
    
    print("\n✅ Batch processing is consistent")


def test_comprehensive_validation(sample_data):
    """Comprehensive validation using built-in validator."""
    print("\n" + "="*80)
    print("TEST: Comprehensive Validation")
    print("="*80)
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    params = SellerParams()
    
    result = validate_gpu_vs_cpu(sample_data, params, Timeframe.m15, tolerance=1e-5)
    
    print(f"\n📊 Validation Results:")
    print(f"   Valid: {result['valid']}")
    print(f"   Signal Count Match: {result['signal_count_match']}")
    print(f"   CPU Signals: {result['cpu_signals']}")
    print(f"   GPU Signals: {result['gpu_signals']}")
    
    if result['issues']:
        print(f"\n   ⚠ Issues Found ({len(result['issues'])}):")
        for issue in result['issues']:
            if isinstance(issue, dict):
                print(f"      {issue}")
            else:
                print(f"      {issue}")
    
    assert result['valid'], f"Validation failed with {len(result['issues'])} issues"
    assert result['signal_count_match'], \
        f"Signal count mismatch: CPU={result['cpu_signals']}, GPU={result['gpu_signals']}"
    
    print("\n✅ Comprehensive validation passed")


def test_performance_benchmark(sample_data):
    """Benchmark GPU speedup for indicator calculation."""
    print("\n" + "="*80)
    print("TEST: Performance Benchmark - Phase 1")
    print("="*80)
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    import time
    
    n_individuals = 24
    params_list = [
        SellerParams(ema_fast=90 + i*2, vol_z=1.8 + i*0.05)
        for i in range(n_individuals)
    ]
    
    # CPU Sequential
    print(f"\n⏱️  Benchmarking CPU (sequential, {n_individuals} individuals)...")
    start = time.time()
    for params in params_list:
        _ = build_features(sample_data.copy(), params, Timeframe.m15, add_fib=False)
    cpu_time = time.time() - start
    print(f"   CPU Time: {cpu_time:.2f}s ({cpu_time/n_individuals*1000:.1f}ms each)")
    
    # GPU Batch
    print(f"\n⏱️  Benchmarking GPU (batch, {n_individuals} individuals)...")
    start = time.time()
    gpu_feats_list, stats = build_features_gpu_batch(
        sample_data, params_list, Timeframe.m15, verbose=False
    )
    gpu_time = time.time() - start
    print(f"   GPU Time: {gpu_time:.2f}s ({gpu_time/n_individuals*1000:.1f}ms each)")
    
    speedup = cpu_time / gpu_time
    
    print(f"\n🚀 Phase 1 Performance:")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Time Saved: {cpu_time - gpu_time:.2f}s")
    
    # Check GPU stats
    print(f"\n📊 GPU Statistics:")
    print(f"   {stats}")
    
    # Performance expectations for Phase 1
    if speedup >= 10:
        rating = "🌟 EXCELLENT"
    elif speedup >= 5:
        rating = "✓ GOOD"
    elif speedup >= 2:
        rating = "⚠ ACCEPTABLE"
    else:
        rating = "✗ NEEDS IMPROVEMENT"
    
    print(f"   Performance: {rating}")
    
    # Phase 1 should achieve at least 2x speedup
    assert speedup >= 2.0, f"Phase 1 speedup {speedup:.2f}x is below minimum 2x target"
    
    print("\n✅ Performance benchmark complete")


def test_nan_handling(sample_data):
    """Test that NaN handling matches CPU exactly."""
    print("\n" + "="*80)
    print("TEST: NaN Handling - Critical for Warmup Period")
    print("="*80)
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    params = SellerParams()
    
    # CPU
    cpu_feats = build_features(sample_data.copy(), params, Timeframe.m15, add_fib=False)
    
    # GPU
    gpu_feats_list, stats = build_features_gpu_batch(
        sample_data, [params], Timeframe.m15, add_fib=False
    )
    gpu_feats = gpu_feats_list[0]
    
    print(f"\n📊 NaN Positions Check:")
    
    for col in ['ema_f', 'ema_s', 'atr', 'vol_z', 'tr_z']:
        cpu_nan_count = cpu_feats[col].isna().sum()
        gpu_nan_count = gpu_feats[col].isna().sum()
        
        match = cpu_nan_count == gpu_nan_count
        icon = "✓" if match else "✗"
        
        print(f"   {icon} {col:10s}: CPU NaN={cpu_nan_count}, GPU NaN={gpu_nan_count}")
        
        assert match, f"{col} NaN count mismatch: CPU={cpu_nan_count}, GPU={gpu_nan_count}"
        
        # Check that NaN positions are in the same locations
        cpu_nan_mask = cpu_feats[col].isna()
        gpu_nan_mask = gpu_feats[col].isna()
        
        nan_position_match = (cpu_nan_mask == gpu_nan_mask).all()
        
        if not nan_position_match:
            diff_count = (cpu_nan_mask != gpu_nan_mask).sum()
            print(f"      ⚠ NaN position mismatch in {diff_count} locations!")
            assert False, f"{col} NaN positions don't match"
    
    print("\n✅ NaN handling matches CPU exactly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
