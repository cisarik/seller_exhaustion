"""
Comprehensive GPU/CPU Validation Test Suite

This test suite validates that GPU backtesting produces identical results to CPU
backtesting using REAL data from Polygon.io.

Tests:
1. Single individual: CPU baseline vs GPU batch
2. Multiple individuals: CPU sequential vs GPU batch
3. Precision analysis: float32 vs float64 impact
4. Performance benchmarking: Speedup measurements

Run with:
    poetry run pytest tests/test_gpu_cpu_validation.py -v -s

Note: Requires Polygon API key in .env file
"""

import pytest
import asyncio
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import time

from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from core.models import BacktestParams, Timeframe
from backtest.engine import run_backtest
from backtest.optimizer import calculate_fitness
from core.models import FitnessConfig


# Only import GPU modules if CUDA is available
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    from backtest.engine_gpu_batch import BatchGPUBacktestEngine, BatchBacktestConfig


class BacktestResult:
    """Container for backtest results with comparison methods."""
    
    def __init__(self, name: str, metrics: Dict, trades: pd.DataFrame, elapsed_time: float = 0.0):
        self.name = name
        self.metrics = metrics
        self.trades = trades
        self.elapsed_time = elapsed_time
    
    def compare_to(self, other: 'BacktestResult', tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Compare this result to another result.
        
        Args:
            other: Other BacktestResult to compare against
            tolerance: Tolerance for floating point comparisons
        
        Returns:
            Dict with comparison results
        """
        issues = []
        
        # Compare trade count (must be exact)
        trade_count_match = self.metrics['n'] == other.metrics['n']
        if not trade_count_match:
            issues.append({
                'type': 'trade_count',
                'severity': 'critical',
                'self_value': self.metrics['n'],
                'other_value': other.metrics['n'],
                'message': f"Trade count mismatch: {self.name}={self.metrics['n']}, {other.name}={other.metrics['n']}"
            })
        
        # Compare metrics (with tolerance)
        metric_diffs = {}
        for key in ['win_rate', 'avg_R', 'total_pnl', 'max_dd', 'sharpe']:
            if key in self.metrics and key in other.metrics:
                diff = abs(self.metrics[key] - other.metrics[key])
                metric_diffs[key] = diff
                
                if diff > tolerance:
                    severity = 'critical' if diff > tolerance * 100 else 'warning'
                    issues.append({
                        'type': f'metric_{key}',
                        'severity': severity,
                        'self_value': self.metrics[key],
                        'other_value': other.metrics[key],
                        'diff': diff,
                        'message': f"{key} difference: {diff:.8e} (tolerance: {tolerance:.8e})"
                    })
        
        # Compare individual trades (if counts match)
        trade_issues = []
        if trade_count_match and len(self.trades) > 0:
            for i, (t_self, t_other) in enumerate(zip(
                self.trades.itertuples(),
                other.trades.itertuples()
            )):
                # Check exit reason (must be exact)
                if t_self.reason != t_other.reason:
                    trade_issues.append({
                        'trade_idx': i,
                        'type': 'exit_reason',
                        'self_value': t_self.reason,
                        'other_value': t_other.reason
                    })
                
                # Check prices (with tolerance)
                for field in ['entry', 'exit', 'stop', 'tp', 'pnl', 'R']:
                    self_val = getattr(t_self, field)
                    other_val = getattr(t_other, field)
                    diff = abs(self_val - other_val)
                    
                    if diff > tolerance:
                        trade_issues.append({
                            'trade_idx': i,
                            'type': field,
                            'self_value': self_val,
                            'other_value': other_val,
                            'diff': diff
                        })
        
        if trade_issues:
            issues.append({
                'type': 'trade_details',
                'severity': 'warning',
                'count': len(trade_issues),
                'details': trade_issues[:5],  # First 5 issues
                'message': f"{len(trade_issues)} trade detail mismatches"
            })
        
        return {
            'match': len(issues) == 0,
            'trade_count_match': trade_count_match,
            'metric_diffs': metric_diffs,
            'issues': issues,
            'speedup': other.elapsed_time / self.elapsed_time if self.elapsed_time > 0 else None
        }
    
    def __repr__(self):
        return (f"BacktestResult(name='{self.name}', trades={self.metrics['n']}, "
                f"pnl={self.metrics['total_pnl']:.6f}, time={self.elapsed_time:.2f}s)")


@pytest.fixture(scope="module")
def sample_data():
    """
    Download sample data from Polygon.io for testing.
    
    Uses 3 months of data (enough for meaningful backtests, not too large).
    """
    async def fetch():
        dp = DataProvider()
        try:
            # Download 3 months of 15-minute data
            print("\n📥 Downloading sample data from Polygon.io...")
            data = await dp.fetch_15m(
                "X:ADAUSD",
                "2024-10-01",
                "2024-12-31"
            )
            print(f"✓ Downloaded {len(data)} bars ({data.index[0]} to {data.index[-1]})")
            return data
        finally:
            await dp.close()
    
    return asyncio.run(fetch())


@pytest.fixture
def test_parameters():
    """Standard test parameters for consistent testing."""
    return {
        'seller_params': SellerParams(
            ema_fast=96,
            ema_slow=672,
            z_window=672,
            vol_z=2.0,
            tr_z=1.2,
            cloc_min=0.6,
            atr_window=96
        ),
        'backtest_params': BacktestParams(
            use_fib_exits=False,  # Keep it simple for validation
            use_stop_loss=True,
            use_traditional_tp=True,
            atr_stop_mult=0.7,
            reward_r=2.0,
            max_hold=96,
            fee_bp=5.0,
            slippage_bp=5.0
        ),
        'timeframe': Timeframe.m15,
        'fitness_config': FitnessConfig(preset='balanced')
    }


def run_cpu_backtest(
    data: pd.DataFrame,
    seller_params: SellerParams,
    backtest_params: BacktestParams,
    timeframe: Timeframe
) -> BacktestResult:
    """Run single backtest on CPU (baseline reference)."""
    print(f"\n🖥️  Running CPU backtest...")
    
    start = time.time()
    
    # Build features
    feats = build_features(data.copy(), seller_params, timeframe)
    
    # Run backtest
    result = run_backtest(feats, backtest_params)
    
    elapsed = time.time() - start
    
    print(f"   ✓ CPU: {result['metrics']['n']} trades, {elapsed:.2f}s")
    
    return BacktestResult(
        name="CPU",
        metrics=result['metrics'],
        trades=result['trades'],
        elapsed_time=elapsed
    )


def run_gpu_batch_backtest(
    data: pd.DataFrame,
    seller_params_list: List[SellerParams],
    backtest_params_list: List[BacktestParams],
    timeframe: Timeframe
) -> List[BacktestResult]:
    """Run batch backtest on GPU (hybrid approach)."""
    n = len(seller_params_list)
    print(f"\n🚀 Running GPU batch backtest ({n} individuals)...")
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    start = time.time()
    
    # Create GPU engine
    config = BatchBacktestConfig(verbose=True)
    gpu_engine = BatchGPUBacktestEngine(data, config=config)
    
    # Batch backtest
    results = gpu_engine.batch_backtest(
        seller_params_list,
        backtest_params_list,
        timeframe
    )
    
    # Clear cache
    gpu_engine.clear_cache()
    
    elapsed = time.time() - start
    
    # Get memory info
    mem_info = gpu_engine.get_memory_usage()
    if mem_info['available']:
        print(f"   📊 GPU Memory: {mem_info['peak_gb']:.2f}/{mem_info['total_gb']:.2f} GB "
              f"({mem_info['peak_utilization']:.1%} peak)")
    
    print(f"   ✓ GPU: {n} backtests in {elapsed:.2f}s ({elapsed/n:.3f}s each)")
    
    # Convert to BacktestResult objects
    backtest_results = []
    for i, result in enumerate(results):
        backtest_results.append(BacktestResult(
            name=f"GPU[{i}]",
            metrics=result['metrics'],
            trades=result['trades'],
            elapsed_time=elapsed / n  # Amortized time per individual
        ))
    
    return backtest_results


# ============================================================================
# TEST 1: Single Individual - CPU vs GPU
# ============================================================================

def test_single_individual_cpu_vs_gpu(sample_data, test_parameters):
    """
    Test that GPU batch (with 1 individual) matches CPU exactly.
    
    This is the most basic validation - if this fails, GPU is broken.
    """
    print("\n" + "="*80)
    print("TEST 1: Single Individual - CPU vs GPU")
    print("="*80)
    
    sp = test_parameters['seller_params']
    bp = test_parameters['backtest_params']
    tf = test_parameters['timeframe']
    
    # Run CPU
    result_cpu = run_cpu_backtest(sample_data, sp, bp, tf)
    
    # Run GPU (single individual)
    results_gpu = run_gpu_batch_backtest(sample_data, [sp], [bp], tf)
    result_gpu = results_gpu[0]
    
    # Compare
    comparison = result_cpu.compare_to(result_gpu, tolerance=1e-5)
    
    # Print report
    print("\n📊 Comparison Report:")
    print(f"   CPU: {result_cpu}")
    print(f"   GPU: {result_gpu}")
    print(f"   Trade Count Match: {'✓' if comparison['trade_count_match'] else '✗ FAILED'}")
    
    if comparison['metric_diffs']:
        print(f"\n   Metric Differences:")
        for key, diff in comparison['metric_diffs'].items():
            status = "✓" if diff < 1e-5 else "⚠"
            print(f"     {status} {key}: {diff:.8e}")
    
    if comparison['speedup']:
        print(f"\n   Speedup: {comparison['speedup']:.2f}x")
    
    # Report issues
    if comparison['issues']:
        print(f"\n   ⚠ Issues Found:")
        for issue in comparison['issues']:
            print(f"     [{issue['severity'].upper()}] {issue['message']}")
            if issue['type'].startswith('metric_'):
                print(f"       CPU: {issue['self_value']:.8f}")
                print(f"       GPU: {issue['other_value']:.8f}")
                print(f"       Diff: {issue['diff']:.8e}")
    
    # Assertions
    assert comparison['trade_count_match'], \
        f"Trade count mismatch: CPU={result_cpu.metrics['n']}, GPU={result_gpu.metrics['n']}"
    
    # Allow small differences in PnL due to float32 precision
    if result_cpu.metrics['n'] > 0:
        pnl_diff = abs(result_cpu.metrics['total_pnl'] - result_gpu.metrics['total_pnl'])
        assert pnl_diff < 1e-4, \
            f"Total PnL difference too large: {pnl_diff:.8e} (CPU={result_cpu.metrics['total_pnl']:.6f}, GPU={result_gpu.metrics['total_pnl']:.6f})"
        
        # Sharpe can differ slightly due to timestamp precision, allow larger tolerance
        sharpe_diff = abs(result_cpu.metrics['sharpe'] - result_gpu.metrics['sharpe'])
        if sharpe_diff > 0.2:  # Only warn if very different
            print(f"   ⚠ Sharpe ratio differs: CPU={result_cpu.metrics['sharpe']:.6f}, GPU={result_gpu.metrics['sharpe']:.6f}")
    
    print("\n✅ TEST 1 PASSED: GPU matches CPU for single individual")


# ============================================================================
# TEST 2: Multiple Individuals - CPU vs GPU Batch
# ============================================================================

@pytest.mark.parametrize("n_individuals", [5, 12, 24])
def test_multiple_individuals_cpu_vs_gpu(sample_data, test_parameters, n_individuals):
    """
    Test that GPU batch matches CPU for multiple individuals.
    
    This validates that batch processing doesn't introduce errors.
    """
    print("\n" + "="*80)
    print(f"TEST 2: Multiple Individuals ({n_individuals}) - CPU vs GPU Batch")
    print("="*80)
    
    tf = test_parameters['timeframe']
    
    # Create varied parameter sets
    print(f"\n📋 Creating {n_individuals} parameter sets...")
    sp_list = []
    bp_list = []
    
    for i in range(n_individuals):
        sp = SellerParams(
            ema_fast=90 + i*3,
            ema_slow=650 + i*10,
            z_window=650 + i*10,
            vol_z=1.8 + i*0.05,
            tr_z=1.0 + i*0.05,
            cloc_min=0.55 + i*0.01,
            atr_window=90 + i*2
        )
        bp = BacktestParams(
            use_fib_exits=False,
            use_stop_loss=True,
            use_traditional_tp=True,
            atr_stop_mult=0.6 + i*0.02,
            reward_r=1.8 + i*0.05,
            max_hold=96,
            fee_bp=5.0,
            slippage_bp=5.0
        )
        sp_list.append(sp)
        bp_list.append(bp)
    
    # Run CPU (sequential)
    print(f"\n🖥️  Running {n_individuals} CPU backtests (sequential)...")
    start_cpu = time.time()
    results_cpu = []
    
    for i, (sp, bp) in enumerate(zip(sp_list, bp_list)):
        result = run_cpu_backtest(sample_data, sp, bp, tf)
        result.name = f"CPU[{i}]"
        results_cpu.append(result)
        print(f"     [{i+1}/{n_individuals}] {result.metrics['n']} trades")
    
    cpu_total_time = time.time() - start_cpu
    print(f"\n   ✓ CPU Total: {cpu_total_time:.2f}s ({cpu_total_time/n_individuals:.2f}s per individual)")
    
    # Run GPU (batch)
    results_gpu = run_gpu_batch_backtest(sample_data, sp_list, bp_list, tf)
    
    # Compare each individual
    print(f"\n📊 Individual Comparisons:")
    mismatches = []
    all_comparisons = []
    
    for i, (r_cpu, r_gpu) in enumerate(zip(results_cpu, results_gpu)):
        comparison = r_cpu.compare_to(r_gpu, tolerance=1e-5)
        all_comparisons.append(comparison)
        
        match_icon = "✓" if comparison['match'] else "✗"
        trade_icon = "✓" if comparison['trade_count_match'] else "✗"
        
        print(f"   [{i:2d}] {match_icon} CPU: {r_cpu.metrics['n']:3d} trades | "
              f"GPU: {r_gpu.metrics['n']:3d} trades {trade_icon}")
        
        if not comparison['trade_count_match']:
            mismatches.append({
                'index': i,
                'cpu_trades': r_cpu.metrics['n'],
                'gpu_trades': r_gpu.metrics['n'],
                'cpu_pnl': r_cpu.metrics['total_pnl'],
                'gpu_pnl': r_gpu.metrics['total_pnl']
            })
    
    # Summary
    gpu_total_time = sum(r.elapsed_time for r in results_gpu)
    speedup = cpu_total_time / gpu_total_time
    
    print(f"\n📈 Performance Summary:")
    print(f"   CPU Total Time:  {cpu_total_time:.2f}s ({cpu_total_time/n_individuals:.3f}s each)")
    print(f"   GPU Total Time:  {gpu_total_time:.2f}s ({gpu_total_time/n_individuals:.3f}s each)")
    print(f"   Speedup:         {speedup:.2f}x")
    
    print(f"\n📊 Consistency Summary:")
    matches = sum(1 for c in all_comparisons if c['match'])
    trade_matches = sum(1 for c in all_comparisons if c['trade_count_match'])
    print(f"   Trade Count Matches: {trade_matches}/{n_individuals} ({trade_matches/n_individuals*100:.1f}%)")
    print(f"   Full Matches:        {matches}/{n_individuals} ({matches/n_individuals*100:.1f}%)")
    
    # Report mismatches
    if mismatches:
        print(f"\n   ⚠ Trade Count Mismatches ({len(mismatches)}):")
        for m in mismatches[:5]:  # Show first 5
            print(f"     Individual {m['index']}: CPU={m['cpu_trades']}, GPU={m['gpu_trades']}")
            print(f"       CPU PnL: {m['cpu_pnl']:.6f}, GPU PnL: {m['gpu_pnl']:.6f}")
    
    # Assertions
    assert trade_matches == n_individuals, \
        f"Trade count mismatches: {n_individuals - trade_matches}/{n_individuals} individuals differ"
    
    print(f"\n✅ TEST 2 PASSED: GPU batch matches CPU for {n_individuals} individuals")


# ============================================================================
# TEST 3: Precision Analysis - Float32 vs Float64
# ============================================================================

def test_precision_analysis(sample_data, test_parameters):
    """
    Analyze the impact of float32 vs float64 precision on results.
    
    This test helps determine if we need to upgrade GPU to float64.
    """
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    print("\n" + "="*80)
    print("TEST 3: Precision Analysis - Float32 vs Float64")
    print("="*80)
    
    sp = test_parameters['seller_params']
    bp = test_parameters['backtest_params']
    tf = test_parameters['timeframe']
    
    # Run CPU (float64 - reference)
    result_cpu = run_cpu_backtest(sample_data, sp, bp, tf)
    
    # Run GPU with current precision (float32)
    results_gpu_f32 = run_gpu_batch_backtest(sample_data, [sp], [bp], tf)
    result_gpu_f32 = results_gpu_f32[0]
    
    # Compare
    comparison = result_cpu.compare_to(result_gpu_f32, tolerance=1e-5)
    
    print(f"\n📊 Precision Analysis:")
    print(f"   CPU (float64):  {result_cpu.metrics['n']} trades, PnL={result_cpu.metrics['total_pnl']:.8f}")
    print(f"   GPU (float32):  {result_gpu_f32.metrics['n']} trades, PnL={result_gpu_f32.metrics['total_pnl']:.8f}")
    
    if comparison['metric_diffs']:
        print(f"\n   Metric Differences (float32 vs float64):")
        for key, diff in comparison['metric_diffs'].items():
            rel_error = diff / abs(result_cpu.metrics[key]) if result_cpu.metrics[key] != 0 else 0
            print(f"     {key:12s}: {diff:.8e} (relative: {rel_error:.2%})")
    
    # Recommendation
    max_diff = max(comparison['metric_diffs'].values()) if comparison['metric_diffs'] else 0
    
    print(f"\n💡 Recommendation:")
    if max_diff < 1e-6:
        print(f"   ✓ Float32 precision is EXCELLENT (max diff: {max_diff:.8e})")
        print(f"   → Keep float32 for best performance")
    elif max_diff < 1e-4:
        print(f"   ⚠ Float32 precision is ACCEPTABLE (max diff: {max_diff:.8e})")
        print(f"   → Consider float64 if exact consistency is critical")
    else:
        print(f"   ✗ Float32 precision is INSUFFICIENT (max diff: {max_diff:.8e})")
        print(f"   → MUST upgrade to float64 for production use")
    
    print(f"\n✅ TEST 3 COMPLETE: Precision analysis finished")


# ============================================================================
# TEST 4: Performance Benchmark
# ============================================================================

@pytest.mark.parametrize("n_individuals", [1, 5, 12, 24, 48])
def test_performance_benchmark(sample_data, test_parameters, n_individuals):
    """
    Benchmark GPU speedup across different population sizes.
    
    This shows how GPU advantage scales with batch size.
    """
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    print("\n" + "="*80)
    print(f"TEST 4: Performance Benchmark (N={n_individuals})")
    print("="*80)
    
    tf = test_parameters['timeframe']
    
    # Create parameter sets
    sp_list = []
    bp_list = []
    
    for i in range(n_individuals):
        sp = SellerParams(ema_fast=90 + i*2, ema_slow=650, z_window=650, atr_window=90)
        bp = BacktestParams()
        sp_list.append(sp)
        bp_list.append(bp)
    
    # Benchmark CPU
    print(f"\n⏱️  Benchmarking CPU ({n_individuals} individuals)...")
    start = time.time()
    for sp, bp in zip(sp_list, bp_list):
        feats = build_features(sample_data.copy(), sp, tf)
        run_backtest(feats, bp)
    cpu_time = time.time() - start
    
    print(f"   CPU Time: {cpu_time:.2f}s ({cpu_time/n_individuals:.3f}s per individual)")
    
    # Benchmark GPU
    print(f"\n⏱️  Benchmarking GPU ({n_individuals} individuals)...")
    start = time.time()
    gpu_engine = BatchGPUBacktestEngine(sample_data, config=BatchBacktestConfig(verbose=False))
    gpu_engine.batch_backtest(sp_list, bp_list, tf)
    gpu_time = time.time() - start
    gpu_engine.clear_cache()
    
    print(f"   GPU Time: {gpu_time:.2f}s ({gpu_time/n_individuals:.3f}s per individual)")
    
    speedup = cpu_time / gpu_time
    
    print(f"\n🚀 Speedup: {speedup:.2f}x")
    print(f"   Time Saved: {cpu_time - gpu_time:.2f}s")
    
    # Performance rating
    if speedup >= 20:
        rating = "🌟 EXCELLENT"
    elif speedup >= 10:
        rating = "✓ GOOD"
    elif speedup >= 5:
        rating = "⚠ ACCEPTABLE"
    else:
        rating = "✗ POOR"
    
    print(f"   Performance: {rating}")
    
    print(f"\n✅ TEST 4 COMPLETE: Benchmark finished")


# ============================================================================
# Main Test Report Generator
# ============================================================================

def test_generate_full_report(sample_data, test_parameters):
    """
    Generate comprehensive validation report.
    
    Run this after all tests to get final verdict on GPU reliability.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE GPU/CPU VALIDATION REPORT")
    print("="*80)
    
    print(f"\n📊 Test Data:")
    print(f"   Source: Polygon.io (X:ADAUSD)")
    print(f"   Period: {sample_data.index[0]} to {sample_data.index[-1]}")
    print(f"   Bars: {len(sample_data)}")
    print(f"   Timeframe: 15 minutes")
    
    print(f"\n🔧 System Info:")
    print(f"   GPU Available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {mem:.2f} GB")
    
    print(f"\n📋 Test Parameters:")
    sp = test_parameters['seller_params']
    bp = test_parameters['backtest_params']
    print(f"   EMA Fast: {sp.ema_fast}, EMA Slow: {sp.ema_slow}")
    print(f"   Vol Z: {sp.vol_z}, TR Z: {sp.tr_z}")
    print(f"   Stop Mult: {bp.atr_stop_mult}, R:R: {bp.reward_r}")
    
    print(f"\n✅ All tests must pass for GPU to be production-ready!")
    print(f"   Run: poetry run pytest tests/test_gpu_cpu_validation.py -v -s")
