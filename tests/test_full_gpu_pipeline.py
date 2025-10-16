"""
Test full GPU pipeline matches CPU exactly.

CRITICAL: 100% trade count match is required.
Phase 3 validation - ensures GPU pipeline produces identical results to CPU.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from backtest.engine_gpu_full import batch_backtest_full_gpu
from backtest.engine import run_backtest
from strategy.seller_exhaustion import build_features, SellerParams
from core.models import BacktestParams, Timeframe


@pytest.fixture(scope="module")
def real_data():
    """Load real market data for testing."""
    from data.provider import DataProvider
    
    async def fetch():
        dp = DataProvider()
        try:
            # Use Oct-Dec 2024 data (known to have signals)
            data = await dp.fetch_15m("X:ADAUSD", "2024-10-01", "2024-12-31")
            print(f"\n📊 Loaded {len(data)} bars from 2024-10-01 to 2024-12-31")
            return data
        finally:
            await dp.close()
    
    return asyncio.run(fetch())


def test_full_pipeline_single_individual(real_data):
    """Test full GPU pipeline matches CPU for single individual."""
    print("\n" + "="*80)
    print("TEST: Full Pipeline - Single Individual")
    print("="*80)
    
    sp = SellerParams()
    # Use traditional exits (GPU doesn't support Fibonacci yet)
    bp = BacktestParams(
        use_fib_exits=False,
        use_stop_loss=True,
        use_traditional_tp=True,
        use_time_exit=True
    )
    tf = Timeframe.m15
    
    # CPU
    print("\n[CPU] Running backtest...")
    cpu_feats = build_features(real_data.copy(), sp, tf, add_fib=False)
    cpu_result = run_backtest(cpu_feats, bp)
    
    # GPU
    print("\n[GPU] Running backtest...")
    gpu_results, stats = batch_backtest_full_gpu(
        real_data, [sp], [bp], tf, verbose=True
    )
    gpu_result = gpu_results[0]
    
    # CRITICAL: Trade counts must match
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"CPU Trades: {cpu_result['metrics']['n']}")
    print(f"GPU Trades: {gpu_result['metrics']['n']}")
    
    assert cpu_result['metrics']['n'] == gpu_result['metrics']['n'], \
        f"❌ Trade count mismatch: CPU={cpu_result['metrics']['n']}, GPU={gpu_result['metrics']['n']}"
    
    print(f"✅ Trade counts match: {cpu_result['metrics']['n']}")
    
    # Compare trades if we have any
    if cpu_result['metrics']['n'] > 0:
        cpu_trades = cpu_result['trades']
        gpu_trades = gpu_result['trades']
        
        print(f"\nComparing {len(cpu_trades)} trades...")
        
        # Check exit reasons match
        mismatches = []
        for i, (cpu_reason, gpu_reason) in enumerate(zip(cpu_trades['reason'], gpu_trades['reason'])):
            if cpu_reason != gpu_reason:
                mismatches.append({
                    'trade': i,
                    'cpu_reason': cpu_reason,
                    'gpu_reason': gpu_reason
                })
        
        if mismatches:
            print(f"\n❌ Exit reason mismatches:")
            for m in mismatches:
                print(f"  Trade {m['trade']}: CPU={m['cpu_reason']}, GPU={m['gpu_reason']}")
            assert False, f"{len(mismatches)} trades have mismatched exit reasons"
        
        print(f"✅ All exit reasons match")
        
        # Check PnL matches (within tolerance)
        pnl_diff = abs(cpu_result['metrics']['total_pnl'] - gpu_result['metrics']['total_pnl'])
        print(f"\nPnL Comparison:")
        print(f"  CPU Total PnL: {cpu_result['metrics']['total_pnl']:.6f}")
        print(f"  GPU Total PnL: {gpu_result['metrics']['total_pnl']:.6f}")
        print(f"  Difference: {pnl_diff:.6e}")
        
        assert pnl_diff < 1e-4, f"❌ PnL diff {pnl_diff} exceeds tolerance (1e-4)"
        print(f"✅ PnL matches within tolerance")
        
        # Check metrics
        print(f"\nMetrics Comparison:")
        for key in ['win_rate', 'avg_R', 'max_dd', 'sharpe']:
            cpu_val = cpu_result['metrics'].get(key, 0)
            gpu_val = gpu_result['metrics'].get(key, 0)
            diff = abs(cpu_val - gpu_val)
            print(f"  {key}: CPU={cpu_val:.4f}, GPU={gpu_val:.4f}, diff={diff:.6e}")
            
            # Allow slightly larger tolerance for derived metrics
            assert diff < 1e-3, f"❌ {key} diff {diff} exceeds tolerance"
        
        print(f"✅ All metrics match")
    
    print(f"\n{'='*80}")
    print("✅ Full pipeline matches CPU exactly!")
    print(f"{'='*80}")


def test_full_pipeline_batch(real_data):
    """Test full GPU pipeline matches CPU for multiple individuals."""
    print("\n" + "="*80)
    print("TEST: Full Pipeline - Batch (24 individuals)")
    print("="*80)
    
    # Create 24 varied parameter sets
    print("\n[Setup] Creating 24 parameter combinations...")
    params_list = []
    
    # Vary key parameters systematically
    for i in range(24):
        sp = SellerParams(
            ema_fast=90 + i*2,          # 90 to 136
            vol_z=1.8 + i*0.05,         # 1.8 to 2.95
            tr_z=1.0 + i*0.03,          # 1.0 to 1.69
        )
        bp = BacktestParams(
            use_fib_exits=False,         # GPU doesn't support Fib yet
            use_stop_loss=True,
            use_traditional_tp=True,
            use_time_exit=True,
            atr_stop_mult=0.6 + i*0.02,  # 0.6 to 1.06
        )
        params_list.append((sp, bp))
    
    tf = Timeframe.m15
    
    # CPU (sequential)
    print("\n[CPU] Running 24 backtests sequentially...")
    import time
    cpu_start = time.time()
    
    cpu_results = []
    for idx, (sp, bp) in enumerate(params_list):
        if idx % 6 == 0:
            print(f"  Progress: {idx}/24...")
        feats = build_features(real_data.copy(), sp, tf, add_fib=False)
        result = run_backtest(feats, bp)
        cpu_results.append(result)
    
    cpu_time = time.time() - cpu_start
    print(f"  ✓ CPU complete in {cpu_time:.2f}s ({cpu_time/24*1000:.0f}ms per individual)")
    
    # GPU (batch)
    print("\n[GPU] Running 24 backtests in batch...")
    gpu_start = time.time()
    
    gpu_results, stats = batch_backtest_full_gpu(
        real_data,
        [sp for sp, _ in params_list],
        [bp for _, bp in params_list],
        tf,
        verbose=True
    )
    
    gpu_time = time.time() - gpu_start
    print(f"  ✓ GPU complete in {gpu_time:.2f}s ({gpu_time/24*1000:.0f}ms per individual)")
    
    # Speedup
    speedup = cpu_time / gpu_time
    print(f"\n⚡ Speedup: {speedup:.1f}x")
    
    # Compare ALL individuals
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    mismatches = []
    exact_matches = 0
    
    for i, (cpu_res, gpu_res) in enumerate(zip(cpu_results, gpu_results)):
        cpu_n = cpu_res['metrics']['n']
        gpu_n = gpu_res['metrics']['n']
        
        if cpu_n == gpu_n:
            exact_matches += 1
        else:
            mismatches.append({
                'idx': i,
                'cpu_trades': cpu_n,
                'gpu_trades': gpu_n,
                'sp': params_list[i][0],
                'bp': params_list[i][1]
            })
    
    print(f"\nTrade Count Matches: {exact_matches}/24 ({exact_matches/24*100:.1f}%)")
    
    if mismatches:
        print(f"\n⚠️ Mismatches found:")
        for m in mismatches[:10]:  # Show first 10
            print(f"  Individual {m['idx']}: CPU={m['cpu_trades']}, GPU={m['gpu_trades']}")
            print(f"    Params: ema_fast={m['sp'].ema_fast}, vol_z={m['sp'].vol_z:.2f}, "
                  f"tr_z={m['sp'].tr_z:.2f}, stop_mult={m['bp'].atr_stop_mult:.2f}")
        
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches)-10} more")
        
        assert False, f"❌ {len(mismatches)}/24 individuals have trade count mismatches"
    
    print(f"✅ All 24 individuals match!")
    
    # Compare metrics summary
    print(f"\nMetrics Summary:")
    total_cpu_trades = sum(r['metrics']['n'] for r in cpu_results)
    total_gpu_trades = sum(r['metrics']['n'] for r in gpu_results)
    print(f"  Total Trades: CPU={total_cpu_trades}, GPU={total_gpu_trades}")
    
    avg_cpu_pnl = np.mean([r['metrics']['total_pnl'] for r in cpu_results])
    avg_gpu_pnl = np.mean([r['metrics']['total_pnl'] for r in gpu_results])
    print(f"  Avg Total PnL: CPU={avg_cpu_pnl:.6f}, GPU={avg_gpu_pnl:.6f}")
    
    print(f"\n{'='*80}")
    print("✅ Batch pipeline validation complete!")
    print(f"{'='*80}")
    print(f"\nPerformance Summary:")
    print(f"  CPU Time: {cpu_time:.2f}s")
    print(f"  GPU Time: {gpu_time:.2f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  GPU Stats: {stats}")


def test_full_pipeline_edge_cases(real_data):
    """Test edge cases: no signals, single signal, max signals."""
    print("\n" + "="*80)
    print("TEST: Full Pipeline - Edge Cases")
    print("="*80)
    
    tf = Timeframe.m15
    
    # Use traditional exits (GPU doesn't support Fib yet)
    bp = BacktestParams(
        use_fib_exits=False,
        use_stop_loss=True,
        use_traditional_tp=True,
        use_time_exit=True
    )
    
    # Case 1: Parameters that produce no signals
    print("\n[Case 1] No signals (very restrictive params)...")
    sp_no_signals = SellerParams(vol_z=10.0, tr_z=10.0)  # Impossible thresholds
    
    gpu_results, _ = batch_backtest_full_gpu(
        real_data, [sp_no_signals], [bp], tf, verbose=False
    )
    
    assert gpu_results[0]['metrics']['n'] == 0, "Should have 0 trades"
    print(f"  ✓ No signals case: {gpu_results[0]['metrics']['n']} trades")
    
    # Case 2: Parameters that produce many signals
    print("\n[Case 2] Many signals (very permissive params)...")
    sp_many_signals = SellerParams(vol_z=0.5, tr_z=0.5, cloc_min=0.3)
    
    cpu_feats = build_features(real_data.copy(), sp_many_signals, tf, add_fib=False)
    cpu_result = run_backtest(cpu_feats, bp)
    
    gpu_results, _ = batch_backtest_full_gpu(
        real_data, [sp_many_signals], [bp], tf, verbose=False
    )
    
    print(f"  CPU Trades: {cpu_result['metrics']['n']}")
    print(f"  GPU Trades: {gpu_results[0]['metrics']['n']}")
    
    assert cpu_result['metrics']['n'] == gpu_results[0]['metrics']['n'], \
        "Many signals case should match"
    print(f"  ✓ Many signals case matches: {gpu_results[0]['metrics']['n']} trades")
    
    # Case 3: Very short max_hold
    print("\n[Case 3] Very short max_hold (1 bar)...")
    sp = SellerParams()
    bp_short_hold = BacktestParams(max_hold=1)
    
    cpu_feats = build_features(real_data.copy(), sp, tf, add_fib=False)
    cpu_result = run_backtest(cpu_feats, bp_short_hold)
    
    gpu_results, _ = batch_backtest_full_gpu(
        real_data, [sp], [bp_short_hold], tf, verbose=False
    )
    
    print(f"  CPU Trades: {cpu_result['metrics']['n']}")
    print(f"  GPU Trades: {gpu_results[0]['metrics']['n']}")
    
    assert cpu_result['metrics']['n'] == gpu_results[0]['metrics']['n'], \
        "Short max_hold case should match"
    print(f"  ✓ Short max_hold case matches: {gpu_results[0]['metrics']['n']} trades")
    
    print(f"\n{'='*80}")
    print("✅ All edge cases passed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
