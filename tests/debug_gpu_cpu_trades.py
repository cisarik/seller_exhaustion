"""
Debug script to compare CPU vs GPU trades in detail.

This will show us exactly where they diverge.
"""

import asyncio
import torch
import pandas as pd
from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from core.models import BacktestParams, Timeframe
from backtest.engine import run_backtest
from backtest.engine_gpu_batch import BatchGPUBacktestEngine


async def main():
    # Fetch data
    dp = DataProvider()
    try:
        print("Fetching data...")
        data = await dp.fetch_15m("X:ADAUSD", "2024-10-01", "2024-12-31")
        print(f"Loaded {len(data)} bars\n")
    finally:
        await dp.close()
    
    # Test parameters
    sp = SellerParams(
        ema_fast=96,
        ema_slow=672,
        z_window=672,
        vol_z=2.0,
        tr_z=1.2,
        cloc_min=0.6,
        atr_window=96
    )
    
    bp = BacktestParams(
        use_fib_exits=False,  # Disabled for this test
        use_stop_loss=True,
        use_traditional_tp=True,
        atr_stop_mult=0.7,
        reward_r=2.0,
        max_hold=96,
        fee_bp=5.0,
        slippage_bp=5.0
    )
    
    tf = Timeframe.m15
    
    # CPU backtest
    print("Running CPU backtest...")
    feats_cpu = build_features(data.copy(), sp, tf)
    result_cpu = run_backtest(feats_cpu, bp)
    trades_cpu = result_cpu['trades']
    print(f"CPU: {len(trades_cpu)} trades\n")
    
    # GPU backtest
    print("Running GPU backtest...")
    gpu_engine = BatchGPUBacktestEngine(data)
    results_gpu = gpu_engine.batch_backtest([sp], [bp], tf)
    trades_gpu = results_gpu[0]['trades']
    print(f"GPU: {len(trades_gpu)} trades\n")
    
    # Compare trades
    print("="*80)
    print("TRADE-BY-TRADE COMPARISON")
    print("="*80)
    
    print(f"\nCPU Trades ({len(trades_cpu)}):")
    if len(trades_cpu) > 0:
        for i, trade in trades_cpu.iterrows():
            print(f"  [{i:2d}] Entry: {trade['entry_ts']} @ {trade['entry']:.5f} | "
                  f"Exit: {trade['exit_ts']} @ {trade['exit']:.5f} | "
                  f"Reason: {trade['reason']:10s} | PnL: {trade['pnl']:7.5f}")
    
    print(f"\nGPU Trades ({len(trades_gpu)}):")
    if len(trades_gpu) > 0:
        for i, trade in trades_gpu.iterrows():
            print(f"  [{i:2d}] Entry: {trade['entry_ts']} @ {trade['entry']:.5f} | "
                  f"Exit: {trade['exit_ts']} @ {trade['exit']:.5f} | "
                  f"Reason: {trade['reason']:10s} | PnL: {trade['pnl']:7.5f}")
    
    # Find differences
    print("\n" + "="*80)
    print("DIFFERENCE ANALYSIS")
    print("="*80)
    
    # Check if first N trades match
    min_trades = min(len(trades_cpu), len(trades_gpu))
    matches = 0
    
    for i in range(min_trades):
        cpu_t = trades_cpu.iloc[i]
        gpu_t = trades_gpu.iloc[i]
        
        # Compare entry timestamps (should be exact)
        entry_match = str(cpu_t['entry_ts']) == str(gpu_t['entry_ts'])
        exit_match = str(cpu_t['exit_ts']) == str(gpu_t['exit_ts'])
        reason_match = cpu_t['reason'] == gpu_t['reason']
        
        if entry_match and exit_match and reason_match:
            matches += 1
        else:
            print(f"\n❌ Trade {i} MISMATCH:")
            print(f"  CPU: Entry={cpu_t['entry_ts']}, Exit={cpu_t['exit_ts']}, Reason={cpu_t['reason']}")
            print(f"  GPU: Entry={gpu_t['entry_ts']}, Exit={gpu_t['exit_ts']}, Reason={gpu_t['reason']}")
    
    print(f"\nFirst {min_trades} trades: {matches} matches, {min_trades - matches} mismatches")
    
    # Extra trades
    if len(trades_cpu) != len(trades_gpu):
        extra_cpu = len(trades_cpu) - min_trades
        extra_gpu = len(trades_gpu) - min_trades
        
        if extra_cpu > 0:
            print(f"\n⚠ CPU has {extra_cpu} extra trades:")
            for i in range(min_trades, len(trades_cpu)):
                trade = trades_cpu.iloc[i]
                print(f"  [{i}] {trade['entry_ts']} - {trade['exit_ts']} ({trade['reason']})")
        
        if extra_gpu > 0:
            print(f"\n⚠ GPU has {extra_gpu} extra trades:")
            for i in range(min_trades, len(trades_gpu)):
                trade = trades_gpu.iloc[i]
                print(f"  [{i}] {trade['entry_ts']} - {trade['exit_ts']} ({trade['reason']})")
    
    # Check signal detection
    print("\n" + "="*80)
    print("SIGNAL DETECTION COMPARISON")
    print("="*80)
    
    cpu_signals = feats_cpu[feats_cpu['exhaustion'] == True]
    print(f"\nCPU found {len(cpu_signals)} exhaustion signals")
    print(f"First 10 signal times:")
    for i, (ts, row) in enumerate(cpu_signals.head(10).iterrows()):
        print(f"  [{i}] {ts}")
    
    print(f"\nCPU took {len(trades_cpu)} trades from {len(cpu_signals)} signals")
    print(f"GPU took {len(trades_gpu)} trades from {len(cpu_signals)} signals (assuming same)")


if __name__ == '__main__':
    asyncio.run(main())
