"""Debug script to compare GPU vs CPU signal detection."""

import asyncio
import pandas as pd
import numpy as np

from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from strategy.seller_exhaustion_gpu import build_features_gpu_batch
from core.models import Timeframe


async def main():
    dp = DataProvider()
    try:
        data = await dp.fetch_15m("X:ADAUSD", "2024-10-01", "2024-12-31")
        print(f"Loaded {len(data)} bars\n")
    finally:
        await dp.close()
    
    params = SellerParams()
    
    # CPU
    print("Running CPU build_features...")
    cpu_feats = build_features(data.copy(), params, Timeframe.m15, add_fib=False)
    cpu_signals = cpu_feats[cpu_feats['exhaustion'] == True]
    
    # GPU
    print("Running GPU build_features_gpu_batch...")
    gpu_feats_list, stats = build_features_gpu_batch(
        data, [params], Timeframe.m15, add_fib=False, verbose=False
    )
    gpu_feats = gpu_feats_list[0]
    gpu_signals = gpu_feats[gpu_feats['exhaustion'] == True]
    
    print(f"\nCPU Signals: {len(cpu_signals)}")
    print(f"GPU Signals: {len(gpu_signals)}\n")
    
    # Find first signal that differs
    cpu_ts = set(cpu_signals.index)
    gpu_ts = set(gpu_signals.index)
    
    only_cpu = cpu_ts - gpu_ts
    only_gpu = gpu_ts - cpu_ts
    
    print(f"Signals only in CPU: {len(only_cpu)}")
    print(f"Signals only in GPU: {len(only_gpu)}\n")
    
    if only_gpu:
        print("First GPU-only signal:")
        first_gpu_only = sorted(only_gpu)[0]
        print(f"Timestamp: {first_gpu_only}\n")
        
        # Compare indicator values at that timestamp
        cpu_row = cpu_feats.loc[first_gpu_only]
        gpu_row = gpu_feats.loc[first_gpu_only]
        
        print("Indicator comparison:")
        for col in ['ema_f', 'ema_s', 'atr', 'vol_z', 'tr_z', 'cloc']:
            cpu_val = cpu_row[col]
            gpu_val = gpu_row[col]
            if pd.isna(cpu_val) or pd.isna(gpu_val):
                print(f"    {col:10s}: CPU={cpu_val}, GPU={gpu_val} (NaN)")
                continue
            diff = abs(float(cpu_val) - float(gpu_val))
            match = "✓" if diff < 1e-5 else "✗"
            print(f"  {match} {col:10s}: CPU={cpu_val:.8f}, GPU={gpu_val:.8f}, diff={diff:.8e}")
        
        print(f"\n  downtrend: CPU={cpu_row['downtrend']}, GPU={gpu_row['downtrend']}")
        
        # Check signal conditions
        print("\nSignal conditions:")
        print(f"  downtrend:   CPU={cpu_row['downtrend']}, GPU={gpu_row['downtrend']}")
        print(f"  vol_z > {params.vol_z}: CPU={cpu_row['vol_z']:.4f} > {params.vol_z} = {cpu_row['vol_z'] > params.vol_z}")
        print(f"                GPU={gpu_row['vol_z']:.4f} > {params.vol_z} = {gpu_row['vol_z'] > params.vol_z}")
        print(f"  tr_z > {params.tr_z}:  CPU={cpu_row['tr_z']:.4f} > {params.tr_z} = {cpu_row['tr_z'] > params.tr_z}")
        print(f"                GPU={gpu_row['tr_z']:.4f} > {params.tr_z} = {gpu_row['tr_z'] > params.tr_z}")
        print(f"  cloc > {params.cloc_min}: CPU={cpu_row['cloc']:.4f} > {params.cloc_min} = {cpu_row['cloc'] > params.cloc_min}")
        print(f"                GPU={gpu_row['cloc']:.4f} > {params.cloc_min} = {gpu_row['cloc'] > params.cloc_min}")
        
        print(f"\n  CPU exhaustion: {cpu_row['exhaustion']}")
        print(f"  GPU exhaustion: {gpu_row['exhaustion']}")


if __name__ == "__main__":
    asyncio.run(main())
