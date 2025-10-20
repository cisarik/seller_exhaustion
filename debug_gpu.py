#!/usr/bin/env python3
"""
GPU Diagnostics Script - Check if GPU is actually being utilized
"""

import sys
import logging

# Configure logging to see all warnings
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("GPU DIAGNOSTICS")
print("=" * 70)

# 1. Check PyTorch
print("\n1. PyTorch Status")
print("-" * 70)
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("✗ CUDA not available - GPU acceleration will not work")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

# 2. Check Spectre
print("\n2. Spectre Status")
print("-" * 70)
try:
    from spectre import factors
    from spectre.data import MemoryLoader
    print("✓ Spectre core modules available")
    
    # Try to check if Spectre has GPU support
    try:
        engine = factors.FactorEngine(MemoryLoader.__dict__)
        print(f"✓ FactorEngine created")
        
        # Check if to_cuda method exists
        if hasattr(engine, 'to_cuda'):
            print("✓ FactorEngine has to_cuda() method")
        else:
            print("✗ FactorEngine does NOT have to_cuda() method")
            
    except Exception as e:
        print(f"⚠ Could not create FactorEngine: {e}")
        
except ImportError as e:
    print(f"✗ Spectre not available: {e}")
    sys.exit(1)

# 3. Check current settings
print("\n3. Current Settings")
print("-" * 70)
try:
    from config.settings import settings
    print(f"USE_SPECTRE: {getattr(settings, 'use_spectre', 'N/A')}")
    print(f"USE_SPECTRE_CUDA: {getattr(settings, 'use_spectre_cuda', 'N/A')}")
    print(f"USE_SPECTRE_TRADING: {getattr(settings, 'use_spectre_trading', 'N/A')}")
except Exception as e:
    print(f"⚠ Could not read settings: {e}")

# 4. Test actual GPU computation
print("\n4. GPU Computation Test")
print("-" * 70)
try:
    import pandas as pd
    import numpy as np
    from strategy.seller_exhaustion import build_features, SellerParams
    from core.models import Timeframe
    import asyncio
    import time
    
    # Create sample data
    print("Creating sample data...")
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min', tz='UTC')
    df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() * 0.01 + 0.5,
        'high': np.random.randn(1000).cumsum() * 0.01 + 0.52,
        'low': np.random.randn(1000).cumsum() * 0.01 + 0.48,
        'close': np.random.randn(1000).cumsum() * 0.01 + 0.5,
        'volume': np.random.rand(1000) * 10000 + 5000,
    }, index=dates)
    
    params = SellerParams()
    
    # Test with CPU first
    print("\nCPU Feature Computation (pandas):")
    start = time.perf_counter()
    feats_cpu = build_features(df.copy(), params, Timeframe.m15, use_spectre=False)
    cpu_time = time.perf_counter() - start
    print(f"  Time: {cpu_time:.3f}s")
    print(f"  Signals: {feats_cpu['exhaustion'].sum()}")
    
    # Test with Spectre (CPU mode)
    print("\nSpectre Feature Computation (CPU mode):")
    start = time.perf_counter()
    feats_spectre_cpu = build_features(df.copy(), params, Timeframe.m15, use_spectre=True)
    spectre_cpu_time = time.perf_counter() - start
    print(f"  Time: {spectre_cpu_time:.3f}s")
    print(f"  Signals: {feats_spectre_cpu['exhaustion'].sum()}")
    print(f"  Speedup vs pandas: {cpu_time / spectre_cpu_time:.2f}x")
    
    # Check if they match
    if feats_cpu['exhaustion'].sum() == feats_spectre_cpu['exhaustion'].sum():
        print("  ✓ Results match")
    else:
        print("  ✗ Results differ!")
            
except Exception as e:
    print(f"✗ Error during GPU test: {e}")
    import traceback
    traceback.print_exc()

# 5. Check GPU memory if available
print("\n5. GPU Memory Status")
print("-" * 70)
try:
    import torch
    if torch.cuda.is_available():
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
        print(f"Reserved memory: {torch.cuda.memory_reserved(0) / 1e9:.3f} GB")
    else:
        print("GPU not available")
except Exception as e:
    print(f"⚠ Could not check GPU memory: {e}")

print("\n" + "=" * 70)
print("END OF DIAGNOSTICS")
print("=" * 70)
