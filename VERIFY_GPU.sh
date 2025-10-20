#!/bin/bash
# Quick GPU verification script

echo "=================================="
echo "GPU Acceleration Verification"
echo "=================================="
echo ""

echo "1. Checking GPU Hardware..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

echo "2. Checking Python Environment..."
.venv/bin/python -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

echo "3. Checking Spectre..."
.venv/bin/python -c "
from spectre import factors
engine = factors.FactorEngine.__dict__
print('✓ Spectre: Available')
print('✓ FactorEngine: Ready')
"
echo ""

echo "4. Checking Current Settings..."
if [ -f .env ]; then
    GPU_SETTING=$(grep "USE_SPECTRE_CUDA" .env || echo "NOT FOUND")
    echo "Settings: $GPU_SETTING"
else
    echo "⚠ .env file not found"
fi
echo ""

echo "5. Quick Performance Test..."
.venv/bin/python3 << 'PYEOF'
import time
import pandas as pd
import numpy as np
from strategy.seller_exhaustion import build_features, SellerParams
from core.models import Timeframe

# Create test data
dates = pd.date_range('2020-01-01', periods=1000, freq='15min', tz='UTC')
df = pd.DataFrame({
    'open': np.random.randn(1000).cumsum() * 0.01 + 0.5,
    'high': np.random.randn(1000).cumsum() * 0.01 + 0.52,
    'low': np.random.randn(1000).cumsum() * 0.01 + 0.48,
    'close': np.random.randn(1000).cumsum() * 0.01 + 0.5,
    'volume': np.random.rand(1000) * 10000 + 5000,
}, index=dates)

params = SellerParams()

# Test
start = time.perf_counter()
feats = build_features(df, params, Timeframe.m15, use_spectre=True)
elapsed = time.perf_counter() - start

print(f'✓ Features computed: {elapsed:.3f}s')
print(f'✓ Signals detected: {feats["exhaustion"].sum()}')
PYEOF
echo ""

echo "✅ GPU Verification Complete!"
echo "=================================="
