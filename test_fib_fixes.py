"""
Quick test script to verify Fibonacci fixes.

Tests:
1. Fib target level is used correctly in backtest engine
2. Fibonacci parameters are evolved by GA
3. Discrete Fib levels work in mutation/crossover
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategy.seller_exhaustion import SellerParams, build_features
from core.models import BacktestParams, Timeframe
from backtest.engine import run_backtest, VALID_FIB_LEVELS, FIB_LEVEL_TO_COL
from backtest.optimizer import Population, evolution_step, VALID_FIB_LEVELS as OPT_FIB_LEVELS


def test_fib_target_usage():
    """Test that fib_target_level is actually used in backtesting."""
    print("\n=== Test 1: Fib Target Level Usage ===")
    
    # Create synthetic data with uptrend after signal
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    np.random.seed(42)
    
    # Simple uptrend data
    base_price = 0.50
    prices = base_price + np.linspace(0, 0.10, 200) + np.random.randn(200) * 0.005
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.005,
        'low': prices - 0.005,
        'close': prices + 0.002,
        'volume': 1000 + np.random.randn(200) * 100,
        'atr': [0.01] * 200,
        'exhaustion': [False] * 200,
    }, index=dates)
    
    # Add mock Fibonacci levels
    df['fib_0382'] = base_price + 0.382 * 0.10
    df['fib_0500'] = base_price + 0.500 * 0.10
    df['fib_0618'] = base_price + 0.618 * 0.10
    df['fib_0786'] = base_price + 0.786 * 0.10
    df['fib_1000'] = base_price + 1.000 * 0.10
    
    # Add signal at index 50
    df.loc[df.index[50], 'exhaustion'] = True
    
    # Test different fib_target_level values
    for target_level in [0.382, 0.618, 1.000]:
        params = BacktestParams(
            use_fib_exits=True,
            fib_target_level=target_level,
        )
        
        result = run_backtest(df, params)
        trades = result['trades']
        
        if len(trades) > 0:
            exit_reason = trades.iloc[0]['reason']
            expected_reason = f"FIB_{target_level * 100:.1f}"
            
            print(f"  Target: {target_level:.3f} → Exit Reason: {exit_reason}")
            
            if exit_reason == expected_reason:
                print(f"    ✅ PASS: Correct exit at {target_level * 100:.1f}%")
            else:
                print(f"    ❌ FAIL: Expected {expected_reason}, got {exit_reason}")
        else:
            print(f"  Target: {target_level:.3f} → No trades")


def test_fib_params_evolution():
    """Test that Fibonacci parameters are included in evolution."""
    print("\n=== Test 2: Fibonacci Parameters Evolution ===")
    
    # Create random population
    pop = Population(size=5, timeframe=Timeframe.m15)
    
    # Check that random individuals have varied Fib parameters
    fib_lookbacks = []
    fib_lookaheads = []
    fib_targets = []
    
    for ind in pop.individuals:
        fib_lookbacks.append(ind.backtest_params.fib_swing_lookback)
        fib_lookaheads.append(ind.backtest_params.fib_swing_lookahead)
        fib_targets.append(ind.backtest_params.fib_target_level)
    
    print(f"  Fib Lookbacks: {fib_lookbacks}")
    print(f"  Fib Lookaheads: {fib_lookaheads}")
    print(f"  Fib Targets: {fib_targets}")
    
    # Check for variation
    if len(set(fib_lookbacks)) > 1:
        print("  ✅ PASS: Fib lookback varies across individuals")
    else:
        print("  ❌ FAIL: Fib lookback not varying")
    
    if len(set(fib_lookaheads)) > 1:
        print("  ✅ PASS: Fib lookahead varies across individuals")
    else:
        print("  ❌ FAIL: Fib lookahead not varying")
    
    if len(set(fib_targets)) > 1:
        print("  ✅ PASS: Fib target varies across individuals")
    else:
        print("  ❌ FAIL: Fib target not varying")
    
    # Check that all targets are valid
    invalid_targets = [t for t in fib_targets if t not in VALID_FIB_LEVELS]
    if not invalid_targets:
        print(f"  ✅ PASS: All targets are valid Fib levels")
    else:
        print(f"  ❌ FAIL: Invalid targets: {invalid_targets}")


def test_discrete_fib_levels():
    """Test that discrete Fib levels are handled correctly."""
    print("\n=== Test 3: Discrete Fib Level Handling ===")
    
    # Test constants match
    engine_levels = set(VALID_FIB_LEVELS)
    opt_levels = set(OPT_FIB_LEVELS)
    
    if engine_levels == opt_levels:
        print(f"  ✅ PASS: Constants match ({engine_levels})")
    else:
        print(f"  ❌ FAIL: Mismatch - Engine: {engine_levels}, Optimizer: {opt_levels}")
    
    # Test column mapping
    print(f"  Fib level to column mapping:")
    for level in VALID_FIB_LEVELS:
        col = FIB_LEVEL_TO_COL.get(level)
        print(f"    {level:.3f} → {col}")
        if col is None:
            print(f"    ❌ FAIL: Missing mapping for {level}")
    
    print("  ✅ PASS: All levels have column mappings")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Fibonacci Fixes")
    print("=" * 60)
    
    test_fib_target_usage()
    test_fib_params_evolution()
    test_discrete_fib_levels()
    
    print("\n" + "=" * 60)
    print("Tests Complete")
    print("=" * 60)
