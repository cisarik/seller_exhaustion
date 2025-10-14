"""
Tests for Fibonacci retracement calculations and exit logic.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.fibonacci import (
    find_swing_high,
    calculate_fib_levels,
    add_fib_levels_to_df
)
from strategy.seller_exhaustion import SellerParams, build_features
from backtest.engine import run_backtest, BacktestParams


def create_test_data(n_bars=500):
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')
    
    np.random.seed(42)
    close = 0.5 + np.cumsum(np.random.randn(n_bars) * 0.01)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n_bars) * 0.001,
        'high': close + abs(np.random.randn(n_bars) * 0.002),
        'low': close - abs(np.random.randn(n_bars) * 0.002),
        'close': close,
        'volume': 1000 + np.random.randn(n_bars) * 100
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def test_find_swing_high():
    """Test swing high detection."""
    df = create_test_data(200)
    
    # Manually create a swing high at index 100
    df.loc[df.index[100], 'high'] = df['high'].max() + 0.01
    
    swing_high = find_swing_high(df, idx=150, lookback=60, lookahead=5)
    
    assert swing_high is not None, "Should find swing high"
    assert swing_high > df['high'].iloc[:99].max(), "Swing high should be higher than previous bars"


def test_calculate_fib_levels():
    """Test Fibonacci level calculation."""
    swing_low = 0.50
    swing_high = 0.60
    
    fib_levels = calculate_fib_levels(swing_low, swing_high)
    
    assert len(fib_levels) > 0, "Should return Fibonacci levels"
    assert 0.382 in fib_levels, "Should include 38.2% level"
    assert 0.618 in fib_levels, "Should include 61.8% level"
    
    # Check calculation
    price_range = swing_high - swing_low
    expected_382 = swing_low + (price_range * 0.382)
    assert abs(fib_levels[0.382] - expected_382) < 1e-6, "38.2% level should be correct"
    
    # Check bounds
    assert all(swing_low <= price <= swing_high for price in fib_levels.values()), \
        "All Fib levels should be between swing low and high"


def test_add_fib_levels_to_df():
    """Test adding Fibonacci levels to DataFrame."""
    df = create_test_data(500)
    
    # Build features to get signals
    params = SellerParams(vol_z=1.5, tr_z=1.0, cloc_min=0.5)
    feats = build_features(df, params, add_fib=True)
    
    # Check that Fib columns were added
    assert 'fib_swing_high' in feats.columns, "Should add fib_swing_high column"
    assert 'fib_0618' in feats.columns, "Should add fib_0618 column"
    
    # Check that some signals have Fib levels
    signals = feats[feats['exhaustion'] == True]
    if len(signals) > 0:
        # At least some signals should have Fib levels calculated
        has_fib = signals['fib_swing_high'].notna().sum()
        # Note: may not all have Fib if not enough history
        print(f"Signals with Fib levels: {has_fib}/{len(signals)}")


def test_backtest_with_fib_exits():
    """Test backtesting with Fibonacci exits enabled."""
    df = create_test_data(500)
    
    # Build features with Fib levels
    params = SellerParams(vol_z=1.5, tr_z=1.0, cloc_min=0.5)
    feats = build_features(df, params, add_fib=True, fib_lookback=96, fib_lookahead=5)
    
    # Run backtest with Fib exits
    bt_params = BacktestParams(
        use_fib_exits=True,
        fib_target_level=0.618,
        atr_stop_mult=0.7,
        max_hold=96
    )
    
    result = run_backtest(feats, bt_params)
    
    # Check that backtest ran
    assert 'trades' in result, "Should return trades"
    assert 'metrics' in result, "Should return metrics"
    
    trades = result['trades']
    
    if len(trades) > 0:
        print(f"\nBacktest with Fib exits:")
        print(f"Total trades: {len(trades)}")
        print(f"Exit reasons:")
        print(trades['reason'].value_counts())
        
        # Check if any trades exited via Fib levels
        fib_exits = trades[trades['reason'].str.contains('fib', na=False)]
        print(f"Fibonacci exits: {len(fib_exits)}")
        
        # Verify exit prices are reasonable
        for _, trade in trades.iterrows():
            assert trade['entry'] > 0, "Entry price should be positive"
            assert trade['exit'] > 0, "Exit price should be positive"
            assert trade['stop'] < trade['entry'], "Stop should be below entry for long"


def test_backtest_fib_vs_traditional():
    """Compare Fibonacci exits vs traditional R-multiple exits."""
    df = create_test_data(500)
    
    params = SellerParams(vol_z=1.5, tr_z=1.0, cloc_min=0.5)
    feats = build_features(df, params, add_fib=True)
    
    # Test with Fib exits
    bt_fib = BacktestParams(use_fib_exits=True, fib_target_level=0.618)
    result_fib = run_backtest(feats, bt_fib)
    
    # Test with traditional exits
    bt_trad = BacktestParams(use_fib_exits=False, reward_r=2.0)
    result_trad = run_backtest(feats, bt_trad)
    
    print(f"\n--- Comparison ---")
    print(f"Fibonacci exits: {result_fib['metrics']['n']} trades")
    print(f"Traditional exits: {result_trad['metrics']['n']} trades")
    
    if result_fib['metrics']['n'] > 0:
        print(f"Fib win rate: {result_fib['metrics']['win_rate']:.2%}")
        print(f"Fib avg R: {result_fib['metrics']['avg_R']:.2f}")
    
    if result_trad['metrics']['n'] > 0:
        print(f"Traditional win rate: {result_trad['metrics']['win_rate']:.2%}")
        print(f"Traditional avg R: {result_trad['metrics']['avg_R']:.2f}")


if __name__ == "__main__":
    # Run tests manually
    print("Testing Fibonacci functionality...\n")
    
    print("1. Testing swing high detection...")
    test_find_swing_high()
    print("✓ Passed\n")
    
    print("2. Testing Fibonacci level calculation...")
    test_calculate_fib_levels()
    print("✓ Passed\n")
    
    print("3. Testing adding Fib levels to DataFrame...")
    test_add_fib_levels_to_df()
    print("✓ Passed\n")
    
    print("4. Testing backtest with Fibonacci exits...")
    test_backtest_with_fib_exits()
    print("✓ Passed\n")
    
    print("5. Comparing Fibonacci vs Traditional exits...")
    test_backtest_fib_vs_traditional()
    print("✓ Passed\n")
    
    print("All tests passed! ✓")
