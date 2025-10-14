import pytest
import pandas as pd
import numpy as np
from backtest.engine import run_backtest, BacktestParams
from strategy.seller_exhaustion import build_features, SellerParams


def test_backtest_params_defaults():
    """Test default backtest parameters."""
    params = BacktestParams()
    assert params.atr_stop_mult == 0.7
    assert params.reward_r == 2.0
    assert params.max_hold == 96
    assert params.fee_bp == 5.0
    assert params.slippage_bp == 5.0


def test_run_backtest_no_signals():
    """Test backtest with no signals."""
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    df = pd.DataFrame({
        'open': np.full(100, 0.5),
        'high': np.full(100, 0.52),
        'low': np.full(100, 0.48),
        'close': np.full(100, 0.5),
        'volume': np.full(100, 1000.0),
        'atr': np.full(100, 0.01),
        'exhaustion': np.full(100, False),
    }, index=dates)
    
    params = BacktestParams()
    result = run_backtest(df, params)
    
    assert result['metrics']['n'] == 0
    assert result['metrics']['win_rate'] == 0.0
    assert result['metrics']['total_pnl'] == 0.0
    assert len(result['trades']) == 0


def test_run_backtest_with_signal():
    """Test backtest with a winning signal."""
    dates = pd.date_range('2024-01-01', periods=20, freq='15min')
    
    # Create scenario: signal at bar 5, price goes up
    df = pd.DataFrame({
        'open': [0.50, 0.50, 0.50, 0.50, 0.50,  # 0-4
                 0.50, 0.51, 0.52, 0.53, 0.54,  # 5-9 (entry at 6)
                 0.55, 0.56, 0.57, 0.58, 0.59,  # 10-14
                 0.60, 0.60, 0.60, 0.60, 0.60], # 15-19
        'high': [0.51] * 20,
        'low': [0.49] * 20,
        'close': [0.50] * 20,
        'volume': [1000.0] * 20,
        'atr': [0.01] * 20,
        'exhaustion': [False] * 5 + [True] + [False] * 14,
    }, index=dates)
    
    # Adjust high to allow TP hit
    df.loc[df.index[10], 'high'] = 0.60  # Should hit TP
    
    params = BacktestParams(atr_stop_mult=0.7, reward_r=2.0, max_hold=96)
    result = run_backtest(df, params)
    
    # Should have at least one trade
    assert result['metrics']['n'] >= 1
    
    # Check trade structure
    if len(result['trades']) > 0:
        trade = result['trades'].iloc[0]
        assert 'entry' in trade
        assert 'exit' in trade
        assert 'pnl' in trade
        assert 'R' in trade
        assert 'reason' in trade


def test_backtest_determinism():
    """Test that backtest produces same results with same inputs."""
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'open': 0.5 + np.random.randn(200) * 0.01,
        'high': 0.52 + np.random.randn(200) * 0.01,
        'low': 0.48 + np.random.randn(200) * 0.01,
        'close': 0.5 + np.random.randn(200) * 0.01,
        'volume': 1000 + np.random.randn(200) * 100,
    }, index=dates)
    
    strategy_params = SellerParams(ema_fast=10, ema_slow=20, z_window=30, atr_window=10)
    feats = build_features(df, strategy_params)
    
    bt_params = BacktestParams()
    
    # Run twice
    result1 = run_backtest(feats, bt_params)
    result2 = run_backtest(feats, bt_params)
    
    # Should be identical
    assert result1['metrics']['n'] == result2['metrics']['n']
    assert result1['metrics']['total_pnl'] == result2['metrics']['total_pnl']
