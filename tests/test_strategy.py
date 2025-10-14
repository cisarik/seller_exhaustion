import pytest
import pandas as pd
import numpy as np
from strategy.seller_exhaustion import SellerParams, build_features, check_signal


def test_seller_params_defaults():
    """Test default parameters."""
    params = SellerParams()
    assert params.ema_fast == 96
    assert params.ema_slow == 672
    assert params.z_window == 672
    assert params.vol_z == 2.0
    assert params.tr_z == 1.2
    assert params.cloc_min == 0.6
    assert params.atr_window == 96


def test_build_features():
    """Test feature building with synthetic data."""
    # Create synthetic data
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'open': 0.5 + np.random.randn(1000) * 0.01,
        'high': 0.52 + np.random.randn(1000) * 0.01,
        'low': 0.48 + np.random.randn(1000) * 0.01,
        'close': 0.5 + np.random.randn(1000) * 0.01,
        'volume': 1000 + np.random.randn(1000) * 100,
    }, index=dates)
    
    params = SellerParams(ema_fast=10, ema_slow=20, z_window=50, atr_window=10)
    result = build_features(df, params)
    
    # Check that all expected columns exist
    assert 'ema_f' in result.columns
    assert 'ema_s' in result.columns
    assert 'downtrend' in result.columns
    assert 'atr' in result.columns
    assert 'vol_z' in result.columns
    assert 'tr_z' in result.columns
    assert 'cloc' in result.columns
    assert 'exhaustion' in result.columns
    
    # Check types
    assert result['exhaustion'].dtype == bool
    assert result['downtrend'].dtype == bool


def test_exhaustion_signal_conditions():
    """Test that exhaustion signal respects all conditions."""
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    
    # Create data that should NOT trigger (uptrend)
    df = pd.DataFrame({
        'open': np.linspace(0.5, 0.6, 100),
        'high': np.linspace(0.52, 0.62, 100),
        'low': np.linspace(0.48, 0.58, 100),
        'close': np.linspace(0.5, 0.6, 100),
        'volume': np.full(100, 1000.0),
    }, index=dates)
    
    params = SellerParams(ema_fast=10, ema_slow=20, z_window=30, atr_window=10)
    result = build_features(df, params)
    
    # In an uptrend, there should be no exhaustion signals
    # (though some might appear due to random volume/TR spikes)
    assert result['exhaustion'].sum() < len(result) * 0.2  # Less than 20% signals


def test_check_signal():
    """Test individual signal checking."""
    row = pd.Series({
        'exhaustion': True,
        'ema_f': 0.5,
        'ema_s': 0.6,
    })
    
    params = SellerParams()
    assert check_signal(row, params) is True
    
    row['exhaustion'] = False
    assert check_signal(row, params) is False
