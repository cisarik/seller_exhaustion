import pytest
import pandas as pd
import numpy as np
from indicators.local import ema, sma, rsi, atr, macd, zscore


def test_ema():
    """Test EMA calculation."""
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = ema(s, span=3)
    
    assert len(result) == len(s)
    assert not result.isna().all()
    assert result.iloc[-1] > result.iloc[0]  # Uptrend


def test_sma():
    """Test SMA calculation."""
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = sma(s, window=3)
    
    assert len(result) == len(s)
    # First window-1 should be NaN
    assert result.iloc[:2].isna().all()
    # After that should have values
    assert not result.iloc[2:].isna().any()
    assert result.iloc[2] == 2.0  # (1+2+3)/3


def test_rsi():
    """Test RSI calculation."""
    # Create trending data
    s = pd.Series(list(range(1, 21)) + list(range(20, 0, -1)))
    result = rsi(s, window=14)
    
    assert len(result) == len(s)
    # RSI should be between 0 and 100
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_atr():
    """Test ATR calculation."""
    high = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    low = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
    
    result = atr(high, low, close, window=5)
    
    assert len(result) == len(high)
    # First window should have some NaN
    assert result.iloc[:4].isna().any()
    # ATR should be positive
    assert (result.dropna() > 0).all()


def test_macd():
    """Test MACD calculation."""
    s = pd.Series(list(range(1, 51)))
    result = macd(s, fast=12, slow=26, signal=9)
    
    assert 'macd' in result.columns
    assert 'signal' in result.columns
    assert 'histogram' in result.columns
    assert len(result) == len(s)


def test_zscore():
    """Test z-score calculation."""
    s = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])
    result = zscore(s, window=5)
    
    assert len(result) == len(s)
    # The spike (100) at index 5 should have high z-score
    assert result.iloc[5] > 1.9  # Should be significantly above mean
