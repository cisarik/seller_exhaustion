import pandas as pd
import numpy as np


def ema(s: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return s.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    prev = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev).abs(),
        (low - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def zscore(s: pd.Series, window: int) -> pd.Series:
    """Calculate rolling z-score."""
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - m) / sd
