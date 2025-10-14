"""
Fibonacci retracement calculations for exit signals.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def find_swing_high(df: pd.DataFrame, idx: int, lookback: int = 20, lookahead: int = 5) -> Optional[float]:
    """
    Find the most recent swing high before the given index.
    
    A swing high is a local maximum where:
    - High[i] > High[i-lookback:i]
    - High[i] > High[i+1:i+lookahead]
    
    Args:
        df: DataFrame with 'high' column
        idx: Current bar index
        lookback: Bars to look back for swing high
        lookahead: Bars ahead of peak for confirmation (default 5)
    
    Returns:
        Swing high price, or None if not found
    """
    if idx < lookback + lookahead:
        return None
    
    # Search backwards from idx for swing high
    for i in range(idx - lookahead, max(lookback, 0), -1):
        if i >= len(df):
            continue
            
        high_price = df.iloc[i]["high"]
        
        # Check if it's higher than lookback period
        left_ok = True
        for j in range(max(0, i - lookback), i):
            if df.iloc[j]["high"] >= high_price:
                left_ok = False
                break
        
        if not left_ok:
            continue
        
        # Check if it's higher than lookahead period
        right_ok = True
        for j in range(i + 1, min(i + lookahead + 1, len(df))):
            if df.iloc[j]["high"] >= high_price:
                right_ok = False
                break
        
        if right_ok:
            return high_price
    
    return None


def calculate_fib_levels(
    swing_low: float,
    swing_high: float,
    levels: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
) -> dict[float, float]:
    """
    Calculate Fibonacci retracement levels.
    
    For a LONG position (buying at swing low, targeting swing high):
    - Fib levels are calculated from low to high
    - Level 0.0 = swing_low (100% retracement)
    - Level 1.0 = swing_high (0% retracement)
    
    Args:
        swing_low: Recent low price (entry area)
        swing_high: Previous high price (target area)
        levels: Fib ratios to calculate
    
    Returns:
        Dict mapping fib_level -> price
        Example: {0.382: 0.512, 0.5: 0.515, 0.618: 0.518, ...}
    """
    if swing_high <= swing_low:
        return {}
    
    price_range = swing_high - swing_low
    fib_prices = {}
    
    for level in levels:
        fib_prices[level] = swing_low + (price_range * level)
    
    return fib_prices


def add_fib_levels_to_df(
    df: pd.DataFrame,
    signal_col: str = "exhaustion",
    lookback: int = 96,
    lookahead: int = 5,
    levels: Tuple[float, ...] = (0.382, 0.5, 0.618, 0.786, 1.0)
) -> pd.DataFrame:
    """
    Add Fibonacci retracement levels to DataFrame for each signal.
    
    For each exhaustion signal:
    1. Find swing high in the lookback period
    2. Use signal low as swing low
    3. Calculate Fib levels
    4. Store levels for use during position management
    
    Args:
        df: DataFrame with OHLC and signal column
        signal_col: Column name for entry signals
        lookback: Bars to look back for swing high
        lookahead: Bars for swing high confirmation
        levels: Fib ratios to calculate
    
    Returns:
        DataFrame with additional columns:
        - fib_swing_high: Swing high used for Fib calculation
        - fib_0382, fib_0500, fib_0618, fib_0786, fib_1000: Price levels
    """
    out = df.copy()
    
    # Initialize columns
    out["fib_swing_high"] = np.nan
    for level in levels:
        col_name = f"fib_{int(level * 1000):04d}"
        out[col_name] = np.nan
    
    # Process each signal
    for i in range(len(out)):
        if not bool(out.iloc[i].get(signal_col, False)):
            continue
        
        # Find swing high
        swing_high = find_swing_high(out, i, lookback, lookahead)
        if swing_high is None:
            continue
        
        swing_low = out.iloc[i]["low"]
        
        # Calculate Fib levels
        fib_prices = calculate_fib_levels(swing_low, swing_high, levels)
        
        # Store in DataFrame
        out.iloc[i, out.columns.get_loc("fib_swing_high")] = swing_high
        
        for level, price in fib_prices.items():
            col_name = f"fib_{int(level * 1000):04d}"
            if col_name in out.columns:
                out.iloc[i, out.columns.get_loc(col_name)] = price
    
    return out
