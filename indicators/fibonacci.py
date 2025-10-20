"""
Fibonacci retracement calculations for exit signals.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def find_swing_high(df: pd.DataFrame, idx: int, lookback: int = 20, lookahead: int = 5) -> Optional[float]:
    """
    Find the most recent swing high before the given index.
    
    Uses a RELAXED approach: finds the highest high in the lookback window
    that has at least `lookahead` bars of confirmation after it.
    
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
    # Use a sliding window approach to find local maxima
    search_start = max(0, idx - lookback)
    search_end = idx - lookahead
    
    if search_start >= search_end:
        return None
    
    # Find all potential swing highs (local maxima)
    candidates = []
    
    for i in range(search_start + lookahead, search_end):
        if i >= len(df):
            continue
            
        high_price = df.iloc[i]["high"]
        
        # Check if this is a local maximum
        # (higher than `lookahead` bars before and after)
        is_local_max = True
        
        # Check bars before
        for j in range(max(0, i - lookahead), i):
            if df.iloc[j]["high"] > high_price:
                is_local_max = False
                break
        
        if not is_local_max:
            continue
        
        # Check bars after
        for j in range(i + 1, min(i + lookahead + 1, len(df))):
            if df.iloc[j]["high"] > high_price:
                is_local_max = False
                break
        
        if is_local_max:
            candidates.append((i, high_price))
    
    # Return the highest swing high found
    if candidates:
        # Sort by price (descending) and return the highest
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][1]
    
    # Fallback: if no swing high found with strict criteria,
    # just return the max high in the lookback window
    window_start = max(0, idx - lookback)
    window_end = idx - lookahead
    if window_start < window_end:
        return df.iloc[window_start:window_end]['high'].max()
    
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
    
    # Process each signal (or all bars if signal_col not in df)
    for i in range(len(out)):
        # Skip if signal column exists and is False
        if signal_col in out.columns:
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
