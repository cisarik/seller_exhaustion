from __future__ import annotations
from typing import Optional
from enum import Enum
from pydantic import BaseModel


class Bar(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorBundle(BaseModel):
    ema_fast: Optional[list[float]] = None
    ema_slow: Optional[list[float]] = None
    sma: Optional[list[float]] = None
    rsi: Optional[list[float]] = None
    macd: Optional[dict] = None


class Trade(BaseModel):
    entry_ts: str
    exit_ts: str
    entry: float
    exit: float
    pnl: float
    R: float
    reason: str


class BacktestParams(BaseModel):
    """
    Backtest execution parameters.
    
    Default strategy:
    - BUY: Seller exhaustion signals only
    - SELL: First Fibonacci level hit only
    - Stop-loss and time exits are DISABLED by default (optional)
    """
    # Exit toggles
    use_stop_loss: bool = False
    use_time_exit: bool = False
    use_fib_exits: bool = True
    use_traditional_tp: bool = False
    
    # Stop-loss parameters
    atr_stop_mult: float = 0.7
    
    # Traditional TP parameters
    reward_r: float = 2.0
    
    # Time exit parameters
    max_hold: int = 96
    
    # Fibonacci parameters
    fib_swing_lookback: int = 96
    fib_swing_lookahead: int = 5
    fib_target_level: float = 0.618
    
    # Transaction costs
    fee_bp: float = 5.0
    slippage_bp: float = 5.0


class Timeframe(str, Enum):
    m1 = "1m"
    m3 = "3m"
    m5 = "5m"
    m10 = "10m"
    m15 = "15m"
    m60 = "60m"


def minutes_to_bars(minutes: int, tf: Timeframe) -> int:
    """Convert time window in minutes to number of bars for a given timeframe.
    Ensures minimum of 1 bar.
    """
    if minutes <= 0:
        return 1
    tf_map = {
        Timeframe.m1: 1,
        Timeframe.m3: 3,
        Timeframe.m5: 5,
        Timeframe.m10: 10,
        Timeframe.m15: 15,
        Timeframe.m60: 60,
    }
    bar_minutes = tf_map.get(tf, 15)
    bars = max(1, int(round(minutes / bar_minutes)))
    return bars
