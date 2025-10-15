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


class FitnessConfig(BaseModel):
    """Fitness function configuration for optimization.
    
    Controls what the genetic algorithm optimizes for:
    - Balanced: Standard multi-objective optimization
    - High Frequency: Maximize trade count (day trading / scalping)
    - Conservative: Prioritize win rate and drawdown control
    - Custom: User-defined weights
    """
    # Preset name
    preset: str = "balanced"  # balanced, high_frequency, conservative, profit_focused, custom
    
    # Fitness component weights (must sum to ~1.0)
    trade_count_weight: float = 0.15      # Higher = more trades preferred
    win_rate_weight: float = 0.25         # Higher = higher win rate preferred
    avg_r_weight: float = 0.30            # Higher = better R-multiples preferred
    total_pnl_weight: float = 0.20        # Higher = more profit preferred
    max_drawdown_penalty: float = 0.10    # Higher = penalize drawdowns more
    
    # Minimum requirements (filters)
    min_trades: int = 10                  # Require at least N trades
    min_win_rate: float = 0.40            # Require at least 40% win rate
    
    @staticmethod
    def get_preset_config(preset_name: str) -> "FitnessConfig":
        """Get predefined fitness configuration by preset name."""
        presets = {
            "balanced": FitnessConfig(
                preset="balanced",
                trade_count_weight=0.15,
                win_rate_weight=0.25,
                avg_r_weight=0.30,
                total_pnl_weight=0.20,
                max_drawdown_penalty=0.10,
                min_trades=10,
                min_win_rate=0.40
            ),
            "high_frequency": FitnessConfig(
                preset="high_frequency",
                trade_count_weight=0.40,      # Maximize trades!
                win_rate_weight=0.20,
                avg_r_weight=0.15,
                total_pnl_weight=0.15,
                max_drawdown_penalty=0.10,
                min_trades=20,                # Need more trades
                min_win_rate=0.45             # Slightly higher win rate required
            ),
            "conservative": FitnessConfig(
                preset="conservative",
                trade_count_weight=0.05,      # Fewer trades OK
                win_rate_weight=0.35,         # Prioritize win rate
                avg_r_weight=0.25,
                total_pnl_weight=0.15,
                max_drawdown_penalty=0.20,    # Penalize drawdowns heavily
                min_trades=5,                 # OK with fewer trades
                min_win_rate=0.50             # Require 50%+ win rate
            ),
            "profit_focused": FitnessConfig(
                preset="profit_focused",
                trade_count_weight=0.10,
                win_rate_weight=0.15,
                avg_r_weight=0.20,
                total_pnl_weight=0.45,        # Maximize profit!
                max_drawdown_penalty=0.10,
                min_trades=10,
                min_win_rate=0.40
            ),
        }
        return presets.get(preset_name, presets["balanced"])


class Timeframe(str, Enum):
    m1 = "1m"
    m3 = "3m"
    m5 = "5m"
    m10 = "10m"
    m15 = "15m"
    m30 = "30m"
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
