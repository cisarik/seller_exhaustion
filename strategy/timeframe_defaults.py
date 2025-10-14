"""
Timeframe-specific default parameters for seller exhaustion strategy.

Critical Design Principle:
Parameters must scale with timeframe! A 15-minute strategy cannot simply be
applied to 1-minute data without adjustment.

Time-Based Approach:
- Define parameters in TIME PERIODS (minutes/hours/days)
- Convert to bars based on timeframe
- Ensures consistent lookback periods across timeframes
"""

from dataclasses import dataclass
from core.models import Timeframe, minutes_to_bars
from strategy.seller_exhaustion import SellerParams
from backtest.engine import BacktestParams


# Universal time periods (in minutes)
# These define the strategy's temporal logic
MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 1440
MINUTES_PER_WEEK = 10080

# Default time windows
DEFAULT_EMA_FAST_TIME = MINUTES_PER_DAY          # 24 hours - short-term trend
DEFAULT_EMA_SLOW_TIME = MINUTES_PER_WEEK         # 7 days - long-term trend
DEFAULT_Z_WINDOW_TIME = MINUTES_PER_WEEK         # 7 days - statistical window
DEFAULT_ATR_WINDOW_TIME = MINUTES_PER_DAY        # 24 hours - volatility window


@dataclass
class TimeframeConfig:
    """Complete configuration for a specific timeframe."""
    timeframe: Timeframe
    
    # Time-based parameters (converted to bars automatically)
    ema_fast_minutes: int
    ema_slow_minutes: int
    z_window_minutes: int
    atr_window_minutes: int
    
    # Backtest parameters (timeframe-specific)
    max_hold_minutes: int      # Maximum position hold time
    fib_lookback_minutes: int  # Fibonacci swing lookback
    
    # Universal thresholds (statistical, not time-dependent)
    vol_z: float = 2.0
    tr_z: float = 1.2
    cloc_min: float = 0.6
    
    # Backtest costs (might vary by timeframe)
    fee_bp: float = 5.0
    slippage_bp: float = 5.0
    
    def to_seller_params(self) -> SellerParams:
        """Convert to SellerParams with time-based fields."""
        return SellerParams(
            ema_fast_minutes=self.ema_fast_minutes,
            ema_slow_minutes=self.ema_slow_minutes,
            z_window_minutes=self.z_window_minutes,
            atr_window_minutes=self.atr_window_minutes,
            vol_z=self.vol_z,
            tr_z=self.tr_z,
            cloc_min=self.cloc_min,
        )
    
    def to_backtest_params(self, base_params: BacktestParams = None) -> BacktestParams:
        """
        Convert to BacktestParams with timeframe-adjusted max_hold.
        
        Args:
            base_params: Optional base parameters to inherit from
        """
        if base_params:
            # Inherit exit toggles and other settings
            return BacktestParams(
                use_stop_loss=base_params.use_stop_loss,
                use_time_exit=base_params.use_time_exit,
                use_fib_exits=base_params.use_fib_exits,
                use_traditional_tp=base_params.use_traditional_tp,
                atr_stop_mult=base_params.atr_stop_mult,
                reward_r=base_params.reward_r,
                max_hold=minutes_to_bars(self.max_hold_minutes, self.timeframe),
                fee_bp=self.fee_bp,
                slippage_bp=self.slippage_bp,
                fib_swing_lookback=minutes_to_bars(self.fib_lookback_minutes, self.timeframe),
                fib_swing_lookahead=base_params.fib_swing_lookahead,
                fib_target_level=base_params.fib_target_level,
            )
        else:
            return BacktestParams(
                max_hold=minutes_to_bars(self.max_hold_minutes, self.timeframe),
                fee_bp=self.fee_bp,
                slippage_bp=self.slippage_bp,
                fib_swing_lookback=minutes_to_bars(self.fib_lookback_minutes, self.timeframe),
            )
    
    def get_bar_counts(self) -> dict:
        """Get actual bar counts for this timeframe (for display/debugging)."""
        return {
            "ema_fast_bars": minutes_to_bars(self.ema_fast_minutes, self.timeframe),
            "ema_slow_bars": minutes_to_bars(self.ema_slow_minutes, self.timeframe),
            "z_window_bars": minutes_to_bars(self.z_window_minutes, self.timeframe),
            "atr_window_bars": minutes_to_bars(self.atr_window_minutes, self.timeframe),
            "max_hold_bars": minutes_to_bars(self.max_hold_minutes, self.timeframe),
            "fib_lookback_bars": minutes_to_bars(self.fib_lookback_minutes, self.timeframe),
        }


# Timeframe-specific configurations
TIMEFRAME_CONFIGS = {
    Timeframe.m1: TimeframeConfig(
        timeframe=Timeframe.m1,
        ema_fast_minutes=MINUTES_PER_DAY,        # 24h = 1440 bars
        ema_slow_minutes=MINUTES_PER_WEEK,       # 7d = 10080 bars
        z_window_minutes=MINUTES_PER_WEEK,       # 7d = 10080 bars
        atr_window_minutes=MINUTES_PER_DAY,      # 24h = 1440 bars
        max_hold_minutes=4 * MINUTES_PER_HOUR,   # 4 hours (don't hold 1m trades too long!)
        fib_lookback_minutes=MINUTES_PER_DAY,    # 24h swing lookback
        fee_bp=5.0,   # Higher fees matter more on 1m (more trades)
        slippage_bp=8.0,  # More slippage on fast timeframe
    ),
    
    Timeframe.m3: TimeframeConfig(
        timeframe=Timeframe.m3,
        ema_fast_minutes=MINUTES_PER_DAY,        # 24h = 480 bars
        ema_slow_minutes=MINUTES_PER_WEEK,       # 7d = 3360 bars
        z_window_minutes=MINUTES_PER_WEEK,       # 7d = 3360 bars
        atr_window_minutes=MINUTES_PER_DAY,      # 24h = 480 bars
        max_hold_minutes=8 * MINUTES_PER_HOUR,   # 8 hours
        fib_lookback_minutes=MINUTES_PER_DAY,    # 24h swing lookback
        fee_bp=5.0,
        slippage_bp=6.0,
    ),
    
    Timeframe.m5: TimeframeConfig(
        timeframe=Timeframe.m5,
        ema_fast_minutes=MINUTES_PER_DAY,        # 24h = 288 bars
        ema_slow_minutes=MINUTES_PER_WEEK,       # 7d = 2016 bars
        z_window_minutes=MINUTES_PER_WEEK,       # 7d = 2016 bars
        atr_window_minutes=MINUTES_PER_DAY,      # 24h = 288 bars
        max_hold_minutes=12 * MINUTES_PER_HOUR,  # 12 hours
        fib_lookback_minutes=MINUTES_PER_DAY,    # 24h swing lookback
        fee_bp=5.0,
        slippage_bp=5.0,
    ),
    
    Timeframe.m10: TimeframeConfig(
        timeframe=Timeframe.m10,
        ema_fast_minutes=MINUTES_PER_DAY,        # 24h = 144 bars
        ema_slow_minutes=MINUTES_PER_WEEK,       # 7d = 1008 bars
        z_window_minutes=MINUTES_PER_WEEK,       # 7d = 1008 bars
        atr_window_minutes=MINUTES_PER_DAY,      # 24h = 144 bars
        max_hold_minutes=MINUTES_PER_DAY,        # 24 hours
        fib_lookback_minutes=MINUTES_PER_DAY,    # 24h swing lookback
        fee_bp=5.0,
        slippage_bp=5.0,
    ),
    
    Timeframe.m15: TimeframeConfig(
        timeframe=Timeframe.m15,
        ema_fast_minutes=MINUTES_PER_DAY,        # 24h = 96 bars (original defaults)
        ema_slow_minutes=MINUTES_PER_WEEK,       # 7d = 672 bars
        z_window_minutes=MINUTES_PER_WEEK,       # 7d = 672 bars
        atr_window_minutes=MINUTES_PER_DAY,      # 24h = 96 bars
        max_hold_minutes=MINUTES_PER_DAY,        # 24 hours = 96 bars
        fib_lookback_minutes=MINUTES_PER_DAY,    # 24h swing lookback
        fee_bp=5.0,
        slippage_bp=5.0,
    ),
}


def get_defaults_for_timeframe(tf: Timeframe) -> TimeframeConfig:
    """
    Get recommended default parameters for a specific timeframe.
    
    Args:
        tf: Timeframe enum
    
    Returns:
        TimeframeConfig with optimal defaults
    
    Raises:
        ValueError: If timeframe not supported
    """
    if tf not in TIMEFRAME_CONFIGS:
        raise ValueError(
            f"Timeframe {tf} not supported. "
            f"Available: {list(TIMEFRAME_CONFIGS.keys())}"
        )
    
    return TIMEFRAME_CONFIGS[tf]


def print_timeframe_comparison():
    """Print comparison table of all timeframe configurations."""
    print("\n" + "="*100)
    print("TIMEFRAME PARAMETER COMPARISON")
    print("="*100)
    print(f"{'Timeframe':<12} {'EMA Fast':<15} {'EMA Slow':<15} {'Z-Window':<15} {'Max Hold':<15} {'Fib Lookback':<15}")
    print(f"{'':12} {'(bars)':<15} {'(bars)':<15} {'(bars)':<15} {'(bars)':<15} {'(bars)':<15}")
    print("-"*100)
    
    for tf in [Timeframe.m1, Timeframe.m3, Timeframe.m5, Timeframe.m10, Timeframe.m15]:
        config = TIMEFRAME_CONFIGS[tf]
        bars = config.get_bar_counts()
        
        print(
            f"{tf.value:<12} "
            f"{bars['ema_fast_bars']:<15} "
            f"{bars['ema_slow_bars']:<15} "
            f"{bars['z_window_bars']:<15} "
            f"{bars['max_hold_bars']:<15} "
            f"{bars['fib_lookback_bars']:<15}"
        )
    
    print("="*100)
    print("\nNote: All timeframes use the SAME time periods:")
    print(f"  - EMA Fast: {DEFAULT_EMA_FAST_TIME} minutes (24 hours)")
    print(f"  - EMA Slow: {DEFAULT_EMA_SLOW_TIME} minutes (7 days)")
    print(f"  - Z-Window: {DEFAULT_Z_WINDOW_TIME} minutes (7 days)")
    print(f"  - ATR Window: {DEFAULT_ATR_WINDOW_TIME} minutes (24 hours)")
    print("\nBar counts differ because timeframes have different minutes per bar.")
    print("This ensures TEMPORAL CONSISTENCY across all timeframes!")
    print("="*100 + "\n")


def validate_parameters_for_timeframe(
    seller_params: SellerParams,
    tf: Timeframe,
    tolerance: float = 0.5
) -> tuple[bool, list[str]]:
    """
    Validate if parameters are appropriate for the given timeframe.
    
    Args:
        seller_params: Parameters to validate
        tf: Timeframe being used
        tolerance: How much deviation from defaults is acceptable (0.0-1.0)
    
    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []
    defaults = get_defaults_for_timeframe(tf)
    
    # If using time-based params, they're probably fine
    if seller_params.ema_fast_minutes is not None:
        return True, []
    
    # Check bar-based parameters against defaults
    default_bars = defaults.get_bar_counts()
    
    # Check EMA fast
    expected_fast = default_bars['ema_fast_bars']
    if abs(seller_params.ema_fast - expected_fast) > expected_fast * tolerance:
        warnings.append(
            f"⚠ ema_fast={seller_params.ema_fast} bars seems inappropriate for {tf.value}. "
            f"Expected ~{expected_fast} bars (24 hours). "
            f"Consider using ema_fast_minutes={DEFAULT_EMA_FAST_TIME} instead."
        )
    
    # Check EMA slow
    expected_slow = default_bars['ema_slow_bars']
    if abs(seller_params.ema_slow - expected_slow) > expected_slow * tolerance:
        warnings.append(
            f"⚠ ema_slow={seller_params.ema_slow} bars seems inappropriate for {tf.value}. "
            f"Expected ~{expected_slow} bars (7 days). "
            f"Consider using ema_slow_minutes={DEFAULT_EMA_SLOW_TIME} instead."
        )
    
    # Check z_window
    expected_z = default_bars['z_window_bars']
    if abs(seller_params.z_window - expected_z) > expected_z * tolerance:
        warnings.append(
            f"⚠ z_window={seller_params.z_window} bars seems inappropriate for {tf.value}. "
            f"Expected ~{expected_z} bars (7 days). "
            f"Consider using z_window_minutes={DEFAULT_Z_WINDOW_TIME} instead."
        )
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


if __name__ == "__main__":
    # Print comparison when run directly
    print_timeframe_comparison()
    
    # Show example usage
    print("\nExample: Getting defaults for 1-minute timeframe")
    print("-" * 60)
    config = get_defaults_for_timeframe(Timeframe.m1)
    print(f"Timeframe: {config.timeframe.value}")
    print(f"Time periods:")
    print(f"  - EMA Fast: {config.ema_fast_minutes} minutes (24 hours)")
    print(f"  - EMA Slow: {config.ema_slow_minutes} minutes (7 days)")
    print(f"  - Max Hold: {config.max_hold_minutes} minutes (4 hours)")
    print(f"\nConverted to bars:")
    bars = config.get_bar_counts()
    for key, value in bars.items():
        print(f"  - {key}: {value} bars")
