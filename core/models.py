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
    """Backtest execution parameters with optional exit toggles."""
    # Fibonacci parameters (ONLY exit mechanism)
    fib_swing_lookback: int = 96      # Bars to look back for swing high
    fib_swing_lookahead: int = 5       # Bars ahead for swing confirmation
    fib_target_level: float = 0.618    # Target Fib level (0.382-1.0)
    
    # Exit toggles
    use_fib_exits: bool = True
    use_stop_loss: bool = False
    use_traditional_tp: bool = False
    use_time_exit: bool = False
    
    # Exit parameters
    atr_stop_mult: float = 0.7
    reward_r: float = 2.0
    max_hold: int = 96
    
    # Transaction costs
    fee_bp: float = 5.0      # Fees in basis points
    slippage_bp: float = 5.0  # Slippage in basis points


class FitnessConfig(BaseModel):
    """Fitness function configuration for optimization.
    
    Controls what the genetic algorithm optimizes for:
    - Balanced: Standard multi-objective optimization
    - High Frequency: Maximize trade count (day trading / scalping)
    - Conservative: Prioritize win rate and drawdown control
    - Custom: User-defined weights
    
    Supports soft penalties (continuous) instead of hard gates.
    Coach can adjust all parameters automatically.
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
    
    # NEW: Fitness function type (Coach can switch between hard gates and soft penalties)
    fitness_function_type: str = "soft_penalties"  # "hard_gates" | "soft_penalties"
    penalty_trades_strength: float = 0.7            # α: strength of trade count penalty (0-1)
    penalty_wr_strength: float = 0.5                # β: strength of win rate penalty (0-1)
    
    # NEW: Curriculum learning (Coach can enable gradual training)
    curriculum_enabled: bool = False
    curriculum_start_min_trades: int = 5            # Start with low requirement
    curriculum_increase_per_gen: int = 2            # Increase by this each N gens
    curriculum_checkpoint_gens: int = 5             # Increase every N generations
    curriculum_max_generations: int = 30            # Stop increasing after this
    
    def get_effective_min_trades(self, generation: int) -> int:
        """Get effective min_trades for this generation (supports curriculum learning)."""
        if not self.curriculum_enabled or generation >= self.curriculum_max_generations:
            return self.min_trades
        
        # Calculate how many checkpoints have passed
        checkpoints_passed = generation // self.curriculum_checkpoint_gens
        effective_min = self.curriculum_start_min_trades + (checkpoints_passed * self.curriculum_increase_per_gen)
        
        return min(effective_min, self.min_trades)
    
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
                min_win_rate=0.40,
                fitness_function_type="soft_penalties",
                penalty_trades_strength=0.7,
                penalty_wr_strength=0.5
            ),
            "high_frequency": FitnessConfig(
                preset="high_frequency",
                trade_count_weight=0.40,      # Maximize trades!
                win_rate_weight=0.20,
                avg_r_weight=0.15,
                total_pnl_weight=0.15,
                max_drawdown_penalty=0.10,
                min_trades=20,                # Need more trades
                min_win_rate=0.45,             # Slightly higher win rate required
                fitness_function_type="soft_penalties",
                penalty_trades_strength=0.7,
                penalty_wr_strength=0.5
            ),
            "conservative": FitnessConfig(
                preset="conservative",
                trade_count_weight=0.05,      # Fewer trades OK
                win_rate_weight=0.35,         # Prioritize win rate
                avg_r_weight=0.25,
                total_pnl_weight=0.15,
                max_drawdown_penalty=0.20,    # Penalize drawdowns heavily
                min_trades=5,                 # OK with fewer trades
                min_win_rate=0.50,             # Require 50%+ win rate
                fitness_function_type="soft_penalties",
                penalty_trades_strength=0.5,
                penalty_wr_strength=0.7
            ),
            "profit_focused": FitnessConfig(
                preset="profit_focused",
                trade_count_weight=0.10,
                win_rate_weight=0.15,
                avg_r_weight=0.20,
                total_pnl_weight=0.45,        # Maximize profit!
                max_drawdown_penalty=0.10,
                min_trades=10,
                min_win_rate=0.40,
                fitness_function_type="soft_penalties",
                penalty_trades_strength=0.7,
                penalty_wr_strength=0.5
            ),
        }
        return presets.get(preset_name, presets["balanced"])


class OptimizationConfig(BaseModel):
    """
    Genetic Algorithm optimization configuration.
    
    Coach can adjust ALL of these parameters automatically.
    """
    # GA hyperparameters (Coach can tune)
    population_size: int = 56
    tournament_size: int = 3
    elite_fraction: float = 0.10
    mutation_probability: float = 0.65
    mutation_rate: float = 0.28
    sigma: float = 0.12
    
    # Diversity mechanisms (Coach can enable/tune)
    immigrant_fraction: float = 0.15                 # 15% new random per gen
    immigrant_strategy: str = "worst_replacement"    # "worst_replacement" | "random"
    stagnation_threshold: int = 5                    # Generations before immigrant trigger
    stagnation_fitness_tolerance: float = 0.01       # Fitness improvement threshold
    
    # Tracking (Coach wants visibility)
    track_diversity: bool = True
    track_stagnation: bool = True
    
    # Bounds override (Coach can expand/narrow search space)
    # Format: {"ema_fast": (50, 240), "vol_z": (1.2, 1.6)}
    override_bounds: dict = {}
    
    class Config:
        extra = "allow"  # Allow additional fields for extensibility


class AdamConfig(BaseModel):
    """
    ADAM optimizer configuration.
    
    Coach can adjust ALL of these parameters automatically.
    """
    # ADAM hyperparameters (Coach can tune)
    learning_rate: float = 0.01
    epsilon: float = 0.02  # For finite differences (2% step ensures meaningful changes in integer params)
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Parallel evaluation
    n_workers: int = 4
    
    class Config:
        extra = "allow"  # Allow additional fields for extensibility


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
