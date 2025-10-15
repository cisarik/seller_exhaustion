from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional
import os

# Clear environment variables on module import to prevent interference with .env file
# This ensures Settings dialog is the ONLY source of truth
_ENV_VARS_TO_CLEAR = [
    'TIMEFRAME', 'TIMEFRAME_UNIT', 'LAST_TICKER', 'LAST_DATE_FROM', 'LAST_DATE_TO',
    'STRATEGY_EMA_FAST', 'STRATEGY_EMA_SLOW', 'STRATEGY_Z_WINDOW',
    'STRATEGY_VOL_Z', 'STRATEGY_TR_Z', 'STRATEGY_CLOC_MIN', 'STRATEGY_ATR_WINDOW',
    'BACKTEST_ATR_STOP_MULT', 'BACKTEST_REWARD_R', 'BACKTEST_MAX_HOLD',
    'BACKTEST_FEE_BP', 'BACKTEST_SLIPPAGE_BP',
]

for var in _ENV_VARS_TO_CLEAR:
    if var in os.environ:
        del os.environ[var]


class Settings(BaseSettings):
    """Application settings with persistence to .env file."""
    
    # API Configuration
    polygon_api_key: str = ""
    data_dir: str = ".data"
    tz: str = "UTC"
    
    # Timeframe Configuration
    timeframe: str = "15"  # Options: 5, 15, 30, 60 (1h), 240 (4h), 720 (12h), 1440 (24h)
    timeframe_unit: str = "minute"  # minute, hour, day
    
    # Last Data Download
    last_ticker: str = "X:ADAUSD"
    last_date_from: str = "2024-01-01"
    last_date_to: str = "2024-12-31"
    
    # Strategy Parameters
    strategy_ema_fast: int = 96
    strategy_ema_slow: int = 672
    strategy_z_window: int = 672
    strategy_vol_z: float = 2.0
    strategy_tr_z: float = 1.2
    strategy_cloc_min: float = 0.6
    strategy_atr_window: int = 96
    
    # Backtest Parameters
    backtest_atr_stop_mult: float = 0.7
    backtest_reward_r: float = 2.0
    backtest_max_hold: int = 96
    backtest_fee_bp: float = 5.0
    backtest_slippage_bp: float = 5.0

    # Genetic Algorithm Parameters
    ga_population_size: int = 24
    ga_mutation_rate: float = 0.3
    ga_sigma: float = 0.1
    ga_elite_fraction: float = 0.1
    ga_tournament_size: int = 3
    ga_mutation_probability: float = 0.9
    
    # Chart Indicator Display
    chart_ema_fast: bool = True
    chart_ema_slow: bool = True
    chart_sma: bool = False
    chart_rsi: bool = False
    chart_macd: bool = False
    chart_volume: bool = False
    chart_signals: bool = True
    chart_entries: bool = True
    chart_exits: bool = True
    
    # Chart View State
    chart_x_min: Optional[float] = None
    chart_x_max: Optional[float] = None
    chart_y_min: Optional[float] = None
    chart_y_max: Optional[float] = None
    
    # Window State
    window_width: int = 1600
    window_height: int = 1000
    splitter_left: int = 1120
    splitter_right: int = 480
    
    @field_validator('chart_x_min', 'chart_x_max', 'chart_y_min', 'chart_y_max', mode='before')
    @classmethod
    def empty_str_to_none(cls, v):
        """Convert empty strings to None for optional float fields."""
        if v == '' or v is None:
            return None
        return v

    class Config:
        env_prefix = ""
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields


settings = Settings()


class SettingsManager:
    """Manager for saving and loading settings to/from .env file."""
    
    @staticmethod
    def save_to_env(settings_dict: dict):
        """Save settings dictionary to .env file."""
        env_path = ".env"
        
        # Read existing .env content
        existing = {}
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        existing[key.strip()] = value.strip()
        
        # Update with new settings
        existing.update({k.upper(): str(v) for k, v in settings_dict.items()})
        
        # Write back to .env
        with open(env_path, 'w') as f:
            f.write("# ADA Trading Agent Configuration\n")
            f.write("# Auto-saved settings\n\n")
            
            f.write("# API Configuration\n")
            f.write(f"POLYGON_API_KEY={existing.get('POLYGON_API_KEY', '')}\n")
            f.write(f"DATA_DIR={existing.get('DATA_DIR', '.data')}\n")
            f.write(f"TZ={existing.get('TZ', 'UTC')}\n\n")
            
            f.write("# Timeframe Configuration\n")
            f.write(f"TIMEFRAME={existing.get('TIMEFRAME', '15')}\n")
            f.write(f"TIMEFRAME_UNIT={existing.get('TIMEFRAME_UNIT', 'minute')}\n\n")
            
            f.write("# Last Data Download\n")
            f.write(f"LAST_TICKER={existing.get('LAST_TICKER', 'X:ADAUSD')}\n")
            f.write(f"LAST_DATE_FROM={existing.get('LAST_DATE_FROM', '2024-01-01')}\n")
            f.write(f"LAST_DATE_TO={existing.get('LAST_DATE_TO', '2024-12-31')}\n\n")
            
            f.write("# Strategy Parameters\n")
            f.write(f"STRATEGY_EMA_FAST={existing.get('STRATEGY_EMA_FAST', '96')}\n")
            f.write(f"STRATEGY_EMA_SLOW={existing.get('STRATEGY_EMA_SLOW', '672')}\n")
            f.write(f"STRATEGY_Z_WINDOW={existing.get('STRATEGY_Z_WINDOW', '672')}\n")
            f.write(f"STRATEGY_VOL_Z={existing.get('STRATEGY_VOL_Z', '2.0')}\n")
            f.write(f"STRATEGY_TR_Z={existing.get('STRATEGY_TR_Z', '1.2')}\n")
            f.write(f"STRATEGY_CLOC_MIN={existing.get('STRATEGY_CLOC_MIN', '0.6')}\n")
            f.write(f"STRATEGY_ATR_WINDOW={existing.get('STRATEGY_ATR_WINDOW', '96')}\n\n")
            
            f.write("# Backtest Parameters\n")
            f.write(f"BACKTEST_ATR_STOP_MULT={existing.get('BACKTEST_ATR_STOP_MULT', '0.7')}\n")
            f.write(f"BACKTEST_REWARD_R={existing.get('BACKTEST_REWARD_R', '2.0')}\n")
            f.write(f"BACKTEST_MAX_HOLD={existing.get('BACKTEST_MAX_HOLD', '96')}\n")
            f.write(f"BACKTEST_FEE_BP={existing.get('BACKTEST_FEE_BP', '5.0')}\n")
            f.write(f"BACKTEST_SLIPPAGE_BP={existing.get('BACKTEST_SLIPPAGE_BP', '5.0')}\n\n")

            f.write("# Genetic Algorithm Parameters\n")
            f.write(f"GA_POPULATION_SIZE={existing.get('GA_POPULATION_SIZE', '24')}\n")
            f.write(f"GA_MUTATION_RATE={existing.get('GA_MUTATION_RATE', '0.3')}\n")
            f.write(f"GA_SIGMA={existing.get('GA_SIGMA', '0.1')}\n")
            f.write(f"GA_ELITE_FRACTION={existing.get('GA_ELITE_FRACTION', '0.1')}\n")
            f.write(f"GA_TOURNAMENT_SIZE={existing.get('GA_TOURNAMENT_SIZE', '3')}\n")
            f.write(f"GA_MUTATION_PROBABILITY={existing.get('GA_MUTATION_PROBABILITY', '0.9')}\n\n")
            
            f.write("# Chart Indicator Display\n")
            f.write(f"CHART_EMA_FAST={existing.get('CHART_EMA_FAST', 'True')}\n")
            f.write(f"CHART_EMA_SLOW={existing.get('CHART_EMA_SLOW', 'True')}\n")
            f.write(f"CHART_SMA={existing.get('CHART_SMA', 'False')}\n")
            f.write(f"CHART_RSI={existing.get('CHART_RSI', 'False')}\n")
            f.write(f"CHART_MACD={existing.get('CHART_MACD', 'False')}\n")
            f.write(f"CHART_VOLUME={existing.get('CHART_VOLUME', 'False')}\n")
            f.write(f"CHART_SIGNALS={existing.get('CHART_SIGNALS', 'True')}\n")
            f.write(f"CHART_ENTRIES={existing.get('CHART_ENTRIES', 'True')}\n")
            f.write(f"CHART_EXITS={existing.get('CHART_EXITS', 'True')}\n\n")
            
            f.write("# Chart View State\n")
            f.write(f"CHART_X_MIN={existing.get('CHART_X_MIN', '')}\n")
            f.write(f"CHART_X_MAX={existing.get('CHART_X_MAX', '')}\n")
            f.write(f"CHART_Y_MIN={existing.get('CHART_Y_MIN', '')}\n")
            f.write(f"CHART_Y_MAX={existing.get('CHART_Y_MAX', '')}\n\n")
            
            f.write("# Window State\n")
            f.write(f"WINDOW_WIDTH={existing.get('WINDOW_WIDTH', '1600')}\n")
            f.write(f"WINDOW_HEIGHT={existing.get('WINDOW_HEIGHT', '1000')}\n")
            f.write(f"SPLITTER_LEFT={existing.get('SPLITTER_LEFT', '1120')}\n")
            f.write(f"SPLITTER_RIGHT={existing.get('SPLITTER_RIGHT', '480')}\n")
    
    @staticmethod
    def reload_settings():
        """Reload settings from .env file by clearing environment variable overrides."""
        global settings
        
        # Clear environment variables that might override .env file
        # This ensures .env file values take precedence
        env_vars_to_clear = [
            'TIMEFRAME', 'TIMEFRAME_UNIT', 'LAST_TICKER', 'LAST_DATE_FROM', 'LAST_DATE_TO',
            'STRATEGY_EMA_FAST', 'STRATEGY_EMA_SLOW', 'STRATEGY_Z_WINDOW',
            'STRATEGY_VOL_Z', 'STRATEGY_TR_Z', 'STRATEGY_CLOC_MIN', 'STRATEGY_ATR_WINDOW',
            'BACKTEST_ATR_STOP_MULT', 'BACKTEST_REWARD_R', 'BACKTEST_MAX_HOLD',
            'BACKTEST_FEE_BP', 'BACKTEST_SLIPPAGE_BP',
        ]
        
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
        
        settings = Settings()
        print(f"✓ Settings reloaded: timeframe={settings.timeframe}m, ticker={settings.last_ticker}")
        return settings
