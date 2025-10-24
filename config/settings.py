from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional
import os
import multiprocessing

# Clear environment variables on module import to prevent interference with .env file
# This ensures Settings dialog is the ONLY source of truth
_ENV_VARS_TO_CLEAR = [
    'TIMEFRAME', 'TIMEFRAME_UNIT', 'LAST_TICKER', 'LAST_DATE_FROM', 'LAST_DATE_TO',
    'STRATEGY_EMA_FAST', 'STRATEGY_EMA_SLOW', 'STRATEGY_Z_WINDOW',
    'STRATEGY_VOL_Z', 'STRATEGY_TR_Z', 'STRATEGY_CLOC_MIN', 'STRATEGY_ATR_WINDOW',
    'BACKTEST_ATR_STOP_MULT', 'BACKTEST_REWARD_R', 'BACKTEST_MAX_HOLD',
    'BACKTEST_FEE_BP', 'BACKTEST_SLIPPAGE_BP',
    'ACCELERATION_MODE', 'CPU_WORKERS', 'OPTIMIZER_WORKERS',
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
    backtest_fee_bp: float = 5.0
    backtest_slippage_bp: float = 5.0

    # Optimizer Parameters (Common)
    optimizer_iterations: int = 50
    
    # Genetic Algorithm Parameters
    ga_population_size: int = 24
    ga_mutation_rate: float = 0.3
    ga_sigma: float = 0.1
    ga_elite_fraction: float = 0.1
    ga_tournament_size: int = 3
    ga_mutation_probability: float = 0.9
    
    # ADAM Optimizer Parameters
    adam_learning_rate: float = 0.01
    adam_epsilon: float = 0.02  # Finite difference step size (was 1e-3, too small for integer params)
    adam_max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon_stability: float = 1e-8
    
    # Optimizer execution
    optimizer_workers: int = max(1, multiprocessing.cpu_count() - 1)
    
    # Evolution Coach Parameters
    coach_enabled: bool = True  # Enable/disable Evolution Coach
    coach_mode: str = "disabled"  # Coach mode: "disabled", "classic" (deterministic) or "openai" (LLM-based)
    coach_islands_enabled: bool = False  # Enable/disable Islands Management (adds complexity)
    coach_analysis_interval: int = 10  # Analyze every N generations (10, 15, 20, etc)
    coach_debug_payloads: bool = False  # When True, log full LLM payloads/responses
    coach_response_timeout: int = 3600  # LLM response timeout in seconds (3600 = 1 hour)
    
    # Logging
    ada_agent_log_level: str = "INFO"  # Root console log level: DEBUG, INFO, WARNING, ERROR
    log_progress_bars: bool = True      # Show tqdm progress bars during optimization
    log_feature_builds: bool = False    # Emit per-build feature logs (INFO); otherwise DEBUG
    
    # Agent Configuration
    agent_provider: str = "novita"  # openai, openrouter, or novita
    
    # OpenAI Parameters
    openai_api_key: str = ""  # OpenAI API key
    openai_model: str = "gpt-4o"  # Default OpenAI model
    openai_base_url: str = "https://api.openai.com/v1"  # OpenAI API base URL
    
    # Openrouter Parameters
    openrouter_api_key: str = ""  # Openrouter API key
    openrouter_model: str = "anthropic/claude-3.5-sonnet"  # Default Openrouter model
    openrouter_base_url: str = "https://openrouter.ai/api/v1"  # OpenRouter API base URL
    
    # Novita Parameters
    novita_api_key: str = ""  # Novita API key
    novita_model: str = "deepseek/deepseek-r1"  # Default Novita model
    novita_base_url: str = "https://api.novita.ai/openai"  # Novita API base URL
   
    # CPU Workers
    cpu_workers: int = 12  # CPU worker processes for optimization
    
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
    
    # Coach Tools
    coach_tool_analyze_population: bool = True
    coach_tool_get_correlation_matrix: bool = True
    coach_tool_get_param_distribution: bool = True
    coach_tool_get_param_bounds: bool = True
    coach_tool_get_generation_history: bool = True
    coach_tool_mutate_individual: bool = True
    coach_tool_insert_llm_individual: bool = True
    coach_tool_create_islands: bool = True
    coach_tool_migrate_between_islands: bool = True
    coach_tool_configure_island_scheduler: bool = True
    coach_tool_inject_immigrants: bool = True
    coach_tool_export_population: bool = True
    coach_tool_import_population: bool = True
    coach_tool_drop_individual: bool = True
    coach_tool_bulk_update_param: bool = True
    coach_tool_update_param_bounds: bool = True
    coach_tool_update_bounds_multi: bool = True
    coach_tool_reseed_population: bool = True
    coach_tool_insert_individual: bool = True
    coach_tool_update_fitness_gates: bool = True
    coach_tool_update_ga_params: bool = True
    coach_tool_update_fitness_weights: bool = True
    coach_tool_set_fitness_function_type: bool = True
    coach_tool_configure_curriculum: bool = True
    coach_tool_set_fitness_preset: bool = True
    coach_tool_set_exit_policy: bool = True
    coach_tool_set_costs: bool = True
    coach_tool_finish_analysis: bool = True
    
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

            f.write("# Optimizer Parameters (Common)\n")
            f.write(f"OPTIMIZER_ITERATIONS={existing.get('OPTIMIZER_ITERATIONS', '50')}\n\n")
            
            f.write("# Genetic Algorithm Parameters\n")
            f.write(f"GA_POPULATION_SIZE={existing.get('GA_POPULATION_SIZE', '24')}\n")
            f.write(f"GA_MUTATION_RATE={existing.get('GA_MUTATION_RATE', '0.3')}\n")
            f.write(f"GA_SIGMA={existing.get('GA_SIGMA', '0.1')}\n")
            f.write(f"GA_ELITE_FRACTION={existing.get('GA_ELITE_FRACTION', '0.1')}\n")
            f.write(f"GA_TOURNAMENT_SIZE={existing.get('GA_TOURNAMENT_SIZE', '3')}\n")
            f.write(f"GA_MUTATION_PROBABILITY={existing.get('GA_MUTATION_PROBABILITY', '0.9')}\n\n")
            
            f.write("# Optimizer Execution\n")
            f.write(f"OPTIMIZER_WORKERS={existing.get('OPTIMIZER_WORKERS', str(max(1, multiprocessing.cpu_count() - 1)))}\n\n")
            
            f.write("# ADAM Optimizer Parameters\n")
            f.write(f"ADAM_LEARNING_RATE={existing.get('ADAM_LEARNING_RATE', '0.01')}\n")
            f.write(f"ADAM_EPSILON={existing.get('ADAM_EPSILON', '0.02')}\n")  # Fixed: was 0.001
            f.write(f"ADAM_MAX_GRAD_NORM={existing.get('ADAM_MAX_GRAD_NORM', '1.0')}\n")
            f.write(f"ADAM_BETA1={existing.get('ADAM_BETA1', '0.9')}\n")
            f.write(f"ADAM_BETA2={existing.get('ADAM_BETA2', '0.999')}\n")
            f.write(f"ADAM_EPSILON_STABILITY={existing.get('ADAM_EPSILON_STABILITY', '1e-8')}\n\n")
            
            f.write("# Evolution Coach Parameters\n")
            f.write(f"COACH_ENABLED={existing.get('COACH_ENABLED', 'True')}\n")
            f.write(f"COACH_MODE={existing.get('COACH_MODE', 'classic')}\n")
            f.write(f"COACH_ISLANDS_ENABLED={existing.get('COACH_ISLANDS_ENABLED', 'False')}\n")
            f.write(f"COACH_ANALYSIS_INTERVAL={existing.get('COACH_ANALYSIS_INTERVAL', '10')}\n")
            f.write(f"COACH_DEBUG_PAYLOADS={existing.get('COACH_DEBUG_PAYLOADS', 'False')}\n")
            f.write(f"COACH_RESPONSE_TIMEOUT={existing.get('COACH_RESPONSE_TIMEOUT', '3600')}\n\n")

            f.write("# Logging\n")
            f.write(f"ADA_AGENT_LOG_LEVEL={existing.get('ADA_AGENT_LOG_LEVEL', 'INFO')}\n")
            f.write(f"LOG_PROGRESS_BARS={existing.get('LOG_PROGRESS_BARS', 'True')}\n")
            f.write(f"LOG_FEATURE_BUILDS={existing.get('LOG_FEATURE_BUILDS', 'False')}\n\n")
            
            f.write("# Agent Configuration\n")
            f.write(f"AGENT_PROVIDER={existing.get('AGENT_PROVIDER', 'novita')}\n\n")
            
            f.write("# OpenAI Parameters\n")
            f.write(f"OPENAI_API_KEY={existing.get('OPENAI_API_KEY', '')}\n")
            f.write(f"OPENAI_MODEL={existing.get('OPENAI_MODEL', 'gpt-4o')}\n")
            f.write(f"OPENAI_BASE_URL={existing.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')}\n\n")
            
            f.write("# Openrouter Parameters\n")
            f.write(f"OPENROUTER_API_KEY={existing.get('OPENROUTER_API_KEY', '')}\n")
            f.write(f"OPENROUTER_MODEL={existing.get('OPENROUTER_MODEL', 'anthropic/claude-3.5-sonnet')}\n")
            f.write(f"OPENROUTER_BASE_URL={existing.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')}\n\n")
            
            f.write("# Novita Parameters\n")
            f.write(f"NOVITA_API_KEY={existing.get('NOVITA_API_KEY', '')}\n")
            f.write(f"NOVITA_MODEL={existing.get('NOVITA_MODEL', 'deepseek/deepseek-r1')}\n")
            f.write(f"NOVITA_BASE_URL={existing.get('NOVITA_BASE_URL', 'https://api.novita.ai/openai')}\n\n")
            
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
            
            f.write("# Coach Tools\n")
            f.write(f"COACH_TOOL_ANALYZE_POPULATION={existing.get('COACH_TOOL_ANALYZE_POPULATION', 'True')}\n")
            f.write(f"COACH_TOOL_GET_CORRELATION_MATRIX={existing.get('COACH_TOOL_GET_CORRELATION_MATRIX', 'True')}\n")
            f.write(f"COACH_TOOL_GET_PARAM_DISTRIBUTION={existing.get('COACH_TOOL_GET_PARAM_DISTRIBUTION', 'True')}\n")
            f.write(f"COACH_TOOL_GET_PARAM_BOUNDS={existing.get('COACH_TOOL_GET_PARAM_BOUNDS', 'True')}\n")
            f.write(f"COACH_TOOL_GET_GENERATION_HISTORY={existing.get('COACH_TOOL_GET_GENERATION_HISTORY', 'True')}\n")
            f.write(f"COACH_TOOL_MUTATE_INDIVIDUAL={existing.get('COACH_TOOL_MUTATE_INDIVIDUAL', 'True')}\n")
            f.write(f"COACH_TOOL_INSERT_LLM_INDIVIDUAL={existing.get('COACH_TOOL_INSERT_LLM_INDIVIDUAL', 'True')}\n")
            f.write(f"COACH_TOOL_CREATE_ISLANDS={existing.get('COACH_TOOL_CREATE_ISLANDS', 'True')}\n")
            f.write(f"COACH_TOOL_MIGRATE_BETWEEN_ISLANDS={existing.get('COACH_TOOL_MIGRATE_BETWEEN_ISLANDS', 'True')}\n")
            f.write(f"COACH_TOOL_CONFIGURE_ISLAND_SCHEDULER={existing.get('COACH_TOOL_CONFIGURE_ISLAND_SCHEDULER', 'True')}\n")
            f.write(f"COACH_TOOL_INJECT_IMMIGRANTS={existing.get('COACH_TOOL_INJECT_IMMIGRANTS', 'True')}\n")
            f.write(f"COACH_TOOL_EXPORT_POPULATION={existing.get('COACH_TOOL_EXPORT_POPULATION', 'True')}\n")
            f.write(f"COACH_TOOL_IMPORT_POPULATION={existing.get('COACH_TOOL_IMPORT_POPULATION', 'True')}\n")
            f.write(f"COACH_TOOL_DROP_INDIVIDUAL={existing.get('COACH_TOOL_DROP_INDIVIDUAL', 'True')}\n")
            f.write(f"COACH_TOOL_BULK_UPDATE_PARAM={existing.get('COACH_TOOL_BULK_UPDATE_PARAM', 'True')}\n")
            f.write(f"COACH_TOOL_UPDATE_PARAM_BOUNDS={existing.get('COACH_TOOL_UPDATE_PARAM_BOUNDS', 'True')}\n")
            f.write(f"COACH_TOOL_UPDATE_BOUNDS_MULTI={existing.get('COACH_TOOL_UPDATE_BOUNDS_MULTI', 'True')}\n")
            f.write(f"COACH_TOOL_RESEED_POPULATION={existing.get('COACH_TOOL_RESEED_POPULATION', 'True')}\n")
            f.write(f"COACH_TOOL_INSERT_INDIVIDUAL={existing.get('COACH_TOOL_INSERT_INDIVIDUAL', 'True')}\n")
            f.write(f"COACH_TOOL_UPDATE_FITNESS_GATES={existing.get('COACH_TOOL_UPDATE_FITNESS_GATES', 'True')}\n")
            f.write(f"COACH_TOOL_UPDATE_GA_PARAMS={existing.get('COACH_TOOL_UPDATE_GA_PARAMS', 'True')}\n")
            f.write(f"COACH_TOOL_UPDATE_FITNESS_WEIGHTS={existing.get('COACH_TOOL_UPDATE_FITNESS_WEIGHTS', 'True')}\n")
            f.write(f"COACH_TOOL_SET_FITNESS_FUNCTION_TYPE={existing.get('COACH_TOOL_SET_FITNESS_FUNCTION_TYPE', 'True')}\n")
            f.write(f"COACH_TOOL_CONFIGURE_CURRICULUM={existing.get('COACH_TOOL_CONFIGURE_CURRICULUM', 'True')}\n")
            f.write(f"COACH_TOOL_SET_FITNESS_PRESET={existing.get('COACH_TOOL_SET_FITNESS_PRESET', 'True')}\n")
            f.write(f"COACH_TOOL_SET_EXIT_POLICY={existing.get('COACH_TOOL_SET_EXIT_POLICY', 'True')}\n")
            f.write(f"COACH_TOOL_SET_COSTS={existing.get('COACH_TOOL_SET_COSTS', 'True')}\n")
            f.write(f"COACH_TOOL_FINISH_ANALYSIS={existing.get('COACH_TOOL_FINISH_ANALYSIS', 'True')}\n")
    
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
