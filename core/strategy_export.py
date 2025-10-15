"""
Strategy configuration export/import for live trading agent.

This module provides a complete parameter export format that captures ALL information
needed by a separate live trading application to execute the strategy.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import json
from pathlib import Path
from strategy.seller_exhaustion import SellerParams
from core.models import BacktestParams, Timeframe


class RiskManagementConfig(BaseModel):
    """Risk management and position sizing configuration."""
    
    # Position sizing
    risk_per_trade_percent: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Percentage of account to risk per trade (0.1-10%)"
    )
    max_position_size_percent: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Maximum position size as % of account (1-100%)"
    )
    
    # Account protection
    max_daily_loss_percent: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Maximum daily loss before stopping (1-20%)"
    )
    max_daily_trades: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum trades per day"
    )
    max_open_positions: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum concurrent open positions"
    )
    
    # Execution
    slippage_tolerance_percent: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Maximum acceptable slippage (0.1-5%)"
    )
    order_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Order timeout in seconds"
    )


class ExchangeConfig(BaseModel):
    """Exchange connection and trading configuration."""
    
    # Exchange selection
    exchange_name: str = Field(
        default="binance",
        description="Exchange name: binance, kraken, coinbase, etc."
    )
    
    # Trading pair
    trading_pair: str = Field(
        default="ADA/USDT",
        description="Trading pair symbol (e.g., ADA/USDT, ADA/USD)"
    )
    base_currency: str = Field(
        default="ADA",
        description="Base currency (e.g., ADA)"
    )
    quote_currency: str = Field(
        default="USDT",
        description="Quote currency (e.g., USDT, USD)"
    )
    
    # API credentials (NEVER export with real values, use placeholders)
    api_key: str = Field(
        default="YOUR_API_KEY_HERE",
        description="Exchange API key (configure in agent app)"
    )
    api_secret: str = Field(
        default="YOUR_API_SECRET_HERE",
        description="Exchange API secret (configure in agent app)"
    )
    api_passphrase: Optional[str] = Field(
        default=None,
        description="API passphrase (if required by exchange)"
    )
    
    # Connection settings
    testnet: bool = Field(
        default=True,
        description="Use testnet/sandbox mode (recommended for testing)"
    )
    enable_rate_limit: bool = Field(
        default=True,
        description="Enable automatic rate limiting"
    )
    
    # Paper trading
    paper_trading: bool = Field(
        default=True,
        description="Enable paper trading mode (no real orders)"
    )
    paper_initial_balance: float = Field(
        default=10000.0,
        ge=100.0,
        description="Initial balance for paper trading"
    )


class DataFeedConfig(BaseModel):
    """Real-time data feed configuration."""
    
    # Data source
    data_source: str = Field(
        default="exchange",
        description="Data source: exchange, polygon, both"
    )
    
    # Polygon.io (for historical/backup)
    polygon_api_key: str = Field(
        default="YOUR_POLYGON_KEY_HERE",
        description="Polygon.io API key (optional, for data backup)"
    )
    
    # WebSocket settings
    use_websocket: bool = Field(
        default=True,
        description="Use WebSocket for real-time data (faster)"
    )
    websocket_ping_interval: int = Field(
        default=20,
        ge=5,
        le=60,
        description="WebSocket ping interval in seconds"
    )
    
    # Fallback settings
    rest_api_interval_seconds: int = Field(
        default=60,
        ge=15,
        le=300,
        description="REST API polling interval if WebSocket unavailable"
    )
    
    # Data validation
    max_missing_bars: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum missing bars before alert"
    )
    validate_ohlcv: bool = Field(
        default=True,
        description="Validate OHLCV data integrity"
    )


class TradingConfig(BaseModel):
    """Complete trading strategy configuration for export to live agent."""
    
    # Metadata
    version: str = Field(
        default="2.1.0",
        description="Config file format version"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Creation timestamp (UTC ISO format)"
    )
    description: str = Field(
        default="",
        description="Strategy description or notes"
    )
    strategy_name: str = Field(
        default="Seller Exhaustion",
        description="Strategy name"
    )
    
    # Timeframe
    timeframe: Timeframe = Field(
        default=Timeframe.m15,
        description="Trading timeframe"
    )
    
    # Strategy parameters
    seller_params: SellerParams = Field(
        default_factory=SellerParams,
        description="Seller exhaustion signal parameters"
    )
    
    # Exit/backtest parameters
    backtest_params: BacktestParams = Field(
        default_factory=BacktestParams,
        description="Exit strategy and cost parameters"
    )
    
    # Risk management
    risk_management: RiskManagementConfig = Field(
        default_factory=RiskManagementConfig,
        description="Position sizing and risk management"
    )
    
    # Exchange configuration
    exchange: ExchangeConfig = Field(
        default_factory=ExchangeConfig,
        description="Exchange connection and credentials"
    )
    
    # Data feed configuration
    data_feed: DataFeedConfig = Field(
        default_factory=DataFeedConfig,
        description="Real-time data feed settings"
    )
    
    # Performance metadata (optional, from backtesting)
    backtest_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Backtest performance metrics (informational only)"
    )


def export_trading_config(
    config: TradingConfig,
    output_path: str | Path,
    pretty: bool = True
) -> None:
    """
    Export trading configuration to JSON file.
    
    Args:
        config: TradingConfig instance
        output_path: Output file path (.json)
        pretty: Pretty-print JSON (default True)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    config_dict = config.model_dump(mode='json')
    
    # Write to file
    with open(output_path, 'w') as f:
        if pretty:
            json.dump(config_dict, f, indent=2, sort_keys=False)
        else:
            json.dump(config_dict, f)
    
    print(f"✓ Trading config exported to: {output_path}")


def import_trading_config(input_path: str | Path) -> TradingConfig:
    """
    Import trading configuration from JSON file.
    
    Args:
        input_path: Input file path (.json)
    
    Returns:
        TradingConfig instance
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or incompatible version
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Config file not found: {input_path}")
    
    # Read file
    with open(input_path, 'r') as f:
        config_dict = json.load(f)
    
    # Check version compatibility
    version = config_dict.get('version', '1.0.0')
    major_version = int(version.split('.')[0])
    
    if major_version > 2:
        raise ValueError(
            f"Incompatible config version: {version} (max supported: 2.x.x)"
        )
    
    # Parse and validate
    config = TradingConfig(**config_dict)
    
    print(f"✓ Trading config imported from: {input_path}")
    print(f"  Strategy: {config.strategy_name}")
    print(f"  Timeframe: {config.timeframe.value}")
    print(f"  Exchange: {config.exchange.exchange_name} ({config.exchange.trading_pair})")
    print(f"  Paper Trading: {config.exchange.paper_trading}")
    
    return config


def create_default_config(
    seller_params: SellerParams,
    backtest_params: BacktestParams,
    timeframe: Timeframe = Timeframe.m15,
    description: str = "",
    backtest_metrics: Optional[Dict[str, Any]] = None
) -> TradingConfig:
    """
    Create a TradingConfig with default risk/exchange settings.
    
    This is the main function to use when exporting from the backtesting app.
    
    Args:
        seller_params: Strategy parameters from backtesting
        backtest_params: Exit parameters from backtesting
        timeframe: Trading timeframe
        description: Strategy description
        backtest_metrics: Optional backtest results
    
    Returns:
        TradingConfig ready for export
    """
    return TradingConfig(
        description=description,
        timeframe=timeframe,
        seller_params=seller_params,
        backtest_params=backtest_params,
        backtest_metrics=backtest_metrics,
        # Risk/exchange settings use safe defaults
        risk_management=RiskManagementConfig(),
        exchange=ExchangeConfig(
            paper_trading=True,  # Always start with paper trading
            testnet=True  # Always start with testnet
        ),
        data_feed=DataFeedConfig()
    )


def validate_config_for_live_trading(config: TradingConfig) -> tuple[bool, list[str]]:
    """
    Validate that configuration is safe for live trading.
    
    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check API credentials
    if config.exchange.api_key == "YOUR_API_KEY_HERE":
        warnings.append("⚠️ API key not configured (using placeholder)")
    
    if config.exchange.api_secret == "YOUR_API_SECRET_HERE":
        warnings.append("⚠️ API secret not configured (using placeholder)")
    
    # Check paper trading
    if not config.exchange.paper_trading:
        warnings.append("⚠️ LIVE TRADING ENABLED - Real money at risk!")
    
    # Check testnet
    if not config.exchange.testnet and not config.exchange.paper_trading:
        warnings.append("⚠️ Production exchange + live trading - EXTREME RISK!")
    
    # Check risk limits
    if config.risk_management.risk_per_trade_percent > 2.0:
        warnings.append(f"⚠️ High risk per trade: {config.risk_management.risk_per_trade_percent}%")
    
    if config.risk_management.max_daily_loss_percent > 10.0:
        warnings.append(f"⚠️ High max daily loss: {config.risk_management.max_daily_loss_percent}%")
    
    # Check timeframe
    if config.timeframe == Timeframe.m1:
        warnings.append("ℹ️ 1-minute timeframe - Very active trading")
    
    is_valid = len(warnings) == 0
    
    return is_valid, warnings


# Example usage for testing
if __name__ == "__main__":
    # Create example config
    config = create_default_config(
        seller_params=SellerParams(),
        backtest_params=BacktestParams(),
        timeframe=Timeframe.m15,
        description="Default 15m seller exhaustion strategy with Fibonacci exits",
        backtest_metrics={
            "total_trades": 45,
            "win_rate": 0.56,
            "avg_R": 0.42,
            "total_pnl": 0.1234,
            "max_drawdown": -0.0456,
            "sharpe": 0.89
        }
    )
    
    # Export
    export_trading_config(config, "example_strategy.json")
    
    # Import and validate
    imported = import_trading_config("example_strategy.json")
    is_valid, warnings = validate_config_for_live_trading(imported)
    
    print(f"\nValidation: {'✓ SAFE' if is_valid else '⚠ WARNINGS'}")
    for warning in warnings:
        print(f"  {warning}")
