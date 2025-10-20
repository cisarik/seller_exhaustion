#!/usr/bin/env python3
import asyncio
from enum import Enum

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table

from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from backtest.engine import run_backtest
from core.models import Timeframe, BacktestParams
from backtest.metrics import print_metrics
from backtest.optimizer_factory import create_optimizer
from core.models import FitnessConfig


class OptimizerChoice(str, Enum):
    evolutionary = "evolutionary"
    adam = "adam"

app = typer.Typer(help="ADA Seller-Exhaustion Agent CLI")
console = Console()


@app.command()
def fetch(
    ticker: str = typer.Option("X:ADAUSD", help="Ticker symbol"),
    from_date: str = typer.Option("2024-01-01", "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option("2025-01-13", "--to", help="End date (YYYY-MM-DD)"),
    tf: Timeframe = typer.Option("15m", case_sensitive=False, help="Timeframe: 1m,3m,5m,10m,15m,60m"),
):
    """Fetch bar data from Polygon.io with selectable timeframe."""
    
    async def _run():
        console.print(f"[cyan]Fetching {ticker} from {from_date} to {to_date}...[/cyan]")
        dp = DataProvider()
        
        try:
            df = await dp.fetch(ticker, tf, from_date, to_date)
            
            console.print(f"[green]✓ Fetched {len(df)} bars[/green]")
            console.print(f"\nFirst 5 rows:")
            console.print(df.head())
            console.print(f"\nLast 5 rows:")
            console.print(df.tail())
            console.print(f"\nData shape: {df.shape}")
            console.print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
        finally:
            await dp.close()
    
    asyncio.run(_run())


@app.command()
def backtest(
    ticker: str = typer.Option("X:ADAUSD", help="Ticker symbol"),
    from_date: str = typer.Option("2024-01-01", "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option("2025-01-13", "--to", help="End date (YYYY-MM-DD)"),
    tf: Timeframe = typer.Option("15m", case_sensitive=False, help="Timeframe: 1m,3m,5m,10m,15m,60m"),
    ema_fast_min: int = typer.Option(96 * 15, help="Fast EMA window in minutes"),
    ema_slow_min: int = typer.Option(672 * 15, help="Slow EMA window in minutes"),
    z_window_min: int = typer.Option(672 * 15, help="Z-score lookback in minutes"),
    vol_z: float = typer.Option(2.0, help="Volume z-score threshold"),
    tr_z: float = typer.Option(1.2, help="True range z-score threshold"),
    cloc_min: float = typer.Option(0.6, help="Minimum close location in candle"),
    fib_target_level: float = typer.Option(0.618, "--fib", help="Fibonacci target level: 0.382, 0.5, 0.618, 0.786, 1.0"),
    fee_bp: float = typer.Option(5.0, help="Fee in basis points"),
    slippage_bp: float = typer.Option(5.0, help="Slippage in basis points"),
    output: str = typer.Option("trades.csv", help="Output CSV file for trades"),
    data: str = typer.Option("", help="Path to cached OHLCV DataFrame (parquet/pickle) to use instead of fetching; e.g. .data/X_ADAUSD_2025-09-14_2025-10-14_15minute.parquet"),
):
    """Run backtest on historical data"""
    
    async def _run():
        console.print(f"[cyan]Running backtest on {ticker}...[/cyan]")
        
        dp = DataProvider()
        
        try:
            # Fetch data
            if data:
                console.print(f"[cyan]Loading cached data from {data}...[/cyan]")
                if data.endswith(".parquet"):
                    df = pd.read_parquet(data)
                elif data.endswith(".pkl") or data.endswith(".pickle"):
                    df = pd.read_pickle(data)
                else:
                    raise ValueError("Unsupported --data file type; use .parquet or .pkl")
                if df.index.name == 'ts':
                    df.index = pd.to_datetime(df.index, utc=True)
                console.print(f"[green]✓ Loaded {len(df)} bars from cache[/green]")
            else:
                console.print(f"[cyan]Fetching data from {from_date} to {to_date}...[/cyan]")
                df = await dp.fetch(ticker, tf, from_date, to_date)
                console.print(f"[green]✓ Fetched {len(df)} bars[/green]")
            
            # Build features
            console.print("[cyan]Building features...[/cyan]")
            params = SellerParams(
                ema_fast_minutes=ema_fast_min,
                ema_slow_minutes=ema_slow_min,
                z_window_minutes=z_window_min,
                atr_window_minutes=ema_fast_min,
                vol_z=vol_z,
                tr_z=tr_z,
                cloc_min=cloc_min,
            )
            feats = build_features(df, params, tf)
            console.print(f"[green]✓ Detected {feats['exhaustion'].sum()} signals[/green]")
            
            # Run backtest
            console.print("[cyan]Running backtest...[/cyan]")
            # Normalize Fibonacci target to valid set if needed
            valid_levels = {0.382, 0.5, 0.618, 0.786, 1.0}
            if fib_target_level not in valid_levels:
                # pick nearest
                fib_target_level = min(valid_levels, key=lambda x: abs(x - fib_target_level))

            bt_params = BacktestParams(
                fib_target_level=float(fib_target_level),
                fee_bp=fee_bp,
                slippage_bp=slippage_bp,
            )
            result = run_backtest(feats, bt_params)
            
            # Display results
            console.print("[green]✓ Backtest complete[/green]\n")
            print_metrics(result["metrics"])
            
            # Save trades
            if len(result["trades"]) > 0:
                result["trades"].to_csv(output, index=False)
                console.print(f"[green]✓ Saved {len(result['trades'])} trades to {output}[/green]")
                
                # Show sample trades
                console.print("\nSample trades:")
                table = Table()
                table.add_column("Entry", style="cyan")
                table.add_column("Exit", style="cyan")
                table.add_column("PnL", style="green")
                table.add_column("R", style="yellow")
                table.add_column("Reason", style="magenta")
                
                for _, trade in result["trades"].head(10).iterrows():
                    pnl_color = "green" if trade["pnl"] > 0 else "red"
                    table.add_row(
                        str(trade["entry"])[:10],
                        str(trade["exit"])[:10],
                        f"[{pnl_color}]{trade['pnl']:.4f}[/{pnl_color}]",
                        f"{trade['R']:.2f}",
                        trade["reason"]
                    )
                
                console.print(table)
            else:
                console.print("[yellow]⚠ No trades generated[/yellow]")
            
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            await dp.close()
    
    asyncio.run(_run())


@app.command()
def ui(
    ga_init_from: str = typer.Option("", help="Path to GA population JSON to initialize and auto-start optimization"),
):
    """Launch the PySide6 UI (optionally initialize GA from file and auto-start)."""
    console.print("[cyan]Launching UI...[/cyan]")
    from app.main import main
    main(ga_init_from=ga_init_from or None)


@app.command("ga-export")
def ga_export(
    output: str = typer.Argument(..., help="Output JSON file path for population"),
    size: int = typer.Option(None, help="Population size (defaults to GA_POPULATION_SIZE)"),
    timeframe: str = typer.Option(None, help="Override timeframe (e.g., 1m,3m,5m,10m,15m,30m,60m); defaults to settings.timeframe"),
):
    """Export a GA population (seeded by current settings) to JSON."""
    from backtest.optimizer import Population, Individual, export_population
    from config.settings import settings as app_settings
    from strategy.seller_exhaustion import SellerParams
    from core.models import BacktestParams

    # Determine TF
    tf = None
    if timeframe is None:
        # Map minutes in settings to enum
        mapping = {
            1: Timeframe.m1, 3: Timeframe.m3, 5: Timeframe.m5, 10: Timeframe.m10,
            15: Timeframe.m15, 30: Timeframe.m30, 60: Timeframe.m60
        }
        try:
            tf = mapping.get(int(app_settings.timeframe), Timeframe.m15)
        except Exception:
            tf = Timeframe.m15
    else:
        tf_map = {
            "1m": Timeframe.m1, "3m": Timeframe.m3, "5m": Timeframe.m5,
            "10m": Timeframe.m10, "15m": Timeframe.m15, "30m": Timeframe.m30, "60m": Timeframe.m60
        }
        tf = tf_map.get(timeframe.lower(), Timeframe.m15)

    # Seed params from .env
    seed_seller = SellerParams(
        ema_fast=int(app_settings.strategy_ema_fast),
        ema_slow=int(app_settings.strategy_ema_slow),
        z_window=int(app_settings.strategy_z_window),
        vol_z=float(app_settings.strategy_vol_z),
        tr_z=float(app_settings.strategy_tr_z),
        cloc_min=float(app_settings.strategy_cloc_min),
        atr_window=int(app_settings.strategy_atr_window),
    )
    seed_backtest = BacktestParams(
        fee_bp=float(app_settings.backtest_fee_bp),
        slippage_bp=float(app_settings.backtest_slippage_bp),
    )
    seed = Individual(seller_params=seed_seller, backtest_params=seed_backtest)

    pop_size = int(size) if size is not None else int(app_settings.ga_population_size)
    population = Population(size=pop_size, seed_individual=seed, timeframe=tf)
    export_population(population, output)
    console.print(f"[green]✓ Exported population ({pop_size}) to {output}[/green]")


@app.command("ga-init-from")
def ga_init_from(
    path: str = typer.Argument(..., help="Population JSON path to initialize UI and auto-start optimization"),
):
    """Launch UI with GA initialized from a population file and auto-start optimization."""
    from app.main import main
    console.print(f"[cyan]Launching UI with GA init from: {path}[/cyan]")
    main(ga_init_from=path)


@app.command()
def optimize(
    ticker: str = typer.Option("X:ADAUSD", "--ticker", help="Ticker symbol"),
    from_date: str = typer.Option("2024-01-01", "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option("2025-01-13", "--to", help="End date (YYYY-MM-DD)"),
    tf: Timeframe = typer.Option("15m", "--tf", case_sensitive=False, help="Timeframe: 1m,3m,5m,10m,15m,60m"),
    optimizer: OptimizerChoice = typer.Option(OptimizerChoice.evolutionary, "--optimizer", "-o", help="Optimizer type: evolutionary|adam"),
    init_from: str = typer.Option("", "--init-from", "-i", help="Path to population JSON to initialize optimizer (for GA uses full pop; for ADAM seeds params)"),
    generations: int = typer.Option(10, "--generations", "-g", help="Number of optimization steps/generations to run"),
    data: str = typer.Option("", "--data", help="Path to cached OHLCV DataFrame (parquet/pickle) to use instead of fetching; e.g. .data/X_ADAUSD_2025-09-14_2025-10-14_15minute.parquet"),
):
    """Run optimization headlessly (CLI) with chosen optimizer and optional population seed file."""

    async def _run():
        console.print(f"[cyan]Optimizing {ticker} on {tf.value} using {optimizer.value.upper()}...[/cyan]")
        dp = DataProvider()
        try:
            # Load data: prefer --data if provided, else fetch by date range
            if data:
                console.print(f"[cyan]Loading cached data from {data}...[/cyan]")
                if data.endswith(".parquet"):
                    df = pd.read_parquet(data)
                elif data.endswith(".pkl") or data.endswith(".pickle"):
                    df = pd.read_pickle(data)
                else:
                    raise ValueError("Unsupported --data file type; use .parquet or .pkl")
                # Normalize index if needed
                if df.index.name == 'ts':
                    df.index = pd.to_datetime(df.index, utc=True)
            else:
                df = await dp.fetch(ticker, tf, from_date, to_date)
            if len(df) == 0:
                raise RuntimeError("No data fetched")

            # Build features with defaults (will be overridden by optimizer seeds)
            seed_params = SellerParams()
            feats = build_features(df, seed_params, tf)

            # Create optimizer (pass initial_population_file when provided)
            kwargs = {}
            if init_from:
                kwargs["initial_population_file"] = init_from

            opt = create_optimizer(optimizer_type=optimizer.value, **kwargs)

            # Initialize from defaults; ADAM môže načítať seed z init_from vo vnútri
            opt.initialize(seed_seller_params=SellerParams(), seed_backtest_params=BacktestParams(), timeframe=tf)

            # Fitness config (balanced by default)
            fitness_cfg = FitnessConfig.get_preset_config("balanced") if hasattr(FitnessConfig, "get_preset_config") else FitnessConfig()

            best_fitness = None
            for i in range(max(1, generations)):
                res = opt.step(feats, tf, fitness_cfg)
                best_fitness = res.fitness
                console.print(
                    f"[green]✓ Step {i+1}/{generations}[/green] "
                    f"fitness={res.fitness:.4f} trades={res.metrics.get('n', 0)} "
                    f"win={res.metrics.get('win_rate', 0.0):.1%} avgR={res.metrics.get('avg_R', 0.0):.2f}"
                )

            sp, bp, fit = opt.get_best_params()
            if sp and bp:
                console.print("\n[bold]Best parameters:[/bold]")
                console.print(f"fitness={fit:.4f}")
                console.print(f"seller_params={sp}")
                console.print(f"backtest_params={bp}")
            else:
                console.print("[yellow]No best params available[/yellow]")

        except Exception as e:
            console.print(f"[red]✗ Optimization error: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            await dp.close()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
