#!/usr/bin/env python3
import asyncio

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


app = typer.Typer(help="ADA Seller-Exhaustion Backtesting CLI")
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
    init_from: str = typer.Option("", "--init-from", "-i", help="Path to population JSON to initialize optimizer"),
    generations: int = typer.Option(10, "--generations", "-g", help="Number of GA generations to run"),
    data: str = typer.Option("", "--data", help="Path to cached OHLCV DataFrame (parquet/pickle) to use instead of fetching; e.g. .data/X_ADAUSD_2025-09-14_2025-10-14_15minute.parquet"),
):
    """Run genetic algorithm optimization headlessly (CLI)."""

    async def _run():
        console.print(f"[cyan]Optimizing {ticker} on {tf.value} (evolutionary GA)...[/cyan]")
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

            # Optimizer needs raw OHLCV (builds features per individual)
            seed_params = SellerParams()

            kwargs = {}
            if init_from:
                kwargs["initial_population_file"] = init_from

            opt = create_optimizer(optimizer_type="evolutionary", **kwargs)

            opt.initialize(seed_seller_params=SellerParams(), seed_backtest_params=BacktestParams(), timeframe=tf)

            fitness_cfg = FitnessConfig.get_preset_config("balanced") if hasattr(FitnessConfig, "get_preset_config") else FitnessConfig()

            best_fitness = None
            for i in range(max(1, generations)):
                res = opt.step(df, tf, fitness_cfg)
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


@app.command("compare-strategies")
def compare_strategies(
    data: str = typer.Option(..., help="Path to cached OHLCV parquet/pkl"),
    tf: Timeframe = typer.Option("60m", "--tf", case_sensitive=False, help="Timeframe"),
    walk_forward: bool = typer.Option(True, help="Run walk-forward on best strategy"),
):
    """Compare all registered strategies on the same data (profit-focused)."""
    from backtest.compare import compare_all, walk_forward_compare
    from rich.table import Table

    if data.endswith(".parquet"):
        df = pd.read_parquet(data)
    elif data.endswith((".pkl", ".pickle")):
        df = pd.read_pickle(data)
    else:
        raise typer.BadParameter("Use .parquet or .pkl")

    if df.index.name == "ts":
        df.index = pd.to_datetime(df.index, utc=True)

    console.print(f"[cyan]Comparing strategies on {len(df)} bars ({tf.value})...[/cyan]")
    bt = BacktestParams()
    results = compare_all(df, tf, bt)

    table = Table(title="Strategy Comparison (sorted by profit score)")
    for col in ("name", "signals", "trades", "win_rate", "total_pnl", "expectancy_r", "cagr_pct", "profit_score"):
        table.add_column(col)
    for _, row in results.iterrows():
        table.add_row(
            str(row["name"]),
            str(int(row["signals"])),
            str(int(row["trades"])),
            f"{row['win_rate']:.0%}",
            f"{row['total_pnl']:+.4f}",
            f"{row['expectancy_r']:.3f}",
            f"{row['cagr_pct']:+.1f}%",
            f"{row['profit_score']:.3f}",
        )
    console.print(table)

    best_id = results.iloc[0]["strategy_id"]
    console.print(f"\n[bold green]Best:[/bold green] {results.iloc[0]['name']} (score={results.iloc[0]['profit_score']:.3f})")

    if walk_forward:
        console.print(f"\n[cyan]Walk-forward validation: {best_id}[/cyan]")
        for fold in walk_forward_compare(df, best_id, tf, bt):
            console.print(
                f"  Fold {fold['fold']}: trades={fold['trades']} pnl={fold['total_pnl']:+.4f} "
                f"wr={fold['win_rate']:.0%} CAGR={fold['cagr_pct']:+.1f}%"
            )


@app.command("fetch-15m")
def fetch_15m(
    ticker: str = typer.Option("X:ADAUSD"),
    from_date: str = typer.Option("2018-01-01", "--from"),
    to_date: str = typer.Option("2026-06-15", "--to"),
    force: bool = typer.Option(False, help="Force re-download"),
):
    """Download and cache 15m bars (extended range for research)."""

    async def _run():
        dp = DataProvider()
        try:
            df = await dp.fetch(ticker, Timeframe.m15, from_date, to_date, force_download=force)
            console.print(f"[green]✓ Cached {len(df)} bars[/green] | {df.index[0]} → {df.index[-1]}")
        finally:
            await dp.close()

    asyncio.run(_run())


@app.command("optimize-strategies")
def optimize_strategies(
    data: str = typer.Option(..., help="15m parquet path"),
    tf: Timeframe = typer.Option("15m", "--tf"),
    generations: int = typer.Option(15, "-g"),
    population: int = typer.Option(12, "-p"),
    train_ratio: float = typer.Option(0.7, help="Train fraction (rest = OOS holdout)"),
    round: int = typer.Option(2, help="Strategy round: 1=original trio, 2=new trio"),
):
    """GA optimize each strategy on TRAIN only; validate on OOS; save configs."""
    from backtest.strategy_ga import (
        get_optimizable_strategies,
        run_strategy_ga,
        split_train_oos,
        evaluate_oos,
        evaluate_strategy_individual,
    )
    from exec.paper_trader import save_optimized_config
    from core.models import FitnessConfig

    strategies = get_optimizable_strategies(round)
    df = _load_dataframe(data)
    train, oos = split_train_oos(df, train_ratio)
    console.print(f"[cyan]Round {round} | Train: {len(train)} bars | OOS: {len(oos)} bars[/cyan]")
    fitness = FitnessConfig.get_preset_config("profit_focused")

    table = Table(title=f"Per-Strategy GA round {round} (train → OOS)")
    table.add_column("strategy")
    table.add_column("train_n")
    table.add_column("train_pnl")
    table.add_column("oos_n")
    table.add_column("oos_pnl")
    table.add_column("oos_cagr")

    for sid in strategies:
        console.print(f"\n[bold]Optimizing {sid}...[/bold]")
        pop = run_strategy_ga(sid, train, tf, generations=generations, population_size=population, fitness_config=fitness)
        best = pop.best_ever or pop.get_best()
        _, train_m = evaluate_strategy_individual(sid, best, train, tf, fitness)
        oos_m = evaluate_oos(sid, best, oos, tf)
        save_optimized_config(sid, best, tf, train_m, oos_m)
        table.add_row(
            sid,
            str(train_m.get("n", 0)),
            f"{train_m.get('total_pnl', 0):+.4f}",
            str(oos_m.get("n", 0)),
            f"{oos_m.get('total_pnl', 0):+.4f}",
            f"{oos_m.get('cagr_pct', 0):+.1f}%",
        )
    console.print(table)
    console.print("[green]✓ Saved to strategies_optimized/[/green]")


@app.command("tune-depth-charge")
def tune_depth_charge_cmd(
    data: str = typer.Option(..., help="15m parquet path"),
    tf: Timeframe = typer.Option("15m", "--tf"),
    train_ratio: float = typer.Option(0.7),
):
    """Tune Depth Charge tri-channel engine (OOS-weighted grid, no GA)."""
    from backtest.depth_tune import tune_depth_charge
    from backtest.strategy_ga import split_train_oos

    df = _load_dataframe(data)
    train, oos = split_train_oos(df, train_ratio)
    console.print(f"[cyan]Depth Charge tune | train={len(train)} oos={len(oos)}[/cyan]")
    best = tune_depth_charge(train, oos, tf)
    dp = best["params"]
    console.print(f"  conviction>={dp.min_conviction:.2f} floor={dp.floor_pct:.2f} vol_pct={dp.vol_pctile_min:.2f}")
    console.print(f"  Train: n={best['train'].get('n',0)} pnl={best['train'].get('total_pnl',0):+.4f}")
    console.print(f"  OOS:   n={best['oos'].get('n',0)} pnl={best['oos'].get('total_pnl',0):+.4f} score={best['oos_score']:.3f}")
    console.print("[green]✓ strategies_optimized/depth_charge_params.json[/green]")


@app.command("tune-fusion")
def tune_fusion(
    data: str = typer.Option(..., help="15m parquet path"),
    tf: Timeframe = typer.Option("15m", "--tf"),
    train_ratio: float = typer.Option(0.7),
):
    """Grid-tune Fusion V2 + Panic Floor (fast, OOS-weighted, no GA)."""
    from backtest.fusion_tune import tune_fusion_v2, tune_panic_floor
    from backtest.strategy_ga import split_train_oos
    from strategy.fusion_v2 import build_features as build_fv2, load_fusion_params
    from strategy.panic_floor import build_features as build_panic
    from backtest.engine import run_backtest
    from backtest.profit import simulate_account

    df = _load_dataframe(data)
    train, oos = split_train_oos(df, train_ratio)
    console.print(f"[cyan]Tuning on train={len(train)} / oos={len(oos)} bars[/cyan]")

    console.print("\n[bold]Fusion V2 grid search...[/bold]")
    fv2 = tune_fusion_v2(train, oos, tf)
    fp, bt = load_fusion_params()
    console.print(f"  Best: agreement={fp.min_agreement} conf={fp.min_confidence:.2f} "
                  f"oversold={fp.require_oversold} floor={fp.oversold_pct:.3f}")
    console.print(f"  Train: n={fv2['train'].get('n',0)} pnl={fv2['train'].get('total_pnl',0):+.4f}")
    console.print(f"  OOS:   n={fv2['oos'].get('n',0)} pnl={fv2['oos'].get('total_pnl',0):+.4f} "
                  f"score={fv2['oos_score']:.3f}")

    console.print("\n[bold]Panic Floor grid search...[/bold]")
    pf = tune_panic_floor(train, oos, tf)
    pp = pf["params"]
    console.print(f"  Best: floor={pp.floor_pct:.3f} vol_pctile={pp.vol_pctile_min:.2f} rsi<{pp.rsi_max}")
    console.print(f"  Train: n={pf['train'].get('n',0)} pnl={pf['train'].get('total_pnl',0):+.4f}")
    console.print(f"  OOS:   n={pf['oos'].get('n',0)} pnl={pf['oos'].get('total_pnl',0):+.4f}")

    table = Table(title="Elite stack OOS comparison")
    table.add_column("strategy")
    table.add_column("trades")
    table.add_column("pnl")
    table.add_column("win_rate")
    table.add_column("profit_factor")
    for sid, metrics in [("fusion_v2", fv2["oos"]), ("panic_floor", pf["oos"])]:
        table.add_row(
            sid,
            str(metrics.get("n", 0)),
            f"{metrics.get('total_pnl', 0):+.4f}",
            f"{metrics.get('win_rate', 0):.0%}",
            f"{metrics.get('profit_factor', 0):.2f}",
        )
    console.print(table)
    console.print("[green]✓ Fusion V2 params → strategies_optimized/fusion_v2_params.json[/green]")


@app.command("regime-update")
def regime_update(
    data: str = typer.Option(..., help="OHLCV parquet for weekly summary"),
    tf: Timeframe = typer.Option("15m", "--tf"),
    force: bool = typer.Option(False),
):
    """Update weekly LLM/deterministic regime gate (REGIME_LLM_MODEL, default gpt-4o-mini)."""
    from strategy.regime_weekly import update_weekly_regime

    df = _load_dataframe(data)
    entry = update_weekly_regime(df, tf, force=force)
    console.print(f"[green]Week {entry['week']}[/green] label={entry['label']} score={entry['score']:.3f}")
    console.print(f"  min_regime_score={entry['min_regime_score']} | {entry['reason'][:120]}")


@app.command("paper-forward")
def paper_forward(
    data: str = typer.Option(..., help="OHLCV parquet (uses last N days as forward test)"),
    strategy: str = typer.Option("depth_charge", help="Strategy id"),
    tf: Timeframe = typer.Option("15m", "--tf"),
    days: int = typer.Option(30, help="Forward window in days"),
    use_regime: bool = typer.Option(True, help="Apply weekly regime gate (GA strategies only)"),
):
    """Paper forward test using frozen config (no re-fit)."""
    from exec.paper_trader import run_paper_forward

    df = _load_dataframe(data)
    result = run_paper_forward(df, strategy, tf, min_days=days, use_regime_gate=use_regime)
    m = result["metrics"]
    a = result["account"]
    console.print(f"[bold]{strategy}[/bold] forward {days}d (+{result.get('warmup_days', 0)}d warmup): {result['n_trades']} trades")
    console.print(f"  PnL={m.get('total_pnl', 0):+.4f} WR={m.get('win_rate', 0):.0%} CAGR={a.get('cagr_pct', 0):+.1f}%")
    console.print(f"  Logged → .data/paper_trades.jsonl")


@app.command("paper-compare")
def paper_compare(
    data: str = typer.Option(..., help="OHLCV parquet"),
    strategies: str = typer.Option("depth_charge,fusion_v2", help="Comma-separated strategy ids"),
    tf: Timeframe = typer.Option("15m", "--tf"),
    days: int = typer.Option(60, help="Forward window in days"),
):
    """Compare paper-forward results across strategies (no re-tune)."""
    from exec.paper_trader import run_paper_forward

    df = _load_dataframe(data)
    table = Table(title=f"Paper Forward {days}d (frozen configs, no re-tune)")
    table.add_column("strategy")
    table.add_column("trades")
    table.add_column("pnl")
    table.add_column("win_rate")
    table.add_column("expectancy_r")
    table.add_column("cagr")

    for sid in [s.strip() for s in strategies.split(",") if s.strip()]:
        try:
            r = run_paper_forward(df, sid, tf, min_days=days, use_regime_gate=False)
            m = r["metrics"]
            a = r["account"]
            table.add_row(
                sid,
                str(r["n_trades"]),
                f"{m.get('total_pnl', 0):+.4f}",
                f"{m.get('win_rate', 0):.0%}",
                f"{m.get('expectancy_r', a.get('expectancy_r', 0)):.3f}",
                f"{a.get('cagr_pct', 0):+.1f}%",
            )
        except Exception as e:
            table.add_row(sid, "—", "error", str(e)[:40], "—", "—")
    console.print(table)


@app.command("walk-forward")
def walk_forward_cmd(
    data: str = typer.Option(..., help="OHLCV parquet"),
    strategies: str = typer.Option("mean_reversion,depth_charge", help="Comma-separated ids"),
    tf: Timeframe = typer.Option("15m", "--tf"),
    test_days: int = typer.Option(60, help="OOS window per fold (days)"),
    step_days: int = typer.Option(60, help="Step between folds (days); =test_days → non-overlapping"),
    warmup_days: int = typer.Option(14, help="Indicator warmup before each test window"),
    last_days: int = typer.Option(0, help="Use only trailing N days (0 = full file)"),
    auto_sanity: bool = typer.Option(False, "--auto-sanity", help="Run params sanity audit before walk-forward"),
    sanity_normalize: bool = typer.Option(True, "--sanity-normalize/--no-sanity-normalize", help="When auto-sanity is enabled, normalize configs"),
    export: str = typer.Option("", help="Optional JSON export path"),
):
    """
    Rolling walk-forward on frozen configs (no re-tune).

    Each fold: warmup + test context for features, metrics only on test window.
    """
    from backtest.walk_forward import walk_forward_report, slice_last_days
    from backtest.param_sanity import list_config_files, analyze_file, normalize_file
    import json

    if auto_sanity:
        sanity_paths = list_config_files(tf)
        if sanity_paths:
            bad = 0
            changed = 0
            for pth in sanity_paths:
                res = normalize_file(pth, tf) if sanity_normalize else analyze_file(pth, tf)
                if res is None:
                    continue
                if not res.valid:
                    bad += 1
                if res.changed:
                    changed += 1
            console.print(
                f"[cyan]Auto-sanity ({tf.value})[/cyan] files={len(sanity_paths)} "
                f"warnings={bad} changed={changed}"
            )
        else:
            console.print(f"[yellow]Auto-sanity: no optimized config files for {tf.value}[/yellow]")

    df = _load_dataframe(data)
    if last_days > 0:
        df = slice_last_days(df, last_days, tf)

    ids = [s.strip() for s in strategies.split(",") if s.strip()]
    report = walk_forward_report(df, ids, tf, test_days, step_days, warmup_days)
    p = report["params"]

    console.print(
        f"[cyan]Walk-forward[/cyan] {p['range']} | {p['bars']} bars | "
        f"test={test_days}d step={step_days}d warmup={warmup_days}d | {tf.value}"
    )

    ranked = sorted(
        report["summaries"].items(),
        key=lambda kv: kv[1].get("robust_profit_score", -1e9),
        reverse=True,
    )

    rank_table = Table(title="Strategy ranking (primary sort: robust_score)")
    rank_table.add_column("rank")
    rank_table.add_column("strategy")
    rank_table.add_column("robust_score")
    rank_table.add_column("sum_pnl")
    rank_table.add_column("positive_folds")
    rank_table.add_column("trades")
    for idx, (sid, summary) in enumerate(ranked, start=1):
        rank_table.add_row(
            str(idx),
            sid,
            f"{summary['robust_profit_score']:+.4f}",
            f"{summary['sum_pnl']:+.4f}",
            f"{summary['positive_folds']}/{summary['folds']}",
            str(summary["total_trades"]),
        )
    console.print(rank_table)

    for sid, summary in ranked:
        console.print(f"\n[bold]{sid}[/bold] — {summary['folds']} folds")
        console.print(
            f"  trades={summary['total_trades']} sum_pnl={summary['sum_pnl']:+.4f} "
            f"median_pnl={summary['median_pnl']:+.4f} "
            f"positive_folds={summary['positive_folds']}/{summary['folds']}"
        )
        console.print(
            f"  median_wr={summary['median_win_rate']:.0%} "
            f"median_expR={summary['median_expectancy_r']:.3f}"
        )
        console.print(f"  robust_score={summary['robust_profit_score']:+.4f}")

    table = Table(title="Per-fold detail")
    table.add_column("strategy")
    table.add_column("fold")
    table.add_column("period")
    table.add_column("trades")
    table.add_column("pnl")
    table.add_column("wr")
    table.add_column("expR")

    for row in report["folds"]:
        period = row["test_start"][:10] + "…" + row["test_end"][:10]
        table.add_row(
            row["strategy_id"],
            str(row["fold"]),
            period,
            str(row["trades"]),
            f"{row['total_pnl']:+.4f}",
            f"{row['win_rate']:.0%}",
            f"{row['expectancy_r']:.3f}",
        )
    console.print(table)

    # Auto-export top candidate for paper-forward
    if ranked:
        import json
        from pathlib import Path

        top_sid, top_summary = ranked[0]
        top_payload = {
            "strategy_id": top_sid,
            "timeframe": tf.value,
            "robust_profit_score": top_summary["robust_profit_score"],
            "sum_pnl": top_summary["sum_pnl"],
            "positive_folds": top_summary["positive_folds"],
            "folds": top_summary["folds"],
            "total_trades": top_summary["total_trades"],
            "source_data": data,
            "paper_forward_command": (
                f"poetry run python cli.py paper-forward "
                f"--data {data} --strategy {top_sid} --tf {tf.value} --days {test_days}"
            ),
        }
        Path(".data").mkdir(exist_ok=True)
        top_path = Path(f".data/top_candidate_{tf.value}.json")
        with top_path.open("w") as f:
            json.dump(top_payload, f, indent=2)
        console.print(
            f"[green]✓ Auto-exported top candidate:[/green] {top_sid} "
            f"(robust_score={top_summary['robust_profit_score']:+.4f}) → {top_path}"
        )

    if export:
        with open(export, "w") as f:
            json.dump(report, f, indent=2)
        console.print(f"[green]✓ Exported → {export}[/green]")


@app.command("fetch-bars")
def fetch_bars(
    ticker: str = typer.Option("X:ADAUSD"),
    tf: Timeframe = typer.Option("15m", "--tf"),
    from_date: str = typer.Option("2025-06-15", "--from"),
    to_date: str = typer.Option("2026-06-15", "--to"),
    force: bool = typer.Option(False),
):
    """Download and cache bars for any timeframe (e.g. 5m / 15m research slices)."""

    async def _run():
        dp = DataProvider()
        try:
            df = await dp.fetch(ticker, tf, from_date, to_date, force_download=force)
            console.print(f"[green]✓ Cached {len(df)} bars[/green] | {df.index[0]} → {df.index[-1]}")
        finally:
            await dp.close()

    asyncio.run(_run())


@app.command("params-sanity")
def params_sanity(
    tf: Timeframe = typer.Option("15m", "--tf"),
    normalize: bool = typer.Option(False, help="Normalize proxy windows to timeframe defaults"),
):
    """
    Audit (and optionally normalize) optimized strategy configs for timeframe consistency.
    """
    from backtest.param_sanity import list_config_files, analyze_file, normalize_file

    paths = list_config_files(tf)
    if not paths:
        console.print(f"[yellow]No optimized config files for {tf.value}[/yellow]")
        return

    table = Table(title=f"Parameter Sanity ({tf.value})")
    table.add_column("strategy")
    table.add_column("file")
    table.add_column("status")
    table.add_column("details")

    bad = 0
    changed = 0
    for p in paths:
        res = normalize_file(p, tf) if normalize else analyze_file(p, tf)
        if res is None:
            continue
        status = "OK" if res.valid else "WARN"
        if not res.valid:
            bad += 1
        if res.changed:
            changed += 1
            status = "CHANGED"
        details = "; ".join(w.replace("⚠ ", "") for w in res.warnings[:2]) if res.warnings else "-"
        table.add_row(res.strategy_id, p.name, status, details[:120])

    console.print(table)
    if normalize:
        console.print(f"[green]Normalized files:[/green] {changed}")
    console.print(f"[cyan]Warnings:[/cyan] {bad}")


@app.command("bootstrap-configs")
def bootstrap_configs_cmd(
    source_tf: Timeframe = typer.Option("15m", "--from-tf"),
    target_tf: Timeframe = typer.Option("5m", "--to-tf"),
    normalize: bool = typer.Option(True, help="Run params-sanity normalize on target timeframe after bootstrap"),
):
    """
    Bootstrap optimized configs from one timeframe to another.

    Core windows are time-scaled; thresholds are preserved.
    """
    from backtest.param_sanity import bootstrap_configs

    created = bootstrap_configs(source_tf, target_tf)
    if not created:
        console.print(f"[yellow]No source configs for {source_tf.value}[/yellow]")
        return

    table = Table(title=f"Bootstrapped configs {source_tf.value} → {target_tf.value}")
    table.add_column("file")
    for p in created:
        table.add_row(p.name)
    console.print(table)
    console.print(f"[green]Created:[/green] {len(created)}")

    if normalize:
        params_sanity(tf=target_tf, normalize=True)


def _load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith((".pkl", ".pickle")):
        df = pd.read_pickle(path)
    else:
        raise typer.BadParameter("Use .parquet or .pkl")
    if df.index.name == "ts":
        df.index = pd.to_datetime(df.index, utc=True)
    return df


if __name__ == "__main__":
    app()
