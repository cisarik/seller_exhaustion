"""
Spectre-based trading runner for the Seller Exhaustion strategy.

This module provides an experimental alternative to the event-driven CPU backtester.
It uses Spectre's CustomAlgorithm + blotter API to simulate trading with the same
entry/exit logic as the CPU engine, targeting parity in results.

Key design decisions:
1. Single-asset focus (X:ADAUSD) with multi-asset extension possible later
2. Fibonacci targets injected as external SeriesDataFactor for entry-signal alignment
3. CustomAlgorithm manages per-position state (entry_ts, entry, target, bars_held)
4. Commission/slippage mapping: approximate per-side percentage from basis points
5. Trade output format matches current engine: columns entry_ts, exit_ts, entry, exit, pnl, R, reason, bars_held
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
import threading
from core.logging_utils import get_logger

logger = get_logger(__name__)

# Optional Spectre imports
try:
    from spectre import factors, trading
    from spectre.data import MemoryLoader
    _SPECTRE_AVAILABLE = True
except Exception as e:
    _SPECTRE_AVAILABLE = False
    logger.warning("Spectre not available: %s", e)

from core.models import BacktestParams, Timeframe
from strategy.seller_exhaustion import SellerParams
from indicators.fibonacci import add_fib_levels_to_df


@dataclass
class PositionState:
    """State tracking for an open position."""
    entry_ts: pd.Timestamp
    entry_price: float
    target_price: float
    risk: float  # For R-multiple calculation
    bars_held: int = 0
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None


def df_to_memory_loader(
    df: pd.DataFrame,
    asset: str = "X:ADAUSD"
) -> tuple[MemoryLoader, pd.DataFrame]:
    """
    Convert a simple DatetimeIndex DataFrame to Spectre MemoryLoader format.
    
    Args:
        df: DataFrame with DatetimeIndex, columns [open, high, low, close, volume]
        asset: Asset identifier (default: X:ADAUSD)
    
    Returns:
        (MemoryLoader, spectre_df): Loader and the transformed MultiIndex DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex")
    
    # Ensure UTC timezone
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    
    # Build Spectre-format DataFrame: MultiIndex [date, asset]
    base = df[["open", "high", "low", "close", "volume"]].copy()
    base.index = idx
    # Ensure all numeric columns are float (Spectre requires float32 for GPU)
    base["volume"] = base["volume"].astype(float)
    base["ex-dividend"] = 0.0
    base["split_ratio"] = 1.0
    base["asset"] = asset
    base["date"] = base.index
    
    spectre_df = base.set_index(["date", "asset"]).sort_index()
    
    # Create MemoryLoader with explicit OHLCV specification (including volume)
    loader = MemoryLoader(spectre_df, ohlcv=('open', 'high', 'low', 'close', 'volume'))
    return loader, spectre_df


def make_fib_lookup_dict(
    feats_df: pd.DataFrame,
    fib_col: str = "fib_0618"
) -> dict:
    """
    Create a lookup dict from timestamp (string) to Fibonacci target price.
    
    Used to quickly access pre-computed targets during rebalance callback.
    
    Args:
        feats_df: DataFrame with Fibonacci columns and DatetimeIndex
        fib_col: Column name for target level (e.g., "fib_0618")
    
    Returns:
        Dict mapping timestamp string -> target price (or NaN if not applicable)
    """
    lookup = {}
    if fib_col in feats_df.columns:
        for idx, val in feats_df[fib_col].items():
            # Convert timestamp to string for consistent lookup
            ts_str = str(idx)
            lookup[ts_str] = float(val) if not pd.isna(val) else np.nan
    return lookup


def build_seller_exhaustion_factors(
    engine: "factors.FactorEngine",
    seller_params: SellerParams,
    tf: Timeframe
) -> Dict[str, "factors.Factor"]:
    """
    Build Spectre factors for the Seller Exhaustion strategy.
    
    Returns a dict of registered factor names -> factor objects for reference.
    All factors are also added to the engine.
    
    Args:
        engine: Spectre FactorEngine
        seller_params: Strategy parameters
        tf: Timeframe (for bar conversion if needed)
    
    Returns:
        Dict mapping factor names to factor objects
    """
    # Resolve bar counts (for now, assume seller_params already has bars, not minutes)
    ema_fast_bars = seller_params.ema_fast
    ema_slow_bars = seller_params.ema_slow
    z_window_bars = seller_params.z_window
    atr_window_bars = seller_params.atr_window
    
    # Base OHLCV factors
    close = factors.OHLCV.close
    high = factors.OHLCV.high
    low = factors.OHLCV.low
    volume = factors.OHLCV.volume
    
    # Indicators
    ema_f = factors.EMA(span=int(ema_fast_bars), inputs=[close])
    ema_s = factors.EMA(span=int(ema_slow_bars), inputs=[close])
    downtrend = ema_f < ema_s
    
    tr = factors.TRANGE(win=2)
    atr_fac = factors.MA(win=int(atr_window_bars), inputs=[tr])
    
    # Volume z-score
    vol_ma = factors.MA(win=int(z_window_bars), inputs=[volume])
    vol_std = factors.STDDEV(win=int(z_window_bars), inputs=[volume])
    vol_z = (volume - vol_ma) / vol_std
    
    # True Range z-score (approximate using ATR * window)
    tr_proxy = atr_fac * float(atr_window_bars)
    tr_ma = factors.MA(win=int(z_window_bars), inputs=[tr_proxy])
    tr_std = factors.STDDEV(win=int(z_window_bars), inputs=[tr_proxy])
    tr_z = (tr_proxy - tr_ma) / tr_std
    
    # Close location in candle
    span = (high - low)
    cloc = (close - low) / span
    
    # Exhaustion signal
    exhaustion = (
        downtrend & 
        (vol_z > seller_params.vol_z) & 
        (tr_z > seller_params.tr_z) & 
        (cloc > seller_params.cloc_min)
    )
    
    # Register factors
    factor_dict = {}
    for name, factor in [
        ("ema_f", ema_f),
        ("ema_s", ema_s),
        ("downtrend", downtrend),
        ("atr", atr_fac),
        ("vol_z", vol_z),
        ("tr_z", tr_z),
        ("cloc", cloc),
        ("exhaustion", exhaustion),
    ]:
        engine.add(factor, name)
        factor_dict[name] = factor
    
    return factor_dict


class SellerExhaustionAlg(trading.CustomAlgorithm):
    """
    Seller Exhaustion trading algorithm for Spectre.
    
    Entry: Long position when exhaustion signal is True at bar t, fills at t+1 open.
    Exit: Position closes when Fibonacci target price is hit intrabar.
    Fees/slippage: Applied as per-side commission based on basis points.
    """
    
    # Class variable to store the last instance (for retrieval after Spectre run)
    _last_instance = None
    _instance_lock = threading.Lock()
    
    def initialize(self):
        """Initialize the algorithm: build factors, config blotter, schedule rebalance."""
        # Store this instance for retrieval after Spectre run
        with SellerExhaustionAlg._instance_lock:
            SellerExhaustionAlg._last_instance = self
        
        # Get factor engine and build factors
        engine = self.get_factor_engine()
        
        # Optional GPU acceleration (if enabled)
        if getattr(self, 'use_cuda', False):
            try:
                engine.to_cuda(enable_stream=True, gpu_id=0)
                logger.info("Spectre engine moved to CUDA with streaming enabled")
            except Exception as e:
                logger.warning("Spectre CUDA requested but unavailable: %s", e)
        
        # Build entry signal factors
        self.factors = build_seller_exhaustion_factors(
            engine,
            self.seller_params,
            self.timeframe
        )
        
        # Create Fibonacci target lookup dict (pre-computed on CPU)
        # Avoids Spectre factor injection issues with DatetimeIndex
        if hasattr(self, 'fib_target_dict') and self.fib_target_dict is not None:
            self.fib_lookup = self.fib_target_dict
            logger.debug("Fibonacci target lookup available: %d entries", len(self.fib_lookup))
        else:
            self.fib_lookup = {}
        
        # Configure blotter: commission from basis points
        # Approximate: fee_bp + slippage_bp total per round trip
        # Apply half to each side (entry and exit)
        total_bp = self.backtest_params.fee_bp + self.backtest_params.slippage_bp
        side_pct = total_bp / 20000.0  # Convert bp to percentage, split sides
        
        self.blotter.set_commission(
            percentage=side_pct,
            per_share=0.0,
            minimum=0.0
        )
        
        # Set capital base
        self.blotter.capital_base = 1.0  # Normalized capital for parity
        
        # Schedule rebalance once per bar close
        # Use a small negative offset to trigger just before market close
        self.schedule_rebalance(
            trading.event.MarketClose(
                callback=self.rebalance,
                offset_ns=-10000
            )
        )
        
        # Internal state
        self.open_position: Optional[PositionState] = None
        self.trades = []
        self.asset = getattr(self, 'asset', "X:ADAUSD")
    
    def rebalance(self, data: pd.DataFrame, history: pd.DataFrame):
        """
        Rebalance callback: entry/exit logic per bar.
        
        data: Current bar(s) for all assets
        history: Historical data (may be empty on first call)
        """
        # Single-asset case
        if len(data.index) == 0 or len(data.index.get_level_values("asset")) == 0:
            return
        
        asset = data.index.get_level_values("asset")[0]
        try:
            row = data.xs(asset, level="asset").iloc[0]
        except Exception:
            return
        
        now = row.name  # datetime
        
        # Check for NEW ENTRY (at t+1 open after signal at t)
        # We look at the PREVIOUS bar's exhaustion signal
        if self.open_position is None:
            # Get exhaustion signal value for current bar
            try:
                exhaustion = bool(row.get("exhaustion", False))
            except Exception:
                exhaustion = False
            
            if exhaustion:
                # Entry at current bar's open (simulates t+1 fill)
                entry_price = float(row.get("open", np.nan))
                
                # Get Fibonacci target from pre-computed lookup dict
                target_price = np.inf  # Default: no target (position never closes)
                try:
                    ts_str = str(now)
                    if ts_str in self.fib_lookup and not pd.isna(self.fib_lookup[ts_str]):
                        target_price = float(self.fib_lookup[ts_str])
                except Exception as e:
                    logger.debug("Fib lookup error: %s", e)
                
                if not pd.isna(entry_price):
                    # Simple 1% risk assumption for R-multiple (matching CPU engine)
                    risk = entry_price * 0.01
                    
                    self.open_position = PositionState(
                        entry_ts=now,
                        entry_price=entry_price,
                        target_price=target_price,
                        risk=risk,
                        bars_held=0
                    )
                    
                    # Order: go long at 100% weight
                    self.blotter.batch_order_target_percent({asset: 1.0})
                    logger.debug(
                        "Entry at %s | price=%.6f | target=%.6f | risk=%.6f",
                        now, entry_price, target_price or np.inf, risk
                    )
                    return
        
        # MANAGE OPEN POSITION
        if self.open_position is not None:
            self.open_position.bars_held += 1
            
            hi = float(row.get("high", np.nan))
            lo = float(row.get("low", np.nan))
            close_price = float(row.get("close", np.nan))
            
            exit_price = None
            reason = None
            
            # Check Fibonacci exit: target hit intrabar
            if not pd.isna(self.open_position.target_price) and hi >= self.open_position.target_price:
                exit_price = self.open_position.target_price
                reason = f"FIB_{self.backtest_params.fib_target_level * 100:.1f}"
            
            # If exiting, record trade and close position
            if exit_price is not None and not pd.isna(exit_price):
                # Compute fees/slippage (Blotter will apply commission separately)
                fee = (self.open_position.entry_price + exit_price) * (
                    (self.backtest_params.fee_bp + self.backtest_params.slippage_bp) / 10000.0
                )
                pnl = exit_price - self.open_position.entry_price - fee
                R = pnl / self.open_position.risk if self.open_position.risk != 0 else 0.0
                
                trade_record = {
                    "entry_ts": str(self.open_position.entry_ts),
                    "exit_ts": str(now),
                    "entry": self.open_position.entry_price,
                    "exit": exit_price,
                    "pnl": pnl,
                    "R": R,
                    "reason": reason,
                    "bars_held": self.open_position.bars_held,
                }
                self.trades.append(trade_record)
                
                logger.debug(
                    "Exit at %s | price=%.6f | pnl=%.8f | R=%.2f | reason=%s",
                    now, exit_price, pnl, R, reason
                )
                
                # Order: flatten position
                self.blotter.batch_order_target_percent({asset: 0.0})
                self.open_position = None
    
    def terminate(self, records: pd.DataFrame):
        """Called at end of backtest. Optionally plot or log results."""
        logger.info(
            "Spectre algorithm terminated | trades=%d | final_position=%s",
            len(self.trades),
            "open" if self.open_position is not None else "closed"
        )


def run_spectre_trading(
    df: pd.DataFrame,
    seller_params: SellerParams,
    backtest_params: BacktestParams,
    timeframe: Timeframe = Timeframe.m15,
    use_cuda: bool = False,
    asset: str = "X:ADAUSD",
) -> Dict[str, Any]:
    """
    Run Spectre-based backtest for Seller Exhaustion strategy.
    
    Args:
        df: DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
        seller_params: Strategy parameters
        backtest_params: Backtest execution parameters
        timeframe: Bar timeframe
        use_cuda: Enable Spectre GPU if available
        asset: Asset identifier (default: X:ADAUSD)
    
    Returns:
        Dict with keys:
            - "trades": DataFrame with columns [entry_ts, exit_ts, entry, exit, pnl, R, reason, bars_held]
            - "metrics": Dict with performance metrics (n, win_rate, avg_R, total_pnl, max_dd, sharpe)
    """
    if not _SPECTRE_AVAILABLE:
        raise RuntimeError("Spectre is not available. Install spectre package.")
    
    try:
        t0 = time.perf_counter()
        
        # Step 1: Build MemoryLoader from DataFrame
        loader, spectre_df = df_to_memory_loader(df, asset=asset)
        logger.info(
            "Spectre trading run | bars=%d | asset=%s | tf=%s",
            len(df),
            asset,
            timeframe.value
        )
        
        # Step 2: Create FactorEngine
        engine = factors.FactorEngine(loader)
        
        # Step 3: Determine warmup rows (same as feature builder)
        def _ema_win(span: int) -> int:
            return int(4.5 * (int(span) + 1))
        
        warmup = max(
            _ema_win(seller_params.ema_fast),
            _ema_win(seller_params.ema_slow),
            seller_params.z_window,
            seller_params.atr_window,
        )
        
        # Precompute Fibonacci targets (CPU-side, pre-computed for fast lookup)
        feats = df.copy()
        feats = add_fib_levels_to_df(
            feats,
            signal_col="exhaustion",
            lookback=96,  # TODO: Make configurable
            lookahead=5,
        )
        
        # Extract target column based on fib_target_level
        fib_target_level = backtest_params.fib_target_level
        fib_col = f"fib_{int(fib_target_level * 1000):04d}"
        fib_target_dict = make_fib_lookup_dict(feats, fib_col)
        logger.debug("Built Fibonacci lookup: %d target prices", len(fib_target_dict))
        
        # Step 4: Configure algorithm by setting class attributes
        # Spectre will instantiate the algorithm with the blotter argument
        SellerExhaustionAlg.seller_params = seller_params
        SellerExhaustionAlg.backtest_params = backtest_params
        SellerExhaustionAlg.timeframe = timeframe
        SellerExhaustionAlg.use_cuda = use_cuda
        SellerExhaustionAlg.asset = asset
        SellerExhaustionAlg.fib_target_dict = fib_target_dict
        
        algo = SellerExhaustionAlg
        
        # Step 5: Run backtest
        date_index = spectre_df.index.get_level_values("date")
        if len(date_index) == 0:
            logger.warning("No data in MemoryLoader")
            return {
                "trades": pd.DataFrame(),
                "metrics": {
                    "n": 0,
                    "win_rate": 0.0,
                    "avg_R": 0.0,
                    "total_pnl": 0.0,
                    "max_dd": 0.0,
                    "sharpe": 0.0,
                }
            }
        
        # Use full date range - Spectre handles warmup internally
        start = date_index[0]
        end = date_index[-1]
        
        logger.debug(
            "Spectre run window | warmup=%d | start=%s | end=%s | total=%d",
            warmup, start, end, len(date_index)
        )
        
        trading.run_backtest(loader, algo, start, end)
        
        # Step 6: Extract and format trades
        # Retrieve the actual algorithm instance that Spectre created
        trades_list = []
        with SellerExhaustionAlg._instance_lock:
            instance = SellerExhaustionAlg._last_instance
            if instance is not None and hasattr(instance, 'trades'):
                trades_list = instance.trades
                logger.debug("Retrieved %d trades from Spectre algorithm instance", len(trades_list))
        
        if len(trades_list) == 0:
            trades_df = pd.DataFrame(
                columns=["entry_ts", "exit_ts", "entry", "exit", "pnl", "R", "reason", "bars_held"]
            )
        else:
            trades_df = pd.DataFrame(trades_list)
        
        # Step 7: Calculate metrics
        if len(trades_df) == 0:
            metrics = {
                "n": 0,
                "win_rate": 0.0,
                "avg_R": 0.0,
                "total_pnl": 0.0,
                "max_dd": 0.0,
                "sharpe": 0.0,
            }
        else:
            win_rate = float((trades_df["pnl"] > 0).mean())
            avg_R = float(trades_df["R"].mean())
            total_pnl = float(trades_df["pnl"].sum())
            
            # Drawdown
            cumulative = trades_df["pnl"].cumsum()
            running_max = cumulative.expanding().max()
            drawdown = cumulative - running_max
            max_dd = float(drawdown.min())
            
            # Sharpe approximation
            sharpe = float(
                trades_df["R"].mean() / trades_df["R"].std()
            ) if len(trades_df) > 1 and trades_df["R"].std() > 0 else 0.0
            
            metrics = {
                "n": int(len(trades_df)),
                "win_rate": win_rate,
                "avg_R": avg_R,
                "total_pnl": total_pnl,
                "max_dd": max_dd,
                "sharpe": sharpe,
            }
        
        dt = time.perf_counter() - t0
        logger.info(
            "Spectre trading complete | trades=%d | pnl=%.8f | sharpe=%.2f | %.3fs",
            metrics["n"],
            metrics["total_pnl"],
            metrics["sharpe"],
            dt,
        )
        
        return {
            "trades": trades_df,
            "metrics": metrics,
        }
    
    except Exception as e:
        logger.error("Spectre trading failed: %s", e, exc_info=True)
        raise
