from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, Any


@dataclass
class BacktestParams:
    """
    Backtest execution parameters.
    
    Default strategy:
    - BUY: Seller exhaustion signals only
    - SELL: First Fibonacci level hit only
    - Stop-loss and time exits are DISABLED by default (optional)
    """
    # Exit toggles
    use_stop_loss: bool = False          # Stop-loss exit (disabled by default)
    use_time_exit: bool = False          # Time-based exit (disabled by default)
    use_fib_exits: bool = True           # Fibonacci exits (enabled by default)
    use_traditional_tp: bool = False     # R-multiple TP (disabled by default)
    
    # Stop-loss parameters (used if use_stop_loss=True)
    atr_stop_mult: float = 0.7
    
    # Traditional TP parameters (used if use_traditional_tp=True)
    reward_r: float = 2.0
    
    # Time exit parameters (used if use_time_exit=True)
    max_hold: int = 96
    
    # Fibonacci parameters (used if use_fib_exits=True)
    fib_swing_lookback: int = 96
    fib_swing_lookahead: int = 5
    fib_target_level: float = 0.618
    
    # Transaction costs (always applied)
    fee_bp: float = 5.0
    slippage_bp: float = 5.0


def run_backtest(df: pd.DataFrame, p: BacktestParams) -> Dict[str, Any]:
    """
    Run event-driven backtest on DataFrame with exhaustion signals.
    
    Entry: t+1 open after signal (seller exhaustion bottom)
    
    Default exit: FIRST Fibonacci level hit
    Optional exits (must be enabled):
    - Stop-loss: signal_low - atr_stop_mult * ATR (if use_stop_loss=True)
    - Traditional TP: entry + reward_r * risk (if use_traditional_tp=True)
    - Time-based: max_hold bars (if use_time_exit=True)
    
    If NO exits are enabled, position stays open until end of data.
    """
    d = df.dropna(subset=["atr"]).copy()
    trades = []
    in_pos = False
    
    # Fibonacci level columns to check
    fib_cols = ["fib_0382", "fib_0500", "fib_0618", "fib_0786", "fib_1000"]
    
    for i in range(len(d.index) - 1):
        t = d.index[i]
        nxt = d.index[i + 1]
        row = d.loc[t]
        nxt_row = d.loc[nxt]
        
        if not in_pos and bool(row.get("exhaustion", False)):
            entry = float(nxt_row["open"])
            
            # Calculate stop (even if not used, needed for risk calculation)
            stop = float(row["low"] - p.atr_stop_mult * row["atr"]) if p.use_stop_loss else 0.0
            risk = max(1e-8, entry - stop) if p.use_stop_loss else entry * 0.01  # 1% risk if no stop
            
            # Determine take profit target (for traditional TP if enabled)
            tp = entry + p.reward_r * risk if p.use_traditional_tp else 0.0
            
            # Store Fib levels for this trade (if Fib exits enabled)
            fib_levels = {}
            if p.use_fib_exits:
                for col in fib_cols:
                    if col in row and not pd.isna(row[col]):
                        fib_levels[col] = float(row[col])
            
            bars = 0
            in_pos = True
            entry_ts = t
            
        elif in_pos:
            bars += 1
            op = float(nxt_row["open"])
            lo = float(nxt_row["low"])
            hi = float(nxt_row["high"])
            cl = float(nxt_row["close"])
            
            exit_price = None
            reason = None
            
            # Check exit conditions in order of priority
            # 1. Stop loss (if enabled)
            if p.use_stop_loss:
                if op <= stop:
                    exit_price = op
                    reason = "stop_gap"
                elif lo <= stop:
                    exit_price = stop
                    reason = "stop"
            
            # 2. Fibonacci exits (if enabled and not stopped out)
            if exit_price is None and p.use_fib_exits and fib_levels:
                # Check each Fib level in order
                for fib_col in fib_cols:
                    if fib_col in fib_levels:
                        fib_price = fib_levels[fib_col]
                        if hi >= fib_price:
                            exit_price = fib_price
                            fib_pct = int(fib_col.split("_")[1]) / 10.0
                            reason = f"fib_{fib_pct:.1f}"
                            break
            
            # 3. Traditional take profit (if enabled and not exited yet)
            if exit_price is None and p.use_traditional_tp and tp > 0 and hi >= tp:
                exit_price = tp
                reason = "tp"
            
            # 4. Time-based exit (if enabled and not exited yet)
            if exit_price is None and p.use_time_exit and bars >= p.max_hold:
                exit_price = cl
                reason = "time"
            
            if exit_price is not None:
                # Calculate fees and slippage
                fee = (entry + exit_price) * (p.fee_bp + p.slippage_bp) / 10000.0
                pnl = exit_price - entry - fee
                R = pnl / risk
                
                trades.append({
                    "entry_ts": str(entry_ts),
                    "exit_ts": str(nxt),
                    "entry": entry,
                    "exit": exit_price,
                    "stop": stop,
                    "tp": tp,
                    "pnl": pnl,
                    "R": R,
                    "reason": reason,
                    "bars_held": bars
                })
                in_pos = False
    
    tr = pd.DataFrame(trades)
    
    if len(tr) == 0:
        return {
            "trades": tr,
            "metrics": {
                "n": 0,
                "win_rate": 0.0,
                "avg_R": 0.0,
                "total_pnl": 0.0,
                "max_dd": 0.0,
                "sharpe": 0.0
            }
        }
    
    # Calculate metrics
    win_rate = float((tr["pnl"] > 0).mean())
    avg_R = float(tr["R"].mean())
    total_pnl = float(tr["pnl"].sum())
    
    # Calculate drawdown
    cumulative = tr["pnl"].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min())
    
    # Simple Sharpe approximation (assuming independent trades)
    sharpe = float(tr["R"].mean() / tr["R"].std()) if len(tr) > 1 and tr["R"].std() > 0 else 0.0
    
    return {
        "trades": tr,
        "metrics": {
            "n": int(len(tr)),
            "win_rate": win_rate,
            "avg_R": avg_R,
            "total_pnl": total_pnl,
            "max_dd": max_dd,
            "sharpe": sharpe
        }
    }
