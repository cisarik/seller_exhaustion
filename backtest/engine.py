from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, Any


@dataclass
class BacktestParams:
    """Backtest execution parameters."""
    atr_stop_mult: float = 0.7
    reward_r: float = 2.0
    max_hold: int = 96
    fee_bp: float = 5.0
    slippage_bp: float = 5.0


def run_backtest(df: pd.DataFrame, p: BacktestParams) -> Dict[str, Any]:
    """
    Run event-driven backtest on DataFrame with exhaustion signals.
    
    Entry: t+1 open after signal
    Stop: signal_low - atr_stop_mult * ATR
    TP: entry + reward_r * (entry - stop)
    Exit: hit stop/TP or max_hold bars
    """
    d = df.dropna(subset=["atr"]).copy()
    trades = []
    in_pos = False
    
    for i in range(len(d.index) - 1):
        t = d.index[i]
        nxt = d.index[i + 1]
        row = d.loc[t]
        nxt_row = d.loc[nxt]
        
        if not in_pos and bool(row.get("exhaustion", False)):
            entry = float(nxt_row["open"])
            stop = float(row["low"] - p.atr_stop_mult * row["atr"])
            risk = max(1e-8, entry - stop)
            tp = entry + p.reward_r * risk
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
            if op <= stop:
                exit_price = op
                reason = "stop_gap"
            elif lo <= stop:
                exit_price = stop
                reason = "stop"
            elif hi >= tp:
                exit_price = tp
                reason = "tp"
            elif bars >= p.max_hold:
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
