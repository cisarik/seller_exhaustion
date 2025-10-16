import pandas as pd
import numpy as np
from typing import Dict, Any
from core.models import BacktestParams


# Valid Fibonacci retracement levels (immutable)
VALID_FIB_LEVELS = [0.382, 0.500, 0.618, 0.786, 1.000]

# Mapping from Fib level to column name
FIB_LEVEL_TO_COL = {
    0.382: "fib_0382",
    0.500: "fib_0500",
    0.618: "fib_0618",
    0.786: "fib_0786",
    1.000: "fib_1000",
}


def run_backtest(df: pd.DataFrame, p: BacktestParams) -> Dict[str, Any]:
    """
    Run event-driven backtest with PURE Fibonacci exits.
    
    Entry: t+1 open after exhaustion signal
    Exit: ONLY when Fibonacci target level is hit
    
    No stop-loss, no traditional TP, no time exits.
    If Fibonacci target never hit, position never closes.
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
            # Entry at next bar open
            entry = float(nxt_row["open"])
            
            # Get Fibonacci target price for this trade
            fib_target_price = None
            target_col = FIB_LEVEL_TO_COL.get(p.fib_target_level)
            if target_col and target_col in row and not pd.isna(row[target_col]):
                fib_target_price = float(row[target_col])
            
            # Simple 1% risk assumption for R-multiple calculation
            risk = entry * 0.01
            
            bars = 0
            in_pos = True
            entry_ts = t
            
        elif in_pos:
            bars += 1
            hi = float(nxt_row["high"])
            
            exit_price = None
            reason = None
            
            # ONLY exit condition: Fibonacci target hit
            if fib_target_price is not None and hi >= fib_target_price:
                exit_price = fib_target_price
                fib_pct = p.fib_target_level * 100
                reason = f"FIB_{fib_pct:.1f}"
            
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
