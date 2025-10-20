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
    Run event-driven backtest with optional exits.
    
    Entry: t+1 open after exhaustion signal.
    Exit priority:
        1. Stop-loss (including gap through stop)
        2. Fibonacci target (if enabled)
        3. Traditional TP (if enabled)
        4. Time-based exit (if enabled)
    """
    d = df.dropna(subset=["atr"]).copy()
    trades = []
    
    in_pos = False
    entry = None
    entry_ts = None
    fib_target_price = None
    stop_price = None
    tp_price = None
    risk = None
    bars = 0
    
    for i in range(len(d.index) - 1):
        t = d.index[i]
        nxt = d.index[i + 1]
        row = d.loc[t]
        nxt_row = d.loc[nxt]
        
        if not in_pos and bool(row.get("exhaustion", False)):
            # Entry at next bar open
            entry = float(nxt_row["open"])
            entry_ts = t
            bars = 0
            
            # Fibonacci target
            fib_target_price = None
            if p.use_fib_exits:
                target_col = FIB_LEVEL_TO_COL.get(p.fib_target_level)
                if target_col and target_col in row and not pd.isna(row[target_col]):
                    fib_target_price = float(row[target_col])
            
            # Stop-loss
            stop_price = None
            atr_value = float(row.get("atr", np.nan))
            if p.use_stop_loss and not np.isnan(atr_value):
                stop_price = float(row.get("low", entry)) - p.atr_stop_mult * atr_value
            
            # Risk used for R-multiple
            if p.use_stop_loss and stop_price is not None and stop_price < entry:
                risk = entry - stop_price
            else:
                risk = entry * 0.01  # Fallback assumption
            
            # Traditional TP
            tp_price = None
            if p.use_traditional_tp:
                tp_price = entry + p.reward_r * risk
            
            in_pos = True
            continue
        
        if not in_pos:
            continue
        
        bars += 1
        exit_price = None
        reason = None
        
        open_price = float(nxt_row["open"])
        high_price = float(nxt_row["high"])
        low_price = float(nxt_row["low"])
        close_price = float(nxt_row["close"])
        
        # 1) Stop-loss (gap)
        if p.use_stop_loss and stop_price is not None:
            if open_price <= stop_price:
                exit_price = stop_price
                reason = "STOP_GAP"
            elif low_price <= stop_price:
                exit_price = stop_price
                reason = "STOP"
        
        # 2) Fibonacci target
        if exit_price is None and p.use_fib_exits and fib_target_price is not None:
            if high_price >= fib_target_price:
                exit_price = fib_target_price
                fib_pct = p.fib_target_level * 100
                reason = f"FIB_{fib_pct:.1f}"
        
        # 3) Traditional TP
        if exit_price is None and p.use_traditional_tp and tp_price is not None:
            if high_price >= tp_price:
                exit_price = tp_price
                reason = f"TP_{p.reward_r:.1f}R"
        
        # 4) Time-based exit
        if exit_price is None and p.use_time_exit and bars >= p.max_hold:
            exit_price = close_price
            reason = "TIME"
        
        if exit_price is not None:
            fee = (entry + exit_price) * (p.fee_bp + p.slippage_bp) / 10000.0
            pnl = exit_price - entry - fee
            trade_r = pnl / risk if risk else 0.0
            
            trades.append({
                "entry_ts": str(entry_ts),
                "exit_ts": str(nxt),
                "entry": entry,
                "exit": exit_price,
                "pnl": pnl,
                "R": trade_r,
                "reason": reason,
                "bars_held": bars
            })
            
            # Reset position state
            in_pos = False
            entry = entry_ts = fib_target_price = stop_price = tp_price = risk = None
            bars = 0
    
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
    
    cumulative = tr["pnl"].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min())
    
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
