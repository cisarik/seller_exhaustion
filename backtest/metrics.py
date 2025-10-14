import pandas as pd
from typing import Dict, Any


def calculate_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive trading metrics from trades DataFrame."""
    if len(trades) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_R": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0
        }
    
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    
    total_trades = len(trades)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    
    avg_R = trades["R"].mean()
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0.0
    
    total_wins = wins["pnl"].sum() if len(wins) > 0 else 0.0
    total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    total_pnl = trades["pnl"].sum()
    
    # Calculate max drawdown
    cumulative = trades["pnl"].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio
    sharpe = (trades["R"].mean() / trades["R"].std()) if len(trades) > 1 and trades["R"].std() > 0 else 0.0
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty print metrics."""
    print("\n=== Backtest Metrics ===")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Avg R: {metrics['avg_R']:.2f}")
    print(f"Avg Win: ${metrics['avg_win']:.4f}")
    print(f"Avg Loss: ${metrics['avg_loss']:.4f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total PnL: ${metrics['total_pnl']:.4f}")
    print(f"Max Drawdown: ${metrics['max_drawdown']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print("=======================\n")
