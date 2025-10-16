"""
Test GPU exit detection logic matches CPU exactly.

CRITICAL: Exit priority must be:
1. Stop gap (open <= stop)
2. Stop hit (low <= stop)
3. TP hit (high >= tp)
4. Time exit (max_hold)
"""

import pytest
import torch
import pandas as pd
import numpy as np
from backtest.engine_gpu_full import find_exits_vectorized_gpu


@pytest.fixture
def device():
    """Get GPU device if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_stop_gap_exit(device):
    """Test gap down through stop (open <= stop)."""
    # Create scenario: entry at 1.0, stop at 0.9, next bar opens at 0.88
    entry_indices = torch.tensor([0], device=device, dtype=torch.long)
    entry_prices = torch.tensor([1.0], device=device, dtype=torch.float32)
    stop_prices = torch.tensor([0.9], device=device, dtype=torch.float32)
    tp_prices = torch.tensor([1.1], device=device, dtype=torch.float32)
    
    # Price data: [entry_bar, exit_bar]
    open_t = torch.tensor([1.0, 0.88], device=device, dtype=torch.float32)   # Opens at 0.88 < 0.9 stop
    high_t = torch.tensor([1.0, 0.95], device=device, dtype=torch.float32)
    low_t = torch.tensor([0.95, 0.85], device=device, dtype=torch.float32)
    close_t = torch.tensor([1.0, 0.90], device=device, dtype=torch.float32)
    
    exits = find_exits_vectorized_gpu(
        entry_indices, entry_prices, stop_prices, tp_prices,
        max_hold=10,
        open_t=open_t, high_t=high_t, low_t=low_t, close_t=close_t,
        use_stop=True, use_tp=True, use_time_exit=True,
        device=device
    )
    
    assert exits['exit_reasons'][0].item() == 1, "Should exit via stop_gap"
    assert abs(exits['exit_prices'][0].item() - 0.88) < 1e-6, "Should exit at open price"
    assert exits['bars_held'][0].item() == 1, "Should exit on next bar"
    print("✓ Stop gap exit test passed")


def test_stop_hit_exit(device):
    """Test stop hit during bar (low <= stop, but open > stop)."""
    # Entry at 1.0, stop at 0.9, bar dips to 0.89 but opens at 0.95
    entry_indices = torch.tensor([0], device=device, dtype=torch.long)
    entry_prices = torch.tensor([1.0], device=device, dtype=torch.float32)
    stop_prices = torch.tensor([0.9], device=device, dtype=torch.float32)
    tp_prices = torch.tensor([1.1], device=device, dtype=torch.float32)
    
    # Price data: [entry_bar, exit_bar]
    open_t = torch.tensor([1.0, 0.95], device=device, dtype=torch.float32)   # Opens above stop
    high_t = torch.tensor([1.0, 0.96], device=device, dtype=torch.float32)
    low_t = torch.tensor([0.95, 0.89], device=device, dtype=torch.float32)   # Dips to 0.89 <= 0.9 stop
    close_t = torch.tensor([1.0, 0.92], device=device, dtype=torch.float32)
    
    exits = find_exits_vectorized_gpu(
        entry_indices, entry_prices, stop_prices, tp_prices,
        max_hold=10,
        open_t=open_t, high_t=high_t, low_t=low_t, close_t=close_t,
        use_stop=True, use_tp=True, use_time_exit=True,
        device=device
    )
    
    assert exits['exit_reasons'][0].item() == 2, "Should exit via stop (not stop_gap)"
    assert abs(exits['exit_prices'][0].item() - 0.9) < 1e-6, "Should exit at stop price"
    assert exits['bars_held'][0].item() == 1, "Should exit on next bar"
    print("✓ Stop hit exit test passed")


def test_tp_exit(device):
    """Test take profit exit (high >= tp)."""
    # Entry at 1.0, TP at 1.1, bar reaches 1.12
    entry_indices = torch.tensor([0], device=device, dtype=torch.long)
    entry_prices = torch.tensor([1.0], device=device, dtype=torch.float32)
    stop_prices = torch.tensor([0.9], device=device, dtype=torch.float32)
    tp_prices = torch.tensor([1.1], device=device, dtype=torch.float32)
    
    # Price data: [entry_bar, exit_bar]
    open_t = torch.tensor([1.0, 1.05], device=device, dtype=torch.float32)
    high_t = torch.tensor([1.0, 1.12], device=device, dtype=torch.float32)   # High reaches 1.12 >= 1.1 TP
    low_t = torch.tensor([0.95, 1.03], device=device, dtype=torch.float32)   # Above stop
    close_t = torch.tensor([1.0, 1.08], device=device, dtype=torch.float32)
    
    exits = find_exits_vectorized_gpu(
        entry_indices, entry_prices, stop_prices, tp_prices,
        max_hold=10,
        open_t=open_t, high_t=high_t, low_t=low_t, close_t=close_t,
        use_stop=True, use_tp=True, use_time_exit=True,
        device=device
    )
    
    assert exits['exit_reasons'][0].item() == 3, "Should exit via TP"
    assert abs(exits['exit_prices'][0].item() - 1.1) < 1e-6, "Should exit at TP price"
    assert exits['bars_held'][0].item() == 1, "Should exit on next bar"
    print("✓ TP exit test passed")


def test_time_exit(device):
    """Test time-based exit (max_hold exceeded, no stop/TP hit)."""
    # Entry at 1.0, stop at 0.9, TP at 1.1, price stays at 1.0 for max_hold bars
    entry_indices = torch.tensor([0], device=device, dtype=torch.long)
    entry_prices = torch.tensor([1.0], device=device, dtype=torch.float32)
    stop_prices = torch.tensor([0.9], device=device, dtype=torch.float32)
    tp_prices = torch.tensor([1.1], device=device, dtype=torch.float32)
    
    # Price data: 5 bars of sideways action
    max_hold = 3
    open_t = torch.tensor([1.0, 1.01, 1.00, 0.99, 1.00], device=device, dtype=torch.float32)
    high_t = torch.tensor([1.0, 1.02, 1.03, 1.01, 1.02], device=device, dtype=torch.float32)
    low_t = torch.tensor([0.95, 0.99, 0.98, 0.97, 0.98], device=device, dtype=torch.float32)
    close_t = torch.tensor([1.0, 1.00, 1.01, 1.00, 1.01], device=device, dtype=torch.float32)
    
    exits = find_exits_vectorized_gpu(
        entry_indices, entry_prices, stop_prices, tp_prices,
        max_hold=max_hold,
        open_t=open_t, high_t=high_t, low_t=low_t, close_t=close_t,
        use_stop=True, use_tp=True, use_time_exit=True,
        device=device
    )
    
    assert exits['exit_reasons'][0].item() == 4, "Should exit via time"
    # Should exit at close of bar after holding for max_hold bars
    # Entry at bar 0, hold through bars 1,2,3, exit at bar 3
    expected_exit_idx = 0 + max_hold
    assert exits['exit_indices'][0].item() == expected_exit_idx, f"Should exit at bar {expected_exit_idx}"
    # Exit price should be close of that bar
    expected_price = close_t[expected_exit_idx].item()
    assert abs(exits['exit_prices'][0].item() - expected_price) < 1e-6, "Should exit at close price"
    print("✓ Time exit test passed")


def test_exit_priority(device):
    """Test that stop_gap has priority over stop over TP."""
    # Create bar where stop_gap, stop, AND tp all trigger
    # Priority should be: stop_gap (1) > stop (2) > TP (3)
    entry_indices = torch.tensor([0], device=device, dtype=torch.long)
    entry_prices = torch.tensor([1.0], device=device, dtype=torch.float32)
    stop_prices = torch.tensor([0.95], device=device, dtype=torch.float32)
    tp_prices = torch.tensor([1.05], device=device, dtype=torch.float32)
    
    # Price data: Bar that gaps down, goes even lower, but also reaches TP
    # This is unrealistic but tests priority
    open_t = torch.tensor([1.0, 0.93], device=device, dtype=torch.float32)   # Gap to 0.93 <= 0.95 stop (stop_gap!)
    high_t = torch.tensor([1.0, 1.10], device=device, dtype=torch.float32)   # Also hits TP (1.10 >= 1.05)
    low_t = torch.tensor([0.95, 0.90], device=device, dtype=torch.float32)   # Also hits stop (0.90 <= 0.95)
    close_t = torch.tensor([1.0, 0.95], device=device, dtype=torch.float32)
    
    exits = find_exits_vectorized_gpu(
        entry_indices, entry_prices, stop_prices, tp_prices,
        max_hold=10,
        open_t=open_t, high_t=high_t, low_t=low_t, close_t=close_t,
        use_stop=True, use_tp=True, use_time_exit=True,
        device=device
    )
    
    # Should exit via stop_gap (highest priority)
    assert exits['exit_reasons'][0].item() == 1, "Should prioritize stop_gap over stop and TP"
    assert abs(exits['exit_prices'][0].item() - 0.93) < 1e-6, "Should exit at open (gap) price"
    print("✓ Exit priority test passed")


def test_multiple_trades(device):
    """Test vectorized exit finding for multiple trades - simplified."""
    # 2 trades: one TP exit, one stop exit
    entry_indices = torch.tensor([0, 3], device=device, dtype=torch.long)
    entry_prices = torch.tensor([1.0, 0.5], device=device, dtype=torch.float32)
    stop_prices = torch.tensor([0.9, 0.45], device=device, dtype=torch.float32)
    tp_prices = torch.tensor([1.1, 0.55], device=device, dtype=torch.float32)
    
    # Price data: 10 bars
    # Trade 0: Enters at bar 0, exits via TP at bar 2
    # Trade 1: Enters at bar 3, exits via stop at bar 5
    open_t = torch.tensor([
        1.00, 1.02, 1.05, # Trade 0: bars 0-2
        0.50, 0.51, 0.48, # Trade 1: bars 3-5, bar 5 opens below stop (stop_gap)
        1.00, 1.00, 1.00, 1.00
    ], device=device, dtype=torch.float32)
    
    high_t = torch.tensor([
        1.01, 1.05, 1.12, # Trade 0: bar 2 high hits TP (1.12 >= 1.1)
        0.52, 0.53, 0.49,
        1.01, 1.01, 1.01, 1.01
    ], device=device, dtype=torch.float32)
    
    low_t = torch.tensor([
        0.98, 1.00, 1.03, # Trade 0: no stops hit (all > 0.9)
        0.48, 0.49, 0.42, # Trade 1: bar 5 low hits stop (0.42 <= 0.45)
        0.99, 0.99, 0.99, 0.99
    ], device=device, dtype=torch.float32)
    
    close_t = torch.ones(10, device=device, dtype=torch.float32)
    
    exits = find_exits_vectorized_gpu(
        entry_indices, entry_prices, stop_prices, tp_prices,
        max_hold=10,
        open_t=open_t, high_t=high_t, low_t=low_t, close_t=close_t,
        use_stop=True, use_tp=True, use_time_exit=True,
        device=device
    )
    
    # Trade 0: Should exit via TP at bar 2
    assert exits['exit_reasons'][0].item() == 3, f"Trade 0 should exit via TP, got {exits['exit_reasons'][0].item()}"
    assert exits['exit_indices'][0].item() == 2, f"Trade 0 should exit at bar 2, got {exits['exit_indices'][0].item()}"
    
    # Trade 1: Should exit via stop at bar 5
    assert exits['exit_reasons'][1].item() == 2, f"Trade 1 should exit via stop, got {exits['exit_reasons'][1].item()}"
    assert exits['exit_indices'][1].item() == 5, f"Trade 1 should exit at bar 5, got {exits['exit_indices'][1].item()}"
    
    print("✓ Multiple trades test passed")


def test_disabled_exits(device):
    """Test that disabled exit types are not triggered."""
    entry_indices = torch.tensor([0], device=device, dtype=torch.long)
    entry_prices = torch.tensor([1.0], device=device, dtype=torch.float32)
    stop_prices = torch.tensor([0.9], device=device, dtype=torch.float32)
    tp_prices = torch.tensor([1.1], device=device, dtype=torch.float32)
    
    # Bar that hits both stop and TP
    open_t = torch.tensor([1.0, 1.05], device=device, dtype=torch.float32)
    high_t = torch.tensor([1.0, 1.15], device=device, dtype=torch.float32)   # Hits TP
    low_t = torch.tensor([0.95, 0.85], device=device, dtype=torch.float32)   # Hits stop
    close_t = torch.tensor([1.0, 1.00], device=device, dtype=torch.float32)
    
    # Test with stop disabled, TP enabled
    exits = find_exits_vectorized_gpu(
        entry_indices, entry_prices, stop_prices, tp_prices,
        max_hold=10,
        open_t=open_t, high_t=high_t, low_t=low_t, close_t=close_t,
        use_stop=False, use_tp=True, use_time_exit=True,
        device=device
    )
    
    assert exits['exit_reasons'][0].item() == 3, "Should exit via TP when stop disabled"
    
    # Test with TP disabled, stop enabled
    exits = find_exits_vectorized_gpu(
        entry_indices, entry_prices, stop_prices, tp_prices,
        max_hold=10,
        open_t=open_t, high_t=high_t, low_t=low_t, close_t=close_t,
        use_stop=True, use_tp=False, use_time_exit=True,
        device=device
    )
    
    assert exits['exit_reasons'][0].item() == 2, "Should exit via stop when TP disabled"
    
    # Test with both disabled, only time exit
    exits = find_exits_vectorized_gpu(
        entry_indices, entry_prices, stop_prices, tp_prices,
        max_hold=2,
        open_t=open_t, high_t=high_t, low_t=low_t, close_t=close_t,
        use_stop=False, use_tp=False, use_time_exit=True,
        device=device
    )
    
    assert exits['exit_reasons'][0].item() == 4, "Should exit via time when stop and TP disabled"
    
    print("✓ Disabled exits test passed")


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
