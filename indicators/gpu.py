"""
GPU-accelerated technical indicators using PyTorch.

Provides massive speedup for indicator calculations and batch processing
of multiple parameter combinations on CUDA-enabled GPUs.
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def get_device():
    """Get best available device (CUDA GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_tensor(data: pd.Series, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert pandas Series to PyTorch tensor."""
    if device is None:
        device = get_device()
    return torch.tensor(data.values, dtype=torch.float32, device=device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor back to numpy array."""
    return tensor.cpu().detach().numpy()


def ema_gpu(x: torch.Tensor, span: int) -> torch.Tensor:
    """
    Calculate Exponential Moving Average on GPU.
    
    IMPORTANT: This uses a Python loop but ALL operations are on GPU.
    The loop itself runs fast on GPU since each iteration is minimal.
    
    Args:
        x: Input tensor (1D) on GPU
        span: EMA span (window size)
    
    Returns:
        EMA tensor of same shape as input, on same device
    """
    alpha = 2.0 / (span + 1.0)
    beta = 1.0 - alpha
    
    # Initialize result tensor on same device as input
    result = torch.zeros_like(x)
    result[0] = x[0]
    
    # Sequential update - PyTorch handles this efficiently on GPU
    # Each iteration: result[i] = alpha * x[i] + beta * result[i-1]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + beta * result[i-1]
    
    return result


def sma_gpu(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Calculate Simple Moving Average on GPU using conv1d.
    
    Args:
        x: Input tensor (1D)
        window: Window size
    
    Returns:
        SMA tensor (first window-1 values are NaN)
    """
    # Reshape for conv1d: (batch, channels, length)
    x_reshaped = x.unsqueeze(0).unsqueeze(0)
    
    # Create averaging kernel
    kernel = torch.ones(1, 1, window, device=x.device) / window
    
    # Apply convolution (valid padding)
    sma = torch.nn.functional.conv1d(x_reshaped, kernel, padding=0)
    
    # Reshape back and pad with NaN
    sma = sma.squeeze()
    result = torch.full_like(x, float('nan'))
    result[window-1:] = sma
    
    return result


def atr_gpu(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, window: int = 14) -> torch.Tensor:
    """
    Calculate Average True Range on GPU.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: ATR period
    
    Returns:
        ATR tensor
    """
    # True Range calculation
    high_low = high - low
    high_close = torch.abs(high - torch.roll(close, 1))
    low_close = torch.abs(low - torch.roll(close, 1))
    
    # Set first value to high-low (no previous close)
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    
    # True range is max of three values
    tr = torch.maximum(high_low, torch.maximum(high_close, low_close))
    
    # ATR is SMA of true range (CRITICAL: CPU uses rolling.mean(), not EMA!)
    return sma_gpu(tr, window)


def rsi_gpu(close: torch.Tensor, window: int = 14) -> torch.Tensor:
    """
    Calculate Relative Strength Index on GPU.
    
    Args:
        close: Close prices
        window: RSI period
    
    Returns:
        RSI tensor (0-100 range)
    """
    # Calculate price changes
    delta = close[1:] - close[:-1]
    delta = torch.cat([torch.zeros(1, device=close.device), delta])
    
    # Separate gains and losses
    gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
    losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))
    
    # Calculate average gains and losses using EMA
    avg_gains = ema_gpu(gains, window)
    avg_losses = ema_gpu(losses, window)
    
    # Calculate RS and RSI
    rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def zscore_gpu(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Calculate rolling z-score on GPU.
    
    Args:
        x: Input tensor
        window: Rolling window size
    
    Returns:
        Z-score tensor
    """
    # Calculate rolling mean
    mean = sma_gpu(x, window)
    
    # Calculate rolling std using conv1d
    x_reshaped = x.unsqueeze(0).unsqueeze(0)
    
    # Squared values for variance
    x_squared = x * x
    x_squared_reshaped = x_squared.unsqueeze(0).unsqueeze(0)
    
    # Rolling sum of squared values
    kernel = torch.ones(1, 1, window, device=x.device) / window
    mean_squared = torch.nn.functional.conv1d(x_squared_reshaped, kernel, padding=0).squeeze()
    
    # Variance = E[X^2] - E[X]^2
    variance = torch.full_like(x, float('nan'))
    variance[window-1:] = mean_squared - mean[window-1:]**2
    
    # Standard deviation
    std = torch.sqrt(torch.maximum(variance, torch.tensor(1e-10, device=x.device)))
    
    # Z-score
    zscore = (x - mean) / (std + 1e-10)
    
    return zscore


def macd_gpu(close: torch.Tensor, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate MACD on GPU.
    
    Args:
        close: Close prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = ema_gpu(close, fast)
    ema_slow = ema_gpu(close, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema_gpu(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


class GPUIndicatorBatch:
    """
    Batch calculator for multiple indicator configurations.
    Processes entire population of parameters in parallel on GPU.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
    
    def batch_ema(self, x: torch.Tensor, spans: list) -> torch.Tensor:
        """
        Calculate EMA for multiple spans in parallel.
        
        Args:
            x: Input data (length N)
            spans: List of span values
        
        Returns:
            Tensor of shape (len(spans), N) with EMA for each span
        """
        results = []
        for span in spans:
            results.append(ema_gpu(x, span))
        return torch.stack(results)
    
    def batch_zscore(self, x: torch.Tensor, windows: list) -> torch.Tensor:
        """
        Calculate z-scores for multiple windows in parallel.
        
        Args:
            x: Input data (length N)
            windows: List of window sizes
        
        Returns:
            Tensor of shape (len(windows), N) with z-scores
        """
        results = []
        for window in windows:
            results.append(zscore_gpu(x, window))
        return torch.stack(results)


def print_gpu_info():
    """Print GPU information for debugging."""
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"✓ PyTorch version: {torch.__version__}")
    else:
        print("⚠ CUDA not available, using CPU")
        print(f"✓ PyTorch version: {torch.__version__}")
