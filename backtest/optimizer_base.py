"""
Abstract base class for optimization algorithms.

Provides common interface for different optimizers (GA, ADAM, PSO, etc.)
to be plugged into the UI seamlessly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd

from strategy.seller_exhaustion import SellerParams
from core.models import BacktestParams, Timeframe, FitnessConfig


@dataclass
class OptimizationResult:
    """Result from one optimization step."""
    best_seller_params: SellerParams
    best_backtest_params: BacktestParams
    fitness: float
    metrics: Dict[str, Any]
    iteration: int  # Generation/iteration number
    additional_info: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    Provides unified interface for:
    - Evolutionary algorithms (GA, DE, PSO)
    - Gradient-based methods (ADAM, SGD)
    - Hybrid approaches
    """
    
    @abstractmethod
    def initialize(
        self,
        seed_seller_params: Optional[SellerParams] = None,
        seed_backtest_params: Optional[BacktestParams] = None,
        timeframe: Timeframe = Timeframe.m15
    ) -> None:
        """
        Initialize optimizer with optional seed parameters.
        
        Args:
            seed_seller_params: Starting point for strategy parameters
            seed_backtest_params: Starting point for backtest parameters
            timeframe: Timeframe for optimization (affects parameter bounds)
        """
        pass
    
    @abstractmethod
    def step(
        self,
        data: pd.DataFrame,
        timeframe: Timeframe,
        fitness_config: FitnessConfig,
        progress_callback: Optional[callable] = None,
        stop_flag: Optional[callable] = None
    ) -> OptimizationResult:
        """
        Run one optimization step.
        
        Args:
            data: Historical OHLCV data
            timeframe: Timeframe enum
            fitness_config: Fitness function configuration
            progress_callback: Optional callback(current, total, message) for progress updates
            stop_flag: Optional callable that returns True if optimization should stop
        
        Returns:
            OptimizationResult with best parameters found
        """
        pass
    
    @abstractmethod
    def get_best_params(self) -> tuple[Optional[SellerParams], Optional[BacktestParams], float]:
        """
        Return best parameters found so far.
        
        Returns:
            (seller_params, backtest_params, fitness) tuple
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Return current optimization statistics.
        
        Returns:
            Dictionary with optimizer-specific stats (e.g., population stats, gradient norms)
        """
        pass
    
    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Return optimization history for plotting.
        
        Returns:
            List of dicts with iteration-by-iteration stats
        """
        pass
    
    @abstractmethod
    def get_optimizer_name(self) -> str:
        """
        Return human-readable optimizer name.
        
        Returns:
            Optimizer name (e.g., "Evolutionary Algorithm", "ADAM")
        """
        pass
    
    @abstractmethod
    def get_acceleration_mode(self) -> str:
        """
        Return current acceleration mode.
        
        Returns:
            Acceleration mode (e.g., "CPU", "Multi-Core", "GPU")
        """
        pass
