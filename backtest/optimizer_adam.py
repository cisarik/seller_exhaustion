"""
ADAM optimizer for strategy parameter optimization.

Uses finite differences to approximate gradients of the fitness function,
then applies ADAM updates to continuously optimize parameters.

Supports multi-core parallelization for gradient computation (each parameter
gradient is computed in parallel).

Note: This is experimental. Gradient-based optimization on discrete,
noisy fitness landscapes is challenging. May converge slowly or get
stuck in local optima.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from backtest.optimizer_base import BaseOptimizer, OptimizationResult
from backtest.optimizer import (
    calculate_fitness,
    get_param_bounds_for_timeframe,
    INTEGER_PARAMS,
    VALID_FIB_LEVELS,
    Population,
)
from strategy.seller_exhaustion import SellerParams, build_features
from core.models import BacktestParams, Timeframe, FitnessConfig
from backtest.engine import run_backtest


# Worker function for parallel gradient computation (must be at module level for pickling)
def _evaluate_perturbed_parameter(
    param_vector: np.ndarray,
    param_index: int,
    param_name: str,
    epsilon: float,
    data_dict: dict,
    timeframe_value: int,
    fitness_config_dict: dict
) -> tuple[int, float, str]:
    """
    Evaluate fitness for a perturbed parameter (worker function).
    
    This function is called in a separate process for parallel gradient computation.
    
    Args:
        param_vector: Current parameter vector (normalized 0-1)
        param_index: Index of parameter to perturb
        param_name: Name of parameter (for logging)
        epsilon: Perturbation size
        data_dict: Data as dict (columns: open, high, low, close, volume)
        timeframe_value: Timeframe value (minutes)
        fitness_config_dict: FitnessConfig as dict
    
    Returns:
        (param_index, fitness, param_name) tuple
    """
    import numpy as np
    import pandas as pd
    from core.models import Timeframe, FitnessConfig
    
    # Reconstruct data
    data_payload = dict(data_dict)  # shallow copy
    index_values = data_payload.pop("index", None)
    data = pd.DataFrame(data_payload)
    if index_values is not None:
        data.index = pd.to_datetime(index_values, utc=True)
        data.index.name = "ts"
    
    # Reconstruct timeframe
    timeframe = Timeframe(timeframe_value)
    
    # Reconstruct fitness config
    fitness_config = FitnessConfig(**fitness_config_dict)
    
    # Perturb parameter
    perturbed = param_vector.copy()
    original_value = perturbed[param_index]
    perturbed[param_index] = min(1.0, perturbed[param_index] + epsilon)
    
    # Convert to SellerParams and BacktestParams
    from backtest.optimizer import get_param_bounds_for_timeframe, INTEGER_PARAMS, VALID_FIB_LEVELS
    bounds = get_param_bounds_for_timeframe(timeframe)
    
    # DEBUG: Print perturbation details for first few parameters
    if param_index < 3:
        print(f"  DEBUG [{param_name}]: epsilon={epsilon:.4f}, normalized: {original_value:.4f} → {perturbed[param_index]:.4f}")
    
    # Denormalize
    idx = 0
    seller_dict = {}
    for key in ['ema_fast', 'ema_slow', 'z_window', 'vol_z', 'tr_z', 'cloc_min', 'atr_window']:
        min_val, max_val = bounds[key]
        val = perturbed[idx] * (max_val - min_val) + min_val
        if key in INTEGER_PARAMS:
            val = int(round(val))
        seller_dict[key] = val
        idx += 1
    
    backtest_dict = {}
    for key in ['fib_swing_lookback', 'fib_swing_lookahead', 'fib_target_level', 'fee_bp', 'slippage_bp']:
        # Special handling for fib_target_level (discrete choice)
        if key == 'fib_target_level':
            # Map 0-1 range back to discrete level
            level_idx = int(round(perturbed[idx] * (len(VALID_FIB_LEVELS) - 1)))
            level_idx = np.clip(level_idx, 0, len(VALID_FIB_LEVELS) - 1)
            val = VALID_FIB_LEVELS[level_idx]
        else:
            min_val, max_val = bounds[key]
            val = perturbed[idx] * (max_val - min_val) + min_val
            if key in INTEGER_PARAMS:
                val = int(round(val))
        backtest_dict[key] = val
        idx += 1
    
    # Build features and run backtest
    from strategy.seller_exhaustion import SellerParams, build_features
    from core.models import BacktestParams
    from backtest.engine import run_backtest
    from backtest.optimizer import calculate_fitness
    
    seller_params = SellerParams(**seller_dict)
    backtest_params = BacktestParams(**backtest_dict)
    
    # DEBUG: Print denormalized values for perturbed parameter
    if param_index < 7:  # Strategy params
        actual_value = seller_dict[list(seller_dict.keys())[param_index]]
        if param_index < 3:
            print(f"  DEBUG [{param_name}]: actual value = {actual_value}")
    
    try:
        feats = build_features(data, seller_params, timeframe)
        result = run_backtest(feats, backtest_params)
        fitness = calculate_fitness(result['metrics'], fitness_config)
    except Exception as e:
        print(f"Error evaluating parameter {param_name}: {e}")
        fitness = -1000.0
    
    return param_index, fitness, param_name


class AdamOptimizer(BaseOptimizer):
    """
    ADAM-based optimizer using finite differences for gradient approximation.
    
    Key features:
    - Adaptive learning rates per parameter
    - Momentum-based updates
    - Gradient clipping for stability
    - Configurable multi-core CPU evaluation for gradient computation
    
    Limitations:
    - Requires many fitness evaluations per step (one per parameter)
    - May get stuck in local optima
    - Works best with smooth fitness landscapes
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 0.02,  # For finite differences (2% step ensures meaningful changes in integer params)
        max_grad_norm: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        n_workers: Optional[int] = None,
        initial_population_file: Optional[str] = None,
    ):
        """
        Initialize ADAM optimizer.
        
        Args:
            learning_rate: Learning rate for ADAM updates
            epsilon: Step size for finite difference approximation
            max_grad_norm: Maximum gradient norm (for clipping)
            adam_beta1: ADAM beta1 parameter (momentum)
            adam_beta2: ADAM beta2 parameter (RMSprop-like)
            adam_epsilon: ADAM epsilon (numerical stability)
            n_workers: Number of worker processes for gradient evaluation (defaults to CPU count)
        """
        self.lr = learning_rate
        self.fd_epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        
        # Parallel evaluation (>=1 worker)
        self.n_workers = max(1, n_workers or multiprocessing.cpu_count())
        
        # Set device (CPU only)
        self.device = torch.device('cpu')
        
        # Parameter tensors (will be initialized)
        self.params_tensor = None
        self.optimizer = None
        self.timeframe = None
        self.initial_population_file = initial_population_file
        
        # Tracking
        self.iteration = 0
        self.best_fitness = -float('inf')
        self.best_seller_params = None
        self.best_backtest_params = None
        self.best_metrics = {}
        self.history = []
        
        # Parameter metadata
        self.param_names = [
            'ema_fast', 'ema_slow', 'z_window', 'vol_z', 'tr_z',
            'cloc_min', 'atr_window', 'fib_swing_lookback', 'fib_swing_lookahead',
            'fib_target_level', 'fee_bp', 'slippage_bp'
        ]
        self.bounds = None
        
        print(f"✓ ADAM optimizer initialized on {self.device}")
    
    def initialize(
        self,
        seed_seller_params: Optional[SellerParams] = None,
        seed_backtest_params: Optional[BacktestParams] = None,
        timeframe: Timeframe = Timeframe.m15
    ) -> None:
        """Initialize optimizer with seed parameters."""
        self.timeframe = timeframe
        self.bounds = get_param_bounds_for_timeframe(timeframe)
        
        # If nie sú poskytnuté seed parametre a máme súbor populácie, načítame z neho
        if (seed_seller_params is None or seed_backtest_params is None) and self.initial_population_file:
            try:
                pop = Population.from_file(self.initial_population_file, timeframe=timeframe)
                best = getattr(pop, 'best_ever', None)
                ind = best if best is not None else (pop.individuals[0] if getattr(pop, 'individuals', []) else None)
                if ind is not None:
                    seed_seller_params = seed_seller_params or ind.seller_params
                    seed_backtest_params = seed_backtest_params or ind.backtest_params
                    print(f"✓ ADAM seed načítaný z populácie: {self.initial_population_file}")
            except Exception as e:
                print(f"⚠ Nepodarilo sa načítať populáciu pre ADAM: {e}. Použijú sa defaulty alebo poskytnuté seedy.")
        
        # Use defaults if still missing
        if seed_seller_params is None:
            seed_seller_params = SellerParams()
        if seed_backtest_params is None:
            seed_backtest_params = BacktestParams()
        
        # Convert to normalized parameter vector
        param_vector = self._params_to_vector(seed_seller_params, seed_backtest_params)
        
        # Create tensor with gradient tracking
        self.params_tensor = torch.tensor(
            param_vector,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        
        # Create ADAM optimizer
        self.optimizer = optim.Adam(
            [self.params_tensor],
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon
        )
        
        # Initialize best with seed
        self.best_seller_params = deepcopy(seed_seller_params)
        self.best_backtest_params = deepcopy(seed_backtest_params)
        
        self.iteration = 0
        self.history = []
        
        print(
            f"✓ ADAM initialized with learning_rate={self.lr}, fd_epsilon={self.fd_epsilon}, "
            f"n_workers={self.n_workers}"
        )
        print(f"  DEBUG: Epsilon value being used: {self.fd_epsilon}")
    
    def _params_to_vector(self, seller_params: SellerParams, backtest_params: BacktestParams) -> np.ndarray:
        """Convert parameters to normalized vector (0-1 range)."""
        vector = []
        
        # SellerParams
        for key in ['ema_fast', 'ema_slow', 'z_window', 'vol_z', 'tr_z', 'cloc_min', 'atr_window']:
            val = getattr(seller_params, key)
            min_val, max_val = self.bounds[key]
            span = max(max_val - min_val, 1e-9)
            normalized = (val - min_val) / span
            vector.append(np.clip(normalized, 0, 1))
        
        # BacktestParams
        for key in ['fib_swing_lookback', 'fib_swing_lookahead', 'fib_target_level', 'fee_bp', 'slippage_bp']:
            val = getattr(backtest_params, key)
            
            # Special handling for fib_target_level (discrete choice)
            if key == 'fib_target_level':
                # Map discrete levels to 0-1 range (index / (n_levels - 1))
                try:
                    idx = VALID_FIB_LEVELS.index(val)
                    normalized = idx / (len(VALID_FIB_LEVELS) - 1)
                except ValueError:
                    # Default to Golden Ratio if not found
                    normalized = 0.5
            else:
                min_val, max_val = self.bounds[key]
                span = max(max_val - min_val, 1e-9)
                normalized = (val - min_val) / span
            
            vector.append(np.clip(normalized, 0, 1))
        
        return np.array(vector, dtype=np.float32)
    
    def _vector_to_params(self, vector: np.ndarray) -> tuple[SellerParams, BacktestParams]:
        """Convert normalized vector to parameters."""
        # Clamp to [0, 1]
        vector = np.clip(vector, 0, 1)
        
        # Denormalize
        idx = 0
        
        # SellerParams
        seller_dict = {}
        for key in ['ema_fast', 'ema_slow', 'z_window', 'vol_z', 'tr_z', 'cloc_min', 'atr_window']:
            min_val, max_val = self.bounds[key]
            val = vector[idx] * (max_val - min_val) + min_val
            
            # Round integers
            if key in INTEGER_PARAMS:
                val = int(round(val))
            
            seller_dict[key] = val
            idx += 1
        
        # BacktestParams
        backtest_dict = {}
        for key in ['fib_swing_lookback', 'fib_swing_lookahead', 'fib_target_level', 'fee_bp', 'slippage_bp']:
            # Special handling for fib_target_level (discrete choice)
            if key == 'fib_target_level':
                # Map 0-1 range back to discrete level
                level_idx = int(round(vector[idx] * (len(VALID_FIB_LEVELS) - 1)))
                level_idx = np.clip(level_idx, 0, len(VALID_FIB_LEVELS) - 1)
                val = VALID_FIB_LEVELS[level_idx]
            else:
                min_val, max_val = self.bounds[key]
                val = vector[idx] * (max_val - min_val) + min_val
                
                # Round integers
                if key in INTEGER_PARAMS:
                    val = int(round(val))
            
            backtest_dict[key] = val
            idx += 1
        
        return SellerParams(**seller_dict), BacktestParams(**backtest_dict)
    
    def _evaluate_fitness(
        self,
        vector: np.ndarray,
        data,
        fitness_config: FitnessConfig
    ) -> tuple[float, dict]:
        """Evaluate fitness for a parameter vector."""
        seller_params, backtest_params = self._vector_to_params(vector)
        
        try:
            # Build features
            feats = build_features(data, seller_params, self.timeframe)
            
            # Run backtest
            result = run_backtest(feats, backtest_params)
            
            # Calculate fitness
            fitness = calculate_fitness(result['metrics'], fitness_config)
            
            return fitness, result['metrics']
            
        except Exception as e:
            print(f"⚠ Evaluation error: {e}")
            return -1000.0, {'n': 0, 'error': str(e)}
    
    
    def _compute_gradient_cpu_sequential(
        self,
        data,
        fitness_config: FitnessConfig,
        progress_callback: Optional[callable],
        stop_flag: Optional[callable]
    ) -> tuple[np.ndarray, float, dict]:
        """Sequential CPU gradient computation (original implementation)."""
        current_vector = self.params_tensor.detach().cpu().numpy()
        
        # Evaluate at current point
        if progress_callback:
            progress_callback(0, len(current_vector) + 1, "Evaluating current parameters...")
        
        f_current, metrics_current = self._evaluate_fitness(current_vector, data, fitness_config)
        print(f"Current fitness: {f_current:.4f} | Trades: {metrics_current.get('n', 0)}")
        
        # Compute gradient for each parameter
        gradient = np.zeros_like(current_vector)
        
        print(f"Computing gradients (finite differences)...")
        for i in range(len(current_vector)):
            # Check stop flag
            if stop_flag and stop_flag():
                print("⏹ Stop requested during gradient computation")
                gradient[i:] = 0  # Zero out remaining
                break
            
            # Update progress
            if progress_callback:
                progress_callback(
                    i + 1,
                    len(current_vector) + 1,
                    f"Computing gradient {i+1}/{len(current_vector)} ({self.param_names[i]})..."
                )
            
            # Perturb parameter i
            perturbed = current_vector.copy()
            perturbed[i] = min(1.0, perturbed[i] + self.fd_epsilon)
            
            # Evaluate
            f_perturbed, _ = self._evaluate_fitness(perturbed, data, fitness_config)
            
            # Finite difference
            gradient[i] = (f_perturbed - f_current) / self.fd_epsilon
            
            # Show progress
            print(f"  [{i+1}/{len(current_vector)}] {self.param_names[i]}: grad={gradient[i]:.6f}")
        
        return gradient, f_current, metrics_current
    
    def _compute_gradient_finite_diff(
        self,
        data,
        fitness_config: FitnessConfig,
        progress_callback: Optional[callable] = None,
        stop_flag: Optional[callable] = None
    ) -> tuple[torch.Tensor, float, dict]:
        """
        Compute gradient using finite differences.
        
        For each parameter i:
            grad_i ≈ (f(x + ε*e_i) - f(x)) / ε
        
        Args:
            data: Historical OHLCV data
            fitness_config: Fitness configuration
            progress_callback: Optional callback(current, total, message) for progress
            stop_flag: Optional callable that returns True if should stop
        
        Returns:
            (gradient_tensor, current_fitness, current_metrics)
        """
        # Multi-core CPU mode
        if self.n_workers > 1:
            current_vector = self.params_tensor.detach().cpu().numpy()
            
            # Evaluate at current point first
            if progress_callback:
                progress_callback(0, len(current_vector) + 1, "Evaluating current parameters...")
            
            f_current, metrics_current = self._evaluate_fitness(current_vector, data, fitness_config)
            print(f"Current fitness: {f_current:.4f} | Trades: {metrics_current.get('n', 0)}")
            
            # Compute gradients in parallel (pass f_current)
            gradient = self._compute_gradient_parallel(
                current_vector,
                f_current,
                data,
                fitness_config,
                progress_callback,
                stop_flag
            )
            
            # Negate for maximization
            gradient = -gradient
            return torch.tensor(gradient, dtype=torch.float32, device=self.device), f_current, metrics_current
        
        # Single-core CPU mode (sequential)
        else:
            gradient, f_current, metrics_current = self._compute_gradient_cpu_sequential(
                data, fitness_config, progress_callback, stop_flag
            )
            # Negate for maximization
            gradient = -gradient
            return torch.tensor(gradient, dtype=torch.float32, device=self.device), f_current, metrics_current
    
    def _compute_gradient_parallel(
        self,
        current_vector: np.ndarray,
        f_current: float,
        data,
        fitness_config: FitnessConfig,
        progress_callback: Optional[callable] = None,
        stop_flag: Optional[callable] = None
    ) -> np.ndarray:
        """
        Compute gradient using multi-core parallel finite differences.
        
        Each parameter's gradient is computed in parallel across CPU cores.
        
        Returns:
            gradient vector (numpy array)
        """
        print(f"Computing gradients (parallel finite differences with {self.n_workers} workers)...")
        
        gradient = np.zeros_like(current_vector)
        
        # Prepare data for serialization (convert DataFrame to dict)
        data_dict = {
            'open': data['open'].values,
            'high': data['high'].values,
            'low': data['low'].values,
            'close': data['close'].values,
            'volume': data['volume'].values,
            'index': data.index.values
        }
        
        # Convert fitness config to dict for serialization
        fitness_config_dict = fitness_config.model_dump()
        
        # Submit all parameter evaluations to process pool
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            futures = {}
            for i in range(len(current_vector)):
                # Check stop flag before submitting
                if stop_flag and stop_flag():
                    print("⏹ Stop requested before parallel computation")
                    break
                
                future = executor.submit(
                    _evaluate_perturbed_parameter,
                    current_vector,
                    i,
                    self.param_names[i],
                    self.fd_epsilon,
                    data_dict,
                    self.timeframe.value,
                    fitness_config_dict
                )
                futures[future] = i
            
            # Collect results as they complete
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                # Check stop flag
                if stop_flag and stop_flag():
                    print("⏹ Stop requested during parallel computation")
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
                
                try:
                    param_index, f_perturbed, param_name = future.result()
                    
                    # Compute gradient
                    gradient[param_index] = (f_perturbed - f_current) / self.fd_epsilon
                    
                    completed += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(
                            completed,
                            total + 1,
                            f"Computing gradient {completed}/{total} ({param_name})..."
                        )
                    
                    # Show progress
                    print(f"  [{completed}/{total}] {param_name}: grad={gradient[param_index]:.6f}")
                    
                except Exception as e:
                    print(f"  Error computing gradient: {e}")
                    # Continue with other parameters
        
        print(f"✓ Parallel gradient computation complete ({completed}/{total} parameters)")
        
        return gradient
    
    def step(
        self,
        data,
        timeframe: Timeframe,
        fitness_config: FitnessConfig,
        progress_callback: Optional[callable] = None,
        stop_flag: Optional[callable] = None
    ) -> OptimizationResult:
        """Run one ADAM optimization step."""
        if self.params_tensor is None:
            raise RuntimeError("Optimizer not initialized. Call initialize() first.")
        
        print(f"\n{'='*60}")
        print(f"ADAM Iteration {self.iteration}")
        print(f"Workers: {self.get_worker_count()}")
        print(f"Fitness Preset: {fitness_config.preset}")
        print(f"{'='*60}")
        
        # Compute gradient via finite differences (with progress updates)
        gradient, fitness, metrics = self._compute_gradient_finite_diff(
            data,
            fitness_config,
            progress_callback=progress_callback,
            stop_flag=stop_flag
        )
        
        # Clip gradient norm for stability
        grad_norm = torch.norm(gradient).item()
        if grad_norm > self.max_grad_norm:
            gradient = gradient * (self.max_grad_norm / grad_norm)
            print(f"⚠ Clipped gradient norm: {grad_norm:.4f} → {self.max_grad_norm:.4f}")
        else:
            print(f"Gradient norm: {grad_norm:.4f}")
        
        # Zero previous gradients
        self.optimizer.zero_grad()
        
        # Manually set gradient
        self.params_tensor.grad = gradient
        
        # ADAM step
        self.optimizer.step()
        
        # Clamp parameters to [0, 1] after update
        with torch.no_grad():
            self.params_tensor.clamp_(0, 1)
        
        # Update best if improved
        current_vector = self.params_tensor.detach().cpu().numpy()
        current_seller, current_backtest = self._vector_to_params(current_vector)
        
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_seller_params = deepcopy(current_seller)
            self.best_backtest_params = deepcopy(current_backtest)
            self.best_metrics = metrics.copy()
            print(f"🌟 NEW BEST: Fitness={fitness:.4f}")
        
        # Record history
        self.history.append({
            'iteration': self.iteration,
            'fitness': fitness,
            'best_fitness': self.best_fitness,
            'grad_norm': grad_norm,
            'n_trades': metrics.get('n', 0),
            'win_rate': metrics.get('win_rate', 0.0)
        })
        
        # Show summary
        print(f"\n📊 Summary:")
        print(f"   Fitness: {fitness:.4f}")
        print(f"   Best Fitness: {self.best_fitness:.4f}")
        print(f"   Trades: {metrics.get('n', 0)}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
        
        self.iteration += 1
        
        return OptimizationResult(
            best_seller_params=deepcopy(self.best_seller_params),
            best_backtest_params=deepcopy(self.best_backtest_params),
            fitness=self.best_fitness,
            metrics=self.best_metrics.copy(),
            iteration=self.iteration,
            additional_info={
                'grad_norm': grad_norm,
                'current_fitness': fitness,
                'improvement': fitness > self.best_fitness
            }
        )
    
    def get_best_params(self):
        """Return best parameters found so far."""
        return (
            deepcopy(self.best_seller_params) if self.best_seller_params else None,
            deepcopy(self.best_backtest_params) if self.best_backtest_params else None,
            self.best_fitness
        )
    
    def get_stats(self) -> dict:
        """Return current optimization statistics."""
        stats = {
            'iteration': self.iteration,
            'best_fitness': self.best_fitness,
        }
        
        if self.params_tensor is not None:
            stats['param_l2_norm'] = torch.norm(self.params_tensor).item()
        
        return stats
    
    def get_history(self) -> list:
        """Return optimization history."""
        return self.history.copy()
    
    def get_optimizer_name(self) -> str:
        """Return optimizer name."""
        return "ADAM"
    
    def get_worker_count(self) -> int:
        """Return configured worker count."""
        return int(self.n_workers)
