from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QGridLayout, QPushButton,
    QFileDialog, QScrollArea, QSpinBox, QDoubleSpinBox, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QMetaObject, Q_ARG, Slot
from PySide6.QtGui import QColor
import pandas as pd
import pyqtgraph as pg
from dataclasses import asdict
from typing import Optional
import asyncio
import threading
import time
from copy import deepcopy

from strategy.seller_exhaustion import SellerParams, build_features
from core.models import BacktestParams, FitnessConfig, OptimizationConfig
from backtest.optimizer import Individual
from backtest.engine import run_backtest
import config.settings as config_settings
from core.logging_utils import get_logger
import os

# New modular optimizer system
from backtest.optimizer_base import BaseOptimizer
from backtest.optimizer_factory import (
    create_optimizer,
    get_available_optimizers,
    get_optimizer_display_name,
)

# Evolution Coach integration
from backtest.coach_manager_openai import OpenAICoachManager
from backtest.coach_classic import ClassicCoachManager
from backtest.coach_protocol import CoachAnalysis

import multiprocessing


logger = get_logger(__name__)


class StatsPanel(QWidget):
    """Comprehensive statistics panel with parameter optimization."""
    
    # Signal emitted when optimization step completes with backtest results
    optimization_step_complete = Signal(Individual, dict)  # (best_individual, backtest_result)
    
    # Signal emitted during multi-step optimization for progress tracking
    progress_updated = Signal(int, int, str)  # (current, total, message)
    
    # Signal emitted after generation completes (for thread-safe UI updates)
    generation_complete = Signal()
    
    # Signal emitted to reset progress bar (thread-safe)
    reset_progress_bar = Signal()
    
    # Signal emitted to hide progress bar immediately (thread-safe)
    hide_progress_bar = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.trades_df = None
        self.metrics = None
        self.optimizer: BaseOptimizer = None  # Current optimizer instance
        self.current_data = None  # Features dataframe (with indicators and signals)
        self.raw_data = None  # Raw OHLCV data (for rebuilding features)
        self.current_tf = None
        self.param_editor = None  # Will be set by main window
        
        # Multi-step optimization state
        self.is_optimizing = False
        self.stop_requested = False
        
        # Temporary storage for thread results
        self.temp_backtest_result = None
        
        # Track previous best fitness to detect improvements
        self.prev_best_fitness = None
        
        # Optional: path to an initial population JSON file
        self.initial_population_file: str | None = None
        
        # Track if we've already logged initialization to prevent duplicates
        self._initialization_logged = False
        self._logged_optimizer_ids = set()  # Track logged optimizer instances by ID
        self._last_init_signature: tuple | None = None
        self._last_init_signature_ts: float = 0.0  # Monotonic timestamp
        
        # Evolution Coach integration (OpenAI Agents or Classic coach)
        self.coach_manager: Optional[OpenAICoachManager] = None
        self.status_callback: callable = None  # Callback to update main window status bar
        self.coach_window: Optional['EvolutionCoachWindow'] = None  # Reference to coach window
        
        self.init_ui()
        
        # Connect generation_complete signal for thread-safe UI updates
        self.generation_complete.connect(self._update_after_generation)
        
        # Connect reset_progress_bar signal for thread-safe UI updates
        self.reset_progress_bar.connect(self._reset_progress_bar)
        
        # Connect hide_progress_bar signal for thread-safe UI updates
        self.hide_progress_bar.connect(self._hide_progress_bar)
        
        # Initialize coach manager
        self._initialize_coach_manager()

    def set_initial_population_file(self, path: str | None):
        """Set a population file to initialize the optimizer from (one-time)."""
        self.initial_population_file = path

    def set_coach_window(self, coach_window):
        """Set the Evolution Coach window for real-time updates."""
        self.coach_window = coach_window
        # Update status callback to also update coach window
        if self.coach_window:
            # Store original callback to avoid recursion
            if not hasattr(self, '_original_status_callback'):
                self._original_status_callback = self.status_callback
            self.status_callback = self._create_coach_status_callback()

        # Update coach manager with coach window and new status callback
        if self.coach_manager:
            self.coach_manager.coach_window = coach_window
            # Update coach manager's status callback to use the new combined callback
            if hasattr(self.coach_manager, 'status_callback'):
                self.coach_manager.status_callback = self.status_callback

    def set_classic_coach_window(self, classic_coach_window):
        """Set the Classic Coach window for deterministic decision visualization."""
        print(f"🔍 DEBUG: set_classic_coach_window called with classic_coach_window={classic_coach_window}")
        self.classic_coach_window = classic_coach_window
        
        # Create combined status callback for Classic Coach (same as Evolution Coach)
        if classic_coach_window:
            # Store original callback to avoid recursion
            if not hasattr(self, '_original_status_callback'):
                self._original_status_callback = self.status_callback
            self.status_callback = self._create_coach_status_callback()
            print(f"🔍 DEBUG: set_classic_coach_window created status_callback: {self.status_callback}")
        
        # Update coach manager with classic coach window
        if self.coach_manager and hasattr(self.coach_manager, 'set_classic_coach_window'):
            self.coach_manager.set_classic_coach_window(classic_coach_window)
        
        # Set coach manager reference in classic coach window
        if self.coach_manager and hasattr(classic_coach_window, 'set_coach_manager'):
            classic_coach_window.set_coach_manager(self.coach_manager)
        
        # Update coach manager's status callback to use the new combined callback
        if self.coach_manager and hasattr(self.coach_manager, 'status_callback'):
            self.coach_manager.status_callback = self.status_callback
        
        # Create a combined status callback that updates both main window and coach tool status
        if not hasattr(self, '_combined_status_callback'):
            def combined_callback(tool_name: str, reason: str = ""):
                # Update main window status bar
                if hasattr(self, '_original_status_callback') and self._original_status_callback:
                    try:
                        self._original_status_callback(tool_name, reason)
                    except Exception as e:
                        print(f"Error in original status callback: {e}")
                
                # Update coach tool status for progress bar (only for OpenAI Agent, not Classic Coach)
                if tool_name.startswith('OpenAI_') and hasattr(self, '_coach_tool_status_callback') and self._coach_tool_status_callback:
                    try:
                        self._coach_tool_status_callback(tool_name, reason)
                    except Exception as e:
                        print(f"Error in coach tool status callback: {e}")
            
            self._combined_status_callback = combined_callback
            self.status_callback = self._combined_status_callback
            
            # Update coach manager's status callback to use the new combined callback
            if self.coach_manager and hasattr(self.coach_manager, 'status_callback'):
                self.coach_manager.status_callback = self.status_callback

    def _create_coach_status_callback(self):
        """Create simple status callback that updates coach window."""
        def status_callback(tool_name: str, reason: str = ""):
            """Simple callback for coach tool status - just update UI."""
            try:
                # Update coach window if available
                if hasattr(self, 'coach_window') and self.coach_window:
                    self.coach_window.add_tool_call(
                        tool_name=tool_name,
                        parameters={},
                        response={},
                        reason=reason
                    )
                    self.coach_window.set_status(f"Tool: {tool_name}", is_analyzing=True)
                
                # Update progress bar in compact params editor
                if hasattr(self, 'param_editor') and self.param_editor:
                    self.param_editor.set_coach_analyzing(True)
            
            except Exception as e:
                logger.error(f"Coach status callback error: {e}")
                import traceback
                traceback.print_exc()
        
        return status_callback

    # -------- Evolution Coach Integration --------
    def _initialize_coach_manager(self):
        """Initialize Evolution Coach manager based on selected mode."""
        try:
            from config.settings import settings

            # Coach must be enabled
            if not settings.coach_enabled:
                logger.info("Evolution Coach disabled in settings")
                self.coach_manager = None
                return

            # Get coach mode from settings (default to 'classic')
            coach_mode = getattr(settings, 'coach_mode', 'classic')
            print(f"🔧 Initializing coach with mode: {coach_mode}")

            if coach_mode == 'classic':
                # Initialize Classic Coach (deterministic, no API keys needed)
                self.coach_manager = ClassicCoachManager(
                        analysis_interval=settings.coach_analysis_interval,
                        verbose=True,
                        total_iterations=getattr(settings, 'optimizer_iterations', 200),
                        enable_islands=getattr(settings, 'coach_islands_enabled', False),
                        enable_fitness_tuning=True,
                        enable_bounds_expansion=True,
                        enable_immigration=True,
                        enable_algorithm_switching=True
                    )
                logger.info("✓ Classic Coach initialized (deterministic mode)")
                
                # Set coach manager reference in classic coach window if it exists
                if hasattr(self, 'classic_coach_window') and self.classic_coach_window:
                    if hasattr(self.classic_coach_window, 'set_coach_manager'):
                        self.classic_coach_window.set_coach_manager(self.coach_manager)
                
                # Ensure status_callback is set for Classic Coach even if window is not opened
                if not hasattr(self, '_original_status_callback'):
                    self._original_status_callback = self.status_callback
                self.status_callback = self._create_coach_status_callback()
                print(f"🔍 DEBUG: _initialize_coach_manager created status_callback for Classic Coach: {self.status_callback}")
                
                # Update UI status bar if callback provided
                if self.status_callback:
                    self.status_callback(f"✓ 🧠 Classic Coach ready (every {self.coach_manager.analysis_interval} gens)")

            elif coach_mode == 'openai':
                # Initialize OpenAI Coach (requires API keys)
                provider = getattr(settings, 'agent_provider', 'novita')
                api_key_configured = False

                if provider == "openai":
                    api_key = getattr(settings, 'openai_api_key', '')
                    if api_key:
                        api_key_configured = True
                    else:
                        logger.warning("⚠ OpenAI API key not configured - Coach unavailable")
                elif provider == "openrouter":
                    api_key = getattr(settings, 'openrouter_api_key', '')
                    if api_key:
                        api_key_configured = True
                    else:
                        logger.warning("⚠ OpenRouter API key not configured - Coach unavailable")
                elif provider == "novita":
                    api_key = getattr(settings, 'novita_api_key', '')
                    if api_key:
                        api_key_configured = True
                    else:
                        logger.warning("⚠ Novita API key not configured - Coach unavailable")

                if not api_key_configured:
                    self.coach_manager = None
                    return

                self.coach_manager = OpenAICoachManager(
                    analysis_interval=settings.coach_analysis_interval,
                    auto_apply=True,
                    verbose=True,
                    openrouter_api_key=getattr(settings, 'openrouter_api_key', ''),
                    openrouter_model=getattr(settings, 'openrouter_model', 'anthropic/claude-3.5-sonnet'),
                    status_callback=self._coach_tool_status_callback,
                    coach_window=getattr(self, 'coach_window', None)
                )
                # Store the original status callback for coach manager
                # (This line is redundant but kept for clarity)
                # Update UI status bar if callback provided
                if self.status_callback:
                    self.status_callback(f"✓ 🤖 OpenAI Coach ready ({provider.title()}, every {self.coach_manager.analysis_interval} gens)")
            else:
                logger.warning(f"Unknown coach mode: {coach_mode}")
                self.coach_manager = None

        except Exception as e:
            logger.exception("Failed to initialize coach manager: %s", e)
            logger.warning("⚠ Evolution Coach unavailable - continuing without coach")
            self.coach_manager = None

    # -------- DEAD CODE REMOVED --------
    # Old non-blocking coach methods (_schedule_coach_analysis, 
    # _handle_coach_analysis_complete, _coach_log, _apply_coach_updates) 
    # have been removed. Using BlockingCoachManager instead.
    
    # -------- Auto Export Helpers --------
    def _get_process_id(self) -> int:
        try:
            return os.getpid()
        except Exception:
            return 0

    def _auto_export_population(self) -> None:
        """Export current optimizer population to populations/<pid>.json if available."""
        try:
            if self.optimizer is None:
                return
            population = getattr(self.optimizer, "population", None)
            if population is None:
                logger.info("No population to export (optimizer has no population attribute)")
                return
            pid = self._get_process_id()
            out_dir = os.path.join(os.getcwd(), "populations")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{pid}.json")
            from backtest.optimizer import export_population
            export_population(population, out_path)
            logger.info("💾 Exported population to %s", out_path)
        except Exception as e:
            logger.exception("Failed to auto-export population: %s", e)

    @staticmethod
    def _format_dict_for_log(payload: dict) -> str:
        """Return deterministic key-sorted string representation for logging."""
        return ", ".join(f"{key}={payload[key]}" for key in sorted(payload))

    @staticmethod
    def _format_dict_compact(payload: dict, float_dp: int = 4) -> str:
        """Deterministic, compact key/value list with basic float rounding."""
        def _fmt(v):
            if isinstance(v, float):
                return f"{v:.{float_dp}f}"
            return v
        return ", ".join(f"{k}={_fmt(payload[k])}" for k in sorted(payload))

    @staticmethod
    def _compact_seller_params(p: SellerParams) -> str:
        return (
            f"ema_f={p.ema_fast} ema_s={p.ema_slow} "
            f"vol_z={p.vol_z:.2f} tr_z={p.tr_z:.2f} cloc={p.cloc_min:.2f}"
        )

    @staticmethod
    def _compact_backtest_params(p: BacktestParams) -> str:
        return (
            f"fee={p.fee_bp:.1f} slip={p.slippage_bp:.1f} "
            f"swing_lb={p.fib_swing_lookback} swing_la={p.fib_swing_lookahead} "
            f"target={p.fib_target_level:.3f}"
        )

    @staticmethod
    def _compact_metrics(metrics: dict) -> str:
        if not metrics:
            return "--"
        trades = metrics.get("n", "--")

        def _fmt_pct(value):
            return f"{value:.2%}" if isinstance(value, (int, float)) else "--"

        def _fmt_float(value, precision=2):
            return f"{value:.{precision}f}" if isinstance(value, (int, float)) else "--"

        win = _fmt_pct(metrics.get("win_rate"))
        avg_r = _fmt_float(metrics.get("avg_R"))
        pnl = _fmt_float(metrics.get("total_pnl"), precision=4)
        dd_value = metrics.get("max_drawdown", metrics.get("max_dd"))
        dd = _fmt_float(dd_value, precision=4)

        return f"n={trades} win={win} avgR={avg_r} pnl={pnl} dd={dd}"

    def _log_parameter_snapshot(
        self,
        seller_params: SellerParams,
        backtest_params: BacktestParams,
        prefix: str = ""
    ) -> None:
        """Log seller/backtest parameter values in a compact format."""
        seller_payload = asdict(seller_params)
        backtest_payload = (
            backtest_params.model_dump()
            if hasattr(backtest_params, "model_dump")
            else dict(backtest_params)
        )

        logger.info("%sSellerParams: %s", prefix, self._format_dict_for_log(seller_payload))
        logger.info("%sBacktestParams: %s", prefix, self._format_dict_for_log(backtest_payload))

    @staticmethod
    def _log_metrics_snapshot(metrics: dict, prefix: str = "") -> None:
        """Log backtest metrics with consistent formatting."""
        if not metrics:
            return

        trades = metrics.get("n") or metrics.get("total_trades")
        win_rate = metrics.get("win_rate")
        avg_r = metrics.get("avg_R")
        total_pnl = metrics.get("total_pnl")
        max_dd = metrics.get("max_drawdown") or metrics.get("max_dd")
        sharpe = metrics.get("sharpe")

        logger.info(
            "%sMetrics | trades=%s | win_rate=%s | avg_R=%s | total_pnl=%s | max_dd=%s | sharpe=%s",
            prefix,
            trades if trades is not None else "--",
            f"{win_rate:.2%}" if isinstance(win_rate, (int, float)) else "--",
            f"{avg_r:.2f}" if isinstance(avg_r, (int, float)) else "--",
            f"{total_pnl:.4f}" if isinstance(total_pnl, (int, float)) else "--",
            f"{max_dd:.4f}" if isinstance(max_dd, (int, float)) else "--",
            f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "--",
        )

    @staticmethod
    def _log_fitness_config(fitness_config, prefix: str = "") -> None:
        """Log fitness configuration weights/thresholds."""
        if fitness_config is None:
            return
        if hasattr(fitness_config, "model_dump"):
            payload = fitness_config.model_dump()
        else:
            payload = getattr(fitness_config, "__dict__", {})
        if not payload:
            return
        items = ", ".join(
            f"{k}={payload[k]:.4f}" if isinstance(payload[k], float) else f"{k}={payload[k]}"
            for k in sorted(payload)
        )
        logger.info("%sFitnessConfig: %s", prefix, items)

    def _log_optimizer_config(self, prefix: str = "") -> None:
        """Log optimizer type, worker usage, and hyperparameters."""
        if self.optimizer is None:
            return

        try:
            worker_info = ""
            if hasattr(self.optimizer, "get_worker_count"):
                try:
                    worker_info = f" | workers={self.optimizer.get_worker_count()}"
                except Exception:
                    worker_info = ""
            logger.info(
                "%sOptimizer: name=%s%s",
                prefix,
                self.optimizer.get_optimizer_name(),
                worker_info,
            )
        except Exception:
            logger.info("%sOptimizer: %s", prefix, type(self.optimizer).__name__)

        config_payload = getattr(self.optimizer, "config", None)
        if isinstance(config_payload, dict) and config_payload:
            logger.info(
                "%sOptimizer hyperparameters: %s",
                prefix,
                self._format_dict_for_log(config_payload),
            )

        population = getattr(self.optimizer, "population", None)
        if population is not None:
            logger.info(
                "%sPopulation status: size=%s | generation=%s",
                prefix,
                getattr(population, "size", "--"),
                getattr(population, "generation", "--"),
            )

    def _initialize_optimizer_if_needed(self):
        """Initialize optimizer with current parameters as seed (called automatically)."""
        if self.current_data is None:
            logger.error("No data loaded.")
            return False
        
        # Get optimizer type from UI and desired worker count from settings
        optimizer_type = self.optimizer_type_combo.currentData()
        try:
            worker_count = int(getattr(config_settings.settings, 'optimizer_workers', multiprocessing.cpu_count()))
        except Exception:
            worker_count = multiprocessing.cpu_count()
        worker_count = max(1, worker_count)
        
        # If optimizer already exists with same type/workers, reuse it
        if self.optimizer is not None:
            same_type = (
                self.optimizer.get_optimizer_name().lower().replace(' ', '_') == optimizer_type
                or (optimizer_type == 'evolutionary' and 'evolutionary' in self.optimizer.get_optimizer_name().lower())
            )
            same_workers = True
            if hasattr(self.optimizer, "get_worker_count"):
                try:
                    same_workers = int(self.optimizer.get_worker_count()) == worker_count
                except Exception:
                    same_workers = False
            if same_type and same_workers:
                logger.info("Reusing existing optimizer (%s)", self.optimizer.get_optimizer_name())
                # Don't log to coach manager when reusing - prevents duplicate logs
                return True
        
        # Get current params from UI as seed
        seller_params, backtest_params, _ = self.get_current_params()
        
        # Create new optimizer via factory
        try:
            self.optimizer = create_optimizer(
                optimizer_type=optimizer_type,
                n_workers=worker_count,
                # If provided, initialize from file (supported by EvolutionaryOptimizer)
                initial_population_file=self.initial_population_file,
            )
            
            # Initialize with seed parameters
            self.optimizer.initialize(
                seed_seller_params=seller_params,
                seed_backtest_params=backtest_params,
                timeframe=self.current_tf
            )
            # Clear after use so subsequent re-inits don't re-use it inadvertently
            self.initial_population_file = None
            
            # Reset best fitness tracking
            self.prev_best_fitness = None
            
            logger.info(
                "Optimizer initialized | type=%s | workers=%s",
                self.optimizer.get_optimizer_name(),
                worker_count,
            )
            self._log_parameter_snapshot(seller_params, backtest_params, prefix="Seed ")
            self._log_optimizer_config(prefix="  ")
            
            # Update dropdown to show active optimizer status
            self.update_optimizer_dropdown_status()
            
            # Log initialization to coach manager (only once per optimizer instance)
            # Use optimizer object ID as unique identifier to prevent duplicates
            optimizer_id = id(self.optimizer)
            
            if optimizer_id not in self._logged_optimizer_ids:
                signature = (
                    self.optimizer.get_optimizer_name(),
                    worker_count,
                    self._compact_seller_params(seller_params),
                    self._compact_backtest_params(backtest_params),
                )
                now = time.monotonic()
                last_ts = getattr(self, "_last_init_signature_ts", 0.0)
                is_duplicate_signature = (
                    self._last_init_signature == signature
                    and (now - last_ts) < 2.0
                )

                if is_duplicate_signature:
                    logger.debug(
                        "Skipping duplicate Evolution Coach init log (signature=%s, Δt=%.3fs)",
                        signature,
                        now - last_ts,
                    )
                else:
                    print(
                        f"OPT init name={self.optimizer.get_optimizer_name()} workers={worker_count}"
                    )
                    # Optimizer hyperparameters
                    hp = getattr(self.optimizer, 'config', {}) or {}
                    if isinstance(hp, dict) and hp:
                        print(f"OPT hp {self._format_dict_compact(hp, float_dp=3)}")
                    # Population status
                    pop = getattr(self.optimizer, 'population', None)
                    if pop is not None:
                        print(
                            f"POP status size={getattr(pop, 'size', '--')} gen={getattr(pop, 'generation', '--')}"
                        )
                    print(
                        f"SEED seller[{self._compact_seller_params(seller_params)}]"
                    )
                    print(
                        f"SEED backtest[{self._compact_backtest_params(backtest_params)}]"
                    )
                    # Full seed dumps for agent completeness
                    from dataclasses import asdict as _asdict
                    full_seller = _asdict(seller_params)
                    full_backtest = (
                        backtest_params.model_dump() if hasattr(backtest_params, 'model_dump') else dict(backtest_params)
                    )
                    print(f"SEED SellerParams: {self._format_dict_compact(full_seller)}")
                    print(f"SEED BacktestParams: {self._format_dict_compact(full_backtest)}")
                    self._last_init_signature = signature
                    self._last_init_signature_ts = now
                self._logged_optimizer_ids.add(optimizer_id)
            
            return True
            
        except Exception as e:
            logger.exception("Error initializing optimizer: %s", e)
            return False
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Metrics section
        self.metrics_group = self.create_metrics_section()
        layout.addWidget(self.metrics_group)
        
        # Equity curve
        self.equity_group = self.create_equity_section()
        layout.addWidget(self.equity_group)
        
        # Optimization controls
        self.optimization_group = self.create_optimization_section()
        layout.addWidget(self.optimization_group)
        
        # Initialize dropdown status
        self.update_optimizer_dropdown_status()
    
    def create_metrics_section(self):
        """Create comprehensive metrics display."""
        group = QGroupBox("Performance Metrics")
        layout = QGridLayout()
        
        # Initialize metric labels
        self.metric_labels = {}
        
        metrics_config = [
            # Row 0
            ("Total Trades", "n", 0, 0),
            ("Win Rate", "win_rate", 0, 1),
            ("Avg R-Multiple", "avg_R", 0, 2),
            
            # Row 1
            ("Total PnL", "total_pnl", 1, 0),
            ("Avg Win", "avg_win", 1, 1),
            ("Avg Loss", "avg_loss", 1, 2),
            
            # Row 2
            ("Profit Factor", "profit_factor", 2, 0),
            ("Max Drawdown", "max_drawdown", 2, 1),
            ("Sharpe Ratio", "sharpe", 2, 2),
        ]
        
        for label_text, key, row, col in metrics_config:
            # Label
            label = QLabel(f"{label_text}:")
            label.setProperty("variant", "secondary")
            layout.addWidget(label, row * 2, col)
            
            # Value
            value = QLabel("--")
            value.setStyleSheet("font-size: 16px; font-weight: bold;")
            layout.addWidget(value, row * 2 + 1, col)
            
            self.metric_labels[key] = value
        
        group.setLayout(layout)
        return group
    
    def create_equity_section(self):
        """Create equity curve and fitness evolution charts."""
        group = QGroupBox("Performance Tracking")
        layout = QVBoxLayout()
        
        # Equity curve
        equity_label = QLabel("<b>Equity Curve</b>")
        layout.addWidget(equity_label)
        
        self.equity_plot = pg.PlotWidget()
        self.equity_plot.setBackground('#0f1a12')
        self.equity_plot.setLabel('left', 'Cumulative PnL', color='#e8f5e9')
        self.equity_plot.setLabel('bottom', 'Trade Number', color='#e8f5e9')
        self.equity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.equity_plot.setMinimumHeight(150)
        
        layout.addWidget(self.equity_plot)
        
        # Fitness evolution plot (no label - Performance Tracking section header is above)
        # fitness_label = QLabel("")
        # layout.addWidget(fitness_label)
        
        self.fitness_plot = pg.PlotWidget()
        self.fitness_plot.setBackground('#0f1a12')
        self.fitness_plot.setLabel('left', 'Fitness Score', color='#e8f5e9')
        self.fitness_plot.setLabel('bottom', 'Generation', color='#e8f5e9')
        self.fitness_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fitness_plot.setMinimumHeight(150)
        
        layout.addWidget(self.fitness_plot)
        
        group.setLayout(layout)
        return group
    
    def set_param_editor(self, param_editor):
        """Set external parameter editor reference."""
        self.param_editor = param_editor
    
    def set_chart_view(self, chart_view):
        """Set external chart view reference for status updates."""
        self.chart_view = chart_view
    
    def _coach_tool_status_callback(self, tool_name: str, reason: str = ""):
        """Callback for coach tool status updates."""
        # print(f"🔍 DEBUG: _coach_tool_status_callback called with tool_name='{tool_name}', reason='{reason}'")
        if hasattr(self, 'chart_view') and hasattr(self.chart_view, 'set_coach_tool_status'):
            # print(f"🔍 DEBUG: Calling chart_view.set_coach_tool_status")
            self.chart_view.set_coach_tool_status(tool_name, reason)
        else:
            print(f"🔍 DEBUG: chart_view not available or missing set_coach_tool_status method")
        
        # Also update progress bar in compact params editor
        if hasattr(self, 'param_editor') and self.param_editor:
            try:
                # Show progress bar for analysis start, hide for completion
                if tool_name.startswith('✅ Coach completed'):
                    # Hide progress bar immediately using Signal
                    self.hide_progress_bar.emit()
                    # print(f"🔍 DEBUG: Hiding progress bar - coach analysis completed")
                else:
                    self.param_editor.set_coach_analyzing(True)
                    # print(f"🔍 DEBUG: Showing progress bar - coach analysis started")
            except Exception as e:
                print(f"Error updating coach progress bar: {e}")
        
        # No need to schedule progress bar reset - it's handled immediately by Signal
    
    def _reset_progress_bar(self):
        """Reset progress bar to normal state."""
        if hasattr(self, 'param_editor') and self.param_editor:
            try:
                self.param_editor.set_coach_analyzing(False)
            except Exception as e:
                print(f"Error resetting coach progress bar: {e}")
    
    def _hide_progress_bar(self):
        """Hide progress bar immediately."""
        if hasattr(self, 'param_editor') and self.param_editor:
            try:
                self.param_editor.set_coach_analyzing(False)
                # print(f"🔍 DEBUG: Progress bar hidden immediately via Signal")
            except Exception as e:
                print(f"Error hiding coach progress bar: {e}")
    
    def reinitialize_coach_manager(self):
        """Reinitialize coach manager after settings changes."""
        # Reset coach manager to force re-initialization with new settings
        self.coach_manager = None
        self._initialize_coach_manager()

        # If coach manager supports dynamic interval updates, apply the new interval
        if self.coach_manager and hasattr(self.coach_manager, 'update_analysis_interval'):
            from config.settings import settings
            new_interval = getattr(settings, 'coach_analysis_interval', 5)
            self.coach_manager.update_analysis_interval(new_interval)
            logger.info(f"Updated coach analysis interval to {new_interval}")
    
    def create_optimization_section(self):
        """Create optimization controls with modular optimizer selection."""
        group = QGroupBox("Strategy Optimization")
        layout = QVBoxLayout()
        
        # Optimizer Type Selection
        optimizer_layout = QHBoxLayout()
        optimizer_layout.addWidget(QLabel("Optimization:"))
        self.optimizer_type_combo = self._create_optimizer_type_combo()
        self.optimizer_type_combo.currentIndexChanged.connect(self._on_optimizer_type_changed)
        optimizer_layout.addWidget(self.optimizer_type_combo, stretch=1)
        layout.addLayout(optimizer_layout)
        
        # Acceleration is configured in Settings → ⚡ Acceleration
        
        # Stop button and fitness preset dropdown (in one row)
        preset_stop_layout = QHBoxLayout()
        
        # Stop button on the left
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setToolTip("Stop ongoing optimization (keeps progress)")
        self.stop_btn.setEnabled(False)  # Disabled until optimization starts
        self.stop_btn.clicked.connect(self.stop_optimization)
        preset_stop_layout.addWidget(self.stop_btn, stretch=1)
        
        # Fitness preset combo on the right (without label)
        self.fitness_preset_combo = self._create_fitness_preset_combo()
        self.fitness_preset_combo.currentIndexChanged.connect(self._on_fitness_preset_changed)
        preset_stop_layout.addWidget(self.fitness_preset_combo, stretch=2)
        
        layout.addLayout(preset_stop_layout)
        
        # Optimize button at the bottom
        self.optimize_btn = QPushButton("🚀 Optimize")
        self.optimize_btn.setObjectName("primaryButton")
        self.optimize_btn.setToolTip("Run optimization (auto-initializes optimizer)")
        self.optimize_btn.clicked.connect(self.run_multi_step_optimize)
        layout.addWidget(self.optimize_btn)
        
        # Note: Parameters are auto-applied when a better solution is found during optimization
        
        group.setLayout(layout)
        return group
    
    def _create_optimizer_type_combo(self):
        """Create optimizer type dropdown."""
        from PySide6.QtWidgets import QComboBox
        
        combo = QComboBox()
        combo.addItem("🧬 Evolutionary Algorithm", "evolutionary")
        combo.addItem("🎯 ADAM", "adam")
        combo.setToolTip("Select optimization algorithm")
        combo.setCurrentIndex(0)  # Default to Evolutionary
        
        return combo
    
    # Worker count (parallelism) is configured via the Settings dialog
   
    def _on_optimizer_type_changed(self, index):
        """Handle optimizer type change."""
        optimizer_type = self.optimizer_type_combo.currentData()
        
        # Reset optimizer (will be re-initialized on next run)
        self.optimizer = None
        
        # Update dropdown to show no optimizer is active
        self.update_optimizer_dropdown_status()
    
    def update_optimizer_dropdown_status(self):
        """Update optimizer dropdown to show current active optimizer status."""
        if self.optimizer is None:
            # No optimizer active - show as selection dropdown
            self.optimizer_type_combo.setEnabled(True)
            self.optimizer_type_combo.setStyleSheet("""
                QComboBox {
                    background-color: #2b2b2b;
                    color: #e8f5e9;
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 4px 8px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid #e8f5e9;
                    margin-right: 5px;
                }
            """)
        else:
            # Optimizer is active - show as status indicator
            optimizer_name = self.optimizer.get_optimizer_name()

            # Map optimizer names to dropdown data values
            if optimizer_name == "Evolutionary Algorithm":
                optimizer_type = "evolutionary"
            elif optimizer_name == "ADAM":
                optimizer_type = "adam"
            else:
                optimizer_type = "evolutionary"  # fallback

            # Find the correct index for the active optimizer
            for i in range(self.optimizer_type_combo.count()):
                if self.optimizer_type_combo.itemData(i) == optimizer_type:
                    self.optimizer_type_combo.setCurrentIndex(i)
                    break

            # Disable dropdown and change style to show status
            self.optimizer_type_combo.setEnabled(False)
            self.optimizer_type_combo.setStyleSheet("""
                QComboBox {
                    background-color: white;
                    color: black;
                    border: 2px solid #4caf50;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid #666;
                    margin-right: 5px;
                }
            """)

        # Log the current optimizer status
        if self.optimizer is not None:
            optimizer_name = self.optimizer.get_optimizer_name()
            logger.info("Optimizer active: %s", optimizer_name)
        else:
            logger.info("No optimizer active")
    
    # Acceleration change handled via Settings tab
    
    def _create_fitness_preset_combo(self):
        """Create fitness preset dropdown combo box."""
        from PySide6.QtWidgets import QComboBox
        
        combo = QComboBox()
        combo.addItem("⚖️ Balanced", "balanced")
        combo.addItem("🚀 High Frequency", "high_frequency")
        combo.addItem("🛡️ Conservative", "conservative")
        combo.addItem("💰 Profit Focused", "profit_focused")
        combo.addItem("✏️ Custom", "custom")
        combo.setToolTip("Select fitness function preset - controls weight values in Best Parameters panel")
        combo.setCurrentIndex(0)  # Default to Balanced
        
        return combo
    
    def _on_fitness_preset_changed(self, index):
        """Handle fitness preset dropdown change - update param editor."""
        if self.param_editor:
            preset_name = self.fitness_preset_combo.currentData()
            self.param_editor.load_fitness_preset(preset_name)
            logger.info("Fitness preset changed to: %s", preset_name)
    
    def update_stats(self, backtest_result):
        """Update all statistics displays with backtest results."""
        try:
            self.trades_df = backtest_result['trades']
            self.metrics = backtest_result['metrics']
            
            # Update metrics
            self.update_metrics()
            
            # Update equity curve
            self.update_equity_curve()
            
        except Exception as e:
            logger.exception("Error updating stats: %s", e)
    
    def update_metrics(self):
        """Update metric labels."""
        if not self.metrics:
            return
        
        # Total trades
        self.metric_labels['n'].setText(str(self.metrics['n']))
        
        # Win rate
        win_rate = self.metrics['win_rate']
        self.metric_labels['win_rate'].setText(f"{win_rate:.1%}")
        if win_rate >= 0.5:
            self.metric_labels['win_rate'].setStyleSheet("font-size: 16px; font-weight: bold; color: #4caf50;")
        else:
            self.metric_labels['win_rate'].setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
        
        # Avg R
        avg_r = self.metrics['avg_R']
        self.metric_labels['avg_R'].setText(f"{avg_r:.2f}R")
        if avg_r > 0:
            self.metric_labels['avg_R'].setStyleSheet("font-size: 16px; font-weight: bold; color: #4caf50;")
        else:
            self.metric_labels['avg_R'].setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
        
        # Total PnL
        total_pnl = self.metrics['total_pnl']
        self.metric_labels['total_pnl'].setText(f"${total_pnl:.4f}")
        if total_pnl > 0:
            self.metric_labels['total_pnl'].setStyleSheet("font-size: 16px; font-weight: bold; color: #4caf50;")
        else:
            self.metric_labels['total_pnl'].setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
        
        # Avg win/loss (if available)
        if 'avg_win' in self.metrics:
            self.metric_labels['avg_win'].setText(f"${self.metrics['avg_win']:.4f}")
            self.metric_labels['avg_loss'].setText(f"${self.metrics['avg_loss']:.4f}")
        else:
            wins = self.trades_df[self.trades_df['pnl'] > 0]
            losses = self.trades_df[self.trades_df['pnl'] <= 0]
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
            avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0.0
            self.metric_labels['avg_win'].setText(f"${avg_win:.4f}")
            self.metric_labels['avg_loss'].setText(f"${avg_loss:.4f}")
        
        # Profit factor (if available)
        if 'profit_factor' in self.metrics:
            pf = self.metrics['profit_factor']
            if pf == float('inf'):
                self.metric_labels['profit_factor'].setText("∞")
            else:
                self.metric_labels['profit_factor'].setText(f"{pf:.2f}")
        else:
            wins = self.trades_df[self.trades_df['pnl'] > 0]
            losses = self.trades_df[self.trades_df['pnl'] <= 0]
            total_wins = wins['pnl'].sum() if len(wins) > 0 else 0.0
            total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0.0
            pf = total_wins / total_losses if total_losses > 0 else float('inf')
            if pf == float('inf'):
                self.metric_labels['profit_factor'].setText("∞")
            else:
                self.metric_labels['profit_factor'].setText(f"{pf:.2f}")
        
        # Max drawdown
        max_dd = self.metrics.get('max_dd', 0.0)
        self.metric_labels['max_drawdown'].setText(f"${max_dd:.4f}")
        self.metric_labels['max_drawdown'].setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
        
        # Sharpe
        sharpe = self.metrics.get('sharpe', 0.0)
        self.metric_labels['sharpe'].setText(f"{sharpe:.2f}")
    
    def update_equity_curve(self):
        """Update equity curve chart."""
        self.equity_plot.clear()
        
        if self.trades_df is None or len(self.trades_df) == 0:
            return
        
        # Calculate cumulative PnL
        cumulative_pnl = self.trades_df['pnl'].cumsum()
        
        # Plot equity curve
        x = range(1, len(cumulative_pnl) + 1)
        self.equity_plot.plot(
            x, cumulative_pnl.values,
            pen=pg.mkPen('#4caf50', width=2),
            name='Equity Curve'
        )
        
        # Add zero line
        self.equity_plot.plot(
            [1, len(cumulative_pnl)], [0, 0],
            pen=pg.mkPen('#666', width=1, style=Qt.DashLine)
        )
        
        # Calculate and plot drawdown
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        
        # Fill drawdown area
        fill = pg.FillBetweenItem(
            pg.PlotDataItem(x, cumulative_pnl.values),
            pg.PlotDataItem(x, running_max.values),
            brush=(255, 0, 0, 50)
        )
        self.equity_plot.addItem(fill)
    
    def update_fitness_plot(self):
        """Update fitness evolution plot."""
        self.fitness_plot.clear()
        
        if self.optimizer is None:
            return
        
        history = self.optimizer.get_history()
        if not history:
            return
        
        # Extract generation and fitness data
        generations = [h['generation'] for h in history]
        best_fitness = [h.get('best_fitness', h.get('fitness', 0)) for h in history]
        
        # For evolutionary algorithms: mean_fitness (population average)
        # For ADAM: current fitness (generation-by-generation fitness)
        mean_fitness = [h.get('mean_fitness') for h in history]
        current_fitness = [h.get('fitness', 0) for h in history]
        
        # Plot best fitness (line only, no symbols)
        self.fitness_plot.plot(
            generations, best_fitness,
            pen=pg.mkPen('#4caf50', width=3),
            name='Best Fitness'
        )
        
        # Plot second line based on optimizer type
        # If mean_fitness exists (evolutionary), plot it
        # Otherwise plot current fitness (ADAM)
        if any(m is not None for m in mean_fitness):
            # Evolutionary algorithm - plot mean fitness
            clean_mean = [m if m is not None else 0 for m in mean_fitness]
            self.fitness_plot.plot(
                generations, clean_mean,
                pen=pg.mkPen('#ff9800', width=2),
                name='Avg Fitness'
            )
        elif len(current_fitness) > 0:
            # ADAM - plot current fitness
            self.fitness_plot.plot(
                generations, current_fitness,
                pen=pg.mkPen('#ff9800', width=2),
                name='Current Fitness'
            )
        
        # Add legend
        self.fitness_plot.addLegend()
    
    def set_current_data(self, data, tf, raw_data=None):
        """
        Store current data and timeframe for optimization.
        
        Args:
            data: Features dataframe (with indicators and signals)
            tf: Timeframe enum
            raw_data: Optional raw OHLCV dataframe. If None, will extract from data.
        """
        self.current_data = data
        self.current_tf = tf
        
        # Store raw OHLCV data for rebuilding features during optimization
        if raw_data is not None:
            self.raw_data = raw_data
        else:
            # Extract raw OHLCV from features dataframe
            # Note: This may have different length due to dropped NaN rows
            self.raw_data = data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Optimization button is always enabled when data is loaded
    
    def get_current_params(self):
        """Get current parameters from external editor.
        
        Returns:
            Tuple of (SellerParams, BacktestParams, FitnessConfig)
        """
        if self.param_editor:
            # Get params from editor - fitness preset is already reflected in the weights
            seller_params, backtest_params, fitness_config = self.param_editor.get_params()
            
            # Update the preset field based on our dropdown selection
            if hasattr(self, 'fitness_preset_combo'):
                preset_name = self.fitness_preset_combo.currentData()
                fitness_config.preset = preset_name
            
            return seller_params, backtest_params, fitness_config
        else:
            # Fallback to defaults if no editor connected
            from core.models import FitnessConfig
            return SellerParams(), BacktestParams(), FitnessConfig()
    
    def set_params_from_individual(self, individual: Individual):
        """Update external editor from an individual."""
        if self.param_editor:
            self.param_editor.set_params(individual.seller_params, individual.backtest_params)
    

    
    def run_optimization_step(self):
        """Run one optimization iteration asynchronously (internal method)."""
        if self.optimizer is None:
            logger.error("Optimizer not initialized")
            return
        
        if self.current_data is None:
            logger.error("No data loaded")
            return
        
        # Disable optimize button during execution
        self.optimize_btn.setEnabled(False)
        
        # Emit progress signal
        self.progress_updated.emit(0, 1, "Running optimization step...")
        
        # Run in thread to avoid UI freeze
        from threading import Thread
        
        def _run_single_step():
            try:
                # Get fitness configuration from parameter editor
                _, _, fitness_config = self.get_current_params()
                self._log_fitness_config(fitness_config, prefix="  ")
                self._log_optimizer_config(prefix="  ")
                try:
                    workers = self.optimizer.get_worker_count()
                except Exception:
                    workers = "unknown"
                print(
                    f"STEP begin workers={workers} preset={getattr(fitness_config, 'preset', '--')}"
                )
                
                # CRITICAL: Pass RAW data, not features!
                data_for_optimizer = self.raw_data if self.raw_data is not None else self.current_data[['open', 'high', 'low', 'close', 'volume']]
                
                # Run one optimization step (works for any optimizer)
                # Pass progress callback and stop flag for responsive UI
                def progress_cb(current, total, message):
                    """Progress callback for optimizer."""
                    QMetaObject.invokeMethod(
                        self,
                        "_emit_progress_signal",
                        Qt.QueuedConnection,
                        Q_ARG(int, current),
                        Q_ARG(int, total),
                        Q_ARG(str, message)
                    )
                
                def stop_check():
                    """Check if user requested stop."""
                    return self.stop_requested
                
                result = self.optimizer.step(
                    data=data_for_optimizer,
                    timeframe=self.current_tf,
                    fitness_config=fitness_config,
                    progress_callback=progress_cb,
                    stop_flag=stop_check
                )
                
                # Extract best parameters
                best_seller = result.best_seller_params
                best_backtest = result.best_backtest_params
                best_fitness = result.fitness
                best_metrics = result.metrics
                
                # Store backtest result for main thread
                backtest_result = None
                
                # Run backtest with best parameters to visualize strategy
                logger.info("Running backtest with current best parameters...")
                try:
                    if self.raw_data is None:
                        logger.warning("No raw data available, skipping backtest")
                    else:
                        # Build features with best params from RAW data
                        feats = build_features(
                            self.raw_data,
                            best_seller,
                            self.current_tf,
                        )
                        
                        # Run backtest
                        backtest_result = run_backtest(feats, best_backtest)
                        
                        logger.info("Backtest complete: %d trades", backtest_result['metrics']['n'])
                        self._log_metrics_snapshot(backtest_result.get('metrics', {}), prefix="  ")
                        self._log_parameter_snapshot(best_seller, best_backtest, prefix="  ")
                        metrics_payload = backtest_result.get('metrics') if backtest_result else {}
                        print(
                            "STEP best "
                            f"fitness={best_fitness:.4f} "
                            f"{self._compact_metrics(metrics_payload)}"
                        )
                    
                except Exception as e:
                    logger.exception("Error running backtest for visualization: %s", e)
                    logger.debug(f"❌ Backtest error: {e}")
                
                # Store results for main thread to process
                self.temp_backtest_result = backtest_result
                
                # Emit signal to update UI in main thread
                self.generation_complete.emit()
                
                # Emit progress completion
                self.progress_updated.emit(1, 1, "✓ Optimization step complete")
                print("STEP done")
                
            except Exception as e:
                logger.exception("Error during optimization step: %s", e)
                self.progress_updated.emit(0, 1, f"❌ Error: {str(e)}")
                logger.debug(f"❌ Optimization step error: {e}")
            finally:
                # Re-enable optimize button using thread-safe method
                QMetaObject.invokeMethod(
                    self.optimize_btn,
                    "setEnabled",
                    Qt.QueuedConnection,
                    Q_ARG(bool, True)
                )
        
        # Start thread
        thread = Thread(target=_run_single_step, daemon=True)
        thread.start()
    
    @Slot()
    def _update_after_generation(self):
        """Update UI after iteration completes (runs in main thread)."""
        try:
            if self.optimizer is None:
                return
            
            # Get current generation from optimizer stats
            stats = self.optimizer.get_stats()
            generation = stats.get('generation', 0)
            
            # Check if coach should analyze (agent-based coach)
            if self.coach_manager and self.coach_manager.should_analyze(generation):
                logger.info("🤖 Evolution Coach Agent triggering at generation %s", generation)
                # Run coach analysis in background thread to avoid blocking UI
                from threading import Thread
                
                def run_coach_analysis():
                    try:
                        # Show coach progress in UI
                        if hasattr(self, 'param_editor') and hasattr(self.param_editor, 'show_coach_progress'):
                            self.param_editor.show_coach_progress("Evolution Coach analyzing population...")
                        
                        # Update status bar with coach start - show mode-specific message
                        coach_mode = getattr(settings, 'coach_mode', 'classic')
                        if coach_mode == 'openai':
                            status_msg = "🤖 OpenAI Coach analyzing population..."
                        elif coach_mode == 'classic':
                            status_msg = "🧠 Classic Coach analyzing population..."
                        else:
                            status_msg = "🤖 Coach analyzing population..."
        
                        if hasattr(self, 'chart_view') and hasattr(self.chart_view, 'set_coach_status'):
                            self.chart_view.set_coach_status(status_msg)
                        
                        # Get current configs
                        _, _, fitness_config = self.get_current_params()
                        
                        # Get population from optimizer
                        population = getattr(self.optimizer, 'population', None)
                        if population is None:
                            logger.warning("Optimizer has no population attribute, skipping coach")
                            return
                        
                        # Get GA config from optimizer (use optimizer's config as source of truth)
                        from core.models import OptimizationConfig
                        optimizer_config = getattr(self.optimizer, 'config', {})
                        ga_config = OptimizationConfig(
                            population_size=optimizer_config.get('population_size', len(population.individuals)),
                            mutation_probability=optimizer_config.get('mutation_probability', 0.9),
                            mutation_rate=optimizer_config.get('mutation_rate', 0.55),
                            sigma=optimizer_config.get('sigma', 0.15),
                            elite_fraction=optimizer_config.get('elite_fraction', 0.1),
                            tournament_size=optimizer_config.get('tournament_size', 3),
                            immigrant_fraction=optimizer_config.get('immigrant_fraction', 0.0)
                        )
                        
                        # Show progress bar before analysis
                        if hasattr(self, 'param_editor') and self.param_editor:
                            try:
                                self.param_editor.set_coach_analyzing(True)
                            except Exception as e:
                                print(f"Error showing coach progress bar: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # Run agent analysis (blocking call but in background thread)
                        import asyncio
                        from config.settings import settings
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            # Check if coach is disabled
                            if settings.coach_mode == "disabled":
                                print("🔍 DEBUG: Coach mode is disabled, skipping analysis")
                                success, summary = True, "Coach disabled"
                            else:
                                print(f"🔍 DEBUG: stats_panel.py calling analyze_and_apply_with_openai_agent with status_callback={self.status_callback}")
                                success, summary = loop.run_until_complete(
                                    self.coach_manager.analyze_and_apply_with_openai_agent(
                                        population=population,
                                        fitness_config=fitness_config,
                                        ga_config=ga_config,
                                        coach_window=self.coach_window,
                                        status_callback=self.status_callback
                                    )
                                )
                            
                            if success:
                                # Hide progress bar after successful analysis
                                if hasattr(self, 'param_editor') and self.param_editor:
                                    try:
                                        self.param_editor.set_coach_analyzing(False)
                                    except Exception as e:
                                        print(f"Error hiding coach progress bar: {e}")
                                
                                # Coach may have modified ga_config - propagate changes back to optimizer
                                if hasattr(self.optimizer, 'config'):
                                    self.optimizer.config.update({
                                        'mutation_probability': ga_config.mutation_probability,
                                        'mutation_rate': ga_config.mutation_rate,
                                        'sigma': ga_config.sigma,
                                        'elite_fraction': ga_config.elite_fraction,
                                        'tournament_size': ga_config.tournament_size,
                                        'population_size': ga_config.population_size,
                                    })
                                    logger.info("Updated optimizer config with Coach modifications")

                                    # Save coach-modified parameters to .env for persistence
                                    self._save_coach_modified_params_to_env()
                                
                                logger.info("✅ Coach agent completed: %s", summary)
                                
                                # Hide coach progress and update status
                                if hasattr(self, 'param_editor') and hasattr(self.param_editor, 'hide_coach_progress'):
                                    self.param_editor.hide_coach_progress()
                                
                                # Update status bar with coach completion - show mode-specific message
                                action_count = summary.get('total_actions', 0)
                                coach_mode = getattr(settings, 'coach_mode', 'classic')
                                if coach_mode == 'openai':
                                    completion_msg = f"✅ OpenAI Coach completed: {action_count} actions taken"
                                elif coach_mode == 'classic':
                                    completion_msg = f"✅ Classic Coach completed: {action_count} actions taken"
                                else:
                                    completion_msg = f"✅ Coach completed: {action_count} actions taken"

                                if hasattr(self, 'chart_view') and hasattr(self.chart_view, 'set_coach_status'):
                                    self.chart_view.set_coach_status(completion_msg, is_recommendation=True)

                                # Update status bar if callback available
                                if self.status_callback:
                                    self.status_callback(f"✅ Coach: {action_count} actions taken at gen {generation}")
                            else:
                                logger.warning("⚠ Coach agent failed: %s", summary.get('error', 'Unknown'))
                                
                                # Hide coach progress and update status
                                if hasattr(self, 'param_editor') and hasattr(self.param_editor, 'hide_coach_progress'):
                                    self.param_editor.hide_coach_progress()
                                
                                # Update status bar with coach failure - show mode-specific message
                                coach_mode = getattr(settings, 'coach_mode', 'classic')
                                if coach_mode == 'openai':
                                    failure_msg = "⚠ OpenAI Coach analysis failed"
                                elif coach_mode == 'classic':
                                    failure_msg = "⚠ Classic Coach analysis failed"
                                else:
                                    failure_msg = "⚠ Coach analysis failed"

                                if hasattr(self, 'chart_view') and hasattr(self.chart_view, 'set_coach_status'):
                                    self.chart_view.set_coach_status(failure_msg)

                                if self.status_callback:
                                    self.status_callback(f"⚠ Coach failed at gen {generation}")
                        finally:
                            loop.close()
                    except Exception as e:
                        logger.exception("Error during coach analysis: %s", e)
                        
                        # Hide coach progress and update status
                        if hasattr(self, 'param_editor') and hasattr(self.param_editor, 'hide_coach_progress'):
                            self.param_editor.hide_coach_progress()
                        
                        # Update status bar with coach error - show mode-specific message
                        coach_mode = getattr(settings, 'coach_mode', 'classic')
                        if coach_mode == 'openai':
                            error_msg = f"❌ OpenAI Coach error: {str(e)[:50]}"
                        elif coach_mode == 'classic':
                            error_msg = f"❌ Classic Coach error: {str(e)[:50]}"
                        else:
                            error_msg = f"❌ Coach error: {str(e)[:50]}"

                        if hasattr(self, 'chart_view') and hasattr(self.chart_view, 'set_coach_status'):
                            self.chart_view.set_coach_status(error_msg)

                        if self.status_callback:
                            self.status_callback(error_msg)
                
                # Start coach thread
                coach_thread = Thread(target=run_coach_analysis, daemon=True)
                coach_thread.start()
            
            # Update fitness evolution plot
            self.update_fitness_plot()
            
            # Get best parameters
            best_seller, best_backtest, best_fitness = self.optimizer.get_best_params()
            
            # Create best individual if found
            if best_seller is not None:
                # Create temporary Individual for compatibility
                best_individual = Individual(
                    seller_params=best_seller,
                    backtest_params=best_backtest,
                    fitness=best_fitness
                )
                
                # AUTO-APPLY: Update param editor immediately with best parameters
                logger.info("Auto-applying best parameters from generation %s", generation)
                self.set_params_from_individual(best_individual)
                
                # Update stats display with backtest results if available
                if self.temp_backtest_result:
                    self.trades_df = self.temp_backtest_result['trades']
                    self.metrics = self.temp_backtest_result['metrics']
                    
                    # Update metrics display
                    self.update_metrics()
                    
                    # Update equity curve
                    self.update_equity_curve()
                    
                    # Emit signal with backtest results for chart update
                    self.optimization_step_complete.emit(best_individual, self.temp_backtest_result)
            
            # Status is now shown in chart view's progress bar via signals (no local status label)
            
        except Exception as e:
            logger.exception("Error updating UI after iteration: %s", e)
    
    def apply_best_parameters(self, auto_save: bool = False):
        """
        Apply best parameters from optimizer to UI and optionally save to settings.
        
        Args:
            auto_save: If True, automatically save parameters to .env file
        """
        if self.optimizer is None:
            logger.warning("No optimizer initialized")
            return
        
        best_seller, best_backtest, best_fitness = self.optimizer.get_best_params()
        if best_seller is None:
            logger.warning("No best parameters to apply")
            return
        
        # Create Individual for compatibility
        best = Individual(
            seller_params=best_seller,
            backtest_params=best_backtest,
            fitness=best_fitness
        )
        
        self.set_params_from_individual(best)
        
        # Get generation from stats
        stats = self.optimizer.get_stats()
        generation = stats.get('generation', 0)
        
        logger.info(
            "Applied best parameters from generation %s | Fitness=%.4f",
            generation,
            best_fitness or 0.0,
        )
        self._log_parameter_snapshot(best_seller, best_backtest, prefix="  ")
        
        # Auto-save to .env if requested
        if auto_save:
            try:
                from config.settings import SettingsManager
                
                # Build settings dict from best individual
                settings_dict = {
                    'strategy_ema_fast': int(best.seller_params.ema_fast),
                    'strategy_ema_slow': int(best.seller_params.ema_slow),
                    'strategy_z_window': int(best.seller_params.z_window),
                    'strategy_vol_z': float(best.seller_params.vol_z),
                    'strategy_tr_z': float(best.seller_params.tr_z),
                    'strategy_cloc_min': float(best.seller_params.cloc_min),
                    'strategy_atr_window': int(best.seller_params.atr_window),
                    'backtest_fee_bp': float(best.backtest_params.fee_bp),
                    'backtest_slippage_bp': float(best.backtest_params.slippage_bp),
                }
                
                SettingsManager.save_to_env(settings_dict)
                SettingsManager.reload_settings()
                
                logger.info("Auto-saved best parameters to .env")
            except Exception as e:
                logger.exception("Failed to auto-save parameters: %s", e)
    
    def load_params_from_settings(self, seller_params, backtest_params):
        """Load parameters into external editor from settings dialog."""
        if self.param_editor:
            self.param_editor.set_params(seller_params, backtest_params)
    
    def clear(self):
        """Clear all displays."""
        self.trades_df = None
        self.metrics = None
        
        # Reset metrics
        for label in self.metric_labels.values():
            label.setText("--")
            label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # Clear equity curve
        self.equity_plot.clear()
        
        # Clear fitness plot
        self.fitness_plot.clear()
    
    def run_multi_step_optimize(self):
        """
        Run multiple optimization steps asynchronously.
        
        Automatically initializes population if needed.
        Shows progress and allows stopping to keep current best.
        """
        if self.current_data is None:
            logger.error("No data loaded. Please download data first.")
            return
        
        if self.is_optimizing:
            logger.warning("Optimization already in progress")
            return
        
        # Auto-initialize optimizer if needed (logs all SEED/FIT config)
        if not self._initialize_optimizer_if_needed():
            return
        
        fitness_config = self.get_current_params()[2]
        
        from config.settings import settings
        n_gens = settings.optimizer_iterations
        
        logger.info("=" * 70)
        logger.info("🚀 Starting Multi-Step Optimization: %s generations", n_gens)
        logger.info("=" * 70)
        try:
            workers = self.optimizer.get_worker_count()
        except Exception:
            workers = "unknown"
        print(
            f"RUN start iters={n_gens} workers={workers}"
        )
        
        # Reset best fitness tracking for this optimization run
        self.prev_best_fitness = None
        best_seller, best_backtest, best_fitness = self.optimizer.get_best_params()
        if best_seller is not None:
            self.prev_best_fitness = best_fitness
        
        # Update UI state
        self.is_optimizing = True
        self.stop_requested = False
        
        self.optimize_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)  # Enable stop button during optimization
        self.fitness_preset_combo.setEnabled(False)
        self.optimizer_type_combo.setEnabled(False)
        
        # Emit initial progress
        self.progress_updated.emit(0, n_gens, "Initializing optimization...")
        
        # Run optimization in separate thread to not block UI
        import threading
        thread = threading.Thread(
            target=self._run_multi_step_thread,
            args=(n_gens,),
            daemon=True
        )
        thread.start()
    
    def _run_multi_step_thread(self, n_gens: int):
        """
        Worker thread for multi-step optimization.
        
        This runs in background thread so UI stays responsive.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            import time
            start_time = time.time()
            
            for gen in range(n_gens):
                if self.stop_requested:
                    self.progress_updated.emit(gen, n_gens, "⏹ Stopped by user")
                    logger.info("⏹ Optimization stopped at generation %s (keeping best individual)", gen)
                    break
                
                # Update progress
                elapsed = time.time() - start_time
                avg_time = elapsed / (gen + 1) if gen > 0 else 1
                remaining = avg_time * (n_gens - gen - 1)
                
                msg = f"Generation {gen+1}/{n_gens} | ETA: {remaining:.0f}s"
                self.progress_updated.emit(gen, n_gens, msg)
                
                # Run one optimization step
                try:
                    # Get fitness configuration from parameter editor
                    _, _, fitness_config = self.get_current_params()
                    
                    # CRITICAL: Pass RAW data (OHLCV only), not features!
                    data_for_optimizer = self.raw_data if self.raw_data is not None else self.current_data[['open', 'high', 'low', 'close', 'volume']]
                    
                    # Run one step using abstract optimizer
                    # Pass stop flag for responsive UI
                    def stop_check():
                        """Check if user requested stop."""
                        return self.stop_requested
                    
                    result = self.optimizer.step(
                        data=data_for_optimizer,
                        timeframe=self.current_tf,
                        fitness_config=fitness_config,
                        stop_flag=stop_check
                    )
                    
                    # Evolve islands in parallel (if any were created by the coach and islands are enabled)
                    if self.coach_manager and getattr(self.coach_manager, 'islands_enabled', False):
                        try:
                            from core.models import OptimizationConfig
                            optimizer_config = getattr(self.optimizer, 'config', {})
                            ga_config = OptimizationConfig(
                                population_size=optimizer_config.get('population_size', len(getattr(self.optimizer, 'population', {}).individuals) if getattr(self.optimizer, 'population', None) else 0),
                                mutation_probability=optimizer_config.get('mutation_probability', 0.9),
                                mutation_rate=optimizer_config.get('mutation_rate', 0.55),
                                sigma=optimizer_config.get('sigma', 0.15),
                                elite_fraction=optimizer_config.get('elite_fraction', 0.1),
                                tournament_size=optimizer_config.get('tournament_size', 3),
                                immigrant_fraction=optimizer_config.get('immigrant_fraction', 0.0)
                            )
                            # Provide the current main population so scheduler can merge elites
                            population = getattr(self.optimizer, 'population', None)
                            self.coach_manager.evolve_islands_step(
                                data=data_for_optimizer,
                                timeframe=self.current_tf,
                                fitness_config=fitness_config,
                                ga_config=ga_config,
                                main_population=population
                            )
                        except Exception:
                            pass
                    
                    # CRITICAL: Record EVERY generation to agent_feed (not just when coach triggers)
                    population = getattr(self.optimizer, 'population', None)
                    if population and self.coach_manager:
                        self.coach_manager.record_generation(population)
                    
                    if result.additional_info:
                        population_stats = result.additional_info.get('population_stats')
                        if isinstance(population_stats, dict) and population_stats:
                            mean_fitness = population_stats.get('mean_fitness', 0.0)
                            std_fitness = population_stats.get('std_fitness', 0.0)
                            min_fitness = population_stats.get('min_fitness', 0.0)
                            max_fitness = population_stats.get('max_fitness', 0.0)
                            best_ever = population_stats.get('best_ever_fitness', max_fitness)

                            population = getattr(self.optimizer, 'population', None)
                            diversity = None
                            mean_trades = None
                            below_pct = None
                            effective_min = None

                            if population is not None and hasattr(population, 'individuals'):
                                try:
                                    diversity = population.get_diversity_metric()
                                except Exception:
                                    diversity = None

                                trade_counts = [
                                    ind.metrics.get('n', 0)
                                    for ind in population.individuals
                                ]
                                if trade_counts:
                                    mean_trades = float(sum(trade_counts)) / len(trade_counts)

                                if hasattr(fitness_config, "get_effective_min_trades"):
                                    try:
                                        effective_min = fitness_config.get_effective_min_trades(population.generation)
                                    except Exception:
                                        effective_min = None

                                if effective_min is not None and trade_counts:
                                    below_count = sum(1 for tc in trade_counts if tc < effective_min)
                                    below_pct = (below_count / len(trade_counts)) * 100.0

                            mean_trades_str = f"{mean_trades:.1f}" if mean_trades is not None else "--"
                            below_min_str = f"{below_pct:.1f}%" if below_pct is not None else "--"
                            diversity_str = f"{diversity:.2f}" if diversity is not None else "--"

                            logger.info(
                                "Population stats | generation=%s | mean_fitness=%.4f | std_fitness=%.4f | "
                                "min=%.4f | max=%.4f | best_ever=%.4f | mean_trades=%s | below_min=%s | diversity=%s",
                                result.generation,
                                mean_fitness,
                                std_fitness,
                                min_fitness,
                                max_fitness,
                                best_ever,
                                mean_trades_str,
                                below_min_str,
                                diversity_str,
                            )

                            stat_parts = [
                                f"mean={mean_fitness:.4f}",
                                f"std={std_fitness:.4f}",
                                f"min={min_fitness:.4f}",
                                f"max={max_fitness:.4f}",
                                f"best_ever={best_ever:.4f}",
                            ]
                            if mean_trades is not None:
                                stat_parts.append(f"mean_trades={mean_trades:.1f}")
                            if below_pct is not None and effective_min is not None:
                                stat_parts.append(f"below_min={below_pct:.1f}% (min={effective_min})")
                            if diversity is not None:
                                stat_parts.append(f"diversity={diversity:.2f}")

                            stat_message = "STAT " + " | ".join(stat_parts)
                            if self.coach_manager:
                                self.coach_manager.add_log(result.generation, stat_message)
                            else:
                                print(stat_message)
                    
                    # Extract results
                    current_best_fitness = result.fitness
                    best_seller = result.best_seller_params
                    best_backtest = result.best_backtest_params
                    best_metrics = result.metrics
                    
                    # Check if we have a new best
                    new_best_found = False
                    if self.prev_best_fitness is None or current_best_fitness > self.prev_best_fitness:
                        new_best_found = True
                        self.prev_best_fitness = current_best_fitness
                        
                        # Run backtest with new best to visualize
                        logger.info("🎯 New best found in generation %s | Fitness=%.4f", gen + 1, current_best_fitness)
                        self._log_metrics_snapshot(best_metrics, prefix="  ")
                        if best_seller is not None and best_backtest is not None:
                            self._log_parameter_snapshot(best_seller, best_backtest, prefix="  ")
                        best_line = (
                            "BEST "
                            f"gen={gen+1} fitness={current_best_fitness:.4f} "
                            f"{self._compact_metrics(best_metrics)}"
                        )
                        if self.coach_manager:
                            self.coach_manager.add_log(gen + 1, best_line)
                        else:
                            print(best_line)
                        # Include full param snapshot for best
                        if best_seller is not None:
                            from dataclasses import asdict as _asdict
                            seller_line = f"BEST seller[{self._format_dict_compact(_asdict(best_seller))}]"
                            if self.coach_manager:
                                self.coach_manager.add_log(gen + 1, seller_line)
                            else:
                                print(seller_line)
                        if best_backtest is not None:
                            btp = best_backtest.model_dump() if hasattr(best_backtest, 'model_dump') else dict(best_backtest)
                            backtest_line = f"BEST backtest_params[{self._format_dict_compact(btp)}]"
                            if self.coach_manager:
                                self.coach_manager.add_log(gen + 1, backtest_line)
                            else:
                                print(backtest_line)
                        
                        # Run backtest for visualization using RAW data
                        logger.info("Running backtest with new best parameters...")
                        
                        try:
                            if self.raw_data is None:
                                logger.warning("No raw data available, skipping visualization backtest")
                                self.temp_backtest_result = None
                                print("WARN no_raw_data")
                            else:
                                # Rebuild features from raw OHLCV data with new parameters
                                feats = build_features(self.raw_data, best_seller, self.current_tf)
                                backtest_result = run_backtest(feats, best_backtest)
                                
                                logger.info("Backtest complete: %d trades", backtest_result['metrics']['n'])
                                
                                # Store for UI update
                                self.temp_backtest_result = backtest_result
                                metrics_payload = backtest_result.get('metrics', {}) if backtest_result else {}
                                backtest_metrics_line = (
                                    "BEST backtest "
                                    f"gen={gen+1} {self._compact_metrics(metrics_payload)}"
                                )
                                if self.coach_manager:
                                    self.coach_manager.add_log(gen + 1, backtest_metrics_line)
                                else:
                                    print(backtest_metrics_line)
                            
                        except Exception as e:
                            logger.exception("Error running visualization backtest: %s", e)
                            logger.debug(f"❌ Visualization backtest error: {e}")
                            self.temp_backtest_result = None
                    
                    # Update UI (from main thread)
                    # Pass whether this is a new best to the UI updater
                    QMetaObject.invokeMethod(
                        self,
                        "_update_after_multi_step_generation",
                        Qt.QueuedConnection,
                        Q_ARG(bool, new_best_found)
                    )
                    
                    # Check if Evolution Coach should analyze this generation
                    if self.coach_manager and self.coach_manager.should_analyze(gen + 1):
                        logger.info("⏸️  PAUSING OPTIMIZATION FOR COACH ANALYSIS (Generation %s)", gen + 1)
                        
                        # Get current population
                        population = getattr(self.optimizer, 'population', None)
                        if population:
                            try:
                                # Log population summary for coach context
                                if hasattr(population, 'individuals') and population.individuals:
                                    pop_size = len(population.individuals)
                                    avg_fitness = sum(ind.fitness for ind in population.individuals) / pop_size if pop_size > 0 else 0.0
                                    best_fitness = max((ind.fitness for ind in population.individuals), default=0.0)
                                    avg_trades = sum(ind.metrics.get('n', 0) for ind in population.individuals) / pop_size if pop_size > 0 else 0.0
                                    print(
                                        f"[COACH  ] Population stats: size={pop_size} avg_fitness={avg_fitness:.4f} "
                                        f"best_fitness={best_fitness:.4f} avg_trades={avg_trades:.1f}"
                                    )
                                
                                # Run agent-based coach analysis
                                _, _, fitness_config = self.get_current_params()
                                
                                # Get GA config from optimizer (use optimizer's config as source of truth)
                                from core.models import OptimizationConfig
                                optimizer_config = getattr(self.optimizer, 'config', {})
                                ga_config = OptimizationConfig(
                                    population_size=optimizer_config.get('population_size', len(population.individuals)),
                                    mutation_probability=optimizer_config.get('mutation_probability', 0.9),
                                    mutation_rate=optimizer_config.get('mutation_rate', 0.55),
                                    sigma=optimizer_config.get('sigma', 0.15),
                                    elite_fraction=optimizer_config.get('elite_fraction', 0.1),
                                    tournament_size=optimizer_config.get('tournament_size', 3),
                                    immigrant_fraction=optimizer_config.get('immigrant_fraction', 0.0)
                                )
                                
                                # Show progress bar before analysis
                                if hasattr(self, 'param_editor') and self.param_editor:
                                    try:
                                        self.param_editor.set_coach_analyzing(True)
                                    except Exception as e:
                                        print(f"Error showing coach progress bar: {e}")
                                        import traceback
                                        traceback.print_exc()
                                
                                # Use OpenAI Agents coach analysis
                                # Check if coach is disabled
                                from config.settings import settings
                                if settings.coach_mode == "disabled":
                                    print("🔍 DEBUG: Coach mode is disabled, skipping analysis")
                                    success, summary = True, "Coach disabled"
                                else:
                                    print(f"🔍 DEBUG: stats_panel.py calling analyze_and_apply_with_openai_agent with status_callback={self.status_callback}")
                                    success, summary = loop.run_until_complete(
                                        self.coach_manager.analyze_and_apply_with_openai_agent(
                                            population=population,
                                            fitness_config=fitness_config,
                                            ga_config=ga_config,
                                            coach_window=self.coach_window,
                                            status_callback=self.status_callback
                                        )
                                    )
                                
                                if success:
                                    # Hide progress bar after successful analysis
                                    if hasattr(self, 'param_editor') and self.param_editor:
                                        try:
                                            self.param_editor.set_coach_analyzing(False)
                                        except Exception as e:
                                            print(f"Error hiding coach progress bar: {e}")
                                    
                                    # Coach may have modified ga_config - propagate changes back to optimizer
                                    if hasattr(self.optimizer, 'config'):
                                        self.optimizer.config.update({
                                            'mutation_probability': ga_config.mutation_probability,
                                            'mutation_rate': ga_config.mutation_rate,
                                            'sigma': ga_config.sigma,
                                            'elite_fraction': ga_config.elite_fraction,
                                            'tournament_size': ga_config.tournament_size,
                                            'population_size': ga_config.population_size,
                                        })
                                        logger.info("Updated optimizer config with Coach modifications")
                                    
                                    # Agent returns 'total_actions'
                                    action_count = summary.get('total_actions', 0)
                                    logger.info("✅ Coach agent completed: %s actions", action_count)

                                    # Update status bar with completion message
                                    if self.status_callback:
                                        status_msg = f"✅ Coach completed: {action_count} actions at gen {gen + 1}"
                                        self.status_callback(status_msg)
                                else:
                                    # Hide progress bar after failed analysis
                                    if hasattr(self, 'param_editor') and self.param_editor:
                                        try:
                                            self.param_editor.set_coach_analyzing(False)
                                        except Exception as e:
                                            print(f"Error hiding coach progress bar: {e}")
                                    
                                    error_msg = summary.get('error', 'Unknown error')
                                    logger.warning("⚠️  Coach agent failed: %s", error_msg)

                                    # Update status bar with failure message
                                    if self.status_callback:
                                        self.status_callback(f"⚠ Coach failed at gen {gen + 1}")
                            except Exception as e:
                                # Hide progress bar after exception
                                if hasattr(self, 'param_editor') and self.param_editor:
                                    try:
                                        self.param_editor.set_coach_analyzing(False)
                                    except Exception as e:
                                        print(f"Error hiding coach progress bar: {e}")
                                
                                logger.exception("Coach analysis error: %s", e)

                                # Update status bar with error message
                                if self.status_callback:
                                    self.status_callback(f"❌ Coach error: {str(e)[:50]}")
                        else:
                            logger.warning("⚠️  No population available for coach analysis")
                        
                        logger.info("▶️  RESUMING OPTIMIZATION (Generation %s)", gen + 1)
                    
                except Exception as e:
                    logger.exception("Error in generation %s: %s", gen + 1, e)
                    logger.debug(f"❌ Generation {gen + 1} error: {e}")
                    break
            
            # Final progress update
            total_time = time.time() - start_time
            if not self.stop_requested:
                self.progress_updated.emit(n_gens, n_gens, f"✓ Complete in {total_time:.1f}s")
                logger.info("=" * 70)
                logger.info("✓ Optimization complete! %s generations in %.1fs", n_gens, total_time)
                logger.info("  Average: %.2fs per generation", total_time / n_gens if n_gens else 0.0)
                logger.info("=" * 70)
                run_summary = f"RUN done time={total_time:.1f}s avg={(total_time / n_gens if n_gens else 0.0):.2f}s"
                # _coach_log removed (dead code)
                
                # Show final best results
                best_seller, best_backtest, best_fitness = self.optimizer.get_best_params()
                if best_seller is not None:
                    logger.info("🏆 Final Best Parameters | Fitness=%.4f", best_fitness or 0.0)
                    
                    # Get metrics from optimizer stats if available
                    stats = self.optimizer.get_stats()
                    best_metrics = stats.get('best_metrics') if isinstance(stats, dict) else None
                    self._log_metrics_snapshot(best_metrics or {}, prefix="  ")
                    self._log_parameter_snapshot(best_seller, best_backtest, prefix="  ")
                    final_line = (
                        "BEST final "
                        f"fitness={(best_fitness or 0.0):.4f} {self._compact_metrics(best_metrics or {})}"
                    )
                    population = getattr(self.optimizer, 'population', None)
                    gen_index = getattr(population, 'generation', n_gens)
                    # _coach_log removed (dead code)
        
        except Exception as e:
            logger.exception("Error in multi-step optimization: %s", e)
            self.progress_updated.emit(0, n_gens, f"❌ Error: {str(e)}")
            logger.debug(f"❌ Optimization run error: {e}")
        
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            asyncio.set_event_loop(None)
            
            # Restore UI state (from main thread)
            QMetaObject.invokeMethod(
                self,
                "_restore_ui_after_optimize",
                Qt.QueuedConnection
            )
    
    @Slot(bool)
    def _update_after_multi_step_generation(self, new_best_found: bool):
        """
        Update UI after each generation during multi-step optimization.
        
        Args:
            new_best_found: True if a new best was found this generation
        """
        try:
            if self.optimizer is None:
                return
            
            # Get current generation
            stats = self.optimizer.get_stats()
            generation = stats.get('generation', 0)
            
            # Always update fitness plot
            self.update_fitness_plot()
            
            # Status is now shown in chart view's progress bar via signals (no local status label)
            best_seller, best_backtest, best_fitness = self.optimizer.get_best_params()
            
            # If new best found, update all displays and chart
            if new_best_found and best_seller is not None:
                # Create Individual for compatibility
                best_individual = Individual(
                    seller_params=best_seller,
                    backtest_params=best_backtest,
                    fitness=best_fitness
                )
                
                # AUTO-APPLY: Update param editor immediately with best parameters
                logger.info("Auto-applying new best parameters from generation %s", generation)
                self.set_params_from_individual(best_individual)
                self._log_parameter_snapshot(best_individual.seller_params, best_individual.backtest_params, prefix="  ")
                
                # Update stats display with backtest results if available
                if self.temp_backtest_result:
                    self.trades_df = self.temp_backtest_result['trades']
                    self.metrics = self.temp_backtest_result['metrics']
                    self._log_metrics_snapshot(self.metrics, prefix="  ")
                    
                    # Update metrics display
                    self.update_metrics()
                    
                    # Update equity curve
                    self.update_equity_curve()
                    
                    # Emit signal with backtest results for chart update
                    logger.debug("Updating chart with new best strategy")
                    self.optimization_step_complete.emit(best_individual, self.temp_backtest_result)
                else:
                    logger.warning("No backtest result available for new best")
            
        except Exception as e:
            logger.exception("Error updating UI after multi-step generation: %s", e)
    
    @Slot()
    @Slot()
    def _restore_ui_after_optimize(self):
        """Restore UI state after optimization completes (called from main thread)."""
        self.is_optimizing = False
        self.optimize_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)  # Disable stop button after optimization
        self.fitness_preset_combo.setEnabled(True)
        
        # Update dropdown to show current optimizer status
        self.update_optimizer_dropdown_status()
        
        # Run final backtest with best parameters
        best_seller, best_backtest, best_fitness = self.optimizer.get_best_params()
        if best_seller is not None:
            try:
                if self.raw_data is None:
                    logger.warning("No raw data available, skipping final backtest")
                else:
                    # Build features from RAW data with best parameters
                    feats = build_features(self.raw_data, best_seller, self.current_tf)
                    backtest_result = run_backtest(feats, best_backtest)
                    
                    self.trades_df = backtest_result['trades']
                    self.metrics = backtest_result['metrics']
                    self.update_metrics()
                    self.update_equity_curve()
                    
                    # Create Individual for compatibility
                    best_individual = Individual(
                        seller_params=best_seller,
                        backtest_params=best_backtest,
                        fitness=best_fitness
                    )
                    
                    # Emit signal for chart visualization
                    self.optimization_step_complete.emit(best_individual, backtest_result)
                    
                    # Auto-save best parameters to .env
                    logger.info("=" * 70)
                    logger.info("💾 Auto-saving best parameters...")
                    logger.info("=" * 70)
                    self.apply_best_parameters(auto_save=True)
                
            except Exception as e:
                logger.exception("Error running final backtest: %s", e)
        
        # Always auto-export the final population snapshot
        self._auto_export_population()
    
    def stop_optimization(self):
        """Stop ongoing multi-step optimization (keeps progress)."""
        if self.is_optimizing:
            self.stop_requested = True
            logger.info("⏹ Stop requested... will finish current generation and keep best parameters")
            print("RUN stop_requested")
            
            # Save current evolution parameters to .env (Coach may have modified them)
            self._save_evolution_params_to_env()
    
    @Slot(int, int, str)
    def _emit_progress_signal(self, current: int, total: int, message: str):
        """Emit progress signal from optimizer (thread-safe)."""
        self.progress_updated.emit(current, total, message)
    
    def _save_evolution_params_to_env(self):
        """
        Save current evolution parameters from optimizer to .env file.

        Coach Agent can modify these during optimization, so we save them
        when Stop is clicked to persist any changes.
        """
        if self.optimizer is None:
            logger.debug("No optimizer to save parameters from")
            return

        try:
            from config.settings import SettingsManager

            # Extract current config from optimizer
            config = getattr(self.optimizer, 'config', {})
            if not config:
                logger.warning("Optimizer has no config to save")
                return

            # Build settings dict for GA parameters
            settings_dict = {}

            # Core GA parameters
            if 'mutation_rate' in config:
                settings_dict['ga_mutation_rate'] = float(config['mutation_rate'])
            if 'sigma' in config:
                settings_dict['ga_sigma'] = float(config['sigma'])
            if 'elite_fraction' in config:
                settings_dict['ga_elite_fraction'] = float(config['elite_fraction'])
            if 'tournament_size' in config:
                settings_dict['ga_tournament_size'] = int(config['tournament_size'])
            if 'mutation_probability' in config:
                settings_dict['ga_mutation_probability'] = float(config['mutation_probability'])
            if 'population_size' in config:
                settings_dict['ga_population_size'] = int(config['population_size'])

            if not settings_dict:
                logger.debug("No GA parameters to save")
                return

            # Save to .env
            SettingsManager.save_to_env(settings_dict)
            SettingsManager.reload_settings()

            logger.info("💾 Saved evolution parameters to .env:")
            for key, value in settings_dict.items():
                logger.info(f"  {key}={value}")

            print(f"ENV saved ga_params={len(settings_dict)}")

        except Exception as e:
            logger.exception("Failed to save evolution parameters: %s", e)

    def _save_coach_modified_params_to_env(self):
        """
        Save GA parameters that were modified by the Evolution Coach to .env file.

        This ensures that coach improvements persist across app restarts.
        Called automatically when coach modifies parameters.
        """
        if self.optimizer is None:
            logger.debug("No optimizer to save coach-modified parameters from")
            return

        try:
            from config.settings import SettingsManager

            # Extract current config from optimizer (which may have been modified by coach)
            config = getattr(self.optimizer, 'config', {})
            if not config:
                logger.debug("Optimizer has no config to save")
                return

            # Build settings dict for GA parameters that coach can modify
            coach_modifiable_params = {}

            # Parameters that coach can modify
            if 'mutation_probability' in config:
                coach_modifiable_params['ga_mutation_probability'] = float(config['mutation_probability'])
            if 'mutation_rate' in config:
                coach_modifiable_params['ga_mutation_rate'] = float(config['mutation_rate'])
            if 'sigma' in config:
                coach_modifiable_params['ga_sigma'] = float(config['sigma'])
            if 'elite_fraction' in config:
                coach_modifiable_params['ga_elite_fraction'] = float(config['elite_fraction'])
            if 'tournament_size' in config:
                coach_modifiable_params['ga_tournament_size'] = int(config['tournament_size'])
            if 'population_size' in config:
                coach_modifiable_params['ga_population_size'] = int(config['population_size'])

            if not coach_modifiable_params:
                logger.debug("No coach-modifiable GA parameters to save")
                return

            # Save to .env
            SettingsManager.save_to_env(coach_modifiable_params)
            SettingsManager.reload_settings()

            logger.info("🤖💾 Coach-modified GA parameters saved to .env:")
            for key, value in coach_modifiable_params.items():
                logger.info(f"  {key}={value}")

            logger.info("COACH_ENV saved ga_params=%d", len(coach_modifiable_params))

        except Exception as e:
            logger.exception("Failed to save coach-modified evolution parameters: %s", e)
