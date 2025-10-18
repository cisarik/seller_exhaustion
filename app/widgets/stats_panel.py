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

from strategy.seller_exhaustion import SellerParams, build_features
from core.models import BacktestParams
from backtest.optimizer import Individual
from backtest.engine import run_backtest
import config.settings as config_settings
from core.logging_utils import get_logger
from core.coach_logging import coach_log_manager
import os

# New modular optimizer system
from backtest.optimizer_base import BaseOptimizer
from backtest.optimizer_factory import (
    create_optimizer,
    get_available_optimizers,
    get_available_accelerations,
    get_optimizer_display_name,
    get_acceleration_display_name
)
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
        
        self.init_ui()
        
        # Connect generation_complete signal for thread-safe UI updates
        self.generation_complete.connect(self._update_after_generation)

    def set_initial_population_file(self, path: str | None):
        """Set a population file to initialize the optimizer from (one-time)."""
        self.initial_population_file = path

    # ---------------- Auto Export Helpers ----------------
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
        """Log optimizer type, acceleration, and hyperparameters."""
        if self.optimizer is None:
            return

        try:
            logger.info(
                "%sOptimizer: name=%s | acceleration=%s",
                prefix,
                self.optimizer.get_optimizer_name(),
                self.optimizer.get_acceleration_mode(),
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
        
        # Get optimizer type and acceleration from UI
        optimizer_type = self.optimizer_type_combo.currentData()
        acceleration = self.acceleration_combo.currentData()
        
        # If optimizer already exists with same type/acceleration, reuse it
        if self.optimizer is not None:
            if (self.optimizer.get_optimizer_name().lower().replace(' ', '_') == optimizer_type or
                (optimizer_type == 'evolutionary' and 'evolutionary' in self.optimizer.get_optimizer_name().lower())):
                logger.info("Reusing existing optimizer (%s)", self.optimizer.get_optimizer_name())
                self._log_optimizer_config(prefix="  ")
                coach_log_manager.append(
                    f"OPT reuse name={self.optimizer.get_optimizer_name()} accel={self.optimizer.get_acceleration_mode()}"
                )
                return True
        
        # Get current params from UI as seed
        seller_params, backtest_params, _ = self.get_current_params()
        
        # Create new optimizer via factory
        try:
            self.optimizer = create_optimizer(
                optimizer_type=optimizer_type,
                acceleration=acceleration,
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
                "Optimizer initialized | type=%s | acceleration=%s",
                self.optimizer.get_optimizer_name(),
                self.optimizer.get_acceleration_mode(),
            )
            self._log_parameter_snapshot(seller_params, backtest_params, prefix="Seed ")
            self._log_optimizer_config(prefix="  ")
            coach_log_manager.append(
                f"OPT init name={self.optimizer.get_optimizer_name()} accel={self.optimizer.get_acceleration_mode()}"
            )
            # Optimizer hyperparameters
            hp = getattr(self.optimizer, 'config', {}) or {}
            if isinstance(hp, dict) and hp:
                coach_log_manager.append(f"OPT hp {self._format_dict_compact(hp, float_dp=3)}")
            # Population status
            pop = getattr(self.optimizer, 'population', None)
            if pop is not None:
                coach_log_manager.append(
                    f"POP status size={getattr(pop, 'size', '--')} gen={getattr(pop, 'generation', '--')}"
                )
            coach_log_manager.append(
                f"SEED seller[{self._compact_seller_params(seller_params)}]"
            )
            coach_log_manager.append(
                f"SEED backtest[{self._compact_backtest_params(backtest_params)}]"
            )
            # Full seed dumps for agent completeness
            from dataclasses import asdict as _asdict
            full_seller = _asdict(seller_params)
            full_backtest = (
                backtest_params.model_dump() if hasattr(backtest_params, 'model_dump') else dict(backtest_params)
            )
            coach_log_manager.append(f"SEED SellerParams: {self._format_dict_compact(full_seller)}")
            coach_log_manager.append(f"SEED BacktestParams: {self._format_dict_compact(full_backtest)}")
            
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
        
        # Acceleration Mode Selection
        accel_layout = QHBoxLayout()
        accel_layout.addWidget(QLabel("Acceleration:"))
        self.acceleration_combo = self._create_acceleration_combo()
        self.acceleration_combo.currentIndexChanged.connect(self._on_acceleration_changed)
        accel_layout.addWidget(self.acceleration_combo, stretch=1)
        layout.addLayout(accel_layout)
        
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
    
    def _create_acceleration_combo(self):
        """Create acceleration mode dropdown."""
        from PySide6.QtWidgets import QComboBox
        
        combo = QComboBox()
        
        # Populate based on first optimizer (evolutionary)
        optimizer_type = "evolutionary"
        available_accels = get_available_accelerations(optimizer_type)
        
        for accel in available_accels:
            display_name = get_acceleration_display_name(accel, optimizer_type)
            combo.addItem(display_name, accel)
        
        # Default to multicore if available
        if 'multicore' in available_accels:
            for i in range(combo.count()):
                if combo.itemData(i) == 'multicore':
                    combo.setCurrentIndex(i)
                    break
        
        combo.setToolTip("Select acceleration mode")
        
        return combo
    
    def _on_optimizer_type_changed(self, index):
        """Handle optimizer type change - update available accelerations."""
        optimizer_type = self.optimizer_type_combo.currentData()
        
        # Update acceleration combo
        self.acceleration_combo.blockSignals(True)
        self.acceleration_combo.clear()
        
        available_accels = get_available_accelerations(optimizer_type)
        for accel in available_accels:
            display_name = get_acceleration_display_name(accel, optimizer_type)
            self.acceleration_combo.addItem(display_name, accel)
        
        # Default to best option
        if optimizer_type == 'evolutionary' and 'multicore' in available_accels:
            for i in range(self.acceleration_combo.count()):
                if self.acceleration_combo.itemData(i) == 'multicore':
                    self.acceleration_combo.setCurrentIndex(i)
                    break
        
        self.acceleration_combo.blockSignals(False)
        
        # Reset optimizer (will be re-initialized on next run)
        self.optimizer = None
        
        logger.info("Optimizer type changed to: %s", get_optimizer_display_name(optimizer_type))
    
    def _on_acceleration_changed(self, index):
        """Handle acceleration change."""
        acceleration = self.acceleration_combo.currentData()
        
        # Reset optimizer (will be re-initialized on next run)
        self.optimizer = None
        
        logger.info("Acceleration changed to: %s", self.acceleration_combo.currentText())
    
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
        
        # Extract iteration and fitness data
        iterations = [h['iteration'] for h in history]
        best_fitness = [h.get('best_fitness', h.get('fitness', 0)) for h in history]
        
        # For evolutionary algorithms: mean_fitness (population average)
        # For ADAM: current fitness (iteration-by-iteration fitness)
        mean_fitness = [h.get('mean_fitness') for h in history]
        current_fitness = [h.get('fitness', 0) for h in history]
        
        # Plot best fitness (line only, no symbols)
        self.fitness_plot.plot(
            iterations, best_fitness,
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
                iterations, clean_mean,
                pen=pg.mkPen('#ff9800', width=2),
                name='Avg Fitness'
            )
        elif len(current_fitness) > 0:
            # ADAM - plot current fitness
            self.fitness_plot.plot(
                iterations, current_fitness,
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
                coach_log_manager.append(
                    f"STEP begin accel={self.optimizer.get_acceleration_mode()} preset={getattr(fitness_config, 'preset', '--')}"
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
                        feats = build_features(self.raw_data, best_seller, self.current_tf)
                        
                        # Run backtest
                        backtest_result = run_backtest(feats, best_backtest)
                        
                        logger.info("Backtest complete: %d trades", backtest_result['metrics']['n'])
                        self._log_metrics_snapshot(backtest_result.get('metrics', {}), prefix="  ")
                        self._log_parameter_snapshot(best_seller, best_backtest, prefix="  ")
                        metrics_payload = backtest_result.get('metrics') if backtest_result else {}
                        coach_log_manager.append(
                            "STEP best "
                            f"fitness={best_fitness:.4f} "
                            f"{self._compact_metrics(metrics_payload)}"
                        )
                    
                except Exception as e:
                    logger.exception("Error running backtest for visualization: %s", e)
                    coach_log_manager.append(f"ERROR backtest {e}")
                
                # Store results for main thread to process
                self.temp_backtest_result = backtest_result
                
                # Emit signal to update UI in main thread
                self.generation_complete.emit()
                
                # Emit progress completion
                self.progress_updated.emit(1, 1, "✓ Optimization step complete")
                coach_log_manager.append("STEP done")
                
            except Exception as e:
                logger.exception("Error during optimization step: %s", e)
                self.progress_updated.emit(0, 1, f"❌ Error: {str(e)}")
                coach_log_manager.append(f"ERROR step {e}")
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
            
            # Get current iteration from optimizer stats
            stats = self.optimizer.get_stats()
            iteration = stats.get('iteration', stats.get('generation', 0))
            
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
                logger.info("Auto-applying best parameters from iteration %s", iteration)
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
        
        # Get iteration from stats
        stats = self.optimizer.get_stats()
        iteration = stats.get('iteration', stats.get('generation', 0))
        
        logger.info(
            "Applied best parameters from iteration %s | Fitness=%.4f",
            iteration,
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
        
        # Auto-initialize optimizer if needed
        if not self._initialize_optimizer_if_needed():
            return
        
        seed_seller, seed_backtest, fitness_config = self.get_current_params()
        self._log_parameter_snapshot(seed_seller, seed_backtest, prefix="Seed ")
        self._log_fitness_config(fitness_config, prefix="Seed ")
        self._log_optimizer_config(prefix="  ")
        coach_log_manager.append(
            f"FIT preset={getattr(fitness_config, 'preset', '--')}"
        )
        # Full fitness config dump
        if hasattr(fitness_config, 'model_dump'):
            fit_payload = fitness_config.model_dump()
        else:
            fit_payload = getattr(fitness_config, '__dict__', {})
        if isinstance(fit_payload, dict) and fit_payload:
            coach_log_manager.append(f"FIT cfg[{self._format_dict_compact(fit_payload)}]")
        coach_log_manager.append(
            f"SEED seller[{self._compact_seller_params(seed_seller)}]"
        )
        coach_log_manager.append(
            f"SEED backtest[{self._compact_backtest_params(seed_backtest)}]"
        )
        
        from config.settings import settings
        n_iters = settings.optimizer_iterations
        
        logger.info("=" * 70)
        logger.info("🚀 Starting Multi-Step Optimization: %s iterations", n_iters)
        logger.info("=" * 70)
        coach_log_manager.append(
            f"RUN start iters={n_iters} accel={self.optimizer.get_acceleration_mode()}"
        )
        
        # Show GPU recommendations
        try:
            from backtest.gpu_manager import get_gpu_manager
            gpu_mgr = get_gpu_manager()
            recs = gpu_mgr.get_recommendations(len(self.current_data))
            
            if recs['available']:
                logger.info("GPU: %s", recs['device'])
                logger.info("VRAM: %.2f GB free", recs['memory']['free_gb'])
                logger.info("Expected speedup: %.1fx", recs['estimated_speedup'])
            logger.info("")
        except:
            pass
        
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
        self.acceleration_combo.setEnabled(False)
        
        # Emit initial progress
        self.progress_updated.emit(0, n_iters, "Initializing optimization...")
        
        # Run optimization in separate thread to not block UI
        import threading
        thread = threading.Thread(
            target=self._run_multi_step_thread,
            args=(n_iters,),
            daemon=True
        )
        thread.start()
    
    def _run_multi_step_thread(self, n_gens: int):
        """
        Worker thread for multi-step optimization.
        
        This runs in background thread so UI stays responsive.
        """
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
                    if result.additional_info:
                        population_stats = result.additional_info.get('population_stats')
                        if isinstance(population_stats, dict) and population_stats:
                            logger.info(
                                "Population stats | generation=%s | mean_fitness=%.4f | std_fitness=%.4f | min=%.4f | max=%.4f",
                                result.iteration,
                                population_stats.get('mean_fitness', 0.0),
                                population_stats.get('std_fitness', 0.0),
                                population_stats.get('min_fitness', 0.0),
                                population_stats.get('max_fitness', 0.0),
                            )
                            coach_log_manager.append(
                                "STAT "
                                f"gen={result.iteration} "
                                f"mean={population_stats.get('mean_fitness', 0.0):.4f} "
                                f"std={population_stats.get('std_fitness', 0.0):.4f} "
                                f"min={population_stats.get('min_fitness', 0.0):.4f} "
                                f"max={population_stats.get('max_fitness', 0.0):.4f}"
                            )
                    
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
                        logger.info("🎯 New best found in iteration %s | Fitness=%.4f", gen + 1, current_best_fitness)
                        self._log_metrics_snapshot(best_metrics, prefix="  ")
                        if best_seller is not None and best_backtest is not None:
                            self._log_parameter_snapshot(best_seller, best_backtest, prefix="  ")
                        coach_log_manager.append(
                            "BEST "
                            f"gen={gen+1} fitness={current_best_fitness:.4f} "
                            f"{self._compact_metrics(best_metrics)}"
                        )
                        # Include full param snapshot for best
                        if best_seller is not None:
                            from dataclasses import asdict as _asdict
                            coach_log_manager.append(
                                f"BEST seller[{self._format_dict_compact(_asdict(best_seller))}]"
                            )
                        if best_backtest is not None:
                            btp = best_backtest.model_dump() if hasattr(best_backtest, 'model_dump') else dict(best_backtest)
                            coach_log_manager.append(
                                f"BEST backtest_params[{self._format_dict_compact(btp)}]"
                            )
                        
                        # Run backtest for visualization using RAW data
                        logger.info("Running backtest with new best parameters...")
                        
                        try:
                            if self.raw_data is None:
                                logger.warning("No raw data available, skipping visualization backtest")
                                self.temp_backtest_result = None
                                coach_log_manager.append("WARN no_raw_data")
                            else:
                                # Rebuild features from raw OHLCV data with new parameters
                                feats = build_features(self.raw_data, best_seller, self.current_tf)
                                backtest_result = run_backtest(feats, best_backtest)
                                
                                logger.info("Backtest complete: %d trades", backtest_result['metrics']['n'])
                                
                                # Store for UI update
                                self.temp_backtest_result = backtest_result
                                metrics_payload = backtest_result.get('metrics', {}) if backtest_result else {}
                                coach_log_manager.append(
                                    "BEST backtest "
                                    f"gen={gen+1} {self._compact_metrics(metrics_payload)}"
                                )
                            
                        except Exception as e:
                            logger.exception("Error running visualization backtest: %s", e)
                            coach_log_manager.append(f"ERROR backtest {e}")
                            self.temp_backtest_result = None
                    
                    # Update UI (from main thread)
                    # Pass whether this is a new best to the UI updater
                    QMetaObject.invokeMethod(
                        self,
                        "_update_after_multi_step_generation",
                        Qt.QueuedConnection,
                        Q_ARG(bool, new_best_found)
                    )
                    
                except Exception as e:
                    logger.exception("Error in generation %s: %s", gen + 1, e)
                    coach_log_manager.append(f"ERROR generation_{gen+1} {e}")
                    break
            
            # Final progress update
            total_time = time.time() - start_time
            if not self.stop_requested:
                self.progress_updated.emit(n_gens, n_gens, f"✓ Complete in {total_time:.1f}s")
                logger.info("=" * 70)
                logger.info("✓ Optimization complete! %s iterations in %.1fs", n_gens, total_time)
                logger.info("  Average: %.2fs per iteration", total_time / n_gens if n_gens else 0.0)
                logger.info("=" * 70)
                coach_log_manager.append(
                    f"RUN done time={total_time:.1f}s avg={(total_time / n_gens if n_gens else 0.0):.2f}s"
                )
                
                # Show final best results
                best_seller, best_backtest, best_fitness = self.optimizer.get_best_params()
                if best_seller is not None:
                    logger.info("🏆 Final Best Parameters | Fitness=%.4f", best_fitness or 0.0)
                    
                    # Get metrics from optimizer stats if available
                    stats = self.optimizer.get_stats()
                    best_metrics = stats.get('best_metrics') if isinstance(stats, dict) else None
                    self._log_metrics_snapshot(best_metrics or {}, prefix="  ")
                    self._log_parameter_snapshot(best_seller, best_backtest, prefix="  ")
                    coach_log_manager.append(
                        "BEST final "
                        f"fitness={(best_fitness or 0.0):.4f} {self._compact_metrics(best_metrics or {})}"
                    )
        
        except Exception as e:
            logger.exception("Error in multi-step optimization: %s", e)
            self.progress_updated.emit(0, n_gens, f"❌ Error: {str(e)}")
            coach_log_manager.append(f"ERROR run {e}")
        
        finally:
            # Restore UI state (from main thread)
            QMetaObject.invokeMethod(
                self,
                "_restore_ui_after_optimize",
                Qt.QueuedConnection
            )
    
    @Slot(bool)
    def _update_after_multi_step_generation(self, new_best_found: bool):
        """
        Update UI after each iteration during multi-step optimization.
        
        Args:
            new_best_found: True if a new best was found this iteration
        """
        try:
            if self.optimizer is None:
                return
            
            # Get current iteration
            stats = self.optimizer.get_stats()
            iteration = stats.get('iteration', stats.get('generation', 0))
            
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
                logger.info("Auto-applying new best parameters from iteration %s", iteration)
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
            logger.exception("Error updating UI after multi-step iteration: %s", e)
    
    @Slot()
    @Slot()
    def _restore_ui_after_optimize(self):
        """Restore UI state after optimization completes (called from main thread)."""
        self.is_optimizing = False
        self.optimize_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)  # Disable stop button after optimization
        self.fitness_preset_combo.setEnabled(True)
        self.optimizer_type_combo.setEnabled(True)
        self.acceleration_combo.setEnabled(True)
        
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
            logger.info("⏹ Stop requested... will finish current iteration and keep best parameters")
            coach_log_manager.append("RUN stop_requested")
    
    @Slot(int, int, str)
    def _emit_progress_signal(self, current: int, total: int, message: str):
        """Emit progress signal from optimizer (thread-safe)."""
        self.progress_updated.emit(current, total, message)
