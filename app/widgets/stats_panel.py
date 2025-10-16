from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QGridLayout, QPushButton,
    QFileDialog, QScrollArea, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QMetaObject, Q_ARG, Slot
from PySide6.QtGui import QColor
import pandas as pd
import pyqtgraph as pg

from strategy.seller_exhaustion import SellerParams, build_features
from core.models import BacktestParams
from backtest.optimizer import Population, Individual, evolution_step
from backtest.engine import run_backtest
import config.settings as config_settings

# Multi-core CPU optimizer (guaranteed correct)
from backtest.optimizer_multicore import evolution_step_multicore
import multiprocessing

# GPU acceleration imports (optional)
try:
    from backtest.optimizer_gpu import GPUOptimizer, has_gpu
    from backtest.engine_gpu import GPUBacktestAccelerator
    GPU_AVAILABLE = has_gpu()
except ImportError:
    GPU_AVAILABLE = False
    GPUOptimizer = None
    print("⚠ GPU acceleration not available (PyTorch not installed)")


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
        self.population = None
        self.current_data = None  # Features dataframe (with indicators and signals)
        self.raw_data = None  # Raw OHLCV data (for rebuilding features)
        self.current_tf = None
        self.param_editor = None  # Will be set by main window
        
        # Multi-step optimization state
        self.is_optimizing = False
        self.stop_requested = False
        
        # Multi-core CPU optimizer (guaranteed correct, good speedup)
        self.use_multicore = True
        self.n_workers = multiprocessing.cpu_count()
        print(f"✓ Multi-core optimizer initialized ({self.n_workers} workers)")
        
        # GPU optimizer (DISABLED due to trade count bugs)
        # TODO: Fix GPU batch engine to match CPU backtest logic exactly
        self.gpu_optimizer = None
        self.use_gpu = False
        # if GPU_AVAILABLE and GPUOptimizer:
        #     try:
        #         self.gpu_optimizer = GPUOptimizer()
        #         self.use_gpu = self.gpu_optimizer.has_gpu
        #     except Exception as e:
        #         print(f"⚠ Could not initialize GPU optimizer: {e}")
        #         self.use_gpu = False
        
        # Temporary storage for thread results
        self.temp_backtest_result = None
        
        # Track previous best fitness to detect improvements
        self.prev_best_fitness = None
        
        self.init_ui()
        
        # Connect generation_complete signal for thread-safe UI updates
        self.generation_complete.connect(self._update_after_generation)

    def _get_ga_settings(self) -> dict:
        """Return the latest genetic algorithm settings."""
        s = config_settings.settings
        return {
            'population_size': max(2, int(s.ga_population_size)),
            'mutation_rate': float(s.ga_mutation_rate),
            'sigma': float(s.ga_sigma),
            'elite_fraction': max(0.0, min(0.5, float(s.ga_elite_fraction))),
            'tournament_size': max(2, int(s.ga_tournament_size)),
            'mutation_probability': max(0.0, min(1.0, float(s.ga_mutation_probability))),
        }
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Strategy Optimization")
        title.setProperty("role", "title")
        layout.addWidget(title)
        
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
        
        # Fitness evolution plot
        fitness_label = QLabel("<b>Fitness Evolution (Best Individual)</b>")
        layout.addWidget(fitness_label)
        
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
        """Create optimization controls."""
        group = QGroupBox("Evolutionary Optimization")
        layout = QVBoxLayout()
        
        # Status display
        status_layout = QGridLayout()
        
        status_layout.addWidget(QLabel("Generation:"), 0, 0)
        self.gen_label = QLabel("0")
        self.gen_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        status_layout.addWidget(self.gen_label, 0, 1)
        
        status_layout.addWidget(QLabel("Best Fitness:"), 1, 0)
        self.best_fitness_label = QLabel("--")
        self.best_fitness_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4caf50;")
        status_layout.addWidget(self.best_fitness_label, 1, 1)
        
        status_layout.addWidget(QLabel("Population:"), 2, 0)
        self.pop_label = QLabel("Ready to optimize")
        status_layout.addWidget(self.pop_label, 2, 1)
        
        status_layout.addWidget(QLabel("Acceleration:"), 3, 0)
        accel_text = f"Multi-Core CPU ({self.n_workers} workers)" if self.use_multicore else "Single-Core CPU"
        self.accel_label = QLabel(accel_text)
        self.accel_label.setStyleSheet("font-weight: bold; color: #4caf50;" if self.use_multicore else "")
        status_layout.addWidget(self.accel_label, 3, 1)
        
        layout.addLayout(status_layout)
        
        # Number of generations spinner
        gen_layout = QHBoxLayout()
        gen_layout.addWidget(QLabel("Generations:"))
        self.n_generations_spin = QSpinBox()
        self.n_generations_spin.setRange(10, 1000)
        self.n_generations_spin.setValue(50)
        self.n_generations_spin.setToolTip("Number of generations for multi-step optimization")
        gen_layout.addWidget(self.n_generations_spin)
        layout.addLayout(gen_layout)
        
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
        self.optimize_btn.setToolTip("Run evolutionary optimization (auto-initializes population)")
        self.optimize_btn.clicked.connect(self.run_multi_step_optimize)
        layout.addWidget(self.optimize_btn)
        
        # Note: Parameters are auto-applied when a better solution is found during optimization
        
        group.setLayout(layout)
        return group
    
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
            print(f"✓ Fitness preset changed to: {preset_name}")
    
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
            print(f"Error updating stats: {e}")
    
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
        
        if self.population is None or not self.population.history:
            return
        
        # Extract generation and fitness data
        generations = [h['generation'] for h in self.population.history]
        best_fitness = [h['best_fitness'] for h in self.population.history]
        mean_fitness = [h['mean_fitness'] for h in self.population.history]
        
        # Plot best fitness
        self.fitness_plot.plot(
            generations, best_fitness,
            pen=pg.mkPen('#4caf50', width=3),
            symbol='o', symbolSize=8,
            symbolBrush='#4caf50',
            name='Best Fitness'
        )
        
        # Plot mean fitness
        self.fitness_plot.plot(
            generations, mean_fitness,
            pen=pg.mkPen('#ff9800', width=2),
            symbol='s', symbolSize=6,
            symbolBrush='#ff9800',
            name='Mean Fitness'
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
    
    def _initialize_population_if_needed(self):
        """Initialize population with current parameters as seed (called automatically)."""
        if self.current_data is None:
            print("Error: No data loaded.")
            return False
        
        # If population already exists and has the right size, reuse it
        ga_cfg = self._get_ga_settings()
        pop_size = ga_cfg['population_size']
        
        if self.population is not None and self.population.size == pop_size:
            print(f"✓ Reusing existing population (size={pop_size}, gen={self.population.generation})")
            return True
        
        # Get current params from UI
        seller_params, backtest_params, _ = self.get_current_params()
        
        # Create seed individual
        seed = Individual(
            seller_params=seller_params,
            backtest_params=backtest_params
        )
        
        # Initialize population with seed
        self.population = Population(size=pop_size, seed_individual=seed)
        
        # Reset best fitness tracking
        self.prev_best_fitness = None
        
        # Update UI
        self.gen_label.setText(str(self.population.generation))
        self.pop_label.setText(f"{pop_size} individuals")
        
        print(
            "✓ Population initialized with "
            f"{self.population.size} individuals | "
            f"mutation_rate={ga_cfg['mutation_rate']:.3f} | "
            f"sigma={ga_cfg['sigma']:.3f} | "
            f"elite_fraction={ga_cfg['elite_fraction']:.3f} | "
            f"tournament_size={ga_cfg['tournament_size']} | "
            f"mutation_probability={ga_cfg['mutation_probability']:.3f}"
        )
        
        return True
    
    def run_optimization_step(self):
        """Run one generation of evolution asynchronously (internal method)."""
        if self.population is None:
            print("Error: Population not initialized")
            return
        
        if self.current_data is None:
            print("Error: No data loaded")
            return
        
        # Disable optimize button during execution
        self.optimize_btn.setEnabled(False)
        
        # Emit progress signal
        self.progress_updated.emit(0, 1, "Running single evolution step...")
        
        # Run in thread to avoid UI freeze
        from threading import Thread
        
        def _run_single_step():
            try:
                ga_cfg = self._get_ga_settings()
                if self.population.size != ga_cfg['population_size']:
                    print(
                        "⚠ Population size differs from settings "
                        f"({self.population.size} != {ga_cfg['population_size']}). "
                        "Reinitialize the population to apply the new size."
                    )

                # Get fitness configuration from parameter editor
                _, _, fitness_config = self.get_current_params()
                
                # Run evolution step (multi-core if available, else single-core)
                print(f"\n{'='*60}")
                mode = f"Multi-Core CPU ({self.n_workers} workers)" if self.use_multicore else "Single-Core CPU"
                print(f"Running Evolution Step [{mode}]...")
                print(f"Fitness Preset: {fitness_config.preset}")
                print(f"{'='*60}")
                
                # CRITICAL: Pass RAW data, not features!
                data_for_optimizer = self.raw_data if self.raw_data is not None else self.current_data[['open', 'high', 'low', 'close', 'volume']]
                
                if self.use_multicore:
                    # Multi-core CPU (correct + fast)
                    self.population = evolution_step_multicore(
                        self.population,
                        data_for_optimizer,  # Pass raw OHLCV, not features!
                        self.current_tf,
                        fitness_config=fitness_config,
                        mutation_rate=ga_cfg['mutation_rate'],
                        sigma=ga_cfg['sigma'],
                        elite_fraction=ga_cfg['elite_fraction'],
                        tournament_size=ga_cfg['tournament_size'],
                        mutation_probability=ga_cfg['mutation_probability'],
                        n_workers=self.n_workers
                    )
                else:
                    # Single-core CPU fallback
                    self.population = evolution_step(
                        self.population,
                        data_for_optimizer,  # Pass raw OHLCV, not features!
                        self.current_tf,
                        fitness_config=fitness_config,
                        mutation_rate=ga_cfg['mutation_rate'],
                        sigma=ga_cfg['sigma'],
                        elite_fraction=ga_cfg['elite_fraction'],
                        tournament_size=ga_cfg['tournament_size'],
                        mutation_probability=ga_cfg['mutation_probability']
                    )
                
                # Store backtest result for main thread
                backtest_result = None
                
                if self.population.best_ever:
                    best = self.population.best_ever
                    
                    # Show best metrics
                    if best.metrics:
                        print(f"\n🏆 Best Individual Metrics:")
                        print(f"   Fitness: {best.fitness:.4f}")
                        print(f"   Trades: {best.metrics.get('n', 0)}")
                        print(f"   Win Rate: {best.metrics.get('win_rate', 0):.2%}")
                        print(f"   Avg R: {best.metrics.get('avg_R', 0):.2f}")
                        print(f"   Total PnL: ${best.metrics.get('total_pnl', 0):.4f}")
                    
                    # Run backtest with best individual to visualize strategy
                    print(f"\n📊 Running backtest with best parameters...")
                    try:
                        if self.raw_data is None:
                            print("⚠ No raw data available, skipping backtest")
                        else:
                            # Build features with best individual's params from RAW data
                            feats = build_features(self.raw_data, best.seller_params, self.current_tf)
                            
                            # Run backtest
                            backtest_result = run_backtest(feats, best.backtest_params)
                            
                            print(f"✓ Backtest complete: {backtest_result['metrics']['n']} trades")
                        
                    except Exception as e:
                        print(f"Error running backtest for visualization: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Store results for main thread to process
                self.temp_backtest_result = backtest_result
                
                # Emit signal to update UI in main thread
                self.generation_complete.emit()
                
                # Emit progress completion
                self.progress_updated.emit(1, 1, "✓ Evolution step complete")
                
            except Exception as e:
                print(f"Error during evolution step: {e}")
                import traceback
                traceback.print_exc()
                self.progress_updated.emit(0, 1, f"❌ Error: {str(e)}")
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
        """Update UI after generation completes (runs in main thread)."""
        try:
            if self.population is None:
                return
            
            # Update generation label
            self.gen_label.setText(str(self.population.generation))
            
            # Update fitness evolution plot
            self.update_fitness_plot()
            
            # Update best individual display
            if self.population.best_ever:
                best = self.population.best_ever
                self.best_fitness_label.setText(f"{best.fitness:.4f}")
                
                # AUTO-APPLY: Update param editor immediately with best parameters
                print(f"✓ Auto-applying best parameters from generation {best.generation}")
                self.set_params_from_individual(best)
                
                # Update stats display with backtest results if available
                if self.temp_backtest_result:
                    self.trades_df = self.temp_backtest_result['trades']
                    self.metrics = self.temp_backtest_result['metrics']
                    
                    # Update metrics display
                    self.update_metrics()
                    
                    # Update equity curve
                    self.update_equity_curve()
                    
                    # Emit signal with backtest results for chart update
                    self.optimization_step_complete.emit(best, self.temp_backtest_result)
            
            # Update population stats
            stats = self.population.get_stats()
            self.pop_label.setText(
                f"Gen {self.population.generation} | "
                f"Pop: {self.population.size} | "
                f"Mean: {stats['mean_fitness']:.2f} | "
                f"Best: {stats['max_fitness']:.2f}"
            )
            
        except Exception as e:
            print(f"Error updating UI after generation: {e}")
            import traceback
            traceback.print_exc()
    
    def apply_best_parameters(self, auto_save: bool = False):
        """
        Apply best parameters from population to UI and optionally save to settings.
        
        Args:
            auto_save: If True, automatically save parameters to .env file
        """
        if self.population is None or self.population.best_ever is None:
            print("No best individual to apply")
            return
        
        best = self.population.best_ever
        self.set_params_from_individual(best)
        
        print(f"✓ Applied best parameters from generation {best.generation}")
        print(f"  Fitness: {best.fitness:.4f}")
        
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
                    'backtest_atr_stop_mult': float(best.backtest_params.atr_stop_mult),
                    'backtest_reward_r': float(best.backtest_params.reward_r),
                    'backtest_max_hold': int(best.backtest_params.max_hold),
                    'backtest_fee_bp': float(best.backtest_params.fee_bp),
                    'backtest_slippage_bp': float(best.backtest_params.slippage_bp),
                }
                
                SettingsManager.save_to_env(settings_dict)
                SettingsManager.reload_settings()
                
                print("✓ Auto-saved best parameters to .env")
            except Exception as e:
                print(f"⚠ Failed to auto-save parameters: {e}")
    
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
            print("Error: No data loaded. Please download data first.")
            return
        
        if self.is_optimizing:
            print("Already optimizing...")
            return
        
        # Auto-initialize population if needed
        if not self._initialize_population_if_needed():
            return
        
        n_gens = self.n_generations_spin.value()
        
        print(f"\n{'='*70}")
        print(f"🚀 Starting Multi-Step Optimization: {n_gens} generations")
        print(f"{'='*70}")
        
        # Show GPU recommendations
        try:
            from backtest.gpu_manager import get_gpu_manager
            gpu_mgr = get_gpu_manager()
            recs = gpu_mgr.get_recommendations(len(self.current_data))
            
            if recs['available']:
                print(f"GPU: {recs['device']}")
                print(f"VRAM: {recs['memory']['free_gb']:.2f} GB free")
                print(f"Expected speedup: {recs['estimated_speedup']:.1f}x")
            print()
        except:
            pass
        
        # Reset best fitness tracking for this optimization run
        self.prev_best_fitness = None
        if self.population.best_ever:
            self.prev_best_fitness = self.population.best_ever.fitness
        
        # Update UI state
        self.is_optimizing = True
        self.stop_requested = False
        
        self.optimize_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)  # Enable stop button during optimization
        self.n_generations_spin.setEnabled(False)
        self.fitness_preset_combo.setEnabled(False)
        
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
        try:
            import time
            start_time = time.time()
            
            for gen in range(n_gens):
                if self.stop_requested:
                    self.progress_updated.emit(gen, n_gens, "⏹ Stopped by user")
                    print(f"\n⏹ Optimization stopped at generation {gen} (keeping best individual)")
                    break
                
                # Update progress
                elapsed = time.time() - start_time
                avg_time = elapsed / (gen + 1) if gen > 0 else 1
                remaining = avg_time * (n_gens - gen - 1)
                
                msg = f"Generation {gen+1}/{n_gens} | ETA: {remaining:.0f}s"
                self.progress_updated.emit(gen, n_gens, msg)
                
                # Run one evolution step
                try:
                    ga_cfg = self._get_ga_settings()
                    
                    # Get fitness configuration from parameter editor
                    _, _, fitness_config = self.get_current_params()
                    
                    # CRITICAL FIX: Pass raw_data (OHLCV only), not features!
                    # Optimizer MUST build features with each individual's parameters
                    data_for_optimizer = self.raw_data if self.raw_data is not None else self.current_data[['open', 'high', 'low', 'close', 'volume']]
                    
                    if self.use_multicore:
                        # Multi-core CPU (correct + fast)
                        self.population = evolution_step_multicore(
                            self.population,
                            data_for_optimizer,  # Pass raw OHLCV, NOT features!
                            self.current_tf,
                            fitness_config=fitness_config,
                            mutation_rate=ga_cfg['mutation_rate'],
                            sigma=ga_cfg['sigma'],
                            elite_fraction=ga_cfg['elite_fraction'],
                            tournament_size=ga_cfg['tournament_size'],
                            mutation_probability=ga_cfg['mutation_probability'],
                            n_workers=self.n_workers
                        )
                    else:
                        # Single-core CPU fallback
                        self.population = evolution_step(
                            self.population,
                            data_for_optimizer,  # Pass raw OHLCV, NOT features!
                            self.current_tf,
                            fitness_config=fitness_config,
                            mutation_rate=ga_cfg['mutation_rate'],
                            sigma=ga_cfg['sigma'],
                            elite_fraction=ga_cfg['elite_fraction'],
                            tournament_size=ga_cfg['tournament_size'],
                            mutation_probability=ga_cfg['mutation_probability']
                        )
                    
                    # Check if we have a new best individual
                    new_best_found = False
                    if self.population.best_ever:
                        current_best_fitness = self.population.best_ever.fitness
                        if self.prev_best_fitness is None or current_best_fitness > self.prev_best_fitness:
                            new_best_found = True
                            self.prev_best_fitness = current_best_fitness
                            
                            # Run backtest with new best individual to visualize
                            print(f"\n🎯 NEW BEST found in generation {gen+1}!")
                            print(f"   Fitness: {current_best_fitness:.4f}")
                            
                            best = self.population.best_ever
                            if best.metrics:
                                print(f"   Trades: {best.metrics.get('n', 0)}")
                                print(f"   Win Rate: {best.metrics.get('win_rate', 0):.2%}")
                                print(f"   Avg R: {best.metrics.get('avg_R', 0):.2f}")
                                print(f"   Total PnL: ${best.metrics.get('total_pnl', 0):.4f}")
                            
                            # Run backtest for visualization using RAW data
                            print(f"\n📊 Running backtest with new best parameters...")
                            
                            try:
                                if self.raw_data is None:
                                    print("⚠ No raw data available, skipping visualization backtest")
                                    self.temp_backtest_result = None
                                else:
                                    # Rebuild features from raw OHLCV data with new parameters
                                    feats = build_features(self.raw_data, best.seller_params, self.current_tf)
                                    backtest_result = run_backtest(feats, best.backtest_params)
                                    
                                    print(f"✓ Backtest complete: {backtest_result['metrics']['n']} trades")
                                    
                                    # Store for UI update
                                    self.temp_backtest_result = backtest_result
                                
                            except Exception as e:
                                print(f"Error running backtest: {e}")
                                import traceback
                                traceback.print_exc()
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
                    print(f"Error in generation {gen+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # Final progress update
            total_time = time.time() - start_time
            if not self.stop_requested:
                self.progress_updated.emit(n_gens, n_gens, f"✓ Complete in {total_time:.1f}s")
                print(f"\n{'='*70}")
                print(f"✓ Optimization complete! {n_gens} generations in {total_time:.1f}s")
                print(f"  Average: {total_time/n_gens:.2f}s per generation")
                print(f"{'='*70}\n")
                
                # Show final best results
                if self.population.best_ever:
                    best = self.population.best_ever
                    print(f"\n🏆 Final Best Individual:")
                    print(f"   Fitness: {best.fitness:.4f}")
                    if best.metrics:
                        print(f"   Trades: {best.metrics.get('n', 0)}")
                        print(f"   Win Rate: {best.metrics.get('win_rate', 0):.2%}")
                        print(f"   Avg R: {best.metrics.get('avg_R', 0):.2f}")
                        print(f"   Total PnL: ${best.metrics.get('total_pnl', 0):.4f}\n")
        
        except Exception as e:
            print(f"Error in multi-step optimization: {e}")
            import traceback
            traceback.print_exc()
            self.progress_updated.emit(0, n_gens, f"❌ Error: {str(e)}")
        
        finally:
            # Restore UI state (from main thread)
            QMetaObject.invokeMethod(
                self,
                "_restore_ui_after_optimize",
                Qt.QueuedConnection
            )
    
    def _update_after_generation(self):
        """Update UI after each generation (called from main thread)."""
        try:
            # Update generation label
            self.gen_label.setText(str(self.population.generation))
            
            # Update fitness plot
            self.update_fitness_plot()
            
            # Update best fitness and auto-apply parameters
            if self.population.best_ever:
                best = self.population.best_ever
                self.best_fitness_label.setText(f"{best.fitness:.4f}")
                
                # AUTO-APPLY: Update param editor immediately with best parameters
                self.set_params_from_individual(best)
        except Exception as e:
            print(f"Error updating UI: {e}")
    
    @Slot(bool)
    def _update_after_multi_step_generation(self, new_best_found: bool):
        """
        Update UI after each generation during multi-step optimization.
        
        Args:
            new_best_found: True if a new best individual was found this generation
        """
        try:
            if self.population is None:
                return
            
            # Always update generation label and fitness plot
            self.gen_label.setText(str(self.population.generation))
            self.update_fitness_plot()
            
            # Update population stats
            stats = self.population.get_stats()
            self.pop_label.setText(
                f"Gen {self.population.generation} | "
                f"Pop: {self.population.size} | "
                f"Mean: {stats['mean_fitness']:.2f} | "
                f"Best: {stats['max_fitness']:.2f}"
            )
            
            # If new best found, update all displays and chart
            if new_best_found and self.population.best_ever:
                best = self.population.best_ever
                self.best_fitness_label.setText(f"{best.fitness:.4f}")
                
                # AUTO-APPLY: Update param editor immediately with best parameters
                print(f"✓ Auto-applying new best parameters from generation {best.generation}")
                self.set_params_from_individual(best)
                
                # Update stats display with backtest results if available
                if self.temp_backtest_result:
                    self.trades_df = self.temp_backtest_result['trades']
                    self.metrics = self.temp_backtest_result['metrics']
                    
                    # Update metrics display
                    self.update_metrics()
                    
                    # Update equity curve
                    self.update_equity_curve()
                    
                    # Emit signal with backtest results for chart update
                    print(f"✓ Updating chart with new best strategy...")
                    self.optimization_step_complete.emit(best, self.temp_backtest_result)
                else:
                    print(f"⚠ No backtest result available for new best")
            
        except Exception as e:
            print(f"Error updating UI after multi-step generation: {e}")
            import traceback
            traceback.print_exc()
    
    @Slot()
    def _restore_ui_after_optimize(self):
        """Restore UI state after optimization completes (called from main thread)."""
        self.is_optimizing = False
        self.optimize_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)  # Disable stop button after optimization
        self.n_generations_spin.setEnabled(True)
        self.fitness_preset_combo.setEnabled(True)
        
        # Run final backtest with best individual
        if self.population.best_ever:
            try:
                best = self.population.best_ever
                
                if self.raw_data is None:
                    print("⚠ No raw data available, skipping final backtest")
                else:
                    # Build features from RAW data with best parameters
                    feats = build_features(self.raw_data, best.seller_params, self.current_tf)
                    backtest_result = run_backtest(feats, best.backtest_params)
                    
                    self.trades_df = backtest_result['trades']
                    self.metrics = backtest_result['metrics']
                    self.update_metrics()
                    self.update_equity_curve()
                    
                    # Emit signal for chart visualization
                    self.optimization_step_complete.emit(best, backtest_result)
                    
                    # Auto-save best parameters to .env
                    print("\n" + "="*70)
                    print("💾 Auto-saving best parameters...")
                    print("="*70)
                    self.apply_best_parameters(auto_save=True)
                
            except Exception as e:
                print(f"Error running final backtest: {e}")
    
    def stop_optimization(self):
        """Stop ongoing multi-step optimization (keeps progress)."""
        if self.is_optimizing:
            self.stop_requested = True
            print("\n⏹ Stop requested... will finish current generation and keep best individual")
