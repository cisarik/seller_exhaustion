from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QGridLayout, QPushButton,
    QFileDialog, QScrollArea, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
import pandas as pd
import pyqtgraph as pg

from strategy.seller_exhaustion import SellerParams, build_features
from core.models import BacktestParams
from backtest.optimizer import Population, Individual, evolution_step
from backtest.engine import run_backtest
import config.settings as config_settings

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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.trades_df = None
        self.metrics = None
        self.population = None
        self.current_data = None
        self.current_tf = None
        
        # GPU optimizer (if available)
        self.gpu_optimizer = None
        self.use_gpu = False
        if GPU_AVAILABLE and GPUOptimizer:
            try:
                self.gpu_optimizer = GPUOptimizer()
                self.use_gpu = self.gpu_optimizer.has_gpu
            except Exception as e:
                print(f"⚠ Could not initialize GPU optimizer: {e}")
                self.use_gpu = False
        
        self.init_ui()

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
        
        # Parameters section (NEW)
        self.params_group = self.create_parameters_section()
        layout.addWidget(self.params_group, stretch=1)
        
        # Optimization controls (NEW)
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
    
    def create_parameters_section(self):
        """Create parameters display with editable values."""
        group = QGroupBox("Strategy Parameters")
        layout = QVBoxLayout()
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(400)
        
        params_widget = QWidget()
        params_layout = QGridLayout(params_widget)
        
        row = 0
        
        # SellerParams section
        seller_label = QLabel("<b>Seller Exhaustion Parameters:</b>")
        params_layout.addWidget(seller_label, row, 0, 1, 3)
        row += 1
        
        self.param_widgets = {}
        
        # Define parameters with labels and types
        seller_params_config = [
            ('ema_fast', 'EMA Fast (bars)', 48, 192, 1),
            ('ema_slow', 'EMA Slow (bars)', 336, 1344, 1),
            ('z_window', 'Z-Score Window (bars)', 336, 1344, 1),
            ('vol_z', 'Volume Z-Score Threshold', 1.0, 3.5, 0.1),
            ('tr_z', 'True Range Z-Score', 0.8, 2.0, 0.1),
            ('cloc_min', 'Close Location Min', 0.4, 0.8, 0.01),
            ('atr_window', 'ATR Window (bars)', 48, 192, 1),
        ]
        
        for param_name, label_text, min_val, max_val, step in seller_params_config:
            label = QLabel(f"{label_text}:")
            params_layout.addWidget(label, row, 0)
            
            if isinstance(step, int):
                widget = QSpinBox()
                widget.setRange(int(min_val), int(max_val))
                widget.setSingleStep(step)
            else:
                widget = QDoubleSpinBox()
                widget.setRange(min_val, max_val)
                widget.setSingleStep(step)
                widget.setDecimals(2)
            
            params_layout.addWidget(widget, row, 1)
            self.param_widgets[param_name] = widget
            row += 1
        
        # BacktestParams section
        row += 1
        backtest_label = QLabel("<b>Backtest Parameters:</b>")
        params_layout.addWidget(backtest_label, row, 0, 1, 3)
        row += 1
        
        backtest_params_config = [
            ('atr_stop_mult', 'ATR Stop Multiplier', 0.3, 1.5, 0.05),
            ('reward_r', 'Reward:Risk Ratio', 1.5, 4.0, 0.1),
            ('max_hold', 'Max Hold (bars)', 48, 192, 1),
            ('fee_bp', 'Fee (basis points)', 2.0, 10.0, 0.5),
            ('slippage_bp', 'Slippage (basis points)', 2.0, 10.0, 0.5),
        ]
        
        for param_name, label_text, min_val, max_val, step in backtest_params_config:
            label = QLabel(f"{label_text}:")
            params_layout.addWidget(label, row, 0)
            
            if isinstance(step, int):
                widget = QSpinBox()
                widget.setRange(int(min_val), int(max_val))
                widget.setSingleStep(step)
            else:
                widget = QDoubleSpinBox()
                widget.setRange(min_val, max_val)
                widget.setSingleStep(step)
                widget.setDecimals(2)
            
            params_layout.addWidget(widget, row, 1)
            self.param_widgets[param_name] = widget
            row += 1
        
        scroll.setWidget(params_widget)
        layout.addWidget(scroll)
        
        group.setLayout(layout)
        return group
    
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
        self.pop_label = QLabel("Not initialized")
        status_layout.addWidget(self.pop_label, 2, 1)
        
        status_layout.addWidget(QLabel("Acceleration:"), 3, 0)
        self.accel_label = QLabel("CPU" if not GPU_AVAILABLE else f"GPU ({'CUDA' if self.use_gpu else 'Not available'})")
        self.accel_label.setStyleSheet("font-weight: bold; color: #4caf50;" if self.use_gpu else "")
        status_layout.addWidget(self.accel_label, 3, 1)
        
        layout.addLayout(status_layout)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.init_pop_btn = QPushButton("Initialize Population")
        self.init_pop_btn.clicked.connect(self.initialize_population)
        btn_layout.addWidget(self.init_pop_btn)
        
        self.step_btn = QPushButton("Step")
        self.step_btn.setObjectName("primaryButton")
        self.step_btn.clicked.connect(self.run_optimization_step)
        self.step_btn.setEnabled(False)
        btn_layout.addWidget(self.step_btn)
        
        self.optimize_btn = QPushButton("Optimize")
        self.optimize_btn.setToolTip("Run multiple generations (coming soon)")
        self.optimize_btn.setEnabled(False)
        btn_layout.addWidget(self.optimize_btn)
        
        layout.addLayout(btn_layout)
        
        # Apply best button
        apply_layout = QHBoxLayout()
        apply_layout.addStretch()
        
        self.apply_best_btn = QPushButton("Apply Best Parameters")
        self.apply_best_btn.clicked.connect(self.apply_best_parameters)
        self.apply_best_btn.setEnabled(False)
        apply_layout.addWidget(self.apply_best_btn)
        
        layout.addLayout(apply_layout)
        
        group.setLayout(layout)
        return group
    
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
    
    def set_current_data(self, data, tf):
        """Store current data and timeframe for optimization."""
        self.current_data = data
        self.current_tf = tf
        self.init_pop_btn.setEnabled(True)
    
    def get_current_params(self):
        """Get current parameters from UI widgets."""
        seller_params = SellerParams(
            ema_fast=self.param_widgets['ema_fast'].value(),
            ema_slow=self.param_widgets['ema_slow'].value(),
            z_window=self.param_widgets['z_window'].value(),
            vol_z=self.param_widgets['vol_z'].value(),
            tr_z=self.param_widgets['tr_z'].value(),
            cloc_min=self.param_widgets['cloc_min'].value(),
            atr_window=self.param_widgets['atr_window'].value(),
        )
        
        backtest_params = BacktestParams(
            atr_stop_mult=self.param_widgets['atr_stop_mult'].value(),
            reward_r=self.param_widgets['reward_r'].value(),
            max_hold=self.param_widgets['max_hold'].value(),
            fee_bp=self.param_widgets['fee_bp'].value(),
            slippage_bp=self.param_widgets['slippage_bp'].value(),
        )
        
        return seller_params, backtest_params
    
    def set_params_from_individual(self, individual: Individual):
        """Update UI widgets from an individual."""
        sp = individual.seller_params
        bp = individual.backtest_params
        
        # Update SellerParams
        self.param_widgets['ema_fast'].setValue(sp.ema_fast)
        self.param_widgets['ema_slow'].setValue(sp.ema_slow)
        self.param_widgets['z_window'].setValue(sp.z_window)
        self.param_widgets['vol_z'].setValue(sp.vol_z)
        self.param_widgets['tr_z'].setValue(sp.tr_z)
        self.param_widgets['cloc_min'].setValue(sp.cloc_min)
        self.param_widgets['atr_window'].setValue(sp.atr_window)
        
        # Update BacktestParams
        self.param_widgets['atr_stop_mult'].setValue(bp.atr_stop_mult)
        self.param_widgets['reward_r'].setValue(bp.reward_r)
        self.param_widgets['max_hold'].setValue(bp.max_hold)
        self.param_widgets['fee_bp'].setValue(bp.fee_bp)
        self.param_widgets['slippage_bp'].setValue(bp.slippage_bp)
    
    def initialize_population(self):
        """Initialize population with current parameters as seed."""
        if self.current_data is None:
            print("Error: No data loaded. Please run backtest first.")
            return
        
        # Get current params from UI
        seller_params, backtest_params = self.get_current_params()
        
        # Create seed individual
        seed = Individual(
            seller_params=seller_params,
            backtest_params=backtest_params
        )
        
        # Initialize population with seed
        ga_cfg = self._get_ga_settings()
        pop_size = ga_cfg['population_size']
        self.population = Population(size=pop_size, seed_individual=seed)
        
        # Update UI
        self.gen_label.setText(str(self.population.generation))
        self.pop_label.setText(f"{pop_size} individuals (initialized)")
        self.step_btn.setEnabled(True)
        
        print(
            "✓ Population initialized with "
            f"{self.population.size} individuals | "
            f"mutation_rate={ga_cfg['mutation_rate']:.3f} | "
            f"sigma={ga_cfg['sigma']:.3f} | "
            f"elite_fraction={ga_cfg['elite_fraction']:.3f} | "
            f"tournament_size={ga_cfg['tournament_size']} | "
            f"mutation_probability={ga_cfg['mutation_probability']:.3f}"
        )
    
    def run_optimization_step(self):
        """Run one generation of evolution and visualize winning strategy."""
        if self.population is None:
            print("Error: Population not initialized")
            return
        
        if self.current_data is None:
            print("Error: No data loaded")
            return
        
        try:
            ga_cfg = self._get_ga_settings()
            if self.population.size != ga_cfg['population_size']:
                print(
                    "⚠ Population size differs from settings "
                    f"({self.population.size} != {ga_cfg['population_size']}). "
                    "Reinitialize the population to apply the new size."
                )

            # Disable button during execution
            self.step_btn.setEnabled(False)
            self.step_btn.setText("Running...")
            
            # Run evolution step (GPU if available, else CPU)
            print(f"\n{'='*60}")
            print(f"Running Evolution Step {'[GPU]' if self.use_gpu else '[CPU]'}...")
            print(f"{'='*60}")
            
            if self.use_gpu and self.gpu_optimizer:
                # GPU-accelerated evolution
                self.population = self.gpu_optimizer.evolution_step(
                    self.population,
                    self.current_data,
                    self.current_tf,
                    mutation_rate=ga_cfg['mutation_rate'],
                    sigma=ga_cfg['sigma'],
                    elite_fraction=ga_cfg['elite_fraction'],
                    tournament_size=ga_cfg['tournament_size'],
                    mutation_probability=ga_cfg['mutation_probability']
                )
            else:
                # CPU fallback
                self.population = evolution_step(
                    self.population,
                    self.current_data,
                    self.current_tf,
                    mutation_rate=ga_cfg['mutation_rate'],
                    sigma=ga_cfg['sigma'],
                    elite_fraction=ga_cfg['elite_fraction'],
                    tournament_size=ga_cfg['tournament_size'],
                    mutation_probability=ga_cfg['mutation_probability']
                )
            
            # Update UI
            self.gen_label.setText(str(self.population.generation))
            
            # Update fitness evolution plot
            self.update_fitness_plot()
            
            backtest_result = None
            
            if self.population.best_ever:
                best = self.population.best_ever
                self.best_fitness_label.setText(f"{best.fitness:.4f}")
                self.apply_best_btn.setEnabled(True)
                
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
                    # Build features with best individual's params
                    raw_data = self.current_data[['open', 'high', 'low', 'close', 'volume']].copy()
                    feats = build_features(raw_data, best.seller_params, self.current_tf)
                    
                    # Run backtest
                    backtest_result = run_backtest(feats, best.backtest_params)
                    
                    # Update stats display with best individual's results
                    self.trades_df = backtest_result['trades']
                    self.metrics = backtest_result['metrics']
                    
                    # Update metrics display
                    self.update_metrics()
                    
                    # Update equity curve
                    self.update_equity_curve()
                    
                    print(f"✓ Backtest complete: {self.metrics['n']} trades visualized")
                    
                except Exception as e:
                    print(f"Error running backtest for visualization: {e}")
                    import traceback
                    traceback.print_exc()
            
            stats = self.population.get_stats()
            self.pop_label.setText(
                f"Gen {self.population.generation} | "
                f"Pop: {self.population.size} | "
                f"Mean: {stats['mean_fitness']:.2f} | "
                f"Best: {stats['max_fitness']:.2f}"
            )
            
            # Emit signal with backtest results for chart update
            if self.population.best_ever:
                self.optimization_step_complete.emit(self.population.best_ever, backtest_result)
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Re-enable button
            self.step_btn.setEnabled(True)
            self.step_btn.setText("Step")
    
    def apply_best_parameters(self):
        """Apply best parameters from population to UI."""
        if self.population is None or self.population.best_ever is None:
            print("No best individual to apply")
            return
        
        best = self.population.best_ever
        self.set_params_from_individual(best)
        
        print(f"✓ Applied best parameters from generation {best.generation}")
        print(f"  Fitness: {best.fitness:.4f}")
    
    def load_params_from_settings(self, seller_params, backtest_params):
        """Load parameters into UI from settings dialog."""
        sp = seller_params
        bp = backtest_params
        
        # Update SellerParams
        self.param_widgets['ema_fast'].setValue(sp.ema_fast)
        self.param_widgets['ema_slow'].setValue(sp.ema_slow)
        self.param_widgets['z_window'].setValue(sp.z_window)
        self.param_widgets['vol_z'].setValue(sp.vol_z)
        self.param_widgets['tr_z'].setValue(sp.tr_z)
        self.param_widgets['cloc_min'].setValue(sp.cloc_min)
        self.param_widgets['atr_window'].setValue(sp.atr_window)
        
        # Update BacktestParams
        self.param_widgets['atr_stop_mult'].setValue(bp.atr_stop_mult)
        self.param_widgets['reward_r'].setValue(bp.reward_r)
        self.param_widgets['max_hold'].setValue(bp.max_hold)
        self.param_widgets['fee_bp'].setValue(bp.fee_bp)
        self.param_widgets['slippage_bp'].setValue(bp.slippage_bp)
    
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
