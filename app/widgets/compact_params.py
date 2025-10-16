"""Compact parameter editor for main window integration."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QScrollArea, QPushButton,
    QCheckBox, QComboBox
)
from PySide6.QtCore import Signal

from strategy.seller_exhaustion import SellerParams
from backtest.engine import BacktestParams
from core.models import Timeframe, FitnessConfig


class CompactParamsEditor(QWidget):
    """Compact parameter editor widget for main window with time-based display."""
    
    params_changed = Signal()  # Emitted when any parameter changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.param_widgets = {}
        self.current_timeframe = Timeframe.m15  # Default to 15m
        self.timeframe_minutes = 15
        self.init_ui()
    
    def set_timeframe(self, timeframe: Timeframe):
        """Update the timeframe for time-based conversions."""
        self.current_timeframe = timeframe
        
        # Map timeframe to minutes
        tf_map = {
            Timeframe.m1: 1,
            Timeframe.m3: 3,
            Timeframe.m5: 5,
            Timeframe.m10: 10,
            Timeframe.m15: 15,
            Timeframe.m30: 30,
            Timeframe.m60: 60,
        }
        self.timeframe_minutes = tf_map.get(timeframe, 15)
        
        # Update tooltips to reflect current timeframe
        self._update_tooltips()
    
    def _minutes_to_display(self, minutes: int) -> str:
        """Convert minutes to human-readable format."""
        if minutes < 60:
            return f"{minutes}m"
        elif minutes < 1440:
            hours = minutes / 60
            return f"{hours:.1f}h" if hours % 1 else f"{int(hours)}h"
        else:
            days = minutes / 1440
            return f"{days:.1f}d" if days % 1 else f"{int(days)}d"
    
    def _update_tooltips(self):
        """Update tooltips to show time + bars for current timeframe."""
        time_params = ['ema_fast', 'ema_slow', 'z_window', 'atr_window', 'max_hold', 'fib_lookback', 'fib_lookahead']
        
        for param_name in time_params:
            if param_name in self.param_widgets:
                widget = self.param_widgets[param_name]
                minutes = widget.value()
                bars = int(minutes / self.timeframe_minutes)
                time_str = self._minutes_to_display(minutes)
                widget.setToolTip(
                    f"{time_str} = {bars} bars on {self.current_timeframe.value}\n"
                    f"({minutes} minutes total)"
                )
    
    def init_ui(self):
        """Initialize compact UI with strategy and backtest parameters."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title = QLabel("Best Parameters")
        title.setProperty("role", "title")
        layout.addWidget(title)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(250)
        scroll.setMaximumWidth(300)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)
        
        # Strategy Parameters Group
        strategy_group = QGroupBox("Strategy Parameters")
        strategy_layout = QFormLayout()
        strategy_layout.setSpacing(4)
        strategy_layout.setContentsMargins(8, 8, 8, 8)
        
        # Time-based parameters (displayed in minutes, converted from bars)
        # Default values for 15m timeframe: 96 bars = 1440 min (24h), 672 bars = 10080 min (7d)
        strategy_params = [
            ('ema_fast', 'EMA Fast:', 720, 2880, 60, 1440, True),      # 12h-48h, default 24h
            ('ema_slow', 'EMA Slow:', 5040, 20160, 360, 10080, True),  # 3.5d-14d, default 7d
            ('z_window', 'Z-Window:', 5040, 20160, 360, 10080, True),  # 3.5d-14d, default 7d
            ('vol_z', 'Vol Z-Score:', 1.0, 3.5, 0.1, 2.0, False),
            ('tr_z', 'TR Z-Score:', 0.8, 2.0, 0.1, 1.2, False),
            ('cloc_min', 'Close Loc:', 0.4, 0.8, 0.01, 0.6, False),
            ('atr_window', 'ATR Window:', 720, 2880, 60, 1440, True),  # 12h-48h, default 24h
        ]
        
        for param_name, label_text, min_val, max_val, step, default, is_time in strategy_params:
            label = QLabel(label_text)
            label.setMaximumWidth(80)
            
            if isinstance(step, int):
                widget = QSpinBox()
                widget.setRange(int(min_val), int(max_val))
                widget.setSingleStep(step)
                widget.setValue(default)
                if is_time:
                    # Add suffix to show it's time-based
                    widget.setSuffix(" min")
            else:
                widget = QDoubleSpinBox()
                widget.setRange(min_val, max_val)
                widget.setSingleStep(step)
                widget.setDecimals(2)
                widget.setValue(default)
            
            widget.setMaximumWidth(100)
            widget.valueChanged.connect(self._on_param_changed)
            
            strategy_layout.addRow(label, widget)
            self.param_widgets[param_name] = widget
        
        strategy_group.setLayout(strategy_layout)
        scroll_layout.addWidget(strategy_group)
        
        # Exit Strategy Group (NEW - separated from backtest)
        exit_group = QGroupBox("Exit Strategy")
        exit_layout = QFormLayout()
        exit_layout.setSpacing(4)
        exit_layout.setContentsMargins(8, 8, 8, 8)
        
        # Fibonacci exits toggle
        self.use_fib_check = QCheckBox("Fibonacci Exits")
        self.use_fib_check.setChecked(True)
        self.use_fib_check.setToolTip("Exit at first Fibonacci retracement level hit")
        self.use_fib_check.stateChanged.connect(self._on_param_changed)
        exit_layout.addRow("", self.use_fib_check)
        
        # Exit strategy parameters
        exit_params = [
            ('fib_lookback', 'Fib Lookback:', 720, 2880, 60, 1440, True),    # 12h-48h, default 24h
            ('fib_lookahead', 'Fib Lookahead:', 60, 240, 15, 75, True),       # 1h-4h, default 1.25h
            ('fib_target', 'Fib Target:', 0.382, 1.0, 0.001, 0.618, False),   # 38.2%-100%, default 61.8%
            ('max_hold', 'Max Hold:', 720, 2880, 60, 1440, True),             # 12h-48h, default 24h
            ('atr_stop_mult', 'Stop Mult:', 0.3, 1.5, 0.05, 0.7, False),
            ('reward_r', 'R:R Ratio:', 1.5, 4.0, 0.1, 2.0, False),
        ]
        
        for param_name, label_text, min_val, max_val, step, default, is_time in exit_params:
            label = QLabel(label_text)
            label.setMaximumWidth(80)
            
            if isinstance(step, int):
                widget = QSpinBox()
                widget.setRange(int(min_val), int(max_val))
                widget.setSingleStep(step)
                widget.setValue(default)
                if is_time:
                    widget.setSuffix(" min")
            else:
                widget = QDoubleSpinBox()
                widget.setRange(min_val, max_val)
                widget.setSingleStep(step)
                widget.setDecimals(2)
                widget.setValue(default)
            
            widget.setMaximumWidth(100)
            widget.valueChanged.connect(self._on_param_changed)
            
            exit_layout.addRow(label, widget)
            self.param_widgets[param_name] = widget
        
        exit_group.setLayout(exit_layout)
        scroll_layout.addWidget(exit_group)
        
        # Transaction Costs Group (NEW - separated for clarity)
        costs_group = QGroupBox("Transaction Costs")
        costs_layout = QFormLayout()
        costs_layout.setSpacing(4)
        costs_layout.setContentsMargins(8, 8, 8, 8)
        
        # Fee
        fee_label = QLabel("Fee (bp):")
        fee_label.setMaximumWidth(80)
        self.param_widgets['fee_bp'] = QDoubleSpinBox()
        self.param_widgets['fee_bp'].setRange(2.0, 10.0)
        self.param_widgets['fee_bp'].setSingleStep(0.5)
        self.param_widgets['fee_bp'].setDecimals(1)
        self.param_widgets['fee_bp'].setValue(5.0)
        self.param_widgets['fee_bp'].setMaximumWidth(100)
        self.param_widgets['fee_bp'].valueChanged.connect(self._on_param_changed)
        costs_layout.addRow(fee_label, self.param_widgets['fee_bp'])
        
        # Slippage
        slip_label = QLabel("Slip (bp):")
        slip_label.setMaximumWidth(80)
        self.param_widgets['slippage_bp'] = QDoubleSpinBox()
        self.param_widgets['slippage_bp'].setRange(2.0, 10.0)
        self.param_widgets['slippage_bp'].setSingleStep(0.5)
        self.param_widgets['slippage_bp'].setDecimals(1)
        self.param_widgets['slippage_bp'].setValue(5.0)
        self.param_widgets['slippage_bp'].setMaximumWidth(100)
        self.param_widgets['slippage_bp'].valueChanged.connect(self._on_param_changed)
        costs_layout.addRow(slip_label, self.param_widgets['slippage_bp'])
        
        costs_group.setLayout(costs_layout)
        scroll_layout.addWidget(costs_group)
        
        # Fitness Function Group (NEW - optimization goals)
        fitness_group = QGroupBox("Fitness Function")
        fitness_layout = QFormLayout()
        fitness_layout.setSpacing(4)
        fitness_layout.setContentsMargins(8, 8, 8, 8)
        
        # Note: Preset selector is in Evolutionary Optimization section
        # This section only shows the weight values that can be adjusted
        # Preset selection controls these values from stats_panel
        
        # Weights section
        weights_label = QLabel("<b>Weights:</b>")
        fitness_layout.addRow(weights_label)
        
        # Define fitness weight params
        fitness_weights = [
            ('trade_count_weight', 'Trade Count:', 0.15, "More trades (HFT)"),
            ('win_rate_weight', 'Win Rate:', 0.25, "Higher win %"),
            ('avg_r_weight', 'Avg R:', 0.30, "Better R-multiples"),
            ('total_pnl_weight', 'Total PnL:', 0.20, "More profit"),
            ('max_dd_penalty', 'DD Penalty:', 0.10, "Penalize drawdowns"),
        ]
        
        for param_name, label_text, default, tooltip in fitness_weights:
            label = QLabel(label_text)
            label.setMaximumWidth(80)
            widget = QDoubleSpinBox()
            widget.setRange(0.0, 1.0)
            widget.setSingleStep(0.05)
            widget.setDecimals(2)
            widget.setValue(default)
            widget.setMaximumWidth(100)
            widget.setToolTip(tooltip)
            widget.valueChanged.connect(self._on_fitness_changed)
            fitness_layout.addRow(label, widget)
            setattr(self, f'{param_name}_spin', widget)
        
        # Min requirements section
        reqs_label = QLabel("<b>Min Requirements:</b>")
        fitness_layout.addRow(reqs_label)
        
        # Min trades
        min_trades_label = QLabel("Min Trades:")
        min_trades_label.setMaximumWidth(80)
        self.min_trades_spin = QSpinBox()
        self.min_trades_spin.setRange(0, 100)
        self.min_trades_spin.setValue(10)
        self.min_trades_spin.setMaximumWidth(100)
        self.min_trades_spin.setToolTip("Minimum trades required")
        self.min_trades_spin.valueChanged.connect(self._on_fitness_changed)
        fitness_layout.addRow(min_trades_label, self.min_trades_spin)
        
        # Min win rate
        min_wr_label = QLabel("Min Win Rate:")
        min_wr_label.setMaximumWidth(80)
        self.min_win_rate_spin = QDoubleSpinBox()
        self.min_win_rate_spin.setRange(0.0, 1.0)
        self.min_win_rate_spin.setSingleStep(0.05)
        self.min_win_rate_spin.setDecimals(2)
        self.min_win_rate_spin.setValue(0.40)
        self.min_win_rate_spin.setMaximumWidth(100)
        self.min_win_rate_spin.setToolTip("Minimum win rate required (0.40 = 40%)")
        self.min_win_rate_spin.valueChanged.connect(self._on_fitness_changed)
        fitness_layout.addRow(min_wr_label, self.min_win_rate_spin)
        
        fitness_group.setLayout(fitness_layout)
        scroll_layout.addWidget(fitness_group)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Action buttons
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(4)
        
        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        btn_layout.addWidget(self.reset_btn)
        
        self.strategy_editor_btn = QPushButton("📊 Strategy Editor")
        self.strategy_editor_btn.setToolTip("Open full Strategy Editor for parameter sets")
        btn_layout.addWidget(self.strategy_editor_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_param_changed(self):
        """Handle parameter change and update tooltips."""
        self._update_tooltips()
        self.params_changed.emit()
    
    def load_fitness_preset(self, preset_name: str):
        """
        Load fitness preset configuration and update UI.
        Called externally from stats_panel when preset dropdown changes.
        
        Args:
            preset_name: Name of preset (balanced, high_frequency, conservative, profit_focused, custom)
        """
        if preset_name == "custom":
            # Don't override custom values
            return
        
        # Load preset configuration
        config = FitnessConfig.get_preset_config(preset_name)
        
        # Update UI (block signals to avoid loops)
        self.trade_count_weight_spin.blockSignals(True)
        self.win_rate_weight_spin.blockSignals(True)
        self.avg_r_weight_spin.blockSignals(True)
        self.total_pnl_weight_spin.blockSignals(True)
        self.max_dd_penalty_spin.blockSignals(True)
        self.min_trades_spin.blockSignals(True)
        self.min_win_rate_spin.blockSignals(True)
        
        self.trade_count_weight_spin.setValue(config.trade_count_weight)
        self.win_rate_weight_spin.setValue(config.win_rate_weight)
        self.avg_r_weight_spin.setValue(config.avg_r_weight)
        self.total_pnl_weight_spin.setValue(config.total_pnl_weight)
        self.max_dd_penalty_spin.setValue(config.max_drawdown_penalty)
        self.min_trades_spin.setValue(config.min_trades)
        self.min_win_rate_spin.setValue(config.min_win_rate)
        
        self.trade_count_weight_spin.blockSignals(False)
        self.win_rate_weight_spin.blockSignals(False)
        self.avg_r_weight_spin.blockSignals(False)
        self.total_pnl_weight_spin.blockSignals(False)
        self.max_dd_penalty_spin.blockSignals(False)
        self.min_trades_spin.blockSignals(False)
        self.min_win_rate_spin.blockSignals(False)
        
        self._on_param_changed()
    
    def _on_fitness_changed(self):
        """Handle fitness weight changes."""
        # Note: Preset combo is in stats_panel, not here
        # Manual weight changes don't need to switch preset
        self._on_param_changed()
    
    def reset_to_defaults(self):
        """Reset all parameters to default values (time-based)."""
        # Strategy defaults (in minutes)
        self.param_widgets['ema_fast'].setValue(1440)      # 24 hours
        self.param_widgets['ema_slow'].setValue(10080)     # 7 days
        self.param_widgets['z_window'].setValue(10080)     # 7 days
        self.param_widgets['vol_z'].setValue(2.0)
        self.param_widgets['tr_z'].setValue(1.2)
        self.param_widgets['cloc_min'].setValue(0.6)
        self.param_widgets['atr_window'].setValue(1440)    # 24 hours
        
        # Exit strategy defaults
        self.use_fib_check.setChecked(True)
        self.param_widgets['fib_lookback'].setValue(1440)  # 24 hours
        self.param_widgets['fib_lookahead'].setValue(75)   # 1.25 hours
        self.param_widgets['fib_target'].setValue(0.618)   # Golden ratio
        self.param_widgets['max_hold'].setValue(1440)      # 24 hours
        self.param_widgets['atr_stop_mult'].setValue(0.7)
        self.param_widgets['reward_r'].setValue(2.0)
        
        # Transaction costs defaults
        self.param_widgets['fee_bp'].setValue(5.0)
        self.param_widgets['slippage_bp'].setValue(5.0)
        
        # Fitness function defaults (Balanced preset)
        self.load_fitness_preset("balanced")
    
    def get_params(self):
        """Get current parameters, converting time-based values to bars.
        
        Returns:
            Tuple of (SellerParams, BacktestParams, FitnessConfig)
        """
        # Convert minutes to bars for time-based parameters
        seller_params = SellerParams(
            ema_fast=int(self.param_widgets['ema_fast'].value() / self.timeframe_minutes),
            ema_slow=int(self.param_widgets['ema_slow'].value() / self.timeframe_minutes),
            z_window=int(self.param_widgets['z_window'].value() / self.timeframe_minutes),
            vol_z=self.param_widgets['vol_z'].value(),
            tr_z=self.param_widgets['tr_z'].value(),
            cloc_min=self.param_widgets['cloc_min'].value(),
            atr_window=int(self.param_widgets['atr_window'].value() / self.timeframe_minutes),
        )
        
        backtest_params = BacktestParams(
            use_fib_exits=self.use_fib_check.isChecked(),
            fib_swing_lookback=int(self.param_widgets['fib_lookback'].value() / self.timeframe_minutes),
            fib_swing_lookahead=int(self.param_widgets['fib_lookahead'].value() / self.timeframe_minutes),
            fib_target_level=self.param_widgets['fib_target'].value(),
            max_hold=int(self.param_widgets['max_hold'].value() / self.timeframe_minutes),
            atr_stop_mult=self.param_widgets['atr_stop_mult'].value(),
            reward_r=self.param_widgets['reward_r'].value(),
            fee_bp=self.param_widgets['fee_bp'].value(),
            slippage_bp=self.param_widgets['slippage_bp'].value(),
        )
        
        # Get fitness configuration
        # Note: Preset name is controlled by stats_panel dropdown
        fitness_config = FitnessConfig(
            preset="custom",  # Actual preset is managed by stats_panel
            trade_count_weight=self.trade_count_weight_spin.value(),
            win_rate_weight=self.win_rate_weight_spin.value(),
            avg_r_weight=self.avg_r_weight_spin.value(),
            total_pnl_weight=self.total_pnl_weight_spin.value(),
            max_drawdown_penalty=self.max_dd_penalty_spin.value(),
            min_trades=self.min_trades_spin.value(),
            min_win_rate=self.min_win_rate_spin.value()
        )
        
        return seller_params, backtest_params, fitness_config
    
    def set_params(self, seller_params: SellerParams, backtest_params: BacktestParams, fitness_config: FitnessConfig = None):
        """Set parameters from models, converting bars to minutes.
        
        Args:
            seller_params: Strategy parameters
            backtest_params: Backtest parameters
            fitness_config: Fitness configuration (optional)
        """
        # Convert bars to minutes for time-based parameters
        self.param_widgets['ema_fast'].setValue(int(seller_params.ema_fast * self.timeframe_minutes))
        self.param_widgets['ema_slow'].setValue(int(seller_params.ema_slow * self.timeframe_minutes))
        self.param_widgets['z_window'].setValue(int(seller_params.z_window * self.timeframe_minutes))
        self.param_widgets['vol_z'].setValue(float(seller_params.vol_z))
        self.param_widgets['tr_z'].setValue(float(seller_params.tr_z))
        self.param_widgets['cloc_min'].setValue(float(seller_params.cloc_min))
        self.param_widgets['atr_window'].setValue(int(seller_params.atr_window * self.timeframe_minutes))
        
        # Update backtest params (convert bars to minutes for time-based)
        self.use_fib_check.setChecked(backtest_params.use_fib_exits)
        self.param_widgets['fib_lookback'].setValue(int(backtest_params.fib_swing_lookback * self.timeframe_minutes))
        self.param_widgets['fib_lookahead'].setValue(int(backtest_params.fib_swing_lookahead * self.timeframe_minutes))
        self.param_widgets['fib_target'].setValue(float(backtest_params.fib_target_level))
        self.param_widgets['max_hold'].setValue(int(backtest_params.max_hold * self.timeframe_minutes))
        self.param_widgets['atr_stop_mult'].setValue(float(backtest_params.atr_stop_mult))
        self.param_widgets['reward_r'].setValue(float(backtest_params.reward_r))
        self.param_widgets['fee_bp'].setValue(float(backtest_params.fee_bp))
        self.param_widgets['slippage_bp'].setValue(float(backtest_params.slippage_bp))
        
        # Update fitness configuration if provided
        if fitness_config:
            # Note: Preset combo is in stats_panel, we only set weight values
            
            # Set weights
            self.trade_count_weight_spin.setValue(fitness_config.trade_count_weight)
            self.win_rate_weight_spin.setValue(fitness_config.win_rate_weight)
            self.avg_r_weight_spin.setValue(fitness_config.avg_r_weight)
            self.total_pnl_weight_spin.setValue(fitness_config.total_pnl_weight)
            self.max_dd_penalty_spin.setValue(fitness_config.max_drawdown_penalty)
            self.min_trades_spin.setValue(fitness_config.min_trades)
            self.min_win_rate_spin.setValue(fitness_config.min_win_rate)
        
        # Update tooltips after setting values
        self._update_tooltips()
