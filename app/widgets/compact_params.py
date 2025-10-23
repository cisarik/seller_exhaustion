"""Compact parameter editor for main window integration."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QScrollArea, QPushButton,
    QCheckBox, QComboBox, QProgressBar
)
from PySide6.QtCore import Signal

from strategy.seller_exhaustion import SellerParams
from backtest.engine import BacktestParams
from core.models import Timeframe, FitnessConfig


class CompactParamsEditor(QWidget):
    """Compact parameter editor widget for main window with time-based display."""
    
    params_changed = Signal()  # Emitted when any parameter changes
    coach_load_requested = Signal(str, str)  # Emitted with (model, prompt_version) when load clicked
    coach_unload_requested = Signal()  # Emitted when unload clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.param_widgets = {}
        self.current_timeframe = Timeframe.m15  # Default to 15m
        self.timeframe_minutes = 15
        self.init_ui()
        
        # Load coach settings from config
        self._load_coach_settings()
    
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
        time_params = ['ema_fast', 'ema_slow', 'z_window', 'atr_window', 'fib_lookback', 'fib_lookahead']
        
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
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(250)
        scroll.setMaximumWidth(300)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)
        
        # Strategy Parameters Group
        strategy_group = QGroupBox("Seller-Exhaustion Entry")
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
        exit_group = QGroupBox("Fibonacci Exit")
        exit_layout = QFormLayout()
        exit_layout.setSpacing(4)
        exit_layout.setContentsMargins(8, 8, 8, 8)
        
        # Fibonacci exits toggle (always enabled, checkbox removed)
        self.use_fib_check = QCheckBox("Fibonacci Exits")
        self.use_fib_check.setChecked(True)
        self.use_fib_check.setVisible(False)  # Hidden but still exists for compatibility
        self.use_fib_check.stateChanged.connect(self._on_param_changed)
        
        # Fibonacci exit parameters (ONLY exit mechanism)
        exit_params = [
            ('fib_lookback', 'Fib Lookback:', 720, 2880, 60, 1440, True),    # 12h-48h, default 24h
            ('fib_lookahead', 'Fib Lookahead:', 60, 240, 15, 75, True),      # 1h-4h, default 1.25h
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
        
        # Fibonacci Target Level (Dropdown for discrete choices)
        fib_target_label = QLabel("Fib Target:")
        fib_target_label.setMaximumWidth(80)
        self.param_widgets['fib_target'] = QComboBox()
        self.param_widgets['fib_target'].setMaximumWidth(100)
        
        # Add valid Fibonacci levels
        fib_levels = [
            (0.382, "38.2%"),
            (0.500, "50.0%"),
            (0.618, "61.8% (Golden)"),
            (0.786, "78.6%"),
            (1.000, "100%"),
        ]
        for value, label in fib_levels:
            self.param_widgets['fib_target'].addItem(label, value)
        
        # Set default to Golden Ratio (61.8%)
        self.param_widgets['fib_target'].setCurrentIndex(2)
        self.param_widgets['fib_target'].currentIndexChanged.connect(self._on_param_changed)
        exit_layout.addRow(fib_target_label, self.param_widgets['fib_target'])
        
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
        
        # Evolution Coach Group (NEW - Enable checkbox + provider dropdown)
        coach_group = QGroupBox("Evolution Coach 🤖")
        coach_layout = QVBoxLayout()
        coach_layout.setSpacing(6)
        coach_layout.setContentsMargins(8, 8, 8, 8)

        # Enable checkbox row
        enable_row = QHBoxLayout()
        enable_label = QLabel("Enable:")
        enable_label.setMaximumWidth(60)
        self.coach_enabled_check = QCheckBox()
        self.coach_enabled_check.setChecked(True)
        self.coach_enabled_check.setToolTip("Enable/disable Evolution Coach agent")
        self.coach_enabled_check.stateChanged.connect(self._on_coach_config_changed)
        enable_row.addWidget(enable_label)
        enable_row.addWidget(self.coach_enabled_check, 1)
        coach_layout.addLayout(enable_row)

        # Coach mode dropdown row
        mode_row = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setMaximumWidth(60)
        self.coach_mode_combo = QComboBox()
        self.coach_mode_combo.addItem("🧠 OpenAI Agents", "openai")
        self.coach_mode_combo.addItem("⚙️ Classic", "classic")
        self.coach_mode_combo.setToolTip("Choose coach mode: OpenAI Agents (LLM-based) or Classic (deterministic)")
        self.coach_mode_combo.currentIndexChanged.connect(self._on_coach_config_changed)
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.coach_mode_combo, 1)
        coach_layout.addLayout(mode_row)
        
        # Coach progress bar (shows when coach is active)
        self.coach_progress = QProgressBar()
        self.coach_progress.setVisible(False)  # Hidden by default
        self.coach_progress.setRange(0, 0)  # Indeterminate progress
        self.coach_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 2px;
            }
        """)
        self.coach_progress.setToolTip("Evolution Coach is analyzing population...")
        coach_layout.addWidget(self.coach_progress)
        
        self._coach_model_loaded = False
        
        coach_group.setLayout(coach_layout)
        scroll_layout.addWidget(coach_group)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    

    def _load_coach_settings(self):
        """Load coach settings from config and update UI."""
        from config.settings import settings

        # Set enabled state
        self.coach_enabled_check.setChecked(getattr(settings, 'coach_enabled', True))

        # Set coach mode - default to 'classic' if not set
        coach_mode = getattr(settings, 'coach_mode', 'classic')
        index = self.coach_mode_combo.findData(coach_mode)
        if index >= 0:
            self.coach_mode_combo.setCurrentIndex(index)
        else:
            # Default to Classic if mode not found
            self.coach_mode_combo.setCurrentIndex(1)  # Index 1 is "Classic"

        # Update button styles based on provider
    def _on_coach_config_changed(self):
        """Handle coach enable/disable and mode change - save to settings."""
        from config.settings import settings, SettingsManager

        # Update enabled setting
        settings.coach_enabled = self.coach_enabled_check.isChecked()

        # Update coach mode setting
        settings.coach_mode = self.coach_mode_combo.currentData()

        # Convert settings to dict and save to .env
        try:
            settings_dict = settings.model_dump()  # Pydantic v2
        except AttributeError:
            settings_dict = settings.dict()  # Pydantic v1 fallback

        SettingsManager.save_to_env(settings_dict)

    def set_coach_analyzing(self, is_analyzing: bool):
        """Set coach analysis state and show/hide progress bar."""
        if is_analyzing:
            self.coach_progress.setVisible(True)
            self.coach_progress.setRange(0, 0)  # Indeterminate progress
        else:
            self.coach_progress.setVisible(False)

    def hide_coach_progress(self):
        """Hide coach progress bar."""
        self.coach_progress.setVisible(False)
        
    
    
    def get_coach_config(self) -> dict:
        """Get Evolution Coach configuration from settings."""
        from config.settings import settings
        return {
            "enabled": self.coach_enabled_check.isChecked(),
            "mode": self.coach_mode_combo.currentData(),
            "provider": getattr(settings, 'coach_provider', 'local'),
            "model": getattr(settings, 'coach_model', 'google/gemma-3-12b'),
            "prompt_version": getattr(settings, 'coach_system_prompt', 'agent01')
        }
    
    def show_coach_progress(self, message: str = "Evolution Coach analyzing..."):
        """Show coach progress bar with message."""
        self.coach_progress.setVisible(True)
        self.coach_progress.setFormat(message)
        self.coach_progress.setToolTip(message)
    
    def hide_coach_progress(self):
        """Hide coach progress bar."""
        self.coach_progress.setVisible(False)
    
    
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
        
        # Fibonacci exit defaults (ONLY exit mechanism)
        self.use_fib_check.setChecked(True)
        self.param_widgets['fib_lookback'].setValue(1440)  # 24 hours
        self.param_widgets['fib_lookahead'].setValue(75)   # 1.25 hours
        self.param_widgets['fib_target'].setCurrentIndex(2)  # Golden ratio (61.8%)
        
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
            fib_swing_lookback=int(self.param_widgets['fib_lookback'].value() / self.timeframe_minutes),
            fib_swing_lookahead=int(self.param_widgets['fib_lookahead'].value() / self.timeframe_minutes),
            fib_target_level=self.param_widgets['fib_target'].currentData(),
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
        
        # Update Fibonacci params (convert bars to minutes)
        self.param_widgets['fib_lookback'].setValue(int(backtest_params.fib_swing_lookback * self.timeframe_minutes))
        self.param_widgets['fib_lookahead'].setValue(int(backtest_params.fib_swing_lookahead * self.timeframe_minutes))
        
        # Set fib_target combobox by finding matching value
        fib_target_combo = self.param_widgets['fib_target']
        for i in range(fib_target_combo.count()):
            if abs(fib_target_combo.itemData(i) - backtest_params.fib_target_level) < 0.001:
                fib_target_combo.setCurrentIndex(i)
                break
        
        # Update transaction costs
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
