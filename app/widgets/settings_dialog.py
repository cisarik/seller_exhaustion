from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QPushButton, QDateEdit, QProgressBar,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QMessageBox, QComboBox
)
from PySide6.QtCore import Qt, QDate, Signal
import asyncio
from datetime import datetime

from config.settings import settings, SettingsManager
from data.provider import DataProvider
from core.models import Timeframe
from strategy.timeframe_defaults import get_defaults_for_timeframe
from strategy.seller_exhaustion import SellerParams
from backtest.engine import BacktestParams


# Timeframe mapping (multiplier, unit, label)
TIMEFRAMES = {
    "1m": (1, "minute", "1 minute"),
    "3m": (3, "minute", "3 minutes"),
    "5m": (5, "minute", "5 minutes"),
    "10m": (10, "minute", "10 minutes"),
    "15m": (15, "minute", "15 minutes"),
    "30m": (30, "minute", "30 minutes"),
    "60m": (60, "minute", "1 hour"),
    "4h": (240, "minute", "4 hours"),
    "12h": (720, "minute", "12 hours"),
    "1d": (1, "day", "1 day"),
}


class SettingsDialog(QDialog):
    """Enhanced settings dialog with persistence and timeframe selection."""
    
    data_downloaded = Signal(object)  # Emits DataFrame when data is downloaded
    settings_saved = Signal()  # Emits when settings are saved
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings - ADA Trading Agent")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
        
        self.dp = None
        self.downloaded_data = None
        
        self.init_ui()
        self.load_from_settings()  # Load saved settings on init
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Tab widget for different settings sections
        tabs = QTabWidget()
        
        # Data Download Tab
        tabs.addTab(self.create_data_tab(), "Data Download")
        
        # Strategy Parameters Tab
        tabs.addTab(self.create_strategy_tab(), "Strategy Parameters")
        
        # Indicator Selection Tab
        tabs.addTab(self.create_indicators_tab(), "Chart Indicators")
        
        # Backtest Parameters Tab
        tabs.addTab(self.create_backtest_tab(), "Backtest Parameters")

        # Optimization Parameters Tab
        tabs.addTab(self.create_optimizer_tab(), "Optimization")
        
        # Acceleration Settings Tab (NEW)
        tabs.addTab(self.create_acceleration_tab(), "⚡ Acceleration")
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("💾 Save Settings")
        self.save_btn.setObjectName("primaryButton")
        self.save_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def create_data_tab(self):
        """Create data download tab with timeframe selector."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Timeframe Selection Group
        tf_group = QGroupBox("Timeframe Selection")
        tf_layout = QFormLayout()
        
        self.timeframe_combo = QComboBox()
        for key, (mult, unit, label) in TIMEFRAMES.items():
            self.timeframe_combo.addItem(f"{label} ({key})", key)
        self.timeframe_combo.setCurrentText("15 minutes (15m)")
        
        # Connect timeframe change to auto-adjust parameters
        self.timeframe_combo.currentIndexChanged.connect(self.on_timeframe_changed)
        
        tf_layout.addRow("Timeframe:", self.timeframe_combo)
        
        # Auto-adjust checkbox
        self.auto_adjust_params = QCheckBox("Auto-adjust strategy parameters for timeframe")
        self.auto_adjust_params.setChecked(True)
        self.auto_adjust_params.setToolTip(
            "When enabled, automatically adjusts EMA, Z-Score, and ATR windows\n"
            "to maintain consistent time periods across different timeframes.\n"
            "Example: 24h lookback = 96 bars on 15m, but 1440 bars on 1m."
        )
        tf_layout.addRow("", self.auto_adjust_params)
        
        info_label = QLabel(
            "💡 Tip: Shorter timeframes (1m-5m) generate more signals but may be noisier.\n"
            "Recommended: Start with 15m or 1h for balanced signal quality.\n\n"
            "⚠️ IMPORTANT: Strategy parameters MUST scale with timeframe!\n"
            "Use auto-adjust or manually configure time-based parameters."
        )
        info_label.setWordWrap(True)
        info_label.setProperty("variant", "secondary")
        tf_layout.addRow(info_label)
        
        tf_group.setLayout(tf_layout)
        layout.addWidget(tf_group)
        
        # Data Download Group
        download_group = QGroupBox("Download Historical Data")
        download_layout = QFormLayout()
        
        # Ticker (fixed to ADA)
        ticker_label = QLabel("X:ADAUSD (Cardano)")
        ticker_label.setProperty("variant", "secondary")
        download_layout.addRow("Ticker:", ticker_label)
        
        # Date range
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate(2024, 1, 1))
        self.date_from.setDisplayFormat("yyyy-MM-dd")
        download_layout.addRow("From Date:", self.date_from)
        
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setDisplayFormat("yyyy-MM-dd")
        download_layout.addRow("To Date:", self.date_to)
        
        # API Key status
        api_key_status = QLabel()
        if settings.polygon_api_key:
            api_key_status.setText("✓ API Key Configured")
            api_key_status.setStyleSheet("color: #4caf50;")
        else:
            api_key_status.setText("✗ API Key Missing (check .env)")
            api_key_status.setStyleSheet("color: #f44336;")
        download_layout.addRow("Status:", api_key_status)
        
        download_group.setLayout(download_layout)
        layout.addWidget(download_group)
        
        # Progress section
        progress_group = QGroupBox("Download Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Ready to download")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Download button
        self.download_btn = QPushButton("📥 Download Data")
        self.download_btn.setObjectName("primaryButton")
        self.download_btn.clicked.connect(lambda: asyncio.create_task(self.download_data()))
        layout.addWidget(self.download_btn)
        
        # Info label
        info = QLabel(
            "💡 Tip: Download at least 7 days of data for proper indicator calculation.\n"
            "Free tier: 5 API calls/minute. Large date ranges may take a few minutes.\n"
            "Settings are auto-saved when you download data or click 'Save Settings'."
        )
        info.setWordWrap(True)
        info.setProperty("variant", "secondary")
        layout.addWidget(info)
        
        layout.addStretch()
        return widget
    
    def create_strategy_tab(self):
        """Create strategy parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Seller Exhaustion Parameters")
        form_layout = QFormLayout()
        
        # EMA periods
        self.ema_fast = QSpinBox()
        self.ema_fast.setRange(10, 2000)
        self.ema_fast.setValue(96)
        self.ema_fast.setSuffix(" bars")
        form_layout.addRow("EMA Fast:", self.ema_fast)
        
        self.ema_slow = QSpinBox()
        self.ema_slow.setRange(100, 5000)
        self.ema_slow.setValue(672)
        self.ema_slow.setSuffix(" bars")
        form_layout.addRow("EMA Slow:", self.ema_slow)
        
        # Z-score window
        self.z_window = QSpinBox()
        self.z_window.setRange(100, 5000)
        self.z_window.setValue(672)
        self.z_window.setSuffix(" bars")
        form_layout.addRow("Z-Score Window:", self.z_window)
        
        # Volume z-score threshold
        self.vol_z = QDoubleSpinBox()
        self.vol_z.setRange(0.5, 10.0)
        self.vol_z.setValue(2.0)
        self.vol_z.setSingleStep(0.1)
        form_layout.addRow("Volume Z-Score:", self.vol_z)
        
        # True range z-score threshold
        self.tr_z = QDoubleSpinBox()
        self.tr_z.setRange(0.5, 10.0)
        self.tr_z.setValue(1.2)
        self.tr_z.setSingleStep(0.1)
        form_layout.addRow("True Range Z-Score:", self.tr_z)
        
        # Close location minimum
        self.cloc_min = QDoubleSpinBox()
        self.cloc_min.setRange(0.0, 1.0)
        self.cloc_min.setValue(0.6)
        self.cloc_min.setSingleStep(0.05)
        form_layout.addRow("Min Close Location:", self.cloc_min)
        
        # ATR window
        self.atr_window = QSpinBox()
        self.atr_window.setRange(10, 500)
        self.atr_window.setValue(96)
        self.atr_window.setSuffix(" bars")
        form_layout.addRow("ATR Window:", self.atr_window)
        
        group.setLayout(form_layout)
        layout.addWidget(group)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_strategy_params)
        layout.addWidget(reset_btn)
        
        # Info
        info = QLabel(
            "💡 These parameters control when seller exhaustion signals are generated.\n"
            "Adjust based on your selected timeframe for optimal results."
        )
        info.setWordWrap(True)
        info.setProperty("variant", "secondary")
        layout.addWidget(info)
        
        layout.addStretch()
        return widget
    
    def create_indicators_tab(self):
        """Create indicator selection tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Chart Indicators")
        indicators_layout = QVBoxLayout()
        
        indicators_layout.addWidget(QLabel("Select indicators to display on chart:"))
        
        # EMA Fast
        self.show_ema_fast = QCheckBox("EMA Fast - Cyan")
        self.show_ema_fast.setChecked(True)
        indicators_layout.addWidget(self.show_ema_fast)
        
        # EMA Slow
        self.show_ema_slow = QCheckBox("EMA Slow - Orange")
        self.show_ema_slow.setChecked(True)
        indicators_layout.addWidget(self.show_ema_slow)
        
        # SMA
        self.show_sma = QCheckBox("SMA (100) - Purple")
        self.show_sma.setChecked(False)
        indicators_layout.addWidget(self.show_sma)
        
        # RSI (separate panel)
        self.show_rsi = QCheckBox("RSI (14) - Separate Panel (Future)")
        self.show_rsi.setChecked(False)
        self.show_rsi.setEnabled(False)
        indicators_layout.addWidget(self.show_rsi)
        
        # MACD (separate panel)
        self.show_macd = QCheckBox("MACD (12,26,9) - Separate Panel (Future)")
        self.show_macd.setChecked(False)
        self.show_macd.setEnabled(False)
        indicators_layout.addWidget(self.show_macd)
        
        # Volume
        self.show_volume = QCheckBox("Volume Bars (Future)")
        self.show_volume.setChecked(False)
        self.show_volume.setEnabled(False)
        indicators_layout.addWidget(self.show_volume)
        
        # Signal markers
        indicators_layout.addWidget(QLabel("\nTrade Markers:"))
        
        self.show_signals = QCheckBox("Entry Signals (Yellow Triangles)")
        self.show_signals.setChecked(True)
        self.show_signals.setToolTip("Show yellow triangles marking seller exhaustion signals")
        indicators_layout.addWidget(self.show_signals)
        
        self.show_entries = QCheckBox("Trade Balls (Green/Red circles, sized by PnL)")
        self.show_entries.setChecked(True)
        self.show_entries.setToolTip(
            "Show trade markers as colored balls:\n"
            "• Green = Profit (larger = more profit)\n"
            "• Red = Loss (larger = more loss)\n"
            "• White = Selected trade"
        )
        indicators_layout.addWidget(self.show_entries)
        
        # Remove the old "Sell Orders" checkbox - it's no longer used
        # (exits are now shown as part of the trade balls)
        
        # Fibonacci Retracements Section
        indicators_layout.addWidget(QLabel("\nFibonacci Retracements (click trade to view):"))
        
        self.show_fib_retracements = QCheckBox("📊 Show Fibonacci Retracements")
        self.show_fib_retracements.setChecked(True)
        self.show_fib_retracements.setToolTip(
            "Show Fibonacci retracement levels for selected trade:\n"
            "- Swing high marker (⭐ star)\n"
            "- Rainbow-colored levels\n"
            "- Golden Ratio (61.8%) highlighted\n"
            "- Click on a trade in Trade History to view its Fib levels"
        )
        indicators_layout.addWidget(self.show_fib_retracements)
        
        # Individual Fib Level Controls (indented)
        fib_levels_layout = QVBoxLayout()
        fib_levels_layout.setContentsMargins(30, 0, 0, 0)  # Indent
        
        self.show_fib_0382 = QCheckBox("38.2% (Blue)")
        self.show_fib_0382.setChecked(True)
        fib_levels_layout.addWidget(self.show_fib_0382)
        
        self.show_fib_0500 = QCheckBox("50.0% (Cyan)")
        self.show_fib_0500.setChecked(True)
        fib_levels_layout.addWidget(self.show_fib_0500)
        
        self.show_fib_0618 = QCheckBox("61.8% ⭐ Golden Ratio (Gold)")
        self.show_fib_0618.setChecked(True)
        fib_levels_layout.addWidget(self.show_fib_0618)
        
        self.show_fib_0786 = QCheckBox("78.6% (Orange)")
        self.show_fib_0786.setChecked(True)
        fib_levels_layout.addWidget(self.show_fib_0786)
        
        self.show_fib_1000 = QCheckBox("100% (Red)")
        self.show_fib_1000.setChecked(True)
        fib_levels_layout.addWidget(self.show_fib_1000)
        
        indicators_layout.addLayout(fib_levels_layout)
        
        group.setLayout(indicators_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        return widget
    
    def create_backtest_tab(self):
        """Create backtest parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Backtest Execution Parameters")
        form_layout = QFormLayout()
        
        # ATR stop multiplier
        self.atr_stop_mult = QDoubleSpinBox()
        self.atr_stop_mult.setRange(0.1, 5.0)
        self.atr_stop_mult.setValue(0.7)
        self.atr_stop_mult.setSingleStep(0.1)
        form_layout.addRow("ATR Stop Multiplier:", self.atr_stop_mult)
        
        # Reward to risk
        self.reward_r = QDoubleSpinBox()
        self.reward_r.setRange(0.5, 20.0)
        self.reward_r.setValue(2.0)
        self.reward_r.setSingleStep(0.1)
        self.reward_r.setSuffix(" R")
        form_layout.addRow("Reward:Risk Ratio:", self.reward_r)
        
        # Max hold
        self.max_hold = QSpinBox()
        self.max_hold.setRange(10, 1000)
        self.max_hold.setValue(96)
        self.max_hold.setSuffix(" bars")
        form_layout.addRow("Max Hold Period:", self.max_hold)
        
        # Fees
        self.fee_bp = QDoubleSpinBox()
        self.fee_bp.setRange(0.0, 500.0)
        self.fee_bp.setValue(5.0)
        self.fee_bp.setSingleStep(0.5)
        self.fee_bp.setSuffix(" bps")
        form_layout.addRow("Trading Fee:", self.fee_bp)
        
        # Slippage
        self.slippage_bp = QDoubleSpinBox()
        self.slippage_bp.setRange(0.0, 500.0)
        self.slippage_bp.setValue(5.0)
        self.slippage_bp.setSingleStep(0.5)
        self.slippage_bp.setSuffix(" bps")
        form_layout.addRow("Slippage:", self.slippage_bp)
        
        group.setLayout(form_layout)
        layout.addWidget(group)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_backtest_params)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        return widget

    def create_optimizer_tab(self):
        """Create optimization / genetic algorithm tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Genetic Algorithm Parameters")
        form_layout = QFormLayout()

        # Population size
        self.ga_population = QSpinBox()
        self.ga_population.setRange(4, 256)
        self.ga_population.setValue(24)
        self.ga_population.setSuffix(" individuals")
        form_layout.addRow("Population Size:", self.ga_population)

        # Mutation rate per parameter
        self.ga_mutation_rate = QDoubleSpinBox()
        self.ga_mutation_rate.setRange(0.0, 1.0)
        self.ga_mutation_rate.setSingleStep(0.05)
        self.ga_mutation_rate.setDecimals(3)
        self.ga_mutation_rate.setValue(0.3)
        form_layout.addRow("Mutation Rate:", self.ga_mutation_rate)

        # Mutation sigma (strength)
        self.ga_sigma = QDoubleSpinBox()
        self.ga_sigma.setRange(0.01, 1.0)
        self.ga_sigma.setSingleStep(0.01)
        self.ga_sigma.setDecimals(3)
        self.ga_sigma.setValue(0.1)
        form_layout.addRow("Mutation Sigma:", self.ga_sigma)

        # Elitism fraction
        self.ga_elite_fraction = QDoubleSpinBox()
        self.ga_elite_fraction.setRange(0.0, 0.5)
        self.ga_elite_fraction.setSingleStep(0.05)
        self.ga_elite_fraction.setDecimals(3)
        self.ga_elite_fraction.setValue(0.1)
        form_layout.addRow("Elite Fraction:", self.ga_elite_fraction)

        # Tournament size
        self.ga_tournament_size = QSpinBox()
        self.ga_tournament_size.setRange(2, 10)
        self.ga_tournament_size.setValue(3)
        form_layout.addRow("Tournament Size:", self.ga_tournament_size)

        # Mutation probability (per offspring)
        self.ga_mutation_probability = QDoubleSpinBox()
        self.ga_mutation_probability.setRange(0.0, 1.0)
        self.ga_mutation_probability.setSingleStep(0.05)
        self.ga_mutation_probability.setDecimals(3)
        self.ga_mutation_probability.setValue(0.9)
        form_layout.addRow("Mutation Probability:", self.ga_mutation_probability)

        group.setLayout(form_layout)
        layout.addWidget(group)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_ga_params)
        layout.addWidget(reset_btn)

        info = QLabel(
            "💡 These controls adjust the evolutionary optimizer.\n"
            "- Higher population improves exploration but needs more computation.\n"
            "- Mutation rate and sigma control how aggressively new solutions mutate.\n"
            "- Elite fraction preserves top performers per generation.\n"
            "- Tournament size alters selection pressure.\n"
            "- Mutation probability sets how often offspring are mutated."
        )
        info.setWordWrap(True)
        info.setProperty("variant", "secondary")
        layout.addWidget(info)

        layout.addStretch()
        return widget
    
    def load_from_settings(self):
        """Load UI values from saved settings."""
        self._loading_settings = True  # Prevent auto-adjust during load
        
        # Timeframe
        tf_key = f"{settings.timeframe}{settings.timeframe_unit[0]}"  # e.g., "15m"
        for i in range(self.timeframe_combo.count()):
            if self.timeframe_combo.itemData(i) == tf_key:
                self.timeframe_combo.setCurrentIndex(i)
                break
        
        # Data download
        try:
            from_parts = settings.last_date_from.split('-')
            self.date_from.setDate(QDate(int(from_parts[0]), int(from_parts[1]), int(from_parts[2])))
        except:
            pass
        
        try:
            to_parts = settings.last_date_to.split('-')
            self.date_to.setDate(QDate(int(to_parts[0]), int(to_parts[1]), int(to_parts[2])))
        except:
            pass
        
        # Strategy parameters
        self.ema_fast.setValue(settings.strategy_ema_fast)
        self.ema_slow.setValue(settings.strategy_ema_slow)
        self.z_window.setValue(settings.strategy_z_window)
        self.vol_z.setValue(settings.strategy_vol_z)
        self.tr_z.setValue(settings.strategy_tr_z)
        self.cloc_min.setValue(settings.strategy_cloc_min)
        self.atr_window.setValue(settings.strategy_atr_window)
        
        # Backtest parameters
        self.atr_stop_mult.setValue(settings.backtest_atr_stop_mult)
        self.reward_r.setValue(settings.backtest_reward_r)
        self.max_hold.setValue(settings.backtest_max_hold)
        self.fee_bp.setValue(settings.backtest_fee_bp)
        self.slippage_bp.setValue(settings.backtest_slippage_bp)

        # Genetic algorithm parameters
        self.ga_population.setValue(settings.ga_population_size)
        self.ga_mutation_rate.setValue(settings.ga_mutation_rate)
        self.ga_sigma.setValue(settings.ga_sigma)
        self.ga_elite_fraction.setValue(settings.ga_elite_fraction)
        self.ga_tournament_size.setValue(settings.ga_tournament_size)
        self.ga_mutation_probability.setValue(settings.ga_mutation_probability)
        
        # Chart indicators
        self.show_ema_fast.setChecked(settings.chart_ema_fast)
        self.show_ema_slow.setChecked(settings.chart_ema_slow)
        self.show_sma.setChecked(settings.chart_sma)
        self.show_rsi.setChecked(settings.chart_rsi)
        self.show_macd.setChecked(settings.chart_macd)
        self.show_volume.setChecked(settings.chart_volume)
        self.show_signals.setChecked(settings.chart_signals)
        self.show_entries.setChecked(settings.chart_entries)
        # show_exits removed - now integrated into show_entries (trade balls)
        
        # Acceleration settings (NEW)
        accel_mode = getattr(settings, 'acceleration_mode', 'multicore')
        for i in range(self.accel_mode_combo.count()):
            if self.accel_mode_combo.itemData(i) == accel_mode:
                self.accel_mode_combo.setCurrentIndex(i)
                break
        
        self.cpu_workers.setValue(getattr(settings, 'cpu_workers', self.cpu_workers.value()))
        self.gpu_batch_size.setValue(getattr(settings, 'gpu_batch_size', 512))  # Default 512 for better GPU utilization
        self.gpu_memory_fraction.setValue(getattr(settings, 'gpu_memory_fraction', 0.85))
        
        self._loading_settings = False  # Re-enable auto-adjust
    
    def on_timeframe_changed(self, index):
        """Handle timeframe change and optionally auto-adjust parameters."""
        if self._loading_settings:
            return  # Skip during initial load
        
        if not self.auto_adjust_params.isChecked():
            return  # User disabled auto-adjust
        
        # Get selected timeframe
        tf_key = self.timeframe_combo.currentData()
        mult, unit, label = TIMEFRAMES[tf_key]
        
        # Map to Timeframe enum
        tf_map = {
            "1m": Timeframe.m1,
            "3m": Timeframe.m3,
            "5m": Timeframe.m5,
            "10m": Timeframe.m10,
            "15m": Timeframe.m15,
            "30m": Timeframe.m30,
            "60m": Timeframe.m60,
            "1h": Timeframe.m60,
        }
        tf = tf_map.get(tf_key, Timeframe.m15)
        
        # Get defaults for this timeframe
        try:
            config = get_defaults_for_timeframe(tf)
            bar_counts = config.get_bar_counts()
            
            # Ask user for confirmation
            reply = QMessageBox.question(
                self,
                "Auto-Adjust Parameters",
                f"<h3>Adjust parameters for {label} timeframe?</h3>"
                f"<p>Current timeframe requires different parameter values to maintain "
                f"consistent time periods (e.g., 24 hours for EMA Fast).</p>"
                f"<h4>Proposed adjustments:</h4>"
                f"<table>"
                f"<tr><td><b>EMA Fast:</b></td><td>{self.ema_fast.value()} bars → {bar_counts['ema_fast_bars']} bars</td><td>(24 hours)</td></tr>"
                f"<tr><td><b>EMA Slow:</b></td><td>{self.ema_slow.value()} bars → {bar_counts['ema_slow_bars']} bars</td><td>(7 days)</td></tr>"
                f"<tr><td><b>Z-Window:</b></td><td>{self.z_window.value()} bars → {bar_counts['z_window_bars']} bars</td><td>(7 days)</td></tr>"
                f"<tr><td><b>ATR Window:</b></td><td>{self.atr_window.value()} bars → {bar_counts['atr_window_bars']} bars</td><td>(24 hours)</td></tr>"
                f"<tr><td><b>Max Hold:</b></td><td>{self.max_hold.value()} bars → {bar_counts['max_hold_bars']} bars</td><td>({config.max_hold_minutes//60} hours)</td></tr>"
                f"</table>"
                f"<p><b>Click Yes</b> to apply these adjustments.<br>"
                f"<b>Click No</b> to keep current values (not recommended).</p>",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Apply adjustments
                self.ema_fast.setValue(bar_counts['ema_fast_bars'])
                self.ema_slow.setValue(bar_counts['ema_slow_bars'])
                self.z_window.setValue(bar_counts['z_window_bars'])
                self.atr_window.setValue(bar_counts['atr_window_bars'])
                self.max_hold.setValue(bar_counts['max_hold_bars'])
                
                # Also adjust costs if significantly different
                if config.fee_bp != self.fee_bp.value() or config.slippage_bp != self.slippage_bp.value():
                    self.fee_bp.setValue(config.fee_bp)
                    self.slippage_bp.setValue(config.slippage_bp)
                
                QMessageBox.information(
                    self,
                    "Parameters Adjusted",
                    f"✓ Parameters have been adjusted for {label} timeframe.\n\n"
                    f"All time periods remain consistent (24h short-term, 7d long-term).\n"
                    f"Remember to save settings before closing!"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Parameters Not Adjusted",
                    "⚠️ Using inappropriate parameters for this timeframe may lead to:\n"
                    "- Poor signal quality\n"
                    "- Inconsistent backtest results\n"
                    "- Failed optimizations\n\n"
                    "Consider using time-based parameters or enabling auto-adjust.\n"
                    "See TIMEFRAME_SCALING_GUIDE.md for details."
                )
        
        except Exception as e:
            print(f"Error auto-adjusting parameters: {e}")
            import traceback
            traceback.print_exc()
    
    def create_acceleration_tab(self):
        """Create acceleration settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Acceleration Mode Group
        mode_group = QGroupBox("Acceleration Mode")
        mode_layout = QVBoxLayout()
        
        self.accel_mode_combo = QComboBox()
        self.accel_mode_combo.addItem("🖥️  CPU (Single Core)", "cpu")
        self.accel_mode_combo.addItem("🖥️ 🖥️  Multi-Core CPU", "multicore")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.accel_mode_combo.addItem(f"🚀 GPU ({gpu_name})", "gpu")
            else:
                self.accel_mode_combo.addItem("🚀 GPU (Not Available)", "gpu_unavailable")
                self.accel_mode_combo.setItemData(2, 0, Qt.UserRole - 1)  # Disable
        except ImportError:
            self.accel_mode_combo.addItem("🚀 GPU (PyTorch Not Installed)", "gpu_unavailable")
            self.accel_mode_combo.setItemData(2, 0, Qt.UserRole - 1)  # Disable
        
        mode_layout.addWidget(QLabel("Select acceleration method:"))
        mode_layout.addWidget(self.accel_mode_combo)
        self.accel_mode_combo.currentIndexChanged.connect(self._on_accel_mode_changed)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Multi-Core CPU Settings
        self.multicore_group = QGroupBox("Multi-Core CPU Settings")
        multicore_layout = QFormLayout()
        
        import multiprocessing
        max_cores = multiprocessing.cpu_count()
        
        self.cpu_workers = QSpinBox()
        self.cpu_workers.setRange(1, max_cores)
        self.cpu_workers.setValue(max(1, max_cores - 1))  # Leave 1 core for system
        self.cpu_workers.setSuffix(f" workers (max: {max_cores})")
        multicore_layout.addRow("Worker Processes:", self.cpu_workers)
        
        info_label = QLabel(
            f"<small>ℹ️ Your system has {max_cores} CPU cores.\n"
            "Using N-1 cores is recommended to keep system responsive.\n"
            "More workers = faster optimization, but higher CPU usage.</small>"
        )
        info_label.setWordWrap(True)
        multicore_layout.addRow(info_label)
        
        self.multicore_group.setLayout(multicore_layout)
        layout.addWidget(self.multicore_group)
        
        # GPU Settings
        self.gpu_group = QGroupBox("GPU Settings")
        gpu_layout = QFormLayout()
        
        self.gpu_batch_size = QSpinBox()
        self.gpu_batch_size.setRange(1, 512)
        self.gpu_batch_size.setValue(512)  # Increased from 150 for better GPU utilization
        self.gpu_batch_size.setSuffix(" individuals")
        gpu_layout.addRow("Batch Size:", self.gpu_batch_size)
        
        self.gpu_memory_fraction = QDoubleSpinBox()
        self.gpu_memory_fraction.setRange(0.1, 1.0)
        self.gpu_memory_fraction.setValue(0.85)
        self.gpu_memory_fraction.setSingleStep(0.05)
        self.gpu_memory_fraction.setDecimals(2)
        gpu_layout.addRow("Memory Usage:", self.gpu_memory_fraction)
        
        gpu_info_label = QLabel(
            "<small>ℹ️ <b>Batch Size</b>: Number of parameter sets processed simultaneously.\n"
            "Larger = better GPU utilization, but more memory usage.\n\n"
            "<b>Memory Usage</b>: Fraction of GPU VRAM to use (0.85 = 85%).\n"
            "Leave headroom for system and display.</small>"
        )
        gpu_info_label.setWordWrap(True)
        gpu_layout.addRow(gpu_info_label)
        
        # GPU Status
        try:
            import torch
            if torch.cuda.is_available():
                gpu_status = QLabel()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_status.setText(
                    f"<b>✓ GPU Detected:</b> {gpu_name}<br>"
                    f"<b>Total Memory:</b> {gpu_memory:.2f} GB<br>"
                    f"<b>CUDA Version:</b> {torch.version.cuda}"
                )
                gpu_status.setWordWrap(True)
                gpu_layout.addRow(gpu_status)
        except:
            pass
        
        self.gpu_group.setLayout(gpu_layout)
        layout.addWidget(self.gpu_group)
        
        # Performance Comparison
        perf_group = QGroupBox("📊 Expected Performance")
        perf_layout = QVBoxLayout()
        
        perf_text = QLabel(
            "<b>Approximate Speedup (24 individuals, 1000 bars):</b><br><br>"
            "🖥️  <b>CPU (Single Core)</b>: ~50s per generation (baseline)<br>"
            "🖥️ 🖥️  <b>Multi-Core CPU</b>: ~10s per generation (~5x faster)<br>"
            "🚀 <b>GPU</b>: ~2-5s per generation (~10-25x faster)<br><br>"
            "<small>Note: Actual speedup depends on hardware, data size, and parameter complexity.</small>"
        )
        perf_text.setWordWrap(True)
        perf_layout.addWidget(perf_text)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Warning for GPU
        warning_group = QGroupBox("⚠️ Important Notes")
        warning_layout = QVBoxLayout()
        
        warning_text = QLabel(
            "<b>GPU Acceleration:</b><br>"
            "• GPU mode is VALIDATED and safe for production use<br>"
            "• Results match CPU exactly (100% tested)<br>"
            "• Automatic fallback to CPU if GPU unavailable<br>"
            "• See GPU_VALIDATION_REPORT.md for details<br><br>"
            "<b>Recommendations:</b><br>"
            "• Start with Multi-Core CPU for reliability<br>"
            "• Use GPU for large populations (>50 individuals)<br>"
            "• Monitor GPU temperature during long runs<br>"
            "• GPU requires ~2GB VRAM for typical workloads"
        )
        warning_text.setWordWrap(True)
        warning_layout.addWidget(warning_text)
        
        warning_group.setLayout(warning_layout)
        layout.addWidget(warning_group)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_acceleration_params)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        
        # Initial state
        self._on_accel_mode_changed(0)
        
        return widget
    
    def _on_accel_mode_changed(self, index):
        """Handle acceleration mode change."""
        mode = self.accel_mode_combo.currentData()
        
        # Show/hide relevant groups
        self.multicore_group.setVisible(mode == "multicore")
        self.gpu_group.setVisible(mode == "gpu")
    
    def reset_acceleration_params(self):
        """Reset acceleration parameters to defaults."""
        import multiprocessing
        max_cores = multiprocessing.cpu_count()
        
        self.accel_mode_combo.setCurrentIndex(1)  # Multi-Core CPU default
        self.cpu_workers.setValue(max(1, max_cores - 1))
        self.gpu_batch_size.setValue(512)  # Increased from 150 for better GPU utilization
        self.gpu_memory_fraction.setValue(0.85)
    
    def save_settings(self):
        """Save all settings to .env file."""
        try:
            # Get timeframe
            tf_key = self.timeframe_combo.currentData()
            mult, unit, _ = TIMEFRAMES[tf_key]
            
            settings_dict = {
                # Timeframe
                'timeframe': str(mult),
                'timeframe_unit': unit,
                
                # Last download
                'last_ticker': 'X:ADAUSD',
                'last_date_from': self.date_from.date().toString("yyyy-MM-dd"),
                'last_date_to': self.date_to.date().toString("yyyy-MM-dd"),
                
                # Strategy
                'strategy_ema_fast': self.ema_fast.value(),
                'strategy_ema_slow': self.ema_slow.value(),
                'strategy_z_window': self.z_window.value(),
                'strategy_vol_z': self.vol_z.value(),
                'strategy_tr_z': self.tr_z.value(),
                'strategy_cloc_min': self.cloc_min.value(),
                'strategy_atr_window': self.atr_window.value(),
                
                # Backtest
                'backtest_atr_stop_mult': self.atr_stop_mult.value(),
                'backtest_reward_r': self.reward_r.value(),
                'backtest_max_hold': self.max_hold.value(),
                'backtest_fee_bp': self.fee_bp.value(),
                'backtest_slippage_bp': self.slippage_bp.value(),
                
                # Genetic algorithm
                'ga_population_size': self.ga_population.value(),
                'ga_mutation_rate': self.ga_mutation_rate.value(),
                'ga_sigma': self.ga_sigma.value(),
                'ga_elite_fraction': self.ga_elite_fraction.value(),
                'ga_tournament_size': self.ga_tournament_size.value(),
                'ga_mutation_probability': self.ga_mutation_probability.value(),
                
                # Acceleration settings (NEW)
                'acceleration_mode': self.accel_mode_combo.currentData(),
                'cpu_workers': self.cpu_workers.value(),
                'gpu_batch_size': self.gpu_batch_size.value(),
                'gpu_memory_fraction': self.gpu_memory_fraction.value(),
                
                # Chart indicators
                'chart_ema_fast': self.show_ema_fast.isChecked(),
                'chart_ema_slow': self.show_ema_slow.isChecked(),
                'chart_sma': self.show_sma.isChecked(),
                'chart_rsi': self.show_rsi.isChecked(),
                'chart_macd': self.show_macd.isChecked(),
                'chart_volume': self.show_volume.isChecked(),
                'chart_signals': self.show_signals.isChecked(),
                'chart_entries': self.show_entries.isChecked(),
                'chart_exits': True,  # Always true for backwards compatibility (not shown in UI)
            }
            
            SettingsManager.save_to_env(settings_dict)
            SettingsManager.reload_settings()
            
            self.settings_saved.emit()
            
            QMessageBox.information(
                self,
                "Settings Saved",
                "All settings have been saved successfully!\n\n"
                "Your configuration will be restored the next time you open the application."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save settings:\n{str(e)}"
            )
    
    async def download_data(self):
        """Download data from Polygon.io with progress updates."""
        try:
            # Validate API key
            if not settings.polygon_api_key:
                QMessageBox.warning(
                    self,
                    "API Key Missing",
                    "Please set POLYGON_API_KEY in your .env file"
                )
                return
            
            # Get dates
            from_date = self.date_from.date().toString("yyyy-MM-dd")
            to_date = self.date_to.date().toString("yyyy-MM-dd")
            
            # Get timeframe
            tf_key = self.timeframe_combo.currentData()
            mult, unit, label = TIMEFRAMES[tf_key]

            tf_map = {
                1: Timeframe.m1,
                3: Timeframe.m3,
                5: Timeframe.m5,
                10: Timeframe.m10,
                15: Timeframe.m15,
                30: Timeframe.m30,
                60: Timeframe.m60,
            }
            tf_enum = tf_map.get(mult)
            if tf_enum is None:
                QMessageBox.warning(
                    self,
                    "Unsupported Timeframe",
                    "Selected timeframe is not yet supported in the main application.\n"
                    "Defaulting to 15-minute timeframe for this download."
                )
                tf_enum = Timeframe.m15
                mult = 15
                unit = "minute"
                label = "15 minutes"
            
            # Update UI
            self.download_btn.setEnabled(False)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            
            # Create provider
            if not self.dp:
                self.dp = DataProvider()

            # Estimate download effort for the requested timeframe
            download_estimate = self.dp.estimate_download(from_date, to_date, mult, unit)

            def format_bars(current: int, total: int | None) -> str:
                if total and total > 0:
                    return f"{current:,}/{total:,}"
                return f"{current:,}"

            self.progress_bar.setRange(0, max(download_estimate.pages, 1))
            self.progress_bar.setValue(0)
            est_time_text = self.format_duration(download_estimate.seconds_total)
            self.progress_label.setText(
                "Free tier: 5 API calls/min.\n"
                f"Downloading ADA data ({label}) {from_date} → {to_date}.\n"
                f"Estimated {download_estimate.pages} request(s) (~{est_time_text})."
            )

            async def on_progress(progress):
                total_pages = max(progress.total_pages, 1)
                if self.progress_bar.maximum() != total_pages:
                    self.progress_bar.setRange(0, total_pages)

                if progress.page == 0:
                    return

                self.progress_bar.setValue(progress.page)

                remaining_text = (
                    f"≈ {self.format_duration(progress.seconds_remaining)} remaining"
                    if progress.seconds_remaining > 0
                    else "Finalizing..."
                )

                bars_text = format_bars(
                    progress.items_received,
                    progress.estimated_total_items,
                )

                self.progress_label.setText(
                    f"Request {progress.page}/{total_pages} complete "
                    f"| Bars fetched: {bars_text} | {remaining_text}"
                )
            
            # Fetch data directly at requested timeframe (force fresh data)
            df = await self.dp.fetch(
                "X:ADAUSD",
                tf_enum,
                from_date,
                to_date,
                progress_callback=on_progress,
                estimate=download_estimate,
                force_download=True,
            )
            self.progress_bar.setValue(self.progress_bar.maximum())
            
            if len(df) == 0:
                self.progress_label.setText("⚠ No data received. Check date range and API key.")
                QMessageBox.warning(
                    self,
                    "No Data",
                    "No data was returned. Please check:\n"
                    "- Date range is valid\n"
                    "- API key is correct\n"
                    "- You haven't exceeded API quota"
                )
            else:
                self.downloaded_data = df
                self.progress_label.setText(
                    f"✓ Downloaded {len(df)} bars ({label}) | "
                    f"Range: {df.index[0]} to {df.index[-1]}"
                )
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(100)
                
                # Auto-save settings after successful download
                self.save_settings()
                
                # Emit signal
                self.data_downloaded.emit(df)
                
                QMessageBox.information(
                    self,
                    "Download Complete",
                    f"Successfully downloaded {len(df)} {label} bars\n"
                    f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}\n\n"
                    f"Settings have been auto-saved."
                )
        
        except Exception as e:
            self.progress_label.setText(f"✗ Error: {str(e)}")
            QMessageBox.critical(
                self,
                "Download Error",
                f"Failed to download data:\n{str(e)}"
            )
        
        finally:
            self.download_btn.setEnabled(True)
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format seconds into a compact human-readable string."""
        if seconds <= 0:
            return "under 1s"

        total_seconds = int(round(seconds))
        minutes, sec = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        parts = []

        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if sec or not parts:
            parts.append(f"{sec}s")

        return " ".join(parts)

    def reset_strategy_params(self):
        """Reset strategy parameters to defaults."""
        self.ema_fast.setValue(96)
        self.ema_slow.setValue(672)
        self.z_window.setValue(672)
        self.vol_z.setValue(2.0)
        self.tr_z.setValue(1.2)
        self.cloc_min.setValue(0.6)
        self.atr_window.setValue(96)
    
    def reset_backtest_params(self):
        """Reset backtest parameters to defaults."""
        self.atr_stop_mult.setValue(0.7)
        self.reward_r.setValue(2.0)
        self.max_hold.setValue(96)
        self.fee_bp.setValue(5.0)
        self.slippage_bp.setValue(5.0)
    
    def set_strategy_params(self, params: SellerParams):
        """Update strategy tab controls from SellerParams."""
        self.ema_fast.setValue(int(params.ema_fast))
        self.ema_slow.setValue(int(params.ema_slow))
        self.z_window.setValue(int(params.z_window))
        self.vol_z.setValue(float(params.vol_z))
        self.tr_z.setValue(float(params.tr_z))
        self.cloc_min.setValue(float(params.cloc_min))
        self.atr_window.setValue(int(params.atr_window))
    
    def set_backtest_params(self, params: BacktestParams):
        """Update backtest tab controls from BacktestParams."""
        self.atr_stop_mult.setValue(float(params.atr_stop_mult))
        self.reward_r.setValue(float(params.reward_r))
        self.max_hold.setValue(int(params.max_hold))
        self.fee_bp.setValue(float(params.fee_bp))
        self.slippage_bp.setValue(float(params.slippage_bp))

    def reset_ga_params(self):
        """Reset genetic algorithm parameters to defaults."""
        self.ga_population.setValue(24)
        self.ga_mutation_rate.setValue(0.3)
        self.ga_sigma.setValue(0.1)
        self.ga_elite_fraction.setValue(0.1)
        self.ga_tournament_size.setValue(3)
        self.ga_mutation_probability.setValue(0.9)
    
    def get_timeframe(self):
        """Get selected timeframe as (multiplier, unit) tuple."""
        tf_key = self.timeframe_combo.currentData()
        mult, unit, _ = TIMEFRAMES[tf_key]
        return mult, unit
    
    def get_strategy_params(self):
        """Get strategy parameters from UI."""
        return SellerParams(
            ema_fast=self.ema_fast.value(),
            ema_slow=self.ema_slow.value(),
            z_window=self.z_window.value(),
            vol_z=self.vol_z.value(),
            tr_z=self.tr_z.value(),
            cloc_min=self.cloc_min.value(),
            atr_window=self.atr_window.value()
        )
    
    def get_backtest_params(self):
        """Get backtest parameters from UI."""
        return BacktestParams(
            atr_stop_mult=self.atr_stop_mult.value(),
            reward_r=self.reward_r.value(),
            max_hold=self.max_hold.value(),
            fee_bp=self.fee_bp.value(),
            slippage_bp=self.slippage_bp.value()
        )
    
    def get_indicator_config(self):
        """Get indicator display configuration."""
        return {
            'ema_fast': self.show_ema_fast.isChecked(),
            'ema_slow': self.show_ema_slow.isChecked(),
            'sma': self.show_sma.isChecked(),
            'rsi': self.show_rsi.isChecked(),
            'macd': self.show_macd.isChecked(),
            'volume': self.show_volume.isChecked(),
            'signals': self.show_signals.isChecked(),
            'entries': self.show_entries.isChecked(),
            # 'exits' not used anymore - trade balls show both entry and exit
            'fib_retracements': self.show_fib_retracements.isChecked(),
            'fib_0382': self.show_fib_0382.isChecked(),
            'fib_0500': self.show_fib_0500.isChecked(),
            'fib_0618': self.show_fib_0618.isChecked(),
            'fib_0786': self.show_fib_0786.isChecked(),
            'fib_1000': self.show_fib_1000.isChecked(),
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.dp:
            await self.dp.close()
