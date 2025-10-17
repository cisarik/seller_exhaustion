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
        
        # Indicator Selection Tab (moved to first position)
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
        # self.atr_stop_mult = QDoubleSpinBox()
        # self.atr_stop_mult.setRange(0.1, 5.0)
        # self.atr_stop_mult.setValue(0.7)
        # self.atr_stop_mult.setSingleStep(0.1)
        # form_layout.addRow("ATR Stop Multiplier:", self.atr_stop_mult)
        
        # Reward to risk
        # self.reward_r = QDoubleSpinBox()
        # self.reward_r.setRange(0.5, 20.0)
        # self.reward_r.setValue(2.0)
        # self.reward_r.setSingleStep(0.1)
        # self.reward_r.setSuffix(" R")
        # form_layout.addRow("Reward:Risk Ratio:", self.reward_r)
        
        # Max hold
        # self.max_hold = QSpinBox()
        # self.max_hold.setRange(10, 1000)
        # self.max_hold.setValue(96)
        # self.max_hold.setSuffix(" bars")
        # form_layout.addRow("Max Hold Period:", self.max_hold)
        
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

        # Optimizer Iterations (Common to all optimizers)
        common_group = QGroupBox("Optimizer Settings")
        common_layout = QFormLayout()
        
        self.optimizer_iterations = QSpinBox()
        self.optimizer_iterations.setRange(10, 1000)
        self.optimizer_iterations.setValue(50)
        self.optimizer_iterations.setSuffix(" iterations")
        self.optimizer_iterations.setToolTip("Number of iterations for multi-step optimization (used by all optimizer types)")
        common_layout.addRow("Iterations:", self.optimizer_iterations)
        
        common_group.setLayout(common_layout)
        layout.addWidget(common_group)

        # Genetic Algorithm Parameters
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

        reset_btn = QPushButton("Reset GA Defaults")
        reset_btn.clicked.connect(self.reset_ga_params)
        layout.addWidget(reset_btn)

        info = QLabel(
            "💡 Genetic Algorithm (Evolutionary) Optimizer:\n"
            "- Higher population improves exploration but needs more computation.\n"
            "- Mutation rate and sigma control how aggressively new solutions mutate.\n"
            "- Elite fraction preserves top performers per generation.\n"
            "- Tournament size alters selection pressure.\n"
            "- Mutation probability sets how often offspring are mutated."
        )
        info.setWordWrap(True)
        info.setProperty("variant", "secondary")
        layout.addWidget(info)

        # ADAM Optimizer Parameters
        adam_group = QGroupBox("ADAM Optimizer Parameters")
        adam_layout = QFormLayout()
        
        # Learning rate
        self.adam_learning_rate = QDoubleSpinBox()
        self.adam_learning_rate.setRange(0.0001, 1.0)
        self.adam_learning_rate.setSingleStep(0.001)
        self.adam_learning_rate.setDecimals(4)
        self.adam_learning_rate.setValue(0.01)
        adam_layout.addRow("Learning Rate:", self.adam_learning_rate)
        
        # Epsilon (for finite differences)
        self.adam_epsilon = QDoubleSpinBox()
        self.adam_epsilon.setRange(1e-5, 0.1)
        self.adam_epsilon.setSingleStep(0.0001)
        self.adam_epsilon.setDecimals(5)
        self.adam_epsilon.setValue(0.001)
        adam_layout.addRow("FD Epsilon:", self.adam_epsilon)
        
        # Max gradient norm
        self.adam_max_grad_norm = QDoubleSpinBox()
        self.adam_max_grad_norm.setRange(0.1, 10.0)
        self.adam_max_grad_norm.setSingleStep(0.1)
        self.adam_max_grad_norm.setDecimals(2)
        self.adam_max_grad_norm.setValue(1.0)
        adam_layout.addRow("Max Gradient Norm:", self.adam_max_grad_norm)
        
        # Beta1 (momentum)
        self.adam_beta1 = QDoubleSpinBox()
        self.adam_beta1.setRange(0.0, 1.0)
        self.adam_beta1.setSingleStep(0.01)
        self.adam_beta1.setDecimals(3)
        self.adam_beta1.setValue(0.9)
        adam_layout.addRow("Beta1 (Momentum):", self.adam_beta1)
        
        # Beta2 (RMSprop-like)
        self.adam_beta2 = QDoubleSpinBox()
        self.adam_beta2.setRange(0.0, 1.0)
        self.adam_beta2.setSingleStep(0.001)
        self.adam_beta2.setDecimals(4)
        self.adam_beta2.setValue(0.999)
        adam_layout.addRow("Beta2 (RMSprop):", self.adam_beta2)
        
        # Epsilon stability
        self.adam_epsilon_stability = QDoubleSpinBox()
        self.adam_epsilon_stability.setRange(1e-10, 1e-6)
        self.adam_epsilon_stability.setSingleStep(1e-9)
        self.adam_epsilon_stability.setDecimals(10)
        self.adam_epsilon_stability.setValue(1e-8)
        adam_layout.addRow("Epsilon Stability:", self.adam_epsilon_stability)
        
        adam_group.setLayout(adam_layout)
        layout.addWidget(adam_group)
        
        adam_reset_btn = QPushButton("Reset ADAM Defaults")
        adam_reset_btn.clicked.connect(self.reset_adam_params)
        layout.addWidget(adam_reset_btn)
        
        adam_info = QLabel(
            "💡 ADAM (Adaptive Moment Estimation) Optimizer:\n"
            "- Learning rate controls step size for parameter updates.\n"
            "- FD epsilon is step size for finite difference gradient approximation.\n"
            "- Max gradient norm clips gradients for stability.\n"
            "- Beta1 controls exponential decay rate for first moment (momentum).\n"
            "- Beta2 controls exponential decay rate for second moment (RMSprop-like).\n"
            "- Epsilon stability prevents division by zero in ADAM updates."
        )
        adam_info.setWordWrap(True)
        adam_info.setProperty("variant", "secondary")
        layout.addWidget(adam_info)

        layout.addStretch()
        return widget
    
    def load_from_settings(self):
        """Load UI values from saved settings."""
        # Backtest parameters
        self.fee_bp.setValue(settings.backtest_fee_bp)
        self.slippage_bp.setValue(settings.backtest_slippage_bp)

        # Optimizer parameters (common)
        self.optimizer_iterations.setValue(settings.optimizer_iterations)
        
        # Genetic algorithm parameters
        self.ga_population.setValue(settings.ga_population_size)
        self.ga_mutation_rate.setValue(settings.ga_mutation_rate)
        self.ga_sigma.setValue(settings.ga_sigma)
        self.ga_elite_fraction.setValue(settings.ga_elite_fraction)
        self.ga_tournament_size.setValue(settings.ga_tournament_size)
        self.ga_mutation_probability.setValue(settings.ga_mutation_probability)
        
        # ADAM optimizer parameters
        self.adam_learning_rate.setValue(settings.adam_learning_rate)
        self.adam_epsilon.setValue(settings.adam_epsilon)
        self.adam_max_grad_norm.setValue(settings.adam_max_grad_norm)
        self.adam_beta1.setValue(settings.adam_beta1)
        self.adam_beta2.setValue(settings.adam_beta2)
        self.adam_epsilon_stability.setValue(settings.adam_epsilon_stability)
        
        # Chart indicators
        self.show_ema_fast.setChecked(settings.chart_ema_fast)
        self.show_ema_slow.setChecked(settings.chart_ema_slow)
        self.show_sma.setChecked(settings.chart_sma)
        self.show_rsi.setChecked(settings.chart_rsi)
        self.show_macd.setChecked(settings.chart_macd)
        self.show_volume.setChecked(settings.chart_volume)
        self.show_signals.setChecked(settings.chart_signals)
        self.show_entries.setChecked(settings.chart_entries)
        
        # Acceleration settings (NEW)
        accel_mode = getattr(settings, 'acceleration_mode', 'multicore')
        for i in range(self.accel_mode_combo.count()):
            if self.accel_mode_combo.itemData(i) == accel_mode:
                self.accel_mode_combo.setCurrentIndex(i)
                break
        
        self.cpu_workers.setValue(getattr(settings, 'cpu_workers', self.cpu_workers.value()))
        self.gpu_batch_size.setValue(getattr(settings, 'gpu_batch_size', 512))  # Default 512 for better GPU utilization
        self.gpu_memory_fraction.setValue(getattr(settings, 'gpu_memory_fraction', 0.85))
    

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
            settings_dict = {
                # Backtest
                'backtest_fee_bp': self.fee_bp.value(),
                'backtest_slippage_bp': self.slippage_bp.value(),
                
                # Optimizer (Common)
                'optimizer_iterations': self.optimizer_iterations.value(),
                
                # Genetic algorithm
                'ga_population_size': self.ga_population.value(),
                'ga_mutation_rate': self.ga_mutation_rate.value(),
                'ga_sigma': self.ga_sigma.value(),
                'ga_elite_fraction': self.ga_elite_fraction.value(),
                'ga_tournament_size': self.ga_tournament_size.value(),
                'ga_mutation_probability': self.ga_mutation_probability.value(),
                
                # ADAM Optimizer
                'adam_learning_rate': self.adam_learning_rate.value(),
                'adam_epsilon': self.adam_epsilon.value(),
                'adam_max_grad_norm': self.adam_max_grad_norm.value(),
                'adam_beta1': self.adam_beta1.value(),
                'adam_beta2': self.adam_beta2.value(),
                'adam_epsilon_stability': self.adam_epsilon_stability.value(),
                
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


    def reset_backtest_params(self):
        """Reset backtest parameters to defaults."""
        # self.atr_stop_mult.setValue(0.7)
        # self.reward_r.setValue(2.0)
        # self.max_hold.setValue(96)
        self.fee_bp.setValue(5.0)
        self.slippage_bp.setValue(5.0)
    
    def set_backtest_params(self, params: BacktestParams):
        """Update backtest tab controls from BacktestParams."""
        # self.atr_stop_mult.setValue(float(params.atr_stop_mult))
        # self.reward_r.setValue(float(params.reward_r))
        # self.max_hold.setValue(int(params.max_hold))
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
    
    def reset_adam_params(self):
        """Reset ADAM optimizer parameters to defaults."""
        self.adam_learning_rate.setValue(0.01)
        self.adam_epsilon.setValue(0.001)
        self.adam_max_grad_norm.setValue(1.0)
        self.adam_beta1.setValue(0.9)
        self.adam_beta2.setValue(0.999)
        self.adam_epsilon_stability.setValue(1e-8)
    
    def get_timeframe(self):
        """Get selected timeframe as (multiplier, unit) tuple.
        
        Note: Timeframe is now managed in the main window.
        Returns default 15m for backwards compatibility.
        """
        return 15, "minute"
    
    def get_strategy_params(self):
        """Get strategy parameters - now returns defaults since params are managed in main window."""
        # Strategy parameters are now managed in the main window compact editor
        # Return defaults for backwards compatibility
        return SellerParams()
    
    def get_backtest_params(self):
        """Get backtest parameters from UI."""
        return BacktestParams(
            # atr_stop_mult=self.atr_stop_mult.value(),
            # reward_r=self.reward_r.value(),
            # max_hold=self.max_hold.value(),
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
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.dp:
            await self.dp.close()
