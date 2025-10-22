from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QPushButton, QDateEdit, QProgressBar,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QMessageBox, QComboBox
)
from PySide6.QtCore import Qt, QDate, Signal
import asyncio
from datetime import datetime
import multiprocessing

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
        
        # Evolution Coach Tab
        tabs.addTab(self.create_coach_tab(), "Evolution Coach")

        # Logging Tab
        tabs.addTab(self.create_logging_tab(), "Logging")
        
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

    def create_logging_tab(self):
        """Create logging configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Console logging group
        console_group = QGroupBox("Console Logging")
        console_layout = QFormLayout()

        # Log level
        self.log_level = QComboBox()
        for lvl in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self.log_level.addItem(lvl, lvl)
        console_layout.addRow("Log Level:", self.log_level)

        # Progress bars
        self.log_progress_bars = QCheckBox("Show progress bars during optimization (tqdm)")
        self.log_progress_bars.setChecked(True)
        console_layout.addRow("Progress Bars:", self.log_progress_bars)

        console_group.setLayout(console_layout)
        layout.addWidget(console_group)

        # Feature logging group
        feature_group = QGroupBox("Feature Build Logging")
        feature_layout = QVBoxLayout()
        self.log_feature_builds = QCheckBox("Log feature builds (INFO level)")
        self.log_feature_builds.setToolTip("When enabled, emits a line per feature build; otherwise logs at DEBUG only.")
        feature_layout.addWidget(self.log_feature_builds)
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)

        # Coach logging group
        coach_group = QGroupBox("Coach Logging")
        coach_layout = QVBoxLayout()
        self.coach_debug_payloads = QCheckBox("Log full LLM payloads and responses (COACH_DEBUG_PAYLOADS)")
        self.coach_debug_payloads.setToolTip("Very noisy. Includes full prompts and responses for diagnostics.")
        coach_layout.addWidget(self.coach_debug_payloads)
        coach_group.setLayout(coach_layout)
        layout.addWidget(coach_group)

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

        cpu_count = multiprocessing.cpu_count()
        self.optimizer_workers = QSpinBox()
        self.optimizer_workers.setRange(1, cpu_count)
        self.optimizer_workers.setValue(max(1, min(cpu_count, settings.optimizer_workers)))
        self.optimizer_workers.setSuffix(f" workers (max: {cpu_count})")
        self.optimizer_workers.setToolTip("Parallel worker processes used for optimization (set to 1 to run sequentially)")
        common_layout.addRow("Worker Processes:", self.optimizer_workers)
        
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
        self.adam_epsilon.setSingleStep(0.001)
        self.adam_epsilon.setDecimals(4)
        self.adam_epsilon.setValue(0.02)
        self.adam_epsilon.setToolTip("Step size for finite difference gradient approximation. Too small = no parameter change, too large = noisy gradients. Recommended: 0.01-0.05")
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
    
    def create_coach_tab(self):
        """Create Evolution Coach settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Enable/Provider Group (NEW)
        enable_group = QGroupBox("Evolution Coach")
        enable_layout = QFormLayout()
        
        # Enable checkbox
        self.coach_enabled = QCheckBox("Enable Evolution Coach Agent")
        self.coach_enabled.setChecked(True)
        self.coach_enabled.setToolTip("Enable/disable the Evolution Coach agent during optimization")
        enable_layout.addRow("Enabled:", self.coach_enabled)
        
        # Provider selection
        self.coach_provider = QComboBox()
        self.coach_provider.addItem("🖥️  Local (LM Studio)", "local")
        self.coach_provider.addItem("🌐 Online (Openrouter)", "openrouter")
        self.coach_provider.setCurrentIndex(0)
        self.coach_provider.setToolTip(
            "Choose LLM provider:\n"
            "• Local: Free, private, requires LM Studio installed\n"
            "• Openrouter: Paid, faster, more powerful models (Claude, GPT-4)"
        )
        self.coach_provider.currentIndexChanged.connect(self._on_provider_changed)
        enable_layout.addRow("Provider:", self.coach_provider)
        
        enable_group.setLayout(enable_layout)
        layout.addWidget(enable_group)
        
        # LM Studio Configuration (local provider only)
        self.lms_group = QGroupBox("LM Studio Configuration")
        lms_layout = QFormLayout()
        
        # Model dropdown (populated by lms ls)
        model_row = QHBoxLayout()
        self.lms_model = QComboBox()
        self.lms_model.setEditable(True)
        self.lms_model.addItem("google/gemma-3-12b", "google/gemma-3-12b")
        self.lms_model.addItem("gemma-2-9b-it", "gemma-2-9b-it")
        self.lms_model.setToolTip("Select model from LM Studio")
        model_row.addWidget(self.lms_model)
        
        # Refresh models button
        refresh_btn = QPushButton("🔄")
        refresh_btn.setMaximumWidth(40)
        refresh_btn.setToolTip("Refresh available models from LM Studio")
        refresh_btn.clicked.connect(self._refresh_lms_models)
        model_row.addWidget(refresh_btn)
        
        lms_layout.addRow("Model:", model_row)
        
        # Context Length
        self.coach_context_length = QSpinBox()
        self.coach_context_length.setRange(1000, 131072)
        self.coach_context_length.setValue(5000)
        self.coach_context_length.setSingleStep(1000)
        self.coach_context_length.setSuffix(" tokens")
        self.coach_context_length.setToolTip("Model context window size (131072 = Gemma max)")
        lms_layout.addRow("Context Length:", self.coach_context_length)
        
        # GPU Offload Ratio
        self.coach_gpu = QDoubleSpinBox()
        self.coach_gpu.setRange(0.0, 1.0)
        self.coach_gpu.setSingleStep(0.1)
        self.coach_gpu.setDecimals(1)
        self.coach_gpu.setValue(0.6)
        self.coach_gpu.setToolTip("GPU offload ratio: 0.0 = CPU only, 1.0 = max GPU")
        lms_layout.addRow("GPU Offload:", self.coach_gpu)
        
        # Auto Reload Model
        self.coach_auto_reload = QCheckBox("Auto reload model after analysis")
        self.coach_auto_reload.setChecked(True)
        self.coach_auto_reload.setToolTip("Automatically unload and reload model to clear context window")
        lms_layout.addRow("Auto Reload:", self.coach_auto_reload)
        
        self.lms_group.setLayout(lms_layout)
        layout.addWidget(self.lms_group)
        
        # Openrouter Configuration (online provider only)
        self.openrouter_group = QGroupBox("Openrouter Configuration")
        openrouter_layout = QFormLayout()
        
        # API Key input
        api_key_row = QHBoxLayout()
        self.openrouter_api_key = QLineEdit()
        self.openrouter_api_key.setEchoMode(QLineEdit.Password)
        self.openrouter_api_key.setPlaceholderText("sk-or-v1-...")
        self.openrouter_api_key.setToolTip(
            "Openrouter API key\n"
            "Get one at: https://openrouter.ai/keys"
        )
        api_key_row.addWidget(self.openrouter_api_key)
        
        # Show/hide API key button
        self.show_key_btn = QPushButton("👁️")
        self.show_key_btn.setMaximumWidth(40)
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.setToolTip("Show/hide API key")
        self.show_key_btn.clicked.connect(lambda checked: 
            self.openrouter_api_key.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        api_key_row.addWidget(self.show_key_btn)
        
        openrouter_layout.addRow("API Key:", api_key_row)
        
        # Model input with popular defaults
        self.openrouter_model = QComboBox()
        self.openrouter_model.setEditable(True)
        self.openrouter_model.addItem("anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-sonnet")
        self.openrouter_model.addItem("openai/gpt-4o", "openai/gpt-4o")
        self.openrouter_model.addItem("anthropic/claude-3-opus", "anthropic/claude-3-opus")
        self.openrouter_model.addItem("google/gemini-pro-1.5", "google/gemini-pro-1.5")
        self.openrouter_model.setToolTip(
            "Openrouter model identifier\n"
            "Browse models at: https://openrouter.ai/models"
        )
        openrouter_layout.addRow("Model:", self.openrouter_model)
        
        # Info label
        info_label = QLabel(
            "💡 <b>Recommended for Evolution Coach:</b><br>"
            "• <b>Claude 3.5 Sonnet</b>: Best reasoning, ~$0.06 per analysis<br>"
            "• <b>GPT-4o</b>: Fast, good quality, ~$0.04 per analysis<br>"
            "• <b>Claude 3 Opus</b>: Most powerful, ~$0.15 per analysis"
        )
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText)
        info_label.setStyleSheet("color: #888; font-size: 11px; padding: 8px;")
        openrouter_layout.addRow(info_label)
        
        self.openrouter_group.setLayout(openrouter_layout)
        layout.addWidget(self.openrouter_group)
        
        # Common Coach Settings Group
        coach_group = QGroupBox("Coach Behavior Settings")
        coach_layout = QFormLayout()
        
        # System Prompt Selection
        self.coach_system_prompt = QComboBox()
        self.coach_system_prompt.addItem("🤖 Agent Mode (agent02)", "agent02")
        self.coach_system_prompt.addItem("🤖 Agent Mode (agent01)", "agent01")
        self.coach_system_prompt.addItem("blocking_coach_v1", "blocking_coach_v1")
        self.coach_system_prompt.addItem("async_coach_v1", "async_coach_v1")
        self.coach_system_prompt.setCurrentIndex(0)  # Default to agent02
        self.coach_system_prompt.setToolTip("Select the system prompt version for the coach\nagent01 = Full agent with tool-calling capabilities")
        coach_layout.addRow("System Prompt:", self.coach_system_prompt)
        
        # Coach Analysis Interval
        self.coach_interval = QSpinBox()
        self.coach_interval.setRange(1, 100)
        self.coach_interval.setValue(10)
        self.coach_interval.setSuffix(" gens")
        self.coach_interval.setToolTip("Coach analyzes every N generations (10, 20, 30, etc)")
        coach_layout.addRow("Analysis Interval:", self.coach_interval)
        
        # Removed: Population Context Window + Max Log Generations (full history is used)
        
        # LLM Response Timeout
        self.coach_response_timeout = QSpinBox()
        self.coach_response_timeout.setRange(30, 36000)  # 30 seconds to 10 hours
        self.coach_response_timeout.setValue(3600)  # Default 1 hour
        self.coach_response_timeout.setSingleStep(60)  # Step by 60 seconds
        self.coach_response_timeout.setSuffix(" sec")
        self.coach_response_timeout.setToolTip(
            "LLM response timeout in seconds (default: 3600 = 1 hour)\n"
            "Increase if LM Studio is slow or model is large"
        )
        coach_layout.addRow("LLM Response Timeout:", self.coach_response_timeout)
        
        # Removed: Agent Max Iterations (fixed at 50 internally)
        
        coach_group.setLayout(coach_layout)
        layout.addWidget(coach_group)
        
        # CPU Workers Group (moved from Optimization)
        cpu_group = QGroupBox("CPU Worker Settings")
        cpu_layout = QFormLayout()
        
        cpu_count = multiprocessing.cpu_count()
        self.cpu_workers = QSpinBox()
        self.cpu_workers.setRange(1, cpu_count)
        self.cpu_workers.setValue(7)
        self.cpu_workers.setSuffix(f" workers (max: {cpu_count})")
        self.cpu_workers.setToolTip("Worker processes for CPU-based operations")
        cpu_layout.addRow("CPU Workers:", self.cpu_workers)
        
        # ADAM Epsilon Stability (related to ADAM optimizer)
        self.adam_epsilon_stability_coach = QDoubleSpinBox()
        self.adam_epsilon_stability_coach.setRange(1e-10, 1e-6)
        self.adam_epsilon_stability_coach.setSingleStep(1e-9)
        self.adam_epsilon_stability_coach.setDecimals(10)
        self.adam_epsilon_stability_coach.setValue(1e-8)
        self.adam_epsilon_stability_coach.setToolTip("ADAM optimizer epsilon stability parameter")
        cpu_layout.addRow("ADAM Epsilon Stability:", self.adam_epsilon_stability_coach)
        
        cpu_group.setLayout(cpu_layout)
        layout.addWidget(cpu_group)
        
        # Reset button
        reset_btn = QPushButton("Reset Coach Defaults")
        reset_btn.clicked.connect(self.reset_coach_params)
        layout.addWidget(reset_btn)
        
        # Info
        coach_info = QLabel(
            "💡 Evolution Coach Agent:\n"
            "- AI agent using tool-calling LLM (Gemma 3 via LM Studio)\n"
            "- Analyzes population state and diagnoses problems\n"
            "- Takes direct actions: mutate individuals, adjust parameters, inject diversity\n"
            "- Agent mode (agent01): Full autonomy with 27 tools available\n"
            "- Analyzes every N generations (e.g., gen 10, 20, 30...)\n"
            "- Typical session: 5-7 tool calls in 3-5 iterations (30-90 seconds)"
        )
        coach_info.setWordWrap(True)
        coach_info.setProperty("variant", "secondary")
        layout.addWidget(coach_info)
        
        layout.addStretch()
        return widget
    
    def reset_coach_params(self):
        """Reset Evolution Coach parameters to defaults."""
        self.coach_enabled.setChecked(True)
        self.coach_provider.setCurrentIndex(0)  # Local
        idx = self.coach_system_prompt.findData("agent01")
        if idx >= 0:
            self.coach_system_prompt.setCurrentIndex(idx)
        self.coach_interval.setValue(10)
        # No population window / max log generations
        self.coach_auto_reload.setChecked(True)
        self.coach_context_length.setValue(5000)
        self.coach_gpu.setValue(0.6)
        self.coach_response_timeout.setValue(3600)  # 1 hour default
        self.cpu_workers.setValue(7)
        self.adam_epsilon_stability_coach.setValue(1e-8)
        self._on_provider_changed(0)
    
    def _on_provider_changed(self, index):
        """Show/hide provider-specific settings based on selection."""
        provider = self.coach_provider.currentData()
        
        # Show/hide provider-specific groups
        self.lms_group.setVisible(provider == "local")
        self.openrouter_group.setVisible(provider == "openrouter")
        
        # No agent iteration control here
    
    def _refresh_lms_models(self):
        """Refresh available LM Studio models via 'lms ls' command."""
        try:
            import subprocess
            result = subprocess.run(
                ["lms", "ls"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse lms ls output
                lines = result.stdout.strip().split('\n')
                models = []
                
                for line in lines:
                    # Skip header lines
                    if 'LLM' in line or 'EMBEDDING' in line or 'PARAMS' in line:
                        continue
                    
                    # Extract model names (format: "model-name    SIZE    ARCH    ...")
                    parts = line.split()
                    if parts and len(parts) >= 2:
                        model_name = parts[0]
                        # Skip empty lines and separator lines
                        if model_name and not model_name.startswith('-'):
                            models.append(model_name)
                
                if models:
                    # Save current selection
                    current = self.lms_model.currentText()
                    
                    # Update dropdown
                    self.lms_model.clear()
                    for model in models:
                        self.lms_model.addItem(model, model)
                    
                    # Restore selection if it exists
                    idx = self.lms_model.findText(current)
                    if idx >= 0:
                        self.lms_model.setCurrentIndex(idx)
                    
                    QMessageBox.information(
                        self,
                        "Models Refreshed",
                        f"Found {len(models)} models in LM Studio:\n\n" + "\n".join(f"• {m}" for m in models[:10])
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "No Models Found",
                        "No models found in LM Studio.\n\n"
                        "Download models using:\n"
                        "lms download <model-name>\n\n"
                        "Example:\n"
                        "lms download google/gemma-3-12b"
                    )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Failed to list LM Studio models.\n\n"
                    "Make sure LM Studio CLI is installed:\n"
                    "curl -fsSL https://lmstudio.ai/lms-cli/install.sh | bash"
                )
        
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "LM Studio CLI Not Found",
                "'lms' command not found in PATH.\n\n"
                "Install LM Studio CLI:\n"
                "curl -fsSL https://lmstudio.ai/lms-cli/install.sh | bash\n\n"
                "Then restart terminal and try again."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to refresh models: {e}"
            )
    
    def load_from_settings(self):
        """Load UI values from saved settings."""
        # Backtest parameters
        self.fee_bp.setValue(settings.backtest_fee_bp)
        self.slippage_bp.setValue(settings.backtest_slippage_bp)
        
        # Optimizer parameters (common)
        self.optimizer_iterations.setValue(settings.optimizer_iterations)
        # Clamp workers to available range
        workers = getattr(settings, 'optimizer_workers', self.optimizer_workers.value())
        self.optimizer_workers.setValue(max(1, min(self.optimizer_workers.maximum(), workers)))
        
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

        # Logging
        # Log level
        lvl = str(getattr(settings, 'ada_agent_log_level', 'INFO')).upper()
        idx = self.log_level.findData(lvl)
        if idx >= 0:
            self.log_level.setCurrentIndex(idx)
        else:
            self.log_level.setCurrentText(lvl)
        # Progress bars and feature builds
        self.log_progress_bars.setChecked(bool(getattr(settings, 'log_progress_bars', True)))
        self.log_feature_builds.setChecked(bool(getattr(settings, 'log_feature_builds', False)))
        # Coach debug payloads
        self.coach_debug_payloads.setChecked(bool(getattr(settings, 'coach_debug_payloads', False)))
        
        # Evolution Coach settings
        # Enable/Provider
        self.coach_enabled.setChecked(getattr(settings, 'coach_enabled', True))
        
        provider = getattr(settings, 'coach_provider', 'local')
        idx = self.coach_provider.findData(provider)
        if idx >= 0:
            self.coach_provider.setCurrentIndex(idx)
        
        # LM Studio settings
        model = getattr(settings, 'coach_model', 'google/gemma-3-12b')
        idx = self.lms_model.findData(model)
        if idx >= 0:
            self.lms_model.setCurrentIndex(idx)
        else:
            # Model not in list, add it
            self.lms_model.addItem(model, model)
            self.lms_model.setCurrentIndex(self.lms_model.count() - 1)
        
        self.coach_context_length.setValue(int(getattr(settings, 'coach_context_length', 5000)))
        self.coach_gpu.setValue(float(getattr(settings, 'coach_gpu', 0.6)))
        self.coach_auto_reload.setChecked(bool(getattr(settings, 'coach_auto_reload_model', True)))
        
        # Openrouter settings
        self.openrouter_api_key.setText(getattr(settings, 'openrouter_api_key', ''))
        openrouter_model = getattr(settings, 'openrouter_model', 'anthropic/claude-3.5-sonnet')
        idx = self.openrouter_model.findData(openrouter_model)
        if idx >= 0:
            self.openrouter_model.setCurrentIndex(idx)
        else:
            self.openrouter_model.setCurrentText(openrouter_model)
        
        # Common coach settings
        if hasattr(settings, 'coach_system_prompt'):
            idx = self.coach_system_prompt.findData(settings.coach_system_prompt)
            if idx >= 0:
                self.coach_system_prompt.setCurrentIndex(idx)
        
        self.coach_interval.setValue(int(getattr(settings, 'coach_analysis_interval', 10)))
        self.coach_response_timeout.setValue(int(getattr(settings, 'coach_response_timeout', 3600)))
        
        # CPU Workers and ADAM epsilon stability
        self.cpu_workers.setValue(getattr(settings, 'cpu_workers', 7))
        self.adam_epsilon_stability_coach.setValue(getattr(settings, 'adam_epsilon_stability', 1e-8))
        
        # Trigger provider change to show/hide appropriate sections
        self._on_provider_changed(self.coach_provider.currentIndex())
    

    
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
                
                # Optimizer execution
                'optimizer_workers': self.optimizer_workers.value(),
                
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
                
                # Evolution Coach settings
                'coach_enabled': self.coach_enabled.isChecked(),
                'coach_provider': self.coach_provider.currentData(),
                'coach_model': self.lms_model.currentData() or self.lms_model.currentText(),
                'openrouter_api_key': self.openrouter_api_key.text(),
                'openrouter_model': self.openrouter_model.currentData() or self.openrouter_model.currentText(),
                'coach_system_prompt': self.coach_system_prompt.currentData(),
                'coach_analysis_interval': self.coach_interval.value(),
                'coach_auto_reload_model': self.coach_auto_reload.isChecked(),
                'coach_context_length': self.coach_context_length.value(),
                'coach_gpu': self.coach_gpu.value(),
                'coach_response_timeout': self.coach_response_timeout.value(),
                # Logging
                'ada_agent_log_level': self.log_level.currentData() or self.log_level.currentText(),
                'log_progress_bars': self.log_progress_bars.isChecked(),
                'log_feature_builds': self.log_feature_builds.isChecked(),
                'coach_debug_payloads': self.coach_debug_payloads.isChecked(),
                
                # CPU Workers
                'cpu_workers': self.cpu_workers.value(),
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
        self.adam_epsilon.setValue(0.02)
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
