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
        
        # General Tab (API keys, basic settings)
        tabs.addTab(self.create_general_tab(), "General")
        
        # Agent Tab (Evolution Coach + OpenRouter/OpenAI selection)
        tabs.addTab(self.create_agent_tab(), "Agent")
        
        # Optimization Parameters Tab
        tabs.addTab(self.create_optimizer_tab(), "Optimization")

        # Logging Tab
        tabs.addTab(self.create_logging_tab(), "Logging")
        
        # Tools Tab
        tabs.addTab(self.create_tools_tab(), "Tools")
        
        # Chart Indicators Tab (moved to end)
        tabs.addTab(self.create_indicators_tab(), "Chart Indicators")
        
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
    

    def create_general_tab(self):
        """Create general settings tab with API keys and basic configuration."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # API Configuration Group
        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout()
        
        # Polygon API Key
        polygon_row = QHBoxLayout()
        self.polygon_api_key = QLineEdit()
        self.polygon_api_key.setEchoMode(QLineEdit.Password)
        self.polygon_api_key.setPlaceholderText("Enter your Polygon.io API key")
        self.polygon_api_key.setToolTip(
            "Polygon.io API key for market data\n"
            "Get one at: https://polygon.io/dashboard"
        )
        polygon_row.addWidget(self.polygon_api_key)
        
        # Show/hide API key button
        self.show_polygon_key_btn = QPushButton("👁️")
        self.show_polygon_key_btn.setMaximumWidth(40)
        self.show_polygon_key_btn.setCheckable(True)
        self.show_polygon_key_btn.setToolTip("Show/hide API key")
        self.show_polygon_key_btn.clicked.connect(lambda checked: 
            self.polygon_api_key.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        polygon_row.addWidget(self.show_polygon_key_btn)
        
        api_layout.addRow("Polygon API Key:", polygon_row)
        
        # Data Directory
        self.data_dir = QLineEdit()
        self.data_dir.setPlaceholderText(".data")
        self.data_dir.setToolTip("Directory for cached market data")
        api_layout.addRow("Data Directory:", self.data_dir)
        
        # Timezone
        self.timezone = QComboBox()
        self.timezone.addItems(["UTC", "EST", "PST", "CET"])
        self.timezone.setToolTip("Timezone for data processing")
        api_layout.addRow("Timezone:", self.timezone)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Data Download Group
        data_group = QGroupBox("Data Download")
        data_layout = QFormLayout()
        
        # Ticker
        self.ticker = QLineEdit()
        self.ticker.setPlaceholderText("X:ADAUSD")
        self.ticker.setToolTip("Trading pair symbol")
        data_layout.addRow("Ticker:", self.ticker)
        
        # Date Range
        date_row = QHBoxLayout()
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_from.setCalendarPopup(True)
        self.date_from.setToolTip("Start date for data download")
        date_row.addWidget(self.date_from)
        
        date_row.addWidget(QLabel("to"))
        
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.setToolTip("End date for data download")
        date_row.addWidget(self.date_to)
        
        data_layout.addRow("Date Range:", date_row)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Timeframe Configuration Group
        timeframe_group = QGroupBox("Timeframe Configuration")
        timeframe_layout = QFormLayout()
        
        # Timeframe
        self.timeframe = QComboBox()
        for key, (mult, unit, label) in TIMEFRAMES.items():
            self.timeframe.addItem(label, key)
        self.timeframe.setCurrentText("15 minutes")
        self.timeframe.setToolTip("Default timeframe for analysis")
        timeframe_layout.addRow("Timeframe:", self.timeframe)
        
        timeframe_group.setLayout(timeframe_layout)
        layout.addWidget(timeframe_group)
        
        # Info
        info_label = QLabel(
            "💡 <b>General Settings:</b><br>"
            "• Configure API keys for data access<br>"
            "• Set default data download parameters<br>"
            "• Choose default timeframe for analysis"
        )
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText)
        info_label.setStyleSheet("color: #888; font-size: 11px; padding: 8px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        return widget

    def create_agent_tab(self):
        """Create agent configuration tab with Evolution Coach and LLM provider settings."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Evolution Coach Group
        coach_group = QGroupBox("Evolution Coach")
        coach_layout = QFormLayout()
        
        # Enable checkbox
        self.coach_enabled = QCheckBox("Enable Evolution Coach Agent")
        self.coach_enabled.setChecked(True)
        self.coach_enabled.setToolTip("Enable/disable the Evolution Coach agent during optimization")
        coach_layout.addRow("Enabled:", self.coach_enabled)
        
        # Islands Management checkbox
        self.coach_islands_enabled = QCheckBox("Enable Islands Management")
        self.coach_islands_enabled.setChecked(False)
        self.coach_islands_enabled.setToolTip(
            "Enable/disable Islands Management (adds complexity)\n"
            "Islands create sub-populations with migration between them.\n"
            "Can complicate agent decision-making and slow optimization."
        )
        coach_layout.addRow("Islands:", self.coach_islands_enabled)
        
        # Analysis Interval
        self.coach_interval = QSpinBox()
        self.coach_interval.setRange(1, 100)
        self.coach_interval.setValue(10)
        self.coach_interval.setSuffix(" gens")
        self.coach_interval.setToolTip("Coach analyzes every N generations (10, 20, 30, etc)")
        coach_layout.addRow("Analysis Interval:", self.coach_interval)
        
        # Response Timeout
        self.coach_response_timeout = QSpinBox()
        self.coach_response_timeout.setRange(30, 36000)  # 30 seconds to 10 hours
        self.coach_response_timeout.setValue(3600)  # Default 1 hour
        self.coach_response_timeout.setSingleStep(60)  # Step by 60 seconds
        self.coach_response_timeout.setSuffix(" sec")
        self.coach_response_timeout.setToolTip(
            "LLM response timeout in seconds (default: 3600 = 1 hour)\n"
            "Increase if LLM is slow or model is large"
        )
        coach_layout.addRow("Response Timeout:", self.coach_response_timeout)
        
        coach_group.setLayout(coach_layout)
        layout.addWidget(coach_group)
        
        # Provider Selection Group
        provider_group = QGroupBox("LLM Provider")
        provider_layout = QFormLayout()
        
        # Provider selection
        self.agent_provider = QComboBox()
        self.agent_provider.addItem("🚀 Novita (Recommended)", "novita")
        self.agent_provider.addItem("🌐 OpenRouter", "openrouter")
        self.agent_provider.addItem("🤖 OpenAI Direct", "openai")
        self.agent_provider.setCurrentIndex(0)
        self.agent_provider.setToolTip(
            "Choose LLM provider:\n"
            "• Novita: Fast, affordable models (DeepSeek, etc.)\n"
            "• OpenRouter: Access to multiple models (Claude, GPT-4, etc.)\n"
            "• OpenAI: Direct access to OpenAI models"
        )
        self.agent_provider.currentIndexChanged.connect(self._on_agent_provider_changed)
        provider_layout.addRow("Provider:", self.agent_provider)
        
        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)
        
        # OpenRouter Configuration
        self.openrouter_group = QGroupBox("OpenRouter Configuration")
        openrouter_layout = QFormLayout()
        
        # API Key input
        openrouter_key_row = QHBoxLayout()
        self.openrouter_api_key = QLineEdit()
        self.openrouter_api_key.setEchoMode(QLineEdit.Password)
        self.openrouter_api_key.setPlaceholderText("sk-or-v1-...")
        self.openrouter_api_key.setToolTip(
            "OpenRouter API key\n"
            "Get one at: https://openrouter.ai/keys"
        )
        openrouter_key_row.addWidget(self.openrouter_api_key)
        
        # Show/hide API key button
        self.show_openrouter_key_btn = QPushButton("👁️")
        self.show_openrouter_key_btn.setMaximumWidth(40)
        self.show_openrouter_key_btn.setCheckable(True)
        self.show_openrouter_key_btn.setToolTip("Show/hide API key")
        self.show_openrouter_key_btn.clicked.connect(lambda checked: 
            self.openrouter_api_key.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        openrouter_key_row.addWidget(self.show_openrouter_key_btn)
        
        openrouter_layout.addRow("API Key:", openrouter_key_row)
        
        # Model selection
        self.openrouter_model = QComboBox()
        self.openrouter_model.setEditable(True)
        self.openrouter_model.addItem("anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-sonnet")
        self.openrouter_model.addItem("openai/gpt-4o", "openai/gpt-4o")
        self.openrouter_model.addItem("anthropic/claude-3-opus", "anthropic/claude-3-opus")
        self.openrouter_model.addItem("google/gemini-pro-1.5", "google/gemini-pro-1.5")
        self.openrouter_model.setToolTip(
            "OpenRouter model identifier\n"
            "Browse models at: https://openrouter.ai/models"
        )
        openrouter_layout.addRow("Model:", self.openrouter_model)
        
        # Base URL
        self.openrouter_base_url = QLineEdit()
        self.openrouter_base_url.setPlaceholderText("https://openrouter.ai/api/v1")
        self.openrouter_base_url.setToolTip("OpenRouter API base URL")
        openrouter_layout.addRow("Base URL:", self.openrouter_base_url)
        
        self.openrouter_group.setLayout(openrouter_layout)
        layout.addWidget(self.openrouter_group)
        
        # Novita Configuration
        self.novita_group = QGroupBox("Novita Configuration")
        novita_layout = QFormLayout()
        
        # API Key input
        novita_key_row = QHBoxLayout()
        self.novita_api_key = QLineEdit()
        self.novita_api_key.setEchoMode(QLineEdit.Password)
        self.novita_api_key.setPlaceholderText("nv-...")
        self.novita_api_key.setToolTip(
            "Novita API key\n"
            "Get one at: https://novita.ai/"
        )
        novita_key_row.addWidget(self.novita_api_key)
        
        # Show/hide API key button
        self.show_novita_key_btn = QPushButton("👁️")
        self.show_novita_key_btn.setMaximumWidth(40)
        self.show_novita_key_btn.setCheckable(True)
        self.show_novita_key_btn.setToolTip("Show/hide API key")
        self.show_novita_key_btn.clicked.connect(lambda checked:
            self.novita_api_key.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        novita_key_row.addWidget(self.show_novita_key_btn)
        
        novita_layout.addRow("API Key:", novita_key_row)
        
        # Model selection
        self.novita_model = QComboBox()
        self.novita_model.setEditable(True)
        self.novita_model.addItem("deepseek/deepseek-r1", "deepseek/deepseek-r1")
        self.novita_model.addItem("deepseek/deepseek-coder", "deepseek/deepseek-coder")
        self.novita_model.addItem("qwen/qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct")
        self.novita_model.addItem("meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-70b-instruct")
        self.novita_model.setToolTip(
            "Novita model identifier\n"
            "Browse models at: https://novita.ai/models"
        )
        novita_layout.addRow("Model:", self.novita_model)
        
        # Base URL
        self.novita_base_url = QLineEdit()
        self.novita_base_url.setPlaceholderText("https://api.novita.ai/openai")
        self.novita_base_url.setToolTip("Novita API base URL")
        novita_layout.addRow("Base URL:", self.novita_base_url)
        
        self.novita_group.setLayout(novita_layout)
        layout.addWidget(self.novita_group)
        
        # OpenAI Configuration
        self.openai_group = QGroupBox("OpenAI Configuration")
        openai_layout = QFormLayout()
        
        # API Key input
        openai_key_row = QHBoxLayout()
        self.openai_api_key = QLineEdit()
        self.openai_api_key.setEchoMode(QLineEdit.Password)
        self.openai_api_key.setPlaceholderText("sk-proj-...")
        self.openai_api_key.setToolTip(
            "OpenAI API key\n"
            "Get one at: https://platform.openai.com/api-keys"
        )
        openai_key_row.addWidget(self.openai_api_key)
        
        # Show/hide API key button
        self.show_openai_key_btn = QPushButton("👁️")
        self.show_openai_key_btn.setMaximumWidth(40)
        self.show_openai_key_btn.setCheckable(True)
        self.show_openai_key_btn.setToolTip("Show/hide API key")
        self.show_openai_key_btn.clicked.connect(lambda checked: 
            self.openai_api_key.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        openai_key_row.addWidget(self.show_openai_key_btn)
        
        openai_layout.addRow("API Key:", openai_key_row)
        
        # Model selection
        self.openai_model = QComboBox()
        self.openai_model.setEditable(True)
        self.openai_model.addItem("gpt-4o", "gpt-4o")
        self.openai_model.addItem("gpt-4o-mini", "gpt-4o-mini")
        self.openai_model.addItem("gpt-4-turbo", "gpt-4-turbo")
        self.openai_model.addItem("gpt-3.5-turbo", "gpt-3.5-turbo")
        self.openai_model.setToolTip(
            "OpenAI model identifier\n"
            "Browse models at: https://platform.openai.com/docs/models"
        )
        openai_layout.addRow("Model:", self.openai_model)
        
        # Base URL
        self.openai_base_url = QLineEdit()
        self.openai_base_url.setPlaceholderText("https://api.openai.com/v1")
        self.openai_base_url.setToolTip("OpenAI API base URL")
        openai_layout.addRow("Base URL:", self.openai_base_url)
        
        self.openai_group.setLayout(openai_layout)
        layout.addWidget(self.openai_group)
        
        # Info
        info_label = QLabel(
            "💡 <b>Evolution Coach Agent:</b><br>"
            "• <b>Evolution Coach</b>: AI agent that analyzes and improves optimization<br>"
            "• <b>Islands Management</b>: Advanced feature (can add complexity)<br>"
            "• <b>Novita</b>: Fast, affordable models (DeepSeek, Qwen, Llama)<br>"
            "• <b>OpenRouter</b>: Access to multiple models (Claude, GPT-4, etc.)<br>"
            "• <b>OpenAI</b>: Direct access to OpenAI models<br>"
            "• All settings are stored securely in .env file"
        )
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText)
        info_label.setStyleSheet("color: #888; font-size: 11px; padding: 8px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        return widget

    def _on_agent_provider_changed(self, index):
        """Show/hide provider-specific settings based on selection."""
        provider = self.agent_provider.currentData()
        
        # Show/hide provider-specific groups
        self.novita_group.setVisible(provider == "novita")
        self.openrouter_group.setVisible(provider == "openrouter")
        self.openai_group.setVisible(provider == "openai")

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
    
        
    def create_tools_tab(self):
        """Create tab for enabling/disabling coach tools."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Coach Tools")
        group_layout = QVBoxLayout()
        
        group_layout.addWidget(QLabel("Enable/disable individual tools for the Evolution Coach:"))
        
        # List of tools from the provided list
        tool_definitions = [
            ('analyze_population', 'Analyze current population state and provide detailed statistics'),
            ('get_correlation_matrix', 'Get correlation matrix between parameters in the population'),
            ('get_param_distribution', 'Get parameter distribution statistics'),
            ('get_param_bounds', 'Get current parameter bounds for optimization'),
            ('get_generation_history', 'Get fitness evolution history across generations'),
            ('mutate_individual', 'Apply mutations to a specific individual'),
            ('insert_llm_individual', 'Insert a new individual created by LLM'),
            ('create_islands', 'Create island populations for parallel evolution'),
            ('migrate_between_islands', 'Migrate individuals between island populations'),
            ('configure_island_scheduler', 'Configure island migration scheduler'),
            ('inject_immigrants', 'Inject new individuals into population'),
            ('export_population', 'Export current population to file'),
            ('import_population', 'Import population from file'),
            ('drop_individual', 'Remove worst individual from population'),
            ('bulk_update_param', 'Update parameter for multiple individuals'),
            ('update_param_bounds', 'Update bounds for a specific parameter'),
            ('update_bounds_multi', 'Update bounds for multiple parameters'),
            ('reseed_population', 'Reseed population with new random individuals'),
            ('insert_individual', 'Insert a new individual into population'),
            ('update_fitness_gates', 'Update fitness gate thresholds'),
            ('update_ga_params', 'Update genetic algorithm parameters'),
            ('update_fitness_weights', 'Update fitness function weights'),
            ('set_fitness_function_type', 'Set fitness function type'),
            ('configure_curriculum', 'Configure curriculum learning parameters'),
            ('set_fitness_preset', 'Set fitness function preset'),
            ('set_exit_policy', 'Set exit policy for optimization'),
            ('set_costs', 'Set transaction costs for backtesting'),
            ('finish_analysis', 'Finish analysis and return final recommendations')
        ]
        
        for tool_name, description in tool_definitions:
            checkbox = QCheckBox(description)
            checkbox.setChecked(True)
            setattr(self, f'coach_tool_{tool_name}', checkbox)
            group_layout.addWidget(checkbox)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        return widget
    
    
    def load_from_settings(self):
        """Load UI values from saved settings."""
        # General settings
        self.polygon_api_key.setText(getattr(settings, 'polygon_api_key', ''))
        self.data_dir.setText(getattr(settings, 'data_dir', '.data'))
        
        # Timezone
        tz = getattr(settings, 'tz', 'UTC')
        idx = self.timezone.findText(tz)
        if idx >= 0:
            self.timezone.setCurrentIndex(idx)
        
        # Ticker and date range
        self.ticker.setText(getattr(settings, 'last_ticker', 'X:ADAUSD'))
        
        # Date range
        from_date = getattr(settings, 'last_date_from', '2024-01-01')
        to_date = getattr(settings, 'last_date_to', '2024-12-31')
        self.date_from.setDate(QDate.fromString(from_date, "yyyy-MM-dd"))
        self.date_to.setDate(QDate.fromString(to_date, "yyyy-MM-dd"))
        
        # Timeframe
        tf = getattr(settings, 'timeframe', '15m')
        idx = self.timeframe.findData(tf)
        if idx >= 0:
            self.timeframe.setCurrentIndex(idx)
        
        # Agent settings (Evolution Coach + LLM Provider)
        self.coach_enabled.setChecked(getattr(settings, 'coach_enabled', True))
        self.coach_islands_enabled.setChecked(getattr(settings, 'coach_islands_enabled', False))
        
        # Provider selection
        provider = getattr(settings, 'agent_provider', 'novita')
        idx = self.agent_provider.findData(provider)
        if idx >= 0:
            self.agent_provider.setCurrentIndex(idx)
        
        # Novita settings
        self.novita_api_key.setText(getattr(settings, 'novita_api_key', ''))
        self.novita_base_url.setText(getattr(settings, 'novita_base_url', 'https://api.novita.ai/openai'))
        novita_model = getattr(settings, 'novita_model', 'deepseek/deepseek-r1')
        idx = self.novita_model.findData(novita_model)
        if idx >= 0:
            self.novita_model.setCurrentIndex(idx)
        else:
            self.novita_model.setCurrentText(novita_model)
        
        # OpenRouter settings
        self.openrouter_api_key.setText(getattr(settings, 'openrouter_api_key', ''))
        self.openrouter_base_url.setText(getattr(settings, 'openrouter_base_url', 'https://openrouter.ai/api/v1'))
        openrouter_model = getattr(settings, 'openrouter_model', 'anthropic/claude-3.5-sonnet')
        idx = self.openrouter_model.findData(openrouter_model)
        if idx >= 0:
            self.openrouter_model.setCurrentIndex(idx)
        else:
            self.openrouter_model.setCurrentText(openrouter_model)
        
        # OpenAI settings
        self.openai_api_key.setText(getattr(settings, 'openai_api_key', ''))
        self.openai_base_url.setText(getattr(settings, 'openai_base_url', 'https://api.openai.com/v1'))
        openai_model = getattr(settings, 'openai_model', 'gpt-4o')
        idx = self.openai_model.findData(openai_model)
        if idx >= 0:
            self.openai_model.setCurrentIndex(idx)
        else:
            self.openai_model.setCurrentText(openai_model)
        
        # Coach settings
        self.coach_interval.setValue(getattr(settings, 'coach_analysis_interval', 10))
        self.coach_response_timeout.setValue(getattr(settings, 'coach_response_timeout', 3600))
        
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
        # Coach Tools
        tool_definitions = [
            ('analyze_population', 'Analyze current population state and provide detailed statistics'),
            ('get_correlation_matrix', 'Get correlation matrix between parameters in the population'),
            ('get_param_distribution', 'Get parameter distribution statistics'),
            ('get_param_bounds', 'Get current parameter bounds for optimization'),
            ('get_generation_history', 'Get fitness evolution history across generations'),
            ('mutate_individual', 'Apply mutations to a specific individual'),
            ('insert_llm_individual', 'Insert a new individual created by LLM'),
            ('create_islands', 'Create island populations for parallel evolution'),
            ('migrate_between_islands', 'Migrate individuals between island populations'),
            ('configure_island_scheduler', 'Configure island migration scheduler'),
            ('inject_immigrants', 'Inject new individuals into population'),
            ('export_population', 'Export current population to file'),
            ('import_population', 'Import population from file'),
            ('drop_individual', 'Remove worst individual from population'),
            ('bulk_update_param', 'Update parameter for multiple individuals'),
            ('update_param_bounds', 'Update bounds for a specific parameter'),
            ('update_bounds_multi', 'Update bounds for multiple parameters'),
            ('reseed_population', 'Reseed population with new random individuals'),
            ('insert_individual', 'Insert a new individual into population'),
            ('update_fitness_gates', 'Update fitness gate thresholds'),
            ('update_ga_params', 'Update genetic algorithm parameters'),
            ('update_fitness_weights', 'Update fitness function weights'),
            ('set_fitness_function_type', 'Set fitness function type'),
            ('configure_curriculum', 'Configure curriculum learning parameters'),
            ('set_fitness_preset', 'Set fitness function preset'),
            ('set_exit_policy', 'Set exit policy for optimization'),
            ('set_costs', 'Set transaction costs for backtesting'),
            ('finish_analysis', 'Finish analysis and return final recommendations')
        ]
        
        for tool_name, _ in tool_definitions:
            checkbox = getattr(self, f'coach_tool_{tool_name}', None)
            if checkbox:
                checkbox.setChecked(getattr(settings, f'coach_tool_{tool_name}', True))
        if idx >= 0:
            self.log_level.setCurrentIndex(idx)
        else:
            self.log_level.setCurrentText(lvl)
        # Progress bars and feature builds
        self.log_progress_bars.setChecked(bool(getattr(settings, 'log_progress_bars', True)))
        self.log_feature_builds.setChecked(bool(getattr(settings, 'log_feature_builds', False)))
        # Coach debug payloads
        self.coach_debug_payloads.setChecked(bool(getattr(settings, 'coach_debug_payloads', False)))
        
    

    
    def save_settings(self):
        """Save all settings to .env file."""
        try:
            settings_dict = {
                # Backtest parameters are now managed in main window
                
                
                # Optimizer (Common)
                'optimizer_iterations': self._safe_get_spin_value('optimizer_iterations', 1),
                
                # Genetic algorithm
                'ga_population_size': self._safe_get_spin_value('ga_population', 24),
                'ga_mutation_rate': self._safe_get_spin_value('ga_mutation_rate', 0.3),
                'ga_sigma': self._safe_get_spin_value('ga_sigma', 0.1),
                'ga_elite_fraction': self._safe_get_spin_value('ga_elite_fraction', 0.1),
                'ga_tournament_size': self._safe_get_spin_value('ga_tournament_size', 3),
                'ga_mutation_probability': self._safe_get_spin_value('ga_mutation_probability', 0.9),
                
                # ADAM Optimizer
                'adam_learning_rate': self._safe_get_spin_value('adam_learning_rate', 0.001),
                'adam_epsilon': self._safe_get_spin_value('adam_epsilon', 0.02),
                'adam_max_grad_norm': self._safe_get_spin_value('adam_max_grad_norm', 1.0),
                'adam_beta1': self._safe_get_spin_value('adam_beta1', 0.9),
                'adam_beta2': self._safe_get_spin_value('adam_beta2', 0.999),
                'adam_epsilon_stability': self._safe_get_spin_value('adam_epsilon_stability', 1e-8),
                
                # Optimizer execution
                'optimizer_workers': self._safe_get_spin_value('optimizer_workers', 1),
                
                # Chart indicators
                'chart_ema_fast': self._safe_get_checkbox_value('show_ema_fast', True),
                'chart_ema_slow': self._safe_get_checkbox_value('show_ema_slow', True),
                'chart_sma': self._safe_get_checkbox_value('show_sma', False),
                'chart_rsi': self._safe_get_checkbox_value('show_rsi', False),
                'chart_macd': self._safe_get_checkbox_value('show_macd', False),
                'chart_volume': self._safe_get_checkbox_value('show_volume', False),
                'chart_signals': self._safe_get_checkbox_value('show_signals', True),
                'chart_entries': self._safe_get_checkbox_value('show_entries', True),
                'chart_exits': True,  # Always true for backwards compatibility (not shown in UI)
                
                # General settings
                'polygon_api_key': self._safe_get_text('polygon_api_key', ''),
                'data_dir': self._safe_get_text('data_dir', '.data'),
                'tz': self._safe_get_combo_data('timezone', 'UTC'),
                'last_ticker': self._safe_get_text('ticker', 'X:ADAUSD'),
                'last_date_from': self._safe_get_date('date_from', '2024-01-01'),
                'last_date_to': self._safe_get_date('date_to', '2024-12-31'),
                'timeframe': self._safe_get_combo_data('timeframe', '15m'),
                
                # Agent settings (Evolution Coach + LLM Provider)
                'coach_enabled': self._safe_get_checkbox_value('coach_enabled', True),
                'coach_islands_enabled': self._safe_get_checkbox_value('coach_islands_enabled', False),
                'agent_provider': self._safe_get_combo_data('agent_provider', 'novita'),
                
                # Novita settings
                'novita_api_key': self._safe_get_text('novita_api_key', ''),
                'novita_model': self._safe_get_combo_data('novita_model', 'deepseek/deepseek-r1'),
                'novita_base_url': self._safe_get_text('novita_base_url', 'https://api.novita.ai/openai'),
                
                # OpenRouter settings
                'openrouter_api_key': self._safe_get_text('openrouter_api_key', ''),
                'openrouter_model': self._safe_get_combo_data('openrouter_model', 'anthropic/claude-3.5-sonnet'),
                'openrouter_base_url': self._safe_get_text('openrouter_base_url', 'https://openrouter.ai/api/v1'),
                
                # OpenAI settings
                'openai_api_key': self._safe_get_text('openai_api_key', ''),
                'openai_model': self._safe_get_combo_data('openai_model', 'gpt-4o'),
                'openai_base_url': self._safe_get_text('openai_base_url', 'https://api.openai.com/v1'),
                
                'coach_analysis_interval': self._safe_get_spin_value('coach_interval', 10),
                'coach_response_timeout': self._safe_get_spin_value('coach_response_timeout', 3600),
                # Logging
                'ada_agent_log_level': self._safe_get_combo_data('log_level', 'INFO'),
                'log_progress_bars': self._safe_get_checkbox_value('log_progress_bars', True),
                'log_feature_builds': self._safe_get_checkbox_value('log_feature_builds', False),
                'coach_debug_payloads': self._safe_get_checkbox_value('coach_debug_payloads', False),
                
                # CPU Workers (moved to optimizer_workers)
                
                # Coach Tools
                'coach_tool_analyze_population': self._safe_get_checkbox_value('coach_tool_analyze_population', True),
                'coach_tool_get_correlation_matrix': self._safe_get_checkbox_value('coach_tool_get_correlation_matrix', True),
                'coach_tool_get_param_distribution': self._safe_get_checkbox_value('coach_tool_get_param_distribution', True),
                'coach_tool_get_param_bounds': self._safe_get_checkbox_value('coach_tool_get_param_bounds', True),
                'coach_tool_get_generation_history': self._safe_get_checkbox_value('coach_tool_get_generation_history', True),
                'coach_tool_mutate_individual': self._safe_get_checkbox_value('coach_tool_mutate_individual', True),
                'coach_tool_insert_llm_individual': self._safe_get_checkbox_value('coach_tool_insert_llm_individual', True),
                'coach_tool_create_islands': self._safe_get_checkbox_value('coach_tool_create_islands', True),
                'coach_tool_migrate_between_islands': self._safe_get_checkbox_value('coach_tool_migrate_between_islands', True),
                'coach_tool_configure_island_scheduler': self._safe_get_checkbox_value('coach_tool_configure_island_scheduler', True),
                'coach_tool_inject_immigrants': self._safe_get_checkbox_value('coach_tool_inject_immigrants', True),
                'coach_tool_export_population': self._safe_get_checkbox_value('coach_tool_export_population', True),
                'coach_tool_import_population': self._safe_get_checkbox_value('coach_tool_import_population', True),
                'coach_tool_drop_individual': self._safe_get_checkbox_value('coach_tool_drop_individual', True),
                'coach_tool_bulk_update_param': self._safe_get_checkbox_value('coach_tool_bulk_update_param', True),
                'coach_tool_update_param_bounds': self._safe_get_checkbox_value('coach_tool_update_param_bounds', True),
                'coach_tool_update_bounds_multi': self._safe_get_checkbox_value('coach_tool_update_bounds_multi', True),
                'coach_tool_reseed_population': self._safe_get_checkbox_value('coach_tool_reseed_population', True),
                'coach_tool_insert_individual': self._safe_get_checkbox_value('coach_tool_insert_individual', True),
                'coach_tool_update_fitness_gates': self._safe_get_checkbox_value('coach_tool_update_fitness_gates', True),
                'coach_tool_update_ga_params': self._safe_get_checkbox_value('coach_tool_update_ga_params', True),
                'coach_tool_update_fitness_weights': self._safe_get_checkbox_value('coach_tool_update_fitness_weights', True),
                'coach_tool_set_fitness_function_type': self._safe_get_checkbox_value('coach_tool_set_fitness_function_type', True),
                'coach_tool_configure_curriculum': self._safe_get_checkbox_value('coach_tool_configure_curriculum', True),
                'coach_tool_set_fitness_preset': self._safe_get_checkbox_value('coach_tool_set_fitness_preset', True),
                'coach_tool_set_exit_policy': self._safe_get_checkbox_value('coach_tool_set_exit_policy', True),
                'coach_tool_set_costs': self._safe_get_checkbox_value('coach_tool_set_costs', True),
                'coach_tool_finish_analysis': self._safe_get_checkbox_value('coach_tool_finish_analysis', True),
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
        # Backtest parameters are now managed in main window
        pass
    
    def set_backtest_params(self, params: BacktestParams):
        """Update backtest tab controls from BacktestParams."""
        # Backtest parameters are now managed in main window
        pass

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
        # Backtest parameters are now managed in main window
        return BacktestParams()  # Return default values
    
    def get_indicator_config(self):
        """Get indicator display configuration."""
        return {
            'ema_fast': self._safe_get_checkbox_value('show_ema_fast', True),
            'ema_slow': self._safe_get_checkbox_value('show_ema_slow', True),
            'sma': self._safe_get_checkbox_value('show_sma', False),
            'rsi': self._safe_get_checkbox_value('show_rsi', False),
            'macd': self._safe_get_checkbox_value('show_macd', False),
            'volume': self._safe_get_checkbox_value('show_volume', False),
            'signals': self._safe_get_checkbox_value('show_signals', True),
            'entries': self._safe_get_checkbox_value('show_entries', True),
        }
    
    def _safe_get_checkbox_value(self, widget_name: str, default_value: bool = False) -> bool:
        """Safely get checkbox value, return default if widget is deleted."""
        try:
            widget = getattr(self, widget_name)
            return widget.isChecked()
        except RuntimeError:
            return default_value
    
    def _safe_get_combo_data(self, widget_name: str, default_value: str = "") -> str:
        """Safely get combo box data, return default if widget is deleted."""
        try:
            widget = getattr(self, widget_name)
            return widget.currentData() or widget.currentText() or default_value
        except RuntimeError:
            return default_value
    
    def _safe_get_text(self, widget_name: str, default_value: str = "") -> str:
        """Safely get text from widget, return default if widget is deleted."""
        try:
            widget = getattr(self, widget_name)
            return widget.text()
        except RuntimeError:
            return default_value
    
    def _safe_get_spin_value(self, widget_name: str, default_value: float = 0.0) -> float:
        """Safely get spin box value, return default if widget is deleted."""
        try:
            widget = getattr(self, widget_name)
            return widget.value()
        except RuntimeError:
            return default_value
    
    def _safe_get_date(self, widget_name: str, default_value: str = "2024-01-01") -> str:
        """Safely get date from QDateEdit, return default if widget is deleted."""
        try:
            widget = getattr(self, widget_name)
            return widget.date().toString("yyyy-MM-dd")
        except RuntimeError:
            return default_value
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.dp:
            await self.dp.close()
