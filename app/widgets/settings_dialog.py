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


# Timeframe mapping (multiplier, unit, label)
TIMEFRAMES = {
    "1m": (1, "minute", "1 minute"),
    "3m": (3, "minute", "3 minutes"),
    "5m": (5, "minute", "5 minutes"),
    "10m": (10, "minute", "10 minutes"),
    "15m": (15, "minute", "15 minutes"),
    "30m": (30, "minute", "30 minutes"),
    "1h": (60, "minute", "1 hour"),
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
        tf_layout.addRow("Timeframe:", self.timeframe_combo)
        
        info_label = QLabel(
            "💡 Tip: Shorter timeframes (1m-5m) generate more signals but may be noisier.\n"
            "Recommended: Start with 15m or 1h for balanced signal quality."
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
        indicators_layout.addWidget(self.show_signals)
        
        self.show_entries = QCheckBox("Buy Orders (Green Arrows ↑)")
        self.show_entries.setChecked(True)
        indicators_layout.addWidget(self.show_entries)
        
        self.show_exits = QCheckBox("Sell Orders (Red/Green Arrows ↓)")
        self.show_exits.setChecked(True)
        indicators_layout.addWidget(self.show_exits)
        
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
        self.show_exits.setChecked(settings.chart_exits)
    
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
                
                # Chart indicators
                'chart_ema_fast': self.show_ema_fast.isChecked(),
                'chart_ema_slow': self.show_ema_slow.isChecked(),
                'chart_sma': self.show_sma.isChecked(),
                'chart_rsi': self.show_rsi.isChecked(),
                'chart_macd': self.show_macd.isChecked(),
                'chart_volume': self.show_volume.isChecked(),
                'chart_signals': self.show_signals.isChecked(),
                'chart_entries': self.show_entries.isChecked(),
                'chart_exits': self.show_exits.isChecked(),
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
            
            # Update UI
            self.download_btn.setEnabled(False)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            
            # Create provider
            if not self.dp:
                self.dp = DataProvider()

            estimate = self.dp.estimate_download(from_date, to_date, mult, unit)

            def format_bars(current: int, total: int | None) -> str:
                if total and total > 0:
                    return f"{current:,}/{total:,}"
                return f"{current:,}"

            self.progress_bar.setRange(0, max(estimate.pages, 1))
            self.progress_bar.setValue(0)
            est_time_text = self.format_duration(estimate.seconds_total)
            self.progress_label.setText(
                "Free tier: 5 API calls/min.\n"
                f"Downloading ADA data ({label}) {from_date} → {to_date}.\n"
                f"Estimated {estimate.pages} request(s) (~{est_time_text})."
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
            
            # Fetch data
            df = await self.dp.fetch_bars(
                "X:ADAUSD",
                from_date,
                to_date,
                mult,
                unit,
                progress_callback=on_progress,
                estimate=estimate,
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
        from strategy.seller_exhaustion import SellerParams
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
        from backtest.engine import BacktestParams
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
            'exits': self.show_exits.isChecked()
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.dp:
            await self.dp.close()
