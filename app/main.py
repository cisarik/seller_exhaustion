import sys
import os
import asyncio
import multiprocessing
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QSplitter, QProgressBar,
    QMessageBox, QWidget, QVBoxLayout
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction

try:
    import qasync
    HAS_QASYNC = True
except ImportError:
    HAS_QASYNC = False

from core.logging_utils import configure_logging, get_logger
from app.theme import DARK_FOREST_QSS
from app.widgets.candle_view import CandleChartWidget
from app.widgets.settings_dialog import SettingsDialog
from app.widgets.stats_panel import StatsPanel
from app.widgets.strategy_editor import StrategyEditor
from app.widgets.compact_params import CompactParamsEditor
from app.widgets.data_bar import DataBar
# Removed: EvolutionCoachWindow (now using console logging)
from strategy.seller_exhaustion import build_features, SellerParams
from backtest.engine import BacktestParams
from core.models import Timeframe
from backtest.engine import run_backtest
from data.provider import DataProvider
from data.cache import DataCache
from config.settings import settings, SettingsManager
from app.session_store import (
    save_session_snapshot,
    load_session_snapshot,
    clear_session_snapshot,
)
from core.strategy_export import (
    create_default_config,
    export_trading_config,
    import_trading_config,
    validate_config_for_live_trading,
)

TIMEFRAME_META = {
    Timeframe.m1: (1, "minute"),
    Timeframe.m3: (3, "minute"),
    Timeframe.m5: (5, "minute"),
    Timeframe.m10: (10, "minute"),
    Timeframe.m15: (15, "minute"),
    Timeframe.m30: (30, "minute"),
    Timeframe.m60: (60, "minute"),
}


logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """Main application window with toolbar, chart, and stats panel."""
    
    def __init__(self, ga_init_from: str | None = None):
        super().__init__()
        self.setWindowTitle("Seller-Exhaustion Entry - Fibonacci Exit Trading Strategy Optimizer")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Data and settings
        self.current_data = None  # Features dataframe (with indicators and signals)
        self.raw_data = None  # Raw OHLCV data (for rebuilding features)
        self.current_tf = Timeframe.m15
        self.settings_dialog = None
        self.strategy_editor = None
        # Removed: evolution_coach_window (now using console logging)
        self.data_provider = DataProvider(use_cache=True)
        self.cache = DataCache(settings.data_dir)
        self.current_ticker = settings.last_ticker
        self.current_range = (settings.last_date_from, settings.last_date_to)
        
        # Optional: path to initial GA population file for auto-start optimization
        self.ga_init_from: str | None = ga_init_from
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create toolbar
        self.create_toolbar()
        
        # Create main central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Add data bar at the top
        self.data_bar = DataBar()
        self.data_bar.download_requested.connect(self.on_data_bar_download_requested)
        self.data_bar.timeframe_changed.connect(self.on_data_bar_timeframe_changed)
        main_layout.addWidget(self.data_bar)
        
        # Create main layout with 3-column splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Chart
        self.chart_view = CandleChartWidget()
        splitter.addWidget(self.chart_view)
        
        # Middle: Compact parameter editor (Seller-Exhaustion Entry)
        self.param_editor = CompactParamsEditor()
        splitter.addWidget(self.param_editor)
        
        # Right: Stats/Optimization panel (Performance Metrics with Optimize button in bottom right)
        self.stats_panel = StatsPanel()
        splitter.addWidget(self.stats_panel)
        
        # Connect stats panel to param editor
        self.stats_panel.set_param_editor(self.param_editor)
        
        # Strategy editor button removed - use toolbar button instead
        
        # Connect optimization signals
        self.stats_panel.optimization_step_complete.connect(self.on_optimization_step_complete)
        self.stats_panel.progress_updated.connect(self.update_progress)
        
        # Connect param changes to trigger re-calculation
        self.param_editor.params_changed.connect(self.on_params_changed)
        
        # Set initial sizes: Chart 50%, Params 25%, Stats 25% (right panel with Optimize button bottom-right)
        splitter.setSizes([800, 400, 400])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter, 1)  # Expand to fill space
        
        # Set the central widget
        self.setCentralWidget(central_widget)
        
        # Hide the main window's status bar (we use the custom black status bar in chart view)
        self.statusBar().hide()

    def _engine_label(self) -> str:
        """Return short label for current backtest engine."""
        workers = getattr(settings, 'optimizer_workers', max(1, multiprocessing.cpu_count() - 1))
        workers = max(1, int(workers))
        if workers > 1:
            return f"CPU ({workers} workers)"
        return "CPU (1 worker)"

    def _indicator_config_from_settings(self) -> dict[str, bool]:
        """Return chart indicator configuration based on saved settings."""
        return {
            'ema_fast': bool(settings.chart_ema_fast),
            'ema_slow': bool(settings.chart_ema_slow),
            'sma': bool(settings.chart_sma),
            'rsi': bool(settings.chart_rsi),
            'macd': bool(settings.chart_macd),
            'volume': bool(settings.chart_volume),
            'signals': bool(settings.chart_signals),
            'entries': bool(settings.chart_entries),
            'exits': bool(settings.chart_exits),
            'fib_retracements': True,  # Default ON; settings field not persisted yet
            'fib_0382': True,
            'fib_0500': True,
            'fib_0618': True,
            'fib_0786': True,
            'fib_1000': True,
        }

    def _infer_date_range(self, df=None) -> tuple[str, str]:
        """Infer date range from a DataFrame or fallback to last known range."""
        source = df if df is not None else self.current_data
        if source is not None and len(source) > 0:
            start = str(source.index[0].date())
            end = str(source.index[-1].date())
            return start, end
        return self.current_range

    def _persist_session(self, seller_params, backtest_params, backtest_result):
        """Persist the last backtest so it can be restored on restart."""
        if self.current_tf not in TIMEFRAME_META:
            return

        ticker = self.current_ticker or settings.last_ticker or "X:ADAUSD"
        start, end = self._infer_date_range()
        multiplier, timespan = TIMEFRAME_META[self.current_tf]

        save_session_snapshot(
            ticker=ticker,
            date_from=start,
            date_to=end,
            timeframe=self.current_tf,
            multiplier=multiplier,
            timespan=timespan,
            seller_params=seller_params,
            backtest_params=backtest_params,
            backtest_result=backtest_result,
        )
    
    def create_toolbar(self):
        """Create main toolbar with actions."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Settings action
        settings_action = QAction("⚙ Settings", self)
        settings_action.setToolTip("Configure data download, strategy parameters, and indicators")
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)
        
        # Strategy Editor action
        strategy_action = QAction("📊 Strategy Editor", self)
        strategy_action.setToolTip("Edit strategy parameters and manage parameter sets")
        strategy_action.triggered.connect(self.show_strategy_editor)
        toolbar.addAction(strategy_action)

        # Evolution Coach logs
        coach_action = QAction("🧠 Evolution Coach", self)
        coach_action.setToolTip("Open concise optimization logs for agent analysis")
        coach_action.triggered.connect(self.show_evolution_coach)
        toolbar.addAction(coach_action)
        
        toolbar.addSeparator()
        
        # Run Backtest action
        self.backtest_action = QAction("▶ Run Backtest", self)
        self.backtest_action.setToolTip("Run backtest on loaded data")
        self.backtest_action.setEnabled(False)
        self.backtest_action.triggered.connect(lambda: asyncio.create_task(self.run_backtest()))
        toolbar.addAction(self.backtest_action)
        
        toolbar.addSeparator()
        
        # Export Strategy action
        export_action = QAction("💾 Export Strategy", self)
        export_action.setToolTip("Export complete strategy configuration for live trading agent")
        export_action.triggered.connect(self.export_strategy_config)
        toolbar.addAction(export_action)
        
        # Import Strategy action
        import_action = QAction("📥 Import Strategy", self)
        import_action.setToolTip("Import strategy configuration")
        import_action.triggered.connect(self.import_strategy_config)
        toolbar.addAction(import_action)
        
        toolbar.addSeparator()
        
        # Clear action
        clear_action = QAction("🗑 Clear Results", self)
        clear_action.setToolTip("Clear backtest results")
        clear_action.triggered.connect(self.clear_results)
        toolbar.addAction(clear_action)
        
        toolbar.addSeparator()
        
        # Help action
        help_action = QAction("❓ Help", self)
        help_action.setToolTip("Show quick help")
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
    
    def on_data_bar_download_requested(self, from_date: str, to_date: str, tf_key: str):
        """Handle download request from data bar."""
        asyncio.create_task(self.download_data_from_bar(from_date, to_date, tf_key))
    
    def on_data_bar_timeframe_changed(self, tf_key: str):
        """Handle timeframe change from data bar."""
        # Map timeframe key to Timeframe enum
        tf_map = {
            "1m": Timeframe.m1,
            "3m": Timeframe.m3,
            "5m": Timeframe.m5,
            "10m": Timeframe.m10,
            "15m": Timeframe.m15,
            "30m": Timeframe.m30,
            "60m": Timeframe.m60,
        }
        
        new_tf = tf_map.get(tf_key, Timeframe.m15)
        if new_tf != self.current_tf:
            self.current_tf = new_tf
            self.chart_view.set_timeframe(new_tf)
            self.param_editor.set_timeframe(new_tf)
            
            # Update status
            self.chart_view.status_label.setText(f"Timeframe changed to {new_tf.value}")
            
            # Persist to settings
            from config.settings import SettingsManager
            tf_mult = int(tf_key.replace("m", "").replace("h", ""))
            if "h" in tf_key:
                tf_mult = int(tf_key.replace("h", "")) * 60
            SettingsManager.save_to_env({"timeframe": str(tf_mult)})
    
    async def download_data_from_bar(self, from_date: str, to_date: str, tf_key: str):
        """Download data from the data bar."""
        try:
            # Validate API key
            if not settings.polygon_api_key:
                QMessageBox.warning(
                    self,
                    "API Key Missing",
                    "Please set POLYGON_API_KEY in your .env file"
                )
                return
            
            # Map timeframe key to enum
            tf_map = {
                "1m": Timeframe.m1,
                "3m": Timeframe.m3,
                "5m": Timeframe.m5,
                "10m": Timeframe.m10,
                "15m": Timeframe.m15,
                "30m": Timeframe.m30,
                "60m": Timeframe.m60,
            }
            
            tf = tf_map.get(tf_key, Timeframe.m15)
            mult_str = tf_key.replace("m", "").replace("h", "")
            
            # Get timeframe multiplier
            if "h" in tf_key:
                mult = int(mult_str) * 60
                unit = "minute"
            else:
                mult = int(mult_str)
                unit = "minute"
            
            # Update UI
            self.data_bar.set_controls_enabled(False)
            self.chart_view.show_action_progress(f"Downloading {tf_key} data...")
            
            # Create provider if needed
            if not self.data_provider:
                self.data_provider = DataProvider(use_cache=True)
            
            # Download data
            def format_bars(current: int, total: int | None) -> str:
                if total and total > 0:
                    return f"{current:,}/{total:,}"
                return f"{current:,}"
            
            async def on_progress(progress):
                total_pages = max(progress.total_pages, 1)
                percentage = (progress.page / total_pages * 100) if total_pages > 0 else 0
                
                if progress.page == 0:
                    return
                
                remaining_text = (
                    f"≈ {self._format_duration(progress.seconds_remaining)} remaining"
                    if progress.seconds_remaining > 0
                    else "Finalizing..."
                )
                
                bars_text = format_bars(
                    progress.items_received,
                    progress.estimated_total_items,
                )
                
                msg = f"Download: {percentage:.0f}% | {bars_text} bars | {remaining_text}"
                self.chart_view.show_action_progress(msg)
            
            # Fetch data
            df = await self.data_provider.fetch(
                "X:ADAUSD",
                tf,
                from_date,
                to_date,
                progress_callback=on_progress,
                force_download=False,
            )
            
            if len(df) == 0:
                QMessageBox.warning(
                    self,
                    "No Data",
                    "No data was returned. Please check:\n"
                    "- Date range is valid\n"
                    "- API key is correct\n"
                    "- You haven't exceeded API quota"
                )
                self.chart_view.hide_action_progress("Download failed")
            else:
                # Process the downloaded data
                params, bt_params, _ = self.param_editor.get_params()
                feats = build_features(df, params, tf)
                
                # Update state
                self.current_data = feats
                self.raw_data = df
                self.current_tf = tf
                self.current_ticker = "X:ADAUSD"
                self.current_range = (str(feats.index[0].date()), str(feats.index[-1].date()))
                
                # Update chart
                self.chart_view.feats = feats
                self.chart_view.params = params
                self.chart_view.set_indicator_config(self._indicator_config_from_settings())
                self.chart_view.render_candles(feats)
                
                # Load parameters into param editor
                self.param_editor.set_params(params, bt_params)
                
                # Pass data to stats panel
                self.stats_panel.set_current_data(feats, tf, raw_data=df)
                
                # Enable backtest button
                self.backtest_action.setEnabled(True)
                
                # Update status
                signals = feats['exhaustion'].sum()
                self.chart_view.status_label.setText(
                    f"✓ Downloaded {len(feats)} bars | {signals} signals | Ready to backtest"
                )
                
                # Persist settings
                SettingsManager.save_to_env({
                    'timeframe': str(mult),
                    'last_date_from': from_date,
                    'last_date_to': to_date,
                    'last_ticker': 'X:ADAUSD',
                })
                
                # Show success message
                self.chart_view.hide_action_progress(f"✓ Downloaded {len(feats)} bars")
        
        except Exception as e:
            QMessageBox.critical(
                self,
                "Download Error",
                f"Failed to download data:\n{str(e)}"
            )
            self.chart_view.hide_action_progress("Download failed")
            import traceback
            traceback.print_exc()
        
        finally:
            self.data_bar.set_controls_enabled(True)
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
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
    
    def show_settings(self):
        """Show settings dialog."""
        if not self.settings_dialog:
            self.settings_dialog = SettingsDialog(self)
            self.settings_dialog.data_downloaded.connect(self.on_data_downloaded)
            # Track settings saved to refresh feature-engine preference
            self.settings_dialog.settings_saved.connect(self.on_settings_saved)
        
        self.settings_dialog.exec()
    
    def show_strategy_editor(self):
        """Show strategy editor dialog."""
        if not self.strategy_editor:
            from PySide6.QtWidgets import QDialog, QVBoxLayout
            
            # Create a dialog wrapper for the strategy editor
            dialog = QDialog(self)
            dialog.setWindowTitle("Strategy Parameter Editor")
            dialog.setMinimumWidth(1200)
            dialog.setMinimumHeight(800)
            
            layout = QVBoxLayout(dialog)
            
            # Create strategy editor widget
            editor = StrategyEditor(dialog)
            
            # Connect signals
            editor.params_changed.connect(self.on_strategy_params_changed)
            editor.params_loaded.connect(self.on_strategy_params_loaded)
            editor.params_saved.connect(self.on_strategy_params_saved)
            
            layout.addWidget(editor)
            
            # Store references
            self.strategy_editor = dialog
            self.strategy_editor_widget = editor
        
        self.strategy_editor.exec()

    def show_evolution_coach(self):
        """Evolution Coach window removed - using console logging now."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Evolution Coach",
            "Evolution Coach logging has been moved to console output.\n\n"
            "Check the terminal where you launched the application to see coach logs."
        )

    def on_data_downloaded(self, df):
        """Handle data download completion."""
        try:
            self.chart_view.show_action_progress("Building features...")
            
            # Get strategy params from compact editor (main window)
            params, bt_params, _ = self.param_editor.get_params()
            
            # Determine timeframe from settings
            tf_mult, tf_unit = self.settings_dialog.get_timeframe()
            tf_map = {
                1: Timeframe.m1,
                3: Timeframe.m3,
                5: Timeframe.m5,
                10: Timeframe.m10,
                15: Timeframe.m15,
                30: Timeframe.m30,
                60: Timeframe.m60,
            }
            tf = tf_map.get(int(tf_mult), Timeframe.m15)
            self.chart_view.set_timeframe(tf)
            self.param_editor.set_timeframe(tf)
            
            # Build features
            feats = build_features(df, params, tf)
            
            # Update chart
            self.chart_view.feats = feats
            self.chart_view.params = params
            
            # Get indicator config and apply
            indicator_config = (
                self.settings_dialog.get_indicator_config()
                if self.settings_dialog
                else self._indicator_config_from_settings()
            )
            self.chart_view.set_indicator_config(indicator_config)
            
            # Render chart
            self.chart_view.render_candles(feats)
            
            # Store data (both features and raw)
            self.current_data = feats
            self.raw_data = df  # Store raw OHLCV for rebuilding features
            self.current_tf = tf
            self.current_ticker = "X:ADAUSD"
            if len(feats) > 0:
                self.current_range = (
                    str(feats.index[0].date()),
                    str(feats.index[-1].date()),
                )
            
            # Load parameters into param editor
            self.param_editor.set_params(params, bt_params)
            
            # Pass data to stats panel for optimization (both features and raw)
            self.stats_panel.set_current_data(feats, tf, raw_data=df)
            
            # Enable backtest button
            self.backtest_action.setEnabled(True)
            
            # Update status
            signals = int(feats['exhaustion'].sum()) if 'exhaustion' in feats else 0
            eng = self._engine_label()
            self.chart_view.status_label.setText(
                f"✓ Features built [{eng}] · {tf.value} · {len(feats):,} bars · {signals:,} signals"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to process data:\n{str(e)}"
            )
            self.chart_view.status_label.setText(f"Error: {str(e)}")
        
        finally:
            self.chart_view.hide_action_progress()
    
    def on_params_changed(self):
        """Handle parameter changes from the compact editor."""
        # Just mark that params changed - backtest will use latest values when run
        if self.current_data is not None:
            self.chart_view.status_label.setText("Parameters changed — run backtest to apply")
            QTimer.singleShot(2000, lambda: self.chart_view.status_label.setText(f"Engine: {self._engine_label()}"))
    
    async def run_backtest(self):
        """Run backtest on current data."""
        if self.current_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please download data first using Settings."
            )
            return
        
        success = False
        try:
            self.backtest_action.setEnabled(False)
            self.chart_view.show_action_progress("Running backtest…")
            
            # Get current params from param editor (user may have edited them)
            # Note: fitness_config (3rd value) not used for single backtest
            seller_params, bt_params, _ = self.param_editor.get_params()
            
            # Rebuild features with current params from RAW data
            if self.raw_data is None:
                # Fallback: extract from current_data (may have dropped rows)
                print("⚠ No raw data available, using features dataframe")
                raw = self.current_data[['open', 'high', 'low', 'close', 'volume']]
            else:
                raw = self.raw_data
            
            feats = build_features(raw, seller_params, self.current_tf)
            self.current_data = feats
            self.current_range = self._infer_date_range(feats)
            self.chart_view.feats = feats
            self.chart_view.params = seller_params
            
            # Update stats panel with new features (keep same raw data)
            self.stats_panel.set_current_data(feats, self.current_tf, raw_data=self.raw_data)
            
            # Run backtest (in executor to avoid blocking)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                run_backtest,
                feats,
                bt_params
            )
            
            # Update stats panel
            self.stats_panel.update_stats(result)
            
            # Update chart with trade markers and pass bt_params for Fib target
            self.chart_view.set_backtest_result(result, bt_params)

            # Persist session snapshot
            self._persist_session(seller_params, bt_params, result)
            
            # Update status
            metrics = result['metrics']
            self.statusBar().showMessage(
                f"✓ Backtest complete: {metrics['n']} trades | Win {metrics['win_rate']:.1%} | Avg R {metrics['avg_R']:.2f} | Total ${metrics['total_pnl']:.4f}"
            )
            self.chart_view.status_label.setText(
                f"✓ Backtest ready [{self._engine_label()}] · trades {metrics['n']} · win {metrics['win_rate']:.1%}"
            )
            
            # Show summary
            QMessageBox.information(
                self,
                "Backtest Complete",
                f"Backtest completed successfully!\n\n"
                f"Total Trades: {metrics['n']}\n"
                f"Win Rate: {metrics['win_rate']:.1%}\n"
                f"Avg R-Multiple: {metrics['avg_R']:.2f}\n"
                f"Total PnL: ${metrics['total_pnl']:.4f}\n\n"
                f"See the Stats panel for detailed results."
            )
            success = True
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Backtest Error",
                f"Failed to run backtest:\n{str(e)}"
            )
            self.chart_view.status_label.setText(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            self.backtest_action.setEnabled(True)
            final_msg = "✓ Backtest complete" if success else "Backtest failed"
            self.chart_view.hide_action_progress(final_msg)
    
    def clear_results(self):
        """Clear backtest results."""
        self.stats_panel.clear()
        self.chart_view.backtest_result = None
        self.chart_view.update_trade_table(None)
        if self.current_data is not None:
            self.chart_view.render_candles(self.current_data)
        self.chart_view.status_label.setText("Results cleared")
        clear_session_snapshot()
    
    def update_progress(self, current: int, total: int, message: str):
        """
        Update progress bar with optimization progress.
        Uses the chart view's status bar progress.
        
        Args:
            current: Current generation number
            total: Total number of generations
            message: Status message
        """
        if total > 0:
            percentage = (current / total * 100) if total > 0 else 0
            
            # Update chart view progress bar
            self.chart_view.action_progress.setVisible(True)
            self.chart_view.action_progress.setRange(0, total)
            self.chart_view.action_progress.setValue(current)
            
            # Update chart view status label
            self.chart_view.status_label.setText(f"{message} ({percentage:.0f}%)")
            
            # Hide progress when complete
            if current >= total:
                QTimer.singleShot(2000, lambda: self.chart_view.action_progress.setVisible(False))
        else:
            self.chart_view.action_progress.setVisible(False)
            self.chart_view.status_label.setText(message)
    
    def on_optimization_step_complete(self, best_individual, backtest_result):
        """Handle optimization step completion and update chart."""
        if backtest_result is None:
            print("⚠ No backtest result to display")
            return
        
        try:
            # Get best individual's parameters
            seller_params = best_individual.seller_params
            backtest_params = best_individual.backtest_params
            
            # Rebuild features for chart display
            raw_data = self.current_data[['open', 'high', 'low', 'close', 'volume']].copy()
            feats = build_features(raw_data, seller_params, self.current_tf)
            
            # Update chart with features and backtest results
            self.chart_view.feats = feats
            self.chart_view.params = seller_params
            self.chart_view.set_backtest_result(backtest_result, backtest_params)
            self.current_data = feats
            self.current_range = self._infer_date_range(feats)
            
            # Update status bar with best individual info
            metrics = backtest_result['metrics']
            gen_text = f"Gen {best_individual.generation}" if hasattr(best_individual, 'generation') and best_individual.generation else "Best"
            self.statusBar().showMessage(
                f"🏆 {gen_text}: "
                f"{metrics['n']} trades | "
                f"Win rate: {metrics['win_rate']:.1%} | "
                f"Avg R: {metrics['avg_R']:.2f} | "
                f"Fitness: {best_individual.fitness:.4f}"
            )
            
            # Update stats panel metrics display
            self.stats_panel.trades_df = backtest_result['trades']
            self.stats_panel.metrics = metrics
            self.stats_panel.update_metrics()
            self.stats_panel.update_equity_curve()

            self._persist_session(seller_params, best_individual.backtest_params, backtest_result)
            
            from core.logging_utils import get_logger
            get_logger(__name__).info(
                "✓ Chart and metrics updated with best strategy: %s trades, %.1f%% win rate",
                metrics['n'], 100.0 * metrics['win_rate']
            )
            
        except Exception as e:
            print(f"❌ Error updating chart: {e}")
            import traceback
            traceback.print_exc()
    
    def on_strategy_params_changed(self):
        """Handle strategy parameter changes from editor."""
        self.statusBar().showMessage("Strategy parameters updated", 3000)
    
    def on_strategy_params_loaded(self, seller_params, backtest_params):
        """Handle strategy parameter loading from editor."""
        # Update settings dialog if it exists
        if self.settings_dialog:
            # Strategy params now managed in main window compact editor
            self.settings_dialog.set_backtest_params(backtest_params)

    def on_settings_saved(self):
        """Refresh runtime flags from settings after the dialog saves."""
        try:
            SettingsManager.reload_settings()
            # Apply new logging level immediately
            from core.logging_utils import configure_logging
            configure_logging(level=getattr(settings, 'ada_agent_log_level', 'INFO'))
        except Exception:
            pass

        status = self._engine_label()
        self.statusBar().showMessage(f"Settings saved. Engine: {status}", 3000)
        if hasattr(self, 'chart_view'):
            self.chart_view.status_label.setText(f"Engine: {status}")
        
        # Update param editor to reflect saved settings (best-effort)
        try:
            sp = SellerParams(
                ema_fast=int(settings.strategy_ema_fast),
                ema_slow=int(settings.strategy_ema_slow),
                z_window=int(settings.strategy_z_window),
                vol_z=float(settings.strategy_vol_z),
                tr_z=float(settings.strategy_tr_z),
                cloc_min=float(settings.strategy_cloc_min),
                atr_window=int(settings.strategy_atr_window),
            )
            bp = BacktestParams(
                fee_bp=float(settings.backtest_fee_bp),
                slippage_bp=float(settings.backtest_slippage_bp),
            )
            if hasattr(self, 'param_editor') and self.param_editor is not None:
                self.param_editor.set_params(sp, bp)
            self.statusBar().showMessage("Parameters loaded from settings", 3000)
        except Exception:
            # Ignore settings parsing errors
            pass
        
        # Re-run backtest if data is loaded
        if self.current_data is not None:
            asyncio.create_task(self.run_backtest())

    def on_strategy_params_saved(self, seller_params, backtest_params):
        """Persist saved parameters and re-run strategy automatically."""
        if self.settings_dialog:
            # Strategy params now managed in main window compact editor
            self.settings_dialog.set_backtest_params(backtest_params)
        if hasattr(self, 'stats_panel'):
            self.stats_panel.load_params_from_settings(seller_params, backtest_params)
        asyncio.create_task(self._persist_params_and_run(seller_params, backtest_params))

    async def _persist_params_and_run(self, seller_params, backtest_params):
        """Persist parameters to settings and rerun strategy asynchronously."""
        try:
            self.chart_view.show_action_progress("Saving parameters…")
            settings_dict = {
                'strategy_ema_fast': seller_params.ema_fast,
                'strategy_ema_slow': seller_params.ema_slow,
                'strategy_z_window': seller_params.z_window,
                'strategy_vol_z': seller_params.vol_z,
                'strategy_tr_z': seller_params.tr_z,
                'strategy_cloc_min': seller_params.cloc_min,
                'strategy_atr_window': seller_params.atr_window,
                'backtest_fee_bp': backtest_params.fee_bp,
                'backtest_slippage_bp': backtest_params.slippage_bp,
            }
            SettingsManager.save_to_env(settings_dict)
            SettingsManager.reload_settings()
            self.statusBar().showMessage("Parameters saved. Running strategy…")

            if self.current_data is not None:
                self.chart_view.show_action_progress("Running strategy with saved parameters…")
                await self.run_backtest()
            else:
                self.chart_view.hide_action_progress("Parameters saved. Download data to run strategy.")
        except Exception as e:
            self.chart_view.hide_action_progress("Parameter save failed")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to persist parameters:\n{str(e)}"
            )
    
    def show_help(self):
        """Show quick help dialog."""
        QMessageBox.information(
            self,
            "Quick Help",
            "<h3>ADA Seller-Exhaustion Trading Agent</h3>"
            "<p><b>Getting Started:</b></p>"
            "<ol>"
            "<li>Click <b>Settings</b> to download historical data</li>"
            "<li>Click <b>📊 Strategy Editor</b> to view/edit strategy parameters with explanations</li>"
            "<li>Configure strategy parameters and indicators</li>"
            "<li>Click <b>Run Backtest</b> to test the strategy</li>"
            "<li>View results in the Stats panel and chart markers</li>"
            "</ol>"
            "<p><b>Chart Markers:</b></p>"
            "<ul>"
            "<li>🔺 Yellow triangles: Entry signals detected</li>"
            "<li>🔼 Green arrows up: Buy orders (trade entries)</li>"
            "<li>🔽 Red/Green arrows down: Sell orders (exits)<br>"
            "   &nbsp;&nbsp;&nbsp;Green = winning trade, Red = losing trade</li>"
            "</ul>"
            "<p><b>Strategy Editor Features:</b></p>"
            "<ul>"
            "<li>View all parameters with detailed explanations</li>"
            "<li>Save/load evolved parameter sets</li>"
            "<li>Export parameters to YAML</li>"
            "<li>Fibonacci retracement exit configuration</li>"
            "</ul>"
            "<p><b>Tips:</b></p>"
            "<ul>"
            "<li>Download at least 7 days of data for accurate indicators</li>"
            "<li>Toggle indicators in Settings → Chart Indicators</li>"
            "<li>Export trades to CSV from the Stats panel</li>"
            "<li>Save good parameter sets via Strategy Editor</li>"
            "</ul>"
        )
    
    def export_strategy_config(self):
        """Export complete strategy configuration to JSON for live trading agent."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        
        try:
            # Get current parameters
            seller_params, bt_params, fitness_config = self.param_editor.get_params()
            
            # Get backtest metrics if available
            backtest_metrics = None
            if hasattr(self.stats_panel, 'last_metrics') and self.stats_panel.last_metrics:
                backtest_metrics = self.stats_panel.last_metrics
            
            # Create trading config
            config = create_default_config(
                seller_params=seller_params,
                backtest_params=bt_params,
                timeframe=self.current_tf,
                description=f"Exported from backtesting app on {self.current_range[0]} to {self.current_range[1]}",
                backtest_metrics=backtest_metrics
            )
            
            # Prompt for save location
            default_filename = f"strategy_{self.current_tf.value}_{self.current_range[0]}_{self.current_range[1]}.json"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Strategy Configuration",
                default_filename,
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Export to file
            export_trading_config(config, file_path, pretty=True)
            
            # Validate configuration
            is_valid, warnings = validate_config_for_live_trading(config)
            
            # Show success message with warnings
            warning_text = "\n".join(warnings) if warnings else "No warnings"
            
            QMessageBox.information(
                self,
                "Strategy Exported Successfully",
                f"<h3>✅ Strategy Configuration Exported</h3>"
                f"<p><b>File:</b> {file_path}</p>"
                f"<p><b>Timeframe:</b> {config.timeframe.value}</p>"
                f"<p><b>Exit Strategy:</b> {'Fibonacci' if config.backtest_params.use_fib_exits else 'Traditional'}</p>"
                f"<p><b>Paper Trading:</b> {'Enabled' if config.exchange.paper_trading else 'DISABLED (LIVE)'}</p>"
                f"<hr>"
                f"<p><b>Validation:</b></p>"
                f"<pre style='color: {'green' if is_valid else 'orange'};'>{warning_text}</pre>"
                f"<hr>"
                f"<p><b>Next Steps:</b></p>"
                f"<ol>"
                f"<li>Copy this file to your trading agent application</li>"
                f"<li>Rename to <code>config.json</code></li>"
                f"<li>Configure exchange credentials in <code>.env</code> file</li>"
                f"<li>Start agent with paper trading enabled</li>"
                f"</ol>"
                f"<p><i>⚠️ See PRD_TRADING_AGENT.md for complete setup instructions</i></p>"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"<h3>❌ Failed to Export Strategy</h3>"
                f"<p><b>Error:</b> {str(e)}</p>"
                f"<p>Please check the logs for details.</p>"
            )
            import traceback
            traceback.print_exc()
    
    def import_strategy_config(self):
        """Import strategy configuration from JSON."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        
        try:
            # Prompt for file
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Strategy Configuration",
                "",
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Import from file
            config = import_trading_config(file_path)
            
            # Validate configuration
            is_valid, warnings = validate_config_for_live_trading(config)
            
            # Show warnings if any
            if warnings:
                warning_text = "\n".join(warnings)
                result = QMessageBox.warning(
                    self,
                    "Configuration Warnings",
                    f"<h3>⚠️ Configuration has warnings:</h3>"
                    f"<pre style='color: orange;'>{warning_text}</pre>"
                    f"<p><b>Continue with import?</b></p>",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if result == QMessageBox.No:
                    return
            
            # Apply to UI
            self.param_editor.set_params(
                config.seller_params,
                config.backtest_params,
                config.get('fitness_config')  # May not exist in older configs
            )
            
            # Update timeframe if different
            if config.timeframe != self.current_tf:
                self.current_tf = config.timeframe
                self.stats_panel.set_current_timeframe(config.timeframe)
            
            # Show success message
            QMessageBox.information(
                self,
                "Strategy Imported Successfully",
                f"<h3>✅ Strategy Configuration Imported</h3>"
                f"<p><b>File:</b> {file_path}</p>"
                f"<p><b>Strategy:</b> {config.strategy_name}</p>"
                f"<p><b>Timeframe:</b> {config.timeframe.value}</p>"
                f"<p><b>Description:</b> {config.description or 'None'}</p>"
                f"<hr>"
                f"<p>Parameters have been loaded into the UI.</p>"
                f"<p><b>Click 'Run Backtest'</b> to test these parameters on your data.</p>"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Failed",
                f"<h3>❌ Failed to Import Strategy</h3>"
                f"<p><b>Error:</b> {str(e)}</p>"
                f"<p>Please check that the file is a valid strategy configuration.</p>"
            )
            import traceback
            traceback.print_exc()
    
    def save_window_state(self):
        """Save window and splitter state to settings."""
        from config.settings import SettingsManager
        
        try:
            central_widget = self.centralWidget()
            # The central widget is now a VBox with DataBar + splitter
            splitter = central_widget.findChild(QSplitter)
            if splitter:
                sizes = splitter.sizes()
                settings_dict = {
                    'window_width': self.width(),
                    'window_height': self.height(),
                    'splitter_chart': sizes[0] if len(sizes) > 0 else 800,
                    'splitter_params': sizes[1] if len(sizes) > 1 else 400,
                    'splitter_stats': sizes[2] if len(sizes) > 2 else 400,
                }
            else:
                settings_dict = {
                    'window_width': self.width(),
                    'window_height': self.height(),
                }
            
            # Save chart view state if available
            if hasattr(self.chart_view, 'plot_widget'):
                view_range = self.chart_view.plot_widget.viewRange()
                if view_range:
                    x_range, y_range = view_range
                    settings_dict.update({
                        'chart_x_min': x_range[0],
                        'chart_x_max': x_range[1],
                        'chart_y_min': y_range[0],
                        'chart_y_max': y_range[1],
                    })
            
            SettingsManager.save_to_env(settings_dict)
        except Exception as e:
            print(f"Warning: Could not save window state: {e}")
    
    def restore_window_state(self):
        """Restore window and splitter state from settings."""
        from config.settings import settings
        
        try:
            # Restore window size
            if settings.window_width and settings.window_height:
                self.resize(settings.window_width, settings.window_height)
            
            # Restore 3-column splitter sizes (Chart, Params, Stats)
            central_widget = self.centralWidget()
            splitter = central_widget.findChild(QSplitter)
            
            if splitter and hasattr(settings, 'splitter_chart') and hasattr(settings, 'splitter_params') and hasattr(settings, 'splitter_stats'):
                splitter.setSizes([
                    settings.splitter_chart,
                    settings.splitter_params,
                    settings.splitter_stats
                ])
            elif splitter and hasattr(settings, 'splitter_left') and hasattr(settings, 'splitter_right'):
                # Backwards compatibility: old 2-column layout
                # Convert to 3-column by splitting right side
                left = settings.splitter_left
                middle = int(settings.splitter_right * 0.5)
                right = int(settings.splitter_right * 0.5)
                splitter.setSizes([left, middle, right])
            
            # Note: Chart view range restored when data is loaded
        except Exception as e:
            print(f"Warning: Could not restore window state: {e}")
    
    async def try_load_cached_data(self):
        """Try to load cached data from last session - CANDLESTICKS ONLY (no trades)."""
        try:
            # Always use current settings for timeframe and date range
            ticker = settings.last_ticker or "X:ADAUSD"
            from_ = settings.last_date_from or "2024-01-01"
            to = settings.last_date_to or "2025-12-31"
            
            # Derive current_tf from settings.timeframe (integer minutes)
            tf_map = {
                1: Timeframe.m1,
                3: Timeframe.m3,
                5: Timeframe.m5,
                10: Timeframe.m10,
                15: Timeframe.m15,
                30: Timeframe.m30,
                60: Timeframe.m60,
            }
            raw_tf = settings.timeframe
            tf_numeric = None
            if isinstance(raw_tf, (int, float)):
                tf_numeric = int(raw_tf)
            elif isinstance(raw_tf, str):
                digits_only = "".join(ch for ch in raw_tf if ch.isdigit())
                if digits_only:
                    tf_numeric = int(digits_only)
            self.current_tf = tf_map.get(tf_numeric, Timeframe.m15)
            self.chart_view.set_timeframe(self.current_tf)
            self.param_editor.set_timeframe(self.current_tf)
            self.current_ticker = ticker
            
            # Load strategy parameters from settings
            seller_params = SellerParams(
                ema_fast=settings.strategy_ema_fast,
                ema_slow=settings.strategy_ema_slow,
                z_window=settings.strategy_z_window,
                vol_z=settings.strategy_vol_z,
                tr_z=settings.strategy_tr_z,
                cloc_min=settings.strategy_cloc_min,
                atr_window=settings.strategy_atr_window,
            )
            bt_params = BacktestParams(
                atr_stop_mult=settings.backtest_atr_stop_mult,
                reward_r=settings.backtest_reward_r,
                max_hold=settings.backtest_max_hold,
                fee_bp=settings.backtest_fee_bp,
                slippage_bp=settings.backtest_slippage_bp,
            )
            
            target_multiplier, target_timespan = TIMEFRAME_META.get(self.current_tf, (15, "minute"))
            
            # STRICT: Only load cache if it matches EXACT timeframe
            target_available = self.cache.has_cached_data(ticker, from_, to, target_multiplier, target_timespan)

            if not target_available:
                self.statusBar().showMessage(
                    f"No cached data for {self.current_tf.value} - Click Settings to download"
                )
                return

            self.statusBar().showMessage(f"Loading cached {self.current_tf.value} data...")
            self.chart_view.show_action_progress(f"Loading cached {self.current_tf.value} data...")

            # Load EXACT timeframe data (no fallback to wrong timeframe) in a worker thread
            df = await asyncio.to_thread(
                self.cache.get_cached_data,
                ticker,
                from_,
                to,
                target_multiplier,
                target_timespan,
            )

            if df is None or len(df) == 0:
                self.statusBar().showMessage(
                    f"Cached data unavailable for {self.current_tf.value} - Click Settings to download"
                )
                return

            # Build features for chart display only (CPU heavy → run off the UI thread)
            feats = await asyncio.to_thread(build_features, df, seller_params, self.current_tf)
            self.chart_view.feats = feats
            self.chart_view.params = seller_params
            self.chart_view.set_indicator_config(self._indicator_config_from_settings())
            
            # Render candlesticks WITHOUT trades (no backtest on startup)
            self.chart_view.render_candles(feats, backtest_result=None)

            # Store data (both features and raw)
            self.current_data = feats
            self.raw_data = df  # Store raw OHLCV for rebuilding features
            self.current_range = self._infer_date_range(feats)

            # Load parameters into param editor
            self.param_editor.set_params(seller_params, bt_params)
            
            # Pass data to stats panel (WITH raw_data for optimization)
            self.stats_panel.set_current_data(feats, self.current_tf, raw_data=df)

            # Enable backtest button but DON'T run backtest automatically
            self.backtest_action.setEnabled(True)

            signals = int(feats["exhaustion"].sum())
            date_range = f"{self.current_range[0]} to {self.current_range[1]}"
            self.statusBar().showMessage(
                f"✓ Loaded {len(feats)} bars ({date_range}, {self.current_tf.value}) | "
                f"{signals} signals | Click 'Run Backtest' to see trades"
            )

            print(f"✓ Auto-loaded cached data: {len(feats)} bars ({self.current_tf.value}), {signals} signals")
            print(f"   Chart ready - Run backtest to see trades")
            
        except Exception as e:
            print(f"⚠ Could not auto-load cached data: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(
                "Could not load cached data - Click Settings to download fresh data"
            )
        
        finally:
            self.chart_view.hide_action_progress()
    
    def closeEvent(self, event):
        """Save state when window is closed."""
        self.save_window_state()
        self.save_parameters_state()
        event.accept()
    
    def save_parameters_state(self):
        """Save current parameters from compact editor to .env for persistence."""
        from config.settings import SettingsManager
        
        try:
            # Get current parameters from compact editor
            seller_params, bt_params, fitness_config = self.param_editor.get_params()
            
            # Convert to settings dict for .env
            # Note: These are in bar counts (already converted by get_params)
            settings_dict = {
                # Strategy parameters (in bars for this timeframe)
                'strategy_ema_fast': seller_params.ema_fast,
                'strategy_ema_slow': seller_params.ema_slow,
                'strategy_z_window': seller_params.z_window,
                'strategy_vol_z': seller_params.vol_z,
                'strategy_tr_z': seller_params.tr_z,
                'strategy_cloc_min': seller_params.cloc_min,
                'strategy_atr_window': seller_params.atr_window,
                
                # Backtest parameters
                'backtest_fee_bp': bt_params.fee_bp,
                'backtest_slippage_bp': bt_params.slippage_bp,
            }
            
            SettingsManager.save_to_env(settings_dict)
            print("✓ Parameters saved to .env")
        except Exception as e:
            print(f"Warning: Could not save parameters state: {e}")
    
    async def initialize(self):
        """Async initialization after window is shown."""
        # Reload settings to ensure .env values override any stale environment variables
        SettingsManager.reload_settings()
        
        # Check if coach model is already loaded (on startup)
        await self._check_coach_model_status_on_startup()
        
        # Restore window state
        self.restore_window_state()
        
        # Sync timeframe with saved settings
        try:
            tf_map = {
                1: Timeframe.m1,
                3: Timeframe.m3,
                5: Timeframe.m5,
                10: Timeframe.m10,
                15: Timeframe.m15,
                30: Timeframe.m30,
                60: Timeframe.m60,
            }
            tf_mult = int(settings.timeframe)
            
            # If configured timeframe not supported, default to 15m
            if tf_mult not in tf_map:
                print(f"⚠ Configured timeframe {tf_mult}m not supported, defaulting to 15m")
                tf_mult = 15
            
            self.current_tf = tf_map[tf_mult]
            self.chart_view.set_timeframe(self.current_tf)
            self.param_editor.set_timeframe(self.current_tf)
            print(f"✓ Timeframe set to: {self.current_tf.value} ({tf_mult} minutes)")
        except Exception as e:
            print(f"Warning: Could not sync timeframe from settings: {e}")
            self.current_tf = Timeframe.m15
            self.chart_view.set_timeframe(self.current_tf)
        
        # Load saved parameters from .env into compact editor
        self.load_parameters_state()
        
        # Try to auto-load cached data
        await self.try_load_cached_data()
        
        # If a GA population file was provided, initialize optimizer from it
        # and auto-start multi-step optimization after the UI is ready
        if self.ga_init_from:
            try:
                # Provide the initial population file to the stats panel
                self.stats_panel.set_initial_population_file(self.ga_init_from)

                # Wait until data is loaded before starting optimization
                def _maybe_start():
                    if self.stats_panel.current_data is not None:
                        logger.info("Auto-starting optimization from population file: %s", self.ga_init_from)
                        self.stats_panel.run_multi_step_optimize()
                    else:
                        # Poll again shortly until data is ready
                        QTimer.singleShot(200, _maybe_start)

                QTimer.singleShot(200, _maybe_start)
            except Exception as e:
                logger.exception("Failed to auto-start optimization from file: %s", e)
    
    async def _check_coach_model_status_on_startup(self):
        """Check if Evolution Coach model is already loaded on app startup."""
        try:
# Removed import for deleted file
            # Removed: coach_log_manager (now using console logging)
            
            model = settings.coach_model
            
            # Create temporary client to check model status
            # Uses agent.txt automatically (no prompt selection needed)
            temp_client = GemmaCoachClient(
                model=model,
                verbose=False
            )
            
            # Check if model is already loaded
            is_loaded = await temp_client.check_model_loaded()
            
            if is_loaded:
                # Model already loaded, update button state
                self.param_editor.set_coach_model_loaded(True)
                self.chart_view.set_coach_status(f"✅ Model already loaded: {model}")
                logger.info("✅ Coach model already loaded on startup: %s", model)
                
                # Store client for later use
                self.coach_client = temp_client
            else:
                # Model not loaded
                self.param_editor.set_coach_model_loaded(False)
                self.chart_view.set_coach_status(f"Ready to load model: {model}")
                logger.info("Coach model not loaded on startup: %s", model)
        
        except Exception as e:
            # Non-critical, just log and continue
            logger.debug("Could not check coach model status on startup: %s", e)
    
    def load_parameters_state(self):
        """Load saved parameters from .env into compact editor."""
        try:
            # Create parameter objects from saved settings
            seller_params = SellerParams(
                ema_fast=settings.strategy_ema_fast,
                ema_slow=settings.strategy_ema_slow,
                z_window=settings.strategy_z_window,
                vol_z=settings.strategy_vol_z,
                tr_z=settings.strategy_tr_z,
                cloc_min=settings.strategy_cloc_min,
                atr_window=settings.strategy_atr_window,
            )
            
            backtest_params = BacktestParams(
                fee_bp=settings.backtest_fee_bp,
                slippage_bp=settings.backtest_slippage_bp,
            )
            
            # Load into compact editor (will convert bars to minutes for display)
            self.param_editor.set_params(seller_params, backtest_params)
            logger.info("Parameters loaded from .env")
        except Exception as e:
            logger.warning("Could not load parameters state: %s", e)
            # Parameters will remain at defaults if loading fails


def main(ga_init_from: str | None = None):
    """Main entry point for the UI application."""
    configure_logging()
    # Create processes/<pid> marker file at startup (empty file)
    try:
        pid = os.getpid()
        proc_dir = os.path.join(os.getcwd(), "processes")
        os.makedirs(proc_dir, exist_ok=True)
        marker_path = os.path.join(proc_dir, str(pid))
        with open(marker_path, "a", encoding="utf-8"):
            pass  # create empty file if not exists
        logger.info("Process marker created: %s", marker_path)
    except Exception as e:
        # Non-fatal
        print(f"Warning: Could not create process marker: {e}")

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_FOREST_QSS)
    
    window = MainWindow(ga_init_from=ga_init_from)
    window.show()
    
    if HAS_QASYNC:
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        with loop:
            loop.create_task(window.initialize())
            loop.run_forever()
    else:
        # Fallback without qasync - use QTimer to run event loop
        logger.warning("qasync not available, async features may not work properly")
        logger.warning("Install qasync for full functionality: poetry add qasync")
        
        timer = QTimer()
        timer.start(100)
        timer.timeout.connect(lambda: None)
        
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
