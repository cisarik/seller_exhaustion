import sys
import asyncio
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

from app.theme import DARK_FOREST_QSS
from app.widgets.candle_view import CandleChartWidget
from app.widgets.settings_dialog import SettingsDialog
from app.widgets.stats_panel import StatsPanel
from app.widgets.strategy_editor import StrategyEditor
from app.widgets.compact_params import CompactParamsEditor
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


class MainWindow(QMainWindow):
    """Main application window with toolbar, chart, and stats panel."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADA Seller-Exhaustion Trading Agent")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Data and settings
        self.current_data = None  # Features dataframe (with indicators and signals)
        self.raw_data = None  # Raw OHLCV data (for rebuilding features)
        self.current_tf = Timeframe.m15
        self.settings_dialog = None
        self.strategy_editor = None
        self.data_provider = DataProvider(use_cache=True)
        self.cache = DataCache(settings.data_dir)
        self.current_ticker = settings.last_ticker
        self.current_range = (settings.last_date_from, settings.last_date_to)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create toolbar
        self.create_toolbar()
        
        # Create main layout with 3-column splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Chart
        self.chart_view = CandleChartWidget()
        splitter.addWidget(self.chart_view)
        
        # Middle: Stats/Optimization panel
        self.stats_panel = StatsPanel()
        splitter.addWidget(self.stats_panel)
        
        # Right: Compact parameter editor
        self.param_editor = CompactParamsEditor()
        splitter.addWidget(self.param_editor)
        
        # Connect stats panel to param editor
        self.stats_panel.set_param_editor(self.param_editor)
        
        # Connect param editor's strategy editor button to open dialog
        self.param_editor.strategy_editor_btn.clicked.connect(self.show_strategy_editor)
        
        # Connect optimization signals
        self.stats_panel.optimization_step_complete.connect(self.on_optimization_step_complete)
        self.stats_panel.progress_updated.connect(self.update_progress)
        
        # Connect param changes to trigger re-calculation
        self.param_editor.params_changed.connect(self.on_params_changed)
        
        # Set initial sizes (chart: 50%, stats: 30%, params: 20%)
        splitter.setSizes([800, 480, 320])
        
        self.setCentralWidget(splitter)
        
        # Hide the main window's status bar (we use the custom black status bar in chart view)
        self.statusBar().hide()

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
    
    def show_settings(self):
        """Show settings dialog."""
        if not self.settings_dialog:
            self.settings_dialog = SettingsDialog(self)
            self.settings_dialog.data_downloaded.connect(self.on_data_downloaded)
        
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
    
    def on_data_downloaded(self, df):
        """Handle data download completion."""
        try:
            self.chart_view.show_action_progress("Building features...")
            
            # Get strategy params from settings
            params = self.settings_dialog.get_strategy_params()
            bt_params = self.settings_dialog.get_backtest_params()
            
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
            signals = feats['exhaustion'].sum()
            self.chart_view.status_label.setText(
                f"✓ Loaded {len(feats)} bars | {signals} signals detected | Ready to backtest"
            )
            
            # Update info
            self.chart_view.info_label.setText(
                f"Date range: {feats.index[0]} to {feats.index[-1]}\n"
                f"Signals detected: {signals}"
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
            self.chart_view.status_label.setText("Parameters changed - run backtest to see effects")
            QTimer.singleShot(2000, lambda: self.chart_view.status_label.setText("Ready"))
    
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
            
            # Update chart with trade markers
            self.chart_view.set_backtest_result(result)

            # Persist session snapshot
            self._persist_session(seller_params, bt_params, result)
            
            # Update status
            metrics = result['metrics']
            self.statusBar().showMessage(
                f"✓ Backtest complete: {metrics['n']} trades | "
                f"Win rate: {metrics['win_rate']:.1%} | "
                f"Avg R: {metrics['avg_R']:.2f} | "
                f"Total PnL: ${metrics['total_pnl']:.4f}"
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
            final_msg = "Ready" if success else "Backtest failed"
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
            
            # Rebuild features for chart display
            raw_data = self.current_data[['open', 'high', 'low', 'close', 'volume']].copy()
            feats = build_features(raw_data, seller_params, self.current_tf)
            
            # Update chart with features and backtest results
            self.chart_view.feats = feats
            self.chart_view.params = seller_params
            self.chart_view.set_backtest_result(backtest_result)
            self.current_data = feats
            self.current_range = self._infer_date_range(feats)
            
            # Update status bar
            metrics = backtest_result['metrics']
            self.statusBar().showMessage(
                f"🏆 Gen {best_individual.generation} Best: "
                f"{metrics['n']} trades | "
                f"Win rate: {metrics['win_rate']:.1%} | "
                f"Avg R: {metrics['avg_R']:.2f} | "
                f"Fitness: {best_individual.fitness:.4f}"
            )

            self._persist_session(seller_params, best_individual.backtest_params, backtest_result)
            
            print(f"✓ Chart updated with winning strategy")
            
        except Exception as e:
            print(f"Error updating chart: {e}")
            import traceback
            traceback.print_exc()
    
    def on_strategy_params_changed(self):
        """Handle strategy parameter changes from editor."""
        self.statusBar().showMessage("Strategy parameters updated", 3000)
    
    def on_strategy_params_loaded(self, seller_params, backtest_params):
        """Handle strategy parameter loading from editor."""
        # Update settings dialog if it exists
        if self.settings_dialog:
            self.settings_dialog.set_strategy_params(seller_params)
            self.settings_dialog.set_backtest_params(backtest_params)
        
        # Update param editor
        if hasattr(self, 'param_editor'):
            self.param_editor.set_params(seller_params, backtest_params)
        
        self.statusBar().showMessage("Parameters loaded from file", 3000)
        
        # Re-run backtest if data is loaded
        if self.current_data is not None:
            asyncio.create_task(self.run_backtest())

    def on_strategy_params_saved(self, seller_params, backtest_params):
        """Persist saved parameters and re-run strategy automatically."""
        if self.settings_dialog:
            self.settings_dialog.set_strategy_params(seller_params)
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
                'backtest_atr_stop_mult': backtest_params.atr_stop_mult,
                'backtest_reward_r': backtest_params.reward_r,
                'backtest_max_hold': backtest_params.max_hold,
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
            sizes = self.centralWidget().sizes()
            settings_dict = {
                'window_width': self.width(),
                'window_height': self.height(),
                'splitter_chart': sizes[0] if len(sizes) > 0 else 800,
                'splitter_stats': sizes[1] if len(sizes) > 1 else 480,
                'splitter_params': sizes[2] if len(sizes) > 2 else 320,
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
            
            # Restore 3-column splitter sizes
            if hasattr(settings, 'splitter_chart') and hasattr(settings, 'splitter_stats') and hasattr(settings, 'splitter_params'):
                self.centralWidget().setSizes([
                    settings.splitter_chart,
                    settings.splitter_stats,
                    settings.splitter_params
                ])
            elif hasattr(settings, 'splitter_left') and hasattr(settings, 'splitter_right'):
                # Backwards compatibility: old 2-column layout
                # Convert to 3-column by taking 20% from right side for params
                left = settings.splitter_left
                right = int(settings.splitter_right * 0.6)
                params = int(settings.splitter_right * 0.4)
                self.centralWidget().setSizes([left, right, params])
            
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

            self.chart_view.info_label.setText(
                f"📊 {ticker} {self.current_tf.value} | "
                f"Date range: {date_range}\n"
                f"Signals: {signals} | Click 'Run Backtest' for trades"
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
        event.accept()
    
    async def initialize(self):
        """Async initialization after window is shown."""
        # Reload settings to ensure .env values override any stale environment variables
        SettingsManager.reload_settings()
        
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
        
        # Try to auto-load cached data
        await self.try_load_cached_data()


def main():
    """Main entry point for the UI application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_FOREST_QSS)
    
    window = MainWindow()
    window.show()
    
    if HAS_QASYNC:
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        with loop:
            loop.create_task(window.initialize())
            loop.run_forever()
    else:
        # Fallback without qasync - use QTimer to run event loop
        print("WARNING: qasync not available, async features may not work properly")
        print("Install qasync for full functionality: poetry add qasync")
        
        timer = QTimer()
        timer.start(100)
        timer.timeout.connect(lambda: None)
        
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
