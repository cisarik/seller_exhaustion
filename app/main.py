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
from strategy.seller_exhaustion import build_features, SellerParams
from backtest.engine import BacktestParams
from core.models import Timeframe
from backtest.engine import run_backtest
from data.provider import DataProvider
from data.cache import DataCache
from config.settings import settings, SettingsManager


class MainWindow(QMainWindow):
    """Main application window with toolbar, chart, and stats panel."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADA Seller-Exhaustion Trading Agent")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Data and settings
        self.current_data = None
        self.current_tf = Timeframe.m15
        self.settings_dialog = None
        self.strategy_editor = None
        self.data_provider = DataProvider(use_cache=True)
        self.cache = DataCache()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create toolbar
        self.create_toolbar()
        
        # Create main layout with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Chart
        self.chart_view = CandleChartWidget()
        splitter.addWidget(self.chart_view)
        
        # Right side: Stats panel
        self.stats_panel = StatsPanel()
        splitter.addWidget(self.stats_panel)
        
        # Connect optimization signals
        self.stats_panel.optimization_step_complete.connect(self.on_optimization_step_complete)
        self.stats_panel.progress_updated.connect(self.update_progress)
        
        # Set initial sizes (chart gets 70%, stats gets 30%)
        splitter.setSizes([1120, 480])
        
        self.setCentralWidget(splitter)
        
        # Status bar with progress
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Initializing...")
    
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
            self.statusBar().showMessage("Building features...")
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            
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
                60: Timeframe.m60,
            }
            tf = tf_map.get(int(tf_mult), Timeframe.m15)
            self.chart_view.set_timeframe(tf)
            
            # Build features
            feats = build_features(df, params, tf)
            
            # Update chart
            self.chart_view.feats = feats
            self.chart_view.params = params
            
            # Get indicator config and apply
            indicator_config = self.settings_dialog.get_indicator_config()
            self.chart_view.set_indicator_config(indicator_config)
            
            # Render chart
            self.chart_view.render_candles(feats)
            
            # Store data
            self.current_data = feats
            self.current_tf = tf
            
            # Load parameters into stats panel
            self.stats_panel.load_params_from_settings(params, bt_params)
            
            # Pass data to stats panel for optimization
            self.stats_panel.set_current_data(feats, tf)
            
            # Enable backtest button
            self.backtest_action.setEnabled(True)
            
            # Update status
            signals = feats['exhaustion'].sum()
            self.statusBar().showMessage(
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
            self.statusBar().showMessage(f"Error: {str(e)}")
        
        finally:
            self.progress.setVisible(False)
    
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
            self.statusBar().showMessage("Running backtest...")
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.backtest_action.setEnabled(False)
            self.chart_view.show_action_progress("Running backtest…")
            
            # Get current params from stats panel (user may have edited them)
            seller_params, bt_params = self.stats_panel.get_current_params()
            
            # Rebuild features with current params
            feats = build_features(self.current_data[['open', 'high', 'low', 'close', 'volume']], 
                                   seller_params, self.current_tf)
            self.current_data = feats
            self.chart_view.feats = feats
            self.chart_view.params = seller_params
            
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
            self.statusBar().showMessage(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            self.progress.setVisible(False)
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
        self.statusBar().showMessage("Results cleared")
    
    def update_progress(self, current: int, total: int, message: str):
        """
        Update progress bar with optimization progress.
        
        Args:
            current: Current generation number
            total: Total number of generations
            message: Status message
        """
        if total > 0:
            self.progress.setVisible(True)
            self.progress.setMaximum(total)
            self.progress.setValue(current)
            percentage = (current / total * 100) if total > 0 else 0
            self.statusBar().showMessage(f"{message} ({percentage:.0f}%)")
        else:
            self.progress.setVisible(False)
            self.statusBar().showMessage(message)
        
        # Hide progress bar after completion
        if current >= total and total > 0:
            QTimer.singleShot(3000, lambda: self.progress.setVisible(False))
    
    def on_optimization_step_complete(self, best_individual, backtest_result):
        """Handle optimization step completion and update chart."""
        print(f"\n🎯 Displaying winning strategy on chart...")
        
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
            
            # Update status bar
            metrics = backtest_result['metrics']
            self.statusBar().showMessage(
                f"🏆 Gen {best_individual.generation} Best: "
                f"{metrics['n']} trades | "
                f"Win rate: {metrics['win_rate']:.1%} | "
                f"Avg R: {metrics['avg_R']:.2f} | "
                f"Fitness: {best_individual.fitness:.4f}"
            )
            
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
        
        # Update stats panel if it exists
        if hasattr(self, 'stats_panel'):
            self.stats_panel.load_params(seller_params, backtest_params)
        
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
    
    def save_window_state(self):
        """Save window and splitter state to settings."""
        from config.settings import SettingsManager
        
        try:
            settings_dict = {
                'window_width': self.width(),
                'window_height': self.height(),
                'splitter_left': self.centralWidget().sizes()[0],
                'splitter_right': self.centralWidget().sizes()[1],
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
            
            # Restore splitter sizes
            if settings.splitter_left and settings.splitter_right:
                self.centralWidget().setSizes([settings.splitter_left, settings.splitter_right])
            
            # Note: Chart view range restored when data is loaded
        except Exception as e:
            print(f"Warning: Could not restore window state: {e}")
    
    async def try_load_cached_data(self):
        """Try to load cached data from last session."""
        try:
            # Get last download parameters from settings
            ticker = settings.last_ticker
            from_ = settings.last_date_from
            to = settings.last_date_to
            tf_mult = int(settings.timeframe)
            tf_unit = settings.timeframe_unit
            
            # Check if cache exists
            has_cache = self.cache.has_cached_data(ticker, from_, to, 1, "minute")
            
            if not has_cache:
                self.statusBar().showMessage(
                    "No cached data found - Click Settings to download data"
                )
                return
            
            # Load from cache
            self.statusBar().showMessage("Loading cached data...")
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            
            # Fetch data (will use cache automatically)
            df = await self.data_provider.fetch(ticker, self.current_tf, from_, to)
            
            if df is None or len(df) == 0:
                self.statusBar().showMessage(
                    "Cache empty - Click Settings to download data"
                )
                self.progress.setVisible(False)
                return
            
            # Load strategy params from settings
            seller_params = SellerParams(
                ema_fast=settings.strategy_ema_fast,
                ema_slow=settings.strategy_ema_slow,
                z_window=settings.strategy_z_window,
                vol_z=settings.strategy_vol_z,
                tr_z=settings.strategy_tr_z,
                cloc_min=settings.strategy_cloc_min,
                atr_window=settings.strategy_atr_window,
            )
            
            # Build features
            feats = build_features(df, seller_params, self.current_tf)
            
            # Update chart
            self.chart_view.feats = feats
            self.chart_view.params = seller_params
            self.chart_view.render_candles(feats)
            
            # Store data
            self.current_data = feats
            
            # Load backtest params and pass to stats panel
            bt_params = BacktestParams(
                atr_stop_mult=settings.backtest_atr_stop_mult,
                reward_r=settings.backtest_reward_r,
                max_hold=settings.backtest_max_hold,
                fee_bp=settings.backtest_fee_bp,
                slippage_bp=settings.backtest_slippage_bp,
            )
            
            self.stats_panel.load_params_from_settings(seller_params, bt_params)
            self.stats_panel.set_current_data(feats, self.current_tf)
            
            # Enable backtest button
            self.backtest_action.setEnabled(True)
            
            # Update status
            signals = feats['exhaustion'].sum()
            date_range = f"{feats.index[0].date()} to {feats.index[-1].date()}"
            self.statusBar().showMessage(
                f"✓ Loaded {len(feats)} bars from cache ({date_range}) | "
                f"{signals} signals | Ready to backtest"
            )
            
            # Update chart info
            self.chart_view.info_label.setText(
                f"📊 {ticker} {self.current_tf.value} | "
                f"Date range: {date_range}\n"
                f"Signals detected: {signals} | Data loaded from cache"
            )
            
            print(f"✓ Auto-loaded cached data: {len(feats)} bars, {signals} signals")
            
        except Exception as e:
            print(f"⚠ Could not auto-load cached data: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(
                "Could not load cached data - Click Settings to download fresh data"
            )
        
        finally:
            self.progress.setVisible(False)
    
    def closeEvent(self, event):
        """Save state when window is closed."""
        self.save_window_state()
        event.accept()
    
    async def initialize(self):
        """Async initialization after window is shown."""
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
                60: Timeframe.m60,
            }
            tf_mult = int(settings.timeframe)
            
            # If configured timeframe not supported, default to 15m
            if tf_mult not in tf_map:
                print(f"⚠ Configured timeframe {tf_mult}m not supported, defaulting to 15m")
                tf_mult = 15
            
            self.current_tf = tf_map[tf_mult]
            self.chart_view.set_timeframe(self.current_tf)
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
