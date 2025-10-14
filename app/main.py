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
from strategy.seller_exhaustion import build_features
from core.models import Timeframe
from backtest.engine import run_backtest


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
        
        # Connect optimization signal
        self.stats_panel.optimization_step_complete.connect(self.on_optimization_step_complete)
        
        # Set initial sizes (chart gets 70%, stats gets 30%)
        splitter.setSizes([1120, 480])
        
        self.setCentralWidget(splitter)
        
        # Status bar with progress
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Ready - Click Settings to download data")
    
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
            tf_map = {1: Timeframe.m1, 3: Timeframe.m3, 5: Timeframe.m5, 10: Timeframe.m10, 15: Timeframe.m15}
            tf = tf_map.get(int(tf_mult), Timeframe.m15)
            
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
        
        try:
            self.statusBar().showMessage("Running backtest...")
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.backtest_action.setEnabled(False)
            
            # Get current params from stats panel (user may have edited them)
            seller_params, bt_params = self.stats_panel.get_current_params()
            
            # Rebuild features with current params
            feats = build_features(self.current_data[['open', 'high', 'low', 'close', 'volume']], 
                                   seller_params, self.current_tf)
            
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
    
    def clear_results(self):
        """Clear backtest results."""
        self.stats_panel.clear()
        self.chart_view.backtest_result = None
        self.chart_view.update_trade_table(None)
        if self.current_data is not None:
            self.chart_view.render_candles(self.current_data)
        self.statusBar().showMessage("Results cleared")
    
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
    
    def closeEvent(self, event):
        """Save state when window is closed."""
        self.save_window_state()
        event.accept()
    
    async def initialize(self):
        """Async initialization after window is shown."""
        # Restore window state
        self.restore_window_state()
        
        # Note: Data auto-load removed - user must explicitly download
        # This prevents unwanted API calls on every startup


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
