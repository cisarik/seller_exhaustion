import asyncio
import logging
from typing import Optional
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, 
    QSplitter, QTableWidget, QTableWidgetItem, 
    QHeaderView, QGroupBox, QFileDialog, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QBrush, QFont
import pyqtgraph as pg
from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem
from pyqtgraph import InfiniteLine, TextItem
from strategy.seller_exhaustion import SellerParams, build_features
from indicators.local import sma, rsi, macd
from core.models import Timeframe
from core.logging_utils import get_logger


logger = get_logger(__name__)


# Fibonacci level colors (rainbow gradient)
FIB_COLORS = {
    0.382: '#2196F3',  # Blue - first level
    0.500: '#00BCD4',  # Cyan - mid level
    0.618: '#FFD700',  # GOLD - Golden Ratio (prominent!)
    0.786: '#FF9800',  # Orange - aggressive
    1.000: '#F44336',  # Red - full retracement
}


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick chart item for PyQtGraph."""
    
    def __init__(self, data, x_values=None):
        pg.GraphicsObject.__init__(self)
        self.data = data
        # x_values should be 1D array-like of the same length as data, typically epoch seconds
        self.x_values = x_values
        self.generatePicture()
    
    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        
        # Determine candle body width based on median spacing of x_values
        if self.x_values is not None and len(self.x_values) > 1:
            diffs = np.diff(self.x_values)
            step = float(np.median(diffs)) if len(diffs) > 0 else 1.0
            w = step * 0.8

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Candlestick spacing | candles=%d | median_step=%.3f s | width=%.3f s | first_x=%s | first_diffs=%s",
                    len(self.x_values),
                    step,
                    w,
                    list(self.x_values[:3]),
                    list(diffs[:3]),
                )
        else:
            w = 0.6

        for i, (t, row) in enumerate(self.data.iterrows()):
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            x = float(self.x_values[i]) if self.x_values is not None else float(i)
            
            # Color based on direction
            if close_price > open_price:
                p.setPen(pg.mkPen('#4caf50'))
                p.setBrush(pg.mkBrush('#4caf50'))
            else:
                p.setPen(pg.mkPen('#f44336'))
                p.setBrush(pg.mkBrush('#f44336'))
            
            # Draw high-low line (wick)
            p.drawLine(pg.QtCore.QPointF(x, low_price), pg.QtCore.QPointF(x, high_price))

            # Draw open-close box (body). Use normalized coordinates so down candles
            # have a visible body instead of collapsing to a line.
            top = max(open_price, close_price)
            bottom = min(open_price, close_price)
            height = max(top - bottom, 1e-6)
            rect = pg.QtCore.QRectF(x - w / 2, bottom, w, height)
            p.drawRect(rect)
        
        p.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class CandleChartWidget(QWidget):
    """Main widget for displaying candlestick charts with indicators."""
    
    # Signal for immediate status bar reset
    reset_status_bar_signal = Signal()
    
    # Signal for delayed status bar reset
    delayed_reset_status_bar_signal = Signal()
    
    # Signal emitted when chart is clicked (for future features)
    chart_clicked = Signal(float, float)
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.feats = None
        self.params = SellerParams()
        self.tf = Timeframe.m15
        self.backtest_result = None
        self.backtest_params = None  # Store backtest params for Fib target
        self.selected_trade_idx = None  # Track selected trade for Fib display
        self.indicator_config = {
            'ema_fast': True,
            'ema_slow': True,
            'sma': False,
            'rsi': False,
            'macd': False,
            'volume': False,
            'signals': True,
            'entries': True,
            'exits': True,
            'fib_retracements': True,  # Show Fibonacci retracements
            'fib_0382': True,  # 38.2% level
            'fib_0500': True,  # 50% level
            'fib_0618': True,  # 61.8% Golden Ratio
            'fib_0786': True,  # 78.6% level
            'fib_1000': True,  # 100% level
        }
        
        self.initUI()
        
        # Connect signals for status bar reset
        self.reset_status_bar_signal.connect(self._reset_status_bar)
        self.delayed_reset_status_bar_signal.connect(self._reset_status_bar)
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Vertical splitter for chart and trade history
        splitter = QSplitter(Qt.Vertical)
        
        # Top: Chart widget
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        # Chart with DateAxis on bottom
        axis = DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': axis})
        self.plot_widget.setBackground('#0f1a12')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Price', color='#e8f5e9')
        self.plot_widget.setLabel('bottom', '', color='#e8f5e9')  # No label on x-axis
        chart_layout.addWidget(self.plot_widget)
        
        splitter.addWidget(chart_widget)
        
        # Bottom: Trade History Table
        trades_widget = QWidget()
        trades_layout = QVBoxLayout(trades_widget)
        trades_layout.setContentsMargins(0, 0, 0, 0)
        trades_layout.setSpacing(0)
        
        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(6)
        self.trade_table.setHorizontalHeaderLabels(
            ["Entry", "Exit", "Entry", "Exit", "PnL", "R"]
        )
        # Make table selectable by row (for trade detail feature)
        self.trade_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.trade_table.setSelectionMode(QTableWidget.SingleSelection)
        self.trade_table.itemSelectionChanged.connect(self._on_trade_selected)
        
        # Set tooltips for headers
        header_tooltips = {
            0: "Entry time (date + time) - click row for trade detail",
            1: "Exit time (date + time) - click row for trade detail",
            2: "Entry price (contract price)",
            3: "Exit price (contract price)",
            4: "Profit/Loss in contract units (negative = loss)",
            5: "R-Multiple: Risk:Reward ratio (1R = risked amount, 2R = 2x risked amount)"
        }
        for col, tooltip in header_tooltips.items():
            self.trade_table.horizontalHeaderItem(col).setToolTip(tooltip)
        
        self.trade_table.setMaximumHeight(200)
        self.trade_table.horizontalHeader().setStretchLastSection(True)
        self.trade_table.setStyleSheet("""
            QTableWidget {
                background-color: #0f1a12;
                color: #e8f5e9;
                border: 1px solid #2f5c39;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #000000;
                color: #4caf50;
                padding: 4px;
                border: 1px solid #2f5c39;
                font-weight: bold;
            }
        """)
        
        # Set column widths - Entry and Exit time columns need to be wider for full datetime display
        self.trade_table.setColumnWidth(0, 160)  # Entry time column - "2025-01-17 13:12"
        self.trade_table.setColumnWidth(1, 160)  # Exit time column - "2025-01-17 13:12"
        self.trade_table.setColumnWidth(2, 90)   # Entry price
        self.trade_table.setColumnWidth(3, 90)   # Exit price
        self.trade_table.setColumnWidth(4, 90)   # PnL
        self.trade_table.setColumnWidth(5, 70)   # R
        
        trades_layout.addWidget(self.trade_table)
        
        splitter.addWidget(trades_widget)
        
        layout.addWidget(splitter)
        
        # Status bar at the bottom (after trade history)
        # Create a container widget for the entire status bar with black background
        status_container = QWidget()
        status_container.setProperty("role", "statusbar")  # Apply black background styling to container
        
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(16, 12, 16, 12)  # Match padding from theme
        status_layout.setSpacing(8)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("background: transparent; color: #4caf50; font-size: 14px; font-weight: bold; padding: 0;")
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.status_label.setCursor(Qt.ArrowCursor)  # Default cursor
        self.status_label.mousePressEvent = self._on_status_clicked
        self._coach_analysis = None  # Store latest coach analysis
        self._status_is_clickable = False
        status_layout.addWidget(self.status_label, stretch=1)
        
        # Progress bar for actions - aligned to the right
        self.action_progress = QProgressBar()
        self.action_progress.setVisible(False)
        self.action_progress.setFixedWidth(220)
        self.action_progress.setStyleSheet("""
            QProgressBar {
                background-color: #000000;
                border: 1px solid #2f5c39;
                color: #4caf50;
                text-align: center;
                border-radius: 3px;
                min-height: 18px;
                max-height: 18px;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 2px;
            }
        """)
        status_layout.addWidget(self.action_progress)
        
        layout.addWidget(status_container)
    
    def set_timeframe(self, timeframe: Timeframe):
        """Set active timeframe (read-only in main chart)."""
        self.tf = timeframe
        self.status_label.setText(f"{timeframe.value.upper()} timeframe ready")
    
    def show_action_progress(self, message: Optional[str] = None):
        """Display inline progress while async work runs."""
        if message:
            self.status_label.setText(message)
        self.action_progress.setVisible(True)
        self.action_progress.setRange(0, 0)
    
    def hide_action_progress(self, message: Optional[str] = None):
        """Hide inline progress indicator."""
        self.action_progress.setVisible(False)
        if message:
            self.status_label.setText(message)
    
    def set_coach_status(self, message: str, analysis=None, is_recommendation: bool = False):
        """
        Update status bar with coach message.

        Args:
            message: Status message to display
            analysis: CoachAnalysis object (if recommendations applied)
            is_recommendation: True if recommendations were applied (shows green bg + clickable)
        """
        self._coach_analysis = analysis
        self._status_is_clickable = is_recommendation and analysis is not None

        if is_recommendation:
            # Green background, clickable - recommendations applied
            self.status_label.setStyleSheet("""
                background-color: #2f5c39;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 12px;
                border-radius: 4px;
                border: 2px solid #4caf50;
            """)
            self.status_label.setCursor(Qt.PointingHandCursor)
        else:
            # Check if this is a coach status message to determine background color
            if message.startswith('🤖') or 'Coach' in message:
                # Coach-related message - use appropriate background
                if 'OpenAI' in message or '🤖' in message:
                    # OpenAI Agents mode - blue background
                    self.status_label.setStyleSheet("""
                        background-color: #1e3a5f;
                        color: #ffffff;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 6px 10px;
                        border-radius: 4px;
                        border: 1px solid #2196f3;
                    """)
                elif 'Classic' in message or '🧠' in message:
                    # Classic Coach mode - green background
                    self.status_label.setStyleSheet("""
                        background-color: #2e5c39;
                        color: #ffffff;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 6px 10px;
                        border-radius: 4px;
                        border: 1px solid #4caf50;
                    """)
                else:
                    # General coach message - neutral background
                    self.status_label.setStyleSheet("""
                        background-color: #3a3a3a;
                        color: #ffffff;
                        font-size: 14px;
                        font-weight: bold;
                        padding: 6px 10px;
                        border-radius: 4px;
                        border: 1px solid #666;
                    """)
            else:
                # Normal transparent background for non-coach messages
                self.status_label.setStyleSheet("""
                    background: transparent;
                    color: #4caf50;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 0;
                """)
            self.status_label.setCursor(Qt.ArrowCursor)

        self.status_label.setText(message)
    
    def set_coach_tool_status(self, tool_name: str, reason: str = "", result: str = ""):
        """
        Update status bar with coach tool execution details.

        Args:
            tool_name: Name of the tool being called
            reason: Reason for calling the tool
            result: Result or response from the tool
        """
        # Build status message with better UX
        if tool_name.startswith('OpenAI_'):
            # OpenAI Agents mode - show more detailed progress
            tool_display = tool_name.replace('OpenAI_', '').replace('_', ' ').title()
            message_parts = [f"🤖 OpenAI Coach: {tool_display}"]

            if reason:
                # Show reason in a more readable format
                if reason == "analyzing_population":
                    message_parts.append("(analyzing population)")
                elif reason == "applying_recommendations":
                    message_parts.append("(applying changes)")
                else:
                    message_parts.append(f"({reason})")

            if result:
                # Truncate long results but show key info
                if len(result) > 80:
                    result = result[:77] + "..."
                message_parts.append(f"→ {result}")

            message = " ".join(message_parts)

            # Set blue background for OpenAI tool execution
            self.status_label.setStyleSheet("""
                background-color: #1e3a5f;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                padding: 6px 10px;
                border-radius: 3px;
                border: 1px solid #2196f3;
            """)

        else:
            # Classic Coach mode - simpler display
            message_parts = [f"🧠 Coach: {tool_name}"]

            if reason:
                message_parts.append(f"({reason})")

            if result:
                # Truncate long results
                if len(result) > 100:
                    result = result[:97] + "..."
                message_parts.append(f"→ {result}")

            message = " ".join(message_parts)

            # Set green background for Classic Coach
            self.status_label.setStyleSheet("""
                background-color: #2e5c39;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                padding: 6px 10px;
                border-radius: 3px;
                border: 1px solid #4caf50;
            """)

        self.status_label.setCursor(Qt.ArrowCursor)
        self._status_is_clickable = False
        self._coach_analysis = None

        self.status_label.setText(message)

        # Schedule status bar reset using Signal or QTimer
        if tool_name.startswith('✅ Coach completed'):
            # Hide status bar immediately when coach analysis is completed using Signal
            self.reset_status_bar_signal.emit()
            # print(f"🔍 DEBUG: Hiding blue status bar immediately - coach analysis completed")
        else:
            # Use thread-safe method to schedule status bar reset
            from PySide6.QtCore import QMetaObject, Q_ARG, Qt
            # Schedule status bar reset in main thread after 5 seconds
            QMetaObject.invokeMethod(
                self,
                "_schedule_status_bar_reset",
                Qt.QueuedConnection,
                Q_ARG(int, 5000)  # 5 seconds delay
            )
    
    def _schedule_status_bar_reset(self, delay_ms: int):
        """Schedule status bar reset with delay (called from main thread)."""
        from PySide6.QtCore import QTimer
        QTimer.singleShot(delay_ms, self._reset_status_bar)
    
    def _reset_status_bar(self):
        """Reset status bar to normal appearance."""
        self.status_label.setStyleSheet("""
            background-color: #2b2b2b;
            color: #ffffff;
            font-size: 13px;
            font-weight: normal;
            padding: 6px 10px;
            border-radius: 3px;
            border: 1px solid #555;
        """)
        self.status_label.setCursor(Qt.ArrowCursor)
        self._status_is_clickable = False
        self._coach_analysis = None
    
    def _on_status_clicked(self, event):
        """Handle status label click - show coach recommendations if available."""
        if self._status_is_clickable and self._coach_analysis:
            from app.widgets.coach_recommendations_dialog import show_coach_recommendations
            show_coach_recommendations(self._coach_analysis, parent=self)
    
    def set_indicator_config(self, config):
        """Update indicator configuration and re-render."""
        self.indicator_config = config
        if self.feats is not None:
            self.render_candles(self.feats, self.backtest_result)
    
    def set_backtest_result(self, result, backtest_params=None):
        """Set backtest results and re-render with trade markers."""
        self.backtest_result = result
        self.backtest_params = backtest_params  # Store for Fib target
        if result and 'trades' in result:
            self.update_trade_table(result['trades'])
        if self.feats is not None:
            self.render_candles(self.feats, result)
    
    def render_candles(self, df: pd.DataFrame, backtest_result=None):
        """Render candlesticks, indicators, and trade markers."""
        self.plot_widget.clear()
        
        # Store full dataframe as feats for Fibonacci access
        self.feats = df
        
        # DEBUG: Print data info
        logger.debug(
            "Render chart | bars=%d | range=%s -> %s | timeframe=%s | columns=%s",
            len(df),
            df.index[0],
            df.index[-1],
            self.tf.value,
            list(df.columns),
        )
        
        # Check actual spacing between bars
        if len(df) > 1 and logger.isEnabledFor(logging.DEBUG):
            time_diffs = df.index.to_series().diff()[1:6]  # First 5 differences
            formatted_diffs = [
                f"{td} ({td.total_seconds()/60:.0f} minutes)" for td in time_diffs
            ]
            logger.debug("First 5 time deltas: %s", formatted_diffs)
            
            median_minutes = df.index.to_series().diff().median().total_seconds() / 60
            logger.debug("Median bar spacing: %.0f minutes", median_minutes)
        
        # Sample if too many candles (for performance)
        # DISABLED: User wants to see all data
        original_df = df
        # if len(df) > 5000:
        #     df = df.iloc[-5000:]
        #     print(f"   Sampled to last 5000 bars: {df.index[0]} to {df.index[-1]}")
        
        # Store for index mapping
        self.displayed_df = df
        
        # Map datetime index to epoch seconds for DateAxis
        # Keep a mapping array to plot with numerical x values (epoch seconds)
        epoch_seconds = (df.index.view('int64') // 1_000_000_000).astype(np.int64)
        self._epoch_seconds = epoch_seconds
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Epoch range: %s (%s) -> %s (%s)",
                epoch_seconds[0],
                pd.Timestamp(epoch_seconds[0], unit='s', tz='UTC'),
                epoch_seconds[-1],
                pd.Timestamp(epoch_seconds[-1], unit='s', tz='UTC'),
            )

        # Create candlestick item drawn in epoch-seconds X coordinates for proper DateAxis labeling
        candles = CandlestickItem(df[['open', 'high', 'low', 'close']], x_values=epoch_seconds)
        self.plot_widget.addItem(candles)
        
        x = epoch_seconds
        
        # Add EMA Fast
        if self.indicator_config.get('ema_fast', True) and 'ema_f' in df.columns:
            valid_f = df['ema_f'].notna()
            self.plot_widget.plot(
                x[valid_f], df['ema_f'][valid_f].values,
                pen=pg.mkPen('#00bcd4', width=2), name='EMA Fast'
            )
        
        # Add EMA Slow
        if self.indicator_config.get('ema_slow', True) and 'ema_s' in df.columns:
            valid_s = df['ema_s'].notna()
            self.plot_widget.plot(
                x[valid_s], df['ema_s'][valid_s].values,
                pen=pg.mkPen('#ff9800', width=2), name='EMA Slow'
            )
        
        # Add SMA
        if self.indicator_config.get('sma', False):
            if 'sma_100' not in df.columns:
                df['sma_100'] = sma(df['close'], 100)
            valid_sma = df['sma_100'].notna()
            self.plot_widget.plot(
                x[valid_sma], df['sma_100'][valid_sma].values,
                pen=pg.mkPen('#9c27b0', width=2), name='SMA 100'
            )
        
        # Mark exhaustion signals (yellow triangles)
        if self.indicator_config.get('signals', True) and 'exhaustion' in df.columns:
            signals = df[df['exhaustion'] == True]
            if len(signals) > 0:
                signal_x = signals.index.view('int64') // 1_000_000_000
                signal_y = signals['low'].values * 0.998  # Slightly below low
                scatter = pg.ScatterPlotItem(
                    x=signal_x, y=signal_y,
                    pen=pg.mkPen('#ffeb3b', width=2),
                    brush=pg.mkBrush('#ffeb3b'),
                    size=12, symbol='t1'
                )
                self.plot_widget.addItem(scatter)
        
        # Mark trade entries as BALLS (entry point, sized by PnL)
        if backtest_result and len(backtest_result['trades']) > 0:
            trades = backtest_result['trades']
            
            if self.indicator_config.get('entries', True):
                self.render_trade_balls(trades, original_df)
        
        # Add legend (only if not already present)
        if not hasattr(self.plot_widget.plotItem, 'legend') or self.plot_widget.plotItem.legend is None:
            self.plot_widget.addLegend()
    
    def update_trade_table(self, trades: Optional[pd.DataFrame]):
        """Update trade history table with backtest results."""
        self.trade_table.setRowCount(0)
        self.selected_trade_idx = None  # Reset selection
        
        if trades is None or len(trades) == 0:
            return
        
        self.trade_table.setRowCount(len(trades))
        
        for i, (idx, trade) in enumerate(trades.iterrows()):
            # Entry Time - full datetime format: "2025-01-17 13:12"
            entry_time = pd.Timestamp(trade['entry_ts']).strftime('%Y-%m-%d %H:%M') if 'entry_ts' in trade else ""
            self.trade_table.setItem(i, 0, QTableWidgetItem(entry_time))
            
            # Exit Time - full datetime format: "2025-01-17 13:12"
            exit_time = pd.Timestamp(trade['exit_ts']).strftime('%Y-%m-%d %H:%M') if 'exit_ts' in trade else ""
            self.trade_table.setItem(i, 1, QTableWidgetItem(exit_time))
            
            # Entry Price
            entry_price = f"{trade.get('entry', 0):.4f}"
            self.trade_table.setItem(i, 2, QTableWidgetItem(entry_price))
            
            # Exit Price
            exit_price = f"{trade.get('exit', 0):.4f}"
            self.trade_table.setItem(i, 3, QTableWidgetItem(exit_price))
            
            # PnL (color-coded)
            pnl = trade.get('pnl', 0)
            pnl_text = f"{pnl:.6f}"
            pnl_item = QTableWidgetItem(pnl_text)
            
            if pnl > 0:
                pnl_item.setForeground(QBrush(QColor('#4caf50')))  # Green
            elif pnl < 0:
                pnl_item.setForeground(QBrush(QColor('#f44336')))  # Red
            else:
                pnl_item.setForeground(QBrush(QColor('#ffeb3b')))  # Yellow
            
            self.trade_table.setItem(i, 4, pnl_item)
            
            # R-Multiple
            r_multiple = f"{trade.get('R', 0):.2f}"
            self.trade_table.setItem(i, 5, QTableWidgetItem(r_multiple))
    
    def _on_trade_selected(self):
        """Handle trade row selection - update chart with trade detail markers."""
        selected_rows = self.trade_table.selectionModel().selectedRows()
        
        if len(selected_rows) == 0:
            # No selection
            self.selected_trade_idx = None
            logger.debug("Trade detail cleared")
        else:
            # Get selected row index
            row_idx = selected_rows[0].row()
            self.selected_trade_idx = row_idx
            
            # Get the selected trade from backtest results
            if self.backtest_result and 'trades' in self.backtest_result:
                trades = self.backtest_result['trades']
                if row_idx < len(trades):
                    trade = trades.iloc[row_idx]
                    entry_time = pd.Timestamp(trade['entry_ts']).strftime('%Y-%m-%d %H:%M')
                    exit_time = pd.Timestamp(trade['exit_ts']).strftime('%Y-%m-%d %H:%M')
                    logger.debug(
                        "Trade detail #%d | Entry %s @ %.4f | Exit %s @ %.4f | PnL=%.6f | R=%.2f",
                        row_idx + 1,
                        entry_time,
                        trade['entry'],
                        exit_time,
                        trade['exit'],
                        trade['pnl'],
                        trade['R'],
                    )
        
        # Re-render chart with updated selection
        if self.feats is not None:
            self.render_candles(self.feats, self.backtest_result)
    
    def render_trade_balls(self, trades: pd.DataFrame, df: pd.DataFrame):
        """
        Render trades as BALLS positioned at entry points:
        - Green balls for profitable trades (size proportional to profit)
        - Red balls for losing trades (size proportional to loss)
        - SEMI-TRANSPARENT if selected (arrows rendered on top)
        
        For selected trade, render green UP arrow at entry and red DOWN arrow at exit.
        
        Args:
            trades: DataFrame of all trades
            df: Full OHLCV DataFrame for timestamp lookups
        """
        if len(trades) == 0:
            return
        
        # Calculate ball sizes based on PnL magnitude
        pnl_values = trades['pnl'].values
        
        # Normalize PnL to reasonable ball sizes (20-80 range)
        max_abs_pnl = max(abs(pnl_values.max()), abs(pnl_values.min()), 0.01)
        
        normal_balls = []  # Full opacity
        selected_ball = None  # Semi-transparent
        
        for i, (_, trade) in enumerate(trades.iterrows()):
            entry_ts = pd.Timestamp(trade['entry_ts'])
            
            # Check if timestamp exists in dataframe
            if entry_ts not in df.index:
                continue
            
            # Position at entry
            entry_x = int(entry_ts.value // 1_000_000_000)
            entry_y = trade['entry']
            
            # Calculate ball size (proportional to |PnL|)
            pnl = trade['pnl']
            size = (20 + (abs(pnl) / max_abs_pnl) * 60) * 0.33  # Range: 6.6-26.4 (1/3 of original)
            
            # Determine color based on profit/loss
            if pnl > 0:
                # Profit = GREEN
                color = '#4CAF50'
                border_color = '#2E7D32'
                border_width = 2
            else:
                # Loss = RED
                color = '#F44336'
                border_color = '#C62828'
                border_width = 2
            
            ball_info = {
                'x': entry_x,
                'y': entry_y,
                'size': size,
                'color': color,
                'border_color': border_color,
                'border_width': border_width,
                'trade': trade
            }
            
            # Separate selected trade from others
            if i == self.selected_trade_idx:
                selected_ball = ball_info
            else:
                normal_balls.append(ball_info)
        
        # Render normal (non-selected) balls with full opacity
        if normal_balls:
            x_coords = [b['x'] for b in normal_balls]
            y_coords = [b['y'] for b in normal_balls]
            sizes = [b['size'] for b in normal_balls]
            
            # Create brushes and pens for each ball
            brushes = [pg.mkBrush(b['color']) for b in normal_balls]
            pens = [pg.mkPen(b['border_color'], width=b['border_width']) for b in normal_balls]
            
            scatter = pg.ScatterPlotItem(
                x=x_coords,
                y=y_coords,
                size=sizes,
                pen=pens,
                brush=brushes,
                symbol='o',
                pxMode=True  # Size in pixels
            )
            self.plot_widget.addItem(scatter)
        
        # Render selected ball with SEMI-TRANSPARENT (50% opacity)
        if selected_ball:
            from PySide6.QtGui import QColor
            
            # Create semi-transparent color for selected ball (50% = 128/255)
            selected_color = QColor(selected_ball['color'])
            selected_color.setAlpha(128)
            
            selected_border = QColor(selected_ball['border_color'])
            selected_border.setAlpha(128)
            
            brush = pg.mkBrush(selected_color)
            pen = pg.mkPen(selected_border, width=selected_ball['border_width'])
            
            scatter = pg.ScatterPlotItem(
                x=[selected_ball['x']],
                y=[selected_ball['y']],
                size=[selected_ball['size']],
                pen=pen,
                brush=brush,
                symbol='o',
                pxMode=True
            )
            self.plot_widget.addItem(scatter)
            
            # Render arrows on top of semi-transparent ball
            self._render_trade_detail_arrows(selected_ball['trade'], df)
        
        selected_str = (
            f" (#{self.selected_trade_idx + 1} semi-transparent with arrows)"
            if self.selected_trade_idx is not None
            else ""
        )
        logger.debug("Rendered %d trade balls%s", len(normal_balls) + (1 if selected_ball else 0), selected_str)
    
    def _render_trade_detail_arrows(self, trade: pd.Series, df: pd.DataFrame):
        """
        Render entry and exit arrows for selected trade:
        - Small GREEN triangle pointing UP at entry price
        - Small RED triangle pointing DOWN at exit price
        - Display entry and exit PRICES on black background
        - Rendered ON TOP of the chart (after balls)
        
        Args:
            trade: Single trade row from backtest results
            df: Full OHLCV DataFrame for timestamp lookups
        """
        entry_ts = pd.Timestamp(trade['entry_ts'])
        exit_ts = pd.Timestamp(trade['exit_ts'])
        
        entry_x = int(entry_ts.value // 1_000_000_000)
        entry_y = trade['entry']
        
        exit_x = int(exit_ts.value // 1_000_000_000)
        exit_y = trade['exit']
        
        # Render Fibonacci retracement levels FIRST (so arrows appear on top)
        self._render_fibonacci_levels(trade, df, entry_x, exit_x, entry_y)
        
        # Entry arrow: GREEN triangle pointing UP at entry price
        # Small size (25px) for clear visibility without dominating
        entry_scatter = pg.ScatterPlotItem(
            x=[entry_x],
            y=[entry_y],
            size=25,  # Smaller, cleaner size
            pen=pg.mkPen('#00FF00', width=2),  # Bright green outline
            brush=pg.mkBrush('#00FF00'),  # Bright green fill
            symbol='t1',  # UP triangle (t1 explicitly points up)
            pxMode=True
        )
        self.plot_widget.addItem(entry_scatter)
        entry_scatter.setZValue(5000)  # HIGHEST z-value - set AFTER adding to plot
        
        # Exit arrow: RED triangle pointing DOWN at exit price
        # Small size (25px) for clear visibility without dominating
        exit_scatter = pg.ScatterPlotItem(
            x=[exit_x],
            y=[exit_y],
            size=25,  # Smaller, cleaner size
            pen=pg.mkPen('#FF0000', width=2),  # Bright red outline
            brush=pg.mkBrush('#FF0000'),  # Bright red fill
            symbol='t3',  # DOWN triangle (t3 explicitly points down)
            pxMode=True
        )
        self.plot_widget.addItem(exit_scatter)
        exit_scatter.setZValue(5000)  # HIGHEST z-value - set AFTER adding to plot
        
        # Add price labels RIGHT NEXT TO triangles with colored text
        # Green text for entry, red text for exit, on solid black background
        # NO arrow symbols - just the price
        entry_price_text = f"{entry_y:.4f}"
        entry_label = TextItem(
            text=entry_price_text,
            color='#00FF00',  # GREEN text for entry
            anchor=(0.5, 1.2),  # Below the triangle
            fill=(0, 0, 0, 255)  # SOLID black background
        )
        entry_font = QFont('Courier', 12)  # Bold readable font
        entry_font.setBold(True)
        entry_label.setFont(entry_font)
        entry_label.setPos(entry_x, entry_y - 0.002)  # Just below triangle
        self.plot_widget.addItem(entry_label)
        entry_label.setZValue(6000)  # Even higher than arrows
        
        exit_price_text = f"{exit_y:.4f}"
        exit_label = TextItem(
            text=exit_price_text,
            color='#FF0000',  # RED text for exit
            anchor=(0.5, -0.2),  # Above the triangle
            fill=(0, 0, 0, 255)  # SOLID black background
        )
        exit_font = QFont('Courier', 12)  # Bold readable font
        exit_font.setBold(True)
        exit_label.setFont(exit_font)
        exit_label.setPos(exit_x, exit_y + 0.002)  # Just above triangle
        self.plot_widget.addItem(exit_label)
        exit_label.setZValue(6000)  # Even higher than arrows
        
        logger.debug(
            "Rendered trade detail | Entry ↑ at %.4f | Exit ↓ at %.4f",
            entry_y,
            exit_y,
        )
    def _render_fibonacci_levels(self, trade: pd.Series, df: pd.DataFrame, entry_x: int, exit_x: int, entry_y: float):
        """
        Render Fibonacci retracement levels for selected trade.
        
        Shows semi-transparent zones and horizontal lines for Fibonacci levels
        from entry (0) to swing high (1.0).
        
        Args:
            trade: Selected trade data
            df: Full OHLCV DataFrame
            entry_x: Entry timestamp in epoch seconds
            exit_x: Exit timestamp in epoch seconds
            entry_y: Entry price
        """
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QPen, QBrush
        
        # Find swing high before entry
        entry_ts = pd.Timestamp(trade['entry_ts'])
        
        # Look back from entry to find recent swing high (last 96 bars = 24h on 15m)
        lookback = 96
        entry_idx = df.index.get_loc(entry_ts) if entry_ts in df.index else None
        
        if entry_idx is None or entry_idx < lookback:
            logger.debug("Cannot render Fibonacci: entry not found or insufficient history")
            return
        
        # Get slice before entry
        lookback_slice = df.iloc[max(0, entry_idx - lookback):entry_idx]
        
        if len(lookback_slice) == 0:
            return
        
        # Find swing high (highest high in lookback period)
        swing_high = lookback_slice['high'].max()
        swing_high_idx = lookback_slice['high'].idxmax()
        swing_high_x = int(swing_high_idx.value // 1_000_000_000)
        
        # Calculate Fibonacci levels from entry (0) to swing high (1.0)
        fib_range = swing_high - entry_y
        
        if fib_range <= 0:
            logger.debug("Invalid Fibonacci range: swing high not above entry")
            return
        
        # Get the Fib target from backtest params (default to 0.618)
        fib_target = 0.618  # Default
        if self.backtest_params and hasattr(self.backtest_params, 'fib_target_level'):
            fib_target = self.backtest_params.fib_target_level
        
        # Define Fibonacci levels with colors
        # Format: (ratio, price, label_base, line_color, label_color, alpha, show_label)
        fib_ratios = [
            (0.0, 'Entry', '#808080', '#808080', 30, False),  # Gray - NO LABEL (duplicate)
            (0.236, '0.236', '#8B0000', '#8B0000', 50, True),  # Dark red
            (0.382, '0.382', '#FF8C00', '#FF8C00', 50, True),  # Dark orange
            (0.5, '0.5', '#228B22', '#228B22', 50, True),  # Forest green
            (0.618, '0.618', '#FFD700', '#FFD700', 60, True),  # GOLDEN (default target)
            (0.786, '0.786', '#20B2AA', '#20B2AA', 50, True),  # Light sea green
            (1.0, '1.0', '#696969', '#FFFFFF', 30, True),  # Gray line, WHITE label
        ]
        
        # Build fib_levels with star on the actual target
        fib_levels = []
        for ratio, label_base, line_color, label_color, alpha, show_label in fib_ratios:
            price = entry_y + fib_range * ratio
            # Add ⭐ to the label if this matches the Fib target
            label = f"{label_base} ⭐" if abs(ratio - fib_target) < 0.001 else label_base
            fib_levels.append((ratio, price, label, line_color, label_color, alpha, show_label))
        
        # Draw dashed diagonal line from entry to swing high
        diagonal_line = pg.PlotDataItem(
            [entry_x, swing_high_x],
            [entry_y, swing_high],
            pen=pg.mkPen('#808080', width=1, style=Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(diagonal_line)
        
        # Render each Fibonacci level
        for i, (ratio, price, label, line_color, label_color, alpha, show_label) in enumerate(fib_levels):
            # Draw semi-transparent horizontal zone (from entry to exit)
            if i < len(fib_levels) - 1:
                next_price = fib_levels[i + 1][1]
                
                # Create semi-transparent color for zone
                zone_color = QColor(line_color)
                zone_color.setAlpha(alpha)
                zone_brush = QBrush(zone_color)
                
                # Create filled region item
                zone = pg.LinearRegionItem(
                    values=[price, next_price],
                    orientation='horizontal',
                    brush=zone_brush,
                    movable=False
                )
                zone.setZValue(-10)  # Behind everything
                self.plot_widget.addItem(zone)
            
            # Draw horizontal line at this level
            # Make target line thicker and more prominent
            is_target = abs(ratio - fib_target) < 0.001
            line = pg.InfiniteLine(
                pos=price,
                angle=0,  # Horizontal
                pen=pg.mkPen(line_color, width=2 if is_target else 1),  # Target line thicker
                movable=False
            )
            line.setZValue(100)  # Above zones but below arrows
            self.plot_widget.addItem(line)
            
            # Add price label on the left (only if show_label is True)
            if show_label:
                label_text = f"{label} ({price:.4f})"
                fib_label = TextItem(
                    text=label_text,
                    color=label_color,
                    anchor=(1, 0.5),  # Right-aligned, vertically centered
                    fill=(0, 0, 0, 200)  # Black background
                )
                fib_font = QFont('Courier', 10)  # Slightly larger
                fib_font.setBold(True)
                fib_label.setFont(fib_font)
                
                # Position label on the left side of the chart
                fib_label.setPos(entry_x - (exit_x - entry_x) * 0.02, price)
                fib_label.setZValue(150)  # Above lines
                self.plot_widget.addItem(fib_label)
        
        logger.debug(
            "Rendered Fibonacci levels | Entry: %.4f | Swing High: %.4f | Range: %.4f",
            entry_y,
            swing_high,
            fib_range
        )
