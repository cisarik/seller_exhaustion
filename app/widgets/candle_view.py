import asyncio
from typing import Optional
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, 
    QSplitter, QTableWidget, QTableWidgetItem, 
    QHeaderView, QGroupBox, QFileDialog, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor
import pyqtgraph as pg
from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem
from pyqtgraph import InfiniteLine, TextItem
from strategy.seller_exhaustion import SellerParams, build_features
from indicators.local import sma, rsi, macd
from core.models import Timeframe


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
            
            # DEBUG: Print spacing info
            print(f"\n🕯️ CandlestickItem spacing:")
            print(f"   Number of candles: {len(self.x_values)}")
            print(f"   Median spacing: {step} seconds ({step/60:.1f} minutes)")
            print(f"   Candle width: {w} seconds ({w/60:.1f} minutes)")
            print(f"   First 3 x_values: {self.x_values[:3]}")
            print(f"   First 3 diffs: {diffs[:3]}")
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
    
    # Signal emitted when chart is clicked (for future features)
    chart_clicked = Signal(float, float)
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.feats = None
        self.params = SellerParams()
        self.tf = Timeframe.m15
        self.backtest_result = None
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
        self.plot_widget.setLabel('bottom', 'Time', color='#e8f5e9')
        chart_layout.addWidget(self.plot_widget)
        
        # Info label
        self.info_label = QLabel("Load data to begin")
        chart_layout.addWidget(self.info_label)
        
        splitter.addWidget(chart_widget)
        
        # Bottom: Trade history table
        self.trades_group = self.create_trades_section()
        splitter.addWidget(self.trades_group)
        
        # Set initial sizes (chart gets 70%, trade history gets 30%)
        splitter.setSizes([700, 300])
        
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
    
    def set_indicator_config(self, config):
        """Update indicator configuration and re-render."""
        self.indicator_config = config
        if self.feats is not None:
            self.render_candles(self.feats, self.backtest_result)
    
    def set_backtest_result(self, result):
        """Set backtest results and re-render with trade markers."""
        self.backtest_result = result
        if self.feats is not None:
            self.render_candles(self.feats, result)
        
        # Update trade table
        if result and 'trades' in result:
            self.update_trade_table(result['trades'])
        else:
            self.update_trade_table(None)
    
    def render_candles(self, df: pd.DataFrame, backtest_result=None):
        """Render candlesticks, indicators, and trade markers."""
        self.plot_widget.clear()
        
        # Store full dataframe as feats for Fibonacci access
        self.feats = df
        
        # DEBUG: Print data info
        print(f"\n📊 Rendering chart:")
        print(f"   Total bars: {len(df)}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Timeframe: {self.tf.value}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check actual spacing between bars
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()[1:6]  # First 5 differences
            print(f"   First 5 time deltas:")
            for i, td in enumerate(time_diffs):
                print(f"     {i+1}: {td} ({td.total_seconds()/60:.0f} minutes)")
            
            median_minutes = df.index.to_series().diff().median().total_seconds() / 60
            print(f"   Median bar spacing: {median_minutes:.0f} minutes")
        
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
        
        print(f"   First epoch second: {epoch_seconds[0]} ({pd.Timestamp(epoch_seconds[0], unit='s', tz='UTC')})")
        print(f"   Last epoch second: {epoch_seconds[-1]} ({pd.Timestamp(epoch_seconds[-1], unit='s', tz='UTC')})")

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
    
    def render_trade_balls(self, trades: pd.DataFrame, df: pd.DataFrame):
        """
        Render trades as BALLS positioned at entry points:
        - Green balls for profitable trades (size proportional to profit)
        - Red balls for losing trades (size proportional to loss)
        - WHITE ball for the currently selected trade
        
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
        
        ball_data = []
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
            
            # Determine color
            if i == self.selected_trade_idx:
                # Selected trade = WHITE with black border
                color = '#FFFFFF'
                border_color = '#000000'
                border_width = 3
            elif pnl > 0:
                # Profit = GREEN
                color = '#4CAF50'
                border_color = '#2E7D32'
                border_width = 2
            else:
                # Loss = RED
                color = '#F44336'
                border_color = '#C62828'
                border_width = 2
            
            ball_data.append({
                'x': entry_x,
                'y': entry_y,
                'size': size,
                'color': color,
                'border_color': border_color,
                'border_width': border_width
            })
        
        if not ball_data:
            print("⚠ No valid trade balls to render")
            return
        
        # Render all balls as scatter plot
        x_coords = [b['x'] for b in ball_data]
        y_coords = [b['y'] for b in ball_data]
        sizes = [b['size'] for b in ball_data]
        
        # Create brushes and pens for each ball
        brushes = [pg.mkBrush(b['color']) for b in ball_data]
        pens = [pg.mkPen(b['border_color'], width=b['border_width']) for b in ball_data]
        
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
        
        selected_str = f" (#{self.selected_trade_idx + 1} highlighted)" if self.selected_trade_idx is not None else ""
        print(f"✓ Rendered {len(ball_data)} trade balls{selected_str}")

    
    def create_trades_section(self):
        """Create trade list table."""
        group = QGroupBox("Trade History")
        layout = QVBoxLayout()
        
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(9)
        self.trades_table.setHorizontalHeaderLabels([
            "#", "Entry Time", "Exit Time", "Entry $", "Exit $",
            "PnL $", "R-Multiple", "Bars", "Exit Reason"
        ])
        
        # Set column resize modes
        header = self.trades_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        
        self.trades_table.setAlternatingRowColors(True)
        self.trades_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Connect selection signal
        self.trades_table.itemSelectionChanged.connect(self.on_trade_selected)
        
        layout.addWidget(self.trades_table)
        
        group.setLayout(layout)
        return group
    
    def update_trade_table(self, trades_df):
        """Update trade list table."""
        self.trades_table.setRowCount(0)
        
        if trades_df is None or len(trades_df) == 0:
            return
        
        self.trades_table.setRowCount(len(trades_df))
        
        for i, (idx, trade) in enumerate(trades_df.iterrows()):
            # Trade number
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            
            # Entry time
            entry_time = str(trade['entry_ts'])[:19]
            self.trades_table.setItem(i, 1, QTableWidgetItem(entry_time))
            
            # Exit time
            exit_time = str(trade['exit_ts'])[:19]
            self.trades_table.setItem(i, 2, QTableWidgetItem(exit_time))
            
            # Entry price
            entry_item = QTableWidgetItem(f"{trade['entry']:.4f}")
            entry_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.trades_table.setItem(i, 3, entry_item)
            
            # Exit price
            exit_item = QTableWidgetItem(f"{trade['exit']:.4f}")
            exit_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.trades_table.setItem(i, 4, exit_item)
            
            # PnL
            pnl = trade['pnl']
            pnl_item = QTableWidgetItem(f"{pnl:.4f}")
            pnl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if pnl > 0:
                pnl_item.setForeground(QColor('#4caf50'))
            else:
                pnl_item.setForeground(QColor('#f44336'))
            self.trades_table.setItem(i, 5, pnl_item)
            
            # R-multiple
            r = trade['R']
            r_item = QTableWidgetItem(f"{r:.2f}R")
            r_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if r > 0:
                r_item.setForeground(QColor('#4caf50'))
            else:
                r_item.setForeground(QColor('#f44336'))
            self.trades_table.setItem(i, 6, r_item)
            
            # Bars held
            bars = trade.get('bars_held', '--')
            bars_item = QTableWidgetItem(str(bars))
            bars_item.setTextAlignment(Qt.AlignCenter)
            self.trades_table.setItem(i, 7, bars_item)
            
            # Exit reason
            reason = trade['reason']
            reason_item = QTableWidgetItem(reason.upper())
            reason_item.setTextAlignment(Qt.AlignCenter)
            if reason == 'tp':
                reason_item.setForeground(QColor('#4caf50'))
            elif reason in ['stop', 'stop_gap']:
                reason_item.setForeground(QColor('#f44336'))
            else:
                reason_item.setForeground(QColor('#ff9800'))
            self.trades_table.setItem(i, 8, reason_item)
    
    def on_trade_selected(self):
        """Handle trade selection from table - highlight selected trade ball."""
        selected_rows = self.trades_table.selectedIndexes()
        
        if not selected_rows:
            # No selection - clear highlight
            self.selected_trade_idx = None
        else:
            # Get the row index (first column of selected row)
            row_idx = selected_rows[0].row()
            self.selected_trade_idx = row_idx
        
        # Re-render balls only (much faster than full chart re-render)
        if self.backtest_result and 'trades' in self.backtest_result:
            trades = self.backtest_result['trades']
            if len(trades) > 0 and self.feats is not None:
                original_df = self.feats[['open', 'high', 'low', 'close', 'volume']].copy()
                # Remove old trade balls
                items_to_remove = [item for item in self.plot_widget.items() if isinstance(item, pg.ScatterPlotItem)]
                for item in items_to_remove:
                    self.plot_widget.removeItem(item)
                # Re-render with new selection
                self.render_trade_balls(trades, original_df)
    
    def export_trades(self):
        """Export trades to CSV file."""
        if self.backtest_result is None or len(self.backtest_result.get('trades', [])) == 0:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Trades",
            "trades.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                self.backtest_result['trades'].to_csv(filename, index=False)
                print(f"✓ Exported {len(self.backtest_result['trades'])} trades to {filename}")
            except Exception as e:
                print(f"Error exporting trades: {e}")
