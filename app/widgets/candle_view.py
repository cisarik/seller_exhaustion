import asyncio
from typing import Optional
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, 
    QSplitter, QTableWidget, QTableWidgetItem, 
    QHeaderView, QGroupBox, QFileDialog
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
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setProperty("role", "statusbar")
        status_layout.addWidget(self.status_label)
        
        # Progress bar replaces old timeframe selector area
        self.action_progress = QProgressBar()
        self.action_progress.setVisible(False)
        self.action_progress.setMaximumWidth(220)
        status_layout.addWidget(self.action_progress)
        
        layout.addLayout(status_layout)
        
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
        
        # Mark trade entries, exits, and Fibonacci retracements
        if backtest_result and len(backtest_result['trades']) > 0:
            trades = backtest_result['trades']
            
            # Draw Fibonacci retracements FIRST (so they're behind markers)
            if self.indicator_config.get('fib_retracements', True) and self.selected_trade_idx is not None:
                print(f"\n🎯 Attempting to render Fibonacci for selected trade #{self.selected_trade_idx + 1}")
                self.render_fibonacci_retracement_for_trade(self.feats, trades, self.selected_trade_idx)
            
            # Render trade markers as BALLS (entry point, sized by PnL)
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
    
    def render_fibonacci_retracement_for_trade(self, df: pd.DataFrame, trades: pd.DataFrame, trade_idx: int):
        """
        Render Fibonacci retracement levels for a SELECTED trade.
        
        This matches the style from fib.png:
        - Find actual swing high from historical data
        - Draw diagonal dashed line from swing low to swing high
        - Draw horizontal Fibonacci levels with labels
        - Highlight 61.8% Golden Ratio
        - Show entry and exit markers
        
        Args:
            df: Full DataFrame with OHLC data
            trades: DataFrame of all trades
            trade_idx: Index of the trade to visualize (0-based)
        """
        print(f"\n🔍 DEBUG: render_fibonacci_retracement_for_trade called")
        print(f"   trade_idx: {trade_idx}")
        print(f"   self.feats is None: {self.feats is None}")
        
        if self.feats is None:
            print("⚠ self.feats is None - cannot render Fibonacci")
            return
            
        print(f"   self.feats columns: {list(self.feats.columns)}")
        print(f"   'fib_swing_high' in columns: {'fib_swing_high' in self.feats.columns}")
        
        if 'fib_swing_high' not in self.feats.columns:
            print("⚠ No Fibonacci data available in features (fib_swing_high column missing)")
            return
        
        if trade_idx < 0 or trade_idx >= len(trades):
            print(f"⚠ Invalid trade index: {trade_idx}")
            return
        
        try:
            trade = trades.iloc[trade_idx]
            entry_ts = pd.Timestamp(trade['entry_ts'])
            exit_ts = pd.Timestamp(trade['exit_ts'])
            
            print(f"🔍 DEBUG: Rendering Fib for trade #{trade_idx + 1}")
            print(f"   Entry: {entry_ts}, Exit: {exit_ts}")
            
            # Find the signal bar (one bar before entry)
            # Get timeframe minutes
            tf_minutes = {'m1': 1, 'm3': 3, 'm5': 5, 'm10': 10, 'm15': 15, 'm30': 30, 'm60': 60}
            minutes = tf_minutes.get(self.tf.value, 15)
            signal_ts = entry_ts - pd.Timedelta(minutes=minutes)
            
            print(f"   Signal bar: {signal_ts} (TF: {self.tf.value}, {minutes}min)")
            
            # Get Fib data from the signal bar in feats
            if signal_ts not in self.feats.index:
                print(f"⚠ Signal timestamp {signal_ts} not found in features")
                print(f"   Features index range: {self.feats.index[0]} to {self.feats.index[-1]}")
                return
            
            signal_bar = self.feats.loc[signal_ts]
            
            print(f"   Signal bar columns: {list(signal_bar.index)}")
            print(f"   Has fib_swing_high: {'fib_swing_high' in signal_bar.index}")
            
            # Check if this signal has Fib levels
            fib_swing_high_value = signal_bar.get('fib_swing_high')
            print(f"   fib_swing_high value: {fib_swing_high_value}")
            
            if pd.isna(fib_swing_high_value):
                print(f"⚠ No Fibonacci levels calculated for trade #{trade_idx + 1}")
                print(f"   Signal bar low: {signal_bar.get('low')}")
                print(f"   This means find_swing_high returned None for this signal")
                print(f"   Available Fib columns in feats: {[c for c in self.feats.columns if 'fib' in c]}")
                
                # Count how many signals have Fib data
                signals_with_fib = self.feats[self.feats['fib_swing_high'].notna()].shape[0]
                total_signals = self.feats[self.feats.get('exhaustion', False) == True].shape[0]
                print(f"   Signals with Fib data: {signals_with_fib}/{total_signals}")
                return
            
            swing_high_price = signal_bar['fib_swing_high']
            swing_low_price = signal_bar['low']
            entry_price = trade['entry']
            exit_price = trade['exit']
            
            # Find the actual swing high timestamp by searching backwards in data
            swing_high_ts = None
            lookback_bars = 96  # Default lookback from strategy
            
            signal_idx = self.feats.index.get_loc(signal_ts)
            for i in range(max(0, signal_idx - lookback_bars), signal_idx):
                if abs(self.feats.iloc[i]['high'] - swing_high_price) < 0.0001:
                    swing_high_ts = self.feats.index[i]
                    break
            
            if swing_high_ts is None:
                # Fallback: approximate as 50% back in lookback period
                approx_bars_back = lookback_bars // 2
                swing_high_ts = signal_ts - pd.Timedelta(minutes=minutes * approx_bars_back)
            
            # Convert timestamps to epoch seconds for plotting
            swing_high_x = int(swing_high_ts.value // 1_000_000_000)
            entry_x = int(entry_ts.value // 1_000_000_000)
            exit_x = int(exit_ts.value // 1_000_000_000)
            signal_x = int(signal_ts.value // 1_000_000_000)
            
            # === 1. Mark Swing High (Star) ===
            swing_high_scatter = pg.ScatterPlotItem(
                x=[swing_high_x], y=[swing_high_price],
                pen=pg.mkPen('#FFFFFF', width=2),
                brush=pg.mkBrush('#FFD700'),
                size=22, symbol='star'
            )
            self.plot_widget.addItem(swing_high_scatter)
            
            # === 2. Draw Diagonal Range Line (Swing Low to Swing High) ===
            # Dashed line showing the full retracement range
            range_line = pg.PlotDataItem(
                x=[signal_x, swing_high_x],
                y=[swing_low_price, swing_high_price],
                pen=pg.mkPen('#888888', width=2, style=Qt.DashLine)
            )
            self.plot_widget.addItem(range_line)
            
            # === 3. Draw Horizontal Fibonacci Levels ===
            # Only include levels that are actually calculated by the strategy
            fib_config = {
                0.382: {'col': 'fib_0382', 'color': '#2196F3', 'label': '38.2%'},
                0.500: {'col': 'fib_0500', 'color': '#00BCD4', 'label': '50.0%'},
                0.618: {'col': 'fib_0618', 'color': '#FFD700', 'label': '61.8% ⭐'},  # Golden
                0.786: {'col': 'fib_0786', 'color': '#FF9800', 'label': '78.6%'},
                1.000: {'col': 'fib_1000', 'color': '#F44336', 'label': '100%'},
            }
            
            for fib_ratio, config in fib_config.items():
                # Check if this level is enabled in config
                fib_col = config['col']
                if not self.indicator_config.get(fib_col, True):
                    continue
                
                # Get Fib price from signal bar
                fib_price = signal_bar.get(fib_col)
                
                if pd.isna(fib_price):
                    continue
                
                color = config['color']
                label_text = config['label']
                
                # Golden Ratio gets special styling
                if fib_ratio == 0.618:
                    width = 3
                    alpha = 220
                else:
                    width = 2
                    alpha = 180
                
                # Draw horizontal line across chart view
                fib_line = InfiniteLine(
                    pos=fib_price,
                    angle=0,
                    pen=pg.mkPen(color, width=width),
                    movable=False
                )
                self.plot_widget.addItem(fib_line)
                
                # Add label on the left side
                fib_label = TextItem(
                    text=f"{label_text} ({fib_price:.4f})",
                    color=color,
                    anchor=(0, 0.5)
                )
                # Position label to the left of the swing high
                label_x = swing_high_x - 1800  # Offset to the left
                fib_label.setPos(label_x, fib_price)
                self.plot_widget.addItem(fib_label)
            
            # === 4. Mark Entry and Exit Prices ===
            # Entry marker (horizontal line at entry)
            entry_line = InfiniteLine(
                pos=entry_price,
                angle=0,
                pen=pg.mkPen('#4CAF50', width=2, style=Qt.DotLine),
                movable=False
            )
            self.plot_widget.addItem(entry_line)
            
            entry_label = TextItem(
                text=f"Entry: ${entry_price:.4f}",
                color='#4CAF50',
                anchor=(1, 0.5)
            )
            entry_label.setPos(entry_x - 300, entry_price)
            self.plot_widget.addItem(entry_label)
            
            # Exit marker (horizontal line at exit)
            exit_color = '#4CAF50' if trade['pnl'] > 0 else '#F44336'
            exit_line = InfiniteLine(
                pos=exit_price,
                angle=0,
                pen=pg.mkPen(exit_color, width=3, style=Qt.SolidLine),
                movable=False
            )
            self.plot_widget.addItem(exit_line)
            
            # Determine exit reason label
            exit_reason = trade.get('reason', '')
            if 'fib' in exit_reason:
                try:
                    fib_pct = float(exit_reason.split('_')[1])
                    exit_label_text = f"Exit: {fib_pct}% (${exit_price:.4f})"
                except:
                    exit_label_text = f"Exit: {exit_reason} (${exit_price:.4f})"
            else:
                exit_label_text = f"Exit: {exit_reason.upper()} (${exit_price:.4f})"
            
            exit_label = TextItem(
                text=exit_label_text,
                color=exit_color,
                anchor=(1, 0.5)
            )
            exit_label.setPos(exit_x - 300, exit_price)
            self.plot_widget.addItem(exit_label)
            
            # === 5. Add Info Box ===
            info_text = f"Trade #{trade_idx + 1}\nPnL: ${trade['pnl']:.4f} ({trade['R']:.2f}R)"
            info_label = TextItem(
                text=info_text,
                color='#FFFFFF',
                anchor=(0, 0)
            )
            info_label.setPos(entry_x + 300, swing_high_price)
            self.plot_widget.addItem(info_label)
            
            print(f"✓ Rendered Fibonacci retracement for Trade #{trade_idx + 1}")
            
        except Exception as e:
            print(f"❌ Error rendering Fibonacci retracement for trade {trade_idx}: {e}")
            import traceback
            traceback.print_exc()
    
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
        
        # Export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        self.export_btn = QPushButton("Export Trades to CSV")
        self.export_btn.clicked.connect(self.export_trades)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        layout.addLayout(export_layout)
        
        group.setLayout(layout)
        return group
    
    def update_trade_table(self, trades_df):
        """Update trade list table."""
        self.trades_table.setRowCount(0)
        
        if trades_df is None or len(trades_df) == 0:
            self.export_btn.setEnabled(False)
            return
        
        self.trades_table.setRowCount(len(trades_df))
        self.export_btn.setEnabled(True)
        
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
        """Handle trade selection from table - show Fibonacci retracement for selected trade."""
        selected_rows = self.trades_table.selectedIndexes()
        
        print(f"🔍 DEBUG: on_trade_selected called, selected_rows={len(selected_rows)}")
        
        if not selected_rows:
            # No selection - clear Fibonacci display
            print("   No trade selected, clearing Fib display")
            self.selected_trade_idx = None
            # Re-render without Fib
            if self.feats is not None:
                self.render_candles(self.feats, self.backtest_result)
            return
        
        # Get the row index (first column of selected row)
        row_idx = selected_rows[0].row()
        self.selected_trade_idx = row_idx
        
        print(f"✓ Selected Trade #{row_idx + 1} (index={row_idx})")
        print(f"   self.selected_trade_idx = {self.selected_trade_idx}")
        print(f"   self.feats is not None: {self.feats is not None}")
        print(f"   self.backtest_result is not None: {self.backtest_result is not None}")
        
        # Re-render chart with Fibonacci for this trade
        if self.feats is not None:
            self.render_candles(self.feats, self.backtest_result)
        else:
            print("⚠ Cannot render: self.feats is None")
    
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
