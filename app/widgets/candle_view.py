import asyncio
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QToolBar, QComboBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
import pyqtgraph as pg
from data.provider import DataProvider
from strategy.seller_exhaustion import SellerParams, build_features
from indicators.local import sma, rsi, macd
from core.models import Timeframe


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick chart item for PyQtGraph."""
    
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()
    
    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        
        w = 0.6
        for i, (t, row) in enumerate(self.data.iterrows()):
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Color based on direction
            if close_price > open_price:
                p.setPen(pg.mkPen('#4caf50'))
                p.setBrush(pg.mkBrush('#4caf50'))
            else:
                p.setPen(pg.mkPen('#f44336'))
                p.setBrush(pg.mkBrush('#f44336'))
            
            # Draw high-low line (wick)
            p.drawLine(pg.QtCore.QPointF(i, low_price), pg.QtCore.QPointF(i, high_price))

            # Draw open-close box (body). Use normalized coordinates so down candles
            # have a visible body instead of collapsing to a line.
            top = max(open_price, close_price)
            bottom = min(open_price, close_price)
            height = max(top - bottom, 1e-6)
            rect = pg.QtCore.QRectF(i - w / 2, bottom, w, height)
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
        self.dp = DataProvider()
        self.params = SellerParams()
        self.tf = Timeframe.m15
        self.backtest_result = None
        self.indicator_config = {
            'ema_fast': True,
            'ema_slow': True,
            'sma': False,
            'rsi': False,
            'macd': False,
            'volume': False,
            'signals': True,
            'entries': True,
            'exits': True
        }
        
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setProperty("role", "statusbar")
        status_layout.addWidget(self.status_label)
        
        # TF selector
        self.tf_combo = QComboBox()
        self.tf_combo.addItems([tf.value for tf in [Timeframe.m1, Timeframe.m3, Timeframe.m5, Timeframe.m10, Timeframe.m15]])
        self.tf_combo.setCurrentText(self.tf.value)
        self.tf_combo.currentTextChanged.connect(self.on_tf_changed)
        status_layout.addWidget(QLabel("TF:"))
        status_layout.addWidget(self.tf_combo)

        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.setObjectName("primaryButton")
        self.refresh_btn.clicked.connect(lambda: asyncio.create_task(self.load_initial()))
        status_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(status_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Chart
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0f1a12')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Price', color='#e8f5e9')
        self.plot_widget.setLabel('bottom', 'Bar Index', color='#e8f5e9')
        layout.addWidget(self.plot_widget)
        
        # Info label
        self.info_label = QLabel("Load data to begin")
        layout.addWidget(self.info_label)
    
    async def load_initial(self):
        """Load initial data and render chart."""
        try:
            self.status_label.setText("Fetching data...")
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            
            # Fetch data
            self.df = await self.dp.fetch("X:ADAUSD", self.tf, "2024-01-01", "2025-01-13")
            
            if len(self.df) == 0:
                self.status_label.setText("No data received")
                return

            summary = self.df.attrs.get("cleaning_summary")
            
            # Build features
            self.status_label.setText("Calculating indicators...")
            self.feats = build_features(self.df, self.params, self.tf)
            
            # Render
            self.status_label.setText("Rendering chart...")
            self.render_candles(self.feats)
            
            self.status_label.setText(f"Loaded {len(self.feats)} bars")

            extra_info = ""
            if summary and hasattr(summary, "changed") and summary.changed():
                dropped = (
                    summary.dropped_duplicates
                    + summary.dropped_non_finite
                    + summary.dropped_non_positive
                    + summary.dropped_negative_volume
                )
                extra_info = f"\nRemoved {dropped} problematic bars during cleaning."
                if summary.missing_expected_bars:
                    extra_info += f" Missing bars vs expected cadence: {summary.missing_expected_bars}."
            
            self.info_label.setText(
                f"Date range: {self.feats.index[0]} to {self.feats.index[-1]}\n"
                f"Signals detected: {self.feats['exhaustion'].sum()}"
                f"{extra_info}"
            )
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Error loading data: {e}")
        finally:
            self.progress.setVisible(False)

    def on_tf_changed(self, value: str):
        # Update timeframe and refresh
        try:
            self.tf = Timeframe(value)
        except Exception:
            self.tf = Timeframe.m15
        asyncio.create_task(self.load_initial())
    
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
    
    def render_candles(self, df: pd.DataFrame, backtest_result=None):
        """Render candlesticks, indicators, and trade markers."""
        self.plot_widget.clear()
        
        # Sample if too many candles (for performance)
        original_df = df
        if len(df) > 5000:
            df = df.iloc[-5000:]
        
        # Store for index mapping
        self.displayed_df = df
        
        # Create candlestick item
        candles = CandlestickItem(df[['open', 'high', 'low', 'close']])
        self.plot_widget.addItem(candles)
        
        x = np.arange(len(df))
        
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
                signal_x = [df.index.get_loc(idx) for idx in signals.index]
                signal_y = signals['low'].values * 0.998  # Slightly below low
                scatter = pg.ScatterPlotItem(
                    x=signal_x, y=signal_y,
                    pen=pg.mkPen('#ffeb3b', width=2),
                    brush=pg.mkBrush('#ffeb3b'),
                    size=12, symbol='t1'
                )
                self.plot_widget.addItem(scatter)
        
        # Mark trade entries and exits
        if backtest_result and len(backtest_result['trades']) > 0:
            trades = backtest_result['trades']
            
            # Buy markers (green arrows up)
            if self.indicator_config.get('entries', True):
                entry_times = []
                entry_prices = []
                
                for _, trade in trades.iterrows():
                    entry_ts = pd.Timestamp(trade['entry_ts'])
                    if entry_ts in df.index:
                        idx = df.index.get_loc(entry_ts)
                        entry_times.append(idx)
                        entry_prices.append(trade['entry'] * 0.998)  # Slightly below
                
                if entry_times:
                    buy_scatter = pg.ScatterPlotItem(
                        x=entry_times, y=entry_prices,
                        pen=pg.mkPen('#4caf50', width=3),
                        brush=pg.mkBrush('#4caf50'),
                        size=15, symbol='t1'  # Triangle up
                    )
                    self.plot_widget.addItem(buy_scatter)
            
            # Sell markers (red arrows down)
            if self.indicator_config.get('exits', True):
                exit_times = []
                exit_prices = []
                exit_colors = []
                
                for _, trade in trades.iterrows():
                    exit_ts = pd.Timestamp(trade['exit_ts'])
                    if exit_ts in df.index:
                        idx = df.index.get_loc(exit_ts)
                        exit_times.append(idx)
                        exit_prices.append(trade['exit'] * 1.002)  # Slightly above
                        
                        # Color based on outcome
                        if trade['pnl'] > 0:
                            exit_colors.append('#4caf50')  # Green for wins
                        else:
                            exit_colors.append('#f44336')  # Red for losses
                
                if exit_times:
                    # Create scatter with different colors per point
                    for i, (x_pos, y_pos, color) in enumerate(zip(exit_times, exit_prices, exit_colors)):
                        sell_scatter = pg.ScatterPlotItem(
                            x=[x_pos], y=[y_pos],
                            pen=pg.mkPen(color, width=3),
                            brush=pg.mkBrush(color),
                            size=15, symbol='t'  # Triangle down
                        )
                        self.plot_widget.addItem(sell_scatter)
        
        # Add legend
        self.plot_widget.addLegend()
