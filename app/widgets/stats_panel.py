from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QGridLayout, QPushButton,
    QFileDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
import pandas as pd
import pyqtgraph as pg


class StatsPanel(QWidget):
    """Comprehensive statistics panel for backtest results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.trades_df = None
        self.metrics = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Backtest Results")
        title.setProperty("role", "title")
        layout.addWidget(title)
        
        # Metrics section
        self.metrics_group = self.create_metrics_section()
        layout.addWidget(self.metrics_group)
        
        # Equity curve
        self.equity_group = self.create_equity_section()
        layout.addWidget(self.equity_group)
        
        # Trade list
        self.trades_group = self.create_trades_section()
        layout.addWidget(self.trades_group, stretch=1)
        
        # Export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        self.export_btn = QPushButton("Export Trades to CSV")
        self.export_btn.clicked.connect(self.export_trades)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        layout.addLayout(export_layout)
    
    def create_metrics_section(self):
        """Create comprehensive metrics display."""
        group = QGroupBox("Performance Metrics")
        layout = QGridLayout()
        
        # Initialize metric labels
        self.metric_labels = {}
        
        metrics_config = [
            # Row 0
            ("Total Trades", "n", 0, 0),
            ("Win Rate", "win_rate", 0, 1),
            ("Avg R-Multiple", "avg_R", 0, 2),
            
            # Row 1
            ("Total PnL", "total_pnl", 1, 0),
            ("Avg Win", "avg_win", 1, 1),
            ("Avg Loss", "avg_loss", 1, 2),
            
            # Row 2
            ("Profit Factor", "profit_factor", 2, 0),
            ("Max Drawdown", "max_drawdown", 2, 1),
            ("Sharpe Ratio", "sharpe", 2, 2),
        ]
        
        for label_text, key, row, col in metrics_config:
            # Label
            label = QLabel(f"{label_text}:")
            label.setProperty("variant", "secondary")
            layout.addWidget(label, row * 2, col)
            
            # Value
            value = QLabel("--")
            value.setStyleSheet("font-size: 16px; font-weight: bold;")
            layout.addWidget(value, row * 2 + 1, col)
            
            self.metric_labels[key] = value
        
        group.setLayout(layout)
        return group
    
    def create_equity_section(self):
        """Create equity curve chart."""
        group = QGroupBox("Equity Curve")
        layout = QVBoxLayout()
        
        self.equity_plot = pg.PlotWidget()
        self.equity_plot.setBackground('#0f1a12')
        self.equity_plot.setLabel('left', 'Cumulative PnL', color='#e8f5e9')
        self.equity_plot.setLabel('bottom', 'Trade Number', color='#e8f5e9')
        self.equity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.equity_plot.setMinimumHeight(200)
        
        layout.addWidget(self.equity_plot)
        group.setLayout(layout)
        return group
    
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
        
        layout.addWidget(self.trades_table)
        group.setLayout(layout)
        return group
    
    def update_stats(self, backtest_result):
        """Update all statistics displays with backtest results."""
        try:
            self.trades_df = backtest_result['trades']
            self.metrics = backtest_result['metrics']
            
            # Update metrics
            self.update_metrics()
            
            # Update equity curve
            self.update_equity_curve()
            
            # Update trade table
            self.update_trade_table()
            
            # Enable export
            self.export_btn.setEnabled(len(self.trades_df) > 0)
            
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def update_metrics(self):
        """Update metric labels."""
        if not self.metrics:
            return
        
        # Total trades
        self.metric_labels['n'].setText(str(self.metrics['n']))
        
        # Win rate
        win_rate = self.metrics['win_rate']
        self.metric_labels['win_rate'].setText(f"{win_rate:.1%}")
        if win_rate >= 0.5:
            self.metric_labels['win_rate'].setStyleSheet("font-size: 16px; font-weight: bold; color: #4caf50;")
        else:
            self.metric_labels['win_rate'].setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
        
        # Avg R
        avg_r = self.metrics['avg_R']
        self.metric_labels['avg_R'].setText(f"{avg_r:.2f}R")
        if avg_r > 0:
            self.metric_labels['avg_R'].setStyleSheet("font-size: 16px; font-weight: bold; color: #4caf50;")
        else:
            self.metric_labels['avg_R'].setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
        
        # Total PnL
        total_pnl = self.metrics['total_pnl']
        self.metric_labels['total_pnl'].setText(f"${total_pnl:.4f}")
        if total_pnl > 0:
            self.metric_labels['total_pnl'].setStyleSheet("font-size: 16px; font-weight: bold; color: #4caf50;")
        else:
            self.metric_labels['total_pnl'].setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
        
        # Avg win/loss (if available)
        if 'avg_win' in self.metrics:
            self.metric_labels['avg_win'].setText(f"${self.metrics['avg_win']:.4f}")
            self.metric_labels['avg_loss'].setText(f"${self.metrics['avg_loss']:.4f}")
        else:
            wins = self.trades_df[self.trades_df['pnl'] > 0]
            losses = self.trades_df[self.trades_df['pnl'] <= 0]
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
            avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0.0
            self.metric_labels['avg_win'].setText(f"${avg_win:.4f}")
            self.metric_labels['avg_loss'].setText(f"${avg_loss:.4f}")
        
        # Profit factor (if available)
        if 'profit_factor' in self.metrics:
            pf = self.metrics['profit_factor']
            if pf == float('inf'):
                self.metric_labels['profit_factor'].setText("∞")
            else:
                self.metric_labels['profit_factor'].setText(f"{pf:.2f}")
        else:
            wins = self.trades_df[self.trades_df['pnl'] > 0]
            losses = self.trades_df[self.trades_df['pnl'] <= 0]
            total_wins = wins['pnl'].sum() if len(wins) > 0 else 0.0
            total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0.0
            pf = total_wins / total_losses if total_losses > 0 else float('inf')
            if pf == float('inf'):
                self.metric_labels['profit_factor'].setText("∞")
            else:
                self.metric_labels['profit_factor'].setText(f"{pf:.2f}")
        
        # Max drawdown
        max_dd = self.metrics.get('max_dd', 0.0)
        self.metric_labels['max_drawdown'].setText(f"${max_dd:.4f}")
        self.metric_labels['max_drawdown'].setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
        
        # Sharpe
        sharpe = self.metrics.get('sharpe', 0.0)
        self.metric_labels['sharpe'].setText(f"{sharpe:.2f}")
    
    def update_equity_curve(self):
        """Update equity curve chart."""
        self.equity_plot.clear()
        
        if self.trades_df is None or len(self.trades_df) == 0:
            return
        
        # Calculate cumulative PnL
        cumulative_pnl = self.trades_df['pnl'].cumsum()
        
        # Plot equity curve
        x = range(1, len(cumulative_pnl) + 1)
        self.equity_plot.plot(
            x, cumulative_pnl.values,
            pen=pg.mkPen('#4caf50', width=2),
            name='Equity Curve'
        )
        
        # Add zero line
        self.equity_plot.plot(
            [1, len(cumulative_pnl)], [0, 0],
            pen=pg.mkPen('#666', width=1, style=Qt.DashLine)
        )
        
        # Calculate and plot drawdown
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        
        # Fill drawdown area
        fill = pg.FillBetweenItem(
            pg.PlotDataItem(x, cumulative_pnl.values),
            pg.PlotDataItem(x, running_max.values),
            brush=(255, 0, 0, 50)
        )
        self.equity_plot.addItem(fill)
    
    def update_trade_table(self):
        """Update trade list table."""
        self.trades_table.setRowCount(0)
        
        if self.trades_df is None or len(self.trades_df) == 0:
            return
        
        self.trades_table.setRowCount(len(self.trades_df))
        
        for i, (idx, trade) in enumerate(self.trades_df.iterrows()):
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
    
    def export_trades(self):
        """Export trades to CSV file."""
        if self.trades_df is None or len(self.trades_df) == 0:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Trades",
            "trades.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                self.trades_df.to_csv(filename, index=False)
                print(f"✓ Exported {len(self.trades_df)} trades to {filename}")
            except Exception as e:
                print(f"Error exporting trades: {e}")
    
    def clear(self):
        """Clear all displays."""
        self.trades_df = None
        self.metrics = None
        
        # Reset metrics
        for label in self.metric_labels.values():
            label.setText("--")
            label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # Clear equity curve
        self.equity_plot.clear()
        
        # Clear trade table
        self.trades_table.setRowCount(0)
        
        # Disable export
        self.export_btn.setEnabled(False)
