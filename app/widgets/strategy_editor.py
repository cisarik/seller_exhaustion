"""
Strategy Editor widget for editing and persisting strategy parameters.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QComboBox, QTextEdit, QSplitter, QListWidget, QMessageBox,
    QInputDialog, QListWidgetItem
)
from PySide6.QtCore import Qt, Signal
from datetime import datetime

from strategy.seller_exhaustion import SellerParams
from backtest.engine import BacktestParams
from strategy.params_store import params_store


class StrategyEditor(QWidget):
    """
    Interactive strategy parameter editor with explanations and persistence.
    
    Features:
    - Edit all strategy and backtest parameters
    - Detailed parameter explanations
    - Save/load parameter sets
    - Export to JSON/YAML
    - View parameter evolution history
    """
    
    params_changed = Signal()  # Emitted when parameters are modified
    params_loaded = Signal(object, object)  # Emitted when params loaded (seller, backtest)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_seller_params = SellerParams()
        self.current_backtest_params = BacktestParams()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Top toolbar
        toolbar = self.create_toolbar()
        layout.addLayout(toolbar)
        
        # Main content: parameter editor + explanation panel
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Parameter editors
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(self.create_strategy_params_group())
        left_layout.addWidget(self.create_backtest_params_group())
        left_layout.addStretch()
        splitter.addWidget(left_widget)
        
        # Right: Saved parameters list + explanation
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.create_saved_params_group())
        right_layout.addWidget(self.create_explanation_panel())
        splitter.addWidget(right_widget)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
    
    def create_toolbar(self):
        """Create toolbar with save/load/export buttons."""
        toolbar = QHBoxLayout()
        
        self.save_btn = QPushButton("💾 Save Params")
        self.save_btn.setToolTip("Save current parameters to file")
        self.save_btn.clicked.connect(self.save_params)
        toolbar.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("📂 Load Params")
        self.load_btn.setToolTip("Load parameters from file")
        self.load_btn.clicked.connect(self.load_params)
        toolbar.addWidget(self.load_btn)
        
        self.export_btn = QPushButton("📤 Export YAML")
        self.export_btn.setToolTip("Export parameters to YAML format")
        self.export_btn.clicked.connect(self.export_yaml)
        toolbar.addWidget(self.export_btn)
        
        self.reset_btn = QPushButton("🔄 Reset Defaults")
        self.reset_btn.setToolTip("Reset all parameters to default values")
        self.reset_btn.clicked.connect(self.reset_defaults)
        toolbar.addWidget(self.reset_btn)
        
        toolbar.addStretch()
        
        return toolbar
    
    def create_strategy_params_group(self):
        """Create strategy parameters editor group."""
        group = QGroupBox("📊 Strategy Parameters - Seller Exhaustion")
        layout = QFormLayout()
        
        # EMA Fast
        self.ema_fast_spin = QSpinBox()
        self.ema_fast_spin.setRange(10, 500)
        self.ema_fast_spin.setValue(96)
        self.ema_fast_spin.setSuffix(" bars (~1 day on 15m)")
        self.ema_fast_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("EMA Fast:", self.ema_fast_spin)
        
        # EMA Slow
        self.ema_slow_spin = QSpinBox()
        self.ema_slow_spin.setRange(100, 2000)
        self.ema_slow_spin.setValue(672)
        self.ema_slow_spin.setSuffix(" bars (~7 days on 15m)")
        self.ema_slow_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("EMA Slow:", self.ema_slow_spin)
        
        # Z-Score Window
        self.z_window_spin = QSpinBox()
        self.z_window_spin.setRange(100, 2000)
        self.z_window_spin.setValue(672)
        self.z_window_spin.setSuffix(" bars")
        self.z_window_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("Z-Score Window:", self.z_window_spin)
        
        # ATR Window
        self.atr_window_spin = QSpinBox()
        self.atr_window_spin.setRange(10, 500)
        self.atr_window_spin.setValue(96)
        self.atr_window_spin.setSuffix(" bars")
        self.atr_window_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("ATR Window:", self.atr_window_spin)
        
        # Volume Z-Score Threshold
        self.vol_z_spin = QDoubleSpinBox()
        self.vol_z_spin.setRange(0.5, 5.0)
        self.vol_z_spin.setValue(2.0)
        self.vol_z_spin.setSingleStep(0.1)
        self.vol_z_spin.setDecimals(2)
        self.vol_z_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("Volume Z-Threshold:", self.vol_z_spin)
        
        # True Range Z-Score Threshold
        self.tr_z_spin = QDoubleSpinBox()
        self.tr_z_spin.setRange(0.5, 3.0)
        self.tr_z_spin.setValue(1.2)
        self.tr_z_spin.setSingleStep(0.1)
        self.tr_z_spin.setDecimals(2)
        self.tr_z_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("True Range Z-Threshold:", self.tr_z_spin)
        
        # Close Location Minimum
        self.cloc_min_spin = QDoubleSpinBox()
        self.cloc_min_spin.setRange(0.0, 1.0)
        self.cloc_min_spin.setValue(0.6)
        self.cloc_min_spin.setSingleStep(0.05)
        self.cloc_min_spin.setDecimals(2)
        self.cloc_min_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("Close Location Min:", self.cloc_min_spin)
        
        group.setLayout(layout)
        return group
    
    def create_backtest_params_group(self):
        """Create backtest parameters editor group."""
        group = QGroupBox("🎯 Backtest Parameters & Exit Strategy")
        layout = QFormLayout()
        
        # === EXIT TOGGLES SECTION ===
        layout.addRow(QLabel("<b>Exit Strategy (toggles):</b>"), QLabel(""))
        
        # Fibonacci Exit Toggle (DEFAULT: ON)
        self.use_fib_check = QCheckBox("✓ Use Fibonacci exits (DEFAULT)")
        self.use_fib_check.setChecked(True)
        self.use_fib_check.stateChanged.connect(self.on_param_changed)
        self.use_fib_check.setToolTip("Exit at first Fibonacci retracement level hit")
        layout.addRow("", self.use_fib_check)
        
        # Stop Loss Toggle (DEFAULT: OFF)
        self.use_stop_check = QCheckBox("Use stop-loss (optional)")
        self.use_stop_check.setChecked(False)
        self.use_stop_check.stateChanged.connect(self.on_param_changed)
        self.use_stop_check.setToolTip("Exit if price drops below signal low - ATR×multiplier")
        layout.addRow("", self.use_stop_check)
        
        # Traditional TP Toggle (DEFAULT: OFF)
        self.use_tp_check = QCheckBox("Use traditional TP (optional)")
        self.use_tp_check.setChecked(False)
        self.use_tp_check.stateChanged.connect(self.on_param_changed)
        self.use_tp_check.setToolTip("Exit at fixed R-multiple take profit")
        layout.addRow("", self.use_tp_check)
        
        # Time Exit Toggle (DEFAULT: OFF)
        self.use_time_check = QCheckBox("Use time-based exit (optional)")
        self.use_time_check.setChecked(False)
        self.use_time_check.stateChanged.connect(self.on_param_changed)
        self.use_time_check.setToolTip("Exit after max hold time if no other exit triggered")
        layout.addRow("", self.use_time_check)
        
        layout.addRow(QLabel(""), QLabel(""))  # Spacer
        
        # === FIBONACCI PARAMETERS ===
        layout.addRow(QLabel("<b>Fibonacci Parameters:</b>"), QLabel(""))
        
        # Fibonacci Swing Lookback
        self.fib_lookback_spin = QSpinBox()
        self.fib_lookback_spin.setRange(20, 500)
        self.fib_lookback_spin.setValue(96)
        self.fib_lookback_spin.setSuffix(" bars")
        self.fib_lookback_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("Fib Swing Lookback:", self.fib_lookback_spin)
        
        # Fibonacci Swing Lookahead
        self.fib_lookahead_spin = QSpinBox()
        self.fib_lookahead_spin.setRange(2, 20)
        self.fib_lookahead_spin.setValue(5)
        self.fib_lookahead_spin.setSuffix(" bars")
        self.fib_lookahead_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("Fib Swing Lookahead:", self.fib_lookahead_spin)
        
        # Fibonacci Target Level with Golden Button
        fib_target_layout = QHBoxLayout()
        
        self.fib_target_combo = QComboBox()
        self.fib_target_combo.addItem("38.2% Fib", 0.382)
        self.fib_target_combo.addItem("50.0% Fib", 0.5)
        self.fib_target_combo.addItem("61.8% Fib (Golden)", 0.618)
        self.fib_target_combo.addItem("78.6% Fib", 0.786)
        self.fib_target_combo.addItem("100% Fib (Full)", 1.0)
        self.fib_target_combo.setCurrentIndex(2)  # 61.8% default
        self.fib_target_combo.currentIndexChanged.connect(self.on_param_changed)
        fib_target_layout.addWidget(self.fib_target_combo)
        
        # Golden Ratio Button
        self.golden_btn = QPushButton("⭐ Set Golden")
        self.golden_btn.setToolTip("Start with 61.8% (Golden Ratio) for balanced risk/reward")
        self.golden_btn.setMaximumWidth(120)
        self.golden_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFD700, stop:0.5 #FFA500, stop:1 #FF8C00);
                color: #000000;
                border: 2px solid #DAA520;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFED4E, stop:0.5 #FFB84D, stop:1 #FFA500);
                border: 2px solid #FFD700;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #DAA520, stop:0.5 #B8860B, stop:1 #8B6914);
            }
        """)
        self.golden_btn.clicked.connect(self.set_golden_ratio)
        fib_target_layout.addWidget(self.golden_btn)
        
        fib_target_layout.addStretch()
        
        fib_target_widget = QWidget()
        fib_target_widget.setLayout(fib_target_layout)
        layout.addRow("Fib Target Level:", fib_target_widget)
        
        layout.addRow(QLabel(""), QLabel(""))  # Spacer
        
        # === STOP LOSS PARAMETERS ===
        layout.addRow(QLabel("<b>Stop-Loss Parameters:</b>"), QLabel(""))
        
        # ATR Stop Multiplier
        self.atr_stop_mult_spin = QDoubleSpinBox()
        self.atr_stop_mult_spin.setRange(0.1, 3.0)
        self.atr_stop_mult_spin.setValue(0.7)
        self.atr_stop_mult_spin.setSingleStep(0.1)
        self.atr_stop_mult_spin.setDecimals(2)
        self.atr_stop_mult_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("ATR Stop Multiplier:", self.atr_stop_mult_spin)
        
        layout.addRow(QLabel(""), QLabel(""))  # Spacer
        
        # === TRADITIONAL TP PARAMETERS ===
        layout.addRow(QLabel("<b>Traditional TP Parameters:</b>"), QLabel(""))
        
        # Reward R Multiple
        self.reward_r_spin = QDoubleSpinBox()
        self.reward_r_spin.setRange(0.5, 5.0)
        self.reward_r_spin.setValue(2.0)
        self.reward_r_spin.setSingleStep(0.5)
        self.reward_r_spin.setDecimals(1)
        self.reward_r_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("Reward R Multiple:", self.reward_r_spin)
        
        layout.addRow(QLabel(""), QLabel(""))  # Spacer
        
        # === TIME EXIT PARAMETERS ===
        layout.addRow(QLabel("<b>Time Exit Parameters:</b>"), QLabel(""))
        
        # Max Hold Bars
        self.max_hold_spin = QSpinBox()
        self.max_hold_spin.setRange(10, 500)
        self.max_hold_spin.setValue(96)
        self.max_hold_spin.setSuffix(" bars (~24h on 15m)")
        self.max_hold_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("Max Hold Time:", self.max_hold_spin)
        
        layout.addRow(QLabel(""), QLabel(""))  # Spacer
        
        # === TRANSACTION COSTS ===
        layout.addRow(QLabel("<b>Transaction Costs:</b>"), QLabel(""))
        
        # Fee BP
        self.fee_bp_spin = QDoubleSpinBox()
        self.fee_bp_spin.setRange(0.0, 50.0)
        self.fee_bp_spin.setValue(5.0)
        self.fee_bp_spin.setSingleStep(1.0)
        self.fee_bp_spin.setDecimals(1)
        self.fee_bp_spin.setSuffix(" bp")
        self.fee_bp_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("Fees:", self.fee_bp_spin)
        
        # Slippage BP
        self.slippage_bp_spin = QDoubleSpinBox()
        self.slippage_bp_spin.setRange(0.0, 50.0)
        self.slippage_bp_spin.setValue(5.0)
        self.slippage_bp_spin.setSingleStep(1.0)
        self.slippage_bp_spin.setDecimals(1)
        self.slippage_bp_spin.setSuffix(" bp")
        self.slippage_bp_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("Slippage:", self.slippage_bp_spin)
        
        group.setLayout(layout)
        return group
    
    def create_saved_params_group(self):
        """Create saved parameters list group."""
        group = QGroupBox("💾 Saved Parameter Sets")
        layout = QVBoxLayout()
        
        self.saved_list = QListWidget()
        self.saved_list.itemDoubleClicked.connect(self.on_saved_item_clicked)
        layout.addWidget(self.saved_list)
        
        btn_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_saved_list)
        btn_layout.addWidget(self.refresh_btn)
        
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.clicked.connect(self.delete_saved_params)
        btn_layout.addWidget(self.delete_btn)
        
        layout.addLayout(btn_layout)
        
        group.setLayout(layout)
        
        # Load saved params on init
        self.refresh_saved_list()
        
        return group
    
    def create_explanation_panel(self):
        """Create parameter explanation panel."""
        group = QGroupBox("📖 Parameter Explanations")
        layout = QVBoxLayout()
        
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMaximumHeight(400)
        self.explanation_text.setHtml(self.get_explanations_html())
        
        layout.addWidget(self.explanation_text)
        group.setLayout(layout)
        return group
    
    def get_explanations_html(self):
        """Generate HTML explanation of all parameters."""
        return """
        <h3>Seller Exhaustion Strategy</h3>
        <p>Detects potential bottoms in downtrends by identifying seller exhaustion patterns.</p>
        
        <h4>Entry Signal Conditions:</h4>
        <ul>
            <li><b>Downtrend:</b> EMA Fast &lt; EMA Slow (price below average)</li>
            <li><b>Volume Spike:</b> Volume Z-score &gt; threshold (unusual selling pressure)</li>
            <li><b>Range Expansion:</b> True Range Z-score &gt; threshold (high volatility)</li>
            <li><b>Close Near High:</b> Close location &gt; threshold (buyers stepping in)</li>
        </ul>
        
        <h4>Strategy Parameters:</h4>
        <ul>
            <li><b>EMA Fast (96):</b> Fast exponential moving average (~1 day on 15m)</li>
            <li><b>EMA Slow (672):</b> Slow exponential moving average (~7 days on 15m)</li>
            <li><b>Z-Score Window (672):</b> Lookback period for calculating z-scores</li>
            <li><b>ATR Window (96):</b> Period for Average True Range calculation</li>
            <li><b>Volume Z-Threshold (2.0):</b> Minimum volume z-score for signal (2.0 = 2 std devs)</li>
            <li><b>TR Z-Threshold (1.2):</b> Minimum true range z-score for volatility spike</li>
            <li><b>Close Location Min (0.6):</b> Minimum close position in candle (0.6 = top 40%)</li>
        </ul>
        
        <h4>Exit Strategy (NEW - Configurable):</h4>
        <p><b>DEFAULT:</b> BUY at exhaustion bottoms, SELL at first Fibonacci level hit ONLY.</p>
        <ul>
            <li><b>✓ Fibonacci Exits (ENABLED by default):</b> Exit at first Fib level hit
                <ul>
                    <li>38.2% - First resistance (conservative)</li>
                    <li>50.0% - Mid-point (balanced)</li>
                    <li>61.8% - Golden ratio (default)</li>
                    <li>78.6% - Near full retracement</li>
                    <li>100% - Full retracement to swing high</li>
                </ul>
            </li>
            <li><b>Stop Loss (optional, OFF by default):</b> Signal low - (ATR × multiplier)</li>
            <li><b>Traditional TP (optional, OFF by default):</b> Entry + (Risk × R-multiple)</li>
            <li><b>Time Exit (optional, OFF by default):</b> Exit after max hold time</li>
        </ul>
        <p><b>Note:</b> You can enable multiple exits. Priority: Stop → Fib → TP → Time</p>
        
        <h4>Backtest Parameters:</h4>
        <ul>
            <li><b>Exit Toggles:</b> Enable/disable each exit type independently</li>
            <li><b>ATR Stop Multiplier (0.7):</b> Stop distance as multiple of ATR (if enabled)</li>
            <li><b>Reward R Multiple (2.0):</b> TP = Risk × R (if traditional TP enabled)</li>
            <li><b>Max Hold (96):</b> Maximum bars to hold (~24h on 15m) (if time exit enabled)</li>
            <li><b>Fibonacci Swing Lookback (96):</b> Bars to search for swing high</li>
            <li><b>Fibonacci Swing Lookahead (5):</b> Confirmation period for swing high</li>
            <li><b>Fibonacci Target Level:</b> Which Fib level to use (38.2%-100%)</li>
            <li><b>Fees (5 bp):</b> Transaction fees in basis points (0.05%)</li>
            <li><b>Slippage (5 bp):</b> Slippage estimate in basis points (0.05%)</li>
        </ul>
        
        <h4>Fibonacci Exit Logic:</h4>
        <p>When enabled, the strategy uses Fibonacci retracement levels instead of fixed R-multiple take profits:</p>
        <ol>
            <li>Find most recent swing high before entry signal</li>
            <li>Calculate Fibonacci levels from signal low to swing high</li>
            <li>Exit at first Fibonacci level hit (or stop/time exit)</li>
            <li>Allows natural profit-taking at key resistance zones</li>
        </ol>
        
        <p><b>💡 Quick Tip:</b> Click the <span style="background: linear-gradient(to bottom, #FFD700, #FFA500); color: black; padding: 2px 8px; border-radius: 3px; font-weight: bold;">⭐ Set Golden</span> button to use 61.8% (Golden Ratio) - the optimal balance between risk and reward!</p>
        
        <p><b>Fibonacci Level Guide:</b></p>
        <ul>
            <li><b>38.2%</b> - Conservative (quick profits, high win rate, lower avg R)</li>
            <li><b>50.0%</b> - Balanced mid-point</li>
            <li><b>61.8%</b> - <span style="color: #FFD700;">★ Golden Ratio (RECOMMENDED)</span> - Optimal risk/reward</li>
            <li><b>78.6%</b> - Aggressive (near full retracement)</li>
            <li><b>100%</b> - Very aggressive (full retracement to swing high)</li>
        </ul>
        """
    
    def get_seller_params(self) -> SellerParams:
        """Get current strategy parameters from UI."""
        return SellerParams(
            ema_fast=self.ema_fast_spin.value(),
            ema_slow=self.ema_slow_spin.value(),
            z_window=self.z_window_spin.value(),
            atr_window=self.atr_window_spin.value(),
            vol_z=self.vol_z_spin.value(),
            tr_z=self.tr_z_spin.value(),
            cloc_min=self.cloc_min_spin.value()
        )
    
    def get_backtest_params(self) -> BacktestParams:
        """Get current backtest parameters from UI."""
        return BacktestParams(
            # Exit toggles
            use_stop_loss=self.use_stop_check.isChecked(),
            use_time_exit=self.use_time_check.isChecked(),
            use_fib_exits=self.use_fib_check.isChecked(),
            use_traditional_tp=self.use_tp_check.isChecked(),
            # Stop-loss parameters
            atr_stop_mult=self.atr_stop_mult_spin.value(),
            # Traditional TP parameters
            reward_r=self.reward_r_spin.value(),
            # Time exit parameters
            max_hold=self.max_hold_spin.value(),
            # Fibonacci parameters
            fib_swing_lookback=self.fib_lookback_spin.value(),
            fib_swing_lookahead=self.fib_lookahead_spin.value(),
            fib_target_level=self.fib_target_combo.currentData(),
            # Transaction costs
            fee_bp=self.fee_bp_spin.value(),
            slippage_bp=self.slippage_bp_spin.value()
        )
    
    def set_seller_params(self, params: SellerParams):
        """Load strategy parameters into UI."""
        self.ema_fast_spin.setValue(params.ema_fast)
        self.ema_slow_spin.setValue(params.ema_slow)
        self.z_window_spin.setValue(params.z_window)
        self.atr_window_spin.setValue(params.atr_window)
        self.vol_z_spin.setValue(params.vol_z)
        self.tr_z_spin.setValue(params.tr_z)
        self.cloc_min_spin.setValue(params.cloc_min)
    
    def set_backtest_params(self, params: BacktestParams):
        """Load backtest parameters into UI."""
        # Exit toggles
        self.use_stop_check.setChecked(params.use_stop_loss)
        self.use_time_check.setChecked(params.use_time_exit)
        self.use_fib_check.setChecked(params.use_fib_exits)
        self.use_tp_check.setChecked(params.use_traditional_tp)
        
        # Stop-loss parameters
        self.atr_stop_mult_spin.setValue(params.atr_stop_mult)
        
        # Traditional TP parameters
        self.reward_r_spin.setValue(params.reward_r)
        
        # Time exit parameters
        self.max_hold_spin.setValue(params.max_hold)
        
        # Fibonacci parameters
        self.fib_lookback_spin.setValue(params.fib_swing_lookback)
        self.fib_lookahead_spin.setValue(params.fib_swing_lookahead)
        
        # Set Fib target combo
        for i in range(self.fib_target_combo.count()):
            if abs(self.fib_target_combo.itemData(i) - params.fib_target_level) < 0.001:
                self.fib_target_combo.setCurrentIndex(i)
                break
        
        # Transaction costs
        self.fee_bp_spin.setValue(params.fee_bp)
        self.slippage_bp_spin.setValue(params.slippage_bp)
    
    def save_params(self):
        """Save current parameters to file."""
        name, ok = QInputDialog.getText(
            self,
            "Save Parameters",
            "Enter a name for this parameter set:",
            text=datetime.now().strftime("params_%Y%m%d_%H%M%S")
        )
        
        if not ok or not name:
            return
        
        seller_params = self.get_seller_params()
        backtest_params = self.get_backtest_params()
        
        metadata = {
            "saved_by": "Strategy Editor",
            "description": "Manual parameter configuration"
        }
        
        try:
            filepath = params_store.save_params(
                seller_params,
                backtest_params,
                metadata,
                name
            )
            
            QMessageBox.information(
                self,
                "Success",
                f"Parameters saved to:\n{filepath}"
            )
            
            self.refresh_saved_list()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save parameters:\n{str(e)}"
            )
    
    def load_params(self):
        """Load parameters from selected file."""
        current_item = self.saved_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a parameter set to load")
            return
        
        self.on_saved_item_clicked(current_item)
    
    def on_saved_item_clicked(self, item: QListWidgetItem):
        """Load parameters when list item is double-clicked."""
        name = item.data(Qt.UserRole)
        
        try:
            data = params_store.load_params(name)
            
            self.set_seller_params(data["seller_params"])
            self.set_backtest_params(data["backtest_params"])
            
            self.current_seller_params = data["seller_params"]
            self.current_backtest_params = data["backtest_params"]
            
            self.params_loaded.emit(
                data["seller_params"],
                data["backtest_params"]
            )
            
            QMessageBox.information(
                self,
                "Success",
                f"Parameters loaded from:\n{name}.json"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load parameters:\n{str(e)}"
            )
    
    def delete_saved_params(self):
        """Delete selected parameter set."""
        current_item = self.saved_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a parameter set to delete")
            return
        
        name = current_item.data(Qt.UserRole)
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                filepath = params_store.storage_dir / f"{name}.json"
                filepath.unlink()
                QMessageBox.information(self, "Success", f"Deleted {name}")
                self.refresh_saved_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete:\n{str(e)}")
    
    def export_yaml(self):
        """Export current parameters to YAML format."""
        name, ok = QInputDialog.getText(
            self,
            "Export to YAML",
            "Enter filename (without extension):",
            text=datetime.now().strftime("params_%Y%m%d")
        )
        
        if not ok or not name:
            return
        
        seller_params = self.get_seller_params()
        backtest_params = self.get_backtest_params()
        
        try:
            filepath = params_store.export_to_yaml(
                seller_params,
                backtest_params,
                name
            )
            
            QMessageBox.information(
                self,
                "Success",
                f"Parameters exported to:\n{filepath}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to export:\n{str(e)}"
            )
    
    def reset_defaults(self):
        """Reset all parameters to defaults."""
        reply = QMessageBox.question(
            self,
            "Confirm Reset",
            "Reset all parameters to default values?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.set_seller_params(SellerParams())
            self.set_backtest_params(BacktestParams())
            self.params_changed.emit()
    
    def refresh_saved_list(self):
        """Refresh the list of saved parameter sets."""
        self.saved_list.clear()
        
        saved_params = params_store.list_saved_params()
        
        for params_info in saved_params:
            name = params_info["name"]
            saved_at = params_info.get("saved_at", "unknown")
            metadata = params_info.get("metadata", {})
            
            # Format list item text
            text = f"{name}"
            if saved_at != "unknown":
                try:
                    dt = datetime.fromisoformat(saved_at)
                    text += f" ({dt.strftime('%Y-%m-%d %H:%M')})"
                except:
                    pass
            
            if "generation" in metadata:
                text += f" [Gen {metadata['generation']}]"
            if "fitness" in metadata:
                text += f" [Fitness: {metadata['fitness']:.3f}]"
            
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, name)
            self.saved_list.addItem(item)
    
    def on_param_changed(self):
        """Handle parameter change events."""
        self.current_seller_params = self.get_seller_params()
        self.current_backtest_params = self.get_backtest_params()
        self.params_changed.emit()
    
    def set_golden_ratio(self):
        """Set Fibonacci target to Golden Ratio (61.8%)."""
        # Find the index for 0.618 (Golden Ratio)
        for i in range(self.fib_target_combo.count()):
            if abs(self.fib_target_combo.itemData(i) - 0.618) < 0.001:
                self.fib_target_combo.setCurrentIndex(i)
                break
        
        # Also enable Fibonacci exits if not already enabled
        self.use_fib_check.setChecked(True)
        
        # Show confirmation
        from PySide6.QtWidgets import QToolTip
        from PySide6.QtGui import QCursor
        QToolTip.showText(
            QCursor.pos(),
            "✓ Golden Ratio set: 61.8% Fibonacci target",
            self.golden_btn,
            self.golden_btn.rect(),
            2000  # Show for 2 seconds
        )
