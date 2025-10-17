from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QComboBox, QDateEdit, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, QDate, Signal
from core.models import Timeframe


# Timeframe mapping (multiplier, unit, label)
TIMEFRAMES = {
    "1m": (1, "minute", "1 minute"),
    "3m": (3, "minute", "3 minutes"),
    "5m": (5, "minute", "5 minutes"),
    "10m": (10, "minute", "10 minutes"),
    "15m": (15, "minute", "15 minutes"),
    "30m": (30, "minute", "30 minutes"),
    "60m": (60, "minute", "1 hour"),
    "4h": (240, "minute", "4 hours"),
    "12h": (720, "minute", "12 hours"),
    "1d": (1, "day", "1 day"),
}


class DataBar(QWidget):
    """Compact data download bar with ticker, timeframe, and date range selectors."""
    
    # Signals
    download_requested = Signal(str, str, str)  # from_date, to_date, timeframe_key
    timeframe_changed = Signal(str)  # Emits timeframe key like "15m"
    
    def __init__(self):
        super().__init__()
        self.setObjectName("dataBar")  # Apply styling from theme
        self.init_ui()
        self.load_defaults()
    
    def init_ui(self):
        """Initialize compact single-line data bar."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)
        
        # Data label and dropdown (hardcoded ADA)
        data_label = QLabel("Data:")
        data_label.setStyleSheet("font-weight: bold; color: #4caf50;")
        layout.addWidget(data_label)
        
        self.data_combo = QComboBox()
        self.data_combo.addItem("ADA/USDT (X:ADAUSD)")
        self.data_combo.setFixedWidth(120)
        self.data_combo.setEnabled(False)  # Hardcoded, can't change
        layout.addWidget(self.data_combo)
        
        # Timeframe label and dropdown
        timeframe_label = QLabel("Timeframe:")
        timeframe_label.setStyleSheet("font-weight: bold; color: #4caf50;")
        layout.addWidget(timeframe_label)
        
        self.timeframe_combo = QComboBox()
        for key, (mult, unit, label) in TIMEFRAMES.items():
            self.timeframe_combo.addItem(f"{label} ({key})", key)
        self.timeframe_combo.setCurrentText("15 minutes (15m)")
        self.timeframe_combo.setFixedWidth(140)
        self.timeframe_combo.currentIndexChanged.connect(self._on_timeframe_changed)
        layout.addWidget(self.timeframe_combo)
        
        # From Date label and input
        from_label = QLabel("From Date:")
        from_label.setStyleSheet("font-weight: bold; color: #4caf50;")
        layout.addWidget(from_label)
        
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate(2024, 1, 1))
        self.date_from.setDisplayFormat("yyyy-MM-dd")
        self.date_from.setFixedWidth(110)
        layout.addWidget(self.date_from)
        
        # To Date label and input
        to_label = QLabel("To Date:")
        to_label.setStyleSheet("font-weight: bold; color: #4caf50;")
        layout.addWidget(to_label)
        
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setDisplayFormat("yyyy-MM-dd")
        self.date_to.setFixedWidth(110)
        layout.addWidget(self.date_to)
        
        # "Now" button (set To Date to today)
        self.now_btn = QPushButton("Now")
        self.now_btn.setFixedWidth(60)
        self.now_btn.setFixedHeight(28)
        self.now_btn.clicked.connect(self._on_now_clicked)
        layout.addWidget(self.now_btn)
        
        # Download button (directly after Now button)
        self.download_btn = QPushButton("📥 Download")
        self.download_btn.setObjectName("primaryButton")
        self.download_btn.setFixedWidth(120)
        self.download_btn.setFixedHeight(28)
        self.download_btn.setStyleSheet("""
            QPushButton#primaryButton {
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
            }
            QPushButton#primaryButton:hover {
                font-size: 10px;
            }
            QPushButton#primaryButton:pressed {
                font-size: 10px;
            }
        """)
        self.download_btn.clicked.connect(self._on_download_clicked)
        layout.addWidget(self.download_btn)
        
        # Stretch to fill remaining space
        layout.addStretch()
    
    def load_defaults(self):
        """Load default values from settings."""
        from config.settings import settings
        
        try:
            # Set timeframe from settings
            tf_str = settings.timeframe
            if isinstance(tf_str, str):
                # Extract numeric part if it's a string like "15m"
                self.timeframe_combo.setCurrentText(tf_str)
            elif isinstance(tf_str, int):
                # Convert to timeframe key
                tf_map = {1: "1m", 3: "3m", 5: "5m", 10: "10m", 15: "15m", 30: "30m", 60: "60m"}
                key = tf_map.get(tf_str, "15m")
                for i in range(self.timeframe_combo.count()):
                    if self.timeframe_combo.itemData(i) == key:
                        self.timeframe_combo.setCurrentIndex(i)
                        break
            
            # Set date range from settings
            if settings.last_date_from:
                self.date_from.setDate(QDate.fromString(settings.last_date_from, "yyyy-MM-dd"))
            if settings.last_date_to:
                self.date_to.setDate(QDate.fromString(settings.last_date_to, "yyyy-MM-dd"))
        except Exception as e:
            print(f"Warning: Could not load data bar defaults: {e}")
    
    def get_timeframe_key(self) -> str:
        """Get current timeframe key (e.g., '15m')."""
        return self.timeframe_combo.currentData()
    
    def get_date_range(self) -> tuple[str, str]:
        """Get current date range as (from_date, to_date) strings."""
        from_str = self.date_from.date().toString("yyyy-MM-dd")
        to_str = self.date_to.date().toString("yyyy-MM-dd")
        return (from_str, to_str)
    
    def set_download_enabled(self, enabled: bool):
        """Enable/disable download button."""
        self.download_btn.setEnabled(enabled)
    
    def set_controls_enabled(self, enabled: bool):
        """Enable/disable all input controls during download."""
        self.timeframe_combo.setEnabled(enabled)
        self.date_from.setEnabled(enabled)
        self.date_to.setEnabled(enabled)
        self.download_btn.setEnabled(enabled)
    
    def _on_timeframe_changed(self):
        """Handle timeframe change."""
        self.timeframe_changed.emit(self.get_timeframe_key())
    
    def _on_now_clicked(self):
        """Handle 'Now' button click - set To Date to today."""
        self.date_to.setDate(QDate.currentDate())
    
    def _on_download_clicked(self):
        """Handle download button click."""
        from_date, to_date = self.get_date_range()
        tf_key = self.get_timeframe_key()
        self.download_requested.emit(from_date, to_date, tf_key)
