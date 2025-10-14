DARK_FOREST_QSS = r"""
QDialog, QWidget { background-color: #0f1a12; color: #e8f5e9; }
QLabel { color: #e8f5e9; }
.form-label, QLabel[variant="secondary"] { color: #b6e0bd; font-weight: bold; }
QTabWidget::pane { border: 1px solid #2f5c39; background: #0f1a12; }
QTabBar::tab { background:#1a2f1f; color:#b6e0bd; padding:8px 16px; border:1px solid #2f5c39; margin-right:2px; }
QTabBar::tab:selected { background:#295c33; color:#e8f5e9; font-weight:bold; border-bottom:2px solid #4caf50; }
QTabBar::tab:hover { background:#213f29; }
QPushButton { padding:8px 16px; font-size:12px; border-radius:4px; background:#182c1d; color:#b6e0bd; border:1px solid #2f5c39; }
QPushButton:hover { background:#213f29; border-color:#4caf50; }
QPushButton:pressed { background:#152820; }
QPushButton#primaryButton { padding:10px 20px; font-size:13px; font-weight:bold; border-radius:6px; background:#2e7d32; color:#e8f5e9; border:2px solid #4caf50; }
QPushButton#primaryButton:hover { background:#388e3c; }
QPushButton#primaryButton:pressed { background:#1b5e20; }
QPushButton[danger="true"] { background:#2c1d1d; color:#e0b6b6; border:1px solid #5c2f2f; }
QPushButton[danger="true"]:hover { background:#3f2929; border-color:#af4c4c; }
QLineEdit, QTextEdit, QComboBox { background:#000; color:#e8f5e9; border:1px solid #2f5c39; border-radius:4px; padding:6px 8px; }
QLineEdit:hover, QTextEdit:hover, QComboBox:hover { border-color:#4caf50; background:#0a0a0a; }
QLineEdit:focus, QTextEdit:focus, QComboBox:focus { border-color:#4caf50; background:#0a0a0a; }
QComboBox QAbstractItemView { background:#000; color:#e8f5e9; selection-background-color:#295c33; }
QLabel[role="title"] { font-size:18px; font-weight:bold; color:#e8f5e9; padding:8px 0px; }
QLabel[role="statusbar"] { color:#4caf50; font-size:14px; font-weight:bold; padding:12px 16px; background:#000; border-top:1px solid #2f5c39; }
QLabel[variant="warn"] { background:#3d2a0f; color:#ffd54f; padding:10px; border:2px solid #ff9800; border-radius:6px; font-weight:bold; }
"""
