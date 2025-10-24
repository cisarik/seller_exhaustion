"""
Unified Optimization Coach Window - Enhanced Version

Dark Forest styled, fully responsive, signals-based real-time updates.
Supports both Classic Coach (deterministic) and OpenAI Agents (LLM) modes.
Eye-candy design matching main UI with white/green buttons and vibrant elements.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QGroupBox, QFormLayout, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QScrollArea, QFrame, QGridLayout, QPushButton,
    QMessageBox, QAbstractItemView
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QTimer as QtTimer
from PySide6.QtGui import QFont, QColor, QTextCursor, QBrush
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from app.theme import DARK_FOREST_QSS
from app.signals import get_coach_signals

# Dark Forest Color Palette
DARK_BG = "#0f1a12"
DARK_ACCENT = "#000"
TEXT_PRIMARY = "#e8f5e9"
TEXT_SECONDARY = "#b6e0bd"
BORDER_COLOR = "#2f5c39"
ACCENT_GREEN = "#4caf50"
BRIGHT_GREEN = "#2e7d32"
SELECTED_COLOR = "#295c33"


class OptimizationCoachWindowV2(QMainWindow):
    """Unified Optimization Coach window with signals-based real-time updates."""
    
    def __init__(self, coach_mode: str = 'classic', parent=None):
        super().__init__(parent)
        self.coach_mode = coach_mode
        self.setWindowTitle(f"🤖 Optimization Coach - {self._get_mode_display_name(coach_mode)}")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Apply dark forest theme
        self.setStyleSheet(DARK_FOREST_QSS)
        
        # Data storage
        self.coach_manager = None
        self.current_analysis_data = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.tool_calls_history: List[Dict[str, Any]] = []
        self.agent_requests: List[str] = []
        self.agent_responses: List[str] = []
        
        # Parent reference for UI updates (if available)
        self.parent_widget = parent
        
        # UI components
        self.init_ui()
        
        # Connect signals
        signals = get_coach_signals()
        signals.coach_message.connect(self._on_coach_message)
        signals.population_state_updated.connect(self._on_population_state_updated)
        signals.recommendation_updated.connect(self._on_recommendation_updated)
        signals.tool_call_complete.connect(self._on_tool_call_complete)
        signals.coach_mode_changed.connect(self._on_coach_mode_changed)
        
        # Connect to phase info signal if available (for progress bar updates)
        if hasattr(signals, 'phase_info_updated'):
            signals.phase_info_updated.connect(self._on_phase_info_and_update_progress)
        
        # Connect additional signals for detailed information (if available)
        if hasattr(signals, 'decision_reasoning_updated'):
            signals.decision_reasoning_updated.connect(self._on_decision_reasoning_updated)
        if hasattr(signals, 'phase_info_updated'):
            signals.phase_info_updated.connect(self._on_phase_info_updated)
        if hasattr(signals, 'crisis_detection_updated'):
            signals.crisis_detection_updated.connect(self._on_crisis_detection_updated)
        if hasattr(signals, 'learning_insights_updated'):
            signals.learning_insights_updated.connect(self._on_learning_insights_updated)
        
        # Auto-refresh timer for polling updates (fallback)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_display)
        self.refresh_timer.start(2000)  # Update every 2 seconds
    
    def _get_mode_display_name(self, mode: str) -> str:
        """Get display name for coach mode."""
        mode_names = {
            'classic': '🧠 Classic Coach',
            'openai': '🤖 OpenAI Agents',
            'disabled': '❌ Disabled'
        }
        return mode_names.get(mode, mode.title())
    
    def init_ui(self):
        """Initialize comprehensive UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Status bar
        self.create_status_bar()
        main_layout.addWidget(self.status_frame)
        
        # Main content with splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter, 1)
        
        # Left: Classic Coach tabs or Communication panel
        if self.coach_mode == 'classic':
            self.create_classic_coach_tabs()
            main_splitter.addWidget(self.classic_tabs_widget)
            main_splitter.setSizes([1600])
        else:
            self.create_communication_panel()
            main_splitter.addWidget(self.communication_widget)
            
            # Right: Tool history (only for OpenAI)
            self.create_tool_history_panel()
            main_splitter.addWidget(self.tool_history_widget)
            
            main_splitter.setSizes([600, 800])
        
        # Control buttons
        self.create_control_buttons()
        main_layout.addLayout(self.control_layout)
    
    def create_status_bar(self):
        """Create status bar with coach info."""
        self.status_frame = QFrame()
        self.status_frame.setFrameStyle(QFrame.StyledPanel)
        self.status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_ACCENT};
                border: 2px solid {BORDER_COLOR};
                border-radius: 6px;
            }}
        """)
        
        layout = QHBoxLayout(self.status_frame)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Coach status
        self.coach_status_label = QLabel(f"🤖 {self._get_mode_display_name(self.coach_mode)}: Ready")
        self.coach_status_label.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT_GREEN};
                font-weight: bold;
                font-size: 14px;
            }}
        """)
        layout.addWidget(self.coach_status_label)
        
        # Divider
        div_label = QLabel("·")
        div_label.setStyleSheet(f"color: {BORDER_COLOR};")
        layout.addWidget(div_label)
        
        # Counter labels
        self.analysis_counter_label = QLabel("Analyses: 0")
        self.analysis_counter_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(self.analysis_counter_label)
        
        div_label2 = QLabel("·")
        div_label2.setStyleSheet(f"color: {BORDER_COLOR};")
        layout.addWidget(div_label2)
        
        self.current_phase_label = QLabel("Phase: —")
        self.current_phase_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(self.current_phase_label)
        
        layout.addStretch()
        
        self.confidence_label = QLabel("Confidence: —")
        self.confidence_label.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 12px; font-weight: bold;")
        layout.addWidget(self.confidence_label)
    
    def create_classic_coach_tabs(self):
        """Create Classic Coach tabs widget."""
        self.classic_tabs_widget = QWidget()
        layout = QVBoxLayout(self.classic_tabs_widget)
        
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Current State Tab
        self._create_current_state_tab()
        
        # Decision Logic Tab
        self._create_decision_logic_tab()
        
        # Historical Analysis Tab
        self._create_historical_analysis_tab()
        
        # Phase Management Tab
        self._create_phase_management_tab()
        
        # Crisis Detection Tab
        self._create_crisis_detection_tab()
    
    def _create_current_state_tab(self):
        """Create current state tab."""
        tab = QWidget()
        self.tab_widget.addTab(tab, "📊 Current State")
        
        layout = QVBoxLayout(tab)
        
        # Metrics grid
        metrics_group = QGroupBox("Population State Analysis")
        metrics_layout = QGridLayout(metrics_group)
        
        # Row 1
        metrics_layout.addWidget(QLabel("Generation:"), 0, 0)
        self.gen_label = QLabel("—")
        self.gen_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        metrics_layout.addWidget(self.gen_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Population:"), 0, 2)
        self.pop_size_label = QLabel("—")
        metrics_layout.addWidget(self.pop_size_label, 0, 3)
        
        # Row 2
        metrics_layout.addWidget(QLabel("Best Fitness:"), 1, 0)
        self.best_fitness_label = QLabel("—")
        self.best_fitness_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        metrics_layout.addWidget(self.best_fitness_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Mean Fitness:"), 1, 2)
        self.mean_fitness_label = QLabel("—")
        metrics_layout.addWidget(self.mean_fitness_label, 1, 3)
        
        # Row 3
        metrics_layout.addWidget(QLabel("Diversity:"), 2, 0)
        self.diversity_label = QLabel("—")
        metrics_layout.addWidget(self.diversity_label, 2, 1)
        
        metrics_layout.addWidget(QLabel("Trend:"), 2, 2)
        self.trend_label = QLabel("—")
        metrics_layout.addWidget(self.trend_label, 2, 3)
        
        layout.addWidget(metrics_group)
        
        # Fitness history
        history_group = QGroupBox("Recent Fitness History (Last 10)")
        history_layout = QVBoxLayout(history_group)
        
        self.fitness_history_text = QTextEdit()
        self.fitness_history_text.setMaximumHeight(150)
        self.fitness_history_text.setReadOnly(True)
        self._style_textedit(self.fitness_history_text)
        history_layout.addWidget(self.fitness_history_text)
        
        layout.addWidget(history_group)
    
    def _create_decision_logic_tab(self):
        """Create decision logic tab."""
        tab = QWidget()
        self.tab_widget.addTab(tab, "🧠 Decision Logic")
        
        layout = QVBoxLayout(tab)
        
        # Recommendation group
        rec_group = QGroupBox("Current Recommendation")
        rec_layout = QVBoxLayout(rec_group)
        
        self.recommendation_text = QTextEdit()
        self.recommendation_text.setMaximumHeight(100)
        self.recommendation_text.setReadOnly(True)
        self._style_textedit(self.recommendation_text)
        rec_layout.addWidget(self.recommendation_text)
        
        layout.addWidget(rec_group)
        
        # Reasoning group
        reasoning_group = QGroupBox("Decision Reasoning")
        reasoning_layout = QVBoxLayout(reasoning_group)
        
        self.reasoning_text = QTextEdit()
        self.reasoning_text.setReadOnly(True)
        self._style_textedit(self.reasoning_text)
        reasoning_layout.addWidget(self.reasoning_text)
        
        layout.addWidget(reasoning_group)
        
        # Parameters table
        params_group = QGroupBox("Parameter Adjustments Applied")
        params_layout = QVBoxLayout(params_group)
        
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Old Value", "New Value"])
        self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._style_table(self.params_table)
        
        params_layout.addWidget(self.params_table)
        layout.addWidget(params_group)
    
    def _create_historical_analysis_tab(self):
        """Create historical analysis tab."""
        tab = QWidget()
        self.tab_widget.addTab(tab, "📈 Historical Analysis")
        
        layout = QVBoxLayout(tab)
        
        # History table
        history_group = QGroupBox("Recommendation History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Generation", "Algorithm", "Confidence", "Actions", "Result"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._style_table(self.history_table)
        
        history_layout.addWidget(self.history_table)
        layout.addWidget(history_group)
        
        # Insights group
        insights_group = QGroupBox("Learning Insights")
        insights_layout = QVBoxLayout(insights_group)
        
        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self._style_textedit(self.insights_text)
        insights_layout.addWidget(self.insights_text)
        
        layout.addWidget(insights_group)
    
    def _create_phase_management_tab(self):
        """Create phase management tab."""
        tab = QWidget()
        self.tab_widget.addTab(tab, "🔄 Phase Management")
        
        layout = QVBoxLayout(tab)
        
        # Phase boundaries
        boundaries_group = QGroupBox("Phase Boundaries")
        boundaries_layout = QGridLayout(boundaries_group)
        
        boundaries_layout.addWidget(QLabel("Exploration End:"), 0, 0)
        self.exploration_end_label = QLabel("—")
        boundaries_layout.addWidget(self.exploration_end_label, 0, 1)
        
        boundaries_layout.addWidget(QLabel("Exploitation End:"), 0, 2)
        self.exploitation_end_label = QLabel("—")
        boundaries_layout.addWidget(self.exploitation_end_label, 0, 3)
        
        layout.addWidget(boundaries_group)
        
        # Phase performance
        perf_group = QGroupBox("Phase Performance History")
        perf_layout = QVBoxLayout(perf_group)
        
        self.phase_perf_table = QTableWidget()
        self.phase_perf_table.setColumnCount(4)
        self.phase_perf_table.setHorizontalHeaderLabels([
            "Phase", "Generation", "Fitness", "Diversity"
        ])
        self.phase_perf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._style_table(self.phase_perf_table)
        
        perf_layout.addWidget(self.phase_perf_table)
        layout.addWidget(perf_group)
        
        # Convergence prediction
        pred_group = QGroupBox("Convergence Prediction")
        pred_layout = QVBoxLayout(pred_group)
        
        self.prediction_text = QTextEdit()
        self.prediction_text.setMaximumHeight(80)
        self.prediction_text.setReadOnly(True)
        self._style_textedit(self.prediction_text)
        pred_layout.addWidget(self.prediction_text)
        
        layout.addWidget(pred_group)
    
    def _create_crisis_detection_tab(self):
        """Create crisis detection tab."""
        tab = QWidget()
        self.tab_widget.addTab(tab, "🚨 Crisis Detection")
        
        layout = QVBoxLayout(tab)
        
        # Crisis status
        status_group = QGroupBox("Crisis Status")
        status_layout = QVBoxLayout(status_group)
        
        self.crisis_status_text = QTextEdit()
        self.crisis_status_text.setMaximumHeight(100)
        self.crisis_status_text.setReadOnly(True)
        self._style_textedit(self.crisis_status_text)
        status_layout.addWidget(self.crisis_status_text)
        
        layout.addWidget(status_group)
        
        # Crisis history
        hist_group = QGroupBox("Crisis Response History")
        hist_layout = QVBoxLayout(hist_group)
        
        self.crisis_table = QTableWidget()
        self.crisis_table.setColumnCount(4)
        self.crisis_table.setHorizontalHeaderLabels([
            "Generation", "Crisis Type", "Response", "Outcome"
        ])
        self.crisis_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._style_table(self.crisis_table)
        
        hist_layout.addWidget(self.crisis_table)
        layout.addWidget(hist_group)
    
    def create_communication_panel(self):
        """Create OpenAI communication panel."""
        self.communication_widget = QWidget()
        layout = QVBoxLayout(self.communication_widget)
        layout.setSpacing(10)
        
        # Request section
        request_group = QGroupBox("📤 Agent Request")
        request_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        request_layout = QVBoxLayout(request_group)
        
        self.request_text = QTextEdit()
        self.request_text.setPlaceholderText("Agent request will appear here...")
        self._style_textedit(self.request_text)
        self.request_text.setReadOnly(True)
        request_layout.addWidget(self.request_text, 1)
        
        layout.addWidget(request_group, 1)
        
        # Response section
        response_group = QGroupBox("📥 Agent Response")
        response_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        response_layout = QVBoxLayout(response_group)
        
        self.response_text = QTextEdit()
        self.response_text.setPlaceholderText("Agent response will appear here...")
        self._style_textedit(self.response_text)
        self.response_text.setReadOnly(True)
        response_layout.addWidget(self.response_text, 1)
        
        layout.addWidget(response_group, 1)
    
    def create_tool_history_panel(self):
        """Create tool history panel."""
        self.tool_history_widget = QWidget()
        layout = QVBoxLayout(self.tool_history_widget)
        layout.setSpacing(10)
        
        # Tool history group
        history_group = QGroupBox("🔧 Tool Call History")
        history_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        history_layout = QVBoxLayout(history_group)
        
        # Tool calls table
        self.tool_calls_table = QTableWidget()
        self.tool_calls_table.setColumnCount(5)
        self.tool_calls_table.setHorizontalHeaderLabels([
            "Time", "Tool Name", "Parameters", "Response", "Reason"
        ])
        
        header = self.tool_calls_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        self._style_table(self.tool_calls_table)
        self.tool_calls_table.itemSelectionChanged.connect(self.on_tool_call_selected)
        
        history_layout.addWidget(self.tool_calls_table)
        
        # Tool details
        self.tool_details = QTextEdit()
        self.tool_details.setPlaceholderText("Select a tool call to view details...")
        self._style_textedit(self.tool_details)
        self.tool_details.setReadOnly(True)
        history_layout.addWidget(self.tool_details, 1)
        
        layout.addWidget(history_group)
    
    def create_control_buttons(self):
        """Create control buttons."""
        self.control_layout = QHBoxLayout()
        
        # Refresh button
        self.refresh_btn = self._create_button("🔄 Refresh", "Manually refresh coach data", "#2196f3")
        self.refresh_btn.clicked.connect(self.update_display)
        self.control_layout.addWidget(self.refresh_btn)
        
        # Clear button
        self.clear_btn = self._create_button("🗑️ Clear", "Clear all history", "#f44336")
        self.clear_btn.clicked.connect(self.clear_history)
        self.control_layout.addWidget(self.clear_btn)
        
        self.control_layout.addStretch()
        
        # Export button
        self.export_btn = self._create_button("📤 Export", "Export coach data", "#2196f3")
        self.export_btn.clicked.connect(self.export_data)
        self.control_layout.addWidget(self.export_btn)
        
        # Close button
        self.close_btn = self._create_button("❌ Close", "Close window", "#666")
        self.close_btn.clicked.connect(self.close)
        self.control_layout.addWidget(self.close_btn)
    
    def _create_button(self, text: str, tooltip: str, color: str) -> QPushButton:
        """Create styled button (white/green theme matching main UI)."""
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        
        # Determine button type and style accordingly
        is_primary = color == "#2196f3" or "Refresh" in text or "Open" in text
        is_danger = "Clear" in text or "Delete" in text or color == "#f44336"
        is_close = "Close" in text or color == "#666"
        
        if is_primary or "🔍 Open" in text or "🔄 Refresh" in text:
            # Primary action: bright green
            style = f"""
                QPushButton {{
                    background-color: {BRIGHT_GREEN};
                    color: {TEXT_PRIMARY};
                    border: 2px solid {ACCENT_GREEN};
                    padding: 10px 18px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: {ACCENT_GREEN};
                    border-color: {TEXT_PRIMARY};
                    color: #000000;
                }}
                QPushButton:pressed {{
                    background-color: #1b5e20;
                    color: {ACCENT_GREEN};
                }}
            """
        elif is_danger:
            # Danger action: red-tinted
            style = f"""
                QPushButton {{
                    background-color: #3d2a2a;
                    color: #ffb3b3;
                    border: 2px solid #ff6b6b;
                    padding: 10px 18px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: #4d3a3a;
                    border-color: #ff8888;
                }}
                QPushButton:pressed {{
                    background-color: #2d1a1a;
                }}
            """
        elif is_close:
            # Close action: subtle grey
            style = f"""
                QPushButton {{
                    background-color: {BORDER_COLOR};
                    color: {TEXT_SECONDARY};
                    border: 1px solid {BORDER_COLOR};
                    padding: 10px 18px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: {SELECTED_COLOR};
                    border-color: {ACCENT_GREEN};
                    color: {ACCENT_GREEN};
                }}
                QPushButton:pressed {{
                    background-color: #1a2f1f;
                }}
            """
        else:
            # Secondary action: default
            style = f"""
                QPushButton {{
                    background-color: {BORDER_COLOR};
                    color: {TEXT_SECONDARY};
                    border: 1px solid {ACCENT_GREEN};
                    padding: 10px 18px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background-color: {SELECTED_COLOR};
                    border-color: {ACCENT_GREEN};
                    color: {ACCENT_GREEN};
                }}
                QPushButton:pressed {{
                    background-color: #1a2f1f;
                }}
            """
        
        btn.setStyleSheet(style)
        return btn
    
    @staticmethod
    def _lighten_color(color: str) -> str:
        """Lighten a color for hover effect. Handles both #RGB and #RRGGBB formats."""
        if color.startswith("#"):
            color = color[1:]
        
        # Handle short (#RGB) and long (#RRGGBB) hex colors
        if len(color) == 3:
            # Short format: #RGB → expand to #RRGGBB
            color = ''.join([c*2 for c in color])
        elif len(color) != 6:
            # Invalid format, return original
            return f"#{color}"
        
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        r = min(255, int(r * 1.2))
        g = min(255, int(g * 1.2))
        b = min(255, int(b * 1.2))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def _style_textedit(widget: QTextEdit):
        """Apply eye-candy Dark Forest styling to text edit."""
        widget.setStyleSheet(f"""
            QTextEdit {{
                background-color: {DARK_ACCENT};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }}
            QTextEdit:hover {{
                border: 1px solid {ACCENT_GREEN};
            }}
            QTextEdit:focus {{
                border: 2px solid {ACCENT_GREEN};
                background-color: rgba(0, 0, 0, 0.5);
            }}
        """)
    
    @staticmethod
    def _style_table(table: QTableWidget):
        """Apply eye-candy Dark Forest styling to table."""
        table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {DARK_ACCENT};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                gridline-color: {BORDER_COLOR};
                alternate-background-color: rgba(78, 175, 80, 0.1);
            }}
            QTableWidget::item {{
                padding: 6px 8px;
                border-bottom: 1px solid {BORDER_COLOR};
            }}
            QTableWidget::item:selected {{
                background-color: {SELECTED_COLOR};
                color: {ACCENT_GREEN};
                font-weight: bold;
            }}
            QTableWidget::item:hover {{
                background-color: {SELECTED_COLOR};
            }}
            QHeaderView::section {{
                background-color: {BORDER_COLOR};
                color: {ACCENT_GREEN};
                padding: 8px 6px;
                border: 1px solid {BORDER_COLOR};
                font-weight: bold;
                text-align: center;
            }}
            QHeaderView::section:hover {{
                background-color: {SELECTED_COLOR};
            }}
        """)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
    
    # Signal handlers
    def _on_coach_message(self, message: str, msg_type: str):
        """Handle coach message signal."""
        if self.coach_mode == 'openai':
            color = "#2196f3" if msg_type == "blue" else "#ffffff"
            if msg_type == "info":
                color = "#4caf50"
            elif msg_type == "warning":
                color = "#ff9800"
            elif msg_type == "error":
                color = "#f44336"
            
            cursor = self.response_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.response_text.setTextCursor(cursor)
            
            fmt = self.response_text.currentCharFormat()
            fmt.setForeground(QColor(color))
            self.response_text.setCurrentCharFormat(fmt)
            self.response_text.insertPlainText(f"{message}\n")
        elif self.coach_mode == 'classic':
            # For Classic Coach, add blue messages to reasoning tab
            if msg_type == "blue":
                cursor = self.reasoning_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.reasoning_text.setTextCursor(cursor)
                self.reasoning_text.insertPlainText(f"✓ {message}\n")
    
    def _on_population_state_updated(self, data: dict):
        """Handle population state update - populate Current State and other tabs."""
        if self.coach_mode != 'classic':
            return
        
        # Current State Tab
        self.gen_label.setText(str(data.get('generation', '—')))
        self.pop_size_label.setText(str(data.get('population_size', '—')))
        self.best_fitness_label.setText(f"{data.get('best_fitness', 0):.4f}")
        self.mean_fitness_label.setText(f"{data.get('mean_fitness', 0):.4f}")
        self.diversity_label.setText(f"{data.get('diversity', 0):.3f}")
        trend = data.get('fitness_trend', '—')
        trend_icon = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
        self.trend_label.setText(f"{trend_icon} {trend}")
        
        # Update fitness history
        if 'fitness_history' in data and data['fitness_history']:
            history_text = "Recent Fitness: " + ", ".join(
                [f"{f:.4f}" for f in data['fitness_history'][-10:]]
            )
            self.fitness_history_text.setPlainText(history_text)
        
        # Update Phase Management tab with current generation info
        if 'generation' in data:
            generation = data['generation']
            # Will be populated by phase_info signal
    
    def _on_recommendation_updated(self, data: dict):
        """Handle recommendation update - populate Decision Logic and History tabs."""
        if self.coach_mode != 'classic':
            return
        
        # Decision Logic Tab
        algorithm = data.get('recommended_algorithm', 'unknown').upper()
        confidence = data.get('confidence', 0)
        icon = "🎯" if confidence > 0.7 else "🤔" if confidence > 0.5 else "❓"
        
        rec_text = f"{icon} {algorithm}\nConfidence: {confidence:.2f}"
        self.recommendation_text.setPlainText(rec_text)
        
        # Update confidence in status bar
        self.confidence_label.setText(f"Confidence: {confidence:.1%}")
        
        # Update analysis counter in status bar
        generation = data.get('generation', 0)
        current_count = int(self.analysis_counter_label.text().split(': ')[1])
        self.analysis_counter_label.setText(f"Analyses: {current_count + 1}")
        
        # Add to History table
        actions_count = data.get('actions', 0)
        
        row_count = self.history_table.rowCount()
        self.history_table.insertRow(row_count)
        
        self.history_table.setItem(row_count, 0, QTableWidgetItem(str(generation)))
        self.history_table.setItem(row_count, 1, QTableWidgetItem(algorithm))
        self.history_table.setItem(row_count, 2, QTableWidgetItem(f"{confidence:.2f}"))
        self.history_table.setItem(row_count, 3, QTableWidgetItem(str(actions_count)))
        self.history_table.setItem(row_count, 4, QTableWidgetItem("✓ Applied"))
        
        self.history_table.scrollToBottom()
        
        # Keep history to last 50 entries
        while self.history_table.rowCount() > 50:
            self.history_table.removeRow(0)
    
    def _on_tool_call_complete(self, tool_name: str, parameters: dict, response: dict):
        """Handle tool call completion."""
        if self.coach_mode == 'openai':
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            tool_call = {
                'timestamp': timestamp,
                'name': tool_name,
                'parameters': parameters,
                'response': response,
                'reason': ''
            }
            
            self.tool_calls_history.append(tool_call)
            self.update_tool_calls_table()
    
    def _on_coach_mode_changed(self, mode: str):
        """Handle coach mode change."""
        if mode != self.coach_mode:
            self.coach_mode = mode
            self.setWindowTitle(f"🤖 Optimization Coach - {self._get_mode_display_name(mode)}")
            self.update_display()
    
    def _on_decision_reasoning_updated(self, data: dict):
        """Handle decision reasoning update."""
        if self.coach_mode != 'classic':
            return
        
        reasoning_text = data.get('reasoning', '')
        self.reasoning_text.setPlainText(reasoning_text)
        
        # Update parameter adjustments table
        if 'parameters_adjusted' in data:
            params = data['parameters_adjusted']
            
            # Determine table height based on number of params
            row_count_needed = len(params)
            if row_count_needed > 0:
                self.params_table.setRowCount(row_count_needed)
                
                for row, (param_name, value_pair) in enumerate(params.items()):
                    # Handle both tuple format (old_val, new_val) and string format
                    if isinstance(value_pair, (tuple, list)):
                        old_val, new_val = value_pair
                    else:
                        old_val = "—"
                        new_val = str(value_pair)
                    
                    # Parameter name
                    self.params_table.setItem(row, 0, QTableWidgetItem(str(param_name)))
                    # Old value
                    self.params_table.setItem(row, 1, QTableWidgetItem(str(old_val)))
                    # New value
                    self.params_table.setItem(row, 2, QTableWidgetItem(str(new_val)))
                
                # Auto-resize columns to content
                self.params_table.resizeColumnsToContents()
            else:
                # No parameters changed
                self.params_table.setRowCount(1)
                self.params_table.setItem(0, 0, QTableWidgetItem("(no parameters adjusted)"))
                self.params_table.setItem(0, 1, QTableWidgetItem("—"))
                self.params_table.setItem(0, 2, QTableWidgetItem("—"))
    
    def _on_phase_info_updated(self, data: dict):
        """Handle phase information update."""
        if self.coach_mode != 'classic':
            return
        
        # Phase Management tab
        self.exploration_end_label.setText(str(data.get('exploration_end', '—')))
        self.exploitation_end_label.setText(str(data.get('exploitation_end', '—')))
        
        # Update prediction text
        current_phase = data.get('current_phase', '—')
        prediction_text = f"""Current Phase: {current_phase}
Generation Range: {data.get('phase_start', '—')} - {data.get('phase_end', '—')}
Estimated Remaining: {data.get('estimated_remaining', '—')} generations
Next Phase Transition: Gen {data.get('next_transition', '—')}"""
        self.prediction_text.setPlainText(prediction_text)
        
        # Update phase in status bar
        self.current_phase_label.setText(f"Phase: {current_phase}")
        
        # Update progress bar in parent widget
        self._update_progress_bar_from_phase_data(data)
    
    def _on_phase_info_and_update_progress(self, data: dict):
        """Handle phase info and update progress bar."""
        self._on_phase_info_updated(data)
    
    def _update_progress_bar_from_phase_data(self, data: dict):
        """Update the phase progress bar in parent widget."""
        if not self.parent_widget:
            return
        
        try:
            # Get current generation from data
            current_gen = data.get('generation', data.get('phase_start', 0))
            
            # Get total generations (use estimated_remaining to calculate)
            phase_end = data.get('phase_end', 0)
            estimated_remaining = data.get('estimated_remaining', 0)
            
            # Try to calculate total
            try:
                if isinstance(estimated_remaining, str):
                    est_remaining_val = int(estimated_remaining) if estimated_remaining != '—' else 0
                else:
                    est_remaining_val = estimated_remaining
                total_gens = phase_end + est_remaining_val
            except (ValueError, TypeError):
                total_gens = phase_end or 100
            
            # Update progress bar
            exploration_end = data.get('exploration_end', 25)
            exploitation_end = data.get('exploitation_end', 50)
            
            # Get analysis counts from data
            current_analysis_count = data.get('current_analysis_count', 0)
            total_analysis_count = data.get('total_analysis_count', 1)
            
            # parent_widget IS the CompactParamsEditor (param_editor)
            if hasattr(self.parent_widget, 'update_phase_progress'):
                print(f"🔧 Updating progress bar: gen={current_gen}, analysis={current_analysis_count}/{total_analysis_count}")
                self.parent_widget.update_phase_progress(
                    current_gen, total_gens, exploration_end, exploitation_end,
                    current_analysis_count, total_analysis_count
                )
        except Exception as e:
            # Silently fail if progress bar update fails
            pass
    
    def _on_crisis_detection_updated(self, data: dict):
        """Handle crisis detection update."""
        if self.coach_mode != 'classic':
            return
        
        # Crisis Detection tab
        crises = data.get('active_crises', [])
        status_text = f"Status: {'🚨 CRISIS DETECTED' if crises else '✓ Normal'}\n\n"
        status_text += "Active Crises:\n"
        
        for crisis in crises:
            status_text += f"• {crisis['type']}: {crisis['description']}\n"
        
        self.crisis_status_text.setPlainText(status_text)
        
        # Add to Crisis table if there are crises
        if crises:
            generation = data.get('generation', 0)
            
            for crisis in crises:
                row_count = self.crisis_table.rowCount()
                self.crisis_table.insertRow(row_count)
                
                self.crisis_table.setItem(row_count, 0, QTableWidgetItem(str(generation)))
                self.crisis_table.setItem(row_count, 1, QTableWidgetItem(crisis['type']))
                self.crisis_table.setItem(row_count, 2, QTableWidgetItem(crisis.get('response', '—')))
                self.crisis_table.setItem(row_count, 3, QTableWidgetItem(crisis.get('outcome', 'pending')))
            
            # Keep table to last 30 entries
            while self.crisis_table.rowCount() > 30:
                self.crisis_table.removeRow(0)
    
    def _on_learning_insights_updated(self, data: dict):
        """Handle learning insights update."""
        if self.coach_mode != 'classic':
            return
        
        insights_text = data.get('insights', '')
        self.insights_text.setPlainText(insights_text)
    
    def on_tool_call_selected(self):
        """Handle tool call selection."""
        selected = self.tool_calls_table.selectedItems()
        if not selected:
            return
        
        row = selected[0].row()
        if row < len(self.tool_calls_history):
            tool_call = self.tool_calls_history[row]
            details = f"Tool: {tool_call['name']}\n"
            details += f"Time: {tool_call['timestamp']}\n"
            details += f"Parameters: {json.dumps(tool_call['parameters'], indent=2)}\n"
            details += f"Response: {json.dumps(tool_call['response'], indent=2)}\n"
            
            self.tool_details.setPlainText(details)
    
    def update_tool_calls_table(self):
        """Update tool calls table."""
        self.tool_calls_table.setRowCount(len(self.tool_calls_history))
        
        for row, tool_call in enumerate(self.tool_calls_history):
            self.tool_calls_table.setItem(row, 0, QTableWidgetItem(tool_call['timestamp']))
            self.tool_calls_table.setItem(row, 1, QTableWidgetItem(tool_call['name']))
            
            params_str = json.dumps(tool_call['parameters'], separators=(',', ':'))
            if len(params_str) > 100:
                params_str = params_str[:100] + "..."
            self.tool_calls_table.setItem(row, 2, QTableWidgetItem(params_str))
            
            response_str = json.dumps(tool_call['response'], separators=(',', ':'))
            if len(response_str) > 100:
                response_str = response_str[:100] + "..."
            self.tool_calls_table.setItem(row, 3, QTableWidgetItem(response_str))
            
            self.tool_calls_table.setItem(row, 4, QTableWidgetItem(tool_call['reason']))
        
        self.tool_calls_table.scrollToBottom()
    
    def set_coach_manager(self, coach_manager):
        """Set coach manager reference."""
        self.coach_manager = coach_manager
    
    def update_display(self):
        """Update display from coach manager."""
        # Real-time updates are handled by signals
        pass
    
    def clear_history(self):
        """Clear all history."""
        self.analysis_history.clear()
        self.tool_calls_history.clear()
        
        if hasattr(self, 'fitness_history_text'):
            self.fitness_history_text.clear()
        if hasattr(self, 'recommendation_text'):
            self.recommendation_text.clear()
        if hasattr(self, 'tool_calls_table'):
            self.tool_calls_table.setRowCount(0)
        
        QMessageBox.information(self, "History Cleared", "All coach history has been cleared.")
    
    def export_data(self):
        """Export coach data."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Coach Data",
            f"coach_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'coach_mode': self.coach_mode,
                    'analysis_history': self.analysis_history,
                    'tool_calls_history': self.tool_calls_history,
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                QMessageBox.information(self, "Export Successful", f"Data exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export: {str(e)}")
    
    def closeEvent(self, event):
        """Handle window close."""
        self.refresh_timer.stop()
        event.accept()
