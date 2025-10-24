"""
Optimization Coach Window - Unified Interface for All Coach Modes

Combines Classic Coach and Evolution Coach windows into a single interface
that dynamically shows content based on the selected coach mode.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QFormLayout, QLabel, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QScrollArea,
    QFrame, QGridLayout, QTabWidget, QPushButton, QStackedWidget
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QPalette
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from app.theme import DARK_FOREST_QSS


class OptimizationCoachWindow(QMainWindow):
    """
    Unified Optimization Coach window that displays different content
    based on the selected coach mode (Classic, OpenAI Agents, or Disabled).
    """

    def __init__(self, coach_mode: str = 'classic', parent=None):
        super().__init__(parent)
        self.coach_mode = coach_mode
        self.setWindowTitle(f"🤖 Optimization Coach - {self._get_mode_display_name(coach_mode)}")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        # Apply dark forest theme
        self.setStyleSheet(DARK_FOREST_QSS)

        # Data storage
        self.coach_manager = None  # Will be set by stats_panel
        self.current_analysis_data = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.tool_calls_history: List[Dict[str, Any]] = []
        self.agent_requests: List[str] = []
        self.agent_responses: List[str] = []

        # UI components
        self.init_ui()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_display)
        self.refresh_timer.start(2000)  # Update every 2 seconds

        # Connect to coach mode changes from parent (compact_params)
        if parent and hasattr(parent, 'coach_mode_combo'):
            parent.coach_mode_combo.currentIndexChanged.connect(self._on_coach_mode_changed)
            # Set initial mode from parent
            self._on_coach_mode_changed(0)

    def _on_coach_mode_changed(self, index):
        """Handle coach mode change from parent dropdown."""
        if self.parent() and hasattr(self.parent(), 'coach_mode_combo'):
            new_mode = self.parent().coach_mode_combo.currentData()
            if new_mode != self.coach_mode:
                self.coach_mode = new_mode
                self.update_content_for_mode()
                self.update_display()

    def _get_mode_display_name(self, mode: str) -> str:
        """Get display name for coach mode."""
        mode_names = {
            'classic': 'Classic Coach',
            'openai': 'OpenAI Agents',
            'disabled': 'Disabled'
        }
        return mode_names.get(mode, mode.title())

    def init_ui(self):
        """Initialize the comprehensive UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Status bar
        self.create_status_bar()
        main_layout.addWidget(self.status_frame)

        # Main content area - stacked widget for different modes
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack)

        # Create content widgets for each mode
        self.create_classic_coach_content()
        self.create_openai_coach_content()
        self.create_disabled_coach_content()

        # Set initial content based on mode
        self.update_content_for_mode()

        # Control buttons
        self.create_control_buttons()
        main_layout.addLayout(self.control_layout)

    def create_status_bar(self):
        """Create status bar with coach state info."""
        self.status_frame = QFrame()
        self.status_frame.setFrameStyle(QFrame.StyledPanel)
        self.status_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 5px;
            }
        """)

        layout = QHBoxLayout(self.status_frame)
        layout.setContentsMargins(10, 5, 10, 5)

        # Coach status
        self.coach_status_label = QLabel(f"🤖 {self._get_mode_display_name(self.coach_mode)}: Ready")
        self.coach_status_label.setStyleSheet("""
            QLabel {
                color: #4caf50;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.coach_status_label)

        # Analysis counter
        self.analysis_counter_label = QLabel("Analyses: 0")
        self.analysis_counter_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.analysis_counter_label)

        # Current phase (for Classic mode)
        self.current_phase_label = QLabel("Phase: None")
        self.current_phase_label.setStyleSheet("color: #2196f3; font-size: 12px;")
        layout.addWidget(self.current_phase_label)

        layout.addStretch()

        # Confidence indicator
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("color: #ff9800; font-size: 12px;")
        layout.addWidget(self.confidence_label)

    def create_classic_coach_content(self):
        """Create content for Classic Coach mode."""
        self.classic_widget = QWidget()
        layout = QVBoxLayout(self.classic_widget)

        # Main tab widget for Classic Coach
        self.classic_tab_widget = QTabWidget()
        layout.addWidget(self.classic_tab_widget)

        # Create tabs
        self.create_current_state_tab()
        self.create_decision_logic_tab()
        self.create_historical_analysis_tab()
        self.create_phase_management_tab()
        self.create_crisis_detection_tab()

        self.content_stack.addWidget(self.classic_widget)

    def create_openai_coach_content(self):
        """Create content for OpenAI Agents mode."""
        self.openai_widget = QWidget()
        layout = QVBoxLayout(self.openai_widget)

        # Main splitter for OpenAI Coach
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)

        # Left panel - Communication
        self.create_communication_panel()
        main_splitter.addWidget(self.communication_widget)

        # Right panel - Tool History
        self.create_tool_history_panel()
        main_splitter.addWidget(self.tool_history_widget)

        # Set splitter proportions
        main_splitter.setSizes([600, 800])

        self.content_stack.addWidget(self.openai_widget)

    def create_disabled_coach_content(self):
        """Create content for Disabled mode."""
        self.disabled_widget = QWidget()
        layout = QVBoxLayout(self.disabled_widget)

        # Centered message
        message_label = QLabel("🤖 Optimization Coach is currently disabled.\n\n"
                              "To enable coach analysis:\n"
                              "• Select '🧠 Classic Coach' for deterministic analysis\n"
                              "• Select '🤖 OpenAI Agents' for LLM-based analysis\n\n"
                              "Classic Coach requires no API keys and provides fast, deterministic optimization guidance.\n"
                              "OpenAI Agents provides intelligent LLM-based analysis but requires API keys.")
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 16px;
                padding: 40px;
                border: 2px dashed #555;
                border-radius: 10px;
                background-color: #1e1e1e;
            }
        """)
        layout.addWidget(message_label, stretch=1)

        self.content_stack.addWidget(self.disabled_widget)

    def update_content_for_mode(self):
        """Update displayed content based on current coach mode."""
        if self.coach_mode == 'classic':
            self.content_stack.setCurrentWidget(self.classic_widget)
            self.current_phase_label.setVisible(True)
        elif self.coach_mode == 'openai':
            self.content_stack.setCurrentWidget(self.openai_widget)
            self.current_phase_label.setVisible(False)
        else:  # disabled
            self.content_stack.setCurrentWidget(self.disabled_widget)
            self.current_phase_label.setVisible(False)

        # Update window title
        self.setWindowTitle(f"🤖 Optimization Coach - {self._get_mode_display_name(self.coach_mode)}")

    # Classic Coach tab methods (copied from classic_coach_window.py)
    def create_current_state_tab(self):
        """Create tab showing current population state analysis."""
        tab = QWidget()
        self.classic_tab_widget.addTab(tab, "📊 Current State")

        layout = QVBoxLayout(tab)

        # Population metrics
        metrics_group = QGroupBox("Population State Analysis")
        metrics_layout = QGridLayout(metrics_group)

        # Row 1: Basic metrics
        metrics_layout.addWidget(QLabel("Generation:"), 0, 0)
        self.gen_label = QLabel("--")
        metrics_layout.addWidget(self.gen_label, 0, 1)

        metrics_layout.addWidget(QLabel("Population Size:"), 0, 2)
        self.pop_size_label = QLabel("--")
        metrics_layout.addWidget(self.pop_size_label, 0, 3)

        # Row 2: Fitness metrics
        metrics_layout.addWidget(QLabel("Best Fitness:"), 1, 0)
        self.best_fitness_label = QLabel("--")
        self.best_fitness_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        metrics_layout.addWidget(self.best_fitness_label, 1, 1)

        metrics_layout.addWidget(QLabel("Mean Fitness:"), 1, 2)
        self.mean_fitness_label = QLabel("--")
        metrics_layout.addWidget(self.mean_fitness_label, 1, 3)

        # Row 3: Diversity and trend
        metrics_layout.addWidget(QLabel("Diversity:"), 2, 0)
        self.diversity_label = QLabel("--")
        metrics_layout.addWidget(self.diversity_label, 2, 1)

        metrics_layout.addWidget(QLabel("Fitness Trend:"), 2, 2)
        self.trend_label = QLabel("--")
        metrics_layout.addWidget(self.trend_label, 2, 3)

        # Row 4: State counters
        metrics_layout.addWidget(QLabel("Stagnation Count:"), 3, 0)
        self.stagnation_label = QLabel("--")
        metrics_layout.addWidget(self.stagnation_label, 3, 1)

        metrics_layout.addWidget(QLabel("Low Diversity Count:"), 3, 2)
        self.low_diversity_label = QLabel("--")
        metrics_layout.addWidget(self.low_diversity_label, 3, 3)

        layout.addWidget(metrics_group)

        # Fitness history chart (text-based)
        history_group = QGroupBox("Recent Fitness History (Last 10)")
        history_layout = QVBoxLayout(history_group)

        self.fitness_history_text = QTextEdit()
        self.fitness_history_text.setMaximumHeight(150)
        self.fitness_history_text.setReadOnly(True)
        self.fitness_history_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        history_layout.addWidget(self.fitness_history_text)

        layout.addWidget(history_group)

    def create_decision_logic_tab(self):
        """Create tab showing decision logic and recommendations."""
        tab = QWidget()
        self.classic_tab_widget.addTab(tab, "🧠 Decision Logic")

        layout = QVBoxLayout(tab)

        # Current recommendation
        recommendation_group = QGroupBox("Current Recommendation")
        rec_layout = QVBoxLayout(recommendation_group)

        self.recommendation_text = QTextEdit()
        self.recommendation_text.setMaximumHeight(100)
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        rec_layout.addWidget(self.recommendation_text)

        layout.addWidget(recommendation_group)

        # Decision reasoning
        reasoning_group = QGroupBox("Decision Reasoning")
        reasoning_layout = QVBoxLayout(reasoning_group)

        self.reasoning_text = QTextEdit()
        self.reasoning_text.setReadOnly(True)
        self.reasoning_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        reasoning_layout.addWidget(self.reasoning_text)

        layout.addWidget(reasoning_group)

        # Parameter adjustments
        params_group = QGroupBox("Parameter Adjustments Applied")
        params_layout = QVBoxLayout(params_group)

        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Old Value", "New Value"])
        self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.params_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                gridline-color: #555;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #333;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555;
                font-weight: bold;
            }
        """)
        self.params_table.setAlternatingRowColors(True)
        params_layout.addWidget(self.params_table)

        layout.addWidget(params_group)

    def create_historical_analysis_tab(self):
        """Create tab showing historical analysis and learning."""
        tab = QWidget()
        self.classic_tab_widget.addTab(tab, "📈 Historical Analysis")

        layout = QVBoxLayout(tab)

        # Recommendations history
        history_group = QGroupBox("Recommendation History")
        history_layout = QVBoxLayout(history_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Generation", "Algorithm", "Confidence", "Actions", "Result"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                gridline-color: #555;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #333;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555;
                font-weight: bold;
            }
        """)
        self.history_table.setAlternatingRowColors(True)
        history_layout.addWidget(self.history_table)

        layout.addWidget(history_group)

        # Learning insights
        insights_group = QGroupBox("Learning Insights")
        insights_layout = QVBoxLayout(insights_group)

        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        insights_layout.addWidget(self.insights_text)

        layout.addWidget(insights_group)

    def create_phase_management_tab(self):
        """Create tab showing phase management and transitions."""
        tab = QWidget()
        self.classic_tab_widget.addTab(tab, "🔄 Phase Management")

        layout = QVBoxLayout(tab)

        # Phase boundaries
        boundaries_group = QGroupBox("Phase Boundaries")
        boundaries_layout = QGridLayout(boundaries_group)

        boundaries_layout.addWidget(QLabel("Exploration End:"), 0, 0)
        self.exploration_end_label = QLabel("--")
        boundaries_layout.addWidget(self.exploration_end_label, 0, 1)

        boundaries_layout.addWidget(QLabel("Exploitation End:"), 0, 2)
        self.exploitation_end_label = QLabel("--")
        boundaries_layout.addWidget(self.exploitation_end_label, 0, 3)

        boundaries_layout.addWidget(QLabel("Planned vs Actual:"), 1, 0)
        self.phase_comparison_label = QLabel("--")
        boundaries_layout.addWidget(self.phase_comparison_label, 1, 1, 1, 3)

        layout.addWidget(boundaries_group)

        # Phase performance
        performance_group = QGroupBox("Phase Performance History")
        perf_layout = QVBoxLayout(performance_group)

        self.phase_perf_table = QTableWidget()
        self.phase_perf_table.setColumnCount(4)
        self.phase_perf_table.setHorizontalHeaderLabels([
            "Phase", "Generation", "Fitness", "Diversity"
        ])
        self.phase_perf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.phase_perf_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                gridline-color: #555;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #333;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555;
                font-weight: bold;
            }
        """)
        self.phase_perf_table.setAlternatingRowColors(True)
        perf_layout.addWidget(self.phase_perf_table)

        layout.addWidget(performance_group)

        # Convergence prediction
        prediction_group = QGroupBox("Convergence Prediction")
        pred_layout = QVBoxLayout(prediction_group)

        self.prediction_text = QTextEdit()
        self.prediction_text.setMaximumHeight(80)
        self.prediction_text.setReadOnly(True)
        self.prediction_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        pred_layout.addWidget(self.prediction_text)

        layout.addWidget(prediction_group)

    def create_crisis_detection_tab(self):
        """Create tab showing crisis detection and response."""
        tab = QWidget()
        self.classic_tab_widget.addTab(tab, "🚨 Crisis Detection")

        layout = QVBoxLayout(tab)

        # Crisis status
        status_group = QGroupBox("Crisis Status")
        status_layout = QVBoxLayout(status_group)

        self.crisis_status_text = QTextEdit()
        self.crisis_status_text.setMaximumHeight(100)
        self.crisis_status_text.setReadOnly(True)
        self.crisis_status_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        status_layout.addWidget(self.crisis_status_text)

        layout.addWidget(status_group)

        # Crisis history
        history_group = QGroupBox("Crisis Response History")
        hist_layout = QVBoxLayout(history_group)

        self.crisis_table = QTableWidget()
        self.crisis_table.setColumnCount(4)
        self.crisis_table.setHorizontalHeaderLabels([
            "Generation", "Crisis Type", "Response", "Outcome"
        ])
        self.crisis_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.crisis_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                gridline-color: #555;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #333;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555;
                font-weight: bold;
            }
        """)
        self.crisis_table.setAlternatingRowColors(True)
        hist_layout.addWidget(self.crisis_table)

        layout.addWidget(history_group)

    # OpenAI Coach panel methods (copied from evolution_coach_window.py)
    def create_communication_panel(self):
        """Create communication panel with request/response textfields."""
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
        self.request_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
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
        self.response_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
        self.response_text.setReadOnly(True)
        response_layout.addWidget(self.response_text, 1)

        layout.addWidget(response_group, 1)

    def create_tool_history_panel(self):
        """Create tool history panel with table."""
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

        # Configure table
        header = self.tool_calls_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Time
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Tool Name
        header.setSectionResizeMode(2, QHeaderView.Stretch)          # Parameters
        header.setSectionResizeMode(3, QHeaderView.Stretch)          # Response
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents) # Reason

        self.tool_calls_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                gridline-color: #555;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #333;
            }
            QTableWidget::item:selected {
                background-color: #4caf50;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555;
                font-weight: bold;
            }
        """)

        self.tool_calls_table.setAlternatingRowColors(True)
        self.tool_calls_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tool_calls_table.itemSelectionChanged.connect(self.on_tool_call_selected)

        history_layout.addWidget(self.tool_calls_table)

        # Tool call details
        self.tool_details = QTextEdit()
        self.tool_details.setPlaceholderText("Select a tool call to view details...")
        self.tool_details.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        self.tool_details.setReadOnly(True)
        history_layout.addWidget(self.tool_details, 1)

        layout.addWidget(history_group)

    def create_control_buttons(self):
        """Create control buttons."""
        self.control_layout = QHBoxLayout()

        # Refresh button
        self.refresh_btn = self._create_button("🔄 Refresh", "Manually refresh coach data")
        self.refresh_btn.clicked.connect(self.update_display)
        self.control_layout.addWidget(self.refresh_btn)

        # Clear history button
        self.clear_btn = self._create_button("🗑️ Clear History", "Clear all historical data")
        self.clear_btn.clicked.connect(self.clear_history)
        self.control_layout.addWidget(self.clear_btn)

        self.control_layout.addStretch()

        # Export button
        self.export_btn = self._create_button("📤 Export Data", "Export coach analysis data")
        self.export_btn.clicked.connect(self.export_data)
        self.control_layout.addWidget(self.export_btn)

        # Close button
        self.close_btn = self._create_button("❌ Close", "Close Optimization Coach window")
        self.close_btn.clicked.connect(self.close)
        self.control_layout.addWidget(self.close_btn)

    def _create_button(self, text: str, tooltip: str) -> QPushButton:
        """Create a styled button."""
        from PySide6.QtWidgets import QPushButton
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
        """)
        return btn

    # Classic Coach update methods
    def _get_current_phase_name(self) -> str:
        """Get the name of the current optimization phase."""
        if not self.coach_manager:
            return "None"

        # Get latest recommendation to determine current phase
        history = self.coach_manager.recommendations_history
        if history:
            latest = history[-1]
            generation = latest.get('generation', 0)
            phase_info = self.coach_manager.get_phase_info(generation)
            return phase_info.name if phase_info else "Unknown"

        return "None"

    def _update_current_state_tab(self):
        """Update the current state tab with latest data."""
        if not self.coach_manager:
            return

        capabilities = self.coach_manager.get_coach_capabilities()
        state = capabilities.get('state', {})

        # Update basic metrics
        history = self.coach_manager.recommendations_history
        if history:
            latest = history[-1]
            pop_state = latest.get('population_state', {})

            self.gen_label.setText(str(pop_state.get('generation', '--')))
            self.pop_size_label.setText(str(pop_state.get('population_size', '--')))
            self.best_fitness_label.setText(f"{pop_state.get('best_fitness', 0):.4f}")
            self.mean_fitness_label.setText(f"{pop_state.get('mean_fitness', 0):.4f}")
            self.diversity_label.setText(f"{pop_state.get('diversity', 0):.3f}")
            self.trend_label.setText(pop_state.get('fitness_trend', '--'))

        self.stagnation_label.setText(str(state.get('consecutive_stagnation', 0)))
        self.low_diversity_label.setText(str(state.get('consecutive_low_diversity', 0)))

        # Update fitness history
        fitness_history = self.coach_manager.fitness_history[-10:] if hasattr(self.coach_manager, 'fitness_history') else []
        history_text = "Generation | Fitness\n" + "-" * 20 + "\n"
        for i, fitness in enumerate(fitness_history):
            gen = len(self.coach_manager.fitness_history) - len(fitness_history) + i + 1
            history_text += f"{gen:10d} | {fitness:.4f}\n"
        self.fitness_history_text.setPlainText(history_text)

    def _update_decision_logic_tab(self):
        """Update the decision logic tab."""
        if not self.coach_manager:
            return

        history = self.coach_manager.recommendations_history
        if not history:
            return

        latest = history[-1]

        # Current recommendation
        algorithm = latest.get('recommended_algorithm', 'unknown')
        confidence = self.coach_manager.recommendation_confidence
        confidence_icon = "🎯" if confidence > 0.7 else "🤔" if confidence > 0.5 else "❓"

        rec_text = f"{confidence_icon} RECOMMENDATION: Use {algorithm.upper()} algorithm\n"
        rec_text += f"Confidence: {confidence:.2f}\n"
        rec_text += f"Generation: {latest.get('generation', 'unknown')}\n"
        rec_text += f"Estimated completion: {self.coach_manager.estimated_remaining_generations or 'unknown'} generations"

        self.recommendation_text.setPlainText(rec_text)

        # Decision reasoning
        reasoning = self._build_decision_reasoning(latest)
        self.reasoning_text.setPlainText(reasoning)

        # Parameter adjustments
        actions_taken = latest.get('actions_taken', [])
        self.params_table.setRowCount(len(actions_taken))
        for row, action in enumerate(actions_taken):
            # Parse action string
            if "Set " in action and " (was " in action:
                parts = action.split(" (was ")
                param_part = parts[0].replace("Set ", "")
                old_value = parts[1].rstrip(")")

                if "=" in param_part:
                    param, new_value = param_part.split("=", 1)
                    self.params_table.setItem(row, 0, QTableWidgetItem(param))
                    self.params_table.setItem(row, 1, QTableWidgetItem(old_value))
                    self.params_table.setItem(row, 2, QTableWidgetItem(new_value))
                else:
                    self.params_table.setItem(row, 0, QTableWidgetItem(action))
                    self.params_table.setItem(row, 1, QTableWidgetItem(""))
                    self.params_table.setItem(row, 2, QTableWidgetItem(""))
            else:
                self.params_table.setItem(row, 0, QTableWidgetItem(action))
                self.params_table.setItem(row, 1, QTableWidgetItem(""))
                self.params_table.setItem(row, 2, QTableWidgetItem(""))

    def _build_decision_reasoning(self, analysis_data: Dict[str, Any]) -> str:
        """Build detailed decision reasoning text."""
        reasoning = "DECISION REASONING:\n\n"

        pop_state = analysis_data.get('population_state', {})
        recommended_algorithm = analysis_data.get('recommended_algorithm', 'unknown')

        reasoning += f"Population State:\n"
        reasoning += f"  - Best Fitness: {pop_state.get('best_fitness', 0):.4f}\n"
        reasoning += f"  - Diversity: {pop_state.get('diversity', 0):.3f}\n"
        reasoning += f"  - Generation: {pop_state.get('generation', 0)}\n"
        reasoning += f"  - Fitness Trend: {pop_state.get('fitness_trend', 'unknown')}\n\n"

        reasoning += f"Coach State:\n"
        reasoning += f"  - Consecutive Stagnation: {self.coach_manager.consecutive_stagnation}\n"
        reasoning += f"  - Consecutive Low Diversity: {self.coach_manager.consecutive_low_diversity}\n"
        reasoning += f"  - Recommendation Confidence: {self.coach_manager.recommendation_confidence:.2f}\n\n"

        reasoning += f"Algorithm Selection Logic:\n"
        if recommended_algorithm == 'ga':
            reasoning += "  ✓ Selected GA (Genetic Algorithm) for:\n"
            reasoning += "    - Broad exploration in parameter space\n"
            reasoning += "    - Recovery from stagnation or low diversity\n"
            reasoning += "    - Early optimization phases\n"
        elif recommended_algorithm == 'adam':
            reasoning += "  ✓ Selected ADAM (Gradient-based) for:\n"
            reasoning += "    - Fine-tuning in promising regions\n"
            reasoning += "    - Smooth optimization landscapes\n"
            reasoning += "    - Late optimization phases\n"

        reasoning += "\nPhase Analysis:\n"
        phase_info = self.coach_manager.get_phase_info(pop_state.get('generation', 0))
        if phase_info:
            reasoning += f"  - Current Phase: {phase_info.name}\n"
            reasoning += f"  - Phase Range: {phase_info.start_generation}-{phase_info.end_generation}\n"
            reasoning += f"  - Description: {phase_info.description}\n"

        return reasoning

    def _update_historical_analysis_tab(self):
        """Update the historical analysis tab."""
        if not self.coach_manager:
            return

        history = self.coach_manager.recommendations_history
        self.history_table.setRowCount(len(history))

        for row, rec in enumerate(history):
            gen = rec.get('generation', 0)
            algorithm = rec.get('recommended_algorithm', 'unknown')
            confidence = rec.get('coach_state', {}).get('recommendation_confidence', 0.5)
            actions = len(rec.get('actions_taken', []))
            result = "Applied"

            self.history_table.setItem(row, 0, QTableWidgetItem(str(gen)))
            self.history_table.setItem(row, 1, QTableWidgetItem(algorithm.upper()))
            self.history_table.setItem(row, 2, QTableWidgetItem(f"{confidence:.2f}"))
            self.history_table.setItem(row, 3, QTableWidgetItem(str(actions)))
            self.history_table.setItem(row, 4, QTableWidgetItem(result))

        # Update insights
        insights = self._build_learning_insights()
        self.insights_text.setPlainText(insights)

    def _build_learning_insights(self) -> str:
        """Build learning insights text."""
        insights = "LEARNING INSIGHTS:\n\n"

        if not self.coach_manager:
            return insights + "No coach manager available."

        capabilities = self.coach_manager.get_coach_capabilities()
        learning = capabilities.get('learning', {})

        insights += f"Phase Transitions:\n"
        transitions = learning.get('actual_phase_transitions', {})
        if transitions:
            for phase, gen in transitions.items():
                insights += f"  - {phase}: Ended at generation {gen}\n"
        else:
            insights += "  - No early phase transitions\n"

        insights += f"\nConvergence Metrics:\n"
        conv_metrics = learning.get('convergence_metrics', {})
        if conv_metrics:
            for metric, value in conv_metrics.items():
                insights += f"  - {metric}: {value}\n"
        else:
            insights += "  - No convergence metrics available\n"

        insights += f"\nHistorical Performance:\n"
        insights += f"  - Phase performance entries: {learning.get('phase_performance_history', 0)}\n"
        insights += f"  - Total recommendations: {capabilities.get('state', {}).get('recommendations_count', 0)}\n"

        return insights

    def _update_phase_management_tab(self):
        """Update the phase management tab."""
        if not self.coach_manager:
            return

        # Phase boundaries
        boundaries = getattr(self.coach_manager, 'phase_boundaries', {})
        self.exploration_end_label.setText(str(boundaries.get('exploration_end', '--')))
        self.exploitation_end_label.setText(str(boundaries.get('exploitation_end', '--')))

        # Phase comparison
        actual_transitions = getattr(self.coach_manager, 'actual_phase_transitions', {})
        if actual_transitions:
            comparison = "Early transitions detected"
            for phase, gen in actual_transitions.items():
                planned = boundaries.get(f"{phase.split('_')[0]}_end", 'unknown')
                comparison += f" | {phase}: {planned}→{gen}"
        else:
            comparison = "Following planned schedule"
        self.phase_comparison_label.setText(comparison)

        # Phase performance
        perf_history = getattr(self.coach_manager, 'phase_performance_history', [])
        self.phase_perf_table.setRowCount(len(perf_history))

        for row, entry in enumerate(perf_history):
            phase = entry.get('phase', 'Unknown')
            gen = entry.get('generation', 0)
            fitness = entry.get('fitness', 0)
            diversity = entry.get('diversity', 0)

            self.phase_perf_table.setItem(row, 0, QTableWidgetItem(phase))
            self.phase_perf_table.setItem(row, 1, QTableWidgetItem(str(gen)))
            self.phase_perf_table.setItem(row, 2, QTableWidgetItem(f"{fitness:.4f}"))
            self.phase_perf_table.setItem(row, 3, QTableWidgetItem(f"{diversity:.3f}"))

        # Convergence prediction
        remaining = getattr(self.coach_manager, 'estimated_remaining_generations', None)
        if remaining:
            pred_text = f"Estimated {remaining} generations remaining until convergence.\n"
            pred_text += "Based on recent fitness improvement trends."
        else:
            pred_text = "Convergence prediction not available.\n"
            pred_text += "Need more fitness history for accurate estimation."

        self.prediction_text.setPlainText(pred_text)

    def _update_crisis_detection_tab(self):
        """Update the crisis detection tab."""
        if not self.coach_manager:
            return

        # Crisis status
        consecutive_stag = getattr(self.coach_manager, 'consecutive_stagnation', 0)
        consecutive_low_div = getattr(self.coach_manager, 'consecutive_low_diversity', 0)

        status_text = "CURRENT CRISIS STATUS:\n\n"

        if consecutive_stag >= 5:
            status_text += f"🚨 STAGNATION CRISIS: {consecutive_stag} generations without improvement\n"
            status_text += "   → Emergency exploration measures activated\n"
        else:
            status_text += f"✅ No stagnation crisis (count: {consecutive_stag})\n"

        if consecutive_low_div >= 3:
            status_text += f"🚨 LOW DIVERSITY CRISIS: {consecutive_low_div} generations of low diversity\n"
            status_text += "   → Immigration and mutation increases activated\n"
        else:
            status_text += f"✅ Diversity healthy (count: {consecutive_low_div})\n"

        # Check for other crisis conditions
        history = self.coach_manager.recommendations_history
        if history:
            latest = history[-1]
            pop_state = latest.get('population_state', {})

            diversity = pop_state.get('diversity', 1.0)
            best_fitness = pop_state.get('best_fitness', 0)
            generation = pop_state.get('generation', 0)

            if diversity < 0.1 and best_fitness < 0.2 and generation > 10:
                status_text += f"\n🚨 GATE CRISIS: Very low diversity ({diversity:.3f}) + poor fitness ({best_fitness:.3f})\n"
                status_text += "   → Maximum exploration response activated\n"

            if diversity < 0.2 and generation > 30:
                status_text += f"\n🚨 BOUNDARY CLUSTERING: Suspected parameter clustering\n"
                status_text += "   → Boundary exploration measures activated\n"

        self.crisis_status_text.setPlainText(status_text)

        # Crisis history
        crisis_events = []
        for rec in history[-10:]:
            actions = rec.get('actions_taken', [])
            for action in actions:
                if any(keyword in action.lower() for keyword in ['emergency', 'crisis', 'maximum', 'immigrant']):
                    crisis_events.append({
                        'generation': rec.get('generation', 0),
                        'type': 'Optimization Crisis',
                        'response': action,
                        'outcome': 'Applied'
                    })

        self.crisis_table.setRowCount(len(crisis_events))
        for row, event in enumerate(crisis_events):
            self.crisis_table.setItem(row, 0, QTableWidgetItem(str(event['generation'])))
            self.crisis_table.setItem(row, 1, QTableWidgetItem(event['type']))
            self.crisis_table.setItem(row, 2, QTableWidgetItem(event['response'][:50] + "..." if len(event['response']) > 50 else event['response']))
            self.crisis_table.setItem(row, 3, QTableWidgetItem(event['outcome']))

    # OpenAI Coach methods
    def on_tool_call_selected(self):
        """Handle tool call selection in table."""
        selected_items = self.tool_calls_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        if row < len(self.tool_calls_history):
            tool_call = self.tool_calls_history[row]
            self.show_tool_call_details(tool_call)

    def show_tool_call_details(self, tool_call: Dict[str, Any]):
        """Show detailed information about selected tool call."""
        details = f"Tool: {tool_call.get('name', 'Unknown')}\n"
        details += f"Time: {tool_call.get('timestamp', 'Unknown')}\n"
        details += f"Parameters: {json.dumps(tool_call.get('parameters', {}), indent=2)}\n"
        details += f"Response: {json.dumps(tool_call.get('response', {}), indent=2)}\n"
        details += f"Reason: {tool_call.get('reason', 'No reason provided')}\n"

        self.tool_details.setPlainText(details)

    def add_tool_call(self, tool_name: str, parameters: Dict[str, Any],
                      response: Dict[str, Any], reason: str = ""):
        """Add a new tool call to the history."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        tool_call = {
            'timestamp': timestamp,
            'name': tool_name,
            'parameters': parameters,
            'response': response,
            'reason': reason
        }

        self.tool_calls_history.append(tool_call)
        # Schedule UI update on main thread
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.update_tool_calls_table)

        # Auto-scroll to bottom
        QTimer.singleShot(10, self.scroll_table_to_bottom)

    def update_tool_calls_table(self):
        """Update the tool calls table with current history."""
        self.tool_calls_table.setRowCount(len(self.tool_calls_history))

        for row, tool_call in enumerate(self.tool_calls_history):
            # Time
            self.tool_calls_table.setItem(row, 0, QTableWidgetItem(tool_call['timestamp']))

            # Tool Name
            self.tool_calls_table.setItem(row, 1, QTableWidgetItem(tool_call['name']))

            # Parameters (truncated)
            params_str = json.dumps(tool_call['parameters'], separators=(',', ':'))
            if len(params_str) > 100:
                params_str = params_str[:100] + "..."
            self.tool_calls_table.setItem(row, 2, QTableWidgetItem(params_str))

            # Response (truncated)
            response_str = json.dumps(tool_call['response'], separators=(',', ':'))
            if len(response_str) > 100:
                response_str = response_str[:100] + "..."
            self.tool_calls_table.setItem(row, 3, QTableWidgetItem(response_str))

            # Reason
            self.tool_calls_table.setItem(row, 4, QTableWidgetItem(tool_call['reason']))

        # Scroll to bottom
        self.tool_calls_table.scrollToBottom()

    def scroll_table_to_bottom(self):
        """Scroll the tool calls table to the bottom."""
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self.tool_calls_table,
            "scrollToBottom",
            Qt.QueuedConnection
        )

    def update_display(self):
        """Update all display elements based on current coach mode and data."""
        # Update status bar
        self._update_status_bar()

        # Update content based on mode
        if self.coach_mode == 'classic':
            self._update_classic_coach_display()
        elif self.coach_mode == 'openai':
            self._update_openai_coach_display()
        # Disabled mode doesn't need updates

    def _update_status_bar(self):
        """Update the status bar with current information."""
        if not self.coach_manager:
            self.coach_status_label.setText("🤖 Not Connected")
            self.analysis_counter_label.setText("Analyses: 0")
            self.current_phase_label.setText("Phase: None")
            self.confidence_label.setText("Confidence: --")
            return

        # Update coach status
        mode_name = self._get_mode_display_name(self.coach_mode)
        self.coach_status_label.setText(f"🤖 {mode_name}: Active")

        # Update analysis counter
        history = getattr(self.coach_manager, 'recommendations_history', [])
        self.analysis_counter_label.setText(f"Analyses: {len(history)}")

        # Update phase (for Classic mode)
        if self.coach_mode == 'classic':
            phase_name = self._get_current_phase_name()
            self.current_phase_label.setText(f"Phase: {phase_name}")
        else:
            self.current_phase_label.setText("Phase: N/A")

        # Update confidence (for Classic mode)
        if self.coach_mode == 'classic' and hasattr(self.coach_manager, 'recommendation_confidence'):
            confidence = self.coach_manager.recommendation_confidence
            self.confidence_label.setText(f"Confidence: {confidence:.2f}")
        else:
            self.confidence_label.setText("Confidence: --")

    def _update_classic_coach_display(self):
        """Update Classic Coach display elements."""
        if not self.coach_manager or self.coach_mode != 'classic':
            return

        # Update all Classic Coach tabs
        self._update_current_state_tab()
        self._update_decision_logic_tab()
        self._update_historical_analysis_tab()
        self._update_phase_management_tab()
        self._update_crisis_detection_tab()

    def _update_openai_coach_display(self):
        """Update OpenAI Coach display elements."""
        if not self.coach_manager or self.coach_mode != 'openai':
            return

        # Update tool calls table if there are new calls
        self.update_tool_calls_table()

    def clear_history(self):
        """Clear all historical data."""
        self.analysis_history.clear()
        self.tool_calls_history.clear()
        self.agent_requests.clear()
        self.agent_responses.clear()

        # Clear UI elements
        if hasattr(self, 'fitness_history_text'):
            self.fitness_history_text.clear()
        if hasattr(self, 'recommendation_text'):
            self.recommendation_text.clear()
        if hasattr(self, 'reasoning_text'):
            self.reasoning_text.clear()
        if hasattr(self, 'insights_text'):
            self.insights_text.clear()
        if hasattr(self, 'tool_calls_table'):
            self.tool_calls_table.setRowCount(0)
        if hasattr(self, 'history_table'):
            self.history_table.setRowCount(0)
        if hasattr(self, 'crisis_table'):
            self.crisis_table.setRowCount(0)
        if hasattr(self, 'phase_perf_table'):
            self.phase_perf_table.setRowCount(0)

        # Clear coach manager history if available
        if self.coach_manager and hasattr(self.coach_manager, 'recommendations_history'):
            self.coach_manager.recommendations_history.clear()
        if self.coach_manager and hasattr(self.coach_manager, 'fitness_history'):
            self.coach_manager.fitness_history.clear()

        # Update status
        self._update_status_bar()

    def export_data(self):
        """Export coach analysis data to file."""
        from PySide6.QtWidgets import QFileDialog
        import json
        from datetime import datetime

        # Prepare export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'coach_mode': self.coach_mode,
            'analysis_history': self.analysis_history,
            'tool_calls_history': self.tool_calls_history,
            'agent_requests': self.agent_requests,
            'agent_responses': self.agent_responses,
        }

        # Add coach manager data if available
        if self.coach_manager:
            if hasattr(self.coach_manager, 'recommendations_history'):
                export_data['coach_recommendations'] = self.coach_manager.recommendations_history
            if hasattr(self.coach_manager, 'fitness_history'):
                export_data['fitness_history'] = self.coach_manager.fitness_history

        # Get save file path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Coach Data",
            f"coach_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Export Successful",
                                      f"Coach data exported to:\n{file_path}")
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Export Failed",
                                   f"Failed to export data:\n{str(e)}")