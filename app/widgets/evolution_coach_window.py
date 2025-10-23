"""
Evolution Coach Window

Podrobné okno pre Evolution Coach agent s:
- Request/Response textfieldy
- Tool Call History tabulka
- Real-time status updates
- Agent analysis progress
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTextEdit, QTableWidget, QTableWidgetItem, QLabel, QPushButton,
    QGroupBox, QHeaderView, QProgressBar, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QTextCursor, QColor
from datetime import datetime
import json
from typing import Dict, List, Any, Optional


class EvolutionCoachWindow(QMainWindow):
    """
    Hlavné okno pre Evolution Coach s real-time monitoring.
    
    Zobrazuje:
    - Request/Response komunikáciu s agentom
    - Tool Call History tabulku
    - Real-time status updates
    - Agent analysis progress
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧠 Evolution Coach - Agent Analysis")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Data storage
        self.tool_calls_history: List[Dict[str, Any]] = []
        self.agent_requests: List[str] = []
        self.agent_responses: List[str] = []
        self.current_analysis_id: Optional[str] = None
        
        # UI components
        self.init_ui()
        
        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_status)
        self.refresh_timer.start(1000)  # Update every second
        
    def init_ui(self):
        """Initialize UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Status bar
        self.create_status_bar()
        main_layout.addWidget(self.status_frame)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Communication
        self.create_communication_panel()
        main_splitter.addWidget(self.communication_widget)
        
        # Right panel - Tool History
        self.create_tool_history_panel()
        main_splitter.addWidget(self.tool_history_widget)
        
        # Set splitter proportions
        main_splitter.setSizes([600, 800])
        
        # Control buttons
        self.create_control_buttons()
        main_layout.addLayout(self.control_layout)
        
    def create_status_bar(self):
        """Create status bar with current analysis info."""
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
        
        # Status label
        self.status_label = QLabel("🔍 Agent Status: Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4caf50;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Analysis info
        self.analysis_info = QLabel("No active analysis")
        self.analysis_info.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.analysis_info)
        
        layout.addStretch()
        
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
        self.request_text.setMaximumHeight(200)
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
        request_layout.addWidget(self.request_text)
        
        layout.addWidget(request_group)
        
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
        response_layout.addWidget(self.response_text)
        
        layout.addWidget(response_group)
        
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
        self.tool_details.setMaximumHeight(150)
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
        history_layout.addWidget(self.tool_details)
        
        layout.addWidget(history_group)
        
    def create_control_buttons(self):
        """Create control buttons."""
        self.control_layout = QHBoxLayout()
        
        # Clear button
        self.clear_btn = QPushButton("🗑️ Clear History")
        self.clear_btn.setToolTip("Clear all tool call history and communication logs")
        self.clear_btn.clicked.connect(self.clear_history)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.control_layout.addWidget(self.clear_btn)
        
        # Export button
        self.export_btn = QPushButton("📤 Export Logs")
        self.export_btn.setToolTip("Export tool call history to JSON file")
        self.export_btn.clicked.connect(self.export_logs)
        self.export_btn.setStyleSheet("""
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
        self.control_layout.addWidget(self.export_btn)
        
        self.control_layout.addStretch()
        
        # Close button
        self.close_btn = QPushButton("❌ Close")
        self.close_btn.setToolTip("Close Evolution Coach window")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #666;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        self.control_layout.addWidget(self.close_btn)
        
    def update_status(self):
        """Update status information."""
        # This will be called by the timer to refresh status
        pass
        
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
        
    def set_agent_request(self, request: str):
        """Set the current agent request."""
        # Ensure UI update happens on main thread
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self.request_text,
            "setPlainText",
            Qt.QueuedConnection,
            Q_ARG(str, request)
        )
        self.agent_requests.append(request)

    def set_agent_response(self, response: str):
        """Set the current agent response."""
        # Ensure UI update happens on main thread
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self.response_text,
            "setPlainText",
            Qt.QueuedConnection,
            Q_ARG(str, response)
        )
        self.agent_responses.append(response)
    
    def update_last_tool_call(self, parameters: dict, response: dict, reason: str):
        """Update the last tool call with new parameters and response."""
        if self.tool_calls_history:
            last_call = self.tool_calls_history[-1]
            last_call['parameters'] = parameters
            last_call['response'] = response
            last_call['reason'] = reason
            # Schedule UI update on main thread
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self.update_tool_calls_table)
        
    def set_status(self, status: str, is_analyzing: bool = False):
        """Set the current status."""
        # Ensure UI update happens on main thread
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self.status_label,
            "setText",
            Qt.QueuedConnection,
            Q_ARG(str, f"🔍 Agent Status: {status}")
        )

        if is_analyzing:
            QMetaObject.invokeMethod(self.progress_bar, "setVisible", Qt.QueuedConnection, Q_ARG(bool, True))
            QMetaObject.invokeMethod(
                self.status_label,
                "setStyleSheet",
                Qt.QueuedConnection,
                Q_ARG(str, """
                QLabel {
                    color: #ff9800;
                    font-weight: bold;
                    font-size: 14px;
                }
                """)
            )
        else:
            QMetaObject.invokeMethod(self.progress_bar, "setVisible", Qt.QueuedConnection, Q_ARG(bool, False))
            QMetaObject.invokeMethod(
                self.status_label,
                "setStyleSheet",
                Qt.QueuedConnection,
                Q_ARG(str, """
                QLabel {
                    color: #4caf50;
                    font-weight: bold;
                    font-size: 14px;
                }
                """)
            )
            
    def set_analysis_info(self, info: str):
        """Set analysis information."""
        # Ensure UI update happens on main thread
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self.analysis_info,
            "setText",
            Qt.QueuedConnection,
            Q_ARG(str, info)
        )
        
    def clear_history(self):
        """Clear all history."""
        self.tool_calls_history.clear()
        self.agent_requests.clear()
        self.agent_responses.clear()

        # Ensure UI updates happen on main thread
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(self.request_text, "clear", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self.response_text, "clear", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self.tool_details, "clear", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self.tool_calls_table, "setRowCount", Qt.QueuedConnection, Q_ARG(int, 0))

        self.set_status("Ready")
        self.set_analysis_info("No active analysis")
        
    def export_logs(self):
        """Export logs to JSON file."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Evolution Coach Logs",
            f"evolution_coach_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'tool_calls_history': self.tool_calls_history,
                    'agent_requests': self.agent_requests,
                    'agent_responses': self.agent_responses
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                    
                self.set_status(f"Logs exported to {file_path}")
                
            except Exception as e:
                self.set_status(f"Export failed: {str(e)}")
                
    def closeEvent(self, event):
        """Handle window close event."""
        self.refresh_timer.stop()
        event.accept()
