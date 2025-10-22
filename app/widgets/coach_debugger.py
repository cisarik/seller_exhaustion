from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QHeaderView
from PySide6.QtCore import Qt

class CoachDebugger(QWidget):
    def __init__(self, tool_calls=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coach Agent Debugger")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)

        # Title label for selected tool details
        self.detail_label = QLabel("Select a tool call to view details")
        self.detail_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.detail_label)

        # Table for tool calls
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Tool Name", "Parameters", "Response"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self.show_details)
        layout.addWidget(self.table)

        # Populate table if data provided
        if tool_calls:
            self.populate_table(tool_calls)

    def populate_table(self, tool_calls):
        self.table.setRowCount(len(tool_calls))
        for row, call in enumerate(tool_calls):
            self.table.setItem(row, 0, QTableWidgetItem(call.get('name', 'Unknown')))
            params_str = str(call.get('arguments', {}))
            self.table.setItem(row, 1, QTableWidgetItem(params_str[:100] + '...' if len(params_str) > 100 else params_str))
            response_str = str(call.get('response', 'No response'))
            self.table.setItem(row, 2, QTableWidgetItem(response_str[:100] + '...' if len(response_str) > 100 else response_str))
            # Store full data in user data
            self.table.item(row, 0).setData(Qt.UserRole, call)

    def show_details(self):
        selected = self.table.selectedItems()
        if selected:
            row = selected[0].row()
            call = self.table.item(row, 0).data(Qt.UserRole)
            details = f"Tool: {call.get('name')}\nParameters: {call.get('arguments')}\nResponse: {call.get('response')}"
            self.detail_label.setText(details)