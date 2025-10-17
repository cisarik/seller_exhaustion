from __future__ import annotations

from typing import Optional

from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Qt, QMetaObject, Q_ARG, Slot
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QWidget,
)

DEFAULT_PLACEHOLDER = (
    "Evolution Coach log output will appear here.\n"
    "Use this concise feed when exporting runs for the agent."
)


class EvolutionCoachWindow(QDialog):
    """Lightweight window that displays evolution-coach formatted logs."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Evolution Coach")
        self.resize(720, 520)
        self.setModal(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.log_view = QTextEdit(self)
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.NoWrap)
        self.log_view.setPlaceholderText(DEFAULT_PLACEHOLDER)
        font = self.log_view.font()
        font.setFamily("Courier")
        font.setPointSize(font.pointSize() - 1)
        self.log_view.setFont(font)

        layout.addWidget(self.log_view, stretch=1)

        button_bar = QHBoxLayout()
        button_bar.setSpacing(8)
        button_bar.addStretch(1)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_logs)
        button_bar.addWidget(self.clear_btn)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setEnabled(False)
        button_bar.addWidget(self.analyze_btn)

        layout.addLayout(button_bar)

        # Populate existing logs
        for line in coach_log_manager.dump():
            self.append_log(line)

        coach_log_manager.add_listener(self.append_log)

    def append_log(self, line: str) -> None:
        """Thread-safe append of a single log line."""
        if not line:
            return

        QMetaObject.invokeMethod(
            self,
            "_write_log_line",
            Qt.QueuedConnection,
            Q_ARG(str, line),
        )

    @Slot(str)
    def _write_log_line(self, line: str) -> None:
        if self.log_view.toPlainText() == "":
            self.log_view.setPlainText(line)
        else:
            cursor = self.log_view.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(f"\n{line}")
            self.log_view.setTextCursor(cursor)
            self.log_view.ensureCursorVisible()

    def clear_logs(self) -> None:
        """Clear the log display."""
        self.log_view.clear()
        coach_log_manager.clear()

    def closeEvent(self, event) -> None:
        """Ensure listener is removed when window closes."""
        coach_log_manager.remove_listener(self.append_log)
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        """Re-register listener when window is shown again."""
        coach_log_manager.remove_listener(self.append_log)
        coach_log_manager.add_listener(self.append_log)
        log_lines = coach_log_manager.dump()
        self.log_view.setPlainText("\n".join(log_lines) if log_lines else "")
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_view.setTextCursor(cursor)
        super().showEvent(event)
from core.coach_logging import coach_log_manager
