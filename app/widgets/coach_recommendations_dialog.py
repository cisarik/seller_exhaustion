"""
Coach Recommendations Dialog

Displays full Evolution Coach analysis including:
- Summary
- Recommendations with reasoning
- Next steps
- Flags (stagnation, diversity)
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QScrollArea, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from backtest.coach_protocol import CoachAnalysis


class CoachRecommendationsDialog(QDialog):
    """Dialog to display full Evolution Coach analysis."""
    
    def __init__(self, analysis: CoachAnalysis, parent=None):
        super().__init__(parent)
        self.analysis = analysis
        self.setWindowTitle(f"Evolution Coach Analysis - Generation {analysis.generation}")
        self.resize(800, 600)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Header with generation and assessment
        header = QLabel(
            f"<h2>🤖 Evolution Coach Analysis</h2>"
            f"<p style='color: #888;'>Generation {self.analysis.generation} | "
            f"Assessment: <b style='color: {self._assessment_color()};'>{self.analysis.overall_assessment.upper()}</b></p>"
        )
        header.setTextFormat(Qt.RichText)
        layout.addWidget(header)
        
        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(12)
        
        # Flags Group
        if self.analysis.stagnation_detected or self.analysis.diversity_concern:
            flags_group = QGroupBox("⚠️ Flags")
            flags_layout = QVBoxLayout()
            
            if self.analysis.stagnation_detected:
                flag_label = QLabel("🔴 <b>Stagnation Detected:</b> Best fitness has not improved significantly")
                flag_label.setTextFormat(Qt.RichText)
                flag_label.setStyleSheet("color: #f44336; padding: 4px;")
                flags_layout.addWidget(flag_label)
            
            if self.analysis.diversity_concern:
                flag_label = QLabel("🟡 <b>Diversity Concern:</b> Population diversity is low")
                flag_label.setTextFormat(Qt.RichText)
                flag_label.setStyleSheet("color: #ff9800; padding: 4px;")
                flags_layout.addWidget(flag_label)
            
            flags_group.setLayout(flags_layout)
            scroll_layout.addWidget(flags_group)
        
        # Summary Group
        summary_group = QGroupBox("📋 Summary")
        summary_layout = QVBoxLayout()
        
        summary_text = QTextEdit()
        summary_text.setPlainText(self.analysis.summary)
        summary_text.setReadOnly(True)
        summary_text.setMaximumHeight(120)
        summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #e8f5e9;
                border: 1px solid #2f5c39;
                padding: 8px;
                font-size: 13px;
            }
        """)
        summary_layout.addWidget(summary_text)
        
        summary_group.setLayout(summary_layout)
        scroll_layout.addWidget(summary_group)
        
        # Recommendations Group
        if self.analysis.recommendations:
            recs_group = QGroupBox(f"📌 Recommendations ({len(self.analysis.recommendations)})")
            recs_layout = QVBoxLayout()
            recs_layout.setSpacing(8)
            
            for i, rec in enumerate(self.analysis.recommendations, 1):
                rec_widget = self._create_recommendation_widget(i, rec)
                recs_layout.addWidget(rec_widget)
            
            recs_group.setLayout(recs_layout)
            scroll_layout.addWidget(recs_group)
        else:
            no_recs = QLabel("✅ No recommendations - configuration looks good!")
            no_recs.setStyleSheet("color: #4caf50; font-weight: bold; padding: 12px;")
            scroll_layout.addWidget(no_recs)
        
        # Next Steps Group
        if self.analysis.next_steps:
            steps_group = QGroupBox(f"🎯 Next Steps ({len(self.analysis.next_steps)})")
            steps_layout = QVBoxLayout()
            
            for step in self.analysis.next_steps:
                step_label = QLabel(f"• {step}")
                step_label.setWordWrap(True)
                step_label.setStyleSheet("color: #e8f5e9; padding: 4px;")
                steps_layout.addWidget(step_label)
            
            steps_group.setLayout(steps_layout)
            scroll_layout.addWidget(steps_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setFixedWidth(100)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _assessment_color(self) -> str:
        """Get color for overall assessment."""
        if self.analysis.overall_assessment == "positive":
            return "#4caf50"  # Green
        elif self.analysis.overall_assessment == "needs_adjustment":
            return "#f44336"  # Red
        else:
            return "#ff9800"  # Orange (neutral)
    
    def _create_recommendation_widget(self, index: int, rec) -> QWidget:
        """Create widget for single recommendation."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Apply background based on confidence
        if rec.confidence >= 0.8:
            bg_color = "#1b2e1f"  # Green tint (high confidence)
            border_color = "#4caf50"
        elif rec.confidence >= 0.6:
            bg_color = "#2e2416"  # Yellow tint (moderate confidence)
            border_color = "#ff9800"
        else:
            bg_color = "#2e1a1a"  # Red tint (low confidence)
            border_color = "#f44336"
        
        widget.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 4px;
            }}
        """)
        
        # Header: number, parameter name, confidence
        header_layout = QHBoxLayout()
        
        num_label = QLabel(f"<b>{index}.</b>")
        num_label.setTextFormat(Qt.RichText)
        num_label.setStyleSheet("color: #888;")
        header_layout.addWidget(num_label)
        
        param_label = QLabel(f"<b>{rec.parameter.upper()}</b>")
        param_label.setTextFormat(Qt.RichText)
        param_label.setStyleSheet("color: #4caf50; font-size: 14px;")
        header_layout.addWidget(param_label)
        
        header_layout.addStretch()
        
        confidence_label = QLabel(f"Confidence: <b>{rec.confidence:.0%}</b>")
        confidence_label.setTextFormat(Qt.RichText)
        confidence_label.setStyleSheet(f"color: {border_color};")
        header_layout.addWidget(confidence_label)
        
        layout.addLayout(header_layout)
        
        # Category
        category_label = QLabel(f"Category: {rec.category.value}")
        category_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(category_label)
        
        # Current → Suggested
        change_label = QLabel(f"<b>Change:</b> {rec.current_value} → <b style='color: #4caf50;'>{rec.suggested_value}</b>")
        change_label.setTextFormat(Qt.RichText)
        change_label.setStyleSheet("color: #e8f5e9; padding: 4px 0;")
        layout.addWidget(change_label)
        
        # Reasoning
        reasoning_label = QLabel(f"<b>Reason:</b> {rec.reasoning}")
        reasoning_label.setTextFormat(Qt.RichText)
        reasoning_label.setWordWrap(True)
        reasoning_label.setStyleSheet("color: #e8f5e9; padding: 4px 0;")
        layout.addWidget(reasoning_label)
        
        return widget


def show_coach_recommendations(analysis: CoachAnalysis, parent=None):
    """
    Convenience function to show coach recommendations dialog.
    
    Args:
        analysis: CoachAnalysis object
        parent: Parent widget
    
    Returns:
        Dialog result (QDialog.Accepted or QDialog.Rejected)
    """
    dialog = CoachRecommendationsDialog(analysis, parent)
    return dialog.exec()
