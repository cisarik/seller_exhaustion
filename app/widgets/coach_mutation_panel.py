"""
Coach Mutation Panel

UI for:
- Manual parameter mutations (mutate individual)
- Drop/remove individuals
- View and accept coach recommendations
- Mutation history tracking
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QSpinBox, QDoubleSpinBox, QComboBox,
    QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from typing import Optional, List, Dict, Any
import logging

from backtest.coach_session import CoachAnalysisSession, IndividualSnapshot
from backtest.coach_protocol import CoachRecommendation, CoachAnalysis
from backtest.optimizer import Population

logger = logging.getLogger(__name__)


class CoachMutationPanel(QWidget):
    """
    Panel for manual mutations and coach recommendation viewing/application.
    
    Features:
    - Manual mutation of individual parameters
    - Drop individuals
    - View coach recommendations
    - Apply/reject recommendations
    - Mutation history
    """
    
    # Signal emitted when user applies mutations
    mutations_applied = Signal()  # User manually applied mutations
    
    # Signal emitted when coach recommendations accepted
    coach_recommendations_accepted = Signal(dict)  # {accepted: [], rejected: []}
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_session: Optional[CoachAnalysisSession] = None
        self.population: Optional[Population] = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # === Manual Mutations Section ===
        manual_group = self._create_manual_mutations_group()
        layout.addWidget(manual_group)
        
        # === Coach Recommendations Section ===
        coach_group = self._create_coach_recommendations_group()
        layout.addWidget(coach_group)
        
        # === Mutation History Section ===
        history_group = self._create_mutation_history_group()
        layout.addWidget(history_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _create_manual_mutations_group(self) -> QGroupBox:
        """Create manual mutations control group."""
        group = QGroupBox("Manual Mutations")
        layout = QVBoxLayout()
        
        # Individual selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Individual:"))
        self.individual_combo = QComboBox()
        selector_layout.addWidget(self.individual_combo)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        
        # Parameter selector
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Parameter:"))
        self.parameter_combo = QComboBox()
        param_layout.addWidget(self.parameter_combo)
        param_layout.addStretch()
        layout.addLayout(param_layout)
        
        # Value editor
        value_layout = QHBoxLayout()
        value_layout.addWidget(QLabel("New Value:"))
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setMinimum(-1000.0)
        self.value_spin.setMaximum(10000.0)
        self.value_spin.setDecimals(4)
        value_layout.addWidget(self.value_spin)
        value_layout.addStretch()
        layout.addLayout(value_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        mutate_btn = QPushButton("✏️  Apply Mutation")
        mutate_btn.clicked.connect(self._on_apply_mutation)
        button_layout.addWidget(mutate_btn)
        
        drop_btn = QPushButton("🗑️  Drop Individual")
        drop_btn.clicked.connect(self._on_drop_individual)
        button_layout.addWidget(drop_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_coach_recommendations_group(self) -> QGroupBox:
        """Create coach recommendations viewing/application group."""
        group = QGroupBox("Coach Recommendations")
        layout = QVBoxLayout()
        
        # Status label
        self.coach_status_label = QLabel("No recommendations yet")
        self.coach_status_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.coach_status_label)
        
        # Recommendations table
        self.recommendations_table = QTableWidget()
        self.recommendations_table.setColumnCount(4)
        self.recommendations_table.setHorizontalHeaderLabels(
            ["Parameter", "Current → Suggested", "Reasoning", "Confidence"]
        )
        self.recommendations_table.horizontalHeader().setStretchLastSection(False)
        self.recommendations_table.setColumnWidth(0, 150)
        self.recommendations_table.setColumnWidth(1, 200)
        self.recommendations_table.setColumnWidth(2, 250)
        self.recommendations_table.setColumnWidth(3, 100)
        layout.addWidget(self.recommendations_table)
        
        # Apply/Reject buttons
        button_layout = QHBoxLayout()
        
        accept_all_btn = QPushButton("✅ Accept All")
        accept_all_btn.clicked.connect(self._on_accept_all_recommendations)
        button_layout.addWidget(accept_all_btn)
        
        reject_all_btn = QPushButton("❌ Reject All")
        reject_all_btn.clicked.connect(self._on_reject_all_recommendations)
        button_layout.addWidget(reject_all_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_mutation_history_group(self) -> QGroupBox:
        """Create mutation history display group."""
        group = QGroupBox("Mutation History")
        layout = QVBoxLayout()
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(
            ["Time", "Operation", "Individual", "Parameter", "Change"]
        )
        self.history_table.setMaximumHeight(150)
        layout.addWidget(self.history_table)
        
        group.setLayout(layout)
        return group
    
    def set_session_and_population(
        self,
        session: CoachAnalysisSession,
        population: Population
    ):
        """
        Set current session and population for mutations.
        
        Updates UI with available individuals and recommendations.
        
        Args:
            session: Frozen CoachAnalysisSession
            population: Population to mutate
        """
        self.current_session = session
        self.population = population
        
        # Update individual selector
        self._update_individual_combo()
        
        # Update coach recommendations if available
        if session.analysis:
            self._update_recommendations_table(session.analysis)
        else:
            self.coach_status_label.setText("Coach analysis in progress...")
    
    def update_with_analysis(self, analysis: CoachAnalysis):
        """Update recommendations table with new coach analysis."""
        if not self.current_session:
            return
        
        self.current_session.analysis = analysis
        self._update_recommendations_table(analysis)
    
    def _update_individual_combo(self):
        """Update individual selector combo box."""
        self.individual_combo.clear()
        
        if not self.current_session or not self.current_session.individuals_snapshot:
            self.individual_combo.addItem("No individuals", -1)
            return
        
        for snap in self.current_session.individuals_snapshot:
            label = (
                f"Ind#{snap.id:02d} "
                f"(fit={snap.fitness:.4f}, trades={snap.metrics.get('n', 0)})"
            )
            self.individual_combo.addItem(label, snap.id)
    
    def _update_recommendations_table(self, analysis: CoachAnalysis):
        """Populate recommendations table from coach analysis."""
        self.recommendations_table.setRowCount(0)
        
        if not analysis or not analysis.recommendations:
            self.coach_status_label.setText("No recommendations from coach")
            return
        
        self.coach_status_label.setText(
            f"✅ {len(analysis.recommendations)} recommendations from coach"
        )
        
        for row, rec in enumerate(analysis.recommendations):
            self.recommendations_table.insertRow(row)
            
            # Parameter
            param_item = QTableWidgetItem(rec.parameter)
            self.recommendations_table.setItem(row, 0, param_item)
            
            # Current → Suggested
            change_text = f"{rec.current_value} → {rec.suggested_value}"
            change_item = QTableWidgetItem(change_text)
            self.recommendations_table.setItem(row, 1, change_item)
            
            # Reasoning
            reason_item = QTableWidgetItem(rec.reasoning[:80])
            reason_item.setToolTip(rec.reasoning)
            self.recommendations_table.setItem(row, 2, reason_item)
            
            # Confidence
            confidence = rec.confidence
            confidence_text = f"{confidence:.0%}"
            confidence_item = QTableWidgetItem(confidence_text)
            
            # Color code confidence
            if confidence >= 0.8:
                confidence_item.setBackground(QColor(100, 200, 100))  # Green
            elif confidence >= 0.6:
                confidence_item.setBackground(QColor(255, 200, 100))  # Orange
            else:
                confidence_item.setBackground(QColor(200, 100, 100))  # Red
            
            self.recommendations_table.setItem(row, 3, confidence_item)
    
    def _on_apply_mutation(self):
        """Apply manual mutation to selected individual."""
        if not self.population or not self.current_session:
            QMessageBox.warning(self, "Error", "No population selected")
            return
        
        ind_id = self.individual_combo.currentData()
        if ind_id < 0:
            QMessageBox.warning(self, "Error", "Select a valid individual")
            return
        
        param_name = self.parameter_combo.currentText()
        if not param_name:
            QMessageBox.warning(self, "Error", "Select a parameter")
            return
        
        new_value = self.value_spin.value()
        
        # Apply mutation via mutation manager
        from backtest.coach_mutations import CoachMutationManager
        manager = CoachMutationManager(verbose=True)
        
        success = manager.mutate_individual_parameter(
            self.population,
            self.current_session,
            ind_id,
            param_name,
            new_value,
            reason="Manual mutation via UI"
        )
        
        if success:
            QMessageBox.information(
                self, "Success",
                f"Mutated Individual #{ind_id} parameter {param_name}"
            )
            self.mutations_applied.emit()
        else:
            QMessageBox.warning(
                self, "Error",
                f"Failed to mutate individual {ind_id}"
            )
    
    def _on_drop_individual(self):
        """Drop selected individual from population."""
        if not self.population or not self.current_session:
            QMessageBox.warning(self, "Error", "No population selected")
            return
        
        ind_id = self.individual_combo.currentData()
        if ind_id < 0:
            QMessageBox.warning(self, "Error", "Select a valid individual")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Drop",
            f"Drop Individual #{ind_id} from population?"
        )
        if reply != QMessageBox.Yes:
            return
        
        from backtest.coach_mutations import CoachMutationManager
        manager = CoachMutationManager(verbose=True)
        
        success = manager.drop_individual(
            self.population,
            self.current_session,
            ind_id,
            reason="Manual drop via UI"
        )
        
        if success:
            # Refresh combo
            self._update_individual_combo()
            self.mutations_applied.emit()
            QMessageBox.information(self, "Success", f"Dropped Individual #{ind_id}")
        else:
            QMessageBox.warning(self, "Error", f"Failed to drop Individual #{ind_id}")
    
    def _on_accept_all_recommendations(self):
        """Accept all coach recommendations."""
        if not self.current_session or not self.current_session.analysis:
            QMessageBox.information(self, "Info", "No recommendations to accept")
            return
        
        from backtest.coach_mutations import CoachMutationManager
        manager = CoachMutationManager(verbose=True)
        
        summary = manager.apply_coach_recommendations(
            self.population,
            self.current_session,
            self.current_session.analysis
        )
        
        QMessageBox.information(
            self, "Applied",
            f"Applied {summary['total_mutations']} mutations:\n"
            f"  Mutate: {summary['mutations_by_type'].get('mutate', 0)}\n"
            f"  Drop: {summary['mutations_by_type'].get('drop', 0)}\n"
            f"  Insert: {summary['mutations_by_type'].get('insert', 0)}"
        )
        
        # Refresh combo
        self._update_individual_combo()
        self.coach_recommendations_accepted.emit({
            "accepted": summary["mutations_applied"],
            "rejected": summary["mutations_failed"]
        })
    
    def _on_reject_all_recommendations(self):
        """Reject all coach recommendations."""
        reply = QMessageBox.question(
            self, "Confirm Reject",
            "Discard all coach recommendations?"
        )
        if reply == QMessageBox.Yes:
            self.coach_status_label.setText("Recommendations rejected")
            self.coach_recommendations_accepted.emit({
                "accepted": [],
                "rejected": []
            })
