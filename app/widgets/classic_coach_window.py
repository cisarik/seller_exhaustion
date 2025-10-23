"""
Classic Coach Window - Deterministic Decision Interface

This window provides a read-only interface to visualize the Classic Coach's
deterministic optimization logic and decision-making process. Shows all
internal state, phase transitions, crisis detection, and recommendations
in real-time.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QFormLayout, QLabel, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QScrollArea,
    QFrame, QGridLayout, QTabWidget, QPushButton
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QPalette
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from app.theme import DARK_FOREST_QSS


class ClassicCoachWindow(QMainWindow):
    """
    Main window for Classic Coach decision visualization.

    Shows the deterministic thought process of the Classic Coach including:
    - Current population state analysis
    - Phase transitions and boundaries
    - Crisis detection logic
    - Algorithm recommendations with confidence
    - Historical learning and trends
    - Parameter adjustments and reasoning
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 Classic Coach - Deterministic Decision Interface")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        # Apply dark forest theme
        self.setStyleSheet(DARK_FOREST_QSS)

        # Data storage
        self.coach_manager = None  # Will be set by stats_panel
        self.current_analysis_data = {}
        self.analysis_history: List[Dict[str, Any]] = []

        # UI components
        self.init_ui()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_display)
        self.refresh_timer.start(2000)  # Update every 2 seconds

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

        # Main tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_current_state_tab()
        self.create_decision_logic_tab()
        self.create_historical_analysis_tab()
        self.create_phase_management_tab()
        self.create_crisis_detection_tab()

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
        self.coach_status_label = QLabel("🤖 Classic Coach: Ready")
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

        # Current phase
        self.current_phase_label = QLabel("Phase: None")
        self.current_phase_label.setStyleSheet("color: #2196f3; font-size: 12px;")
        layout.addWidget(self.current_phase_label)

        layout.addStretch()

        # Confidence indicator
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("color: #ff9800; font-size: 12px;")
        layout.addWidget(self.confidence_label)

    def create_current_state_tab(self):
        """Create tab showing current population state analysis."""
        tab = QWidget()
        self.tab_widget.addTab(tab, "📊 Current State")

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
        self.tab_widget.addTab(tab, "🧠 Decision Logic")

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
        self.tab_widget.addTab(tab, "📈 Historical Analysis")

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
        self.tab_widget.addTab(tab, "🔄 Phase Management")

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
        self.tab_widget.addTab(tab, "🚨 Crisis Detection")

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
        self.close_btn = self._create_button("❌ Close", "Close Classic Coach window")
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

    def set_coach_manager(self, coach_manager):
        """Set the coach manager reference."""
        self.coach_manager = coach_manager
        self.update_display()

    def update_display(self):
        """Update all display elements with current coach data."""
        if not self.coach_manager:
            return

        try:
            # Update status bar
            capabilities = self.coach_manager.get_coach_capabilities()
            state = capabilities.get('state', {})

            self.coach_status_label.setText("🤖 Classic Coach: Active")
            self.analysis_counter_label.setText(f"Analyses: {state.get('recommendations_count', 0)}")
            self.current_phase_label.setText(f"Phase: {self._get_current_phase_name()}")
            self.confidence_label.setText(f"Confidence: {state.get('recommendation_confidence', 0):.2f}")

            # Update current state tab
            self._update_current_state_tab()

            # Update decision logic tab
            self._update_decision_logic_tab()

            # Update historical analysis tab
            self._update_historical_analysis_tab()

            # Update phase management tab
            self._update_phase_management_tab()

            # Update crisis detection tab
            self._update_crisis_detection_tab()

        except Exception as e:
            print(f"Error updating Classic Coach display: {e}")

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

        # Update basic metrics (we'll need to get this from the latest analysis)
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
            # Parse action string (format: "Set param=value (was old_value)")
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
            result = "Applied"  # Could be enhanced with actual results

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

        # Crisis history (simplified - would need to track actual crises)
        # For now, just show recent recommendations that might indicate crisis responses
        crisis_events = []
        for rec in history[-10:]:  # Last 10 recommendations
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

    def clear_history(self):
        """Clear all historical data."""
        if self.coach_manager:
            self.coach_manager.reset()

        self.analysis_history.clear()
        self.update_display()

        # Show confirmation
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "History Cleared", "All Classic Coach historical data has been cleared.")

    def export_data(self):
        """Export coach analysis data."""
        from PySide6.QtWidgets import QFileDialog

        if not self.coach_manager:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Classic Coach Data",
            f"classic_coach_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'coach_capabilities': self.coach_manager.get_coach_capabilities(),
                    'recommendations_history': self.coach_manager.recommendations_history,
                    'analysis_history': self.analysis_history
                }

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Export Complete", f"Data exported to {file_path}")

            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Export Failed", f"Failed to export data: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event."""
        self.refresh_timer.stop()
        event.accept()