"""Phase-aware progress bar for Classic Coach visualization."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PySide6.QtCore import QRect, QSize


class PhaseProgressBar(QWidget):
    """
    Progress bar showing 3 optimization phases with dynamic updates.
    
    For Classic Coach: Shows Exploration → Exploitation → Refinement
    Segments change as phases progress.
    """
    
    phase_changed = Signal(str, int)  # (phase_name, percentage)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(32)
        self.setMaximumHeight(38)
        
        # Phase configuration
        self.phases = [
            {'name': 'Exploration', 'color': '#2196F3', 'icon': '🔍'},
            {'name': 'Exploitation', 'color': '#FF9800', 'icon': '📈'},
            {'name': 'Refinement', 'color': '#4CAF50', 'icon': '✓'},
        ]
        
        # State
        self.current_phase = 0  # 0=Exploration, 1=Exploitation, 2=Refinement
        self.phase_progress = 0  # 0-100 within current phase
        self.total_generation = 1
        self.current_generation = 1
        
        # Analysis count tracking (for Classic Coach)
        self.current_analysis_count = 0
        self.total_analysis_count = 0  # Start at 0 to indicate "not initialized"
        
        # Hide initially until first analysis
        self.setVisible(False)
        
        # Phase boundaries
        self.exploration_end = 25
        self.exploitation_end = 50
        
        self.setStyleSheet("""
            PhaseProgressBar {
                background-color: #0f1a12;
                border-radius: 4px;
            }
        """)
    
    def set_phase_boundaries(self, exploration_end: int, exploitation_end: int, total: int):
        """Update phase boundaries based on coach analysis."""
        self.exploration_end = exploration_end
        self.exploitation_end = exploitation_end
        self.total_generation = total
        self.update_generation_progress()
        self.update()
    
    def set_generation_progress(self, current_gen: int, total_gens: int):
        """Update progress based on current generation."""
        self.current_generation = current_gen
        self.total_generation = total_gens
        self.update_generation_progress()
        self.update()
    
    def set_analysis_counts(self, current_count: int, total_count: int):
        """Update analysis counts for Classic Coach display."""
        self.current_analysis_count = current_count
        self.total_analysis_count = max(1, total_count)  # Avoid division by zero
        # Update progress percentage based on analysis count
        self.phase_progress = int((current_count / self.total_analysis_count) * 100)
        # Determine which phase we're in (don't recalculate phase_progress)
        self._update_current_phase_only()
        
        print(f"📊 PhaseProgressBar.set_analysis_counts: {current_count}/{total_count}, phase={self.current_phase}, progress={self.phase_progress}%")
        
        # Show the progress bar now that we have analysis data
        if current_count > 0:
            self.setVisible(True)
            print(f"✓ PhaseProgressBar is now visible")
        self.update()
    
    def _update_current_phase_only(self):
        """Update only current_phase index without changing phase_progress."""
        gen = self.current_generation
        
        if gen <= self.exploration_end:
            self.current_phase = 0  # Exploration
        elif gen <= self.exploitation_end:
            self.current_phase = 1  # Exploitation
        else:
            self.current_phase = 2  # Refinement
    
    def update_generation_progress(self):
        """Calculate which phase we're in and progress within it."""
        gen = self.current_generation
        
        if gen <= self.exploration_end:
            # In Exploration phase
            self.current_phase = 0
            self.phase_progress = int((gen / self.exploration_end) * 100) if self.exploration_end > 0 else 0
        elif gen <= self.exploitation_end:
            # In Exploitation phase
            self.current_phase = 1
            range_size = self.exploitation_end - self.exploration_end
            progress_in_phase = gen - self.exploration_end
            self.phase_progress = int((progress_in_phase / range_size) * 100) if range_size > 0 else 0
        else:
            # In Refinement phase
            self.current_phase = 2
            range_size = self.total_generation - self.exploitation_end
            progress_in_phase = gen - self.exploitation_end
            self.phase_progress = int((progress_in_phase / range_size) * 100) if range_size > 0 else 0
        
        # Emit signal
        self.phase_changed.emit(self.phases[self.current_phase]['name'], self.phase_progress)
    
    def paintEvent(self, event):
        """Paint the current phase progress bar (simple black + green)."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        padding = 8
        
        # Black background
        painter.fillRect(rect, QColor('#000000'))
        
        # Border (green)
        pen = QPen(QColor('#4CAF50'), 2)
        painter.setPen(pen)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        
        # Inner area for progress
        inner_rect = rect.adjusted(padding, padding, -padding, -padding)
        
        # Draw progress fill (green)
        progress_width = (inner_rect.width() * self.phase_progress) / 100
        progress_rect = QRect(
            inner_rect.x(),
            inner_rect.y(),
            int(progress_width),
            inner_rect.height()
        )
        painter.fillRect(progress_rect, QColor('#4CAF50'))
        
        # Draw current phase info
        phase_name = self.phases[self.current_phase]['name']
        phase_icon = self.phases[self.current_phase]['icon']
        
        # Format text: "Phase Name (X of Y)" - show analysis count
        if self.total_analysis_count > 0:
            # Show analysis count (for Classic Coach)
            display_text = f"{phase_icon} {phase_name} ({self.current_analysis_count} of {self.total_analysis_count})"
        else:
            # Not initialized - shouldn't be visible, but show placeholder
            display_text = f"⏳ {phase_name} phase"
        
        # Draw text centered
        font = QFont('Arial', 10, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor('#FFFFFF'))
        painter.drawText(rect, Qt.AlignCenter, display_text)
        
        painter.end()
    
    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(400, 35)


class AnimatedProgressBar(QWidget):
    """
    Animated progress bar for OpenAI Agents Coach.
    Continuously animates while agent is working.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(32)
        self.setMaximumHeight(38)
        
        self.is_animating = False
        self.animation_offset = 0
        self.has_completed_analysis = False  # Track if any analysis completed
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_animation_tick)
        self.timer.setInterval(50)  # 20fps animation
        
        self.setStyleSheet("""
            AnimatedProgressBar {
                background-color: #0f1a12;
                border-radius: 4px;
                border: 1px solid #2f5c39;
            }
        """)
    
    def start_animation(self):
        """Start the progress bar animation."""
        self.is_animating = True
        self.animation_offset = 0
        self.timer.start()
    
    def stop_animation(self):
        """Stop the progress bar animation."""
        self.is_animating = False
        self.has_completed_analysis = True
        self.timer.stop()
        self.update()
    
    def _on_animation_tick(self):
        """Update animation position."""
        self.animation_offset = (self.animation_offset + 2) % 100
        self.update()
    
    def paintEvent(self, event):
        """Paint the animated progress bar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # Draw background
        painter.fillRect(rect, QColor('#1a2818'))
        
        # Draw border
        pen = QPen(QColor('#2f5c39'), 1)
        painter.setPen(pen)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        
        if self.is_animating:
            # Draw animated stripes
            stripe_width = 20
            stripe_spacing = 5
            total_offset = self.animation_offset + stripe_width + stripe_spacing
            
            x = rect.x() - total_offset
            while x < rect.right():
                stripe_rect = QRect(x, rect.y(), stripe_width, rect.height())
                color = QColor('#2196F3')
                color.setAlpha(150)
                painter.fillRect(stripe_rect.intersected(rect), color)
                x += stripe_width + stripe_spacing
            
            # Draw text
            font = QFont('Arial', 10, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor('#4CAF50'))
            painter.drawText(rect, Qt.AlignCenter, "🤖 Agent analyzing...")
        elif self.has_completed_analysis:
            # Draw completed state (only if we actually completed an analysis)
            painter.fillRect(rect, QColor('#1b5e20'))
            font = QFont('Arial', 10, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor('#81C784'))
            painter.drawText(rect, Qt.AlignCenter, "✓ Analysis complete")
        else:
            # Initial state - waiting for first analysis
            font = QFont('Arial', 10, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor('#888888'))
            painter.drawText(rect, Qt.AlignCenter, "⏳ Waiting for coach analysis...")
        
        painter.end()
    
    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(400, 35)
