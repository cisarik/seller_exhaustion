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
        self.setMinimumHeight(40)
        self.setMaximumHeight(50)
        
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
        
        # Calculate generation range text
        if self.current_phase == 0:
            range_text = f"(1-{self.exploration_end})"
        elif self.current_phase == 1:
            range_text = f"({self.exploration_end+1}-{self.exploitation_end})"
        else:
            range_text = f"({self.exploitation_end+1}-{self.total_generation})"
        
        # Format text: "Phase Name (range) - X%"
        display_text = f"{phase_icon} {phase_name} {range_text} - {self.phase_progress}%"
        
        # Draw text centered
        font = QFont('Arial', 11, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor('#FFFFFF'))
        painter.drawText(rect, Qt.AlignCenter, display_text)
        
        painter.end()
    
    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(400, 50)


class AnimatedProgressBar(QWidget):
    """
    Animated progress bar for OpenAI Agents Coach.
    Continuously animates while agent is working.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(30)
        self.setMaximumHeight(40)
        
        self.is_animating = False
        self.animation_offset = 0
        
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
        else:
            # Draw completed state
            painter.fillRect(rect, QColor('#1b5e20'))
            font = QFont('Arial', 10, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor('#81C784'))
            painter.drawText(rect, Qt.AlignCenter, "✓ Analysis complete")
        
        painter.end()
    
    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(400, 40)
