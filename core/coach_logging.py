"""Logging sinks for Evolution Coach and debug messages.

Two separate logs:
- coach_log_manager: User-facing coach window (clean, relevant info only)
- debug_log_manager: Internal debug log (initialization, diagnostics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal
from threading import Lock


Callback = Callable[[str], None]


@dataclass
class LogManager:
    """Thread-safe log buffer with subscriber callbacks."""

    name: str = "Log"
    max_lines: int = 5000
    _lines: List[str] = field(default_factory=list)
    _callbacks: List[Callback] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, init=False)

    def add_listener(self, callback: Callback) -> None:
        """Register a callback to receive appended lines."""
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_listener(self, callback: Callback) -> None:
        """Remove an existing callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def append(self, line: str) -> None:
        """
        Append a log line and notify listeners.
        
        Args:
            line: Log line to append
        """
        if not line:
            return

        with self._lock:
            self._lines.append(line)
            if len(self._lines) > self.max_lines:
                excess = len(self._lines) - self.max_lines
                del self._lines[:excess]

            callbacks = list(self._callbacks)

        for cb in callbacks:
            try:
                cb(line)
            except Exception:
                # Ignore listener errors to avoid breaking logging flow
                continue

    def dump(self, n_lines: int = 0) -> List[str]:
        """
        Return a copy of stored lines.
        
        Args:
            n_lines: If > 0, return only last N lines; if 0, return all
        
        Returns:
            List of log lines
        """
        with self._lock:
            lines = list(self._lines)
        
        if n_lines > 0:
            return lines[-n_lines:]
        return lines

    def clear(self) -> None:
        """Clear log lines."""
        with self._lock:
            self._lines.clear()
    
    def get_line_count(self) -> int:
        """Get total number of lines."""
        with self._lock:
            return len(self._lines)


# COACH LOG: User-facing coach window (clean, relevant)
# Show:
#   - What's sent to coach
#   - Coach responses
#   - Mutations applied
#   - Recommendations accepted/rejected
# Hide:
#   - Initialization messages
#   - Model loading status
#   - Debug info
coach_log_manager = LogManager(name="CoachLog", max_lines=1000)


# DEBUG LOG: Internal diagnostic log (for developers)
# Show:
#   - Initialization messages
#   - Model loading/unloading
#   - Configuration changes
#   - Error diagnostics
#   - Performance metrics
debug_log_manager = LogManager(name="DebugLog", max_lines=5000)
