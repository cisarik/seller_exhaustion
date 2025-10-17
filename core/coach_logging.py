"""Lightweight logging sink for Evolution Coach messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List
from threading import Lock


Callback = Callable[[str], None]


@dataclass
class CoachLogManager:
    """Thread-safe log buffer with subscriber callbacks."""

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
        """Append a log line and notify listeners."""
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

    def dump(self) -> List[str]:
        """Return a copy of stored lines."""
        with self._lock:
            return list(self._lines)

    def clear(self) -> None:
        """Clear log lines."""
        with self._lock:
            self._lines.clear()


coach_log_manager = CoachLogManager()
