"""Lightweight logging utilities for the Seller-Exhaustion tooling."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional


class HumanFormatter(logging.Formatter):
    """Minimal, eye-friendly formatter for console output."""

    SYMBOLS = {
        "INFO": "✓",
        "WARNING": "⚠",
        "ERROR": "✗",
        "CRITICAL": "❗",
        "DEBUG": "…",
    }

    def format(self, record: logging.LogRecord) -> str:
        symbol = self.SYMBOLS.get(record.levelname, record.levelname[:1])
        short_name = record.name.split(".")[-1] if record.name else ""
        message = record.getMessage()

        if record.levelno >= logging.WARNING and short_name and short_name != "root":
            prefix = f"{symbol} [{short_name}]"
        else:
            prefix = symbol

        return f"{prefix} {message}"
ENV_LOG_LEVEL = "ADA_AGENT_LOG_LEVEL"


@lru_cache(maxsize=1)
def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once with a consistent format.

    The level can be supplied explicitly or via the ADA_AGENT_LOG_LEVEL
    environment variable (defaults to INFO). Subsequent calls are no-ops.
    """
    env_level = level or os.getenv(ENV_LOG_LEVEL, "INFO")
    numeric_level = getattr(logging, str(env_level).upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)
    handler.setFormatter(HumanFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger with default configuration applied."""
    configure_logging()
    return logging.getLogger(name)
