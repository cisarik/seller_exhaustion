"""
Environment bootstrap — run before other imports in CLI/tests.

Suppresses noisy third-party warnings that confuse agents and clutter CI logs.
The py2app/pkg_resources warning comes from pyenv's global Python 3.10 site-packages
when `poetry` bootstraps; filtering here keeps project output clean for venv runs.
"""

from __future__ import annotations

import os
import warnings


def suppress_known_third_party_warnings() -> None:
    """Filter deprecation noise from py2app, setuptools pkg_resources, etc."""
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", module="py2app")
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

    # Allow override via env (e.g. PYTHONWARNINGS=default for debugging)
    if os.getenv("ADA_SHOW_ALL_WARNINGS", "").lower() in ("1", "true", "yes"):
        warnings.resetwarnings()


suppress_known_third_party_warnings()
