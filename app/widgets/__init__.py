"""Convenience imports for the app widgets package."""

from .candle_view import CandleChartWidget
from .compact_params import CompactParamsEditor
from .data_bar import DataBar
# Removed: EvolutionCoachWindow (deleted, now using console logging)
from .settings_dialog import SettingsDialog
from .stats_panel import StatsPanel
from .strategy_editor import StrategyEditor

__all__ = [
    "CandleChartWidget",
    "CompactParamsEditor",
    "DataBar",
    # Removed: "EvolutionCoachWindow",
    "SettingsDialog",
    "StatsPanel",
    "StrategyEditor",
]
