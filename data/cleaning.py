from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _as_index(df: pd.DataFrame) -> pd.Index:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    raise TypeError("Expected a DatetimeIndex for OHLCV data.")


@dataclass
class CleaningSummary:
    """Book-keeping for the cleaning pipeline."""

    total_rows: int
    dropped_duplicates: int = 0
    dropped_non_finite: int = 0
    dropped_non_positive: int = 0
    dropped_negative_volume: int = 0
    corrected_boundaries: int = 0
    missing_expected_bars: int = 0

    def changed(self) -> bool:
        return any(
            getattr(self, field)
            for field in (
                "dropped_duplicates",
                "dropped_non_finite",
                "dropped_non_positive",
                "dropped_negative_volume",
                "corrected_boundaries",
            )
        )

    def log(self, *, prefix: str = "") -> None:
        if not logger.isEnabledFor(logging.INFO):
            return

        parts = [
            f"rows={self.total_rows}",
            f"drop_duplicates={self.dropped_duplicates}",
            f"drop_non_finite={self.dropped_non_finite}",
            f"drop_non_positive={self.dropped_non_positive}",
            f"drop_negative_volume={self.dropped_negative_volume}",
            f"corrected_boundaries={self.corrected_boundaries}",
            f"missing_expected_bars={self.missing_expected_bars}",
        ]
        msg = prefix + "cleaning summary: " if prefix else "cleaning summary: "
        logger.info(msg + ", ".join(parts))


def clean_ohlcv(
    df: pd.DataFrame,
    *,
    expected_freq: Optional[str] = "15min",
    price_cols: Sequence[str] = ("open", "high", "low", "close"),
    volume_col: str = "volume",
) -> tuple[pd.DataFrame, CleaningSummary]:
    """
    Apply defensive cleaning for OHLCV data retrieved from remote APIs.

    Steps:
        1. Drop duplicated timestamps and sort ascending.
        2. Remove rows with non-finite or non-positive price fields.
        3. Remove rows with negative volume.
        4. Ensure `high`/`low` boundaries contain both `open` and `close`.
        5. (Optional) Report missing timestamps relative to expected frequency.
    """
    if df.empty:
        summary = CleaningSummary(total_rows=0)
        return df, summary

    cleaned = df.copy()
    idx = _as_index(cleaned)

    summary = CleaningSummary(total_rows=len(cleaned))

    # Drop duplicates (keep last occurrence)
    if idx.has_duplicates:
        before = len(cleaned)
        cleaned = cleaned.loc[~idx.duplicated(keep="last")]
        summary.dropped_duplicates = before - len(cleaned)
        idx = _as_index(cleaned)

    # Remove rows with any non-finite values.
    cols_to_check = list(price_cols) + [volume_col]
    non_finite_mask = ~np.isfinite(cleaned[cols_to_check]).all(axis=1)
    if non_finite_mask.any():
        summary.dropped_non_finite += int(non_finite_mask.sum())
        cleaned = cleaned.loc[~non_finite_mask]

    if cleaned.empty:
        summary.log(prefix="data_provider.")
        return cleaned, summary

    # Remove rows with non-positive prices.
    price_frame = cleaned.loc[:, price_cols]
    non_positive_mask = (price_frame <= 0).any(axis=1)
    if non_positive_mask.any():
        summary.dropped_non_positive = int(non_positive_mask.sum())
        cleaned = cleaned.loc[~non_positive_mask]

    # Remove rows with negative volume.
    negative_volume_mask = cleaned[volume_col] < 0
    if negative_volume_mask.any():
        summary.dropped_negative_volume = int(negative_volume_mask.sum())
        cleaned = cleaned.loc[~negative_volume_mask]

    if cleaned.empty:
        summary.log(prefix="data_provider.")
        return cleaned, summary

    # Ensure open/close lie within high/low boundaries.
    high_col = price_cols[1] if len(price_cols) > 1 else "high"
    low_col = price_cols[2] if len(price_cols) > 2 else "low"

    # Guarantee columns exist.
    if high_col not in cleaned.columns or low_col not in cleaned.columns:
        raise KeyError("high/low columns required for boundary correction.")

    boundary_stack = cleaned[list(price_cols)]
    max_price = boundary_stack.max(axis=1)
    min_price = boundary_stack.min(axis=1)

    # Count rows requiring adjustments.
    boundary_mask = (cleaned[high_col] != max_price) | (cleaned[low_col] != min_price)
    if boundary_mask.any():
        summary.corrected_boundaries = int(boundary_mask.sum())
        cleaned.loc[boundary_mask, high_col] = max_price.loc[boundary_mask]
        cleaned.loc[boundary_mask, low_col] = min_price.loc[boundary_mask]

    # Optional: report missing timestamps.
    if expected_freq:
        full_index = pd.date_range(
            start=cleaned.index.min(),
            end=cleaned.index.max(),
            freq=expected_freq,
        )
        summary.missing_expected_bars = int(len(full_index.difference(cleaned.index)))

    summary.log(prefix="data_provider.")
    cleaned.attrs["cleaning_summary"] = summary
    return cleaned, summary
