from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Optional, Callable, Awaitable

import pandas as pd

from data.cleaning import clean_ohlcv
from data.polygon_client import AggregationProgress, PolygonClient
from core.models import Timeframe


@dataclass
class DownloadEstimate:
    bars: int
    pages: int
    seconds_per_request: float
    seconds_total: float


class DataProvider:
    def __init__(self):
        self.poly = PolygonClient()

    async def fetch_bars(
        self,
        ticker: str,
        from_: str,
        to: str,
        multiplier: int = 15,
        timespan: str = "minute",
        *,
        progress_callback: Optional[Callable[[AggregationProgress], Awaitable[None] | None]] = None,
        estimate: Optional[DownloadEstimate] = None,
    ) -> pd.DataFrame:
        """
        Fetch bars with specified timeframe and return as pandas DataFrame.
        
        Args:
            ticker: Ticker symbol (e.g., X:ADAUSD)
            from_: Start date (YYYY-MM-DD)
            to: End date (YYYY-MM-DD)
            multiplier: Timeframe multiplier (e.g., 15, 30, 60)
            timespan: Timeframe unit (minute, hour, day)
        
        Returns:
            DataFrame with OHLCV data and UTC DatetimeIndex
        """
        bars = await self.poly.aggregates(
            ticker,
            from_,
            to,
            multiplier,
            timespan,
            estimate=estimate,
            progress_callback=progress_callback,
        )
        
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        df = pd.DataFrame([b.model_dump() for b in bars])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()

        df, summary = clean_ohlcv(df)
        # Preserve the cleaning summary on the returned frame so consumers can inspect it.
        cleaned = df[["open", "high", "low", "close", "volume"]]
        cleaned.attrs["cleaning_summary"] = summary
        return cleaned

    async def fetch_15m(self, ticker: str, from_: str, to: str) -> pd.DataFrame:
        """Fetch 15-minute bars (backwards compatibility)."""
        return await self.fetch_bars(ticker, from_, to, 15, "minute")

    async def fetch(self, ticker: str, tf: Timeframe, from_: str, to: str) -> pd.DataFrame:
        """Fetch bars for given timeframe, with fallback to 1m + local resample on rate-limit/empty."""
        tf_to_multiplier = {
            Timeframe.m1: 1,
            Timeframe.m3: 3,
            Timeframe.m5: 5,
            Timeframe.m10: 10,
            Timeframe.m15: 15,
        }

        multiplier = tf_to_multiplier.get(tf, 15)
        try:
            return await self.fetch_bars(
                ticker,
                from_,
                to,
                multiplier,
                "minute",
            )
        except PolygonClient.RateLimitError:
            # Fallback: download 1m and resample locally
            base_df = await self.fetch_bars(
                ticker,
                from_,
                to,
                tf_to_multiplier[Timeframe.m1],
                "minute",
            )

            if len(base_df) == 0 or multiplier == 1:
                return base_df

            df = base_df.copy()
            # Resample to target timeframe with OHLCV rules
            ohlcv = df.resample(f"{multiplier}min", label="right", closed="right").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna(subset=["open", "high", "low", "close"])  # drop incomplete intervals

            # Preserve cleaning summary if present
            summary = df.attrs.get("cleaning_summary")
            ohlcv.attrs["cleaning_summary"] = summary
            return ohlcv

    def estimate_download(
        self,
        from_: str,
        to: str,
        multiplier: int,
        timespan: str,
    ) -> DownloadEstimate:
        """Estimate number of bars, requests, and time required for a download."""
        unit_seconds = {
            "minute": 60,
            "hour": 3600,
            "day": 86400,
        }
        unit_sec = unit_seconds.get(timespan)
        if unit_sec is None or multiplier <= 0:
            return DownloadEstimate(
                bars=0,
                pages=1,
                seconds_per_request=self.poly.request_interval,
                seconds_total=0.0,
            )

        start = datetime.fromisoformat(from_)
        end = datetime.fromisoformat(to)

        # Include the entire end day
        end = end + timedelta(days=1)

        total_seconds = max((end - start).total_seconds(), 0)
        bar_duration = unit_sec * multiplier
        if bar_duration <= 0:
            bar_duration = unit_sec

        estimated_bars = max(1, math.ceil(total_seconds / bar_duration))
        estimated_pages = max(1, math.ceil(estimated_bars / self.poly.PAGE_SIZE))
        seconds_per_request = self.poly.request_interval
        estimated_seconds_total = estimated_pages * seconds_per_request

        return DownloadEstimate(
            bars=estimated_bars,
            pages=estimated_pages,
            seconds_per_request=seconds_per_request,
            seconds_total=estimated_seconds_total,
        )

    async def close(self):
        """Close underlying clients."""
        await self.poly.close()
