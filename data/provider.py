from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Optional, Callable, Awaitable

import pandas as pd

from data.cleaning import clean_ohlcv
from data.polygon_client import AggregationProgress, PolygonClient
from data.cache import DataCache
from core.models import Timeframe
from config.settings import settings


@dataclass
class DownloadEstimate:
    bars: int
    pages: int
    seconds_per_request: float
    seconds_total: float


class DataProvider:
    def __init__(self, use_cache: bool = True):
        self.poly = PolygonClient()
        cache_dir = settings.data_dir if use_cache else ".data"
        self.cache = DataCache(cache_dir) if use_cache else None

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
        force_download: bool = False,
        use_cache_only: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch bars with specified timeframe and return as pandas DataFrame.
        
        Automatically uses cached data if available, unless force_download=True.
        
        Args:
            ticker: Ticker symbol (e.g., X:ADAUSD)
            from_: Start date (YYYY-MM-DD)
            to: End date (YYYY-MM-DD)
            multiplier: Timeframe multiplier (e.g., 15, 30, 60)
            timespan: Timeframe unit (minute, hour, day)
            force_download: If True, skip cache and download fresh data
            use_cache_only: If True, only return cached data without hitting the API
        
        Returns:
            DataFrame with OHLCV data and UTC DatetimeIndex
        """
        # Try cache first (if enabled and not forcing download)
        if self.cache and not force_download:
            cached_df = self.cache.get_cached_data(ticker, from_, to, multiplier, timespan)
            if cached_df is not None:
                return cached_df

            if use_cache_only:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if use_cache_only:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        # Download from API
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
        
        # Cache the cleaned data
        if self.cache:
            self.cache.save_cached_data(cleaned, ticker, from_, to, multiplier, timespan)
        
        return cleaned

    async def fetch_15m(self, ticker: str, from_: str, to: str) -> pd.DataFrame:
        """Fetch 15-minute bars (backwards compatibility)."""
        return await self.fetch_bars(ticker, from_, to, 15, "minute")

    async def fetch(
        self,
        ticker: str,
        tf: Timeframe,
        from_: str,
        to: str,
        *,
        progress_callback: Optional[Callable[[AggregationProgress], Awaitable[None] | None]] = None,
        estimate: Optional[DownloadEstimate] = None,
        force_download: bool = False,
        use_cache_only: bool = False,
    ) -> pd.DataFrame:
        """Always fetch 1m bars, clean them, then locally resample to target timeframe.

        Dôvody:
        - konzistentné čistenie a deterministické výsledky
        - minimalizácia problémov s Polygon agregáciami naprieč TF
        - lepšia kontrola nad orezaním neúplných intervalov
        """
        tf_to_multiplier = {
            Timeframe.m1: 1,
            Timeframe.m3: 3,
            Timeframe.m5: 5,
            Timeframe.m10: 10,
            Timeframe.m15: 15,
            Timeframe.m60: 60,
        }

        multiplier = tf_to_multiplier.get(tf, 15)
        base_multiplier = tf_to_multiplier[Timeframe.m1]
        base_timespan = "minute"

        base_estimate = estimate
        if base_estimate is None and not use_cache_only:
            base_estimate = self.estimate_download(
                from_,
                to,
                base_multiplier,
                base_timespan,
            )

        # Vždy stiahnuť 1m (s rate-limit retry už rieši PolygonClient)
        base_df = await self.fetch_bars(
            ticker,
            from_,
            to,
            base_multiplier,
            base_timespan,
            progress_callback=progress_callback,
            estimate=base_estimate,
            force_download=force_download,
            use_cache_only=use_cache_only,
        )

        if len(base_df) == 0:
            if self.cache and use_cache_only and multiplier != 1:
                fallback = self.cache.get_cached_data(ticker, from_, to, multiplier, "minute")
                if fallback is not None:
                    return fallback
            return base_df

        if multiplier == 1:
            return base_df

        df = base_df.copy()
        # Resample na cieľový TF podľa OHLCV pravidiel, pravé uzatváranie
        ohlcv = df.resample(f"{multiplier}min", label="right", closed="right").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        # Drop neúplné intervaly (chýbajúci open/close)
        ohlcv = ohlcv.dropna(subset=["open", "high", "low", "close"])  # volume môže byť 0

        # Zachovať cleaning summary z 1m
        summary = df.attrs.get("cleaning_summary")
        ohlcv.attrs["cleaning_summary"] = summary
        
        # Cache the resampled data for faster loading next time
        if self.cache and multiplier > 1:
            self.cache.save_cached_data(ohlcv, ticker, from_, to, multiplier, "minute")
        
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
