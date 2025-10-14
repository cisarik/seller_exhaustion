from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from config.settings import settings
from core.models import Bar

BASE = "https://api.polygon.io"


@dataclass
class AggregationProgress:
    page: int
    total_pages: int
    items_received: int
    estimated_total_items: Optional[int]
    seconds_remaining: float
    seconds_per_request: float


class PolygonClient:
    REQUESTS_PER_MINUTE = 5
    PAGE_SIZE = 50_000

    def __init__(self, api_key: Optional[str] = None, timeout: float = 20.0):
        self.api_key = api_key or settings.polygon_api_key
        self.client = httpx.AsyncClient(timeout=timeout)
        self.request_interval = 0.0 if self.REQUESTS_PER_MINUTE <= 0 else 60.0 / self.REQUESTS_PER_MINUTE
        self._last_request_ts = 0.0

    async def _maybe_call(
        self,
        callback: Optional[Callable[[AggregationProgress], Awaitable[None] | None]],
        progress: AggregationProgress,
    ) -> None:
        if not callback:
            return
        result = callback(progress)
        if inspect.isawaitable(result):
            await result

    async def _throttle(self) -> None:
        if self.request_interval <= 0:
            return

        now = time.monotonic()
        elapsed = now - self._last_request_ts
        wait_time = max(0.0, self.request_interval - elapsed)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self._last_request_ts = time.monotonic()

    class RateLimitError(Exception):
        """Raised when rate limit persists beyond reasonable retries."""

    async def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {**params, "apiKey": self.api_key}
        retries = 0
        max_retries = 5

        while True:
            await self._throttle()
            try:
                response = await self.client.get(url, params=params)
            except httpx.RequestError as exc:
                if retries >= max_retries:
                    raise
                retries += 1
                backoff = min(30.0, 2 ** retries)
                await asyncio.sleep(backoff)
                continue

            if response.status_code == 429:
                retry_after_header = response.headers.get("Retry-After")
                try:
                    retry_after = float(retry_after_header) if retry_after_header else None
                except ValueError:
                    retry_after = None

                wait_time = retry_after if retry_after is not None else max(self.request_interval, 2 ** retries)
                await asyncio.sleep(wait_time)
                retries = min(retries + 1, max_retries)
                if retries >= max_retries:
                    raise PolygonClient.RateLimitError("Polygon API rate limit exceeded")
                continue

            if 500 <= response.status_code < 600:
                if retries >= max_retries:
                    response.raise_for_status()
                retries += 1
                backoff = min(30.0, 2 ** retries)
                await asyncio.sleep(backoff)
                continue

            response.raise_for_status()
            return response.json()

    async def aggregates(
        self,
        ticker: str,
        date_from: str,
        date_to: str,
        multiplier: int = 15,
        timespan: str = "minute",
        *,
        estimate: Optional[Any] = None,
        progress_callback: Optional[Callable[[AggregationProgress], Awaitable[None] | None]] = None,
    ) -> List[Bar]:
        """
        Fetch aggregates with specified timeframe.
        
        Args:
            ticker: Ticker symbol (e.g., X:ADAUSD)
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            multiplier: Timeframe multiplier (e.g., 15 for 15 minutes)
            timespan: Timeframe unit (minute, hour, day)
        
        Returns:
            List of Bar objects
        """
        url = f"{BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{date_from}/{date_to}"
        params = {"adjusted": "true", "sort": "asc", "limit": self.PAGE_SIZE}
        items: List[Dict[str, Any]] = []
        page = 0
        total_items = 0

        estimated_pages = None
        estimated_items = None
        seconds_per_request = self.request_interval

        if estimate is not None:
            estimated_pages = getattr(estimate, "pages", None)
            estimated_items = getattr(estimate, "bars", None)
            seconds_per_request = getattr(estimate, "seconds_per_request", seconds_per_request)

        if estimated_pages:
            seconds_remaining = estimated_pages * seconds_per_request
            await self._maybe_call(
                progress_callback,
                AggregationProgress(
                    page=0,
                    total_pages=estimated_pages,
                    items_received=0,
                    estimated_total_items=estimated_items,
                    seconds_remaining=seconds_remaining,
                    seconds_per_request=seconds_per_request,
                ),
            )
        
        while True:
            data = await self._get(url, params)
            results = data.get("results", [])
            items.extend(results)
            page += 1
            total_items += len(results)
            
            next_url = data.get("next_url")

            estimated_total_pages = estimated_pages
            if estimated_total_pages is None:
                estimated_total_pages = page + (1 if next_url else 0)
                if estimated_total_pages == 0:
                    estimated_total_pages = 1
            else:
                estimated_total_pages = max(estimated_total_pages, page + (1 if next_url else 0))

            remaining_pages = max(estimated_total_pages - page, 0)
            seconds_remaining = remaining_pages * seconds_per_request

            await self._maybe_call(
                progress_callback,
                AggregationProgress(
                    page=page,
                    total_pages=estimated_total_pages,
                    items_received=total_items,
                    estimated_total_items=estimated_items,
                    seconds_remaining=seconds_remaining,
                    seconds_per_request=seconds_per_request,
                ),
            )

            if not next_url:
                break
            
            url = next_url
            params = {"apiKey": self.api_key}
        
        return [
            Bar(
                ts=i["t"],
                open=i["o"],
                high=i["h"],
                low=i["l"],
                close=i["c"],
                volume=i["v"]
            )
            for i in items
        ]
    
    async def aggregates_15m(self, ticker: str, date_from: str, date_to: str) -> List[Bar]:
        """Fetch 15-minute aggregates (backwards compatibility)."""
        return await self.aggregates(ticker, date_from, date_to, 15, "minute")

    async def indicator_sma(
        self,
        ticker: str,
        window: int,
        timespan: str = "minute",
        multiplier: int = 15,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch SMA indicator from Polygon."""
        url = f"{BASE}/v1/indicators/sma/{ticker}"
        params = {
            "timespan": timespan,
            "multiplier": multiplier,
            "window": window,
            "series_type": "close",
            "expand_underlying": "false"
        }
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        return await self._get(url, params)

    async def indicator_ema(
        self,
        ticker: str,
        window: int,
        timespan: str = "minute",
        multiplier: int = 15,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch EMA indicator from Polygon."""
        url = f"{BASE}/v1/indicators/ema/{ticker}"
        params = {
            "timespan": timespan,
            "multiplier": multiplier,
            "window": window,
            "series_type": "close",
            "expand_underlying": "false"
        }
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        return await self._get(url, params)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
