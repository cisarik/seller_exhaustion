"""
Data caching system for downloaded market data.

Caches OHLCV data to parquet files in .data/ directory to avoid
redundant API calls on app restarts.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd


class DataCache:
    """Manage local cache of downloaded market data."""
    
    def __init__(self, cache_dir: str = ".data"):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory to store cached data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _make_cache_key(
        self,
        ticker: str,
        from_: str,
        to: str,
        multiplier: int,
        timespan: str
    ) -> str:
        """
        Generate cache filename for given parameters.
        
        Format: {ticker}_{from}_{to}_{multiplier}{timespan}.parquet
        Example: X:ADAUSD_2024-01-01_2024-12-31_15minute.parquet
        """
        # Sanitize ticker (replace : with _)
        safe_ticker = ticker.replace(":", "_")
        
        # Create filename
        filename = f"{safe_ticker}_{from_}_{to}_{multiplier}{timespan}.parquet"
        
        return filename
    
    def get_cached_data(
        self,
        ticker: str,
        from_: str,
        to: str,
        multiplier: int,
        timespan: str
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available.
        
        Args:
            ticker: Ticker symbol (e.g., X:ADAUSD)
            from_: Start date (YYYY-MM-DD)
            to: End date (YYYY-MM-DD)
            multiplier: Timeframe multiplier (e.g., 15, 30, 60)
            timespan: Timeframe unit (minute, hour, day)
        
        Returns:
            DataFrame with cached data or None if not found
        """
        cache_file = self.cache_dir / self._make_cache_key(
            ticker, from_, to, multiplier, timespan
        )
        
        if not cache_file.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            
            # Restore UTC timezone on index
            if df.index.name == 'ts':
                df.index = pd.to_datetime(df.index, utc=True)
            
            print(f"✓ Loaded {len(df)} bars from cache: {cache_file.name}")
            return df
            
        except Exception as e:
            print(f"⚠ Error reading cache file {cache_file}: {e}")
            return None
    
    def save_cached_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        from_: str,
        to: str,
        multiplier: int,
        timespan: str
    ):
        """
        Save data to cache.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Ticker symbol (e.g., X:ADAUSD)
            from_: Start date (YYYY-MM-DD)
            to: End date (YYYY-MM-DD)
            multiplier: Timeframe multiplier (e.g., 15, 30, 60)
            timespan: Timeframe unit (minute, hour, day)
        """
        if df is None or len(df) == 0:
            print("⚠ Not caching empty DataFrame")
            return
        
        cache_file = self.cache_dir / self._make_cache_key(
            ticker, from_, to, multiplier, timespan
        )
        
        try:
            # Save to parquet (efficient format for timeseries)
            df.to_parquet(cache_file)
            print(f"✓ Cached {len(df)} bars to: {cache_file.name}")
            
        except Exception as e:
            print(f"⚠ Error saving cache file {cache_file}: {e}")
    
    def clear_cache(self):
        """Delete all cached files."""
        if not self.cache_dir.exists():
            return
        
        count = 0
        for file in self.cache_dir.glob("*.parquet"):
            try:
                file.unlink()
                count += 1
            except Exception as e:
                print(f"⚠ Error deleting {file}: {e}")
        
        print(f"✓ Cleared {count} cached file(s)")
    
    def get_cache_info(self) -> list[Dict[str, Any]]:
        """
        Get information about cached files.
        
        Returns:
            List of dicts with cache file metadata
        """
        if not self.cache_dir.exists():
            return []
        
        cache_files = []
        
        for file in self.cache_dir.glob("*.parquet"):
            try:
                stat = file.stat()
                
                # Parse filename
                name = file.stem
                parts = name.split("_")
                
                cache_files.append({
                    "filename": file.name,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": pd.Timestamp(stat.st_mtime, unit='s'),
                    "ticker": parts[0] if len(parts) > 0 else "unknown",
                })
                
            except Exception as e:
                print(f"⚠ Error reading cache file info {file}: {e}")
        
        return cache_files
    
    def has_cached_data(
        self,
        ticker: str,
        from_: str,
        to: str,
        multiplier: int,
        timespan: str
    ) -> bool:
        """Check if cached data exists for given parameters."""
        cache_file = self.cache_dir / self._make_cache_key(
            ticker, from_, to, multiplier, timespan
        )
        return cache_file.exists()
