"""
Data caching system for downloaded market data.

Caches OHLCV data to parquet files in .data/ directory to avoid
redundant API calls on app restarts.
"""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from data.cleaning import CleaningSummary

try:  # Determine parquet engine availability once
    import pyarrow  # type: ignore  # noqa: F401

    PARQUET_ENGINE = "pyarrow"
except ImportError:  # pragma: no cover - optional dependency
    try:
        import fastparquet  # type: ignore  # noqa: F401

        PARQUET_ENGINE = "fastparquet"
    except ImportError:
        PARQUET_ENGINE = None


class DataCache:
    """Manage local cache of downloaded market data."""
    
    def __init__(self, cache_dir: str = ".data"):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory to store cached data files
        """
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(exist_ok=True)
        self._parquet_engine = PARQUET_ENGINE
        self._warned_no_parquet = False

    def _serialize_attr(self, value: Any) -> Any:
        """
        Convert DataFrame attributes to parquet-friendly values.

        PyArrow stores attrs as JSON metadata; ensure everything is serializable.
        """
        if is_dataclass(value):
            return {k: self._serialize_attr(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {k: self._serialize_attr(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_attr(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        # Fallback: convert to string representation
        return str(value)

    def _sanitize_attrs(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of attrs with parquet-safe values."""
        return {key: self._serialize_attr(val) for key, val in attrs.items()}

    def _rehydrate_attrs(self, df: pd.DataFrame):
        """Restore known dataclass attrs after loading from disk."""
        summary = df.attrs.get("cleaning_summary")
        if isinstance(summary, dict):
            try:
                df.attrs["cleaning_summary"] = CleaningSummary(**summary)
            except TypeError:
                # Leave as dict if structure unexpected
                pass
    
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
        pickle_file = cache_file.with_suffix(".pkl")
        
        if cache_file.exists() and self._parquet_engine:
            try:
                df = pd.read_parquet(cache_file, engine=self._parquet_engine)
                
                if df.index.name == 'ts':
                    df.index = pd.to_datetime(df.index, utc=True)

                self._rehydrate_attrs(df)
                
                print(f"✓ Loaded {len(df)} bars from cache: {cache_file.name}")
                return df
            except Exception as e:
                print(f"⚠ Error reading parquet cache {cache_file}: {e}")
        
        if pickle_file.exists():
            try:
                df = pd.read_pickle(pickle_file)
                
                if df.index.name == 'ts':
                    df.index = pd.to_datetime(df.index, utc=True)

                self._rehydrate_attrs(df)
                
                print(f"✓ Loaded {len(df)} bars from cache: {pickle_file.name}")
                return df
            except Exception as e:
                print(f"⚠ Error reading pickle cache {pickle_file}: {e}")
        
        if cache_file.exists() and not self._parquet_engine and not self._warned_no_parquet:
            print(
                "⚠ Parquet cache available but no parquet engine installed. "
                "Install 'pyarrow' or 'fastparquet' for faster caching."
            )
            self._warned_no_parquet = True
        
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
        pickle_file = cache_file.with_suffix(".pkl")
        
        original_attrs = df.attrs.copy()
        parquet_safe_attrs = self._sanitize_attrs(original_attrs)

        try:
            parquet_saved = False

            if self._parquet_engine:
                df.attrs = parquet_safe_attrs
                try:
                    df.to_parquet(cache_file, engine=self._parquet_engine)
                    parquet_saved = True
                except Exception as parquet_err:
                    print(
                        f"⚠ Error saving parquet cache {cache_file}: {parquet_err}\n"
                        "  ↳ Falling back to pickle cache."
                    )
                finally:
                    df.attrs = original_attrs
            else:
                if not self._warned_no_parquet:
                    print(
                        "ℹ️ Parquet engine not installed; falling back to pickle cache. "
                        "Install 'pyarrow' or 'fastparquet' for parquet support."
                    )
                    self._warned_no_parquet = True

            if parquet_saved:
                print(f"✓ Cached {len(df)} bars to: {cache_file.name}")
                return

            # Parquet not saved (either disabled or errored); pickle as fallback.
            df.to_pickle(pickle_file)
            print(f"✓ Cached {len(df)} bars to: {pickle_file.name}")
        except Exception as e:
            df.attrs = original_attrs
            print(f"⚠ Error saving cache file {pickle_file}: {e}")
    
    def clear_cache(self):
        """Delete all cached files."""
        if not self.cache_dir.exists():
            return
        
        count = 0
        for pattern in ("*.parquet", "*.pkl"):
            for file in self.cache_dir.glob(pattern):
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
                    "format": "parquet",
                })
                
            except Exception as e:
                print(f"⚠ Error reading cache file info {file}: {e}")
        
        for file in self.cache_dir.glob("*.pkl"):
            try:
                stat = file.stat()
                name = file.stem
                parts = name.split("_")
                cache_files.append({
                    "filename": file.name,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": pd.Timestamp(stat.st_mtime, unit='s'),
                    "ticker": parts[0] if len(parts) > 0 else "unknown",
                    "format": "pickle",
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
        if cache_file.exists():
            return True
        return cache_file.with_suffix(".pkl").exists()
