from dataclasses import dataclass
import time
import pandas as pd
import numpy as np
from indicators.local import ema, atr, zscore
from indicators.fibonacci import add_fib_levels_to_df
from core.models import Timeframe, minutes_to_bars
from core.logging_utils import get_logger

logger = get_logger(__name__)

# Optional Spectre import (CPU-first). Falls back to pandas if unavailable.
try:
    from spectre import factors
    from spectre.data import MemoryLoader
    _SPECTRE_AVAILABLE = True
except Exception:
    _SPECTRE_AVAILABLE = False


@dataclass
class SellerParams:
    """Parameters for the Seller Exhaustion strategy.
    Prefer time-based parameters; bar-based remain for backwards compatibility.
    """
    # Backwards-compatible bar-based defaults (assumes 15m)
    ema_fast: int = 96
    ema_slow: int = 672
    z_window: int = 672
    atr_window: int = 96

    # Time-based parameters
    ema_fast_minutes: int | None = None
    ema_slow_minutes: int | None = None
    z_window_minutes: int | None = None
    atr_window_minutes: int | None = None

    # Thresholds
    vol_z: float = 2.0
    tr_z: float = 1.2
    cloc_min: float = 0.6


def build_features(
    df: pd.DataFrame,
    p: SellerParams,
    tf: Timeframe = Timeframe.m15,
    add_fib: bool = True,
    fib_lookback: int = 96,
    fib_lookahead: int = 5,
    use_spectre: bool | None = None,
) -> pd.DataFrame:
    """
    Build features for seller exhaustion detection.
    
    Default path uses Spectre's FactorEngine for performance (CPU-first).
    Falls back to pandas implementation if Spectre is unavailable or disabled.
    
    Signal occurs when:
    - Downtrend (EMA_fast < EMA_slow)
    - Volume z-score spike > vol_z threshold
    - True Range z-score spike > tr_z threshold
    - Close in top portion of candle (cloc > cloc_min)
    
    Additionally calculates Fibonacci retracement levels for exits.
    """
    # Decide engine
    if use_spectre is None:
        use_spectre = _SPECTRE_AVAILABLE

    if use_spectre and _SPECTRE_AVAILABLE:
        try:
            t0 = time.perf_counter()
            out = _build_features_spectre(df, p, tf, add_fib, fib_lookback, fib_lookahead)
            dt = time.perf_counter() - t0
            logger.info(
                "Features built via Spectre | tf=%s | bars=%d | signals=%d | %.3fs",
                tf.value,
                len(out),
                int(out.get("exhaustion", pd.Series(dtype=bool)).sum() if "exhaustion" in out else 0),
                dt,
            )
            return out
        except Exception:
            # Fallback to pandas if Spectre path fails
            pass

    t0 = time.perf_counter()
    out = _build_features_pandas(df, p, tf, add_fib, fib_lookback, fib_lookahead)
    dt = time.perf_counter() - t0
    logger.info(
        "Features built via pandas | tf=%s | bars=%d | signals=%d | %.3fs",
        tf.value,
        len(out),
        int(out.get("exhaustion", pd.Series(dtype=bool)).sum() if "exhaustion" in out else 0),
        dt,
    )
    return out


def _build_features_pandas(
    df: pd.DataFrame,
    p: SellerParams,
    tf: Timeframe,
    add_fib: bool,
    fib_lookback: int,
    fib_lookahead: int,
) -> pd.DataFrame:
    out = df.copy()

    # Resolve window lengths
    ema_fast_bars = minutes_to_bars(p.ema_fast_minutes, tf) if p.ema_fast_minutes is not None else p.ema_fast
    ema_slow_bars = minutes_to_bars(p.ema_slow_minutes, tf) if p.ema_slow_minutes is not None else p.ema_slow
    z_window_bars = minutes_to_bars(p.z_window_minutes, tf) if p.z_window_minutes is not None else p.z_window
    atr_window_bars = minutes_to_bars(p.atr_window_minutes, tf) if p.atr_window_minutes is not None else p.atr_window

    # Calculate EMAs
    out["ema_f"] = ema(out["close"], ema_fast_bars)
    out["ema_s"] = ema(out["close"], ema_slow_bars)

    # Downtrend filter
    out["downtrend"] = out["ema_f"] < out["ema_s"]

    # Calculate ATR
    out["atr"] = atr(out["high"], out["low"], out["close"], atr_window_bars)

    # Volume z-score
    out["vol_z"] = zscore(out["volume"], z_window_bars)

    # True Range z-score (approximate using ATR * window, to match legacy)
    tr_proxy = out["atr"] * atr_window_bars
    out["tr_z"] = zscore(tr_proxy, z_window_bars)

    # Close location in candle (0 = low, 1 = high)
    span = (out["high"] - out["low"]).replace(0, np.nan)
    out["cloc"] = (out["close"] - out["low"]) / span

    # Generate exhaustion signal
    out["exhaustion"] = (
        out["downtrend"] &
        (out["vol_z"] > p.vol_z) &
        (out["tr_z"] > p.tr_z) &
        (out["cloc"] > p.cloc_min)
    )

    # Add Fibonacci retracement levels
    if add_fib:
        out = add_fib_levels_to_df(
            out,
            signal_col="exhaustion",
            lookback=fib_lookback,
            lookahead=fib_lookahead,
        )

    return out


def _build_features_spectre(
    df: pd.DataFrame,
    p: SellerParams,
    tf: Timeframe,
    add_fib: bool,
    fib_lookback: int,
    fib_lookahead: int,
) -> pd.DataFrame:
    """Spectre-accelerated feature builder.

    Uses spectre.data.MemoryLoader to avoid disk I/O and runs on CPU by default.
    """
    if not _SPECTRE_AVAILABLE:
        raise RuntimeError("Spectre is not available")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex")

    # Prepare spectre-formatted DataFrame: MultiIndex ['date','asset'] with adjustments
    # Use a single asset label. If original DataFrame has a name, reuse it; otherwise default.
    asset = getattr(df.index, "name", None) or "X:ADAUSD"
    # Ensure UTC tz
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    base = df.copy()
    base.index = idx
    base = base[["open", "high", "low", "close", "volume"]].copy()
    base["ex-dividend"] = 0.0
    base["split_ratio"] = 1.0
    base["asset"] = asset
    base["date"] = base.index
    spectre_df = base.set_index(["date", "asset"]).sort_index()

    # Build MemoryLoader and FactorEngine
    loader = MemoryLoader(spectre_df)
    engine = factors.FactorEngine(loader)
    # Optional: move factor engine to GPU if enabled
    try:
        import config.settings as cfg_settings
        use_cuda = bool(getattr(cfg_settings.settings, 'use_spectre_cuda', False))
    except Exception:
        use_cuda = False
    if use_cuda:
        try:
            engine.to_cuda(enable_stream=True, gpu_id=0)
            logger.info("Spectre engine moved to CUDA with streaming enabled")
        except Exception as e:
            logger.warning("Spectre CUDA requested but unavailable: %s", e)
            use_cuda = False

    # Resolve window lengths
    ema_fast_bars = minutes_to_bars(p.ema_fast_minutes, tf) if p.ema_fast_minutes is not None else p.ema_fast
    ema_slow_bars = minutes_to_bars(p.ema_slow_minutes, tf) if p.ema_slow_minutes is not None else p.ema_slow
    z_window_bars = minutes_to_bars(p.z_window_minutes, tf) if p.z_window_minutes is not None else p.z_window
    atr_window_bars = minutes_to_bars(p.atr_window_minutes, tf) if p.atr_window_minutes is not None else p.atr_window

    # Define base OHLCV factors
    close = factors.OHLCV.close
    high = factors.OHLCV.high
    low = factors.OHLCV.low
    volume = factors.OHLCV.volume

    # Indicators
    ema_f = factors.EMA(span=int(ema_fast_bars), inputs=[close])
    ema_s = factors.EMA(span=int(ema_slow_bars), inputs=[close])
    downtrend = ema_f < ema_s

    tr = factors.TRANGE(win=2)
    atr_fac = factors.MA(win=int(atr_window_bars), inputs=[tr])

    # Time-series z-scores
    vol_ma = factors.MA(win=int(z_window_bars), inputs=[volume])
    vol_std = factors.STDDEV(win=int(z_window_bars), inputs=[volume])
    vol_z = (volume - vol_ma) / vol_std

    # Match legacy: zscore of (ATR * atr_window)
    tr_proxy = atr_fac * float(atr_window_bars)
    tr_ma = factors.MA(win=int(z_window_bars), inputs=[tr_proxy])
    tr_std = factors.STDDEV(win=int(z_window_bars), inputs=[tr_proxy])
    tr_z = (tr_proxy - tr_ma) / tr_std

    # Close location within candle
    span = (high - low)
    # Avoid divide by zero: clamp span
    cloc = (close - low) / span

    # Exhaustion condition
    exhaustion = (downtrend & (vol_z > p.vol_z) & (tr_z > p.tr_z) & (cloc > p.cloc_min))

    # Register factors
    engine.add(ema_f, "ema_f")
    engine.add(ema_s, "ema_s")
    engine.add(downtrend, "downtrend")
    engine.add(atr_fac, "atr")
    engine.add(vol_z, "vol_z")
    engine.add(tr_z, "tr_z")
    engine.add(cloc, "cloc")
    engine.add(exhaustion, "exhaustion")

    # Determine warmup rows required to avoid insufficient-history warnings
    def _ema_win(span: int) -> int:
        # Spectre's EMA(win) uses ~4.5 * (span + 1)
        return int(4.5 * (int(span) + 1))

    warmup = max(
        _ema_win(int(ema_fast_bars)),
        _ema_win(int(ema_slow_bars)),
        int(z_window_bars),
        int(atr_window_bars),
    )
    date_index = spectre_df.index.get_level_values("date")
    if len(date_index) == 0:
        # No data; return empty aligned frame
        return df.copy()
    start_idx = min(warmup, len(date_index) - 1)
    start = date_index[start_idx]
    end = date_index[-1]

    logger.debug(
        "Spectre run window | warmup=%d | start=%s | end=%s | total=%d",
        warmup,
        start,
        end,
        len(date_index),
    )

    # Run engine for the effective range
    feats_multi = engine.run(start, end)

    # Extract single asset back to a simple DatetimeIndex DataFrame
    feats = feats_multi.xs(asset, level="asset").copy()

    # Ensure boolean dtypes for logical columns
    if "downtrend" in feats.columns:
        feats["downtrend"] = feats["downtrend"].astype(bool)
    if "exhaustion" in feats.columns:
        feats["exhaustion"] = feats["exhaustion"].astype(bool)

    # Reindex to full original index (pre-warmup rows become NaN for features)
    feats = feats.reindex(df.index)

    # Merge with original OHLCV to ensure all columns are present and aligned
    out = pd.concat([
        df[["open", "high", "low", "close", "volume"]],
        feats[["ema_f", "ema_s", "downtrend", "atr", "vol_z", "tr_z", "cloc", "exhaustion"]],
    ], axis=1)

    # Add Fibonacci retracement levels for exits
    if add_fib:
        out = add_fib_levels_to_df(
            out,
            signal_col="exhaustion",
            lookback=fib_lookback,
            lookahead=fib_lookahead,
        )

    return out


def check_signal(row: pd.Series, p: SellerParams) -> bool:
    """Check if a single bar meets seller exhaustion criteria."""
    return bool(row.get("exhaustion", False))
