from dataclasses import dataclass
import time
import pandas as pd
import numpy as np
from indicators.local import ema, atr, zscore
from indicators.fibonacci import add_fib_levels_to_df
from core.models import Timeframe, minutes_to_bars
from config.settings import settings
from core.logging_utils import get_logger

logger = get_logger(__name__)


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
) -> pd.DataFrame:
    """
    Build features for seller exhaustion detection using pandas.
    
    Signal occurs when:
    - Downtrend (EMA_fast < EMA_slow)
    - Volume z-score spike > vol_z threshold
    - True Range z-score spike > tr_z threshold
    - Close in top portion of candle (cloc > cloc_min)
    
    Additionally calculates Fibonacci retracement levels for exits.
    """
    t0 = time.perf_counter()
    out = _build_features_pandas(df, p, tf, add_fib, fib_lookback, fib_lookahead)
    dt = time.perf_counter() - t0
    msg = (
        "Features built | tf=%s | bars=%d | signals=%d | %.3fs",
        tf.value,
        len(out),
        int(out.get("exhaustion", pd.Series(dtype=bool)).sum() if "exhaustion" in out else 0),
        dt,
    )
    if getattr(settings, 'log_feature_builds', False):
        logger.info(*msg)
    else:
        logger.debug(*msg)
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
    span = out["high"] - out["low"]
    span = span.mask(span == 0, np.nan)
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


def check_signal(row: pd.Series, p: SellerParams) -> bool:
    """Check if a single bar meets seller exhaustion criteria."""
    return bool(row.get("exhaustion", False))
