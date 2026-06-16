"""Parameter sanity checks and normalization for optimized strategy configs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.models import Timeframe, minutes_to_bars
from strategy.seller_exhaustion import SellerParams
from strategy.timeframe_defaults import get_defaults_for_timeframe, validate_parameters_for_timeframe


CONFIG_DIR = Path("strategies_optimized")


@dataclass
class SanityResult:
    path: Path
    strategy_id: str
    timeframe: Timeframe
    valid: bool
    warnings: list[str]
    changed: bool = False


def _tf_minutes(tf: Timeframe) -> int:
    return {
        Timeframe.m1: 1,
        Timeframe.m3: 3,
        Timeframe.m5: 5,
        Timeframe.m10: 10,
        Timeframe.m15: 15,
        Timeframe.m30: 30,
        Timeframe.m60: 60,
    }.get(tf, 15)


def _bars_to_minutes(bars: int, tf: Timeframe) -> int:
    return int(max(1, bars) * _tf_minutes(tf))


def _parse_timeframe(raw: Any, fallback: Timeframe) -> Timeframe:
    if isinstance(raw, str):
        for tf in Timeframe:
            if tf.value == raw:
                return tf
    return fallback


def _normalize_proxy_params(sp: dict[str, Any], tf: Timeframe) -> dict[str, Any]:
    """
    Normalize proxy SellerParams in optimized configs.

    We enforce time-consistent defaults for core lookback windows so strategy-specific
    mappers do not drift into incompatible temporal horizons.
    """
    out = dict(sp)
    defaults = get_defaults_for_timeframe(tf)
    bars = defaults.get_bar_counts()
    tmins = _tf_minutes(tf)

    out["ema_fast"] = int(bars["ema_fast_bars"])
    out["ema_slow"] = int(bars["ema_slow_bars"])
    out["z_window"] = int(bars["z_window_bars"])
    out["atr_window"] = int(bars["atr_window_bars"])

    out["ema_fast_minutes"] = int(out["ema_fast"] * tmins)
    out["ema_slow_minutes"] = int(out["ema_slow"] * tmins)
    out["z_window_minutes"] = int(out["z_window"] * tmins)
    out["atr_window_minutes"] = int(out["atr_window"] * tmins)
    return out


def _normalize_backtest(bp: dict[str, Any], tf: Timeframe) -> dict[str, Any]:
    out = dict(bp)
    defaults = get_defaults_for_timeframe(tf)
    out["max_hold"] = int(minutes_to_bars(defaults.max_hold_minutes, tf))
    out["fib_swing_lookback"] = int(minutes_to_bars(defaults.fib_lookback_minutes, tf))
    return out


def analyze_file(path: Path, fallback_tf: Timeframe = Timeframe.m15) -> SanityResult | None:
    with path.open() as f:
        raw = json.load(f)

    if "seller_params" not in raw:
        return None

    tf = _parse_timeframe(raw.get("timeframe"), fallback_tf)
    sid = str(raw.get("strategy_id", path.stem))
    sp = SellerParams(**raw["seller_params"])
    valid, warnings = validate_parameters_for_timeframe(sp, tf)
    return SanityResult(path=path, strategy_id=sid, timeframe=tf, valid=valid, warnings=warnings)


def normalize_file(path: Path, fallback_tf: Timeframe = Timeframe.m15) -> SanityResult | None:
    with path.open() as f:
        raw = json.load(f)

    if "seller_params" not in raw:
        return None

    tf = _parse_timeframe(raw.get("timeframe"), fallback_tf)
    sid = str(raw.get("strategy_id", path.stem))

    before_sp = dict(raw["seller_params"])
    after_sp = _normalize_proxy_params(before_sp, tf)
    raw["seller_params"] = after_sp

    if "backtest_params" in raw:
        raw["backtest_params"] = _normalize_backtest(raw["backtest_params"], tf)

    changed = before_sp != after_sp
    if changed:
        with path.open("w") as f:
            json.dump(raw, f, indent=2)

    sp = SellerParams(**raw["seller_params"])
    valid, warnings = validate_parameters_for_timeframe(sp, tf)
    return SanityResult(path=path, strategy_id=sid, timeframe=tf, valid=valid, warnings=warnings, changed=changed)


def list_config_files(tf: Timeframe | None = None) -> list[Path]:
    files = sorted(CONFIG_DIR.glob("*_*.json"))
    if tf is None:
        return files
    suffix = f"_{tf.value}.json"
    return [p for p in files if p.name.endswith(suffix)]


def bootstrap_configs(source_tf: Timeframe, target_tf: Timeframe) -> list[Path]:
    """
    Bootstrap optimized configs from source timeframe to target timeframe.

    Time-based windows are converted via minutes; thresholds are preserved.
    """
    created: list[Path] = []
    for src in list_config_files(source_tf):
        with src.open() as f:
            raw = json.load(f)
        if "seller_params" not in raw:
            continue

        sp = dict(raw.get("seller_params", {}))
        bp = dict(raw.get("backtest_params", {}))
        sid = str(raw.get("strategy_id", src.stem.split("_")[0]))

        # seller params: convert core windows through minutes
        for key in ("ema_fast", "ema_slow", "z_window", "atr_window"):
            if key in sp:
                mins = _bars_to_minutes(int(sp[key]), source_tf)
                sp[key] = int(minutes_to_bars(mins, target_tf))
        # keep thresholds (vol_z/tr_z/cloc_min etc.) unchanged
        sp["ema_fast_minutes"] = _bars_to_minutes(int(sp.get("ema_fast", 1)), target_tf)
        sp["ema_slow_minutes"] = _bars_to_minutes(int(sp.get("ema_slow", 1)), target_tf)
        sp["z_window_minutes"] = _bars_to_minutes(int(sp.get("z_window", 1)), target_tf)
        sp["atr_window_minutes"] = _bars_to_minutes(int(sp.get("atr_window", 1)), target_tf)

        # backtest params: rescale time windows, preserve toggles/thresholds/cost params
        if "max_hold" in bp:
            mins = _bars_to_minutes(int(bp["max_hold"]), source_tf)
            bp["max_hold"] = int(minutes_to_bars(mins, target_tf))
        if "fib_swing_lookback" in bp:
            mins = _bars_to_minutes(int(bp["fib_swing_lookback"]), source_tf)
            bp["fib_swing_lookback"] = int(minutes_to_bars(mins, target_tf))

        out = dict(raw)
        out["timeframe"] = target_tf.value
        out["seller_params"] = sp
        out["backtest_params"] = bp

        dst = CONFIG_DIR / f"{sid}_{target_tf.value}.json"
        with dst.open("w") as f:
            json.dump(out, f, indent=2)
        created.append(dst)

    return created
