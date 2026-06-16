import json
from pathlib import Path

from core.models import Timeframe
from backtest.param_sanity import analyze_file, normalize_file


def test_analyze_and_normalize_mean_reversion_15m():
    path = Path("strategies_optimized/mean_reversion_15m.json")
    if not path.exists():
        return

    before = analyze_file(path, Timeframe.m15)
    assert before is not None

    original = path.read_text()
    try:
        after = normalize_file(path, Timeframe.m15)
        assert after is not None
        assert after.valid

        data = json.loads(path.read_text())
        sp = data["seller_params"]
        assert sp["ema_fast"] == 96
        assert sp["ema_slow"] == 672
        assert sp["z_window"] == 672
        assert sp["atr_window"] == 96
    finally:
        path.write_text(original)
