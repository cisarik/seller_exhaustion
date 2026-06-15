"""
Strategy-aware genetic algorithm evaluation.

Maps standard Individual (seller_params + backtest_params) to each strategy's
native parameter objects, then evaluates with profit_focused fitness.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

import pandas as pd

from backtest.engine import run_backtest
from backtest.optimizer import (
    Individual,
    Population,
    calculate_fitness,
    evolution_step,
)
from backtest.profit import profit_score, simulate_account
from core.models import BacktestParams, FitnessConfig, Timeframe
from strategy.registry import build
from strategy.seller_exhaustion import SellerParams
from strategy import seller_aggressive, mean_reversion, vol_squeeze
from strategy import capitulation_reversal, liquidity_sweep, divergence_bounce
from strategy.mean_reversion import MeanReversionParams
from strategy.vol_squeeze import VolSqueezeParams
from strategy.capitulation_reversal import CapitulationParams
from strategy.liquidity_sweep import LiquiditySweepParams
from strategy.divergence_bounce import DivergenceParams

OPTIMIZABLE_STRATEGIES = ("seller_aggressive", "mean_reversion", "vol_squeeze")
OPTIMIZABLE_STRATEGIES_V2 = ("capitulation_reversal", "liquidity_sweep", "divergence_bounce")


def get_optimizable_strategies(round: int = 1) -> tuple[str, ...]:
    return OPTIMIZABLE_STRATEGIES_V2 if round == 2 else OPTIMIZABLE_STRATEGIES


def individual_to_features(
    strategy_id: str,
    individual: Individual,
    data: pd.DataFrame,
    tf: Timeframe,
) -> pd.DataFrame:
    sp = individual.seller_params
    bt = individual.backtest_params

    if strategy_id == "seller_aggressive":
        return seller_aggressive.build_features(data, sp, tf, bt)
    if strategy_id == "mean_reversion":
        params = MeanReversionParams(
            ema_fast=sp.ema_fast,
            ema_slow=sp.ema_slow,
            atr_window=sp.atr_window,
            rsi_max=max(18.0, min(45.0, 12.0 + sp.vol_z * 8.0)),
            vol_mult=max(1.0, min(2.5, 0.7 + sp.tr_z * 0.45)),
            cloc_min=sp.cloc_min,
        )
        return mean_reversion.build_features(data, params, tf, bt)
    if strategy_id == "vol_squeeze":
        params = VolSqueezeParams(
            ema_fast=sp.ema_fast,
            ema_slow=sp.ema_slow,
            atr_window=sp.atr_window,
            vol_z_min=max(0.8, min(3.0, sp.vol_z * 0.75)),
            expansion_mult=max(1.1, min(2.5, 0.9 + sp.tr_z * 0.5)),
            cloc_min=sp.cloc_min,
        )
        return vol_squeeze.build_features(data, params, tf, bt)
    if strategy_id == "capitulation_reversal":
        params = CapitulationParams(
            ema_fast=sp.ema_fast,
            ema_slow=sp.ema_slow,
            z_window=sp.z_window,
            atr_window=sp.atr_window,
            vol_z_min=max(1.0, min(3.5, sp.vol_z)),
            wick_min=max(0.30, min(0.60, 0.28 + sp.tr_z * 0.12)),
            rsi_max=max(22.0, min(48.0, 18.0 + sp.vol_z * 6.0)),
            cloc_min=sp.cloc_min,
        )
        return capitulation_reversal.build_features(data, params, tf, bt)
    if strategy_id == "liquidity_sweep":
        params = LiquiditySweepParams(
            ema_fast=sp.ema_fast,
            ema_slow=sp.ema_slow,
            atr_window=sp.atr_window,
            swing_lookback=max(16, min(120, int(sp.z_window / 14))),
            sweep_pct=max(0.0005, min(0.004, sp.tr_z * 0.001)),
            vol_mult=max(1.1, min(2.8, 0.8 + sp.vol_z * 0.35)),
            rsi_max=max(28.0, min(50.0, 20.0 + sp.vol_z * 5.0)),
            cloc_min=sp.cloc_min,
        )
        return liquidity_sweep.build_features(data, params, tf, bt)
    if strategy_id == "divergence_bounce":
        params = DivergenceParams(
            ema_fast=sp.ema_fast,
            ema_slow=sp.ema_slow,
            atr_window=sp.atr_window,
            vol_z_window=sp.z_window,
            div_lookback=max(12, min(96, int(sp.z_window / 21))),
            rsi_max=max(28.0, min(42.0, 22.0 + sp.vol_z * 4.0)),
            vol_z_min=max(0.5, min(1.8, sp.tr_z * 0.45)),
            min_rsi_delta=max(1.5, min(8.0, sp.cloc_min * 8.0)),
            cloc_min=sp.cloc_min,
        )
        return divergence_bounce.build_features(data, params, tf, bt)

    return build(strategy_id, data, tf, bt)


def evaluate_strategy_individual(
    strategy_id: str,
    individual: Individual,
    data: pd.DataFrame,
    tf: Timeframe,
    fitness_config: FitnessConfig | None = None,
    generation: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    try:
        feats = individual_to_features(strategy_id, individual, data, tf)
        result = run_backtest(feats, individual.backtest_params)
        metrics = result["metrics"]
        fitness = calculate_fitness(metrics, fitness_config, generation=generation)

        if fitness_config and fitness_config.preset == "profit_focused":
            account = simulate_account(result["trades"])
            ps = profit_score(metrics, account)
            fitness = 0.55 * ps + 0.45 * fitness

        return fitness, metrics
    except Exception:
        return -1000.0, {"n": 0}


def split_train_oos(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * train_ratio)
    cut = max(500, min(cut, len(df) - 500))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def run_strategy_ga(
    strategy_id: str,
    train_data: pd.DataFrame,
    tf: Timeframe,
    generations: int = 20,
    population_size: int = 16,
    fitness_config: FitnessConfig | None = None,
    seed: Individual | None = None,
) -> Population:
    """Run GA on train set for one strategy (sequential, strategy-aware eval)."""
    import backtest.optimizer as opt

    fitness_config = fitness_config or FitnessConfig.get_preset_config("profit_focused")
    pop = Population(size=population_size, seed_individual=seed, timeframe=tf)

    def _eval(ind, data, tf_, fitness_config=None, generation=0):
        return evaluate_strategy_individual(
            strategy_id, ind, data, tf_, fitness_config, generation
        )

    original = opt.evaluate_individual
    opt.evaluate_individual = _eval
    try:
        for _ in range(generations):
            pop = evolution_step(
                pop,
                train_data,
                tf,
                fitness_config=fitness_config,
                mutation_rate=0.35,
                sigma=0.12,
                elite_fraction=0.15,
                tournament_size=3,
                mutation_probability=0.85,
            )
    finally:
        opt.evaluate_individual = original

    return pop


def evaluate_oos(
    strategy_id: str,
    individual: Individual,
    oos_data: pd.DataFrame,
    tf: Timeframe,
) -> Dict[str, Any]:
    feats = individual_to_features(strategy_id, individual, oos_data, tf)
    return run_backtest(feats, individual.backtest_params)["metrics"]
