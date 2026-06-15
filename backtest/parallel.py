"""
Parallel CPU evaluation for the genetic algorithm.

Uses multiprocessing with spawn context (Qt-safe). Each worker runs the same
pandas feature pipeline and backtest engine as the single-threaded path.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from backtest.engine import run_backtest
from backtest.optimizer import Individual, calculate_fitness
from config.settings import settings
from core.logging_utils import get_logger
from core.models import FitnessConfig, Timeframe
from strategy.seller_exhaustion import build_features

logger = get_logger(__name__)


def dataframe_to_payload(df: pd.DataFrame) -> dict:
    """Serialize DataFrame for pickling across process boundaries."""
    return {
        "values": df.values,
        "index": df.index,
        "columns": df.columns.tolist(),
    }


def payload_to_dataframe(payload: dict) -> pd.DataFrame:
    """Reconstruct DataFrame from worker payload."""
    return pd.DataFrame(payload["values"], index=payload["index"], columns=payload["columns"])


def evaluate_individual_worker(args: Tuple) -> Tuple[int, float, Dict[str, Any]]:
    """
    Evaluate one individual in a worker process.

    Returns (index_in_batch, fitness, metrics).
    """
    idx, seller_params, backtest_params, data_dict, tf, fitness_config, generation = args
    data = payload_to_dataframe(data_dict)

    try:
        feats = build_features(data, seller_params, tf)
        result = run_backtest(feats, backtest_params)
        fitness = calculate_fitness(result["metrics"], fitness_config, generation=generation)
        return idx, float(fitness), result["metrics"]
    except Exception:
        logger.debug("Worker evaluation failed for index %s", idx, exc_info=True)
        return idx, -100.0, {
            "n": 0,
            "win_rate": 0.0,
            "avg_R": 0.0,
            "total_pnl": 0.0,
            "max_dd": 0.0,
        }


def evaluate_population_parallel(
    individuals: List[Individual],
    data: pd.DataFrame,
    tf: Timeframe,
    fitness_config: Optional[FitnessConfig] = None,
    generation: int = 0,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
) -> None:
    """
    Evaluate all unevaluated individuals in-place using a process pool.

    Args:
        individuals: Population members (fitness == 0.0 are evaluated)
        data: Raw OHLCV DataFrame
        tf: Active timeframe
        fitness_config: Fitness weights / curriculum config
        generation: Current GA generation (for curriculum learning)
        n_workers: Process count (defaults to CPU count)
        show_progress: Whether to show tqdm bar
    """
    unevaluated = [ind for ind in individuals if ind.fitness == 0.0]
    if not unevaluated:
        return

    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = max(1, min(n_workers, len(unevaluated)))

    data_dict = dataframe_to_payload(data)
    args_list = [
        (i, ind.seller_params, ind.backtest_params, data_dict, tf, fitness_config, generation)
        for i, ind in enumerate(unevaluated)
    ]

    try:
        ctx = mp.get_context("spawn")
    except ValueError:
        ctx = mp.get_context()

    disable_bar = not (show_progress and getattr(settings, "log_progress_bars", True))
    logger.info("[Gen %s] Parallel eval: %s individuals on %s workers", generation, len(unevaluated), n_workers)

    with ctx.Pool(processes=n_workers) as pool:
        with tqdm(
            total=len(args_list),
            desc=f"Gen {generation} eval",
            unit="ind",
            leave=False,
            disable=disable_bar,
        ) as pbar:
            for idx, fitness, metrics in pool.imap_unordered(evaluate_individual_worker, args_list):
                unevaluated[idx].fitness = fitness
                unevaluated[idx].metrics = metrics
                pbar.update(1)
