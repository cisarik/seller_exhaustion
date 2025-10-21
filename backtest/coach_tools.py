"""
Evolution Coach Tools

Provides 27 tools for agent-based coaching with full GA control.
Tools organized by category:
1. Observability (8 tools) - Query population state
2. Individual Manipulation (3 tools) - Direct control over individuals  
3. GA Algorithm Steering (6 tools) - Evolution mechanics
4. Fitness Function Control (9 tools) - What we optimize
5. Control Flow (1 tool) - Session management
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from backtest.optimizer import Population
from backtest.coach_session import CoachAnalysisSession
from backtest.coach_mutations import CoachMutationManager
from core.models import FitnessConfig, OptimizationConfig, BacktestParams
from strategy.seller_exhaustion import SellerParams
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    tool_name: str
    result: Dict[str, Any]
    error: Optional[str] = None


class CoachToolkit:
    """
    Toolkit for Evolution Coach Agent.
    
    Provides analytical and mutation tools for agent to steer evolution.
    """
    
    def __init__(
        self,
        population: Population,
        session: CoachAnalysisSession,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig,
        mutation_manager: CoachMutationManager
    ):
        self.population = population
        self.session = session
        self.fitness_config = fitness_config
        self.ga_config = ga_config
        self.mutation_manager = mutation_manager
        
        # Track actions taken
        self.actions_log: List[Dict[str, Any]] = []
    
    # ========================================================================
    # CATEGORY 1: OBSERVABILITY (8 tools)
    # ========================================================================
    
    async def analyze_population(
        self,
        group_by: str = "fitness",
        top_n: int = 5,
        bottom_n: int = 3,
        include_params: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive population statistics and identify patterns.
        
        Use this tool to understand:
        - Current fitness distribution (mean, std, min, max)
        - Diversity level (0.0-1.0, where <0.15 is very low)
        - Top and bottom performers
        - Gate compliance (% below min_trades)
        - Stagnation status
        
        Args:
            group_by: Sort criterion - "fitness", "trade_count", "win_rate", "avg_r"
            top_n: Number of top individuals to show
            bottom_n: Number of bottom individuals to show
            include_params: Include full parameter sets
        """
        try:
            individuals = self.population.individuals
            
            # Sort by grouping criterion
            if group_by == "fitness":
                sorted_inds = sorted(individuals, key=lambda x: x.fitness, reverse=True)
            elif group_by == "trade_count":
                sorted_inds = sorted(individuals, key=lambda x: x.metrics.get('n', 0), reverse=True)
            elif group_by == "win_rate":
                sorted_inds = sorted(individuals, key=lambda x: x.metrics.get('win_rate', 0), reverse=True)
            elif group_by == "avg_r":
                sorted_inds = sorted(individuals, key=lambda x: x.metrics.get('avg_R', 0), reverse=True)
            else:
                sorted_inds = individuals
            
            # Get top and bottom
            top_individuals = [self._individual_summary(i, idx, include_params) for idx, i in enumerate(sorted_inds[:top_n])]
            bottom_individuals = [self._individual_summary(i, idx, include_params) for idx, i in enumerate(sorted_inds[-bottom_n:])]
            
            # Calculate fitness stats
            fitness_vals = [i.fitness for i in individuals]
            
            # Calculate metrics
            trade_counts = [i.metrics.get('n', 0) for i in individuals]
            win_rates = [i.metrics.get('win_rate', 0) for i in individuals if i.metrics.get('n', 0) > 0]
            avg_rs = [i.metrics.get('avg_R', 0) for i in individuals if i.metrics.get('n', 0) > 0]
            pnls = [i.metrics.get('total_pnl', 0) for i in individuals if i.metrics.get('n', 0) > 0]
            
            # Calculate diversity
            diversity = self.population.get_diversity_metric()
            
            # Count below gates
            min_trades = self.fitness_config.get_effective_min_trades(self.population.generation)
            below_gates = sum(1 for ind in individuals if ind.metrics.get('n', 0) < min_trades)
            
            result = {
                "success": True,
                "population_size": len(individuals),
                "generation": self.population.generation,
                "fitness": {
                    "mean": float(np.mean(fitness_vals)),
                    "std": float(np.std(fitness_vals)),
                    "min": float(np.min(fitness_vals)),
                    "max": float(np.max(fitness_vals)),
                    "median": float(np.median(fitness_vals)),
                    "quartiles": [float(q) for q in np.percentile(fitness_vals, [25, 50, 75])]
                },
                "metrics": {
                    "mean_trades": float(np.mean(trade_counts)) if trade_counts else 0,
                    "std_trades": float(np.std(trade_counts)) if trade_counts else 0,
                    "mean_win_rate": float(np.mean(win_rates)) if win_rates else 0,
                    "mean_avg_r": float(np.mean(avg_rs)) if avg_rs else 0,
                    "mean_pnl": float(np.mean(pnls)) if pnls else 0
                },
                "gates": {
                    "min_trades": min_trades,
                    "min_win_rate": self.fitness_config.min_win_rate,
                    "below_min_trades": below_gates,
                    "below_min_trades_pct": 100.0 * below_gates / len(individuals),
                    "passing_all_gates": len(individuals) - below_gates
                },
                "diversity": {
                    "metric": float(diversity),
                    "interpretation": self._interpret_diversity(diversity)
                },
                "top_individuals": top_individuals,
                "bottom_individuals": bottom_individuals
            }
            
            self.actions_log.append({"action": "analyze_population", "result": "success"})
            return result
        
        except Exception as e:
            logger.exception("analyze_population failed")
            return {"success": False, "error": str(e)}
    
    async def get_param_distribution(
        self,
        parameter_name: str,
        bins: int = 5,
        correlate_with: Optional[str] = None,
        show_by_fitness_quartile: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze how a specific parameter is distributed across population.
        
        Use this tool to:
        - See which parameter values lead to high fitness
        - Identify boundary clustering (population hitting bounds)
        - Find correlations with fitness/metrics
        - Compare top vs bottom performers
        
        Args:
            parameter_name: Parameter to analyze (e.g., "ema_fast", "vol_z")
            bins: Number of histogram bins
            correlate_with: Metric to correlate with - "fitness", "trade_count", "win_rate", "avg_r"
            show_by_fitness_quartile: Split analysis by fitness quartile
        """
        try:
            individuals = self.population.individuals
            
            # Extract parameter values
            param_values = []
            for ind in individuals:
                # Get from seller_params or backtest_params
                if hasattr(ind.seller_params, parameter_name):
                    param_values.append(getattr(ind.seller_params, parameter_name))
                elif hasattr(ind.backtest_params, parameter_name):
                    param_values.append(getattr(ind.backtest_params, parameter_name))
                else:
                    return {"success": False, "error": f"Parameter {parameter_name} not found"}
            
            if not param_values:
                return {"success": False, "error": "No parameter values found"}
            
            # Statistics
            stats = {
                "min": float(np.min(param_values)),
                "max": float(np.max(param_values)),
                "mean": float(np.mean(param_values)),
                "std": float(np.std(param_values)),
                "median": float(np.median(param_values))
            }
            
            # Histogram
            hist_counts, hist_bins = np.histogram(param_values, bins=bins)
            histogram = {
                "bins": [f"{hist_bins[i]:.2f}-{hist_bins[i+1]:.2f}" for i in range(len(hist_counts))],
                "counts": hist_counts.tolist(),
                "percentages": (100 * hist_counts / len(param_values)).tolist()
            }
            
            # Correlation
            correlation_result = None
            if correlate_with:
                if correlate_with == "fitness":
                    metric_values = [i.fitness for i in individuals]
                elif correlate_with == "trade_count":
                    metric_values = [i.metrics.get('n', 0) for i in individuals]
                elif correlate_with == "win_rate":
                    metric_values = [i.metrics.get('win_rate', 0) for i in individuals]
                elif correlate_with == "avg_r":
                    metric_values = [i.metrics.get('avg_R', 0) for i in individuals]
                else:
                    metric_values = None
                
                if metric_values:
                    r, p = pearsonr(param_values, metric_values)
                    correlation_result = {
                        "with": correlate_with,
                        "pearson_r": float(r),
                        "p_value": float(p),
                        "interpretation": self._interpret_correlation(r),
                        "significant": p < 0.05
                    }
            
            # Quartile analysis
            by_quartile = None
            if show_by_fitness_quartile:
                sorted_by_fitness = sorted(enumerate(individuals), key=lambda x: x[1].fitness, reverse=True)
                n = len(sorted_by_fitness)
                quartiles = {
                    "top_25%": [param_values[idx] for idx, _ in sorted_by_fitness[:n//4]],
                    "q2": [param_values[idx] for idx, _ in sorted_by_fitness[n//4:n//2]],
                    "q3": [param_values[idx] for idx, _ in sorted_by_fitness[n//2:3*n//4]],
                    "bottom_25%": [param_values[idx] for idx, _ in sorted_by_fitness[3*n//4:]]
                }
                
                by_quartile = {
                    name: {
                        "mean": float(np.mean(vals)) if vals else 0,
                        "std": float(np.std(vals)) if vals else 0,
                        "range": [float(np.min(vals)), float(np.max(vals))] if vals else [0, 0]
                    }
                    for name, vals in quartiles.items()
                }
            
            result = {
                "success": True,
                "parameter": parameter_name,
                "stats": stats,
                "histogram": histogram,
                "correlation": correlation_result,
                "by_quartile": by_quartile
            }
            
            self.actions_log.append({"action": "get_param_distribution", "parameter": parameter_name})
            return result
        
        except Exception as e:
            logger.exception(f"get_param_distribution failed for {parameter_name}")
            return {"success": False, "error": str(e)}
    
    async def get_param_bounds(
        self,
        parameters: Optional[List[str]] = None,
        include_clustering: bool = True
    ) -> Dict[str, Any]:
        """
        Query current search space bounds and identify boundary clustering.
        
        Use this tool to:
        - Check current parameter bounds
        - Identify boundary clustering (>30% at bounds)
        - Decide whether to expand bounds
        
        Args:
            parameters: Specific params to query, or None for all
            include_clustering: Show boundary clustering analysis
        """
        try:
            from backtest.optimizer import get_param_bounds_for_timeframe
            from core.models import Timeframe
            
            # Get current bounds (assuming 15m timeframe for now)
            bounds = get_param_bounds_for_timeframe(Timeframe.m15)
            
            result = {
                "success": True,
                "bounds": {}
            }
            
            # Filter to requested parameters
            if parameters:
                bounds = {k: v for k, v in bounds.items() if k in parameters}
            
            # Add bounds and clustering info
            for param_name, (min_val, max_val) in bounds.items():
                param_info = {
                    "min": float(min_val),
                    "max": float(max_val),
                    "type": "int" if isinstance(min_val, int) else "float"
                }
                
                if include_clustering:
                    clustering = self._calculate_boundary_clustering(param_name, min_val, max_val)
                    param_info["clustering"] = clustering
                
                result["bounds"][param_name] = param_info
            
            self.actions_log.append({"action": "get_param_bounds"})
            return result
        
        except Exception as e:
            logger.exception("get_param_bounds failed")
            return {"success": False, "error": str(e)}
    
    # ========================================================================
    # CATEGORY 2: INDIVIDUAL MANIPULATION (3 tools)
    # ========================================================================
    
    async def mutate_individual(
        self,
        individual_id: int,
        parameter_name: str,
        new_value: Any,
        reason: str,
        respect_bounds: bool = True
    ) -> Dict[str, Any]:
        """
        Directly modify a specific parameter of a specific individual.
        
        Use this tool to:
        - Explore nearby regions around successful individuals
        - Test hypotheses about parameter effects
        - Repair obviously broken individuals
        - Create directed exploration
        
        Args:
            individual_id: 0-indexed individual ID
            parameter_name: Parameter to mutate
            new_value: New value
            reason: Explanation for this mutation
            respect_bounds: Enforce parameter bounds
        """
        try:
            if individual_id < 0 or individual_id >= len(self.population.individuals):
                return {"success": False, "error": f"Invalid individual_id: {individual_id}"}
            
            individual = self.population.individuals[individual_id]
            
            # Get old value
            if hasattr(individual.seller_params, parameter_name):
                old_value = getattr(individual.seller_params, parameter_name)
                setattr(individual.seller_params, parameter_name, new_value)
            elif hasattr(individual.backtest_params, parameter_name):
                old_value = getattr(individual.backtest_params, parameter_name)
                setattr(individual.backtest_params, parameter_name, new_value)
            else:
                return {"success": False, "error": f"Parameter {parameter_name} not found"}
            
            # Reset fitness for re-evaluation
            individual.fitness = 0.0
            
            result = {
                "success": True,
                "individual_id": individual_id,
                "parameter": parameter_name,
                "old_value": old_value,
                "new_value": new_value,
                "change": new_value - old_value if isinstance(new_value, (int, float)) else "N/A",
                "reason": reason,
                "impact": {
                    "fitness_reset": True,
                    "will_compete_in_next_gen": True
                }
            }
            
            self.actions_log.append({
                "action": "mutate_individual",
                "individual_id": individual_id,
                "parameter": parameter_name,
                "old_value": old_value,
                "new_value": new_value
            })
            
            return result
        
        except Exception as e:
            logger.exception(f"mutate_individual failed for individual {individual_id}")
            return {"success": False, "error": str(e)}
    
    async def update_fitness_gates(
        self,
        min_trades: Optional[int] = None,
        min_win_rate: Optional[float] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Update fitness gate requirements (hard thresholds).
        
        Use this tool to:
        - Lower gates when too many individuals fail (>80% below threshold)
        - Raise gates to increase selectivity
        - Balance signal frequency vs quality
        
        Args:
            min_trades: Minimum trades required
            min_win_rate: Minimum win rate required
            reason: Explanation for this change
        """
        try:
            old_min_trades = self.fitness_config.min_trades
            old_min_wr = self.fitness_config.min_win_rate
            
            changes = {}
            if min_trades is not None:
                self.fitness_config.min_trades = min_trades
                changes["min_trades"] = {"old": old_min_trades, "new": min_trades}
            
            if min_win_rate is not None:
                self.fitness_config.min_win_rate = min_win_rate
                changes["min_win_rate"] = {"old": old_min_wr, "new": min_win_rate}
            
            result = {
                "success": True,
                "changes_applied": changes,
                "reason": reason
            }
            
            self.actions_log.append({"action": "update_fitness_gates", "changes": changes})
            return result
        
        except Exception as e:
            logger.exception("update_fitness_gates failed")
            return {"success": False, "error": str(e)}
    
    async def update_ga_params(
        self,
        mutation_probability: Optional[float] = None,
        mutation_rate: Optional[float] = None,
        sigma: Optional[float] = None,
        tournament_size: Optional[int] = None,
        elite_fraction: Optional[float] = None,
        immigrant_fraction: Optional[float] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Adjust genetic algorithm evolution mechanics.
        
        Use this tool to:
        - Increase exploration when stagnant or converged early
        - Decrease exploration when refining good solutions
        - Inject diversity when low (<0.15)
        
        Args:
            mutation_probability: Chance each individual mutates (0.0-1.0)
            mutation_rate: How much parameters change (0.0-1.0)
            sigma: Gaussian mutation std dev
            tournament_size: Selection pressure (2-8)
            elite_fraction: Top % preserved unchanged (0.0-0.4)
            immigrant_fraction: Random injection rate (0.0-0.3)
            reason: Explanation for changes
        """
        try:
            changes = {}
            
            if mutation_probability is not None:
                old = self.ga_config.mutation_probability
                self.ga_config.mutation_probability = mutation_probability
                changes["mutation_probability"] = {"old": old, "new": mutation_probability}
            
            if mutation_rate is not None:
                old = self.ga_config.mutation_rate
                self.ga_config.mutation_rate = mutation_rate
                changes["mutation_rate"] = {"old": old, "new": mutation_rate}
            
            if sigma is not None:
                old = self.ga_config.sigma
                self.ga_config.sigma = sigma
                changes["sigma"] = {"old": old, "new": sigma}
            
            if tournament_size is not None:
                old = self.ga_config.tournament_size
                self.ga_config.tournament_size = tournament_size
                changes["tournament_size"] = {"old": old, "new": tournament_size}
            
            if elite_fraction is not None:
                old = self.ga_config.elite_fraction
                self.ga_config.elite_fraction = elite_fraction
                changes["elite_fraction"] = {"old": old, "new": elite_fraction}
            
            if immigrant_fraction is not None:
                old = self.ga_config.immigrant_fraction
                self.ga_config.immigrant_fraction = immigrant_fraction
                changes["immigrant_fraction"] = {"old": old, "new": immigrant_fraction}
            
            result = {
                "success": True,
                "changes_applied": changes,
                "reason": reason
            }
            
            self.actions_log.append({"action": "update_ga_params", "changes": changes})
            return result
        
        except Exception as e:
            logger.exception("update_ga_params failed")
            return {"success": False, "error": str(e)}
    
    async def finish_analysis(
        self,
        summary: str,
        overall_assessment: str = "neutral",
        stagnation_detected: bool = False,
        diversity_concern: bool = False
    ) -> Dict[str, Any]:
        """
        Complete analysis session and return control to GA.
        
        Call this tool when:
        - You've made all necessary interventions
        - Ready to let evolution run with your changes
        - Max ~5-7 tool calls made
        
        Args:
            summary: 1-2 sentence summary of actions taken
            overall_assessment: "positive" | "neutral" | "needs_adjustment"
            stagnation_detected: Is evolution stagnant?
            diversity_concern: Is diversity too low?
        """
        summary_data = {
            "success": True,
            "summary": summary,
            "overall_assessment": overall_assessment,
            "stagnation_detected": stagnation_detected,
            "diversity_concern": diversity_concern,
            "total_actions": len(self.actions_log),
            "actions_log": self.actions_log
        }
        
        self.actions_log.append({"action": "finish_analysis"})
        return summary_data
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _individual_summary(self, individual, idx: int, include_params: bool = False) -> Dict[str, Any]:
        """Create summary of individual for display."""
        summary = {
            "id": idx,
            "fitness": individual.fitness,
            "metrics": {
                "n": individual.metrics.get('n', 0),
                "win_rate": individual.metrics.get('win_rate', 0),
                "avg_r": individual.metrics.get('avg_R', 0),
                "pnl": individual.metrics.get('total_pnl', 0)
            },
            "key_params": {
                "ema_fast": individual.seller_params.ema_fast,
                "ema_slow": individual.seller_params.ema_slow,
                "vol_z": individual.seller_params.vol_z,
                "tr_z": individual.seller_params.tr_z
            }
        }
        
        if include_params:
            # Handle both dataclass and Pydantic models
            from dataclasses import is_dataclass
            
            # seller_params is a dataclass
            if is_dataclass(individual.seller_params):
                seller_dict = asdict(individual.seller_params)
            else:
                seller_dict = individual.seller_params.model_dump() if hasattr(individual.seller_params, 'model_dump') else dict(individual.seller_params)
            
            # backtest_params is a Pydantic model
            if hasattr(individual.backtest_params, 'model_dump'):
                backtest_dict = individual.backtest_params.model_dump()
            elif is_dataclass(individual.backtest_params):
                backtest_dict = asdict(individual.backtest_params)
            else:
                backtest_dict = dict(individual.backtest_params)
            
            summary["full_params"] = {
                "seller_params": seller_dict,
                "backtest_params": backtest_dict
            }
        
        return summary
    
    def _interpret_diversity(self, diversity: float) -> str:
        """Interpret diversity metric."""
        if diversity < 0.1:
            return "very_low"
        elif diversity < 0.2:
            return "low"
        elif diversity < 0.4:
            return "moderate"
        else:
            return "high"
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret Pearson correlation coefficient."""
        abs_r = abs(r)
        if abs_r > 0.7:
            direction = "positive" if r > 0 else "negative"
            return f"strong_{direction}"
        elif abs_r > 0.4:
            direction = "positive" if r > 0 else "negative"
            return f"moderate_{direction}"
        elif abs_r > 0.2:
            return "weak"
        else:
            return "negligible"
    
    def _calculate_boundary_clustering(self, param_name: str, min_val: float, max_val: float) -> Dict[str, Any]:
        """Calculate how many individuals are at or near bounds."""
        individuals = self.population.individuals
        param_values = []
        
        for ind in individuals:
            if hasattr(ind.seller_params, param_name):
                param_values.append(getattr(ind.seller_params, param_name))
            elif hasattr(ind.backtest_params, param_name):
                param_values.append(getattr(ind.backtest_params, param_name))
        
        if not param_values:
            return {"at_min": 0, "at_max": 0, "interpretation": "No data"}
        
        # Count at bounds (exact match)
        at_min = sum(1 for v in param_values if abs(v - min_val) < 1e-6)
        at_max = sum(1 for v in param_values if abs(v - max_val) < 1e-6)
        
        # Count near bounds (within 10% of range)
        range_width = max_val - min_val
        threshold = 0.1 * range_width
        near_min = sum(1 for v in param_values if v < min_val + threshold)
        near_max = sum(1 for v in param_values if v > max_val - threshold)
        
        in_middle = len(param_values) - near_min - near_max
        
        # Interpretation
        pct_at_bounds = 100 * (at_min + at_max) / len(param_values)
        if pct_at_bounds > 30:
            interpretation = "Heavy clustering at bounds - consider expanding"
        elif pct_at_bounds > 15:
            interpretation = "Moderate boundary pressure"
        else:
            interpretation = "Good distribution"
        
        return {
            "at_min": at_min,
            "at_max": at_max,
            "near_min": near_min,
            "near_max": near_max,
            "in_middle": in_middle,
            "interpretation": interpretation
        }
