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
        mutation_manager: CoachMutationManager,
        islands_registry: Optional[Dict[int, Population]] = None,
        island_policy_reference: Optional[Dict[str, Any]] = None
    ):
        self.population = population
        self.session = session
        self.fitness_config = fitness_config
        self.ga_config = ga_config
        self.mutation_manager = mutation_manager
        
        # Track actions taken
        self.actions_log: List[Dict[str, Any]] = []
        # Optional island model registry (persistent across sessions if provided)
        self._islands: Dict[int, Population] = islands_registry if islands_registry is not None else {}
        # Optional policy dict reference owned by manager
        self._island_policy = island_policy_reference if island_policy_reference is not None else {}
    
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

    async def get_correlation_matrix(
        self,
        include_params: Optional[List[str]] = None,
        correlate_with: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute Pearson correlations between parameters and selected metrics."""
        try:
            # Parameters to evaluate
            all_params = [
                'ema_fast','ema_slow','z_window','atr_window','vol_z','tr_z','cloc_min',
                'fib_swing_lookback','fib_swing_lookahead','fib_target_level','fee_bp','slippage_bp'
            ]
            params = include_params or all_params
            metrics_wanted = correlate_with or ['fitness','trade_count','win_rate','avg_r']

            # Gather vectors
            def get_param(ind, name):
                if hasattr(ind.seller_params, name):
                    return getattr(ind.seller_params, name)
                if hasattr(ind.backtest_params, name):
                    return getattr(ind.backtest_params, name)
                return None

            inds = self.population.individuals
            by_param = {p: [get_param(ind, p) for ind in inds] for p in params}
            by_metric = {
                'fitness': [ind.fitness for ind in inds],
                'trade_count': [ind.metrics.get('n', 0) for ind in inds],
                'win_rate': [ind.metrics.get('win_rate', 0) for ind in inds],
                'avg_r': [ind.metrics.get('avg_R', 0) for ind in inds],
            }

            correlations = {}
            for metric in metrics_wanted:
                correlations[metric] = {}
                y = by_metric.get(metric)
                for p in params:
                    x = by_param[p]
                    try:
                        r, pval = pearsonr(x, y)
                        correlations[metric][p] = {
                            'r': float(r),
                            'p': float(pval),
                            'sig': bool(pval < 0.05)
                        }
                    except Exception:
                        correlations[metric][p] = {'r': 0.0, 'p': 1.0, 'sig': False}

            # Rank importance by |r| with fitness
            rank = sorted(
                (
                    (p, abs(correlations['fitness'][p]['r']))
                    for p in params if 'fitness' in correlations and p in correlations['fitness']
                ), key=lambda t: t[1], reverse=True
            )
            ranked = [{'param': p, 'abs_r': v} for p, v in rank]

            self.actions_log.append({"action": "get_correlation_matrix"})
            return {"success": True, "correlations": correlations, "ranked_by_fitness_abs_r": ranked}
        except Exception as e:
            logger.exception("get_correlation_matrix failed")
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

    # ========================================================================
    # CATEGORY 2b: LLM-INDIVIDUALS & ISLANDS (NEW)
    # ========================================================================
    
    async def insert_llm_individual(
        self,
        destination: str,
        individual: Dict[str, Any],
        island_id: Optional[int] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Insert a new individual provided by the LLM into the main population or a specific island.
        The individual's fitness is reset for evaluation in the next generation.
        """
        try:
            sp_dict = individual.get("seller_params") or {}
            bp_dict = individual.get("backtest_params") or {}
            from strategy.seller_exhaustion import SellerParams
            from core.models import BacktestParams
            from backtest.optimizer import Individual as PopIndividual
            sp = SellerParams(**sp_dict)
            bp = BacktestParams(**bp_dict)
            ind = PopIndividual(seller_params=sp, backtest_params=bp, fitness=0.0, metrics={}, generation=self.population.generation)

            if destination == "main":
                self.population.individuals.append(ind)
                self.population.size = len(self.population.individuals)
                target = "main"
                index = self.population.size - 1
            elif destination == "island":
                if not hasattr(self, "_islands"):
                    self._islands = {}
                if island_id is None or island_id not in self._islands:
                    return {"success": False, "error": "Invalid or missing island_id"}
                self._islands[island_id].individuals.append(ind)
                self._islands[island_id].size = len(self._islands[island_id].individuals)
                target = f"island:{island_id}"
                index = self._islands[island_id].size - 1
            else:
                return {"success": False, "error": f"Unknown destination: {destination}"}

            self.actions_log.append({
                "action": "insert_llm_individual",
                "destination": destination,
                "island_id": island_id,
                "reason": reason
            })
            return {"success": True, "destination": target, "index": index}
        except Exception as e:
            logger.exception("insert_llm_individual failed")
            return {"success": False, "error": str(e)}

    async def create_islands(self, count: int = 2, strategy: str = "split") -> Dict[str, Any]:
        """Create multiple sub‑populations (islands) from the current population."""
        try:
            if count < 2:
                return {"success": False, "error": "count must be >= 2"}
            # Reset previous islands
            self._islands = {}
            from copy import deepcopy
            if strategy == "split":
                inds = list(self.population.individuals)
                # Create island containers
                for i in range(count):
                    island = Population(size=0, timeframe=self.population.timeframe)
                    island.individuals = []
                    island.size = 0
                    island.bounds = self.population.bounds
                    island.generation = self.population.generation
                    self._islands[i] = island
                # Distribute individuals round‑robin
                for idx, ind in enumerate(inds):
                    target_id = idx % count
                    self._islands[target_id].individuals.append(deepcopy(ind))
                    self._islands[target_id].size = len(self._islands[target_id].individuals)
            else:
                return {"success": False, "error": f"Unsupported strategy: {strategy}"}

            islands_info = [{"id": k, "size": v.size} for k, v in self._islands.items()]
            self.actions_log.append({"action": "create_islands", "count": count, "strategy": strategy})
            return {"success": True, "islands": islands_info}
        except Exception as e:
            logger.exception("create_islands failed")
            return {"success": False, "error": str(e)}

    async def migrate_between_islands(self, src_island: int, dst_island: int, individual_id: int, reason: str = "") -> Dict[str, Any]:
        """Migrate an individual from one island to another."""
        try:
            if not hasattr(self, "_islands") or src_island not in self._islands or dst_island not in self._islands:
                return {"success": False, "error": "Invalid island id"}
            src = self._islands[src_island]
            dst = self._islands[dst_island]
            if individual_id < 0 or individual_id >= len(src.individuals):
                return {"success": False, "error": "Invalid individual_id"}
            ind = src.individuals.pop(individual_id)
            src.size = len(src.individuals)
            dst.individuals.append(ind)
            dst.size = len(dst.individuals)
            self.actions_log.append({"action": "migrate_between_islands", "src": src_island, "dst": dst_island, "individual_id": individual_id, "reason": reason})
            return {"success": True}
        except Exception as e:
            logger.exception("migrate_between_islands failed")
            return {"success": False, "error": str(e)}

    async def configure_island_scheduler(
        self,
        migration_cadence: Optional[int] = None,
        migration_size: Optional[int] = None,
        merge_to_main_cadence: Optional[int] = None,
        merge_top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Configure island migration cadence/size and island→main merge policy."""
        try:
            changes = {}
            def apply(name, val):
                if val is not None:
                    old = self._island_policy.get(name)
                    self._island_policy[name] = int(val)
                    changes[name] = {"old": old, "new": int(val)}
            apply("migration_cadence", migration_cadence)
            apply("migration_size", migration_size)
            apply("merge_to_main_cadence", merge_to_main_cadence)
            apply("merge_top_k", merge_top_k)
            self.actions_log.append({"action": "configure_island_scheduler", "changes": changes})
            return {"success": True, "changes": changes}
        except Exception as e:
            logger.exception("configure_island_scheduler failed")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # CATEGORY 3: GA / POPULATION UTILITIES (NEW)
    # ========================================================================

    async def inject_immigrants(self, fraction: float = 0.15, strategy: str = "worst_replacement") -> Dict[str, Any]:
        """Inject random immigrants into the main population to boost diversity."""
        try:
            added = self.population.add_immigrants(fraction=fraction, strategy=strategy, generation=self.population.generation)
            self.actions_log.append({"action": "inject_immigrants", "fraction": fraction, "strategy": strategy, "added": added})
            return {"success": True, "added": int(added)}
        except Exception as e:
            logger.exception("inject_immigrants failed")
            return {"success": False, "error": str(e)}

    async def export_population(self, path: str) -> Dict[str, Any]:
        """Export current population to JSON file."""
        try:
            from backtest.optimizer import export_population as exp
            exp(self.population, path)
            self.actions_log.append({"action": "export_population", "path": path})
            return {"success": True, "path": path}
        except Exception as e:
            logger.exception("export_population failed")
            return {"success": False, "error": str(e)}

    async def import_population(self, path: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Import population from JSON and replace current individuals (size preserved if limit provided)."""
        try:
            from backtest.optimizer import import_population as imp
            imported = imp(path, timeframe=self.population.timeframe, limit=limit)
            self.population.individuals = imported.individuals
            self.population.size = len(imported.individuals)
            self.population.bounds = imported.bounds
            self.population.generation = imported.generation
            self.actions_log.append({"action": "import_population", "path": path, "size": self.population.size})
            return {"success": True, "size": self.population.size}
        except Exception as e:
            logger.exception("import_population failed")
            return {"success": False, "error": str(e)}

    async def drop_individual(self, individual_id: int, replace_with: str = "immigrant") -> Dict[str, Any]:
        """Drop an individual and optionally replace with a new immigrant to keep size constant."""
        try:
            if individual_id < 0 or individual_id >= len(self.population.individuals):
                return {"success": False, "error": "Invalid individual_id"}
            self.population.individuals.pop(individual_id)
            self.population.size = len(self.population.individuals)
            if replace_with == "immigrant":
                tmp = Population(size=1, timeframe=self.population.timeframe)
                self.population.individuals.append(tmp.individuals[0])
                self.population.size = len(self.population.individuals)
            self.actions_log.append({"action": "drop_individual", "id": individual_id, "replaced": replace_with == 'immigrant'})
            return {"success": True, "size": self.population.size}
        except Exception as e:
            logger.exception("drop_individual failed")
            return {"success": False, "error": str(e)}

    async def bulk_update_param(self, individual_ids: List[int], parameter_name: str, new_value) -> Dict[str, Any]:
        """Set a parameter to a new value for a group of individuals."""
        try:
            changed = 0
            for iid in individual_ids:
                if 0 <= iid < len(self.population.individuals):
                    ind = self.population.individuals[iid]
                    if hasattr(ind.seller_params, parameter_name):
                        setattr(ind.seller_params, parameter_name, new_value)
                        changed += 1
                    elif hasattr(ind.backtest_params, parameter_name):
                        setattr(ind.backtest_params, parameter_name, new_value)
                        changed += 1
                    ind.fitness = 0.0
            self.actions_log.append({"action": "bulk_update_param", "parameter": parameter_name, "changed": changed})
            return {"success": True, "changed": changed}
        except Exception as e:
            logger.exception("bulk_update_param failed")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # CATEGORY 4: BOUNDS & INSERTION (ADVANCED)
    # ========================================================================

    async def update_param_bounds(
        self,
        parameter: str,
        new_min=None,
        new_max=None,
        reason: str = "",
        retroactive: bool = False
    ) -> Dict[str, Any]:
        """Expand/contract parameter search bounds and optionally clamp existing individuals."""
        try:
            if parameter not in self.population.bounds:
                return {"success": False, "error": f"Unknown parameter: {parameter}"}
            old_min, old_max = self.population.bounds[parameter]
            min_v = old_min if new_min is None else new_min
            max_v = old_max if new_max is None else new_max
            # Apply override
            self.population.apply_bounds_override({parameter: (min_v, max_v)})
            impact = {"individuals_at_old_min": 0, "individuals_now_in_bounds": 0}
            if retroactive:
                # Clamp existing values
                import math
                for ind in self.population.individuals:
                    # Determine which object has the parameter
                    target = None
                    if hasattr(ind.seller_params, parameter):
                        target = ind.seller_params
                    elif hasattr(ind.backtest_params, parameter):
                        target = ind.backtest_params
                    if target is None:
                        continue
                    val = getattr(target, parameter)
                    if val == old_min:
                        impact["individuals_at_old_min"] += 1
                    if val < min_v or val > max_v:
                        new_val = min(max(val, min_v), max_v)
                        if isinstance(val, int):
                            new_val = int(round(new_val))
                        setattr(target, parameter, new_val)
                        impact["individuals_now_in_bounds"] += 1
                        ind.fitness = 0.0
            self.actions_log.append({"action": "update_param_bounds", "parameter": parameter, "old": [old_min, old_max], "new": [min_v, max_v], "retroactive": retroactive, "reason": reason})
            return {
                "success": True,
                "parameter": parameter,
                "old_bounds": {"min": old_min, "max": old_max},
                "new_bounds": {"min": min_v, "max": max_v},
                "population_impact": impact,
            }
        except Exception as e:
            logger.exception("update_param_bounds failed")
            return {"success": False, "error": str(e)}

    async def update_bounds_multi(
        self,
        bounds: Dict[str, Dict[str, float]],
        retroactive: bool = False
    ) -> Dict[str, Any]:
        """Update bounds for multiple parameters at once. bounds={'ema_fast': {'min':24,'max':192}, ...}"""
        try:
            applied = {}
            for param, b in bounds.items():
                mn = b.get('min'); mx = b.get('max')
                if param in self.population.bounds:
                    old = self.population.bounds[param]
                    self.population.apply_bounds_override({param: (mn if mn is not None else old[0], mx if mx is not None else old[1])})
                    applied[param] = {'old': old, 'new': self.population.bounds[param]}
            # Optionally clamp existing individuals
            clamped = 0
            if retroactive:
                for ind in self.population.individuals:
                    for param in bounds.keys():
                        target = ind.seller_params if hasattr(ind.seller_params, param) else (ind.backtest_params if hasattr(ind.backtest_params, param) else None)
                        if target is None: continue
                        val = getattr(target, param)
                        mn, mx = self.population.bounds[param]
                        if val < mn or val > mx:
                            new_val = max(min(val, mx), mn)
                            if isinstance(val, int): new_val = int(round(new_val))
                            setattr(target, param, new_val); clamped += 1; ind.fitness = 0.0
            self.actions_log.append({"action": "update_bounds_multi", "applied": applied, "retroactive": retroactive})
            return {"success": True, "applied": applied, "clamped": clamped}
        except Exception as e:
            logger.exception("update_bounds_multi failed")
            return {"success": False, "error": str(e)}

    async def reseed_population(self, fraction: float = 0.2, strategy: str = "worst_replacement") -> Dict[str, Any]:
        """Replace a fraction of the population immediately with random newcomers (hard reseed)."""
        try:
            added = self.population.add_immigrants(fraction=fraction, strategy=strategy, generation=self.population.generation)
            self.actions_log.append({"action": "reseed_population", "fraction": fraction, "added": added})
            return {"success": True, "reseeded": int(added)}
        except Exception as e:
            logger.exception("reseed_population failed")
            return {"success": False, "error": str(e)}

    async def insert_individual(
        self,
        strategy: str = "coach_designed",
        parameters: Optional[Dict[str, Any]] = None,
        clone_from_id: Optional[int] = None,
        mutations: Optional[Dict[str, Any]] = None,
        parent_ids: Optional[List[int]] = None,
        blend_strategy: str = "average",
        reason: str = "",
        position: Optional[int] = None
    ) -> Dict[str, Any]:
        """Add a new individual using multiple strategies (coach_designed, random, clone_best, hybrid)."""
        try:
            from strategy.seller_exhaustion import SellerParams
            from core.models import BacktestParams
            from backtest.optimizer import Individual as PopIndividual, Population as Pop
            new_ind = None

            if strategy == "coach_designed":
                if not parameters:
                    return {"success": False, "error": "parameters required for coach_designed"}
                sp_dict = parameters.get("seller_params") or {}
                bp_dict = parameters.get("backtest_params") or {}
                sp = SellerParams(**sp_dict)
                bp = BacktestParams(**bp_dict)
                new_ind = PopIndividual(seller_params=sp, backtest_params=bp, fitness=0.0, metrics={}, generation=self.population.generation)

            elif strategy == "random":
                tmp = Pop(size=1, timeframe=self.population.timeframe)
                new_ind = tmp.individuals[0]
                new_ind.generation = self.population.generation

            elif strategy == "clone_best":
                src_id = clone_from_id if clone_from_id is not None else max(range(len(self.population.individuals)), key=lambda i: self.population.individuals[i].fitness)
                base = self.population.individuals[src_id]
                sp = SellerParams(**base.seller_params.__dict__)
                bp = BacktestParams(**(base.backtest_params.model_dump() if hasattr(base.backtest_params, 'model_dump') else base.backtest_params.__dict__))
                # Apply mutations
                if mutations:
                    for k,v in mutations.items():
                        if hasattr(sp, k):
                            setattr(sp, k, v)
                        elif hasattr(bp, k):
                            setattr(bp, k, v)
                new_ind = PopIndividual(seller_params=sp, backtest_params=bp, fitness=0.0, metrics={}, generation=self.population.generation)

            elif strategy == "hybrid":
                if not parent_ids or len(parent_ids) < 2:
                    return {"success": False, "error": "parent_ids must include at least two ids"}
                p1 = self.population.individuals[parent_ids[0]]
                p2 = self.population.individuals[parent_ids[1]]
                # Blend
                def pick_num(a,b):
                    return (a+b)/2 if blend_strategy == 'average' else (a if p1.fitness >= p2.fitness else b)
                sp = SellerParams(
                    ema_fast=int(round(pick_num(p1.seller_params.ema_fast, p2.seller_params.ema_fast))),
                    ema_slow=int(round(pick_num(p1.seller_params.ema_slow, p2.seller_params.ema_slow))),
                    z_window=int(round(pick_num(p1.seller_params.z_window, p2.seller_params.z_window))),
                    vol_z=pick_num(p1.seller_params.vol_z, p2.seller_params.vol_z),
                    tr_z=pick_num(p1.seller_params.tr_z, p2.seller_params.tr_z),
                    cloc_min=pick_num(p1.seller_params.cloc_min, p2.seller_params.cloc_min),
                    atr_window=int(round(pick_num(p1.seller_params.atr_window, p2.seller_params.atr_window))),
                )
                from backtest.optimizer import VALID_FIB_LEVELS
                def pick_discrete(a,b):
                    return a if blend_strategy == 'best_of_each' and p1.fitness >= p2.fitness else (b if blend_strategy == 'best_of_each' else (a if np.random.rand()<0.5 else b))
                bp = BacktestParams(
                    fib_swing_lookback=int(round(pick_num(p1.backtest_params.fib_swing_lookback, p2.backtest_params.fib_swing_lookback))),
                    fib_swing_lookahead=int(round(pick_num(p1.backtest_params.fib_swing_lookahead, p2.backtest_params.fib_swing_lookahead))),
                    fib_target_level=pick_discrete(p1.backtest_params.fib_target_level, p2.backtest_params.fib_target_level),
                    fee_bp=pick_num(p1.backtest_params.fee_bp, p2.backtest_params.fee_bp),
                    slippage_bp=pick_num(p1.backtest_params.slippage_bp, p2.backtest_params.slippage_bp),
                )
                new_ind = PopIndividual(seller_params=sp, backtest_params=bp, fitness=0.0, metrics={}, generation=self.population.generation)
            else:
                return {"success": False, "error": f"Unknown strategy: {strategy}"}

            # Insert into population
            if position is None or position < 0 or position > len(self.population.individuals):
                self.population.individuals.append(new_ind)
                pos = len(self.population.individuals) - 1
            else:
                self.population.individuals.insert(position, new_ind)
                pos = position
            self.population.size = len(self.population.individuals)
            self.actions_log.append({"action": "insert_individual", "strategy": strategy, "position": pos, "reason": reason})
            return {"success": True, "new_individual_id": pos, "position": pos, "strategy": strategy}
        except Exception as e:
            logger.exception("insert_individual failed")
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
    # CATEGORY 5: FITNESS CONFIGURATION (NEW)
    # ========================================================================

    async def update_fitness_weights(
        self,
        trade_count_weight: Optional[float] = None,
        win_rate_weight: Optional[float] = None,
        avg_r_weight: Optional[float] = None,
        total_pnl_weight: Optional[float] = None,
        max_drawdown_penalty: Optional[float] = None,
        penalty_trades_strength: Optional[float] = None,
        penalty_wr_strength: Optional[float] = None
    ) -> Dict[str, Any]:
        """Adjust fitness weights and penalty strengths; renormalize weights to sum≈1.0."""
        try:
            fc = self.fitness_config
            changes = {}
            def apply(name, val):
                if val is not None:
                    old = getattr(fc, name)
                    setattr(fc, name, float(val))
                    changes[name] = {"old": old, "new": float(val)}
            apply('trade_count_weight', trade_count_weight)
            apply('win_rate_weight', win_rate_weight)
            apply('avg_r_weight', avg_r_weight)
            apply('total_pnl_weight', total_pnl_weight)
            apply('max_drawdown_penalty', max_drawdown_penalty)
            apply('penalty_trades_strength', penalty_trades_strength)
            apply('penalty_wr_strength', penalty_wr_strength)
            # Renormalize non-penalty weights to ~1.0
            s = fc.trade_count_weight + fc.win_rate_weight + fc.avg_r_weight + fc.total_pnl_weight
            if s > 0:
                fc.trade_count_weight /= s
                fc.win_rate_weight /= s
                fc.avg_r_weight /= s
                fc.total_pnl_weight /= s
            self.actions_log.append({"action": "update_fitness_weights", "changes": changes})
            return {"success": True, "changes": changes}
        except Exception as e:
            logger.exception("update_fitness_weights failed")
            return {"success": False, "error": str(e)}

    async def set_fitness_function_type(self, fitness_function_type: str) -> Dict[str, Any]:
        """Switch between 'hard_gates' and 'soft_penalties'."""
        try:
            old = self.fitness_config.fitness_function_type
            self.fitness_config.fitness_function_type = fitness_function_type
            self.actions_log.append({"action": "set_fitness_function_type", "old": old, "new": fitness_function_type})
            return {"success": True, "old": old, "new": fitness_function_type}
        except Exception as e:
            logger.exception("set_fitness_function_type failed")
            return {"success": False, "error": str(e)}

    async def configure_curriculum(
        self,
        enabled: Optional[bool] = None,
        start_min_trades: Optional[int] = None,
        increase_per_gen: Optional[int] = None,
        checkpoint_gens: Optional[int] = None,
        max_generations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Enable/adjust curriculum learning parameters for min_trades over generations."""
        try:
            fc = self.fitness_config
            changes = {}
            if enabled is not None:
                old = fc.curriculum_enabled; fc.curriculum_enabled = bool(enabled); changes['enabled'] = {"old": old, "new": bool(enabled)}
            def seti(name, val):
                if val is not None:
                    old = getattr(fc, name)
                    setattr(fc, name, int(val))
                    changes[name] = {"old": old, "new": int(val)}
            seti('curriculum_start_min_trades', start_min_trades)
            seti('curriculum_increase_per_gen', increase_per_gen)
            seti('curriculum_checkpoint_gens', checkpoint_gens)
            seti('curriculum_max_generations', max_generations)
            self.actions_log.append({"action": "configure_curriculum", "changes": changes})
            return {"success": True, "changes": changes}
        except Exception as e:
            logger.exception("configure_curriculum failed")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # CATEGORY 6: HISTORY & PRESETS (NEW)
    # ========================================================================

    async def get_generation_history(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Return generation history from agent_feed (full if last_n=None)."""
        try:
            from core.agent_feed import agent_feed
            gens = agent_feed.get_generations(last_n=last_n)
            data = [
                {
                    'generation': g.generation,
                    'best_fitness': g.best_fitness,
                    'mean_fitness': g.mean_fitness,
                    'diversity': g.diversity,
                    'best_metrics': g.best_metrics,
                    'coach_triggered': g.coach_triggered
                } for g in gens
            ]
            return {"success": True, "generations": data}
        except Exception as e:
            logger.exception("get_generation_history failed")
            return {"success": False, "error": str(e)}

    async def set_fitness_preset(self, preset: str) -> Dict[str, Any]:
        """Apply a FitnessConfig preset quickly (balanced, high_frequency, conservative, profit_focused)."""
        try:
            from core.models import FitnessConfig
            new_fc = FitnessConfig.get_preset_config(preset)
            old = self.fitness_config.preset
            # Copy fields
            for k, v in new_fc.model_dump().items():
                setattr(self.fitness_config, k, v)
            self.actions_log.append({"action": "set_fitness_preset", "old": old, "new": preset})
            return {"success": True, "preset": preset}
        except Exception as e:
            logger.exception("set_fitness_preset failed")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # CATEGORY 7: EXIT / COST CONTROLS (NEW)
    # ========================================================================

    async def set_exit_policy(
        self,
        use_fib_exits: Optional[bool] = None,
        use_stop_loss: Optional[bool] = None,
        use_traditional_tp: Optional[bool] = None,
        use_time_exit: Optional[bool] = None,
        individual_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Set exit toggles globally or for a specific individual."""
        try:
            targets = []
            if individual_id is None:
                targets = self.population.individuals
            else:
                if 0 <= individual_id < len(self.population.individuals):
                    targets = [self.population.individuals[individual_id]]
                else:
                    return {"success": False, "error": "Invalid individual_id"}
            changed = 0
            for ind in targets:
                bp = ind.backtest_params
                def setb(name, val):
                    nonlocal changed
                    if val is not None:
                        setattr(bp, name, bool(val)); changed += 1
                setb('use_fib_exits', use_fib_exits)
                setb('use_stop_loss', use_stop_loss)
                setb('use_traditional_tp', use_traditional_tp)
                setb('use_time_exit', use_time_exit)
                ind.fitness = 0.0
            self.actions_log.append({"action": "set_exit_policy", "changed": changed, "scope": "individual" if individual_id is not None else "global"})
            return {"success": True, "changed": changed}
        except Exception as e:
            logger.exception("set_exit_policy failed")
            return {"success": False, "error": str(e)}

    async def set_costs(self, fee_bp: Optional[float] = None, slippage_bp: Optional[float] = None, individual_id: Optional[int] = None) -> Dict[str, Any]:
        """Adjust transaction cost assumptions globally or per individual."""
        try:
            targets = self.population.individuals if individual_id is None else [self.population.individuals[individual_id]]
            changed = 0
            for ind in targets:
                if fee_bp is not None:
                    ind.backtest_params.fee_bp = float(fee_bp); changed += 1
                if slippage_bp is not None:
                    ind.backtest_params.slippage_bp = float(slippage_bp); changed += 1
                ind.fitness = 0.0
            self.actions_log.append({"action": "set_costs", "changed": changed})
            return {"success": True, "changed": changed}
        except Exception as e:
            logger.exception("set_costs failed")
            return {"success": False, "error": str(e)}
    
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
