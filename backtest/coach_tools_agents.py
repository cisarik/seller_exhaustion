"""
OpenAI Agents Framework Tool Definitions

Converts CoachToolkit methods to OpenAI Agents @function_tool decorators.
Uses proper OpenAI Agents patterns for tool calling.
"""

from typing import Dict, Any, List, Optional
from agents import function_tool, RunContextWrapper
from agents.tool import FunctionTool

from backtest.coach_tools import CoachToolkit
from backtest.optimizer import Population
from backtest.coach_session import CoachAnalysisSession
from core.models import FitnessConfig, OptimizationConfig
from strategy.seller_exhaustion import SellerParams
import logging

logger = logging.getLogger(__name__)


class CoachToolsAgents:
    """
    OpenAI Agents-compatible tool wrapper.

    Wraps CoachToolkit methods with @function_tool decorators
    for proper OpenAI Agents integration.
    """

    def __init__(
        self,
        population: Population,
        session: CoachAnalysisSession,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig,
        mutation_manager,
        islands_registry: Optional[Dict[int, Population]] = None,
        island_policy_reference: Optional[Dict[str, Any]] = None
    ):
        # Create the underlying toolkit
        self.toolkit = CoachToolkit(
            population=population,
            session=session,
            fitness_config=fitness_config,
            ga_config=ga_config,
            mutation_manager=mutation_manager,
            islands_registry=islands_registry,
            island_policy_reference=island_policy_reference
        )

        # Disable strict JSON schema for all tools to avoid additionalProperties issues
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr_value = getattr(self, attr_name)
                if hasattr(attr_value, 'strict_json_schema'):
                    attr_value.strict_json_schema = False

    # ========================================================================
    # CATEGORY 1: OBSERVABILITY (8 tools)
    # ========================================================================

    @function_tool
    async def analyze_population(
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
        return await self.toolkit.analyze_population(
            group_by=group_by,
            top_n=top_n,
            bottom_n=bottom_n,
            include_params=include_params
        )

    @function_tool
    async def get_correlation_matrix(
        include_params: Optional[List[str]] = None,
        correlate_with: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute Pearson correlations between parameters and selected metrics."""
        return await self.toolkit.get_correlation_matrix(
            include_params=include_params,
            correlate_with=correlate_with
        )

    @function_tool
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
        return await self.toolkit.get_param_distribution(
            parameter_name=parameter_name,
            bins=bins,
            correlate_with=correlate_with,
            show_by_fitness_quartile=show_by_fitness_quartile
        )

    @function_tool
    async def get_param_bounds(
        self,
        parameters: Optional[str] = None,
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
        return await self.toolkit.get_param_bounds(
            parameters=parameters,
            include_clustering=include_clustering
        )

    @function_tool
    async def get_generation_history(
        self,
        last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Return generation history from agent_feed (full if last_n=None)."""
        return await self.toolkit.get_generation_history(last_n=last_n)

    # ========================================================================
    # CATEGORY 2: INDIVIDUAL MANIPULATION (3 tools)
    # ========================================================================

    @function_tool
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
        return await self.toolkit.mutate_individual(
            individual_id=individual_id,
            parameter_name=parameter_name,
            new_value=new_value,
            reason=reason,
            respect_bounds=respect_bounds
        )

    @function_tool
    async def insert_llm_individual(
        self,
        destination: str,
        seller_params: Any,
        backtest_params: Any,
        island_id: Optional[int] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Insert a new individual provided by the LLM into the main population or a specific island.
        The individual's fitness is reset for evaluation in the next generation.
        """
        individual = {
            "seller_params": seller_params,
            "backtest_params": backtest_params
        }
        return await self.toolkit.insert_llm_individual(
            destination=destination,
            individual=individual,
            island_id=island_id,
            reason=reason
        )

    @function_tool
    async def create_islands(
        self,
        count: int = 2,
        strategy: str = "split"
    ) -> Dict[str, Any]:
        """Create multiple sub-populations (islands) from the current population."""
        return await self.toolkit.create_islands(count=count, strategy=strategy)

    @function_tool
    async def migrate_between_islands(
        self,
        src_island: int,
        dst_island: int,
        individual_id: int,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Migrate an individual from one island to another."""
        return await self.toolkit.migrate_between_islands(
            src_island=src_island,
            dst_island=dst_island,
            individual_id=individual_id,
            reason=reason
        )

    @function_tool
    async def configure_island_scheduler(
        self,
        migration_cadence: Optional[int] = None,
        migration_size: Optional[int] = None,
        merge_to_main_cadence: Optional[int] = None,
        merge_top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Configure island migration cadence/size and island→main merge policy."""
        return await self.toolkit.configure_island_scheduler(
            migration_cadence=migration_cadence,
            migration_size=migration_size,
            merge_to_main_cadence=merge_to_main_cadence,
            merge_top_k=merge_top_k
        )

    # ========================================================================
    # CATEGORY 3: GA / POPULATION UTILITIES
    # ========================================================================

    @function_tool
    async def inject_immigrants(
        self,
        fraction: float = 0.15,
        strategy: str = "worst_replacement"
    ) -> Dict[str, Any]:
        """Inject random immigrants into the main population to boost diversity."""
        return await self.toolkit.inject_immigrants(fraction=fraction, strategy=strategy)

    @function_tool
    async def export_population(
        self,
        path: str
    ) -> Dict[str, Any]:
        """Export current population to JSON file."""
        return await self.toolkit.export_population(path=path)

    @function_tool
    async def import_population(
        self,
        path: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Import population from JSON and replace current individuals (size preserved if limit provided)."""
        return await self.toolkit.import_population(path=path, limit=limit)

    @function_tool
    async def drop_individual(
        self,
        individual_id: int,
        replace_with: str = "immigrant"
    ) -> Dict[str, Any]:
        """Drop an individual and optionally replace with a new immigrant to keep size constant."""
        return await self.toolkit.drop_individual(individual_id=individual_id, replace_with=replace_with)

    @function_tool
    async def bulk_update_param(
        self,
        individual_ids: List[int],
        parameter_name: str,
        new_value: Any
    ) -> Dict[str, Any]:
        """Set a parameter to a new value for a group of individuals."""
        return await self.toolkit.bulk_update_param(
            individual_ids=individual_ids,
            parameter_name=parameter_name,
            new_value=new_value
        )

    @function_tool
    async def update_param_bounds(
        self,
        parameter: str,
        new_min: Optional[float] = None,
        new_max: Optional[float] = None,
        reason: str = "",
        retroactive: bool = False
    ) -> Dict[str, Any]:
        """Expand/contract parameter search bounds and optionally clamp existing individuals."""
        return await self.toolkit.update_param_bounds(
            parameter=parameter,
            new_min=new_min,
            new_max=new_max,
            reason=reason,
            retroactive=retroactive
        )

    @function_tool
    async def update_bounds_multi(
        self,
        bounds: Any,
        retroactive: bool = False
    ) -> Dict[str, Any]:
        """Update bounds for multiple parameters at once. bounds={'ema_fast': {'min':24,'max':192}, ...}"""
        return await self.toolkit.update_bounds_multi(bounds=bounds, retroactive=retroactive)

    @function_tool
    async def reseed_population(
        self,
        fraction: float = 0.2,
        strategy: str = "worst_replacement"
    ) -> Dict[str, Any]:
        """Replace a fraction of the population immediately with random newcomers (hard reseed)."""
        return await self.toolkit.reseed_population(fraction=fraction, strategy=strategy)

    @function_tool
    async def insert_individual(
        self,
        strategy: str = "coach_designed",
        parameters: Optional[Any] = None,
        clone_from_id: Optional[int] = None,
        mutations: Optional[Any] = None,
        parent_ids: Optional[List[int]] = None,
        blend_strategy: str = "average",
        reason: str = "",
        position: Optional[int] = None
    ) -> Dict[str, Any]:
        """Add a new individual using multiple strategies (coach_designed, random, clone_best, hybrid)."""
        return await self.toolkit.insert_individual(
            strategy=strategy,
            parameters=parameters,
            clone_from_id=clone_from_id,
            mutations=mutations,
            parent_ids=parent_ids,
            blend_strategy=blend_strategy,
            reason=reason,
            position=position
        )

    # ========================================================================
    # CATEGORY 4: FITNESS CONFIGURATION
    # ========================================================================

    @function_tool
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
        return await self.toolkit.update_fitness_gates(
            min_trades=min_trades,
            min_win_rate=min_win_rate,
            reason=reason
        )

    @function_tool
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
        return await self.toolkit.update_ga_params(
            mutation_probability=mutation_probability,
            mutation_rate=mutation_rate,
            sigma=sigma,
            tournament_size=tournament_size,
            elite_fraction=elite_fraction,
            immigrant_fraction=immigrant_fraction,
            reason=reason
        )

    @function_tool
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
        return await self.toolkit.update_fitness_weights(
            trade_count_weight=trade_count_weight,
            win_rate_weight=win_rate_weight,
            avg_r_weight=avg_r_weight,
            total_pnl_weight=total_pnl_weight,
            max_drawdown_penalty=max_drawdown_penalty,
            penalty_trades_strength=penalty_trades_strength,
            penalty_wr_strength=penalty_wr_strength
        )

    @function_tool
    async def set_fitness_function_type(
        self,
        fitness_function_type: str
    ) -> Dict[str, Any]:
        """Switch between 'hard_gates' and 'soft_penalties'."""
        return await self.toolkit.set_fitness_function_type(fitness_function_type=fitness_function_type)

    @function_tool
    async def configure_curriculum(
        self,
        enabled: Optional[bool] = None,
        start_min_trades: Optional[int] = None,
        increase_per_gen: Optional[int] = None,
        checkpoint_gens: Optional[int] = None,
        max_generations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Enable/adjust curriculum learning parameters for min_trades over generations."""
        return await self.toolkit.configure_curriculum(
            enabled=enabled,
            start_min_trades=start_min_trades,
            increase_per_gen=increase_per_gen,
            checkpoint_gens=checkpoint_gens,
            max_generations=max_generations
        )

    @function_tool
    async def set_fitness_preset(
        self,
        preset: str
    ) -> Dict[str, Any]:
        """Apply a FitnessConfig preset quickly (balanced, high_frequency, conservative, profit_focused)."""
        return await self.toolkit.set_fitness_preset(preset=preset)

    # ========================================================================
    # CATEGORY 5: EXIT / COST CONTROLS
    # ========================================================================

    @function_tool
    async def set_exit_policy(
        self,
        use_fib_exits: Optional[bool] = None,
        use_stop_loss: Optional[bool] = None,
        use_traditional_tp: Optional[bool] = None,
        use_time_exit: Optional[bool] = None,
        individual_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Set exit toggles globally or for a specific individual."""
        return await self.toolkit.set_exit_policy(
            use_fib_exits=use_fib_exits,
            use_stop_loss=use_stop_loss,
            use_traditional_tp=use_traditional_tp,
            use_time_exit=use_time_exit,
            individual_id=individual_id
        )

    @function_tool
    async def set_costs(
        self,
        fee_bp: Optional[float] = None,
        slippage_bp: Optional[float] = None,
        individual_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Adjust transaction cost assumptions globally or per individual."""
        return await self.toolkit.set_costs(
            fee_bp=fee_bp,
            slippage_bp=slippage_bp,
            individual_id=individual_id
        )

    # ========================================================================
    # CATEGORY 6: CONTROL FLOW (1 tool)
    # ========================================================================

    @function_tool
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
        return await self.toolkit.finish_analysis(
            summary=summary,
            overall_assessment=overall_assessment,
            stagnation_detected=stagnation_detected,
            diversity_concern=diversity_concern
        )