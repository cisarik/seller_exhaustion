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
from core.models import FitnessConfig, OptimizationConfig, AdamConfig
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
        adam_config: Optional[AdamConfig] = None,
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
            adam_config=adam_config,
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
        self,
        group_by: str = "fitness",
        top_n: int = 5,
        bottom_n: int = 3,
        include_params: bool = False,
        reason: str = ""
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
            reason: Explanation for this analysis
        """
        return await self.toolkit.analyze_population(
            group_by=group_by,
            top_n=top_n,
            bottom_n=bottom_n,
            include_params=include_params,
            reason=reason
        )

    @function_tool
    async def get_correlation_matrix(
        self,
        include_params: Optional[List[str]] = None,
        correlate_with: Optional[List[str]] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Compute Pearson correlations between parameters and selected metrics."""
        return await self.toolkit.get_correlation_matrix(
            include_params=include_params,
            correlate_with=correlate_with,
            reason=reason
        )

    @function_tool
    async def get_param_distribution(
        self,
        parameter_name: str,
        bins: int = 5,
        correlate_with: Optional[str] = None,
        show_by_fitness_quartile: bool = True,
        reason: str = ""
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
            reason: Explanation for this analysis
        """
        return await self.toolkit.get_param_distribution(
            parameter_name=parameter_name,
            bins=bins,
            correlate_with=correlate_with,
            show_by_fitness_quartile=show_by_fitness_quartile,
            reason=reason
        )

    @function_tool
    async def get_param_bounds(
        self,
        parameters: Optional[str] = None,
        include_clustering: bool = True,
        reason: str = ""
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
            reason: Explanation for this query
        """
        return await self.toolkit.get_param_bounds(
            parameters=parameters,
            include_clustering=include_clustering,
            reason=reason
        )

    @function_tool
    async def get_generation_history(
        self,
        last_n: Optional[int] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Return generation history from agent_feed (full if last_n=None)."""
        return await self.toolkit.get_generation_history(last_n=last_n, reason=reason)

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
        strategy: str = "split",
        reason: str = ""
    ) -> Dict[str, Any]:
        """Create multiple sub-populations (islands) from the current population."""
        return await self.toolkit.create_islands(count=count, strategy=strategy, reason=reason)

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
        merge_top_k: Optional[int] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Configure island migration cadence/size and island→main merge policy."""
        return await self.toolkit.configure_island_scheduler(
            migration_cadence=migration_cadence,
            migration_size=migration_size,
            merge_to_main_cadence=merge_to_main_cadence,
            merge_top_k=merge_top_k,
            reason=reason
        )

    # ========================================================================
    # CATEGORY 3: GA / POPULATION UTILITIES
    # ========================================================================

    @function_tool
    async def inject_immigrants(
        self,
        fraction: float = 0.15,
        strategy: str = "worst_replacement",
        reason: str = ""
    ) -> Dict[str, Any]:
        """Inject random immigrants into the main population to boost diversity."""
        return await self.toolkit.inject_immigrants(fraction=fraction, strategy=strategy, reason=reason)

    @function_tool
    async def export_population(
        self,
        path: str,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Export current population to JSON file."""
        return await self.toolkit.export_population(path=path, reason=reason)

    @function_tool
    async def import_population(
        self,
        path: str,
        limit: Optional[int] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Import population from JSON and replace current individuals (size preserved if limit provided)."""
        return await self.toolkit.import_population(path=path, limit=limit, reason=reason)

    @function_tool
    async def drop_individual(
        self,
        individual_id: int,
        replace_with: str = "immigrant",
        reason: str = ""
    ) -> Dict[str, Any]:
        """Drop an individual and optionally replace with a new immigrant to keep size constant."""
        return await self.toolkit.drop_individual(individual_id=individual_id, replace_with=replace_with, reason=reason)

    @function_tool
    async def bulk_update_param(
        self,
        individual_ids: List[int],
        parameter_name: str,
        new_value: Any,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Set a parameter to a new value for a group of individuals."""
        return await self.toolkit.bulk_update_param(
            individual_ids=individual_ids,
            parameter_name=parameter_name,
            new_value=new_value,
            reason=reason
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
        retroactive: bool = False,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Update bounds for multiple parameters at once. bounds={'ema_fast': {'min':24,'max':192}, ...}"""
        return await self.toolkit.update_bounds_multi(bounds=bounds, retroactive=retroactive, reason=reason)

    @function_tool
    async def reseed_population(
        self,
        fraction: float = 0.2,
        strategy: str = "worst_replacement",
        reason: str = ""
    ) -> Dict[str, Any]:
        """Replace a fraction of the population immediately with random newcomers (hard reseed)."""
        return await self.toolkit.reseed_population(fraction=fraction, strategy=strategy, reason=reason)

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
        penalty_wr_strength: Optional[float] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Adjust fitness weights and penalty strengths; renormalize weights to sum≈1.0."""
        return await self.toolkit.update_fitness_weights(
            trade_count_weight=trade_count_weight,
            win_rate_weight=win_rate_weight,
            avg_r_weight=avg_r_weight,
            total_pnl_weight=total_pnl_weight,
            max_drawdown_penalty=max_drawdown_penalty,
            penalty_trades_strength=penalty_trades_strength,
            penalty_wr_strength=penalty_wr_strength,
            reason=reason
        )

    @function_tool
    async def set_fitness_function_type(
        self,
        fitness_function_type: str,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Switch between 'hard_gates' and 'soft_penalties'."""
        return await self.toolkit.set_fitness_function_type(fitness_function_type=fitness_function_type, reason=reason)

    @function_tool
    async def configure_curriculum(
        self,
        enabled: Optional[bool] = None,
        start_min_trades: Optional[int] = None,
        increase_per_gen: Optional[int] = None,
        checkpoint_gens: Optional[int] = None,
        max_generations: Optional[int] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Enable/adjust curriculum learning parameters for min_trades over generations."""
        return await self.toolkit.configure_curriculum(
            enabled=enabled,
            start_min_trades=start_min_trades,
            increase_per_gen=increase_per_gen,
            checkpoint_gens=checkpoint_gens,
            max_generations=max_generations,
            reason=reason
        )

    @function_tool
    async def set_fitness_preset(
        self,
        preset: str,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Apply a FitnessConfig preset quickly (balanced, high_frequency, conservative, profit_focused)."""
        return await self.toolkit.set_fitness_preset(preset=preset, reason=reason)

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
        individual_id: Optional[int] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Set exit toggles globally or for a specific individual."""
        return await self.toolkit.set_exit_policy(
            use_fib_exits=use_fib_exits,
            use_stop_loss=use_stop_loss,
            use_traditional_tp=use_traditional_tp,
            use_time_exit=use_time_exit,
            individual_id=individual_id,
            reason=reason
        )

    @function_tool
    async def set_costs(
        self,
        fee_bp: Optional[float] = None,
        slippage_bp: Optional[float] = None,
        individual_id: Optional[int] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Adjust transaction cost assumptions globally or per individual."""
        return await self.toolkit.set_costs(
            fee_bp=fee_bp,
            slippage_bp=slippage_bp,
            individual_id=individual_id,
            reason=reason
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

    @function_tool
    async def validate_population_health(
        self,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Validate population health and identify potential issues.
        
        Use this tool to:
        - Check for population health issues
        - Identify bounds violations
        - Detect convergence problems
        - Validate fitness distribution
        
        Args:
            reason: Explanation for this validation
        """
        return await self.toolkit.validate_population_health(reason=reason)

    # ============================================================================
    # OPTIMIZER CONTROL TOOLS
    # ============================================================================

    @function_tool
    async def set_active_optimizer(
        self,
        optimizer_type: str,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Switch between optimization algorithms: 'GA' (Genetic Algorithm) or 'ADAM' (Gradient-based).
        
        This is a powerful tool that allows the coach to dynamically change the optimization strategy:
        
        **GA (Genetic Algorithm)**:
        - Population-based evolutionary approach
        - Good for exploration and escaping local optima
        - Uses crossover, mutation, and selection
        - Better for complex, multi-modal fitness landscapes
        - Slower convergence but more thorough search
        
        **ADAM (Adaptive Moment Estimation)**:
        - Gradient-based optimization
        - Faster convergence to local optima
        - Good for smooth, continuous parameter spaces
        - More efficient for fine-tuning near good solutions
        - Can get stuck in local optima
        
        **When to use each**:
        - Start with GA for broad exploration
        - Switch to ADAM when population converges to a promising region
        - Use GA again if ADAM gets stuck in local optima
        - ADAM is excellent for final fine-tuning
        
        Args:
            optimizer_type: Either 'GA' or 'ADAM'
            reason: Explanation for switching optimizers
        """
        return await self.toolkit.set_active_optimizer(
            optimizer_type=optimizer_type,
            reason=reason
        )

    @function_tool
    async def configure_ga_parameters(
        self,
        population_size: Optional[int] = None,
        mutation_rate: Optional[float] = None,
        mutation_sigma: Optional[float] = None,
        elite_fraction: Optional[float] = None,
        tournament_size: Optional[int] = None,
        mutation_probability: Optional[float] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Configure Genetic Algorithm (GA) parameters for evolutionary optimization.
        
        **Population Size** (population_size):
        - Number of individuals in each generation
        - Larger = more diversity, slower computation
        - Smaller = faster, less diversity
        - Recommended: 20-50 for most problems
        
        **Mutation Rate** (mutation_rate):
        - Probability that an individual will be mutated
        - Higher = more exploration, less exploitation
        - Lower = more exploitation, less exploration
        - Recommended: 0.1-0.5
        
        **Mutation Sigma** (mutation_sigma):
        - Standard deviation for Gaussian mutation
        - Higher = larger parameter changes
        - Lower = smaller, fine-tuning changes
        - Recommended: 0.05-0.2
        
        **Elite Fraction** (elite_fraction):
        - Fraction of best individuals preserved each generation
        - Higher = more exploitation, slower convergence
        - Lower = more exploration, faster convergence
        - Recommended: 0.1-0.3
        
        **Tournament Size** (tournament_size):
        - Number of individuals competing in selection
        - Higher = more selective pressure
        - Lower = less selective pressure
        - Recommended: 3-7
        
        **Mutation Probability** (mutation_probability):
        - Probability of mutating each parameter
        - Higher = more parameter changes per mutation
        - Lower = fewer parameter changes
        - Recommended: 0.5-1.0
        
        Args:
            population_size: Number of individuals (20-100)
            mutation_rate: Mutation probability (0.0-1.0)
            mutation_sigma: Mutation strength (0.01-0.5)
            elite_fraction: Elite preservation (0.0-0.5)
            tournament_size: Selection pressure (2-10)
            mutation_probability: Parameter mutation rate (0.0-1.0)
            reason: Explanation for parameter changes
        """
        return await self.toolkit.configure_ga_parameters(
            population_size=population_size,
            mutation_rate=mutation_rate,
            mutation_sigma=mutation_sigma,
            elite_fraction=elite_fraction,
            tournament_size=tournament_size,
            mutation_probability=mutation_probability,
            reason=reason
        )

    @function_tool
    async def configure_adam_parameters(
        self,
        learning_rate: Optional[float] = None,
        epsilon: Optional[float] = None,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        max_perturbations: Optional[int] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Configure ADAM (Adaptive Moment Estimation) optimizer parameters.
        
        **Learning Rate** (learning_rate):
        - Step size for parameter updates
        - Higher = faster convergence, risk of overshooting
        - Lower = slower convergence, more stable
        - Recommended: 0.001-0.1
        
        **Epsilon** (epsilon):
        - Small constant for numerical stability
        - Prevents division by zero in gradient calculations
        - Higher = more stable, less sensitive
        - Lower = more sensitive, potential instability
        - Recommended: 1e-8 to 1e-4
        
        **Beta1** (beta1):
        - Exponential decay rate for first moment estimates
        - Controls momentum of gradient updates
        - Higher = more momentum, smoother updates
        - Lower = less momentum, more responsive
        - Recommended: 0.9-0.99
        
        **Beta2** (beta2):
        - Exponential decay rate for second moment estimates
        - Controls adaptive learning rate
        - Higher = more stable learning rate adaptation
        - Lower = more adaptive learning rate
        - Recommended: 0.999-0.9999
        
        **Max Perturbations** (max_perturbations):
        - Maximum number of parameter perturbations per generation
        - Higher = more thorough gradient estimation
        - Lower = faster computation, less accurate gradients
        - Recommended: 5-20
        
        Args:
            learning_rate: Step size (0.001-0.1)
            epsilon: Numerical stability constant (1e-8 to 1e-4)
            beta1: First moment decay (0.9-0.99)
            beta2: Second moment decay (0.999-0.9999)
            max_perturbations: Gradient estimation samples (5-50)
            reason: Explanation for parameter changes
        """
        return await self.toolkit.configure_adam_parameters(
            learning_rate=learning_rate,
            epsilon=epsilon,
            beta1=beta1,
            beta2=beta2,
            max_perturbations=max_perturbations,
            reason=reason
        )