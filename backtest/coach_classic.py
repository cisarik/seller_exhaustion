"""
Classic Coach Manager - Deterministic optimization scheduler.

This coach provides deterministic, rule-based optimization guidance without
requiring any LLM calls or internet connectivity. It implements phase-based
optimization scheduling with intelligent GA/ADAM switching based on population
state and fitness metrics.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from core.logging_utils import get_logger

try:
    from app.signals import get_coach_signals
    SIGNALS_AVAILABLE = True
except ImportError:
    SIGNALS_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class OptimizationPhase:
    """Represents an optimization phase with specific algorithm and parameters."""
    name: str
    algorithm: str  # 'ga' or 'adam'
    start_generation: int
    end_generation: int
    ga_params: Dict[str, Any] = None
    adam_params: Dict[str, Any] = None
    description: str = ""

    def is_active(self, generation: int) -> bool:
        """Check if this phase is active for the given generation."""
        return self.start_generation <= generation <= self.end_generation


@dataclass
class PopulationState:
    """Current state of the population for decision making."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    diversity: float
    population_size: int
    fitness_trend: str  # 'improving', 'plateau', 'declining'


class ClassicCoachManager:
    """
    Deterministic coach that schedules optimization phases and makes algorithm switching decisions.

    This coach analyzes population state and applies rule-based logic to:
    1. Schedule optimization phases (exploration, exploitation, refinement)
    2. Switch between GA and ADAM algorithms based on population state
    3. Adjust GA hyperparameters dynamically
    4. Provide deterministic recommendations without LLM calls
    """

    def __init__(self, analysis_interval: int = 5, verbose: bool = True, total_iterations: int = 200, **coach_params):
        """
        Initialize the Classic Coach Manager.

        Args:
            analysis_interval: How often to analyze (every N generations)
            verbose: Whether to log detailed analysis
            total_iterations: Total optimization iterations from settings (OPTIMIZER_ITERATIONS)
            **coach_params: Additional coach configuration parameters (similar to AI agent)
        """
        self.analysis_interval = analysis_interval
        self.verbose = verbose
        self.total_iterations = total_iterations

        # Calculate dynamic phase boundaries based on total_iterations
        self._calculate_phase_boundaries()

        # Store coach configuration parameters (similar to AI agent capabilities)
        self.coach_params = coach_params or {}

        # Extract key configuration options (matching AI agent capabilities)
        self.enable_islands = coach_params.get('enable_islands', True)
        self.enable_fitness_tuning = coach_params.get('enable_fitness_tuning', True)
        self.enable_individual_mutation = coach_params.get('enable_individual_mutation', False)  # Classic coach doesn't mutate individuals
        self.enable_bounds_expansion = coach_params.get('enable_bounds_expansion', True)
        self.enable_immigration = coach_params.get('enable_immigration', True)
        self.enable_algorithm_switching = coach_params.get('enable_algorithm_switching', True)

        # Phase management
        self.current_phase: Optional[OptimizationPhase] = None
        self.phases: List[OptimizationPhase] = []
        self.last_analysis_generation = 0
        self.recommendations_history: List[Dict[str, Any]] = []

        # Enhanced state tracking for advanced strategies
        self.consecutive_low_diversity = 0
        self.consecutive_stagnation = 0
        self.last_best_fitness = 0.0
        self.fitness_improvement_threshold = 0.01  # Minimum improvement to not be considered stagnation

        # Historical learning and trend analysis
        self.fitness_history: List[float] = []  # Last 20 fitness values for trend analysis
        self.recommendation_success_rate: Dict[str, float] = {}  # Track success of different strategies
        self.parameter_importance: Dict[str, float] = {}  # Track which parameters correlate with fitness
        self.phase_performance_history: List[Dict[str, Any]] = []  # Track phase effectiveness

        # Adaptive phase management
        self.actual_phase_transitions: Dict[str, int] = {}  # When phases actually ended
        self.convergence_metrics: Dict[str, float] = {}  # Track convergence speed

        # Confidence and prediction
        self.recommendation_confidence = 0.5  # 0-1 confidence in current recommendation
        self.estimated_remaining_generations = None  # Prediction for optimization completion

        # UI integration
        self.classic_coach_window = None  # Reference to Classic Coach window for real-time updates

    def _calculate_phase_boundaries(self):
        """Calculate dynamic phase boundaries based on total optimization iterations."""
        total = self.total_iterations

        # Scale phases proportionally to total iterations
        # Original ratios: Exploration=20, Exploitation=30, Refinement=rest
        # But ensure minimum phase lengths
        exploration_end = max(20, int(total * 0.25))  # At least 20 generations
        exploitation_end = max(exploration_end + 30, int(total * 0.625))  # At least 30 generations

        self.phase_boundaries = {
            'exploration_end': exploration_end,
            'exploitation_end': exploitation_end,
            'refinement_start': exploitation_end + 1
        }

        if self.verbose:
            logger.info(f"Classic Coach phases calculated for {total} iterations:")
            logger.info(f"  Exploration: 1-{exploration_end} generations")
            logger.info(f"  Exploitation: {exploration_end+1}-{exploitation_end} generations")
            logger.info(f"  Refinement: {exploitation_end+1}+ generations")

    def should_analyze(self, generation: int) -> bool:
        """Check if coach should analyze at this generation."""
        return generation >= self.last_analysis_generation + self.analysis_interval

    def update_analysis_interval(self, new_interval: int):
        """
        Update the analysis interval dynamically.

        This allows the coach to adapt its behavior based on the new interval:
        - Shorter intervals: More conservative changes, focus on stability
        - Longer intervals: More aggressive changes, focus on exploration
        """
        old_interval = self.analysis_interval
        self.analysis_interval = max(1, new_interval)  # Minimum 1 generation

        # Adapt coach behavior based on interval change
        if new_interval < old_interval:
            # Shorter interval - be more conservative
            self.fitness_improvement_threshold = max(0.005, self.fitness_improvement_threshold * 0.8)
            if self.verbose:
                logger.info(f"🤖 Classic Coach: Analysis interval shortened ({old_interval}→{new_interval})")
                logger.info(f"   → More conservative: fitness threshold = {self.fitness_improvement_threshold:.4f}")
        elif new_interval > old_interval:
            # Longer interval - be more aggressive
            self.fitness_improvement_threshold = min(0.05, self.fitness_improvement_threshold * 1.2)
            if self.verbose:
                logger.info(f"🤖 Classic Coach: Analysis interval lengthened ({old_interval}→{new_interval})")
                logger.info(f"   → More aggressive: fitness threshold = {self.fitness_improvement_threshold:.4f}")

        # Reset analysis generation counter to avoid immediate analysis
        # Use current generation from last analysis or 0 if none
        current_gen = getattr(self, 'last_analysis_generation', 0)
        self.last_analysis_generation = max(0, current_gen - new_interval)

    def record_generation(self, population) -> None:
        """Record generation data for analysis (no-op for classic coach)."""
        pass

    def add_log(self, generation: int, message: str) -> None:
        """Add log message (no-op for classic coach)."""
        pass

    async def analyze_and_apply_with_openai_agent(
        self,
        population,
        fitness_config,
        ga_config,
        coach_window=None,
        status_callback=None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Main analysis method - implements deterministic optimization scheduling.

        This replaces the LLM-based analysis with rule-based logic that:
        1. Analyzes current population state
        2. Determines optimal algorithm and parameters
        3. Applies changes directly to ga_config

        Args:
            population: Current population object
            fitness_config: Current fitness configuration
            ga_config: GA configuration to modify
            coach_window: Optional coach window for UI updates
            status_callback: Optional callback for status bar updates

        Returns:
            Tuple of (success, summary_dict)
        """
        try:
            generation = getattr(population, 'generation', 0)
            
            # Store status_callback for use in this analysis
            self.status_callback = status_callback

            # Get population state
            pop_state = self._analyze_population_state(population, generation)

            # Determine optimal phase and algorithm
            recommended_algorithm, recommended_params = self._get_optimal_algorithm(pop_state)

            # Apply recommendations
            actions_taken = self._apply_recommendations(ga_config, recommended_algorithm, recommended_params, pop_state)

            # Update status bar with key decisions
            print(f"🔍 DEBUG: coach_classic.py self.status_callback = {self.status_callback}")
            if self.status_callback:
                try:
                    phase_info = self.get_phase_info(generation)
                    phase_name = phase_info.name if phase_info else "Unknown"

                    # Create informative status message
                    if recommended_algorithm == 'adam':
                        status_msg = f"🤖 Classic Coach: Switched to ADAM mode (Phase: {phase_name}, Gen {generation})"
                    else:
                        status_msg = f"🤖 Classic Coach: Using GA mode (Phase: {phase_name}, Gen {generation})"

                    # Add confidence and actions info
                    confidence_indicator = "🎯" if self.recommendation_confidence > 0.7 else "🤔" if self.recommendation_confidence > 0.5 else "❓"
                    status_msg += f" {confidence_indicator} {len(actions_taken)} actions"

                    print(f"🔍 DEBUG: coach_classic.py calling status_callback with 'ClassicCoach_Analysis', '{status_msg}'")
                    self.status_callback("ClassicCoach_Analysis", status_msg)
                except Exception as e:
                    logger.warning(f"Error updating status bar: {e}")

            # Log analysis
            if self.verbose:
                logger.info(f"🤖 Classic Coach Analysis (Gen {generation}):")
                logger.info(f"   Population State: fitness={pop_state.best_fitness:.4f}, diversity={pop_state.diversity:.2f}")
                logger.info(f"   Recommendation: {recommended_algorithm.upper()} algorithm")
                logger.info(f"   Actions Taken: {len(actions_taken)} parameter adjustments")

            # Enhanced coach window updates with more information
            if coach_window:
                try:
                    # Use the new Classic coach specific methods
                    phase_info = self.get_phase_info(generation)
                    phase_name = phase_info.name if phase_info else "Unknown"

                    # Set status with enhanced Classic coach information
                    coach_window.set_classic_coach_status(
                        status=f"{recommended_algorithm.upper()} recommended",
                        phase=phase_name,
                        confidence=self.recommendation_confidence,
                        actions=len(actions_taken)
                    )

                    # Set analysis info with structured Classic coach data
                    coach_window.set_classic_coach_analysis_info(
                        generation=generation,
                        phase=phase_name,
                        actions=len(actions_taken),
                        confidence=self.recommendation_confidence,
                        remaining_gens=self.estimated_remaining_generations
                    )

                    # Enhanced tool call with more context
                    tool_call_params = {
                        'generation': generation,
                        'population_state': {
                            'best_fitness': pop_state.best_fitness,
                            'diversity': pop_state.diversity,
                            'population_size': pop_state.population_size,
                            'fitness_trend': pop_state.fitness_trend
                        },
                        'recommended_algorithm': recommended_algorithm,
                        'recommended_params': recommended_params,
                        'actions_taken': actions_taken,
                        'coach_state': {
                            'consecutive_stagnation': self.consecutive_stagnation,
                            'consecutive_low_diversity': self.consecutive_low_diversity,
                            'recommendation_confidence': self.recommendation_confidence,
                            'estimated_remaining_generations': self.estimated_remaining_generations
                        }
                    }

                    coach_window.add_tool_call(
                        tool_name="ClassicCoach_Analysis",
                        parameters=tool_call_params,
                        response={
                            'success': True,
                            'actions_count': len(actions_taken),
                            'confidence': self.recommendation_confidence,
                            'estimated_completion': self.estimated_remaining_generations
                        },
                        reason=f"Enhanced phase-based optimization analysis with historical learning"
                    )

                    # Set request/response text for Classic coach
                    request_text = f"Classic Coach Analysis Request (Generation {generation}):\n"
                    request_text += f"- Phase: {phase_name}\n"
                    request_text += f"- Population: size={pop_state.population_size}, diversity={pop_state.diversity:.2f}\n"
                    request_text += f"- Fitness: best={pop_state.best_fitness:.4f}, trend={pop_state.fitness_trend}\n"
                    request_text += f"- Coach State: stagnation={self.consecutive_stagnation}, confidence={self.recommendation_confidence:.2f}"

                    response_text = f"Classic Coach Analysis Response:\n"
                    response_text += f"- Recommended Algorithm: {recommended_algorithm.upper()}\n"
                    response_text += f"- Actions Taken: {len(actions_taken)}\n"
                    for i, action in enumerate(actions_taken[:5], 1):  # Show first 5 actions
                        response_text += f"  {i}. {action}\n"
                    if len(actions_taken) > 5:
                        response_text += f"  ... and {len(actions_taken) - 5} more actions\n"
                    response_text += f"- Confidence: {self.recommendation_confidence:.2f}\n"
                    if self.estimated_remaining_generations:
                        response_text += f"- Estimated Completion: {self.estimated_remaining_generations} generations"

                    coach_window.set_agent_request(request_text)
                    coach_window.set_agent_response(response_text)

                except Exception as e:
                    logger.warning(f"Error updating coach window: {e}")

            # Also update the classic_coach_window reference if it exists
            if self.classic_coach_window:
                try:
                    # Trigger update in Classic Coach window using Qt thread-safe method
                    from PySide6.QtCore import QMetaObject, Qt
                    QMetaObject.invokeMethod(
                        self.classic_coach_window,
                        "update_display",
                        Qt.QueuedConnection
                    )
                except Exception as e:
                    logger.warning(f"Error triggering classic coach window update: {e}")

            # Enhanced state tracking with trend analysis
            fitness_improvement = pop_state.best_fitness - self.last_best_fitness
            if abs(fitness_improvement) < self.fitness_improvement_threshold:
                self.consecutive_stagnation += 1
            else:
                self.consecutive_stagnation = 0

            if pop_state.diversity < 0.15:
                self.consecutive_low_diversity += 1
            else:
                self.consecutive_low_diversity = 0

            self.last_best_fitness = pop_state.best_fitness

            # Estimate remaining generations based on convergence rate
            self._estimate_remaining_generations(pop_state)

            # Store recommendation for history
            recommendation = {
                'generation': generation,
                'population_state': pop_state.__dict__,
                'recommended_algorithm': recommended_algorithm,
                'recommended_params': recommended_params,
                'actions_taken': actions_taken,
                'consecutive_stagnation': self.consecutive_stagnation,
                'consecutive_low_diversity': self.consecutive_low_diversity
            }
            self.recommendations_history.append(recommendation)

            # Update last analysis generation
            self.last_analysis_generation = generation

            # Emit signals for real-time window updates
            if SIGNALS_AVAILABLE:
                try:
                    signals = get_coach_signals()
                    
                    # Emit population state update with full fitness history
                    signals.population_state_updated.emit({
                        'generation': pop_state.generation,
                        'population_size': pop_state.population_size,
                        'best_fitness': pop_state.best_fitness,
                        'mean_fitness': pop_state.mean_fitness,
                        'diversity': pop_state.diversity,
                        'fitness_trend': pop_state.fitness_trend,
                        'fitness_history': self.fitness_history[-10:] if self.fitness_history else []
                    })
                    
                    # Emit recommendation update
                    signals.recommendation_updated.emit({
                        'recommended_algorithm': recommended_algorithm,
                        'confidence': self.recommendation_confidence,
                        'generation': generation,
                        'actions': len(actions_taken)
                    })
                    
                    # Emit detailed decision reasoning
                    reasoning_lines = [
                        f"🎯 Algorithm Selection: {recommended_algorithm.upper()}",
                        f"📊 Population Analysis:",
                        f"  • Best Fitness: {pop_state.best_fitness:.4f}",
                        f"  • Mean Fitness: {pop_state.mean_fitness:.4f}",
                        f"  • Diversity: {pop_state.diversity:.3f}",
                        f"  • Population Size: {pop_state.population_size}",
                        f"🔍 Coach State:",
                        f"  • Stagnation Streak: {self.consecutive_stagnation}",
                        f"  • Low Diversity Streak: {self.consecutive_low_diversity}",
                        f"  • Confidence Level: {self.recommendation_confidence:.1%}"
                    ]
                    
                    # Build parameters adjusted dict if any
                    params_adjusted = {}
                    if recommended_algorithm == 'adam':
                        params_adjusted['algorithm_switch'] = ('GA', 'ADAM')
                    
                    signals.decision_reasoning_updated.emit({
                        'reasoning': '\n'.join(reasoning_lines),
                        'parameters_adjusted': params_adjusted
                    })
                    
                    # Emit phase information
                    phase_info = self.get_phase_info(generation)
                    phase_name = phase_info.name if phase_info else "Unknown"
                    phase_start = phase_info.start_generation if phase_info else 0
                    phase_end = phase_info.end_generation if phase_info else self.total_iterations
                    
                    signals.phase_info_updated.emit({
                        'current_phase': phase_name,
                        'phase_start': phase_start,
                        'phase_end': phase_end,
                        'exploration_end': self.phase_boundaries.get('exploration_end', '—'),
                        'exploitation_end': self.phase_boundaries.get('exploitation_end', '—'),
                        'estimated_remaining': self.estimated_remaining_generations or '—',
                        'next_transition': phase_end + 1 if phase_info else '—'
                    })
                    
                    # Emit crisis detection if applicable
                    active_crises = []
                    if self.consecutive_stagnation > 3:
                        active_crises.append({
                            'type': 'STAGNATION',
                            'description': f'{self.consecutive_stagnation} generations without improvement',
                            'response': 'Increasing mutation rate',
                            'outcome': 'in_progress'
                        })
                    if pop_state.diversity < 0.1:
                        active_crises.append({
                            'type': 'LOW_DIVERSITY',
                            'description': f'Diversity: {pop_state.diversity:.3f} (below 0.1 threshold)',
                            'response': 'Applying immigration strategy',
                            'outcome': 'in_progress'
                        })
                    
                    if active_crises:
                        signals.crisis_detection_updated.emit({
                            'generation': generation,
                            'active_crises': active_crises
                        })
                    
                    # Emit learning insights
                    insights_lines = [
                        f"📚 Learning Insights (Gen {generation}):",
                        f"• Strategy Success: {len([r for r in self.recommendations_history if r['recommended_algorithm'] == 'ga'])}/{len(self.recommendations_history)} GA recommendations",
                        f"• Average Confidence: {sum(r.get('recommendation_confidence', 0.5) for r in self.recommendations_history) / max(len(self.recommendations_history), 1):.2f}",
                        f"• Best Ever Fitness: {self.last_best_fitness:.4f}",
                        f"• Phase: {phase_name}"
                    ]
                    
                    signals.learning_insights_updated.emit({
                        'insights': '\n'.join(insights_lines)
                    })
                    
                    # Emit coach message in blue
                    signals.coach_message.emit(
                        f"✓ 🧠 Classic Coach Analysis (Gen {generation}): {recommended_algorithm.upper()} | {len(actions_taken)} actions | confidence={self.recommendation_confidence:.2f}",
                        "blue"
                    )
                except Exception as e:
                    logger.debug(f"Signals emission skipped: {e}")

            # Return success and summary
            summary = {
                'total_actions': len(actions_taken),
                'recommended_algorithm': recommended_algorithm,
                'population_state': pop_state.__dict__,
                'actions_taken': actions_taken
            }

            return True, summary

        except Exception as e:
            logger.exception(f"Classic coach analysis failed: {e}")
            return False, {'error': str(e)}

    def _analyze_population_state(self, population, generation: int) -> PopulationState:
        """Analyze current population state for decision making with enhanced metrics."""
        try:
            # Extract basic metrics
            individuals = getattr(population, 'individuals', [])
            if not individuals:
                # Return default state if no individuals
                return PopulationState(
                    generation=generation,
                    best_fitness=0.0,
                    mean_fitness=0.0,
                    std_fitness=0.0,
                    min_fitness=0.0,
                    max_fitness=0.0,
                    diversity=0.5,
                    population_size=0,
                    fitness_trend='unknown'
                )

            # Calculate fitness statistics
            fitnesses = [ind.fitness for ind in individuals if hasattr(ind, 'fitness')]
            if not fitnesses:
                fitnesses = [0.0]

            best_fitness = max(fitnesses)
            mean_fitness = sum(fitnesses) / len(fitnesses)
            min_fitness = min(fitnesses)
            max_fitness = best_fitness

            # Calculate standard deviation
            if len(fitnesses) > 1:
                variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
                std_fitness = variance ** 0.5
            else:
                std_fitness = 0.0

            # Calculate diversity (simplified metric)
            diversity = std_fitness / max(abs(mean_fitness) + 1e-6, 0.1)  # Normalize by mean fitness
            diversity = min(diversity, 1.0)  # Cap at 1.0

            # Enhanced fitness trend analysis using historical data
            fitness_trend = self._analyze_fitness_trend(best_fitness)

            # Update historical learning
            self._update_historical_learning(best_fitness, diversity, generation)

            return PopulationState(
                generation=generation,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                std_fitness=std_fitness,
                min_fitness=min_fitness,
                max_fitness=max_fitness,
                diversity=diversity,
                population_size=len(individuals),
                fitness_trend=fitness_trend
            )

        except Exception as e:
            logger.warning(f"Error analyzing population state: {e}")
            # Return safe default state
            return PopulationState(
                generation=generation,
                best_fitness=0.0,
                mean_fitness=0.0,
                std_fitness=0.0,
                min_fitness=0.0,
                max_fitness=0.0,
                diversity=0.5,
                population_size=0,
                fitness_trend='unknown'
            )

    def _analyze_fitness_trend(self, current_fitness: float) -> str:
        """Analyze fitness trend using historical data."""
        # Add current fitness to history
        self.fitness_history.append(current_fitness)

        # Keep only last 20 generations
        if len(self.fitness_history) > 20:
            self.fitness_history.pop(0)

        if len(self.fitness_history) < 3:
            return 'unknown'

        # Calculate trend using linear regression slope
        n = len(self.fitness_history)
        x = list(range(n))
        y = self.fitness_history

        # Simple slope calculation
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Classify trend
        if slope > 0.001:
            return 'improving'
        elif slope < -0.001:
            return 'declining'
        else:
            return 'plateau'

    def _update_historical_learning(self, best_fitness: float, diversity: float, generation: int):
        """Update historical learning data for better future decisions."""
        # Track fitness improvement
        if self.last_best_fitness > 0:
            improvement = best_fitness - self.last_best_fitness
            if improvement > 0.001:  # Significant improvement
                # Track what led to improvement (could be enhanced with more context)
                pass

        # Track convergence metrics
        if diversity < 0.2 and generation > 10:
            self.convergence_metrics['early_convergence_generation'] = generation
            self.convergence_metrics['convergence_fitness'] = best_fitness

        # Track phase performance
        current_phase = self.get_phase_info(generation)
        if current_phase:
            phase_key = f"{current_phase.name}_{generation}"
            self.phase_performance_history.append({
                'phase': current_phase.name,
                'generation': generation,
                'fitness': best_fitness,
                'diversity': diversity
            })

    def _get_optimal_algorithm(self, pop_state: PopulationState) -> Tuple[str, Dict[str, Any]]:
        """
        Determine the optimal algorithm and parameters based on population state.

        Enhanced decision matrix with historical learning and adaptive strategies:
        - Historical trend analysis for better predictions
        - Success rate tracking of different strategies
        - Adaptive phase transitions based on actual performance
        - Confidence scoring for recommendations
        """

        # Update recommendation confidence based on historical success
        self._update_recommendation_confidence(pop_state)

        # PHASE-BASED STRATEGY with adaptive transitions
        exploration_end = self.phase_boundaries['exploration_end']
        exploitation_end = self.phase_boundaries['exploitation_end']

        # Adaptive phase transitions based on actual performance
        actual_exploration_end = self.actual_phase_transitions.get('exploration_end', exploration_end)
        actual_exploitation_end = self.actual_phase_transitions.get('exploitation_end', exploitation_end)

        # Check for early phase transitions based on convergence
        if pop_state.generation <= exploration_end:
            # Check if we've actually converged early
            if self._should_transition_to_exploitation(pop_state):
                actual_exploration_end = pop_state.generation
                self.actual_phase_transitions['exploration_end'] = actual_exploration_end

        if pop_state.generation <= actual_exploitation_end and pop_state.generation > actual_exploration_end:
            # Check if we've actually converged to refinement phase
            if self._should_transition_to_refinement(pop_state):
                actual_exploitation_end = pop_state.generation
                self.actual_phase_transitions['exploitation_end'] = actual_exploitation_end

        # CRISIS DETECTION with enhanced logic
        crisis_recommendation = self._detect_crisis_situations(pop_state)
        if crisis_recommendation:
            self.recommendation_confidence = 0.9  # High confidence in crisis response
            return crisis_recommendation

        # PHASE-BASED DECISIONS with historical learning
        if pop_state.generation <= actual_exploration_end:
            return self._get_exploration_strategy(pop_state)
        elif pop_state.generation <= actual_exploitation_end:
            return self._get_exploitation_strategy(pop_state)
        else:
            return self._get_refinement_strategy(pop_state)

    def _update_recommendation_confidence(self, pop_state: PopulationState):
        """Update confidence in recommendations based on historical performance."""
        # Simple confidence model: higher confidence with more data
        history_size = len(self.recommendations_history)
        if history_size > 5:
            self.recommendation_confidence = min(0.8, 0.5 + (history_size - 5) * 0.05)
        else:
            self.recommendation_confidence = 0.5

        # Reduce confidence if we're in a crisis situation
        if pop_state.diversity < 0.1 or self.consecutive_stagnation > 3:
            self.recommendation_confidence *= 0.8

    def _should_transition_to_exploitation(self, pop_state: PopulationState) -> bool:
        """Determine if we should transition from exploration to exploitation early."""
        # Transition if we have good fitness and reasonable diversity
        return (pop_state.best_fitness >= 0.3 and
                pop_state.diversity >= 0.2 and
                pop_state.generation >= 15)

    def _should_transition_to_refinement(self, pop_state: PopulationState) -> bool:
        """Determine if we should transition from exploitation to refinement early."""
        # Transition if we have very good fitness and low diversity (converged)
        return (pop_state.best_fitness >= 0.5 and
                pop_state.diversity <= 0.25 and
                pop_state.generation >= 30)

    def _detect_crisis_situations(self, pop_state: PopulationState) -> Tuple[str, Dict[str, Any]] | None:
        """Detect crisis situations that require immediate intervention."""
        # GATE CRISIS: Very low diversity + poor fitness + established population
        if (pop_state.diversity < 0.1 and
            pop_state.best_fitness < 0.2 and
            pop_state.generation > 10):
            return 'ga', {
                'mutation_rate': 0.7,        # Emergency exploration
                'mutation_probability': 0.95,
                'sigma': 0.2,
                'elite_fraction': 0.05,
                'tournament_size': 2,
                'immigrant_fraction': 0.25
            }

        # STAGNATION CRISIS: No improvement for many generations
        if (self.consecutive_stagnation >= 5 and
            pop_state.best_fitness < 0.5):
            return 'ga', {
                'mutation_rate': 0.75,       # Maximum exploration
                'mutation_probability': 0.95,
                'sigma': 0.22,
                'elite_fraction': 0.03,
                'tournament_size': 2,
                'immigrant_fraction': 0.3
            }

        # PREMATURE CONVERGENCE: Low diversity early in optimization
        if (pop_state.diversity < 0.15 and
            pop_state.generation > 15 and
            pop_state.generation <= self.phase_boundaries['exploration_end']):
            return 'ga', {
                'mutation_rate': 0.65,
                'mutation_probability': 0.9,
                'sigma': 0.18,
                'elite_fraction': 0.08,
                'tournament_size': 3,
                'immigrant_fraction': 0.25
            }

        return None

    def _get_exploration_strategy(self, pop_state: PopulationState) -> Tuple[str, Dict[str, Any]]:
        """Get optimal strategy for exploration phase."""
        # Base exploration parameters
        base_params = {
            'mutation_rate': 0.5,
            'mutation_probability': 0.85,
            'sigma': 0.15,
            'elite_fraction': 0.12,
            'tournament_size': 3,
            'immigrant_fraction': 0.18
        }

        # Adapt based on current state
        if pop_state.diversity < 0.2:
            # Need more diversity - increase exploration
            base_params.update({
                'mutation_rate': 0.6,
                'immigrant_fraction': 0.25
            })
        elif pop_state.diversity > 0.4:
            # Too much diversity - focus a bit more
            base_params.update({
                'mutation_rate': 0.4,
                'elite_fraction': 0.18
            })

        return 'ga', base_params

    def _get_exploitation_strategy(self, pop_state: PopulationState) -> Tuple[str, Dict[str, Any]]:
        """Get optimal strategy for exploitation phase."""
        # Check if ADAM would be beneficial
        if (pop_state.best_fitness >= 0.4 and
            pop_state.diversity <= 0.3 and
            pop_state.fitness_trend in ['plateau', 'improving']):
            # ADAM for fine-tuning in promising region
            return 'adam', {
                'learning_rate': 0.02,
                'max_perturbations': 15,
                'epsilon': 1e-5,
                'beta1': 0.92,
                'beta2': 0.9995
            }
        else:
            # Continue with GA but more focused
            return 'ga', {
                'mutation_rate': 0.35,
                'mutation_probability': 0.75,
                'sigma': 0.1,
                'elite_fraction': 0.22,
                'tournament_size': 4,
                'immigrant_fraction': 0.12
            }

    def _get_refinement_strategy(self, pop_state: PopulationState) -> Tuple[str, Dict[str, Any]]:
        """Get optimal strategy for refinement phase."""
        if pop_state.best_fitness >= 0.6 and pop_state.diversity <= 0.2:
            # ADAM for precise final tuning
            return 'adam', {
                'learning_rate': 0.005,
                'max_perturbations': 8,
                'epsilon': 1e-7,
                'beta1': 0.97,
                'beta2': 0.9999
            }
        elif pop_state.diversity <= 0.15:
            # Stuck in local optimum - break out with GA
            return 'ga', {
                'mutation_rate': 0.6,
                'mutation_probability': 0.95,
                'sigma': 0.2,
                'elite_fraction': 0.08,
                'tournament_size': 3,
                'immigrant_fraction': 0.25
            }
        else:
            # Final ADAM tuning
            return 'adam', {
                'learning_rate': 0.001,
                'max_perturbations': 6,
                'epsilon': 1e-8,
                'beta1': 0.98,
                'beta2': 0.9999
            }

    def _apply_recommendations(
        self,
        ga_config,
        algorithm: str,
        params: Dict[str, Any],
        pop_state: PopulationState
    ) -> List[str]:
        """
        Apply parameter recommendations to the GA config with enhanced deterministic strategies.

        This implements sophisticated rule-based optimization similar to the AI agent,
        but using deterministic heuristics instead of LLM analysis.

        Args:
            ga_config: The GA configuration object to modify
            algorithm: Recommended algorithm ('ga' or 'adam')
            params: Recommended parameters
            pop_state: Current population state

        Returns:
            List of actions taken (for logging)
        """
        actions_taken = []

        try:
            # Apply algorithm-specific parameters
            if algorithm == 'ga':
                for param_name, value in params.items():
                    if hasattr(ga_config, param_name):
                        old_value = getattr(ga_config, param_name)
                        if old_value != value:
                            setattr(ga_config, param_name, value)
                            actions_taken.append(f"Set {param_name}={value} (was {old_value})")

                # Apply additional GA tuning based on population state
                additional_actions = self._apply_ga_tuning(ga_config, pop_state)
                actions_taken.extend(additional_actions)

            elif algorithm == 'adam':
                actions_taken.append(f"Recommended ADAM with {params}")
                # ADAM parameters would be handled by the optimizer factory

            # Apply population health interventions
            health_actions = self._apply_population_health_interventions(ga_config, pop_state)
            actions_taken.extend(health_actions)

            if not actions_taken:
                actions_taken.append("No parameter changes needed")

        except Exception as e:
            logger.warning(f"Error applying recommendations: {e}")
            actions_taken.append(f"Error applying recommendations: {e}")

        return actions_taken

    def _apply_ga_tuning(self, ga_config, pop_state: PopulationState) -> List[str]:
        """Apply sophisticated GA parameter tuning based on population state."""
        actions = []

        # Diversity-based tuning (similar to AI agent strategies)
        if pop_state.diversity < 0.15:
            # Low diversity - increase exploration
            if hasattr(ga_config, 'mutation_rate') and ga_config.mutation_rate < 0.6:
                ga_config.mutation_rate = min(0.6, ga_config.mutation_rate + 0.1)
                actions.append(f"Increased mutation_rate to {ga_config.mutation_rate} (low diversity)")

            if hasattr(ga_config, 'immigrant_fraction') and ga_config.immigrant_fraction < 0.2:
                ga_config.immigrant_fraction = min(0.2, ga_config.immigrant_fraction + 0.05)
                actions.append(f"Increased immigrant_fraction to {ga_config.immigrant_fraction} (diversity injection)")

        elif pop_state.diversity > 0.4:
            # High diversity - focus on exploitation
            if hasattr(ga_config, 'mutation_rate') and ga_config.mutation_rate > 0.3:
                ga_config.mutation_rate = max(0.3, ga_config.mutation_rate - 0.05)
                actions.append(f"Decreased mutation_rate to {ga_config.mutation_rate} (high diversity)")

            if hasattr(ga_config, 'elite_fraction') and ga_config.elite_fraction < 0.25:
                ga_config.elite_fraction = min(0.25, ga_config.elite_fraction + 0.05)
                actions.append(f"Increased elite_fraction to {ga_config.elite_fraction} (preserve good solutions)")

        # Fitness-based tuning
        if pop_state.best_fitness > 0.6:
            # Good fitness - fine-tune
            if hasattr(ga_config, 'mutation_sigma') and ga_config.mutation_sigma > 0.05:
                ga_config.mutation_sigma = max(0.05, ga_config.mutation_sigma - 0.02)
                actions.append(f"Decreased mutation_sigma to {ga_config.mutation_sigma} (fine-tuning)")

        elif pop_state.best_fitness < 0.3:
            # Poor fitness - increase exploration
            if hasattr(ga_config, 'mutation_sigma') and ga_config.mutation_sigma < 0.15:
                ga_config.mutation_sigma = min(0.15, ga_config.mutation_sigma + 0.02)
                actions.append(f"Increased mutation_sigma to {ga_config.mutation_sigma} (more exploration)")

        return actions

    def _apply_population_health_interventions(self, ga_config, pop_state: PopulationState) -> List[str]:
        """Apply population health interventions inspired by AI agent strategic playbooks."""
        actions = []

        # GATE CRISIS PLAYBOOK (from AI agent)
        # "Gate crisis (100% below threshold): Lower min_trades to 5-10"
        # If diversity is very low and fitness poor, assume gate crisis
        if pop_state.diversity < 0.1 and pop_state.best_fitness < 0.2 and pop_state.generation > 10:
            # Simulate lowering fitness gates by adjusting GA to be more permissive
            if hasattr(ga_config, 'tournament_size') and ga_config.tournament_size > 2:
                ga_config.tournament_size = max(2, ga_config.tournament_size - 1)
                actions.append(f"Decreased tournament_size to {ga_config.tournament_size} (gate crisis: reduce selection pressure)")

            if hasattr(ga_config, 'elite_fraction') and ga_config.elite_fraction > 0.05:
                ga_config.elite_fraction = max(0.05, ga_config.elite_fraction - 0.05)
                actions.append(f"Decreased elite_fraction to {ga_config.elite_fraction} (gate crisis: more exploration)")

        # PREMATURE CONVERGENCE PLAYBOOK (from AI agent)
        # "Premature convergence (diversity < 0.15): Inject immigrants 15-25%"
        if pop_state.diversity < 0.15 and pop_state.generation > 15:
            if hasattr(ga_config, 'immigrant_fraction') and ga_config.immigrant_fraction < 0.25:
                ga_config.immigrant_fraction = min(0.25, ga_config.immigrant_fraction + 0.08)
                actions.append(f"Increased immigrant_fraction to {ga_config.immigrant_fraction} (premature convergence)")

            if hasattr(ga_config, 'mutation_rate') and ga_config.mutation_rate < 0.7:
                ga_config.mutation_rate = min(0.7, ga_config.mutation_rate + 0.15)
                actions.append(f"Increased mutation_rate to {ga_config.mutation_rate} (restart exploration)")

        # BOUNDARY CLUSTERING PLAYBOOK (from AI agent)
        # "Boundary clustering (>30% at bounds): Mutate explorers beyond bound"
        if pop_state.diversity < 0.2 and pop_state.generation > 30:
            if hasattr(ga_config, 'mutation_probability') and ga_config.mutation_probability < 0.9:
                ga_config.mutation_probability = min(0.9, ga_config.mutation_probability + 0.1)
                actions.append(f"Increased mutation_probability to {ga_config.mutation_probability} (boundary clustering)")

            if hasattr(ga_config, 'sigma') and ga_config.sigma < 0.18:
                ga_config.sigma = min(0.18, ga_config.sigma + 0.03)
                actions.append(f"Increased sigma to {ga_config.sigma} (boundary exploration)")

        # STAGNATION PLAYBOOK (from AI agent)
        # "Stagnation (no improvement 10+ gens): High mutation_rate 0.6-0.8"
        if pop_state.generation > 40 and pop_state.best_fitness < 0.4:
            if hasattr(ga_config, 'mutation_rate') and ga_config.mutation_rate < 0.65:
                ga_config.mutation_rate = min(0.65, ga_config.mutation_rate + 0.15)
                actions.append(f"Increased mutation_rate to {ga_config.mutation_rate} (combat stagnation)")

            if hasattr(ga_config, 'mutation_probability') and ga_config.mutation_probability < 0.9:
                ga_config.mutation_probability = min(0.9, ga_config.mutation_probability + 0.1)
                actions.append(f"Increased mutation_probability to {ga_config.mutation_probability} (stagnation recovery)")

        # SPARSE SIGNALS PLAYBOOK (from AI agent)
        # "Sparse signals (best individuals have n≈0–3): Lower vol_z/tr_z for top candidates"
        # We can't directly modify individual parameters, but we can adjust GA to favor exploration
        if pop_state.generation > 20 and pop_state.best_fitness < 0.3:
            if hasattr(ga_config, 'mutation_sigma') and ga_config.mutation_sigma < 0.12:
                ga_config.mutation_sigma = min(0.12, ga_config.mutation_sigma + 0.02)
                actions.append(f"Increased mutation_sigma to {ga_config.mutation_sigma} (sparse signals: more parameter variation)")

        # OVERTRADING WITH POOR QUALITY (from AI agent)
        # "Overtrading with poor quality: Raise thresholds slightly"
        # If fitness is moderate but diversity is high, might indicate overtrading
        if pop_state.best_fitness > 0.3 and pop_state.best_fitness < 0.5 and pop_state.diversity > 0.3:
            if hasattr(ga_config, 'elite_fraction') and ga_config.elite_fraction < 0.25:
                ga_config.elite_fraction = min(0.25, ga_config.elite_fraction + 0.03)
                actions.append(f"Increased elite_fraction to {ga_config.elite_fraction} (overtrading: strengthen selection)")

        return actions

    def get_phase_info(self, generation: int) -> Optional[OptimizationPhase]:
        """Get information about the current optimization phase (dynamic based on total_iterations)."""
        exploration_end = self.phase_boundaries['exploration_end']
        exploitation_end = self.phase_boundaries['exploitation_end']

        if generation <= exploration_end:
            return OptimizationPhase(
                name="Exploration",
                algorithm="ga",
                start_generation=1,
                end_generation=exploration_end,
                description=f"Broad parameter space exploration (1-{exploration_end} of {self.total_iterations})"
            )
        elif generation <= exploitation_end:
            return OptimizationPhase(
                name="Exploitation",
                algorithm="adam",
                start_generation=exploration_end + 1,
                end_generation=exploitation_end,
                description=f"Converge to local optima ({exploration_end+1}-{exploitation_end} of {self.total_iterations})"
            )
        else:
            return OptimizationPhase(
                name="Refinement",
                algorithm="adam",
                start_generation=exploitation_end + 1,
                end_generation=self.total_iterations,
                description=f"Fine-tune parameters ({exploitation_end+1}-{self.total_iterations} of {self.total_iterations})"
            )

    def get_recommendations_history(self) -> List[Dict[str, Any]]:
        """Get history of all recommendations made."""
        return self.recommendations_history.copy()

    def reset(self):
        """Reset coach state for new optimization run."""
        self.current_phase = None
        self.last_analysis_generation = 0
        self.recommendations_history.clear()

        # Reset enhanced state tracking
        self.consecutive_low_diversity = 0
        self.consecutive_stagnation = 0
        self.last_best_fitness = 0.0

        # Reset historical learning
        self.fitness_history.clear()
        self.recommendation_success_rate.clear()
        self.parameter_importance.clear()
        self.phase_performance_history.clear()
        self.actual_phase_transitions.clear()
        self.convergence_metrics.clear()

        # Reset confidence and prediction
        self.recommendation_confidence = 0.5
        self.estimated_remaining_generations = None

        # Reset classic coach window reference
        self.classic_coach_window = None

        if self.verbose:
            logger.info("🤖 Classic Coach state reset - ready for new optimization run")

    def set_classic_coach_window(self, classic_coach_window):
        """Set the Classic Coach window for decision visualization."""
        self.classic_coach_window = classic_coach_window
        if self.verbose:
            logger.info("🤖 Classic Coach window connected for decision visualization")

    def _estimate_remaining_generations(self, pop_state: PopulationState):
        """Estimate how many more generations until convergence."""
        if len(self.fitness_history) < 5:
            self.estimated_remaining_generations = None
            return

        # Simple convergence estimation based on recent trend
        recent_improvements = []
        for i in range(1, min(5, len(self.fitness_history))):
            improvement = self.fitness_history[-i] - self.fitness_history[-i-1]
            recent_improvements.append(improvement)

        avg_improvement = sum(recent_improvements) / len(recent_improvements)

        if avg_improvement > 0.001:
            # Still improving - estimate based on convergence rate
            fitness_gap = max(0, 0.8 - pop_state.best_fitness)  # Assume 0.8 is "good enough"
            if avg_improvement > 0:
                estimated_gens = int(fitness_gap / avg_improvement)
                self.estimated_remaining_generations = min(estimated_gens, 100)  # Cap at 100
            else:
                self.estimated_remaining_generations = None
        else:
            # Plateaued or declining - could take longer
            self.estimated_remaining_generations = None

    def get_coach_capabilities(self) -> Dict[str, Any]:
        """Get the coach's capabilities and current configuration."""
        return {
            'name': 'Classic Coach',
            'type': 'deterministic',
            'capabilities': {
                'algorithm_switching': self.enable_algorithm_switching,
                'fitness_tuning': self.enable_fitness_tuning,
                'bounds_expansion': self.enable_bounds_expansion,
                'immigration': self.enable_immigration,
                'individual_mutation': self.enable_individual_mutation,
                'islands': self.enable_islands,
                'historical_learning': True,
                'trend_analysis': True,
                'adaptive_phases': True,
                'convergence_prediction': True
            },
            'configuration': self.coach_params,
            'state': {
                'consecutive_stagnation': self.consecutive_stagnation,
                'consecutive_low_diversity': self.consecutive_low_diversity,
                'last_best_fitness': self.last_best_fitness,
                'analysis_interval': self.analysis_interval,
                'recommendation_confidence': self.recommendation_confidence,
                'estimated_remaining_generations': self.estimated_remaining_generations,
                'fitness_history_size': len(self.fitness_history),
                'recommendations_count': len(self.recommendations_history)
            },
            'learning': {
                'phase_performance_history': len(self.phase_performance_history),
                'actual_phase_transitions': self.actual_phase_transitions,
                'convergence_metrics': self.convergence_metrics
            }
        }