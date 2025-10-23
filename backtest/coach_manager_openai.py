"""
OpenAI Agents Framework Coach Manager

Uses proper OpenAI Agents framework with Runner.run() for tool-calling agent.
Replaces custom agent executor with official OpenAI Agents library.
"""

import asyncio
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from pathlib import Path
import logging
import threading
import json

logger = logging.getLogger(__name__)

from backtest.coach_session import CoachAnalysisSession
from backtest.coach_mutations import CoachMutationManager, MutationRecord
from backtest.coach_protocol import (
    CoachAnalysis, EvolutionState, load_coach_prompt
)
from backtest.coach_agent_openai import CoachAgentOpenAI
from backtest.optimizer import Population
from core.models import FitnessConfig, OptimizationConfig, AdamConfig
from core.agent_feed import agent_feed

logger = logging.getLogger(__name__)


class OpenAICoachManager:
    """
    OpenAI Agents framework coach manager.

    Uses proper Agent and Runner classes from OpenAI Agents library
    instead of custom agent executor.
    """

    def __init__(
        self,
        analysis_interval: Optional[int] = None,
        auto_apply: bool = True,
        verbose: bool = True,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "anthropic/claude-3.5-sonnet",
        status_callback: Optional[callable] = None,
        coach_window=None,
    ):
        """
        Initialize OpenAI Agents coach manager.

        Args:
            analysis_interval: Analyze every N generations
            auto_apply: Automatically apply recommendations
            verbose: Print detailed logs
            openrouter_api_key: OpenRouter API key
            openrouter_model: OpenRouter model name
            status_callback: Callback for status updates (tool_name, reason)
        """
        # Load from settings
        from config.settings import settings

        self.analysis_interval = analysis_interval or settings.coach_analysis_interval
        self.auto_apply = auto_apply
        self.verbose = verbose
        self.debug_payloads = getattr(settings, 'coach_debug_payloads', False)

        # OpenRouter settings
        self.openrouter_api_key = openrouter_api_key or getattr(settings, 'openrouter_api_key', '')
        self.openrouter_model = openrouter_model or getattr(settings, 'openrouter_model', 'anthropic/claude-3.5-sonnet')
        self.status_callback = status_callback
        self.coach_window = coach_window

        # Island model state (only if enabled)
        self.islands: Dict[int, Population] = {}
        self.island_policy = {
            "migration_cadence": 5,
            "migration_size": 1,
            "merge_to_main_cadence": 0,
            "merge_top_k": 1
        }
        
        # Check if islands management is enabled
        self.islands_enabled = getattr(settings, 'coach_islands_enabled', False)
        if not self.islands_enabled and self.verbose:
            print("🚫 Islands Management disabled in settings")

        # State tracking
        self.mutation_manager = CoachMutationManager(verbose=verbose)
        self.adam_config = AdamConfig()  # Default ADAM configuration

        self.last_session: Optional[CoachAnalysisSession] = None
        self.session_history: List[CoachAnalysisSession] = []

        self._last_analysis_generation: int = 0
        self._first_analysis_done: bool = False

        # Log collection
        self.generation_logs: List[Tuple[int, str]] = []

        # Log initialization (only if verbose)
        if self.verbose:
            print(f"✓ 🤖 Coach Manager initialized: model={self.openrouter_model}, interval={self.analysis_interval} gens")
            
            if self.debug_payloads:
                print("✓ 🧪 Debug payload logging enabled")

    def should_analyze(self, generation: int) -> bool:
        """Check if we should trigger coach analysis this generation."""
        try:
            gen_num = int(generation) if isinstance(generation, str) else generation
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid generation value: {generation} ({type(generation)})")
            return False

        should_trigger = gen_num - self._last_analysis_generation >= self.analysis_interval

        if should_trigger:
            self._last_analysis_generation = gen_num

        return should_trigger

    def record_generation(self, population: Population, coach_triggered: bool = False, coach_recommendations: int = 0, mutations_applied: int = 0, ga_changes_applied: int = 0):
        """Record generation data to agent_feed."""
        if not population.individuals:
            return

        fitnesses = [ind.fitness for ind in population.individuals if ind.fitness is not None]
        if not fitnesses:
            return

        best_fitness = max(fitnesses)
        mean_fitness = sum(fitnesses) / len(fitnesses)

        # Calculate diversity
        fitness_variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        diversity = fitness_variance ** 0.5

        # Get best individual
        best_ind = max(population.individuals, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))

        # Record to agent_feed
        agent_feed.record_generation(
            generation=population.generation,
            population_size=len(population.individuals),
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            worst_fitness=min(fitnesses),
            diversity=diversity,
            best_params=best_ind.to_dict() if hasattr(best_ind, 'to_dict') else {},
            best_metrics=best_ind.metrics if hasattr(best_ind, 'metrics') else {},
            coach_triggered=coach_triggered,
            coach_recommendations_count=coach_recommendations,
            mutations_applied=mutations_applied,
            ga_changes_applied=ga_changes_applied
        )

        # Console log
        print(f"[Gen {population.generation:3d}] fitness: best={best_fitness:.4f} mean={mean_fitness:.4f} diversity={diversity:.4f}")

    def add_log(self, generation: int, log_line: str, to_debug: bool = False):
        """Add log line for this generation."""
        self.generation_logs.append((generation, log_line))

        if to_debug:
            logger.debug(f"[Gen {generation:3d}] {log_line}")
        else:
            print(f"[Gen {generation:3d}] {log_line}")

    def get_recent_logs(self, n_generations: Optional[int] = None) -> List[str]:
        """Get recent evolution history from agent_feed."""
        recent_gens = agent_feed.get_generations(last_n=None)

        if not recent_gens:
            return ["No generation data available"]

        log_lines = []
        log_lines.append(f"RECENT LOG HISTORY (last {len(recent_gens)} generations):")
        log_lines.append("")

        for gen_record in recent_gens:
            line = (f"[Gen {gen_record.generation:3d}] "
                   f"fitness: best={gen_record.best_fitness:.4f} "
                   f"mean={gen_record.mean_fitness:.4f} "
                   f"diversity={gen_record.diversity:.2f}")

            if gen_record.coach_triggered:
                line += f" [COACH: {gen_record.coach_recommendations_count} recs, "
                line += f"{gen_record.mutations_applied} mutations]"

            log_lines.append(line)

        # Add summary
        log_lines.append("")
        summary = agent_feed.get_summary()
        if summary['total_generations'] > 0:
            log_lines.append(f"Summary: {summary['total_generations']} gens total, "
                           f"best ever: {summary['best_ever']['fitness']:.4f} @ gen {summary['best_ever']['generation']}, "
                           f"recent mean: {summary['recent_mean_fitness']:.4f}")

        return log_lines

    async def create_analysis_session(
        self,
        population: Population,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig
    ) -> CoachAnalysisSession:
        """Create a frozen analysis session from current population."""
        if self.verbose:
            print(f"❄️  Freezing population at Gen {population.generation}")

        session = CoachAnalysisSession.from_population(
            population, fitness_config, ga_config
        )

        self.last_session = session
        self.session_history.append(session)

        if self.verbose:
            metrics = session.get_population_metrics()
            print(f"📸 Session created: {session.population_size} individuals, diversity={metrics['diversity']:.2f}")

        return session

    async def analyze_and_apply_with_openai_agent(
        self,
        population: Population,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig,
        current_data=None,  # Unused, kept for API compatibility
        coach_window=None,  # Coach window for UI updates
        status_callback=None  # Status callback for UI updates
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Full OpenAI Agents analysis workflow.

        Uses proper Agent and Runner classes from OpenAI Agents library.

        Args:
            population: Population to analyze and modify
            fitness_config: Fitness configuration
            ga_config: GA configuration
            current_data: Unused (kept for API compatibility)

        Returns:
            (success, summary_dict)
        """
        print(f"[AGENT  ] 🤖 Starting OpenAI Agents analysis at Gen {population.generation}")

        # Step 1: FREEZE population
        session = await self.create_analysis_session(
            population, fitness_config, ga_config
        )

        # Step 2: Create OpenAI Agents coach agent
        # Each analysis creates a fresh coach agent to prevent context window overflow
        # when running many analyses (e.g., 500+ analyses)
        coach_agent = CoachAgentOpenAI(
            population=population,
            session=session,
            fitness_config=fitness_config,
            ga_config=ga_config,
            mutation_manager=self.mutation_manager,
            adam_config=getattr(self, 'adam_config', None),
            islands_registry=self.islands if self.islands_enabled else None,
            island_policy_reference=self.island_policy if self.islands_enabled else None,
            openrouter_api_key=self.openrouter_api_key,
            openrouter_model=self.openrouter_model,
            verbose=self.verbose,
            status_callback=status_callback or self.status_callback,
            coach_window=coach_window or getattr(self, 'coach_window', None)
        )

        # Step 3: Build initial observation
        observation = self._build_agent_observation(session, fitness_config, ga_config)

        # Step 4: RUN OpenAI Agents analysis
        # Each analysis starts with a fresh conversation history to prevent context window overflow
        # when running many analyses (e.g., 500+ analyses)
        logger.info(f"[AGENT  ] 🚀 Running OpenAI Agents analysis...")
        logger.debug(f"[AGENT  ]   Observation: {observation[:500]}...")

        try:
            result = await coach_agent.run_analysis(observation)

            if result.get("success"):
                logger.info(f"[AGENT  ] ✅ Agent completed: "
                     f"{result['iterations']} iterations, "
                     f"{result['tool_calls_count']} tool calls")

                # Log actions taken
                actions_taken = result.get("actions_taken", [])
                if actions_taken:
                    logger.info(f"[AGENT  ] 📋 Actions taken: {len(actions_taken)}")
                    for i, action in enumerate(actions_taken[:5], 1):
                        action_name = action.get("action", "unknown")
                        logger.info(f"[AGENT  ]   {i}. {action_name}")
                        logger.debug(f"[AGENT  ]      Action details: {action}")

                    if len(actions_taken) > 5:
                        logger.info(f"[AGENT  ]   ... and {len(actions_taken) - 5} more actions")
                else:
                    logger.warning("[AGENT  ] ⚠️  No actions logged")

                logger.info(f"[AGENT  ] ✅ OpenAI Agents workflow complete - "
                     f"GA will resume with modified population")

                # Show debugger with tool history (disabled to avoid Qt thread issues)
                # if 'tool_history' in result:
                #     from app.widgets.coach_debugger import CoachDebugger
                #     debugger = CoachDebugger(result['tool_history'])
                #     debugger.show()

                return True, {
                    "total_actions": len(actions_taken),
                    "iterations": result['iterations'],
                    "tool_calls": result['tool_calls_count'],
                    "actions_log": actions_taken,
                    "summary": result.get('summary', 'Agent completed analysis')
                }
            else:
                error = result.get("error", "Unknown error")
                logger.error("OpenAI Agents failed: %s", error)
                logger.debug(f"Full error result: {result}")
                return False, {
                    "error": error,
                    "summary": f"Agent failed: {error}"
                }

        except Exception as e:
            logger.exception("OpenAI Agents analysis failed: %s", e)
            logger.debug(f"Exception details: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False, {
                "error": str(e),
                "summary": f"Analysis error: {str(e)}"
            }

    def _build_agent_observation(self, session: CoachAnalysisSession, fitness_config: FitnessConfig, ga_config: OptimizationConfig) -> str:
        """Build initial observation message for agent."""
        pop_metrics = session.get_population_metrics()

        obs = f"""POPULATION STATE - Generation {session.generation}

OVERVIEW:
- Population size: {session.population_size}
- Mean fitness: {pop_metrics['mean_fitness']:.4f}
- Std fitness: {pop_metrics['std_fitness']:.4f}
- Diversity: {pop_metrics['diversity']:.2f} ({self._interpret_diversity_level(pop_metrics['diversity'])})

GATE COMPLIANCE:
- Min trades required: {fitness_config.min_trades}
- Individuals below min_trades: {pop_metrics.get('below_min_trades_count', 0)} ({pop_metrics.get('below_min_trades_pct', 0):.1f}%)

TOP PERFORMERS:
"""

        # Add top 3 individuals
        sorted_inds = sorted(session.individuals_snapshot, key=lambda x: x.fitness, reverse=True)
        for i, ind in enumerate(sorted_inds[:3], 1):
            obs += f"""  {i}. Individual #{ind.id}: fitness={ind.fitness:.4f}
     Metrics: trades={ind.metrics.get('n', 0)}, win_rate={ind.metrics.get('win_rate', 0):.2f}, avg_R={ind.metrics.get('avg_R', 0):.2f}
     Key params: {str(ind.parameters)[:80]}

"""

        obs += "\nBOTTOM PERFORMERS:\n"
        for i, ind in enumerate(sorted_inds[-3:], 1):
            obs += f"""  {i}. Individual #{ind.id}: fitness={ind.fitness:.4f}
     Metrics: trades={ind.metrics.get('n', 0)}, win_rate={ind.metrics.get('win_rate', 0):.2f}
     Key params: {str(ind.parameters)[:80]}

"""

        obs += f"""
GA CONFIGURATION:
- Mutation rate: {ga_config.mutation_rate}
- Elite fraction: {ga_config.elite_fraction}
- Tournament size: {ga_config.tournament_size}
- Immigrant fraction: {ga_config.immigrant_fraction}

"""

        # Add generation history
        recent_gens = agent_feed.get_generations(last_n=None)

        if recent_gens:
            obs += f"EVOLUTION HISTORY ({len(recent_gens)} generations):\n\n"

            # Header row
            obs += f"{'Gen':>4} | {'Best Fit':>8} | {'Mean Fit':>8} | {'Diversity':>9} | {'Trades':>6} | {'Events':>20}\n"
            obs += "-" * 80 + "\n"

            # Generation rows
            for gen_record in recent_gens:
                events = []
                if gen_record.coach_triggered:
                    events.append(f"Coach:{gen_record.coach_recommendations_count}r")
                if gen_record.mutations_applied > 0:
                    events.append(f"{gen_record.mutations_applied}mut")
                if gen_record.ga_changes_applied > 0:
                    events.append(f"{gen_record.ga_changes_applied}ga")
                events_str = " ".join(events) if events else "-"

                obs += (
                    f"{gen_record.generation:4d} | "
                    f"{gen_record.best_fitness:8.4f} | "
                    f"{gen_record.mean_fitness:8.4f} | "
                    f"{gen_record.diversity:9.4f} | "
                    f"{gen_record.best_metrics.get('n', 0):6d} | "
                    f"{events_str:>20}\n"
                )

            # Trend analysis
            if len(recent_gens) >= 3:
                obs += "\nTREND ANALYSIS:\n"

                # Fitness trend
                fitness_improvements = [
                    recent_gens[i].best_fitness - recent_gens[i-1].best_fitness
                    for i in range(1, len(recent_gens))
                ]
                avg_improvement = sum(fitness_improvements) / len(fitness_improvements)

                if abs(avg_improvement) < 0.001:
                    obs += f"⚠️ STAGNATION: fitness nearly flat ({avg_improvement:.5f}/gen avg change)\n"
                elif avg_improvement < -0.001:
                    obs += f"⚠️ REGRESSION: fitness declining ({avg_improvement:.4f}/gen)\n"
                else:
                    obs += f"✓ IMPROVING: {avg_improvement:.4f}/gen average improvement\n"

                # Diversity trend
                diversity_vals = [g.diversity for g in recent_gens]
                diversity_change = diversity_vals[-1] - diversity_vals[0]

                if diversity_vals[-1] < 0.10:
                    obs += f"⚠️ CONVERGENCE: diversity very low ({diversity_vals[-1]:.4f})\n"
                elif diversity_change < -0.05:
                    obs += f"⚠️ COLLAPSING: diversity dropping fast ({diversity_change:.4f})\n"

                # Trade count trend
                trade_counts = [g.best_metrics.get('n', 0) for g in recent_gens]
                recent_avg_trades = sum(trade_counts[-3:]) / 3

                if recent_avg_trades < fitness_config.min_trades:
                    failing_count = sum(1 for tc in trade_counts[-3:] if tc < fitness_config.min_trades)
                    obs += f"⚠️ GATE CRISIS: {failing_count}/3 recent best below min_trades ({fitness_config.min_trades})\n"

            obs += "\n"

        # Summary
        summary = agent_feed.get_summary()
        if summary['total_generations'] > 0:
            obs += f"""SUMMARY STATISTICS:
- Total generations tracked: {summary['total_generations']}
- Best ever: {summary['best_ever']['fitness']:.4f} @ gen {summary['best_ever']['generation']}
- Recent mean fitness: {summary['recent_mean_fitness']:.4f}
- Recent best fitness: {summary['recent_best_fitness']:.4f}
- Recent diversity: {summary['recent_diversity']:.4f}

"""

        obs += """Your task: Analyze this population state and take actions to improve evolution.
Start by calling analyze_population() to get detailed statistics.
"""

        return obs

    def _interpret_diversity_level(self, diversity: float) -> str:
        """Interpret diversity metric."""
        if diversity < 0.1:
            return "VERY LOW - converged"
        elif diversity < 0.2:
            return "LOW"
        elif diversity < 0.4:
            return "MODERATE"
        else:
            return "HIGH"

    def get_stats(self) -> Dict[str, Any]:
        """Get coach manager statistics."""
        return {
            "total_sessions": len(self.session_history),
            "total_mutations": self.mutation_manager.get_stats()["total_mutations"],
            "last_session": (
                self.last_session.to_dict() if self.last_session else None
            ),
            "mutation_stats": self.mutation_manager.get_stats(),
            "islands": {k: getattr(v, 'size', 0) for k, v in self.islands.items()},
        }

    def evolve_islands_step(self, data, timeframe, fitness_config, ga_config: OptimizationConfig, main_population: Optional[Population] = None) -> None:
        """Evolve islands one generation and perform periodic migrations."""
        if not self.islands_enabled:
            return
        if not self.islands:
            return
        try:
            from backtest.optimizer import evolution_step
            from copy import deepcopy

            # Evolve all islands
            for island_id, pop in list(self.islands.items()):
                new_pop = evolution_step(
                    population=pop,
                    data=data,
                    tf=timeframe,
                    fitness_config=fitness_config,
                    ga_config=ga_config
                )
                self.islands[island_id] = new_pop
                self.add_log(new_pop.generation, f"ISLAND {island_id} evolved → best={new_pop.get_best().fitness:.4f}")

            # Periodic ring migration
            ids = sorted(self.islands.keys())
            if not ids:
                return
            ref_gen = self.islands[ids[0]].generation
            mig_cad = int(self.island_policy.get("migration_cadence", 0))
            mig_sz = int(self.island_policy.get("migration_size", 1))
            if mig_cad > 0 and ref_gen % mig_cad == 0 and len(ids) > 1:
                for i, src_id in enumerate(ids):
                    dst_id = ids[(i + 1) % len(ids)]
                    src = self.islands[src_id]
                    dst = self.islands[dst_id]
                    # Select migrants
                    migrants = sorted(src.individuals, key=lambda x: x.fitness, reverse=True)[: mig_sz]
                    migrants = [deepcopy(m) for m in migrants]
                    # Replace worst in destination
                    dst_sorted = sorted(dst.individuals, key=lambda x: x.fitness, reverse=True)
                    keep = dst_sorted[: max(0, len(dst_sorted) - mig_sz)]
                    dst.individuals = keep + migrants
                    dst.size = len(dst.individuals)
                    self.add_log(ref_gen, f"MIGRATE {len(migrants)} from island {src_id} → {dst_id}")

            # Periodic merge from islands to main population
            merge_cad = int(self.island_policy.get("merge_to_main_cadence", 0))
            merge_k = int(self.island_policy.get("merge_top_k", 0))
            if main_population is not None and merge_cad > 0 and merge_k > 0 and ref_gen % merge_cad == 0:
                try:
                    # Collect elites from all islands
                    elites = []
                    for island_id, pop in self.islands.items():
                        top = sorted(pop.individuals, key=lambda x: x.fitness, reverse=True)[: merge_k]
                        elites.extend([deepcopy(ind) for ind in top])
                    if elites:
                        # Replace worst in main population
                        dst_sorted = sorted(main_population.individuals, key=lambda x: x.fitness, reverse=True)
                        keep = dst_sorted[: max(0, len(dst_sorted) - len(elites))]
                        main_population.individuals = keep + elites
                        main_population.size = len(main_population.individuals)
                        self.add_log(ref_gen, f"MERGE {len(elites)} island elites → main population")
                except Exception as me:
                    logger.exception("Island→main merge failed: %s", me)
        except Exception as e:
            logger.exception("Island evolution step failed")


# Example usage
if __name__ == "__main__":
    print("OpenAI Agents Coach Manager")
    print("\nKey Features:")
    print("  - Uses proper OpenAI Agents framework")
    print("  - Agent and Runner classes")
    print("  - @function_tool decorators")
    print("  - OpenRouter API integration")
    print("\nUsage:")
    print("  manager = OpenAICoachManager()")
    print("  if manager.should_analyze(gen):")
    print("      success, summary = await manager.analyze_and_apply_with_openai_agent(pop, fitness, ga)")
    print("  # GA continues with agent-modified population")