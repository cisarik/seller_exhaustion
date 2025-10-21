"""
Evolution Coach Manager - Agent Mode

Implements agent-based Evolution Coach with tool calling:
- Population is FROZEN during agent analysis
- Agent gets FULL population snapshot with all individual data
- Agent makes multiple tool calls to diagnose and take actions
- Tools execute directly on population (mutations, GA params, fitness gates)
- Evolution resumes after agent completes

Agent workflow:
1. GA runs N generations
2. Should analyze? YES → PAUSE optimization
3. Create frozen session (population snapshot)
4. Send full population data to agent (all parameters + metrics)
5. Agent analyzes and takes actions via tools (analyze_population, mutate_individual, etc.)
6. Agent calls finish_analysis() when done
7. RESUME optimization from modified population
"""

import asyncio
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from pathlib import Path
import logging
import threading
import json

from backtest.coach_session import CoachAnalysisSession
from backtest.coach_mutations import CoachMutationManager, MutationRecord
from backtest.coach_protocol import (
    CoachAnalysis, EvolutionState, load_coach_prompt
)
from backtest.llm_coach import GemmaCoachClient
from backtest.coach_tools import CoachToolkit
from backtest.coach_agent_executor import AgentExecutor
from backtest.optimizer import Population
from core.models import FitnessConfig, OptimizationConfig
from core.agent_feed import agent_feed

logger = logging.getLogger(__name__)


class BlockingCoachManager:
    """
    Manages Evolution Coach in agent mode with tool calling.
    
    Agent-based analysis:
    - Optimization PAUSES during agent analysis
    - Agent gets FULL population snapshot with all individual data  
    - Agent makes multiple tool calls to diagnose and act
    - Actions applied directly to population via tools
    - Multi-step reasoning and iterative refinement
    
    Workflow:
    1. GA runs N generations
    2. Should analyze? YES → PAUSE optimization
    3. Create frozen session (population snapshot)
    4. Build observation from population state
    5. Agent loop: observe → think → call tools → repeat
    6. Agent finishes via finish_analysis() tool
    7. RESUME optimization from agent-modified population
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: Optional[str] = None,
        prompt_version: str = "agent01",
        system_prompt: Optional[str] = None,
        analysis_interval: Optional[int] = None,
        max_log_generations: Optional[int] = None,
        auto_apply: bool = True,
        auto_reload_model: Optional[bool] = None,
        verbose: bool = True
    ):
        """
        Initialize Blocking Coach Manager.
        
        Args:
            base_url: LM Studio endpoint
            model: Model name (loaded from settings if None)
            prompt_version: Coach prompt version (blocking_coach_v1 recommended)
            system_prompt: System prompt version (if None, uses prompt_version)
            analysis_interval: Analyze every N generations (e.g., 10)
            max_log_generations: Keep last N generations in logs
            auto_apply: Automatically apply recommendations
            auto_reload_model: Auto unload/reload model after recommendations
            verbose: Print detailed logs
        """
        # Load from settings if not provided
        from config.settings import settings
        
        self.base_url = base_url
        self.model = model or settings.coach_model
        self.prompt_version = prompt_version or "blocking_coach_v1"
        self.system_prompt = system_prompt or getattr(
            settings, 'coach_system_prompt', self.prompt_version
        )
        self.analysis_interval = analysis_interval or getattr(
            settings, 'coach_analysis_interval', 10
        )
        self.population_window = getattr(
            settings, 'coach_population_window', 10
        )
        self.max_log_generations = max_log_generations or getattr(
            settings, 'coach_max_log_generations', 25
        )
        self.auto_apply = auto_apply
        self.auto_reload_model = auto_reload_model if auto_reload_model is not None else getattr(
            settings, 'coach_auto_reload_model', True
        )
        self.verbose = verbose
        self.debug_payloads = getattr(settings, 'coach_debug_payloads', False)
        
        # Log initialization to console
        print(f"✓ 🤖 Evolution Coach Manager initialized: "
              f"interval={self.analysis_interval} gens, window={self.population_window} gens")
        if self.debug_payloads:
            logger.info("Evolution Coach debug payload logging ENABLED (full LLM traffic will be logged)")
            print("✓ 🧪 Coach debug payload logging enabled")
        
        # State tracking
        self.coach_client: Optional[GemmaCoachClient] = None
        self.mutation_manager = CoachMutationManager(verbose=verbose)
        
        self.last_session: Optional[CoachAnalysisSession] = None
        self.session_history: List[CoachAnalysisSession] = []
        
        self._last_analysis_generation: int = 0  # First analysis at gen = analysis_interval (e.g., gen 10)
        self._first_analysis_done: bool = False
        
        # Log collection
        self.generation_logs: List[Tuple[int, str]] = []
    
    def should_analyze(self, generation: int) -> bool:
        """
        Check if we should trigger coach analysis this generation.
        
        INTERVAL-BASED: Analyze every N generations (e.g., every 10 gens)
        
        First analysis: gen 10
        Subsequent: gen 20, 30, 40, etc.
        """
        # Ensure generation is int (defensive programming)
        try:
            gen_num = int(generation) if isinstance(generation, str) else generation
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid generation value: {generation} ({type(generation)})")
            return False
        
        # Check if enough generations have passed since last analysis
        should_trigger = gen_num - self._last_analysis_generation >= self.analysis_interval
        
        if should_trigger:
            self._last_analysis_generation = gen_num  # Mark this as analysis generation
        
        return should_trigger
    
    def record_generation(self, population: Population, coach_triggered: bool = False, coach_recommendations: int = 0, mutations_applied: int = 0, ga_changes_applied: int = 0):
        """
        Record generation data to agent_feed.
        
        Args:
            population: Current population
            coach_triggered: Whether coach was triggered this generation
            coach_recommendations: Number of coach recommendations
            mutations_applied: Number of mutations applied
            ga_changes_applied: Number of GA parameter changes
        """
        if not population.individuals:
            return
        
        # Calculate population metrics
        fitnesses = [ind.fitness for ind in population.individuals if ind.fitness is not None]
        if not fitnesses:
            return
        
        best_fitness = max(fitnesses)
        mean_fitness = sum(fitnesses) / len(fitnesses)
        worst_fitness = min(fitnesses)
        
        # Calculate diversity (std dev of fitness)
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
            worst_fitness=worst_fitness,
            diversity=diversity,
            best_params=best_ind.to_dict() if hasattr(best_ind, 'to_dict') else {},
            best_metrics=best_ind.metrics if hasattr(best_ind, 'metrics') else {},
            coach_triggered=coach_triggered,
            coach_recommendations_count=coach_recommendations,
            mutations_applied=mutations_applied,
            ga_changes_applied=ga_changes_applied
        )
        
        # Also log to console for real-time feedback
        print(f"[Gen {population.generation:3d}] fitness: best={best_fitness:.4f} mean={mean_fitness:.4f} diversity={diversity:.4f}")
    
    def add_log(self, generation: int, log_line: str, to_debug: bool = False):
        """
        Add log line for this generation (legacy method, kept for compatibility).
        
        Args:
            generation: Generation number
            log_line: Log message
            to_debug: If True, log to debug logger instead
        """
        self.generation_logs.append((generation, log_line))
        
        if to_debug:
            # Diagnostic logs go to logger
            logger.debug(f"[Gen {generation:3d}] {log_line}")
        else:
            # Print to console
            print(f"[Gen {generation:3d}] {log_line}")
        
        # Trim old logs
        if len(self.generation_logs) > self.max_log_generations * 10:
            min_gen = generation - self.max_log_generations
            self.generation_logs = [
                (g, line) for g, line in self.generation_logs if g >= min_gen
            ]
    
    def get_recent_logs(self, n_generations: Optional[int] = None) -> List[str]:
        """
        Get recent evolution history from agent_feed formatted for coach.
        
        Args:
            n_generations: Number of generations to include. 
                          If None, uses self.population_window
        
        Returns:
            List of formatted log lines from agent_feed
        """
        if n_generations is None:
            n_generations = self.population_window
        
        # Get recent generations from agent_feed
        recent_gens = agent_feed.get_generations(last_n=n_generations)
        
        if not recent_gens:
            return ["No generation data available"]
        
        # Format as log lines for coach
        log_lines = []
        log_lines.append(f"RECENT LOG HISTORY (last {len(recent_gens)} generations):")
        log_lines.append("")
        
        for gen_record in recent_gens:
            # Format: [Gen XXX] fitness: best=X.XXXX mean=X.XXXX diversity=X.XX
            line = (f"[Gen {gen_record.generation:3d}] "
                   f"fitness: best={gen_record.best_fitness:.4f} "
                   f"mean={gen_record.mean_fitness:.4f} "
                   f"diversity={gen_record.diversity:.2f}")
            
            # Add coach info if triggered
            if gen_record.coach_triggered:
                line += f" [COACH: {gen_record.coach_recommendations_count} recs, "
                line += f"{gen_record.mutations_applied} mutations]"
            
            log_lines.append(line)
        
        # Add summary statistics
        log_lines.append("")
        summary = agent_feed.get_summary()
        if summary['total_generations'] > 0:
            log_lines.append(f"Summary: {summary['total_generations']} gens total, "
                           f"best ever: {summary['best_ever']['fitness']:.4f} @ gen {summary['best_ever']['generation']}, "
                           f"recent mean: {summary['recent_mean_fitness']:.4f}")
        
        return log_lines
    
    def clear_agent_feed(self):
        """
        DEPRECATED: Do NOT clear agent_feed - we want continuous history!
        
        Agent feed should accumulate ALL generations for historical analysis.
        The max_generations limit in AgentFeed handles automatic cleanup.
        """
        # DO NOT CLEAR - we want to keep all generation history!
        logger.debug("⚠️  clear_agent_feed() called but doing nothing (agent_feed keeps history)")
    
    async def create_analysis_session(
        self,
        population: Population,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig
    ) -> CoachAnalysisSession:
        """
        Create a frozen analysis session from current population.
        
        CRITICAL: This FREEZES the population state.
        Coach will analyze this exact state.
        Recommendations will apply ONLY to this state.
        
        Args:
            population: Population to snapshot
            fitness_config: Current fitness configuration
            ga_config: Current GA configuration
        
        Returns:
            Frozen CoachAnalysisSession
        """
        # Log to coach window (user sees this)
        print(
            f"[COACH  ] ❄️  Freezing population at Gen {population.generation}"
        )
        
        session = CoachAnalysisSession.from_population(
            population, fitness_config, ga_config
        )
        
        self.last_session = session
        self.session_history.append(session)
        
        # Show session details
        print(
            f"[COACH  ] 📸 Session {session.session_id}: "
            f"{session.population_size} individuals, "
            f"diversity={session.get_population_metrics()['diversity']:.2f}"
        )
        
        return session
    

    

    

    
    async def analyze_and_apply_with_agent(
        self,
        population: Population,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig,
        current_data=None  # Unused, kept for API compatibility
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Full agent-based analysis workflow with tool-calling agent.
        
        This is the AGENT MODE alternative to analyze_and_apply_blocking.
        Instead of single-shot JSON response, agent makes multiple tool calls to:
        - Analyze population
        - Diagnose problems
        - Take corrective actions (mutate, adjust params, inject diversity)
        - Finish when done
        
        Workflow:
        1. FREEZE population (create session)
        2. CREATE toolkit and agent
        3. RUN agent analysis (agent makes tool calls)
        4. RETURN to caller (GA resumes with modified population)
        
        Args:
            population: Population to analyze and modify
            fitness_config: Fitness configuration  
            ga_config: GA configuration
            current_data: Unused (kept for API compatibility)
        
        Returns:
            (success, summary_dict)
        """
        from config.settings import settings
        
        print(
            f"[AGENT  ] 🤖 Starting agent-based analysis at Gen {population.generation}"
        )
        
        # Step 1: FREEZE population
        session = await self.create_analysis_session(
            population, fitness_config, ga_config
        )
        
        # Step 2: Initialize client if needed
        await self._initialize_client()
        
        # Step 3: Create toolkit for agent
        toolkit = CoachToolkit(
            population=population,
            session=session,
            fitness_config=fitness_config,
            ga_config=ga_config,
            mutation_manager=self.mutation_manager
        )
        
        # Step 4: Create agent executor
        max_iterations = getattr(settings, 'coach_agent_max_iterations', 10)
        agent = AgentExecutor(
            llm_client=self.coach_client,
            toolkit=toolkit,
            max_iterations=max_iterations,
            verbose=self.verbose
        )
        
        print(
            f"[AGENT  ] 🔧 Agent created with max_iterations={max_iterations}"
        )
        
        # Step 5: Build initial observation
        observation = self._build_agent_observation(session, fitness_config, ga_config)
        
        # Step 6: RUN agent analysis
        print(f"[AGENT  ] 🚀 Running agent analysis...")
        
        try:
            result = await agent.run_analysis(observation)
            
            if result.get("success"):
                print(
                    f"[AGENT  ] ✅ Agent completed: "
                    f"{result['iterations']} iterations, "
                    f"{result['tool_calls_count']} tool calls"
                )
                
                # Log actions taken
                if toolkit.actions_log:
                    print(
                        f"[AGENT  ] 📋 Actions taken: {len(toolkit.actions_log)}"
                    )
                    for i, action in enumerate(toolkit.actions_log[:5], 1):
                        action_name = action.get("action", "unknown")
                        print(f"[AGENT  ]   {i}. {action_name}")
                    
                    if len(toolkit.actions_log) > 5:
                        print(
                            f"[AGENT  ]   ... and {len(toolkit.actions_log) - 5} more actions"
                        )
                else:
                    print("[AGENT  ] ⚠️  No actions logged")
                
                # Store toolkit for later inspection
                self.last_toolkit = toolkit
                
                # Step 7: Optionally reload model to clear context
                if self.auto_reload_model:
                    print(
                        f"[AGENT  ] 🔄 Reloading model to clear context window"
                    )
                    try:
                        await self.coach_client.reload_model()
                        print(f"[AGENT  ] ✅ Model reloaded")
                    except Exception as e:
                        logger.warning(f"Failed to reload model: {e}")
                
                print(
                    f"[AGENT  ] ✅ Agent workflow complete - "
                    f"GA will resume with modified population"
                )
                
                # DO NOT clear agent_feed - we want continuous history!
                
                return True, {
                    "total_actions": len(toolkit.actions_log),
                    "iterations": result['iterations'],
                    "tool_calls": result['tool_calls_count'],
                    "actions_log": toolkit.actions_log
                }
            else:
                error = result.get("error", "Unknown error")
                # Log to console only
                logger.error("Agent failed: %s", error)
                return False, {"error": error}
                
        except Exception as e:
            # Log errors to console only - not useful for agent
            logger.exception("Agent analysis failed: %s", e)
            return False, {"error": str(e)}
    
    def _build_agent_observation(self, session: CoachAnalysisSession, fitness_config: FitnessConfig, ga_config: OptimizationConfig) -> str:
        """
        Build initial observation message for agent.
        
        Args:
            session: Frozen session with population data
            fitness_config: Current fitness configuration
            ga_config: Current GA configuration
        
        Returns:
            Formatted observation message for agent
        """
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
        
        # Add generation history from agent_feed
        recent_gens = agent_feed.get_generations(last_n=self.population_window)
        
        if recent_gens:
            obs += f"EVOLUTION HISTORY (last {len(recent_gens)} generations):\n\n"
            
            # Header row
            obs += f"{'Gen':>4} | {'Best Fit':>8} | {'Mean Fit':>8} | {'Diversity':>9} | {'Trades':>6} | {'Events':>20}\n"
            obs += "-" * 80 + "\n"
            
            # Generation rows
            for gen_record in recent_gens:
                # Build events string
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
        
        # Summary from agent_feed
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
    
    async def _initialize_client(self):
        """Initialize coach client if not already done."""
        if not self.coach_client:
            self.coach_client = GemmaCoachClient(
                base_url=self.base_url,
                model=self.model,
                prompt_version="agent01",  # Use agent prompt
                system_prompt="agent01",
                verbose=self.verbose,
                debug_payloads=self.debug_payloads
            )
            logger.debug(f"✓ 🔌 LM Studio agent client initialized")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coach manager statistics."""
        return {
            "total_sessions": len(self.session_history),
            "total_mutations": self.mutation_manager.get_stats()["total_mutations"],
            "last_session": (
                self.last_session.to_dict() if self.last_session else None
            ),
            "mutation_stats": self.mutation_manager.get_stats(),
        }


# Example usage
if __name__ == "__main__":
    print("Evolution Coach Manager - Agent Mode")
    print("\nKey Features:")
    print("  - Agent-based analysis with tool calling")
    print("  - Full population data sent to agent")
    print("  - Direct individual mutations via tools")
    print("  - Frozen session tracking")
    print("  - Multi-step reasoning and diagnosis")
    print("\nUsage:")
    print("  manager = BlockingCoachManager()")
    print("  if manager.should_analyze(gen):")
    print("      success, summary = await manager.analyze_and_apply_with_agent(pop, fitness, ga)")
    print("  # GA continues with agent-modified population")
