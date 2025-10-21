"""
Evolution Coach Manager - Blocking Version

Implements BLOCKING optimization semantics:
- Population is FROZEN during coach analysis
- Coach analyzes exact population state (with all individual data)
- Recommendations applied to SAME population state
- Evolution resumes after mutations applied

This replaces the non-blocking version to solve:
1. Population drift (evolution continuing while coach analyzes)
2. Stale recommendations (coach analyzed old state, applies to new state)
3. Difficulty tracking mutations (population changed between analysis and application)
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
from core.coach_logging import coach_log_manager, debug_log_manager

logger = logging.getLogger(__name__)


class BlockingCoachManager:
    """
    Manages Evolution Coach with BLOCKING semantics.
    
    KEY DIFFERENCE from non-blocking:
    - Optimization PAUSES during coach analysis
    - Coach gets FULL population snapshot with all individual data
    - Recommendations applied to SAME population state
    - Can include direct individual mutations
    
    Workflow:
    1. GA runs N generations
    2. Should analyze? YES → PAUSE optimization
    3. Create frozen session (population snapshot)
    4. Send full population data to coach (all parameters + metrics)
    5. Coach analyzes and recommends (may include mutations)
    6. Apply recommendations (mutations to same frozen population)
    7. RESUME optimization from modified population
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: Optional[str] = None,
        prompt_version: str = "blocking_coach_v1",
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
        
        # Log to debug log (not user-facing coach log) - use startup style
        debug_log_manager.append(
            f"✓ 🤖 Evolution Coach Manager initialized: "
            f"interval={self.analysis_interval} gens, window={self.population_window} gens"
        )
        if self.debug_payloads:
            logger.info("Evolution Coach debug payload logging ENABLED (full LLM traffic will be logged)")
            debug_log_manager.append("✓ 🧪 Coach debug payload logging enabled")
        
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
    
    def add_log(self, generation: int, log_line: str, to_debug: bool = False):
        """
        Add log line for this generation.
        
        Args:
            generation: Generation number
            log_line: Log message
            to_debug: If True, log to debug log instead of coach log
        """
        self.generation_logs.append((generation, log_line))
        
        if to_debug:
            # Diagnostic logs go to debug log (startup, etc.) - simple format
            debug_log_manager.append(f"  {log_line}")
        else:
            # Relevant coach logs go to coach log
            coach_log_manager.append(f"[Gen {generation:3d}] {log_line}")
        
        # Trim old logs
        if len(self.generation_logs) > self.max_log_generations * 10:
            min_gen = generation - self.max_log_generations
            self.generation_logs = [
                (g, line) for g, line in self.generation_logs if g >= min_gen
            ]
    
    def get_recent_logs(self, n_generations: Optional[int] = None) -> List[str]:
        """
        Get recent logs from last N generations.
        
        Args:
            n_generations: Number of generations to include. 
                          If None, uses self.population_window
        
        Returns:
            List of log lines
        """
        if n_generations is None:
            n_generations = self.population_window
        
        # Find logs from last N generations
        if not self.generation_logs:
            return []
        
        last_gen = max(g for g, _ in self.generation_logs)
        min_gen = max(0, last_gen - n_generations + 1)
        
        return [line for g, line in self.generation_logs if g >= min_gen]
    
    def clear_coach_log(self):
        """
        Clear coach log after sending to coach.
        
        This allows user to see how new logs accumulate before next analysis.
        Debug log is NOT cleared (kept for diagnostics).
        """
        lines_before = coach_log_manager.get_line_count()
        coach_log_manager.clear()
        debug_log_manager.append(
            f"✓ 🧹 Coach log cleared ({lines_before} lines sent to coach)"
        )
    
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
        coach_log_manager.append(
            f"[COACH  ] ❄️  Freezing population at Gen {population.generation}"
        )
        
        session = CoachAnalysisSession.from_population(
            population, fitness_config, ga_config
        )
        
        self.last_session = session
        self.session_history.append(session)
        
        # Show session details
        coach_log_manager.append(
            f"[COACH  ] 📸 Session {session.session_id}: "
            f"{session.population_size} individuals, "
            f"diversity={session.get_population_metrics()['diversity']:.2f}"
        )
        
        return session
    
    async def analyze_session_blocking(
        self,
        session: CoachAnalysisSession
    ) -> Optional[CoachAnalysis]:
        """
        Analyze a frozen population session (BLOCKING).
        
        Sends FULL population data to coach including:
        - All individual parameters
        - All individual metrics
        - Population statistics
        - Current GA configuration
        
        Coach provides recommendations including possible mutations.
        
        This is a BLOCKING call - waits for coach response.
        
        Args:
            session: Frozen session to analyze
        
        Returns:
            Coach analysis with recommendations
        """
        coach_log_manager.append(
            f"[COACH  ] 🤖 Requesting coach analysis for {session.session_label}"
        )
        
        # Initialize client if needed
        if not self.coach_client:
            self.coach_client = GemmaCoachClient(
                base_url=self.base_url,
                model=self.model,
                prompt_version=self.prompt_version,
                system_prompt=self.system_prompt,
                verbose=self.verbose,
                debug_payloads=self.debug_payloads
            )
            # Log initialization to debug (not user-facing) - use startup style
            debug_log_manager.append(f"✓ 🔌 LM Studio client initialized")
        
        try:
            # Send FULL population data to coach
            coach_log_manager.append(
                f"[COACH  ] 📤 Sending population data: "
                f"{session.population_size} individuals with full parameters"
            )
            
            # Estimate token count
            session_data = session.to_dict_for_coach()
            estimated_tokens = len(json.dumps(session_data)) // 4
            coach_log_manager.append(
                f"[COACH  ] 📊 Population data: ~{estimated_tokens:,} tokens"
            )
            if self.debug_payloads:
                payload_json = json.dumps(session_data, indent=2, default=str)
                logger.info("Coach session payload (Gen %s):\n%s", session.generation, payload_json)
                debug_log_manager.append(f"🧪 Coach payload size={len(payload_json)} chars")
            
            # Get recent logs for context
            recent_logs = self.get_recent_logs(n_generations=self.population_window)
            coach_log_manager.append(
                f"[COACH  ] 📜 Context logs: {len(recent_logs)} lines"
            )
            if self.debug_payloads and recent_logs:
                logger.info("Coach context logs (latest %s lines):\n%s", len(recent_logs), "\n".join(recent_logs))
            
            # BLOCKING: Wait for coach analysis
            coach_log_manager.append(f"[COACH  ] ⏳ Waiting for coach response...")
            coach_log_manager.append(f"[COACH  ] ⏱️  Timeout: {self.coach_client.timeout}s")
            
            # Analyze using LLM client with full session data
            # (Extended interface to include full population)
            analysis = await self.coach_client.analyze_evolution_with_session(
                session, recent_logs
            )
            
            if analysis:
                session.analysis = analysis
                session.analysis_timestamp = datetime.utcnow()
                
                coach_log_manager.append(
                    f"[COACH  ] ✅ Analysis complete: {analysis.overall_assessment}"
                )
                coach_log_manager.append(
                    f"[COACH  ] 📋 Recommendations: {len(analysis.recommendations)}"
                )
                
                if self.debug_payloads:
                    analysis_json = analysis.to_json()
                    logger.info("Coach analysis JSON (Gen %s):\n%s", session.generation, analysis_json)
                    debug_log_manager.append(f"🧪 Coach analysis JSON size={len(analysis_json)} chars")
                    if analysis.recommendations:
                        for idx, rec in enumerate(analysis.recommendations, start=1):
                            logger.info(
                                "Coach recommendation %s: param=%s value=%s -> %s, reasoning=%s",
                                idx,
                                rec.parameter,
                                rec.current_value,
                                rec.suggested_value,
                                rec.reasoning
                            )
                
                if analysis.recommendations:
                    # Group recommendations by category
                    mutations = [r for r in analysis.recommendations if r.parameter.startswith(("mutate_", "drop_", "insert_"))]
                    ga_params = [r for r in analysis.recommendations if r.category.value in ("ga_hyperparams", "diversity")]
                    fitness_gates = [r for r in analysis.recommendations if r.category.value == "fitness_gates"]
                    
                    for i, rec in enumerate(analysis.recommendations, 1):
                        coach_log_manager.append(
                            f"[COACH  ]   {i}. [{rec.category.value}] {rec.parameter}: {rec.reasoning[:60]}..."
                        )
                    
                    if mutations:
                        coach_log_manager.append(f"[COACH  ]   🧬 {len(mutations)} mutation(s) recommended")
                    if ga_params:
                        coach_log_manager.append(f"[COACH  ]   ⚙️  {len(ga_params)} GA param change(s)")
                    if fitness_gates:
                        coach_log_manager.append(f"[COACH  ]   🎯 {len(fitness_gates)} fitness gate change(s)")
                else:
                    # NEW: Explicit warning if no mutations returned
                    coach_log_manager.append(
                        f"[COACH  ] ⚠️  Coach analysis returned no recommendations "
                        f"(assessment: {analysis.overall_assessment})"
                    )
                
                # Mark first analysis done
                if not self._first_analysis_done:
                    self._first_analysis_done = True
                    coach_log_manager.append(
                        f"[COACH  ] 🎉 First analysis complete at Gen {session.generation}"
                    )
            else:
                # NEW: Explicit failure logging
                coach_log_manager.append(
                    f"[COACH  ] ❌ Analysis returned None (LLM may not have responded)"
                )
            
            return analysis
        
        except asyncio.TimeoutError:
            # NEW: Handle timeout specifically
            timeout_seconds = self.coach_client.timeout if hasattr(self.coach_client, 'timeout') else 120
            coach_log_manager.append(f"[COACH  ] ❌ TIMEOUT: LLM response exceeded {timeout_seconds}s")
            coach_log_manager.append(f"[COACH  ] → Check if LM Studio is running and responsive")
            coach_log_manager.append(f"[COACH  ] → Run: lms ps")
            logger.exception("Coach analysis timeout")
            return None
        
        except Exception as e:
            # NEW: Detailed error logging to user-facing log
            coach_log_manager.append(f"[COACH  ] ❌ Analysis error: {type(e).__name__}")
            coach_log_manager.append(f"[COACH  ] Details: {str(e)[:120]}")
            
            # NEW: Add traceback context for debugging
            import traceback
            tb_lines = traceback.format_exc().split('\n')
            # Find relevant lines (with coach/manager context)
            relevant = [l for l in tb_lines if l.strip() and any(x in l.lower() for x in ['coach', 'manager', 'llm', 'file'])]
            if relevant and len(relevant) > 1:
                # Log the file/line where error occurred
                coach_log_manager.append(f"[COACH  ] Context: {relevant[-2][:80]}")
            
            logger.exception("Coach analysis error")
            debug_log_manager.append(f"❌ Coach analysis error: {type(e).__name__}: {e}")
            return None
    
    async def apply_session_recommendations_blocking(
        self,
        population: Population,
        session: CoachAnalysisSession
    ) -> Dict[str, Any]:
        """
        Apply coach recommendations to population (BLOCKING).
        
        CRITICAL: Applies to the SAME population state that was analyzed.
        Supports mutations:
        - mutate_<id>_<param>: Mutate individual parameter
        - drop_<id>: Drop individual
        - insert_*: Insert new individual (if coach provides)
        
        Args:
            population: Population to mutate (should be same as frozen session)
            session: Frozen session with analysis results
        
        Returns:
            Mutation summary
        """
        if not session.analysis:
            return {"total_mutations": 0, "mutations_by_type": {}}
        
        coach_log_manager.append(
            f"[COACH  ] 🔧 Applying recommendations to {session.session_label}"
        )
        
        # Apply mutations using mutation manager
        summary = self.mutation_manager.apply_coach_recommendations(
            population, session, session.analysis
        )
        
        coach_log_manager.append(
            f"[COACH  ] ✅ Applied {summary['total_mutations']} mutations"
        )
        
        return summary
    
    async def analyze_and_apply_blocking(
        self,
        population: Population,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Full blocking workflow: Freeze → Analyze → Apply → Return.
        
        BLOCKING CALL: Returns after analysis and mutation application.
        
        Workflow:
        1. FREEZE population (create session)
        2. ANALYZE with full population data
        3. APPLY recommendations/mutations
        4. RETURN to caller (GA resumes with modified population)
        
        Args:
            population: Population to analyze and modify
            fitness_config: Fitness configuration
            ga_config: GA configuration
        
        Returns:
            (success, mutation_summary)
        """
        coach_log_manager.append(
            f"[COACH  ] 🔄 Starting blocking analysis workflow at Gen {population.generation}"
        )
        
        # Step 1: FREEZE population
        session = await self.create_analysis_session(
            population, fitness_config, ga_config
        )
        
        # Step 2: ANALYZE
        analysis = await self.analyze_session_blocking(session)
        
        if not analysis:
            coach_log_manager.append(f"[COACH  ] ❌ Analysis failed, skipping")
            return False, {}
        
        # Step 3: APPLY recommendations
        summary = await self.apply_session_recommendations_blocking(
            population, session
        )
        
        # Log mutation summary
        if summary.get("total_mutations", 0) > 0:
            coach_log_manager.append(
                f"[COACH  ] ✅ Applied {summary['total_mutations']} mutations: "
                f"{summary['mutations_by_type']}"
            )
            if summary.get("mutations_failed"):
                coach_log_manager.append(
                    f"[COACH  ] ⚠️  {len(summary['mutations_failed'])} mutations failed: "
                    f"{summary['mutations_failed']}"
                )
        else:
            coach_log_manager.append(
                f"[COACH  ] ℹ️  No mutations applied (0 mutation-type recommendations)"
            )
        
        # Step 4: Optionally reload model to clear context
        if self.auto_reload_model:
            coach_log_manager.append(
                f"[COACH  ] 🔄 Reloading model to clear context window"
            )
            try:
                await self.coach_client.reload_model()
                coach_log_manager.append(f"[COACH  ] ✅ Model reloaded")
            except Exception as e:
                logger.warning(f"Failed to reload model: {e}")
        
        coach_log_manager.append(
            f"[COACH  ] ✅ Blocking workflow complete - "
            f"GA will resume with modified population"
        )
        
        # IMPORTANT: Clear coach log after sending to coach
        # This allows user to see how new logs accumulate before next analysis
        self.clear_coach_log()
        
        return True, summary
    
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
        
        coach_log_manager.append(
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
        
        coach_log_manager.append(
            f"[AGENT  ] 🔧 Agent created with max_iterations={max_iterations}"
        )
        
        # Step 5: Build initial observation
        observation = self._build_agent_observation(session, fitness_config, ga_config)
        
        # Step 6: RUN agent analysis
        coach_log_manager.append(f"[AGENT  ] 🚀 Running agent analysis...")
        
        try:
            result = await agent.run_analysis(observation)
            
            if result.get("success"):
                coach_log_manager.append(
                    f"[AGENT  ] ✅ Agent completed: "
                    f"{result['iterations']} iterations, "
                    f"{result['tool_calls_count']} tool calls"
                )
                
                # Log actions taken
                if toolkit.actions_log:
                    coach_log_manager.append(
                        f"[AGENT  ] 📋 Actions taken: {len(toolkit.actions_log)}"
                    )
                    for i, action in enumerate(toolkit.actions_log[:5], 1):
                        action_name = action.get("action", "unknown")
                        coach_log_manager.append(f"[AGENT  ]   {i}. {action_name}")
                    
                    if len(toolkit.actions_log) > 5:
                        coach_log_manager.append(
                            f"[AGENT  ]   ... and {len(toolkit.actions_log) - 5} more actions"
                        )
                else:
                    coach_log_manager.append("[AGENT  ] ⚠️  No actions logged")
                
                # Store toolkit for later inspection
                self.last_toolkit = toolkit
                
                # Step 7: Optionally reload model to clear context
                if self.auto_reload_model:
                    coach_log_manager.append(
                        f"[AGENT  ] 🔄 Reloading model to clear context window"
                    )
                    try:
                        await self.coach_client.reload_model()
                        coach_log_manager.append(f"[AGENT  ] ✅ Model reloaded")
                    except Exception as e:
                        logger.warning(f"Failed to reload model: {e}")
                
                coach_log_manager.append(
                    f"[AGENT  ] ✅ Agent workflow complete - "
                    f"GA will resume with modified population"
                )
                
                # Clear coach log
                self.clear_coach_log()
                
                return True, {
                    "total_actions": len(toolkit.actions_log),
                    "iterations": result['iterations'],
                    "tool_calls": result['tool_calls_count'],
                    "actions_log": toolkit.actions_log
                }
            else:
                error = result.get("error", "Unknown error")
                coach_log_manager.append(f"[AGENT  ] ❌ Agent failed: {error}")
                return False, {"error": error}
                
        except Exception as e:
            logger.exception("Agent analysis failed: %s", e)
            coach_log_manager.append(f"[AGENT  ] ❌ Exception: {str(e)}")
            debug_log_manager.append(f"❌ Agent exception: {e}")
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

Your task: Analyze this population state and take actions to improve evolution.
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
            debug_log_manager.append(f"✓ 🔌 LM Studio agent client initialized")
    
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
    print("Blocking Evolution Coach Manager")
    print("\nKey Features:")
    print("  - BLOCKING optimization (pauses during coach analysis)")
    print("  - Full population data sent to coach")
    print("  - Direct individual mutations")
    print("  - Frozen session tracking")
    print("  - Accurate recommendation application")
    print("\nUsage:")
    print("  manager = BlockingCoachManager()")
    print("  if manager.should_analyze(gen):")
    print("      success, summary = await manager.analyze_and_apply_blocking(pop, fitness, ga)")
    print("  # GA continues from modified population")
