"""
Evolution Coach Manager

Handles async coach orchestration:
- Non-blocking coach analysis (doesn't interrupt GA)
- Model loading/unloading for memory management
- Recommendation queuing and automatic application
- Application tracking for coach feedback loop
"""

import asyncio
from typing import Optional, Tuple, List
from datetime import datetime
from pathlib import Path
import logging

from backtest.coach_protocol import (
    CoachAnalysis, RecommendationApplication, EvolutionState
)
from backtest.coach_integration import (
    build_evolution_state, apply_coach_recommendations, format_coach_output
)
from backtest.llm_coach import GemmaCoachClient
from backtest.optimizer import Population
from core.models import FitnessConfig, OptimizationConfig
from core.coach_logging import coach_log_manager

logger = logging.getLogger(__name__)


class CoachManager:
    """
    Manages Evolution Coach lifecycle and recommendation application.
    
    Features:
    - Async coach analysis (non-blocking)
    - Model loading/unloading
    - Log trimming to last N generations
    - Application tracking
    - Recommendation queue
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: Optional[str] = None,
        prompt_version: Optional[str] = None,
        system_prompt: Optional[str] = None,
        first_analysis_generation: Optional[int] = None,
        max_log_generations: Optional[int] = None,
        auto_apply: bool = True,
        auto_reload_model: Optional[bool] = None,
        verbose: bool = True
    ):
        """
        Initialize Coach Manager.
        
        CRITICAL WORKFLOW:
        1. First analysis at generation N (first_analysis_generation)
        2. Apply recommendations → Unload/Reload model → Trigger next analysis
        3. Repeat: Analysis happens ONLY after model reload (not on fixed interval)
        
        Args:
            base_url: LM Studio endpoint
            model: Model name (loaded from settings if None)
            prompt_version: Coach prompt version (loaded from settings if None)
            system_prompt: System prompt version (loaded from settings if None)
            first_analysis_generation: Generation to trigger first analysis (loaded from settings if None)
            max_log_generations: Keep last N generations in logs (loaded from settings if None)
            auto_apply: Automatically apply recommendations
            auto_reload_model: Auto unload/reload model after recommendations (loaded from settings if None)
            verbose: Print detailed logs
        """
        # Load from settings if not provided
        from config.settings import settings
        
        self.base_url = base_url
        self.model = model or settings.coach_model
        self.prompt_version = prompt_version or settings.coach_prompt_version
        self.system_prompt = system_prompt or getattr(settings, 'coach_system_prompt', settings.coach_prompt_version)
        self.first_analysis_generation = first_analysis_generation or settings.coach_first_analysis_generation
        self.max_log_generations = max_log_generations or settings.coach_max_log_generations
        self.auto_apply = auto_apply
        self.auto_reload_model = auto_reload_model if auto_reload_model is not None else settings.coach_auto_reload_model
        self.verbose = verbose
        
        coach_log_manager.append(f"[COACH  ] 📋 Manager initialized: system_prompt={self.system_prompt}")
        
        # State tracking
        self.coach_client: Optional[GemmaCoachClient] = None
        self.last_analysis: Optional[CoachAnalysis] = None
        self.application_history: List[RecommendationApplication] = []
        self.pending_analysis: Optional[asyncio.Task] = None
        self._first_analysis_done: bool = False
        self._trigger_analysis_next: bool = False  # Flag to trigger analysis after model reload
        
        # Log collection
        self.generation_logs: List[Tuple[int, str]] = []  # (gen, log_line)
    
    def should_analyze(self, generation: int) -> bool:
        """
        Check if we should trigger coach analysis this generation.
        
        CRITICAL LOGIC:
        - First analysis: at first_analysis_generation (e.g., Gen 10)
        - Subsequent analyses: ONLY when _trigger_analysis_next flag is set
          (this flag is set after model reload following recommendation application)
        """
        if not self._first_analysis_done:
            # First analysis at specified generation
            return generation >= self.first_analysis_generation
        else:
            # Subsequent analyses: only when flag is set (after model reload)
            if self._trigger_analysis_next:
                self._trigger_analysis_next = False  # Reset flag
                return True
            return False
    
    def add_log(self, generation: int, log_line: str):
        """
        Add log line for this generation.
        
        Automatically trims to last N generations.
        """
        # Add to internal logs
        self.generation_logs.append((generation, log_line))
        
        # Also send to coach log window
        coach_log_manager.append(f"[Gen {generation:3d}] {log_line}")
        
        # Trim to last max_log_generations
        before_trim = len(self.generation_logs)
        if before_trim > self.max_log_generations * 10:  # ~10 lines per gen
            min_gen = generation - self.max_log_generations
            self.generation_logs = [
                (g, line) for g, line in self.generation_logs if g >= min_gen
            ]
            after_trim = len(self.generation_logs)
            if self.verbose and after_trim < before_trim:
                trimmed_count = before_trim - after_trim
                coach_log_manager.append(f"[TRIM   ] Trimmed {trimmed_count} old log lines (keeping last {self.max_log_generations} gens)")
    
    def get_recent_logs(self, n_lines: int = 100) -> List[str]:
        """Get last N log lines."""
        return [line for _, line in self.generation_logs[-n_lines:]]
    
    def format_application_history(self) -> str:
        """Format application history for coach prompt."""
        if not self.application_history:
            return "No previous recommendations applied yet."
        
        lines = []
        for app in self.application_history[-3:]:  # Last 3 applications
            lines.append(
                f"Applied recommendations at Gen {app.generation}: "
                f"{app.applied_count} changes ({', '.join(app.recommendations)})"
            )
        return "\n".join(lines)
    
    async def analyze_async(
        self,
        population: Population,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig
    ) -> Optional[CoachAnalysis]:
        """
        Analyze population asynchronously (non-blocking).
        
        Returns immediately, analysis runs in background.
        Use wait_for_analysis() to get result.
        """
        if self.pending_analysis and not self.pending_analysis.done():
            if self.verbose:
                print("⏳ Coach analysis already in progress, skipping...")
            return None
        
        # Log analysis trigger
        coach_log_manager.append(f"[COACH  ] 🤖 Triggering coach analysis for Gen {population.generation}")
        coach_log_manager.append(f"[COACH  ] Sending last {self.max_log_generations} generations to coach")
        
        # Start async analysis task
        self.pending_analysis = asyncio.create_task(
            self._run_analysis(population, fitness_config, ga_config)
        )
        
        return None
    
    async def _run_analysis(
        self,
        population: Population,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig
    ) -> Optional[CoachAnalysis]:
        """Internal: Run coach analysis."""
        try:
            # Build evolution state with application history
            state = build_evolution_state(population, fitness_config, ga_config)
            state.recent_applications = self.application_history[-3:]  # Last 3
            
            # Get recent logs
            recent_logs = self.get_recent_logs(n_lines=100)
            
            # Add application history to logs
            app_history = self.format_application_history()
            recent_logs.insert(0, f"=== Application History ===\n{app_history}\n")
            
            # Log what we're sending to coach
            coach_log_manager.append(f"[COACH  ] Sending evolution state:")
            coach_log_manager.append(f"[COACH  ]   - Mean fitness: {state.mean_fitness:.4f} ± {state.std_fitness:.4f}")
            coach_log_manager.append(f"[COACH  ]   - Best fitness: {state.best_fitness:.4f}")
            coach_log_manager.append(f"[COACH  ]   - Below min_trades: {state.below_min_trades_percent:.1f}%")
            coach_log_manager.append(f"[COACH  ]   - Diversity: {state.diversity_metric:.2f}")
            coach_log_manager.append(f"[COACH  ]   - Stagnant: {state.is_stagnant}")
            coach_log_manager.append(f"[COACH  ]   - Log lines: {len(recent_logs)}")
            
            # Estimate token count (rough: ~4 chars per token)
            total_chars = len(str(state.__dict__)) + sum(len(log) for log in recent_logs)
            estimated_tokens = total_chars // 4
            coach_log_manager.append(f"[COACH  ]   - Estimated input tokens: ~{estimated_tokens:,}")
            
            # Log current parameters
            coach_log_manager.append(f"[PARAMS ] Current fitness config:")
            coach_log_manager.append(f"[PARAMS ]   - fitness_type: {state.fitness_config_dict.get('fitness_function_type', 'N/A')}")
            coach_log_manager.append(f"[PARAMS ]   - min_trades: {state.fitness_config_dict.get('min_trades', 'N/A')}")
            coach_log_manager.append(f"[PARAMS ]   - min_win_rate: {state.fitness_config_dict.get('min_win_rate', 'N/A')}")
            
            if self.verbose:
                print(f"📊 Coach analyzing Gen {state.generation} (last {len(recent_logs)} log lines)...")
            
            # Initialize coach client if needed
            if not self.coach_client:
                self.coach_client = GemmaCoachClient(
                    base_url=self.base_url,
                    model=self.model,
                    prompt_version=self.prompt_version,
                    system_prompt=self.system_prompt,
                    verbose=self.verbose
                )
                coach_log_manager.append(f"[COACH  ] ✅ Client initialized with system_prompt={self.system_prompt}")
            
            # Analyze
            coach_log_manager.append(f"[COACH  ] ⏳ Waiting for coach response...")
            analysis = await self.coach_client.analyze_evolution(state, recent_logs)
            
            if analysis:
                self.last_analysis = analysis
                
                # Log response
                coach_log_manager.append(f"[COACH  ] ✅ Coach response received:")
                coach_log_manager.append(f"[COACH  ]   - Assessment: {analysis.overall_assessment.upper()}")
                coach_log_manager.append(f"[COACH  ]   - Recommendations: {len(analysis.recommendations)}")
                coach_log_manager.append(f"[COACH  ]   - Stagnation: {'Yes' if analysis.stagnation_detected else 'No'}")
                coach_log_manager.append(f"[COACH  ]   - Diversity concern: {'Yes' if analysis.diversity_concern else 'No'}")
                
                # Log each recommendation
                for i, rec in enumerate(analysis.recommendations, 1):
                    coach_log_manager.append(
                        f"[COACH  ]   {i}. {rec.parameter}: {rec.current_value} → {rec.suggested_value} "
                        f"(confidence: {rec.confidence:.0%})"
                    )
                
                # Mark first analysis as done
                if not self._first_analysis_done:
                    self._first_analysis_done = True
                    coach_log_manager.append(f"[COACH  ] ✅ First coach analysis complete at Gen {state.generation}")
                    if self.verbose:
                        print(f"✅ First coach analysis complete at Gen {state.generation}")
                
                if self.verbose:
                    print(format_coach_output(analysis))
            
            return analysis
        
        except Exception as e:
            logger.exception("Coach analysis error")
            print(f"❌ Coach analysis failed: {e}")
            return None
    
    async def wait_for_analysis(self) -> Optional[CoachAnalysis]:
        """Wait for pending analysis to complete and return result."""
        if not self.pending_analysis:
            return self.last_analysis
        
        try:
            analysis = await self.pending_analysis
            return analysis
        except Exception as e:
            logger.exception("Error waiting for coach analysis")
            return None
    
    async def reload_model(self):
        """
        Unload and reload model to clear context window.
        
        CRITICAL: This clears the context window AND triggers next analysis.
        After reload, should_analyze() will return True on next generation.
        """
        if not self.coach_client:
            return
        
        if self.verbose:
            print("🔄 Reloading model to clear context window...")
        
        try:
            # Unload
            await self.coach_client.unload_model()
            
            # Small delay to ensure clean unload
            await asyncio.sleep(0.5)
            
            # Reload
            await self.coach_client.load_model()
            
            # CRITICAL: Set flag to trigger next analysis
            self._trigger_analysis_next = True
            
            if self.verbose:
                print("✅ Model reloaded - context window cleared")
                print("   Next analysis will be triggered after model reload")
        
        except Exception as e:
            logger.exception("Error reloading model")
            print(f"⚠️  Failed to reload model: {e}")
    
    def apply_recommendations(
        self,
        analysis: CoachAnalysis,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig
    ) -> Tuple[FitnessConfig, OptimizationConfig]:
        """
        Apply coach recommendations and track application.
        
        Returns new configs with recommendations applied.
        
        Note: Model reload (if enabled) should be done separately after this.
        """
        if not analysis or not analysis.recommendations:
            return fitness_config, ga_config
        
        # Apply recommendations
        new_fitness, new_ga = apply_coach_recommendations(
            analysis, fitness_config, ga_config
        )
        
        # Track application
        applied_params = [rec.parameter for rec in analysis.recommendations]
        application = RecommendationApplication(
            generation=analysis.generation,
            applied_count=len(analysis.recommendations),
            recommendations=applied_params,
            timestamp=datetime.utcnow().isoformat()
        )
        self.application_history.append(application)
        
        # Log application
        log_msg = (
            f"Applied Coach recommendations at Gen {analysis.generation}: "
            f"{len(analysis.recommendations)} changes"
        )
        self.add_log(analysis.generation, log_msg)
        print(f"✅ {log_msg}")
        
        return new_fitness, new_ga
    
    async def analyze_and_apply(
        self,
        population: Population,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig
    ) -> Tuple[FitnessConfig, OptimizationConfig]:
        """
        Convenience method: Analyze, apply, and reload model.
        
        Workflow:
        1. Analyze evolution state (blocks until complete)
        2. Apply recommendations if auto_apply=True
        3. Reload model if auto_reload_model=True (clears context window)
        
        Returns updated configs.
        """
        # Start analysis
        await self.analyze_async(population, fitness_config, ga_config)
        
        # Wait for result
        analysis = await self.wait_for_analysis()
        
        if not analysis:
            return fitness_config, ga_config
        
        updated_fitness = fitness_config
        updated_ga = ga_config
        
        # Apply if enabled
        if self.auto_apply and analysis.recommendations:
            updated_fitness, updated_ga = self.apply_recommendations(
                analysis, fitness_config, ga_config
            )
        
        # Reload model to clear context window (CRITICAL)
        if self.auto_reload_model:
            await self.reload_model()
        
        return updated_fitness, updated_ga
    
    def unload_model(self):
        """
        Unload model to free memory.
        
        Note: With LM Studio, model stays loaded in server.
        This just disconnects the client.
        """
        if self.coach_client:
            self.coach_client = None
            if self.verbose:
                print("🔌 Coach client disconnected")
    
    def get_stats(self) -> dict:
        """Get coach manager statistics."""
        return {
            "total_analyses": len(self.application_history),
            "total_recommendations_applied": sum(
                app.applied_count for app in self.application_history
            ),
            "last_analysis_generation": (
                self.last_analysis.generation if self.last_analysis else None
            ),
            "pending_analysis": self.pending_analysis is not None and not self.pending_analysis.done(),
            "log_lines": len(self.generation_logs),
        }


# Example usage
if __name__ == "__main__":
    print("Evolution Coach Manager")
    print("\nFeatures:")
    print("  - Async non-blocking coach analysis")
    print("  - Model loading/unloading")
    print("  - Log trimming to last N generations")
    print("  - Application tracking")
    print("  - Automatic recommendation application")
    print("\nUsage:")
    print("  manager = CoachManager()")
    print("  # In GA loop:")
    print("  if manager.should_analyze(gen):")
    print("      await manager.analyze_async(pop, fitness, ga)")
    print("  analysis = await manager.wait_for_analysis()")
    print("  if analysis:")
    print("      fitness, ga = manager.apply_recommendations(analysis, fitness, ga)")
