"""
Gemma 3 Evolution Coach LLM Client

Handles async communication with local Gemma 3 model via LM Studio.
Uses LM Studio CLI (lms) for model loading/unloading.
Uses lmstudio Python SDK for inference.
Robust JSON parsing and error handling.
"""

import asyncio
import json
import re
import subprocess
from typing import Optional, List
from datetime import datetime
import logging

try:
    import lmstudio as lms
    HAS_LMSTUDIO = True
except ImportError:
    HAS_LMSTUDIO = False

from backtest.coach_protocol import (
    CoachAnalysis, CoachRecommendation, RecommendationCategory,
    EvolutionState, load_coach_prompt
)
from core.coach_logging import coach_log_manager, debug_log_manager

logger = logging.getLogger(__name__)


class GemmaCoachClient:
    """
    LLM client for Gemma 3 12B Evolution Coach.
    
    Connects to local LM Studio instance and sends evolution logs for analysis.
    Uses LM Studio Python SDK for model loading/unloading.
    Automatically parses JSON recommendations and applies them to GA.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "google/gemma-3-12b",
        timeout: Optional[float] = None,
        temperature: float = 0.3,
        prompt_version: str = "async_coach_v1",
        system_prompt: Optional[str] = None,
        context_length: int = 5000,  # Experimental: enough for 25 gens?
        gpu: float = 0.6,  # GPU offload ratio 0.0-1.0
        verbose: bool = True,
        debug_payloads: bool = False
    ):
        """
        Initialize Gemma Coach client.
        
        Args:
            base_url: LM Studio API endpoint (default localhost:1234)
            model: Model identifier (e.g., "google/gemma-3-12b")
            timeout: Request timeout in seconds
            temperature: LLM temperature (0.3 = deterministic)
            prompt_version: Coach prompt version to load
            system_prompt: Optional system prompt version (if None, uses prompt_version)
            context_length: Model context window size (default: 5000 tokens)
            gpu: GPU offload ratio 0.0-1.0 (default: 0.6 = 60% GPU)
            verbose: Print debug info
            debug_payloads: If True, log full request/response payloads
        """
        if not HAS_LMSTUDIO:
            raise ImportError("lmstudio package required. Install with: pip install lmstudio")
        
        # Load timeout from settings if not provided
        if timeout is None:
            from config.settings import settings
            timeout = getattr(settings, 'coach_response_timeout', 3600)
        
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.context_length = context_length
        self.gpu = gpu
        self.verbose = verbose
        self.debug_payloads = debug_payloads
        
        # Load system prompt - use system_prompt if provided, otherwise use prompt_version
        prompt_key = system_prompt if system_prompt else prompt_version
        self.system_prompt = load_coach_prompt(prompt_key)
        # Log to debug (not user-facing coach log) - use startup style
        debug_log_manager.append(f"✓ 📋 Loaded coach prompt: {prompt_key}")
        if self.debug_payloads:
            logger.info("GemmaCoachClient debug payload logging ENABLED (full LLM traffic will be logged)")
            debug_log_manager.append("✓ 🧪 GemmaCoachClient payload logging enabled")
        
        # Track if model is loaded
        self._model_loaded = False
        self._lms_client = None
    
    async def analyze_evolution(
        self,
        evolution_state: EvolutionState,
        raw_logs: Optional[List[str]] = None
    ) -> Optional[CoachAnalysis]:
        """
        Send evolution state to Gemma Coach for analysis.
        
        Args:
            evolution_state: Current GA state
            raw_logs: Optional raw log lines for context
        
        Returns:
            CoachAnalysis with recommendations, or None if error
        """
        try:
            # Build user message
            user_message = self._build_user_message(evolution_state, raw_logs)
            
            if self.verbose:
                print("📊 Sending evolution state to Gemma Coach...")
                print(f"   Generation: {evolution_state.generation}")
                print(f"   Population fitness: {evolution_state.mean_fitness:.4f} ± {evolution_state.std_fitness:.4f}")
            
            # Call LLM
            response_text = await self._call_llm(user_message)
            
            if not response_text:
                print("❌ Empty response from coach")
                return None
            
            # Parse JSON from response
            analysis = self._parse_response(response_text, evolution_state.generation)
            
            if self.verbose and analysis:
                print(f"✅ Coach analysis complete: {len(analysis.recommendations)} recommendations")
                if analysis.stagnation_detected:
                    print("   ⚠️  Stagnation detected")
                if analysis.diversity_concern:
                    print("   ⚠️  Diversity concern flagged")
            
            return analysis
        
        except Exception as e:
            print(f"❌ Coach error: {e}")
            logger.exception("Evolution Coach error")
            return None
    
    async def analyze_evolution_with_session(
        self,
        session,  # CoachAnalysisSession
        raw_logs: Optional[List[str]] = None
    ) -> Optional[CoachAnalysis]:
        """
        NEW: Send frozen population session to Gemma Coach for detailed analysis.
        
        This is for BLOCKING coach semantics:
        - Send FULL population data (all individuals + parameters)
        - Coach can recommend direct mutations
        - Recommendations apply to exact same population state
        
        Args:
            session: CoachAnalysisSession with full population snapshot
            raw_logs: Optional raw log lines for context
        
        Returns:
            CoachAnalysis with recommendations (may include mutations)
        """
        try:
            # Build user message with full population data
            session_dict = session.to_dict_for_coach()
            user_message = self._build_session_message(session_dict, raw_logs)
            
            if self.verbose:
                print("📊 Sending frozen population session to Coach...")
                print(f"   Session: {session.session_id}")
                print(f"   Generation: {session.generation}")
                print(f"   Population size: {session.population_size}")
                print(f"   Diversity: {session.get_population_metrics()['diversity']:.2f}")
            
            # Call LLM
            response_text = await self._call_llm(user_message)
            
            if not response_text:
                print("❌ Empty response from coach")
                return None
            
            # Parse JSON from response
            analysis = self._parse_response(response_text, session.generation)
            
            if self.verbose and analysis:
                print(f"✅ Coach analysis complete: {len(analysis.recommendations)} recommendations")
                # Count mutation types
                mutations = [r for r in analysis.recommendations if "mutate" in r.parameter or r.parameter.startswith("drop")]
                if mutations:
                    print(f"   🧬 {len(mutations)} mutations recommended")
                if analysis.stagnation_detected:
                    print("   ⚠️  Stagnation detected")
                if analysis.diversity_concern:
                    print("   ⚠️  Diversity concern flagged")
            
            return analysis
        
        except Exception as e:
            print(f"❌ Coach error: {e}")
            logger.exception("Evolution Coach error")
            return None
    
    def _build_session_message(self, session_dict: dict, raw_logs: Optional[List[str]] = None) -> str:
        """Build user message from session data."""
        import json
        
        # Start with session overview
        message = f"""You are analyzing a frozen population state for the Evolution Coach.

CURRENT STATE:
- Session ID: {session_dict.get('session_id')}
- Generation: {session_dict.get('generation')}
- Population Size: {session_dict.get('population_size')}

POPULATION METRICS:
"""
        metrics = session_dict.get('population_metrics', {})
        message += f"- Mean Fitness: {metrics.get('mean_fitness', 0):.4f} ± {metrics.get('std_fitness', 0):.4f}\n"
        message += f"- Best Fitness: {metrics.get('best_fitness', 0):.4f}\n"
        message += f"- Diversity: {metrics.get('diversity', 0):.2f} (0=homogeneous, 1=diverse)\n"
        message += f"- Mean Trades: {metrics.get('mean_trades', 0):.1f} ± {metrics.get('std_trades', 0):.1f}\n"
        
        # Include full population data
        message += f"\nPOPULATION DATA (all {len(session_dict.get('individuals', []))} individuals):\n"
        individuals = session_dict.get('individuals', [])[:50]  # Limit for token budget
        for ind in individuals:
            message += f"\nIndividual #{ind['id']}:\n"
            message += f"  Fitness: {ind.get('fitness', 0):.4f}\n"
            message += f"  Metrics: {json.dumps(ind.get('metrics', {}), indent=2)}\n"
            message += f"  Parameters (sample): {json.dumps({k: ind['parameters'].get(k) for k in list(ind['parameters'].keys())[:5]}, indent=2)}\n"
        
        if len(individuals) < len(session_dict.get('individuals', [])):
            message += f"\n... and {len(session_dict.get('individuals', [])) - len(individuals)} more individuals\n"
        
        # Add recent logs
        if raw_logs:
            message += f"\n\nRECENT LOG HISTORY ({len(raw_logs)} lines):\n"
            message += "\n".join(raw_logs[-50:])  # Last 50 lines
        
        message += "\n\nProvide recommendations in JSON format."
        
        return message
    
    async def check_model_loaded(self) -> bool:
        """
        Check if the model is already loaded using 'lms ps'.
        
        Returns:
            True if model is currently loaded, False otherwise
        """
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["lms", "ps"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                # Look for the model name in the output
                if self.model in output:
                    self._model_loaded = True
                    return True
                else:
                    self._model_loaded = False
                    return False
            else:
                coach_log_manager.append(f"[LMS    ] ⚠️  Failed to check model status")
                return False
        
        except Exception as e:
            logger.exception("Error checking model status")
            coach_log_manager.append(f"[LMS    ] ⚠️  Error checking model: {e}")
            return False
    
    async def load_model(self):
        """Load model using lms CLI."""
        # First check if model is already loaded
        already_loaded = await self.check_model_loaded()
        if already_loaded:
            coach_log_manager.append(f"[LMS    ] ✅ Model already loaded")
            return
        
        try:
            # Build lms load command
            cmd = ["lms", "load", self.model, f"--gpu={self.gpu}", f"--context-length={self.context_length}"]
            
            # Execute lms load command
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self._model_loaded = True
                coach_log_manager.append(f"[LMS    ] ✅ Model loaded successfully")
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                coach_log_manager.append(f"[LMS    ] ❌ Failed to load model: {error_msg}")
                raise RuntimeError(f"lms load failed: {error_msg}")
        
        except subprocess.TimeoutExpired:
            coach_log_manager.append(f"[LMS    ] ❌ Model loading timed out (120s)")
            raise RuntimeError("Model loading timed out")
        except FileNotFoundError:
            coach_log_manager.append(f"[LMS    ] ❌ 'lms' command not found - is LM Studio installed?")
            raise RuntimeError("'lms' command not found. Make sure LM Studio CLI is installed and in PATH.")
        except Exception as e:
            logger.exception("Model loading error")
            coach_log_manager.append(f"[LMS    ] ❌ Error: {e}")
            raise
    
    async def unload_model(self):
        """
        Unload model using lms CLI to free context window.
        
        NOTE: Does NOT clear _lms_client because the lmstudio SDK's default
        client is a singleton that can only be created once. We keep the client
        alive and just reload the model on the LM Studio side.
        """
        if not self._model_loaded:
            if self.verbose:
                print("Model not loaded, nothing to unload")
            return
        
        try:
            # Execute lms unload command
            result = await asyncio.to_thread(
                subprocess.run,
                ["lms", "unload"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self._model_loaded = False
                coach_log_manager.append(f"[LMS    ] ✅ Model unloaded successfully")
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                coach_log_manager.append(f"[LMS    ] ⚠️  Unload error: {error_msg}")
                # Still mark as unloaded
                self._model_loaded = False
        
        except Exception as e:
            logger.exception("Model unloading error")
            coach_log_manager.append(f"[LMS    ] ⚠️  Error: {e}")
            # Still mark as unloaded
            self._model_loaded = False
    
    async def _call_llm(self, user_message: str) -> Optional[str]:
        """
        Call Gemma 3 model via LM Studio Python SDK.
        
        Args:
            user_message: Formatted user message
        
        Returns:
            Response text from model
        """
        try:
            if self.debug_payloads:
                logger.info("=" * 80)
                logger.info("LLM REQUEST (Full Prompt + Payload)")
                logger.info("=" * 80)
                logger.info("SYSTEM PROMPT (%s chars):\n%s", len(self.system_prompt), self.system_prompt)
                logger.info("-" * 80)
                logger.info("USER MESSAGE (%s chars):\n%s", len(user_message), user_message)
                logger.info("=" * 80)
                debug_log_manager.append(f"🧪 LLM request: system={len(self.system_prompt)} chars, user={len(user_message)} chars")
            
            # Ensure model is loaded
            await self.load_model()
            
            coach_log_manager.append(f"[COACH  ] 📤 Sending {len(user_message)} chars to LLM")
            
            # Get LM Studio client (singleton - only created once)
            # IMPORTANT: SDK's get_default_client() can only be called ONCE
            # After first call, it returns the same client instance
            # NOTE: Don't pass self.base_url - SDK finds localhost:1234 automatically
            if not self._lms_client:
                try:
                    self._lms_client = await asyncio.to_thread(
                        lms.get_default_client
                    )
                except Exception as e:
                    if "Default client is already created" in str(e):
                        # Client already exists, get it without args
                        self._lms_client = await asyncio.to_thread(
                            lms.get_default_client
                        )
                    else:
                        raise
            
            # Get the loaded model handle
            model = await asyncio.to_thread(self._lms_client.llm.model)
            
            # Create chat with system prompt
            chat = lms.Chat(self.system_prompt)
            chat.add_user_message(user_message)
            
            # Generate response WITH TIMEOUT
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        model.respond,
                        chat,
                        config={
                            "temperature": self.temperature,
                            "maxTokens": 4000,  # Enough for structured response
                        }
                    ),
                    timeout=self.timeout
                )
                
                # Extract text content
                response_text = response.content
                
                coach_log_manager.append(f"[COACH  ] 📥 Received response from LLM ({len(response_text or '') or 0} chars)")
                if self.debug_payloads:
                    logger.info("=" * 80)
                    logger.info("LLM RESPONSE (Full Payload)")
                    logger.info("=" * 80)
                    logger.info("RESPONSE TEXT (%s chars):\n%s", len(response_text or ""), response_text)
                    logger.info("=" * 80)
                    debug_log_manager.append(f"🧪 LLM response size={len(response_text or '')} chars")
                
                return response_text
            
            except asyncio.TimeoutError:
                # NEW: Handle timeout specifically with detailed logging
                coach_log_manager.append(f"[COACH  ] ⏱️  LLM timeout after {self.timeout}s")
                coach_log_manager.append(f"[COACH  ] → Check if LM Studio is responsive: lms ps")
                raise TimeoutError(
                    f"LLM response timeout after {self.timeout}s. "
                    f"Check if LM Studio server is running and responsive."
                )
        
        except asyncio.TimeoutError:
            # Re-raise timeout for parent handler
            logger.exception("LLM call timeout")
            raise
        
        except Exception as e:
            logger.exception("LLM call error")
            coach_log_manager.append(f"[COACH  ] ❌ LLM error: {type(e).__name__}: {str(e)[:80]}")
            return None
    
    def _build_user_message(
        self,
        evolution_state: EvolutionState,
        raw_logs: Optional[List[str]] = None
    ) -> str:
        """Build formatted user message for Coach."""
        message = f"""
Analyze this evolution state and recommend parameter adjustments:

GENERATION: {evolution_state.generation}
TIMESTAMP: {datetime.now().isoformat()}

POPULATION STATISTICS:
  Mean Fitness: {evolution_state.mean_fitness:.4f} ± {evolution_state.std_fitness:.4f}
  Best Fitness: {evolution_state.best_fitness:.4f}
  Best Individual: {evolution_state.best_trades} trades, WR {evolution_state.best_win_rate:.1%}, Avg R {evolution_state.best_avg_r:.2f}, PnL ${evolution_state.best_pnl:.4f}
  Below min_trades: {evolution_state.below_min_trades_percent:.1f}% of population
  Mean trade count: {evolution_state.mean_trade_count:.1f}
  Population diversity: {evolution_state.diversity_metric:.2f} (0=homogeneous, 1=diverse)
  Stagnant: {evolution_state.is_stagnant}

CURRENT CONFIGURATION:
Fitness:
{self._format_dict(evolution_state.fitness_config_dict, indent=2)}

GA:
{self._format_dict(evolution_state.ga_config_dict, indent=2)}
"""
        
        if raw_logs:
            message += f"\nRECENT LOGS (last {len(raw_logs)} lines):\n"
            message += "\n".join(raw_logs[-20:])
        
        message += "\n\nProvide your analysis and recommendations in JSON format."
        return message
    
    @staticmethod
    def _format_dict(d: dict, indent: int = 0) -> str:
        """Format dictionary for display."""
        lines = []
        prefix = " " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{prefix}{k}:")
                lines.append(GemmaCoachClient._format_dict(v, indent + 2))
            elif isinstance(v, (list, tuple)):
                lines.append(f"{prefix}{k}: {v}")
            else:
                lines.append(f"{prefix}{k}: {v}")
        return "\n".join(lines)
    
    def _parse_response(self, response_text: str, generation: int) -> Optional[CoachAnalysis]:
        """
        Parse JSON from LLM response.
        
        Handles cases where LLM includes extra text before/after JSON.
        """
        # NEW: Validate response is not empty
        if not response_text or not response_text.strip():
            coach_log_manager.append(f"[COACH  ] ❌ Empty response from LLM")
            return None
        
        # Try to extract JSON object from response
        json_str = self._extract_json(response_text)
        
        if not json_str:
            # NEW: More detailed error logging
            coach_log_manager.append(f"[COACH  ] ❌ No JSON found in response")
            coach_log_manager.append(f"[COACH  ] Response start: {response_text[:100]}...")
            if self.verbose:
                logger.error(f"Full LLM response (no JSON found):\n{response_text}")
            return None
        
        try:
            # NEW: Log parsing attempt
            coach_log_manager.append(f"[COACH  ] 🔍 Parsing JSON ({len(json_str)} chars)")
            
            data = json.loads(json_str)
            analysis = CoachAnalysis.from_dict(data)
            
            # NEW: Validate that analysis has meaningful content
            if not analysis.recommendations and not analysis.summary:
                coach_log_manager.append(
                    f"[COACH  ] ⚠️  Empty analysis (no recommendations, no summary)"
                )
            
            return analysis
        
        except json.JSONDecodeError as e:
            # NEW: Detailed JSON error logging
            coach_log_manager.append(f"[COACH  ] ❌ JSON parse error at position {e.pos}: {e.msg}")
            coach_log_manager.append(f"[COACH  ] Invalid JSON: {json_str[:80]}...")
            if self.verbose:
                logger.error(f"JSON parse error: {e}\nFull JSON:\n{json_str}")
            return None
        
        except Exception as e:
            # NEW: Catch CoachAnalysis validation errors
            coach_log_manager.append(f"[COACH  ] ❌ Validation error: {type(e).__name__}")
            coach_log_manager.append(f"[COACH  ] Details: {str(e)[:100]}")
            if self.verbose:
                logger.exception(f"Coach analysis validation error")
            return None
    
    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """
        Extract JSON object from text, handling extra content.
        
        Strategies:
        1. Look for {...} at beginning/end
        2. Search for pattern matching JSON structure
        3. Try to find balanced braces
        """
        # Strategy 1: Find first { and match to }
        start = text.find('{')
        if start == -1:
            return None
        
        # Count braces to find matching }
        brace_count = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        if end <= start:
            return None
        
        return text[start:end]


async def run_coach_analysis(
    evolution_state: EvolutionState,
    raw_logs: Optional[List[str]] = None,
    base_url: str = "http://localhost:1234",
    model: str = "gemma-2-9b-it"
) -> Optional[CoachAnalysis]:
    """
    Convenience function: Run coach analysis on evolution state.
    
    Args:
        evolution_state: GA state
        raw_logs: Optional log lines
        base_url: LM Studio URL
        model: Model name
    
    Returns:
        CoachAnalysis or None if error
    """
    coach = GemmaCoachClient(base_url=base_url, model=model)
    return await coach.analyze_evolution(evolution_state, raw_logs)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Create sample evolution state
    state = EvolutionState(
        generation=5,
        population_size=56,
        mean_fitness=-15.2,
        std_fitness=35.4,
        best_fitness=0.6,
        best_trades=11,
        best_win_rate=0.91,
        best_avg_r=3.96,
        best_pnl=0.3086,
        below_min_trades_percent=57.1,
        mean_trade_count=8.5,
        diversity_metric=0.42,
        recent_improvement=0.0,
        is_stagnant=True,
        fitness_config_dict={
            "preset": "balanced",
            "min_trades": 10,
            "min_win_rate": 0.40,
            "fitness_function_type": "hard_gates",
        },
        ga_config_dict={
            "population_size": 56,
            "tournament_size": 6,
            "elite_fraction": 0.25,
            "mutation_probability": 0.90,
            "mutation_rate": 0.70,
            "immigrant_fraction": 0.0,
        }
    )
    
    # Run analysis
    async def main():
        analysis = await run_coach_analysis(state)
        if analysis:
            print(f"\n{analysis.to_json()}")
    
    asyncio.run(main())
