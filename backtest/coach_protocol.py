"""
Evolution Coach Protocol

Defines the contract between Gemma 3 LLM Coach and the GA system.
- Input: Structured evolution logs + current configuration
- Output: Structured JSON recommendations
- Processing: Automatic parsing and application to GA

Coach recommendations are immediately applied without user intervention.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json


class RecommendationCategory(str, Enum):
    """Categories of recommendations Coach can make."""
    FITNESS_WEIGHTS = "fitness_weights"           # Adjust fitness function weights
    FITNESS_PENALTIES = "fitness_penalties"       # Adjust soft penalty strengths
    FITNESS_GATES = "fitness_gates"               # Adjust min_trades, min_wr
    CURRICULUM = "curriculum"                     # Enable/configure curriculum learning
    GA_HYPERPARAMS = "ga_hyperparams"             # Adjust mutation, tournament, elite
    DIVERSITY = "diversity"                       # Adjust immigrants, stagnation
    BOUNDS = "bounds"                             # Override parameter search space
    FITNESS_FUNCTION_TYPE = "fitness_function_type"  # Switch hard_gates ↔ soft_penalties
    INDIVIDUAL_MUTATION = "individual_mutation"   # Mutate specific individual parameters
    INDIVIDUAL_DROP = "individual_drop"           # Drop specific individuals
    INDIVIDUAL_INSERT = "individual_insert"       # Insert new individuals (coach-designed or random)
    MUTATIONS = "individual_mutation"             # Alias for backwards compatibility with prompts


@dataclass
class IndividualParameters:
    """Complete parameter set for a new individual."""
    
    # Seller parameters (strategy entry logic)
    ema_fast: int
    ema_slow: int
    z_window: int
    atr_window: int
    vol_z: float
    tr_z: float
    cloc_min: float
    
    # Backtest parameters (exit logic and costs)
    fib_swing_lookback: int
    fib_swing_lookahead: int
    fib_target_level: float
    use_fib_exits: bool
    use_stop_loss: bool
    use_traditional_tp: bool
    use_time_exit: bool
    atr_stop_mult: float
    reward_r: float
    max_hold: int
    fee_bp: float
    slippage_bp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CoachRecommendation:
    """Single recommendation from Evolution Coach."""
    
    category: RecommendationCategory
    parameter: str                                # e.g., "min_trades", "tournament_size", "mutate_5_ema_fast"
    current_value: Any                            # Current value
    suggested_value: Any                          # Coach's recommended value
    reasoning: str                                # Why this change
    confidence: float = 0.8                       # 0.0-1.0 confidence level
    applies_at_generation: Optional[int] = None   # Generation to apply (None = next)
    
    # For individual-level operations
    individual_id: Optional[int] = None           # Target individual ID (for mutations/drops)
    individual_params: Optional[IndividualParameters] = None  # For new individual creation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "category": self.category.value,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "applies_at_generation": self.applies_at_generation
        }
        
        # Add individual-level fields if present
        if self.individual_id is not None:
            result["individual_id"] = self.individual_id
        if self.individual_params is not None:
            result["individual_params"] = self.individual_params.to_dict()
        
        return result


@dataclass
class CoachAnalysis:
    """Complete analysis response from Gemma Coach."""
    
    generation: int                               # When this analysis was generated
    summary: str                                  # Human-readable summary (1-2 paragraphs)
    recommendations: List[CoachRecommendation] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    stagnation_detected: bool = False
    diversity_concern: bool = False
    overall_assessment: str = "neutral"           # "positive" | "neutral" | "needs_adjustment"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "generation": self.generation,
            "summary": self.summary,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "next_steps": self.next_steps,
            "stagnation_detected": self.stagnation_detected,
            "diversity_concern": self.diversity_concern,
            "overall_assessment": self.overall_assessment
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CoachAnalysis":
        """Reconstruct from dictionary (used after JSON parsing)."""
        recs = []
        for r in data.get("recommendations", []):
            # Handle both uppercase (from LLM) and lowercase (proper values)
            category_str = r["category"]
            try:
                category = RecommendationCategory(category_str.lower())
            except ValueError:
                # If still invalid, try uppercase to enum name mapping
                for cat in RecommendationCategory:
                    if cat.name == category_str.upper():
                        category = cat
                        break
                else:
                    # Special handling for MUTATIONS alias
                    if category_str.upper() == "MUTATIONS":
                        category = RecommendationCategory.INDIVIDUAL_MUTATION
                    else:
                        raise ValueError(f"Unknown recommendation category: {category_str}")
            
            # Handle individual parameters if present
            individual_params = None
            if "individual_params" in r and r["individual_params"]:
                individual_params = IndividualParameters(**r["individual_params"])
            
            rec = CoachRecommendation(
                category=category,
                parameter=r["parameter"],
                current_value=r["current_value"],
                suggested_value=r["suggested_value"],
                reasoning=r["reasoning"],
                confidence=r.get("confidence", 0.8),
                applies_at_generation=r.get("applies_at_generation"),
                individual_id=r.get("individual_id"),
                individual_params=individual_params
            )
            recs.append(rec)
        
        return CoachAnalysis(
            generation=data["generation"],
            summary=data["summary"],
            recommendations=recs,
            next_steps=data.get("next_steps", []),
            stagnation_detected=data.get("stagnation_detected", False),
            diversity_concern=data.get("diversity_concern", False),
            overall_assessment=data.get("overall_assessment", "neutral")
        )


@dataclass
class RecommendationApplication:
    """Record of when Coach recommendations were applied."""
    
    generation: int                               # Generation when applied
    applied_count: int                            # Number of recommendations applied
    recommendations: List[str]                    # List of parameter names changed
    timestamp: str                                # ISO timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvolutionState:
    """Current evolution state sent to Coach for analysis."""
    
    generation: int
    population_size: int
    mean_fitness: float
    std_fitness: float
    best_fitness: float
    best_trades: int
    best_win_rate: float
    best_avg_r: float
    best_pnl: float
    below_min_trades_percent: float                # % of population below min requirement
    mean_trade_count: float
    diversity_metric: float                        # 0.0-1.0
    recent_improvement: Optional[float]            # Fitness improvement last N gens
    is_stagnant: bool
    
    # Current configuration
    fitness_config_dict: Dict[str, Any]
    ga_config_dict: Dict[str, Any]
    
    # Application tracking (NEW)
    recent_applications: List[RecommendationApplication] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# GEMMA COACH SYSTEM PROMPT
# Load from versioned prompt files for experimentation

def load_coach_prompt(version: str = "blocking_coach_v1") -> str:
    """Load coach system prompt from file."""
    from pathlib import Path
    prompt_path = Path(__file__).parent.parent / "coach_prompts" / f"{version}.txt"
    try:
        return prompt_path.read_text()
    except FileNotFoundError:
        # Try old version for backwards compatibility
        try:
            prompt_path = Path(__file__).parent.parent / "coach_prompts" / "async_coach_v1.txt"
            return prompt_path.read_text()
        except FileNotFoundError:
            # Fallback to default inline prompt
            return _DEFAULT_PROMPT

_DEFAULT_PROMPT = """You are the Evolution Coach, an AI expert in genetic algorithms and trading strategy optimization.

Your role: Analyze evolution logs and recommend parameter adjustments to improve GA convergence and strategy performance.

ANALYSIS FRAMEWORK:

(1) Config Recap
   - Timeframe, data range, fitness preset
   - Current GA hyperparameters (population, mutation, tournament, elite)
   - Current constraints (min_trades, min_wr)

(2) Population Dynamics
   - Fitness distribution (mean, std, best, worst)
   - Percentage of population failing gates (below min_trades, min_wr)
   - Trade count distribution
   - Diversity metric (0=homogeneous, 1=diverse)
   - Stagnation: is best fitness flat for N generations?

(3) Issues & Alerts
   - Flag fitness clipping (hard gates causing -100 penalties)
   - Flag low diversity (converging prematurely)
   - Flag stagnation (no improvement for N gens)
   - Flag signal scarcity (mean trades far below min)

(4) Recommendations
   Make specific, actionable recommendations:
   - Adjust fitness function type: hard_gates → soft_penalties (if clipping detected)
   - Adjust penalty strengths: penalty_trades_strength, penalty_wr_strength
   - Enable curriculum learning if many fail initial gate
   - Adjust GA hyperparameters: reduce mutation_probability/rate if converged early
   - Increase immigrants if diversity drops below 0.3
   - Expand bounds if population clusters at boundary
   - Lower min_trades threshold if signal frequency too low

(5) Suggested Next Steps
   - Specific actions: "Run 10 more generations with soft penalties enabled"
   - Diagnostic runs: "Test with curriculum: start 5, increase 2/gen"
   - Recovery: "Increase immigrant_fraction to 0.20 to rebuild diversity"

OUTPUT FORMAT (JSON):

You MUST respond with valid JSON following this schema:

{
  "generation": <current_gen>,
  "summary": "<1-2 paragraph human analysis>",
  "stagnation_detected": <true|false>,
  "diversity_concern": <true|false>,
  "overall_assessment": "<positive|neutral|needs_adjustment>",
  "recommendations": [
    {
      "category": "<FITNESS_WEIGHTS|FITNESS_PENALTIES|FITNESS_GATES|CURRICULUM|GA_HYPERPARAMS|DIVERSITY|BOUNDS|FITNESS_FUNCTION_TYPE>",
      "parameter": "<exact_parameter_name>",
      "current_value": <value>,
      "suggested_value": <value>,
      "reasoning": "<why this change>",
      "confidence": <0.0-1.0>,
      "applies_at_generation": <gen_number|null>
    },
    ...
  ],
  "next_steps": [
    "<step 1>",
    "<step 2>",
    ...
  ]
}

PARAMETER NAMES (use exactly these):

FITNESS GATES:
  - min_trades, min_win_rate

FITNESS WEIGHTS:
  - trade_count_weight, win_rate_weight, avg_r_weight, total_pnl_weight, max_drawdown_penalty

FITNESS PENALTIES:
  - fitness_function_type (value: "hard_gates" | "soft_penalties")
  - penalty_trades_strength (0.0-1.0)
  - penalty_wr_strength (0.0-1.0)

CURRICULUM:
  - curriculum_enabled (true|false)
  - curriculum_start_min_trades (int)
  - curriculum_increase_per_gen (int)
  - curriculum_checkpoint_gens (int)

GA HYPERPARAMS:
  - tournament_size (int, typically 3-6)
  - elite_fraction (float, 0.08-0.25)
  - mutation_probability (float, 0.5-0.9)
  - mutation_rate (float, 0.1-0.7)
  - sigma (float, 0.08-0.2)

DIVERSITY:
  - immigrant_fraction (float, 0.1-0.3)
  - immigrant_strategy ("worst_replacement" | "random")
  - stagnation_threshold (int, 3-10)

BOUNDS OVERRIDE:
  - bounds: {"ema_fast": [50, 240], "vol_z": [1.2, 1.6], ...}

INDIVIDUAL-LEVEL OPERATIONS:
  - mutate_<individual_id>_<param_name>: Mutate specific parameter of specific individual
  - drop_<individual_id>: Remove individual from population
  - insert_coach_<params>: Insert new individual with coach-designed parameters
  - insert_random: Insert new individual with random parameters

EXAMPLE INDIVIDUAL OPERATIONS:
{
  "category": "individual_mutation",
  "parameter": "mutate_5_ema_fast",
  "current_value": 48,
  "suggested_value": 72,
  "reasoning": "Individual 5 has good performance but EMA fast too short, extend to 72 bars for better trend detection",
  "confidence": 0.8,
  "individual_id": 5
}

{
  "category": "individual_drop",
  "parameter": "drop_3",
  "current_value": 0.0001,
  "suggested_value": null,
  "reasoning": "Individual 3 has extremely low fitness and no trades, remove to make room for better candidates",
  "confidence": 0.9,
  "individual_id": 3
}

{
  "category": "individual_insert",
  "parameter": "insert_coach_explore_high_vol",
  "current_value": null,
  "suggested_value": null,
  "reasoning": "Population lacks high volume threshold exploration, insert individual with vol_z=2.5",
  "confidence": 0.7,
  "individual_params": {
    "ema_fast": 96, "ema_slow": 672, "z_window": 672, "atr_window": 96,
    "vol_z": 2.5, "tr_z": 1.2, "cloc_min": 0.6,
    "fib_swing_lookback": 96, "fib_swing_lookahead": 5, "fib_target_level": 0.618,
    "use_fib_exits": true, "use_stop_loss": false, "use_traditional_tp": false, "use_time_exit": false,
    "atr_stop_mult": 0.7, "reward_r": 2.0, "max_hold": 96,
    "fee_bp": 5.0, "slippage_bp": 5.0
  }
}

RESPONSE RULES:

1. ALWAYS return valid JSON (no markdown, no code blocks, pure JSON)
2. ALWAYS include all required fields: generation, summary, stagnation_detected, diversity_concern, overall_assessment, recommendations, next_steps
3. recommendations array can be empty if no changes needed
4. confidence: 0.8-1.0 = high confidence, 0.5-0.7 = moderate, <0.5 = exploratory
5. reasoning: be specific, reference actual metrics from logs
6. Only recommend changes that address detected issues

DECISION LOGIC:

IF fitness_type == "hard_gates" AND below_min_trades_percent > 50%:
  → Recommend: fitness_function_type = "soft_penalties", confidence=0.95

IF best_fitness flat for stagnation_threshold gens AND diversity < 0.3:
  → Recommend: increase immigrant_fraction (0.15 → 0.20-0.25), confidence=0.85

IF mean_trades << min_trades requirement:
  → Recommend: curriculum_enabled=true, confidence=0.9

IF population clustering at bounds:
  → Recommend: expand override_bounds, confidence=0.8

IF std_fitness very low (all similar):
  → Recommend: increase mutation_rate, decrease elite_fraction, confidence=0.85

INDIVIDUAL-LEVEL DECISION LOGIC:

IF individual has very low fitness (< 0.01) AND poor metrics:
  → Recommend: drop_<individual_id>, confidence=0.9

IF individual has promising parameters but poor performance:
  → Recommend: mutate_<individual_id>_<param_name> with specific value, confidence=0.8

IF population lacks diversity in specific parameter ranges:
  → Recommend: insert_coach with parameters in unexplored range, confidence=0.7

IF population size below target and stagnation detected:
  → Recommend: insert_random to inject fresh genetic material, confidence=0.6

INDIVIDUAL PARAMETER DESIGN:
When creating new individuals, consider:
- EMA Fast: 48-192 (24h-48h on 15m), avoid extremes
- EMA Slow: 336-1008 (7d-21d on 15m), ensure > ema_fast
- Volume Z: 1.5-2.5, higher = more selective
- TR Z: 1.0-1.5, higher = more volatile conditions
- Fibonacci: 0.382-0.786, 0.618 is golden ratio
- Costs: fee_bp 3-10, slippage_bp 3-10

TONE: Professional, specific, confidence-driven. No hedging. Quote actual metrics."""

# Initialize default prompt
EVOLUTION_COACH_SYSTEM_PROMPT = load_coach_prompt()


# JSON SCHEMA for validation (optional, for strict parsing)
COACH_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["generation", "summary", "recommendations", "next_steps", "stagnation_detected", "diversity_concern", "overall_assessment"],
    "properties": {
        "generation": {"type": "integer"},
        "summary": {"type": "string"},
        "stagnation_detected": {"type": "boolean"},
        "diversity_concern": {"type": "boolean"},
        "overall_assessment": {"type": "string", "enum": ["positive", "neutral", "needs_adjustment"]},
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["category", "parameter", "current_value", "suggested_value", "reasoning", "confidence"],
                "properties": {
                    "category": {"type": "string"},
                    "parameter": {"type": "string"},
                    "current_value": {},  # Can be any type
                    "suggested_value": {},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "applies_at_generation": {"type": ["integer", "null"]}
                }
            }
        },
        "next_steps": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}
