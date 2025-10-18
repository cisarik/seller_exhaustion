"""
Evolution Coach Protocol

Defines the contract between Gemma 3 LLM Coach and the GA system.
- Input: Structured evolution logs + current configuration
- Output: Structured JSON recommendations
- Processing: Automatic parsing and application to GA

Coach recommendations are immediately applied without user intervention.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
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


@dataclass
class CoachRecommendation:
    """Single recommendation from Evolution Coach."""
    
    category: RecommendationCategory
    parameter: str                                # e.g., "min_trades", "tournament_size"
    current_value: Any                            # Current value
    suggested_value: Any                          # Coach's recommended value
    reasoning: str                                # Why this change
    confidence: float = 0.8                       # 0.0-1.0 confidence level
    applies_at_generation: Optional[int] = None   # Generation to apply (None = next)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "applies_at_generation": self.applies_at_generation
        }


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
            rec = CoachRecommendation(
                category=RecommendationCategory(r["category"]),
                parameter=r["parameter"],
                current_value=r["current_value"],
                suggested_value=r["suggested_value"],
                reasoning=r["reasoning"],
                confidence=r.get("confidence", 0.8),
                applies_at_generation=r.get("applies_at_generation")
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# GEMMA COACH SYSTEM PROMPT
# This is the exact prompt to send to Gemma 3

EVOLUTION_COACH_SYSTEM_PROMPT = """You are the Evolution Coach, an AI expert in genetic algorithms and trading strategy optimization.

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

TONE: Professional, specific, confidence-driven. No hedging. Quote actual metrics."""


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
