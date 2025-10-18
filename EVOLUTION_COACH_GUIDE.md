# Evolution Coach Protocol - Complete Implementation Guide

## Overview

The Evolution Coach is a fully autonomous agent that analyzes GA evolution logs and recommends parameter adjustments to improve convergence. **No manual intervention required** - recommendations flow directly into GA configuration.

```
GA Runs (Gens 1-5)
    ↓ (logs: STAT, BEST, fitness values)
    ↓
Gemma 3 Coach (local LLM) analyzes evolution state
    ↓ (returns: JSON with recommendations + reasoning)
    ↓
Coach Integration Layer automatically parses & applies
    ↓ (updates FitnessConfig + OptimizationConfig)
    ↓
GA Runs (Gens 6-10) with OPTIMIZED configuration
    ↓
Loop continues autonomously
```

---

## Files Implemented

### 1. **core/models.py** (Updated)
- Extended `FitnessConfig` with:
  - `fitness_function_type`: "hard_gates" | "soft_penalties"
  - `penalty_trades_strength` / `penalty_wr_strength`: Soft penalty tuning
  - `curriculum_*`: Curriculum learning configuration
  - `get_effective_min_trades(generation)`: Dynamic min requirement based on gen

- Added `OptimizationConfig`:
  - All GA hyperparameters (Coach can tune)
  - `immigrant_fraction` / `stagnation_threshold`: Diversity control
  - `override_bounds`: Dict to modify parameter search space
  - `track_diversity` / `track_stagnation`: Enable monitoring

### 2. **backtest/optimizer.py** (Enhanced)

#### Updated Functions:

**`calculate_fitness(metrics, config, generation)`**
- Now supports `fitness_function_type`:
  - `"hard_gates"`: Original behavior (hard -100 clipping)
  - `"soft_penalties"`: Continuous penalty function (recommended)
- Supports curriculum learning: effective min_trades increases per generation
- Maintains fitness gradient for GA to exploit

**`evaluate_individual(individual, data, tf, fitness_config, generation)`**
- Passes `generation` to calculate_fitness for curriculum support

**`evolution_step(population, data, tf, fitness_config, ga_config, ...)`**
- New parameter: `ga_config: OptimizationConfig` (Coach applies changes here)
- Backward compatible with old parameters
- Features:
  - Applies bounds overrides
  - Tracks diversity + stagnation
  - Automatically adds random immigrants when diversity drops
  - Records detailed history (diversity, below_min_trades, etc.)
  - Detects stagnation for Coach alerts

#### Population Class Enhancements:

**`add_immigrants(fraction, strategy, generation)`**
- Adds N% random individuals to maintain diversity
- Strategies: "worst_replacement" | "random"
- Coach controls fraction per generation

**`get_diversity_metric()`**
- Returns 0.0-1.0 diversity score
- Based on pairwise parameter distances
- Coach triggers immigrants if < 0.3

**`apply_bounds_override(override_bounds)`**
- Coach can narrow/expand parameter search space
- Example: `{"ema_fast": (50, 240), "vol_z": (1.2, 1.6)}`

### 3. **backtest/coach_protocol.py** (NEW)

Defines the evolution coach protocol:

- **CoachRecommendation**: Single recommendation with category, parameter, current/suggested values, reasoning, confidence
- **CoachAnalysis**: Complete response (summary, recommendations, next_steps, flags)
- **EvolutionState**: GA state snapshot sent to Coach
- **EVOLUTION_COACH_SYSTEM_PROMPT**: Exact prompt for Gemma (detailed framework)
- **COACH_RESPONSE_SCHEMA**: JSON schema for validation

**Key Features:**
- Structured JSON I/O with Coach
- Categories: FITNESS_WEIGHTS, FITNESS_PENALTIES, GA_HYPERPARAMS, DIVERSITY, BOUNDS, CURRICULUM, etc.
- Confidence scores (0.0-1.0) for each recommendation
- Clear reasoning for each change

### 4. **backtest/llm_coach.py** (NEW)

Async client for Gemma 3 (local LM Studio):

**GemmaCoachClient**
- Connects to `http://localhost:1234` (LM Studio default)
- Async/await for non-blocking calls
- Robust JSON parsing (handles LLM text before/after JSON)
- Error handling for connection issues

**Key Methods:**
- `analyze_evolution(evolution_state)`: Send state to Coach, get analysis
- `_extract_json(text)`: Robustly extract JSON from LLM response
- `_parse_response(response_text)`: Convert to CoachAnalysis

**Usage:**
```python
coach = GemmaCoachClient(base_url="http://localhost:1234")
analysis = await coach.analyze_evolution(evolution_state)
# Returns: CoachAnalysis with recommendations
```

### 5. **backtest/coach_integration.py** (NEW)

Bridge between GA and Coach:

**`build_evolution_state(population, fitness_config, ga_config)`**
- Converts GA state → EvolutionState
- Calculates: below_min_trades%, diversity, stagnation, improvement
- Ready to send to Coach

**`get_coach_analysis(population, fitness_config, ga_config)`**
- Wrapper: build state + call Coach
- Fully async

**`apply_coach_recommendations(analysis, fitness_config, ga_config)`**
- Automatically applies Coach's recommendations
- Returns new (fitness_config, ga_config)
- Handles all 8 recommendation categories
- Prints applied changes

**`format_coach_output(analysis)`**
- Pretty-prints Coach analysis for console/UI

---

## How the Protocol Works

### 1. Coach Initialization

User runs GA with default or custom configuration:
```python
fitness_cfg = FitnessConfig(preset="high_frequency", curriculum_enabled=True)
ga_cfg = OptimizationConfig(population_size=56, immigrant_fraction=0.15)
```

### 2. Evolution Runs

GA executes N generations with normal selection/crossover/mutation:
```python
population = evolution_step(population, data, tf, fitness_cfg, ga_cfg)
```

The function now:
- Applies any bounds overrides from Coach
- Calculates diversity + stagnation
- Adds immigrants if needed
- Tracks detailed metrics

### 3. Coach Analysis

After N generations, user triggers analysis:
```python
# Build state from current population
state = build_evolution_state(population, fitness_cfg, ga_cfg)

# Send to Gemma Coach for analysis
analysis = await get_coach_analysis(population, fitness_cfg, ga_cfg)

# Coach returns: summary, recommendations, flags, next_steps
print(format_coach_output(analysis))
```

### 4. Automatic Application

App automatically applies recommendations **without user button click**:
```python
# Parse Coach's JSON response
analysis = CoachAnalysis.from_dict(json_response)

# Apply recommendations to config
fitness_cfg, ga_cfg = apply_coach_recommendations(analysis, fitness_cfg, ga_cfg)

# Continue GA with new config
population = evolution_step(population, data, tf, fitness_cfg, ga_cfg)
```

---

## Example: Evolution Coach Recommendations

### Input (from GA logs at Gen 5):
```
Generation: 5
Mean Fitness: -15.2 ± 35.4
Best Fitness: 0.6
Below min_trades: 57.1%
Diversity: 0.42
Stagnation: True (5 gens flat)
```

### Coach Analysis (from Gemma):
```json
{
  "generation": 5,
  "summary": "Population fitness is clipped at -100 due to hard gates preventing evolution gradient. 57% failing min_trades requirement suggests signal frequency too low.",
  "stagnation_detected": true,
  "diversity_concern": true,
  "overall_assessment": "needs_adjustment",
  "recommendations": [
    {
      "category": "FITNESS_FUNCTION_TYPE",
      "parameter": "fitness_function_type",
      "current_value": "hard_gates",
      "suggested_value": "soft_penalties",
      "reasoning": "Hard gates cause -100 clipping for 57% of population, eliminating gradient",
      "confidence": 0.95
    },
    {
      "category": "FITNESS_GATES",
      "parameter": "min_trades",
      "current_value": 10,
      "suggested_value": 5,
      "reasoning": "Majority can't reach current min_trades; lower to 5 initially",
      "confidence": 0.9
    },
    {
      "category": "CURRICULUM",
      "parameter": "curriculum_enabled",
      "current_value": false,
      "suggested_value": true,
      "reasoning": "Enable curriculum to gradually increase min_trades",
      "confidence": 0.85
    },
    {
      "category": "DIVERSITY",
      "parameter": "immigrant_fraction",
      "current_value": 0.0,
      "suggested_value": 0.15,
      "reasoning": "Diversity at 0.42, add 15% immigrants to maintain exploration",
      "confidence": 0.8
    }
  ],
  "next_steps": [
    "Switch to soft penalties to restore fitness gradient",
    "Enable curriculum learning with start=5, increase 2/gen",
    "Add 15% random immigrants to boost diversity",
    "Run 10 generations with new config to verify improvement"
  ]
}
```

### Applied Configuration:
```python
# Before
fitness_cfg.fitness_function_type = "hard_gates"  # ← CHANGED
fitness_cfg.min_trades = 10                       # ← CHANGED
fitness_cfg.curriculum_enabled = False            # ← CHANGED
ga_cfg.immigrant_fraction = 0.0                   # ← CHANGED

# After  
fitness_cfg.fitness_function_type = "soft_penalties"
fitness_cfg.min_trades = 5
fitness_cfg.curriculum_enabled = True
ga_cfg.immigrant_fraction = 0.15
```

### Result (Gen 6-10):
- ✅ Soft penalties restore gradient → GA can distinguish good from bad
- ✅ Lower min_trades allows more individuals to pass gate initially
- ✅ Curriculum gradually increases requirement (5→7→9→11→...)
- ✅ Immigrants maintain 0.15 diversity → exploration continues
- ✅ **Fitness improves** from stagnation!

---

## Gemma 3 System Prompt

The Coach uses a comprehensive system prompt (in `coach_protocol.py`) that:

1. **Explains Coach role**: Analyze evolution logs, diagnose issues, recommend fixes
2. **Defines analysis framework**: Config, population dynamics, issues, recommendations, next steps
3. **Specifies JSON output format**: Strict schema for reliable parsing
4. **Lists all tunable parameters**: Parameter names, ranges, categories
5. **Provides decision logic**: IF-THEN rules for detecting issues and recommending solutions
6. **Sets tone**: Professional, specific, confidence-driven

Example decision logic:
```
IF fitness_type == "hard_gates" AND below_min_trades_percent > 50%:
  → Recommend: fitness_function_type = "soft_penalties", confidence=0.95

IF mean_trades << min_trades requirement:
  → Recommend: curriculum_enabled=true, confidence=0.9

IF population diversity < 0.3:
  → Recommend: increase immigrant_fraction, confidence=0.85
```

---

## Integration with UI (stats_panel.py)

The UI needs one addition - an "Analyze with Coach" trigger:

```python
class EvolutionCoachButton(QPushButton):
    """Button to trigger Coach analysis (automatic recommendation flow)."""
    
    def __init__(self, stats_panel):
        super().__init__("🤖 Analyze with Gemma Coach")
        self.stats_panel = stats_panel
        self.clicked.connect(self.on_clicked)
    
    async def on_clicked(self):
        """Trigger Coach analysis and auto-apply recommendations."""
        print("📊 Sending evolution state to Gemma Coach...")
        
        # Get current state
        population = self.stats_panel.population
        fitness_cfg = self.stats_panel.fitness_config
        ga_cfg = self.stats_panel.ga_config
        logs = self.stats_panel.log_collector.get_recent(50)
        
        # Analyze with Coach (async)
        analysis = await get_coach_analysis(
            population, fitness_cfg, ga_cfg, logs,
            base_url="http://localhost:1234",
            model="gemma-2-9b-it"
        )
        
        if analysis:
            # Display analysis
            print(format_coach_output(analysis))
            
            # AUTO-APPLY recommendations (no button click needed!)
            new_fitness, new_ga = apply_coach_recommendations(analysis, fitness_cfg, ga_cfg)
            
            # Update UI state
            self.stats_panel.update_config(new_fitness, new_ga)
            
            print("✅ Recommendations applied! Next generation will use optimized config.")
        else:
            print("❌ Coach analysis failed (check LM Studio connection)")
```

---

## Running Evolution Coach

### Prerequisites

1. **Gemma 3 Model in LM Studio**
   - Download from Hugging Face (e.g., `gemma-2-9b-it`)
   - Load in LM Studio
   - Server runs on `http://localhost:1234`

2. **Python Packages**
   ```bash
   pip install httpx  # Already in poetry.lock
   ```

### Workflow

**Step 1: Start LM Studio**
```bash
lm-studio-server --port 1234
# Server running on http://localhost:1234
```

**Step 2: Run GA with Coach-compatible config**
```python
from core.models import FitnessConfig, OptimizationConfig
from backtest.coach_integration import build_evolution_state, get_coach_analysis, apply_coach_recommendations

# Create configs Coach can modify
fitness_cfg = FitnessConfig(
    preset="high_frequency",
    curriculum_enabled=False,  # Coach might enable this
    fitness_function_type="hard_gates"  # Coach might switch to "soft_penalties"
)

ga_cfg = OptimizationConfig(
    population_size=56,
    immigrant_fraction=0.0,  # Coach will increase if needed
    track_diversity=True,
    track_stagnation=True
)

# Run generations
for gen in range(50):
    population = evolution_step(population, data, tf, fitness_cfg, ga_cfg)
    
    # Every 5 generations, trigger Coach analysis
    if gen % 5 == 4:
        print(f"\n📊 Coach analysis at generation {gen+1}")
        
        # Async call (requires event loop)
        analysis = await get_coach_analysis(population, fitness_cfg, ga_cfg)
        
        if analysis:
            # Apply recommendations
            fitness_cfg, ga_cfg = apply_coach_recommendations(analysis, fitness_cfg, ga_cfg)
```

**Step 3: Monitor Console Output**
```
=== Generation 5 ===
Evaluating 8 individuals...
  [1/8] Fitness: -0.45 | Trades: 14 | WR: 52%
  ...
Population stats: mean=-0.18, std=0.45, best=0.34
  Below min_trades: 32/56 (57.1%)
  Diversity: 0.42, Stagnation: True

📊 Coach analysis at generation 5
⏳ Sending evolution state to Gemma Coach...

📋 SUMMARY:
Population fitness clipped by hard gates. 57% failing min_trades suggests low signal frequency.
Recommend: soft penalties, curriculum learning, increase immigrants.

⚠️  FLAGS:
  • Stagnation: Yes
  • Diversity: Concern
  • Overall: NEEDS_ADJUSTMENT

📌 RECOMMENDATIONS (4):
  1. FITNESS_FUNCTION_TYPE
     Suggested: soft_penalties (was hard_gates)
     Confidence: 95%

  2. MIN_TRADES
     Suggested: 5 (was 10)
     Confidence: 90%
  
  ... 2 more recommendations ...

✅ Applied: 4/4 recommendations
  • fitness.fitness_function_type = "soft_penalties" (was "hard_gates")
  • fitness.min_trades = 5 (was 10)
  • fitness.curriculum_enabled = True (was False)
  • ga.immigrant_fraction = 0.15 (was 0.0)

📊 Coach analysis complete in 12.3s
✅ Recommendations applied! Next generation will use optimized config.

=== Generation 6 ===
Evaluating 8 individuals...
  [1/8] Fitness: 0.12 | Trades: 16 | WR: 50%   ← Better diversity!
  [2/8] Fitness: 0.05 | Trades: 14 | WR: 55%
```

---

## Key Advantages

1. **Autonomous Evolution**: Coach adapts GA configuration without user intervention
2. **Continuous Learning**: Analyzes evolution state every N generations
3. **Interpretable**: Coach explains reasoning for each recommendation
4. **Flexible**: Coach can recommend any parameter Coach needs
5. **Safe**: Recommendations have confidence scores; low-confidence ones can be skipped
6. **Reproducible**: All Coach decisions logged for review

---

## Testing Coach Locally

Test Coach analysis without running full GA:

```python
import asyncio
from backtest.coach_protocol import EvolutionState
from backtest.llm_coach import run_coach_analysis

# Create test state (simulating stagnation problem)
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
    fitness_config_dict={...},
    ga_config_dict={...}
)

# Get Coach analysis
async def test():
    analysis = await run_coach_analysis(state)
    if analysis:
        print(analysis.to_json())

asyncio.run(test())
```

---

## Summary

The Evolution Coach protocol provides:

✅ **Structured evolution state** → Coach understands GA dynamics  
✅ **Gemma 3 LLM analysis** → Intelligent recommendations with reasoning  
✅ **Automatic parsing** → JSON → CoachAnalysis → recommendations  
✅ **Config application** → Recommendations automatically applied to FitnessConfig + OptimizationConfig  
✅ **No manual intervention** → Fully autonomous feedback loop  

The Coach can now recommend any parameter change the GA supports, and the app automatically applies it to the next generation. **Evolution is now self-improving!**
