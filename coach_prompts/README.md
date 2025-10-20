# Evolution Coach Prompts

This directory contains versioned system prompts for the Evolution Coach LLM.

## Purpose

Different prompts enable experimentation with coaching strategies:
- **Aggressive** - More confident recommendations, faster parameter changes
- **Conservative** - Higher confidence thresholds, cautious adjustments
- **Diversity-focused** - Emphasizes population diversity over convergence
- **Performance-focused** - Prioritizes fitness improvement over stability

## Current Prompts

### `async_coach_v1.txt` (Default)

**Purpose**: Balanced coach with async awareness

**Features**:
- Understands it sees trimmed logs (last N generations)
- Tracks when recommendations were applied
- Assesses impact of previous recommendations
- Balanced confidence levels (0.5-0.95)

**Use for**: General-purpose evolution coaching

**When to use**:
- Default starting point
- Balanced approach to optimization
- Most production use cases

---

## Creating New Prompts

### 1. Copy existing prompt

```bash
cp async_coach_v1.txt my_experiment_v1.txt
```

### 2. Modify prompt

Key sections to customize:

**Decision Logic** (line ~120):
```
IF fitness_type == "hard_gates" AND below_min_trades_percent > 50%:
  → Recommend: fitness_function_type = "soft_penalties", confidence=0.95
```

**Confidence Levels** (line ~80):
```
**Confidence scoring:**
- 0.9-1.0 = High confidence, strong signal
- 0.7-0.8 = Moderate confidence, clear trend
- 0.5-0.6 = Exploratory, worth trying
```

**Tone** (line ~180):
```
TONE: Professional, specific, confidence-driven. Quote actual metrics.
```

### 3. Test new prompt

```python
from backtest.coach_manager import CoachManager

coach = CoachManager(prompt_version="my_experiment_v1")
```

---

## Prompt Design Guidelines

### ✅ DO:
- Be specific about JSON schema requirements
- Include exact parameter names
- Define clear decision logic
- Specify confidence levels
- Quote actual metrics from logs
- Acknowledge application history

### ❌ DON'T:
- Use vague language ("might", "perhaps", "maybe")
- Omit confidence scores
- Forget to mention trimmed logs
- Ignore previous recommendations
- Use non-standard JSON format

---

## Example Modifications

### Aggressive Coach

**Goal**: Faster convergence, more confident recommendations

```diff
- confidence: 0.8-1.0 = high confidence
+ confidence: 0.9-1.0 = high confidence

- IF diversity < 0.3:
+ IF diversity < 0.4:
    → Recommend: increase immigrant_fraction, confidence=0.85
+   → Recommend: increase immigrant_fraction, confidence=0.95

- IF mean_trades << min_trades:
+ IF mean_trades < min_trades * 0.8:
    → Recommend: curriculum_enabled=true, confidence=0.9
+   → Recommend: curriculum_enabled=true, confidence=0.98
```

### Conservative Coach

**Goal**: Only recommend changes with high confidence

```diff
- confidence: 0.8-1.0 = high confidence
+ confidence: 0.95-1.0 = high confidence

- IF diversity < 0.3:
+ IF diversity < 0.2:
    → Recommend: increase immigrant_fraction, confidence=0.85
+   → Recommend: increase immigrant_fraction, confidence=0.90

+ ONLY recommend if confidence >= 0.85
```

### Diversity-Focused Coach

**Goal**: Maintain high diversity, prevent premature convergence

```diff
+ PRIORITY 1: Population diversity
+ PRIORITY 2: Stagnation prevention  
+ PRIORITY 3: Fitness improvement

- IF diversity < 0.3:
+ IF diversity < 0.5:
    → Recommend: increase immigrant_fraction, confidence=0.85
+   → Recommend: increase immigrant_fraction to 0.20-0.30, confidence=0.95

+ IF std_fitness < 0.1 * mean_fitness:
+   → Recommend: increase mutation_rate, confidence=0.90
```

---

## Testing Prompts

### Quick Test

```python
import asyncio
from backtest.coach_protocol import EvolutionState
from backtest.llm_coach import GemmaCoachClient

# Create test state
state = EvolutionState(
    generation=10,
    population_size=56,
    mean_fitness=0.5,
    std_fitness=0.15,
    best_fitness=0.8,
    # ... fill in other fields
)

# Test with custom prompt
async def test():
    client = GemmaCoachClient(prompt_version="my_experiment_v1")
    analysis = await client.analyze_evolution(state)
    print(analysis.to_json())

asyncio.run(test())
```

### A/B Test

```python
# Compare two prompts on same evolution state
async def compare_prompts():
    coach_a = GemmaCoachClient(prompt_version="async_coach_v1")
    coach_b = GemmaCoachClient(prompt_version="aggressive_v1")
    
    analysis_a = await coach_a.analyze_evolution(state)
    analysis_b = await coach_b.analyze_evolution(state)
    
    print("=== Prompt A ===")
    print(f"Recommendations: {len(analysis_a.recommendations)}")
    print(f"Avg confidence: {sum(r.confidence for r in analysis_a.recommendations) / len(analysis_a.recommendations):.2f}")
    
    print("\n=== Prompt B ===")
    print(f"Recommendations: {len(analysis_b.recommendations)}")
    print(f"Avg confidence: {sum(r.confidence for r in analysis_b.recommendations) / len(analysis_b.recommendations):.2f}")
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| async_coach_v1 | 2025-01-XX | Initial async-aware coach with application tracking |

---

## Best Practices

1. **Version everything**: Always create new version, don't overwrite
2. **Test thoroughly**: Run at least 20 generations before production
3. **Document changes**: Add version entry above
4. **Compare results**: A/B test against default before switching
5. **Monitor confidence**: Track avg confidence scores per prompt
6. **Track effectiveness**: Measure convergence speed with each prompt

---

## Future Experiments

Ideas for new prompts:

- **Multi-objective coach**: Optimize for multiple goals simultaneously
- **Curriculum-focused coach**: Specializes in curriculum learning strategies
- **Bounds expert**: Focuses on parameter space exploration
- **Meta-coach**: Learns from previous evolution runs

---

## Contributing

When creating new prompts that work well:
1. Add to this directory with descriptive name
2. Document purpose and use cases
3. Add to version history table
4. Share results on team channel
