# Evolution Coach - Async Implementation Guide

## Overview

The Evolution Coach now operates **asynchronously** with automatic parameter application. No manual intervention required - recommendations are applied immediately when the coach responds, without interrupting the GA optimization.

## Key Features

### ✅ Asynchronous Analysis
- Coach analysis runs in background (non-blocking)
- GA continues running while coach thinks
- Recommendations applied as soon as coach responds

### ✅ Log Trimming
- Automatically trims logs to last N generations (default: 25)
- Manages context window for large prompts
- Ensures coach always sees current parameter values

### ✅ Application Tracking
- Tracks when recommendations were applied
- Coach sees "Applied recommendations at Gen X" in logs
- Enables coach to assess if previous changes worked

### ✅ Model Management
- Load/unload Gemma 3 model as needed
- Frees memory when not in use
- Automatic loading before analysis

### ✅ Prompt Versioning
- Prompts saved in `coach_prompts/` directory
- Easy experimentation with different prompts
- Version control for prompt evolution

---

## Setup

### 1. Install LM Studio Python SDK

```bash
pip install lmstudio
```

### 2. Download and Load Gemma 3 Model

1. Open LM Studio desktop app
2. Go to "Discover" tab
3. Search for "google/gemma-3-12b" (or "gemma-2-9b-it" for smaller model)
4. Download the model
5. Go to "My Models" and load it
6. Enable server mode (port 1234)

### 3. Verify LM Studio Server

```bash
# Check if server is running
curl http://localhost:1234/v1/models

# Should return list of loaded models
```

---

## Quick Start

### Basic Usage

```python
import asyncio
from backtest.coach_manager import CoachManager
from backtest.optimizer import Population, evolution_step
from core.models import FitnessConfig, OptimizationConfig, Timeframe

# Initialize coach manager
coach = CoachManager(
    base_url="http://localhost:1234",
    model="google/gemma-3-12b",
    analysis_interval=5,           # Analyze every 5 generations
    max_log_generations=25,        # Keep last 25 generations in logs
    auto_apply=True,               # Automatically apply recommendations
    verbose=True
)

# Initialize GA configs
fitness_cfg = FitnessConfig(preset="balanced")
ga_cfg = OptimizationConfig(population_size=56, track_diversity=True)

# Initialize population
population = Population(size=56, seed_individual=None)

# Run GA with coach
async def run_evolution_with_coach():
    for gen in range(50):
        # Log generation start
        coach.add_log(gen, f"=== Generation {gen} ===")
        
        # Run evolution step
        population = evolution_step(
            population, data, Timeframe.m15, 
            fitness_cfg, ga_cfg
        )
        
        # Log generation stats
        stats = population.get_stats()
        coach.add_log(gen, f"Mean fitness: {stats['mean_fitness']:.4f}")
        coach.add_log(gen, f"Best fitness: {stats['max_fitness']:.4f}")
        
        # Trigger coach analysis (non-blocking)
        if coach.should_analyze(gen):
            # Start analysis in background
            await coach.analyze_async(population, fitness_cfg, ga_cfg)
        
        # Check if analysis finished
        if coach.pending_analysis and coach.pending_analysis.done():
            analysis = await coach.wait_for_analysis()
            
            if analysis and coach.auto_apply:
                # Apply recommendations automatically
                fitness_cfg, ga_cfg = coach.apply_recommendations(
                    analysis, fitness_cfg, ga_cfg
                )
                print(f"✅ Applied {len(analysis.recommendations)} recommendations")

# Run
asyncio.run(run_evolution_with_coach())
```

### Blocking Analysis (Wait for Coach)

If you want to wait for coach before continuing:

```python
async def run_with_blocking_coach():
    for gen in range(50):
        # ... run evolution step ...
        
        if coach.should_analyze(gen):
            # Analyze and apply (blocks until complete)
            fitness_cfg, ga_cfg = await coach.analyze_and_apply(
                population, fitness_cfg, ga_cfg
            )
```

---

## Architecture

### Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    GA EVOLUTION LOOP                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
         [Gen 0-4: Normal GA evolution]
                           │
                           ▼
                    ┌──────────────┐
                    │  Gen 5       │
                    │  Should      │
                    │  analyze?    │
                    └──────┬───────┘
                           │ YES
                           ▼
         ┌──────────────────────────────────┐
         │  START ASYNC COACH ANALYSIS      │
         │  (non-blocking, runs in bg)      │
         └──────────┬───────────────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
    [GA continues         [Coach thinks
     Gen 6, 7, 8...]       in background]
         │                     │
         │                     ▼
         │              [Coach responds
         │               with JSON]
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌────────────────────────┐
         │  APPLY RECOMMENDATIONS │
         │  (automatic, no button)│
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │  LOG APPLICATION       │
         │  "Applied at Gen 8"    │
         └────────┬───────────────┘
                  │
                  ▼
         [Gen 9+: GA uses optimized config]
                  │
                  ▼
         [Gen 10: Coach sees application log,
          assesses if changes helped]
```

### Key Components

**CoachManager** (`backtest/coach_manager.py`):
- Orchestrates entire coach lifecycle
- Handles async analysis
- Manages log trimming
- Tracks application history
- Provides convenience methods

**GemmaCoachClient** (`backtest/llm_coach.py`):
- LM Studio SDK integration
- Model loading/unloading
- Async LLM calls
- JSON parsing

**Coach Protocol** (`backtest/coach_protocol.py`):
- Data structures (CoachAnalysis, RecommendationApplication)
- System prompt loading from files
- Response schema

**Coach Integration** (`backtest/coach_integration.py`):
- Build evolution state from population
- Apply recommendations to configs
- Format output for console

---

## Prompt Versioning

### Using Different Prompts

```python
# Load specific prompt version
coach = CoachManager(
    prompt_version="async_coach_v1"  # Default
)

# Or use custom prompt
from backtest.llm_coach import GemmaCoachClient
client = GemmaCoachClient(prompt_version="my_custom_prompt_v2")
```

### Creating New Prompt Versions

1. Copy existing prompt:
```bash
cp coach_prompts/async_coach_v1.txt coach_prompts/my_experiment_v1.txt
```

2. Edit new prompt file:
```bash
nano coach_prompts/my_experiment_v1.txt
```

3. Test with new version:
```python
coach = CoachManager(prompt_version="my_experiment_v1")
```

### Prompt Evolution Tips

**What to experiment with:**

- **Confidence thresholds**: Adjust when coach should be confident
- **Decision logic**: Change IF-THEN rules
- **Tone**: More aggressive vs conservative recommendations
- **Metrics focus**: Emphasize different metrics (diversity, stagnation, etc.)
- **Output format**: Add/remove fields from JSON

**Example modifications:**

```text
# Original
IF diversity < 0.3:
  → Recommend: increase immigrant_fraction, confidence=0.85

# More aggressive
IF diversity < 0.4:
  → Recommend: increase immigrant_fraction, confidence=0.95
```

---

## Log Management

### Automatic Trimming

```python
# Trim to last N generations
coach = CoachManager(max_log_generations=25)

# Manually trim logs
from backtest.coach_integration import trim_logs_to_n_generations
trimmed = trim_logs_to_n_generations(full_logs, current_gen=50, n_generations=25)
```

### Log Format

Logs should include generation markers for trimming to work:

```python
# Good format
coach.add_log(5, "=== Generation 5 ===")
coach.add_log(5, "Mean fitness: 0.45")
coach.add_log(5, "Below min_trades: 32%")

# Also works
coach.add_log(5, "Gen 5: Best fitness improved to 0.67")
```

### Context Window Management

**Problem**: Large prompts exceed model context window

**Solution**:
1. Trim logs to last N generations (default: 25)
2. Load/unload model between analyses
3. Increase model context if needed (Gemma 3 supports up to 8k tokens)

**Monitor context usage**:
```python
# Check log size
stats = coach.get_stats()
print(f"Log lines: {stats['log_lines']}")

# If too large, reduce max_log_generations
coach.max_log_generations = 15  # Reduce from 25 to 15
```

---

## Application Tracking

### How It Works

1. Coach recommends changes at Gen 5
2. Changes applied immediately
3. Log entry: "Applied Coach recommendations at Gen 5: 4 changes (min_trades, immigrant_fraction, ...)"
4. At Gen 10, coach sees application log
5. Coach compares metrics before Gen 5 vs after Gen 5
6. Coach adjusts confidence/strategy based on results

### Example Flow

```
Gen 5:
  Coach: "Switch to soft_penalties (confidence: 0.95)"
  Applied: fitness_function_type = "soft_penalties"
  Log: "Applied recommendations at Gen 5: 1 change (fitness_function_type)"

Gen 10:
  Coach sees log: "Applied recommendations at Gen 5"
  Coach analyzes: Gen 1-4 vs Gen 6-10
  Fitness improved? → "Soft penalties working, maintain approach"
  Fitness worse? → "Soft penalties not helping, try curriculum instead"
```

### Viewing Application History

```python
# Get application history
for app in coach.application_history:
    print(f"Gen {app.generation}: {app.applied_count} changes")
    print(f"  Parameters: {', '.join(app.recommendations)}")
    print(f"  Time: {app.timestamp}")

# Get formatted history for coach prompt
history = coach.format_application_history()
print(history)
```

---

## Model Management

### Loading Models

```python
# Automatic loading (recommended)
client = GemmaCoachClient(model="google/gemma-3-12b")
# Model loads on first analysis call

# Manual loading
await client.load_model()
```

### Unloading Models

```python
# Free memory after analysis
await client.unload_model()

# Or via coach manager
coach.unload_model()
```

### Memory Optimization

**Strategy 1**: Load/unload between analyses
```python
async def optimized_evolution():
    for gen in range(100):
        # ... run GA ...
        
        if coach.should_analyze(gen):
            await coach.analyze_async(population, fitness_cfg, ga_cfg)
            analysis = await coach.wait_for_analysis()
            
            # Apply recommendations
            fitness_cfg, ga_cfg = coach.apply_recommendations(...)
            
            # Unload to free memory
            await coach.coach_client.unload_model()
```

**Strategy 2**: Keep loaded for multiple analyses
```python
# Load once at start
await coach.coach_client.load_model()

# Run many analyses
for gen in range(100):
    if coach.should_analyze(gen):
        await coach.analyze_async(...)
        # Model stays loaded

# Unload at end
await coach.coach_client.unload_model()
```

---

## Troubleshooting

### Issue: Coach not connecting

**Symptoms**:
```
❌ Cannot connect to LM Studio at http://localhost:1234
```

**Solutions**:
1. Check LM Studio is running: `curl http://localhost:1234/v1/models`
2. Verify server enabled in LM Studio settings
3. Check port (default: 1234)
4. Try restarting LM Studio

### Issue: Model not found

**Symptoms**:
```
❌ Failed to load model google/gemma-3-12b: Model not found
```

**Solutions**:
1. Check model is downloaded in LM Studio
2. Verify model identifier matches LM Studio name
3. Try loading model manually in LM Studio first

### Issue: Context window exceeded

**Symptoms**:
```
❌ LLM call error: Context length exceeded
```

**Solutions**:
1. Reduce `max_log_generations`: `coach.max_log_generations = 15`
2. Trim logs manually before sending
3. Use smaller model (gemma-2-9b-it instead of gemma-3-12b)
4. Simplify system prompt

### Issue: JSON parsing error

**Symptoms**:
```
⚠️ Could not find JSON in coach response
```

**Solutions**:
1. Check coach response in verbose mode
2. Verify system prompt enforces JSON output
3. Increase temperature slightly (0.3 → 0.4) for more structured output
4. Check if model is following instructions

### Issue: Recommendations not applying

**Symptoms**:
- Coach suggests changes but configs unchanged

**Solutions**:
1. Check `auto_apply=True` in CoachManager
2. Verify `apply_recommendations()` is called
3. Check for errors in console output
4. Verify parameter names match exactly (case-sensitive)

---

## Performance Tips

### 1. Adjust Analysis Interval

```python
# Analyze every 5 gens (default)
coach = CoachManager(analysis_interval=5)

# Less frequent = faster evolution, less coach guidance
coach = CoachManager(analysis_interval=10)

# More frequent = slower evolution, more coach guidance
coach = CoachManager(analysis_interval=3)
```

### 2. Model Selection

```python
# Faster, smaller context
coach = CoachManager(model="gemma-2-9b-it")

# Slower, larger context, better reasoning
coach = CoachManager(model="google/gemma-3-12b")
```

### 3. Async vs Blocking

```python
# Non-blocking (recommended for speed)
await coach.analyze_async(...)  # Returns immediately
# ... GA continues ...
analysis = await coach.wait_for_analysis()  # Check when done

# Blocking (recommended for reliability)
fitness_cfg, ga_cfg = await coach.analyze_and_apply(...)  # Waits
```

---

## Advanced Usage

### Custom Fitness Function Evolution

```python
# Start with balanced
fitness_cfg = FitnessConfig(preset="balanced")

# Coach may recommend switching to soft penalties
# After 5-10 gens with soft penalties, coach may recommend curriculum

# Coach tracks progression:
# Gen 0-5: balanced + hard gates
# Gen 6-15: balanced + soft penalties (coach recommendation)
# Gen 16+: high_frequency + curriculum (coach recommendation)
```

### Multi-Model Experiments

```python
# Run parallel experiments with different models
async def experiment():
    coach_small = CoachManager(model="gemma-2-9b-it")
    coach_large = CoachManager(model="google/gemma-3-12b")
    
    # Compare recommendations
    analysis_small = await coach_small.analyze_async(...)
    analysis_large = await coach_large.analyze_async(...)
    
    # Apply consensus recommendations
    if analysis_small.recommendations == analysis_large.recommendations:
        # Both agree, apply with high confidence
        fitness_cfg, ga_cfg = coach_small.apply_recommendations(...)
```

### Coach Ensemble

```python
# Use multiple prompts and vote
async def ensemble_coach():
    coaches = [
        CoachManager(prompt_version="async_coach_v1"),
        CoachManager(prompt_version="aggressive_v1"),
        CoachManager(prompt_version="conservative_v1"),
    ]
    
    analyses = []
    for coach in coaches:
        analysis = await coach.analyze_and_apply(...)
        analyses.append(analysis)
    
    # Count recommendation votes
    from collections import Counter
    all_recs = []
    for analysis in analyses:
        for rec in analysis.recommendations:
            all_recs.append((rec.parameter, rec.suggested_value))
    
    # Apply recommendations with 2+ votes
    consensus = [k for k, v in Counter(all_recs).items() if v >= 2]
```

---

## Summary

**Evolution Coach Async System provides:**

✅ **Non-blocking analysis** - GA continues while coach thinks  
✅ **Automatic application** - No manual buttons, recommendations flow immediately  
✅ **Log trimming** - Manage context window for large evolution runs  
✅ **Application tracking** - Coach sees when recommendations were applied  
✅ **Model management** - Load/unload for memory optimization  
✅ **Prompt versioning** - Easy experimentation with different coaching strategies  

**Key Takeaway**: Set it up once, let it run. Coach monitors evolution, detects issues, recommends fixes, and applies them automatically. The GA self-optimizes! 🚀
