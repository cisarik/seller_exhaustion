# Agent Mode Fix - Tool Calls Not Working

**Date**: 2025-01-21  
**Status**: ✅ Fixed

---

## Problem

Agent's tools weren't being called. Instead, seeing validation errors:

```
[COACH  ] 🔍 Parsing JSON (810 chars)
[COACH  ] 📝 Injected generation=10 into response
[COACH  ] ❌ Validation error: KeyError
[COACH  ] Details: 'summary'
✗ [llm_coach] Coach analysis validation error
⚠ [coach_manager_blocking] Coach analysis returned None (LLM may not have responded)
⚠ [coach_manager_blocking] Coach analysis failed at Gen 10, skipping
⚠ [stats_panel] ⚠️  Coach analysis returned no mutations
[COACH  ] ⚠️  No recommendations
```

---

## Root Cause

**Mismatch between prompt mode and analysis method:**

### What Was Happening

```python
# .env configuration
COACH_SYSTEM_PROMPT=agent01  # Agent prompt that outputs TOOL CALLS

# stats_panel.py (multi-step mode, line 1489)
success, summary = loop.run_until_complete(
    self.coach_manager.analyze_and_apply_blocking(  # ❌ WRONG!
        population=population,
        fitness_config=fitness_config,
        ga_config=None
    )
)
```

### The Issue

1. **Agent prompt (`agent01`)** tells LLM to output:
   ```json
   {
     "thinking": "...",
     "tool_calls": [...]
   }
   ```

2. **But `analyze_and_apply_blocking()`** expects JSON with:
   ```json
   {
     "generation": 10,
     "summary": "...",  # ← Missing!
     "recommendations": [...]
   }
   ```

3. **Result:** `CoachAnalysis.from_dict()` fails with `KeyError: 'summary'`

---

## Why Two Code Paths?

In `stats_panel.py` there were **two separate coach invocations**:

1. **Single-step mode** (`_update_after_generation()`, line ~1405):
   ```python
   # ✅ Correct: Uses agent executor
   success, summary = loop.run_until_complete(
       self.coach_manager.analyze_and_apply_with_agent(...)
   )
   ```

2. **Multi-step mode** (`_run_multi_step_thread()`, line ~1489):
   ```python
   # ❌ Wrong: Uses blocking/JSON mode
   success, summary = loop.run_until_complete(
       self.coach_manager.analyze_and_apply_blocking(...)  
   )
   ```

The test (`test_agent.py`) was running in a context similar to multi-step mode, hitting the wrong code path!

---

## Solution

### Fix: Auto-detect prompt mode and use correct analysis method

**File**: `app/widgets/stats_panel.py` (line 1488-1525)

```python
# Run coach analysis with current fitness config
_, _, fitness_config = self.get_current_params()

# Get GA config from optimizer
from core.models import OptimizationConfig
ga_config = OptimizationConfig(
    population_size=len(population.individuals),
    mutation_probability=getattr(population, 'mutation_probability', 0.9),
    mutation_rate=getattr(population, 'mutation_rate', 0.55),
    sigma=getattr(population, 'sigma', 0.15),
    elite_fraction=getattr(population, 'elite_fraction', 0.1),
    tournament_size=getattr(population, 'tournament_size', 3),
    immigrant_fraction=getattr(population, 'immigrant_fraction', 0.0)
)

# Check which prompt version is configured to use correct analysis method
# agent01 prompt → use analyze_and_apply_with_agent (tool calls)
# blocking_coach_v1 prompt → use analyze_and_apply_blocking (JSON response)
prompt_version = getattr(self.coach_manager, 'system_prompt', 'agent01')

if 'agent' in prompt_version.lower():
    # Agent mode: expects tool calls, not JSON
    success, summary = loop.run_until_complete(
        self.coach_manager.analyze_and_apply_with_agent(
            population=population,
            fitness_config=fitness_config,
            ga_config=ga_config
        )
    )
else:
    # Blocking mode: expects JSON response with 'summary' field
    success, summary = loop.run_until_complete(
        self.coach_manager.analyze_and_apply_blocking(
            population=population,
            fitness_config=fitness_config,
            ga_config=ga_config
        )
    )
```

### Fix: Handle different summary formats

**File**: `app/widgets/stats_panel.py` (line 1527-1535)

```python
if success:
    # Agent mode returns 'total_actions', blocking mode returns 'total_mutations'
    action_count = summary.get('total_actions', summary.get('total_mutations', 0))
    logger.info("✅ Coach completed: %s actions/mutations", action_count)
    print(f"[COACH  ] ✅ Applied {action_count} actions/mutations")
else:
    error_msg = summary.get('error', 'Unknown error')
    logger.warning("⚠️  Coach analysis failed: %s", error_msg)
    print(f"[COACH  ] ⚠️  {error_msg}")
```

---

## How It Works Now

### Agent Mode (`agent01` prompt)
```
1. LLM outputs: {"thinking": "...", "tool_calls": [...]}
2. AgentExecutor._parse_tool_calls() extracts tool calls
3. Tools execute: analyze_population(), mutate_individual(), etc.
4. Agent iterates until finish_analysis() called
5. Returns: {"success": True, "total_actions": 5, "iterations": 3}
```

### Blocking Mode (`blocking_coach_v1` prompt)
```
1. LLM outputs: {"generation": 10, "summary": "...", "recommendations": [...]}
2. llm_coach._parse_response() validates JSON
3. CoachAnalysis.from_dict() creates analysis object
4. apply_coach_recommendations() applies mutations
5. Returns: {"success": True, "total_mutations": 3}
```

---

## Testing

### Run the agent test:
```bash
python test_agent.py
```

### Expected output:
```
5️⃣  Running agent analysis...
   This will take 30-90 seconds depending on LLM speed...

🤖 Evolution Coach Agent starting...
   Max iterations: 10

🔄 Iteration 1/10
   📤 Sending XXX chars to LLM...
   📥 Received XXX chars from LLM
   🔧 Executing: analyze_population()
✅ analyze_population succeeded
   🔧 Executing: update_fitness_gates(min_trades=5)
✅ update_fitness_gates succeeded
   ...
✅ Agent called finish_analysis - session complete

✅ Agent analysis complete
   Iterations: 3
   Tool calls: 5
   Actions taken: 5

6️⃣  Verifying agent actions...
   ✅ Success: True
   📊 Iterations: 3
   🔧 Tool calls: 5
   📝 Actions: 5

   📋 Actions taken by agent:
      1. analyze_population
         → Analyzed population state
      2. update_fitness_gates
         → min_trades: 20 → 5
      3. update_ga_params
         → immigrant_fraction: 0.0 → 0.15
      4. mutate_individual
         → Mutated Individual #1: vol_z 1.2 → 1.1
      5. finish_analysis
         → Completed analysis

   🔍 Validation:
      ✅ Agent analyzed population (good start)
      ✅ Agent addressed gate crisis (expected)
      ✅ Agent adjusted GA parameters
      ✅ Agent called finish_analysis

✅ TEST PASSED - Agent successfully analyzed stagnating population!
```

---

## Files Modified

1. **app/widgets/stats_panel.py** (+28 lines)
   - Added prompt mode detection
   - Route to correct analysis method
   - Handle different summary formats

---

## Benefits

✅ **Auto-detection**: No manual configuration needed  
✅ **Backward compatible**: Both agent and blocking modes work  
✅ **Clear separation**: Each mode uses correct analysis path  
✅ **Better error messages**: Shows specific error instead of generic validation failure

---

## Related Files

- `backtest/coach_agent_executor.py` - Agent loop implementation
- `backtest/coach_manager_blocking.py` - Two analysis methods
- `backtest/llm_coach.py` - LLM client with JSON parsing
- `coach_prompts/agent01.txt` - Agent prompt (tool calls)
- `coach_prompts/blocking_coach_v1.txt` - Blocking prompt (JSON)
- `test_agent.py` - Agent test script

---

## Summary

**Problem**: Agent prompt was being used with JSON-expecting code path  
**Cause**: Multi-step optimization used wrong analysis method  
**Fix**: Auto-detect prompt mode and route to correct method  
**Result**: Tools now execute properly, agent works as designed
