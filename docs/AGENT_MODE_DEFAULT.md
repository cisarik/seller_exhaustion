# Agent Mode Now Default - JSON Mode Removed

**Date**: 2025-01-21  
**Status**: ✅ Complete

---

## Summary

Removed JSON/blocking coach mode entirely from the codebase. Agent mode with tool calling is now the only mode and the default everywhere.

---

## Problem Fixed

The original issue was that the code had two different coach modes:
1. **JSON mode** (`blocking_coach_v1` prompt) - Expected structured JSON response with 'summary' field
2. **Agent mode** (`agent01` prompt) - Uses tool calls for multi-step reasoning

This caused confusion and the agent's tools weren't being called because stats_panel.py was using the wrong code path.

---

## Changes Made

### 1. Simplified stats_panel.py

**File**: `app/widgets/stats_panel.py`

**Before** (mode detection logic):
```python
# Check which prompt version is configured to use correct analysis method
prompt_version = getattr(self.coach_manager, 'system_prompt', 'agent01')

if 'agent' in prompt_version.lower():
    # Agent mode
    success, summary = loop.run_until_complete(
        self.coach_manager.analyze_and_apply_with_agent(...)
    )
else:
    # Blocking mode
    success, summary = loop.run_until_complete(
        self.coach_manager.analyze_and_apply_blocking(...)
    )
```

**After** (always agent mode):
```python
# Use agent-based analysis (tool calls)
success, summary = loop.run_until_complete(
    self.coach_manager.analyze_and_apply_with_agent(
        population=population,
        fitness_config=fitness_config,
        ga_config=ga_config
    )
)
```

Success message handling also simplified:
```python
if success:
    # Agent returns 'total_actions'
    action_count = summary.get('total_actions', 0)
    logger.info("✅ Coach agent completed: %s actions", action_count)
    print(f"[COACH  ] ✅ Agent took {action_count} actions")
```

### 2. Removed Blocking Methods from coach_manager_blocking.py

**File**: `backtest/coach_manager_blocking.py`

**Removed** (~312 lines):
1. `async def analyze_session_blocking()` - JSON-based analysis
2. `async def apply_session_recommendations_blocking()` - Apply JSON recommendations  
3. `async def analyze_and_apply_blocking()` - Full blocking workflow

**Kept**:
- `analyze_and_apply_with_agent()` - The only method now

### 3. Updated Docstrings

**Module-level docstring**:
```python
"""
Evolution Coach Manager - Agent Mode

Implements agent-based Evolution Coach with tool calling:
- Population is FROZEN during agent analysis
- Agent gets FULL population snapshot with all individual data
- Agent makes multiple tool calls to diagnose and take actions
- Tools execute directly on population (mutations, GA params, fitness gates)
- Evolution resumes after agent completes
...
"""
```

**Class docstring**:
```python
class BlockingCoachManager:
    """
    Manages Evolution Coach in agent mode with tool calling.
    
    Agent-based analysis:
    - Optimization PAUSES during agent analysis
    - Agent gets FULL population snapshot with all individual data  
    - Agent makes multiple tool calls to diagnose and act
    - Actions applied directly to population via tools
    - Multi-step reasoning and iterative refinement
    ...
    """
```

### 4. Updated Default Parameters

**File**: `backtest/coach_manager_blocking.py`

```python
def __init__(
    self,
    base_url: str = "http://localhost:1234",
    model: Optional[str] = None,
    prompt_version: str = "agent01",  # ← Changed from "blocking_coach_v1"
    system_prompt: Optional[str] = None,
    ...
):
```

**File**: `.env` (already had):
```
COACH_SYSTEM_PROMPT=agent01
```

### 5. Removed Blocking Prompt File

**Deleted**: `coach_prompts/blocking_coach_v1.txt`

**Kept**: `coach_prompts/agent01.txt` (the only prompt now)

### 6. Updated Example Code

**File**: `backtest/coach_manager_blocking.py` (bottom)

**Before**:
```python
if __name__ == "__main__":
    print("Blocking Evolution Coach Manager")
    print("\nKey Features:")
    print("  - BLOCKING optimization (pauses during coach analysis)")
    print("  - Full population data sent to coach")
    ...
    print("      success, summary = await manager.analyze_and_apply_blocking(pop, fitness, ga)")
```

**After**:
```python
if __name__ == "__main__":
    print("Evolution Coach Manager - Agent Mode")
    print("\nKey Features:")
    print("  - Agent-based analysis with tool calling")
    print("  - Full population data sent to agent")
    print("  - Direct individual mutations via tools")
    print("  - Multi-step reasoning and diagnosis")
    ...
    print("      success, summary = await manager.analyze_and_apply_with_agent(pop, fitness, ga)")
```

---

## Benefits

✅ **Simplicity**: One mode, one code path, easier to maintain  
✅ **Power**: Agent mode is more capable (multi-step reasoning, diagnosis)  
✅ **No confusion**: No mode detection logic, no wrong code paths  
✅ **Better error messages**: Tools show what agent is doing step-by-step  
✅ **Less code**: ~350 lines removed (methods + prompt + routing logic)

---

## Files Modified

1. **app/widgets/stats_panel.py** (-22 lines)
   - Removed mode detection logic
   - Always use `analyze_and_apply_with_agent()`
   - Simplified success message handling

2. **backtest/coach_manager_blocking.py** (-312 lines, +20 lines)
   - Removed 3 blocking methods
   - Updated module docstring
   - Updated class docstring
   - Changed default `prompt_version` to `"agent01"`
   - Updated example code

3. **coach_prompts/blocking_coach_v1.txt** (deleted)

**Total**: ~330 lines removed, codebase is simpler

---

## Migration Guide

### For Users

**No action needed!** Agent mode is now the default.

**If you were using** `COACH_SYSTEM_PROMPT=blocking_coach_v1` **in .env**:
- Change to: `COACH_SYSTEM_PROMPT=agent01`
- Or remove the line (agent01 is default)

### For Developers

**Code that was calling blocking methods:**

```python
# OLD (no longer exists)
success, summary = await coach_manager.analyze_and_apply_blocking(
    population, fitness_config, ga_config
)

# NEW (use this)
success, summary = await coach_manager.analyze_and_apply_with_agent(
    population, fitness_config, ga_config
)
```

**Summary structure changed:**

```python
# OLD (blocking mode)
mutations_count = summary.get('total_mutations', 0)

# NEW (agent mode)
actions_count = summary.get('total_actions', 0)
```

---

## Testing

### Run agent test:
```bash
python test_agent.py
```

### Expected output:
```
5️⃣  Running agent analysis...

🤖 Evolution Coach Agent starting...
   Max iterations: 10

🔄 Iteration 1/10
   🔧 Executing: analyze_population()
✅ analyze_population succeeded
   🔧 Executing: update_fitness_gates(min_trades=5)
✅ update_fitness_gates succeeded
   🔧 Executing: mutate_individual(...)
✅ mutate_individual succeeded
   ...
✅ Agent called finish_analysis - session complete

✅ Agent analysis complete
   Iterations: 3
   Tool calls: 5
   Actions taken: 5

✅ TEST PASSED - Agent successfully analyzed stagnating population!
```

---

## Architecture

### Agent Workflow (Only Mode)

```
1. GA Evolution
   ↓ (every N generations)
2. PAUSE optimization
   ↓
3. Create frozen session (population snapshot)
   ↓
4. Initialize agent with toolkit
   ↓
5. Build initial observation
   ↓
6. Agent Loop:
   ├─ LLM thinks about problem
   ├─ Calls tools (analyze_population, mutate_individual, etc.)
   ├─ Observes tool results
   ├─ Thinks about next action
   └─ Repeat until finish_analysis()
   ↓
7. RESUME optimization with agent-modified population
```

### Available Tools (27 total)

**Observability**:
- `analyze_population()` - Get population statistics
- `get_param_distribution()` - Analyze parameter distribution
- `get_param_bounds()` - Query search space bounds

**Individual Control**:
- `mutate_individual()` - Directly modify individual parameters

**Fitness Function**:
- `update_fitness_gates()` - Update min_trades, min_win_rate
- `update_ga_params()` - Adjust GA hyperparameters

**Control Flow**:
- `finish_analysis()` - Complete session and return

---

## Related Documentation

- **AGENT_MODE_FIX.md** - Original fix for tool calls not working
- **README_AGENT.md** - Complete agent mode documentation
- **AGENT_UI_INTEGRATION_COMPLETE.md** - UI integration details
- **coach_prompts/agent01.txt** - Agent prompt specification
- **backtest/coach_agent_executor.py** - Agent loop implementation
- **backtest/coach_tools.py** - Tool implementations (27 tools)

---

## Summary

**Problem**: Two modes (JSON + Agent) caused confusion and wrong code paths  
**Solution**: Remove JSON mode, make agent mode the only option  
**Result**: Simpler codebase, more powerful analysis, no confusion  
**Impact**: ~350 lines removed, clearer architecture, better UX
