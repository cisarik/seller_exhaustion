# Evolution Coach Client Lifecycle Fix

## Problem
Error: **"Default client is already created, cannot set its API host"**

This error occurred when trying to reload the model after unloading it to clear the context window.

## Root Cause
The LM Studio SDK only allows **ONE client instance** to be created. When the workflow was:
1. Load model → Create client
2. Run analysis
3. Unload model
4. Load model again → Try to create NEW client ❌ **ERROR: Client already exists**

The old client was never cleared, preventing a fresh client creation on reload.

## Solution
In `backtest/llm_coach.py`, the `unload_model()` method now:
1. Executes `lms unload` command (as before)
2. Sets `self._model_loaded = False` (as before)
3. **CRITICAL**: Sets `self._lms_client = None` (NEW)

This ensures that when the model is loaded again, a fresh client can be created without conflicts.

## Code Changes

```python
async def unload_model(self):
    """
    Unload model using lms CLI and reset client.
    
    CRITICAL: Always clears _lms_client so a fresh client can be created
    on next load. This prevents "Default client is already created" errors
    when reloading model to clear context window.
    """
    # ... unload logic ...
    if result.returncode == 0:
        self._model_loaded = False
        self._lms_client = None  # ← CRITICAL FIX
        # ...
```

## Workflow (Now Working)

```
Generation 5: Analyze with LLM
    ├─ Load model (create client)
    ├─ analyze_evolution()
    └─ Client ready for LLM call ✓

Generation 5 Complete: Apply recommendations
    ├─ unload_model()
    │  ├─ Execute: lms unload
    │  ├─ Set: _model_loaded = False
    │  └─ Clear: _lms_client = None  ← Fresh start!
    └─ Context window freed ✓

Generation 10: Analyze with LLM
    ├─ Load model (create NEW client) ← No conflicts!
    ├─ analyze_evolution()
    └─ Client ready for LLM call ✓
```

## Test Coverage

Three test suites verify this fix:

### 1. Client Lifecycle Management
- ✅ Client not created on initialization
- ✅ Client created when needed
- ✅ Client is reused (not recreated unnecessarily)

### 2. Model Check Without Conflicts  
- ✅ Model status check doesn't create LM client
- ✅ check_model_loaded() uses only subprocess

### 3. Full Reload Cycle
- ✅ Unload properly clears client
- ✅ Reload creates fresh start
- ✅ No "Default client is already created" error
- ✅ Multiple analysis cycles work correctly
- ✅ Context window successfully freed between analyses

Run tests:
```bash
pytest tests/test_coach_llm_client_lifecycle.py -v
pytest tests/test_coach_reload_cycle.py -v
```

## How It Works in Production

When your coach workflow runs during optimization:

```
Generation 5: Coach triggers first analysis
  1. load_model() → Creates LM Studio client
  2. analyze_evolution() → LLM generates recommendations
  3. Apply recommendations to GA config
  4. unload_model() → Clears client, frees context window ✓

Generation 10: Coach triggers second analysis
  1. load_model() → Creates FRESH LM Studio client (no conflicts!)
  2. analyze_evolution() → LLM generates new recommendations
  3. Apply recommendations
  4. unload_model() → Clear for next cycle

... repeats as needed ...
```

## Benefits

✅ **No More Client Conflicts** - Fresh client created on each load  
✅ **Context Window Management** - Can clear between analyses  
✅ **Multiple Analysis Cycles** - Unlimited reload/analyze cycles  
✅ **Clean State** - Each analysis starts with blank context  
✅ **Memory Efficient** - Old client properly disposed  

## Prerequisites

- LM Studio running locally
- Model loaded: `lms load google/gemma-3-12b --gpu=0.6`
- `lms` CLI command available in PATH
- `check_model_loaded()` returns True before first analysis

## Troubleshooting

If you still see "Default client already created" errors:

1. **Check LM Studio state:**
   ```bash
   lms ps  # Should show model READY
   ```

2. **Kill stale LM Studio process:**
   ```bash
   pkill -f lmstudio
   sleep 2
   lms load google/gemma-3-12b --gpu=0.6
   ```

3. **Verify subprocess execution:**
   - Check coach logs for `[LMS    ] ✅ Model unloaded successfully`
   - Check logs for `[LMS    ] 🔄 Client reset for fresh context window on reload`

4. **Debug client state:**
   - Enable verbose logging: `GemmaCoachClient(verbose=True)`
   - Watch for: `_lms_client=None` after unload
   - Verify: Fresh client created on next load
