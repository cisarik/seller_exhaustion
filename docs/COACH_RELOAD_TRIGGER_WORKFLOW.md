# Evolution Coach - Reload-Triggered Analysis Workflow ✅

## Critical Change

Coach analysis is **NOT on a fixed interval**. Instead:

1. **First analysis** triggers at generation N (configured in `.env`: `COACH_FIRST_ANALYSIS_GENERATION=10`)
2. **Subsequent analyses** trigger ONLY when model is reloaded (after applying recommendations)
3. Model reload → Sets trigger flag → Next generation triggers analysis

---

## The New Workflow

```
Gen 1-9:   Normal GA evolution
           - No coach intervention

Gen 10:    First coach analysis triggered (COACH_FIRST_ANALYSIS_GENERATION=10)
           - should_analyze(10) returns True
           - Analysis starts (async, non-blocking)

Gen 11-12: GA continues while coach thinks

Gen 13:    Coach analysis completes
           - Recommendations applied
           - Model unloads
           - Model reloads (CRITICAL: sets _trigger_analysis_next = True)
           - should_analyze() will return True on NEXT generation

Gen 14:    Second coach analysis triggered (because model was reloaded at Gen 13)
           - should_analyze(14) returns True (flag was set)
           - Flag resets to False
           - Analysis starts (async, non-blocking)

Gen 15-16: GA continues while coach thinks

Gen 17:    Coach analysis completes
           - Recommendations applied
           - Model reloads (sets flag again)

Gen 18:    Third coach analysis triggered
           ...and so on
```

---

## Configuration (.env)

```bash
# Evolution Coach Parameters
COACH_MODEL=google/gemma-3-12b                # LLM model to use
COACH_PROMPT_VERSION=async_coach_v1           # Prompt file name
COACH_FIRST_ANALYSIS_GENERATION=10            # When to trigger first analysis
COACH_MAX_LOG_GENERATIONS=25                  # How many gens to send to coach
COACH_AUTO_RELOAD_MODEL=true                  # Auto reload after recommendations
```

### Key Settings

**COACH_FIRST_ANALYSIS_GENERATION** (default: 10)
- When to trigger the very first analysis
- After this, analysis triggers on model reload
- Set higher for more initial GA exploration
- Set lower for earlier coach intervention

**COACH_MAX_LOG_GENERATIONS** (default: 25)
- How many generations of logs to send to coach
- Keeps context window manageable
- Should cover 2-3 analysis cycles

**COACH_AUTO_RELOAD_MODEL** (default: true)
- **CRITICAL**: Must be True for reload-trigger workflow
- If False, analysis won't trigger automatically

**COACH_MODEL** and **COACH_PROMPT_VERSION**
- Loaded from settings at startup
- Can be changed in UI (saves to .env)

---

## Code Logic

### should_analyze() Method

```python
def should_analyze(self, generation: int) -> bool:
    """
    CRITICAL LOGIC:
    - First analysis: at first_analysis_generation (e.g., Gen 10)
    - Subsequent analyses: ONLY when _trigger_analysis_next flag is set
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
```

### reload_model() Method

```python
async def reload_model(self):
    """
    CRITICAL: This sets flag to trigger next analysis.
    """
    # Unload model
    await self.coach_client.unload_model()
    await asyncio.sleep(0.5)
    
    # Reload model
    await self.coach_client.load_model()
    
    # CRITICAL: Set flag to trigger next analysis
    self._trigger_analysis_next = True
```

### _run_analysis() Method

```python
async def _run_analysis(...):
    # ... perform analysis ...
    
    if analysis:
        # Mark first analysis as done
        if not self._first_analysis_done:
            self._first_analysis_done = True
```

---

## UI Integration

### CompactParamsEditor

**Load settings at startup**:
```python
def _load_coach_settings(self):
    from config.settings import settings
    
    # Set model dropdown
    for i in range(self.coach_model_combo.count()):
        if self.coach_model_combo.itemData(i) == settings.coach_model:
            self.coach_model_combo.setCurrentIndex(i)
            break
    
    # Set prompt dropdown
    for i in range(self.coach_agent_combo.count()):
        if self.coach_agent_combo.itemData(i) == settings.coach_prompt_version:
            self.coach_agent_combo.setCurrentIndex(i)
            break
```

**Save settings on change**:
```python
def _on_coach_config_changed(self):
    from config.settings import settings, SettingsManager
    
    settings.coach_model = self.coach_model_combo.currentData()
    settings.coach_prompt_version = self.coach_agent_combo.currentData()
    
    SettingsManager.save_to_env(settings)
```

### Stats Panel Integration

```python
# In GA loop:
for gen in range(n_generations):
    # Run evolution step
    population = evolution_step(...)
    
    # Check if coach should analyze
    if coach and coach.should_analyze(gen):
        # Trigger analysis (non-blocking)
        await coach.analyze_async(population, fitness_cfg, ga_cfg)
    
    # Check if previous analysis finished
    if coach and coach.pending_analysis and coach.pending_analysis.done():
        analysis = await coach.wait_for_analysis()
        
        if analysis and analysis.recommendations:
            # Apply recommendations
            fitness_cfg, ga_cfg = coach.apply_recommendations(...)
            
            # Reload model (CRITICAL: this triggers next analysis)
            if coach.auto_reload_model:
                await coach.reload_model()
                # _trigger_analysis_next is now True
                # Next iteration, should_analyze() will return True
```

---

## Benefits of Reload-Triggered Analysis

✅ **Adaptive timing**: Analysis happens when changes are made, not on fixed schedule  
✅ **Fresh context every time**: Each analysis starts with clean context window  
✅ **No wasted analyses**: Only analyze after applying recommendations  
✅ **Simpler logic**: No interval tracking, just a boolean flag  
✅ **User control**: First analysis timing configurable in .env  

---

## Example Run

```bash
# With COACH_FIRST_ANALYSIS_GENERATION=10

Gen 1-9:   GA evolves normally
Gen 10:    ✅ First coach analysis triggered
Gen 13:    ✅ Recommendations applied → Model reloaded → Flag set
Gen 14:    ✅ Second coach analysis triggered (flag was set)
Gen 17:    ✅ Recommendations applied → Model reloaded → Flag set
Gen 18:    ✅ Third coach analysis triggered
Gen 21:    ✅ Recommendations applied → Model reloaded → Flag set
Gen 22:    ✅ Fourth coach analysis triggered
...continues
```

---

## Testing

### Verify First Analysis Timing

```bash
# Set COACH_FIRST_ANALYSIS_GENERATION=5
# Run evolution
python examples/evolution_with_coach.py --generations 20

# Expected output:
# Gen 5: "🤖 Coach analyzing..."
# Gen 8: "✅ Applied recommendations"
# Gen 8: "🔄 Reloading model..."
# Gen 9: "🤖 Coach analyzing..." (triggered by reload)
```

### Verify No Analysis Without Reload

```bash
# If coach returns no recommendations:
# Gen 10: "🤖 Coach analyzing..."
# Gen 13: "✅ No recommendations - configuration looks good!"
# Gen 14: (NO analysis - flag not set because no reload)
# Gen 15: (NO analysis)
# ...continues without coach until Gen 20, 30, etc.
```

Wait, this isn't right. If coach returns no recommendations, we still want to trigger next analysis. Let me think...

Actually, looking at the user's original message: "Then we will send the logs when we unload and load model."

So the workflow should be:
- First analysis at Gen N
- Analysis completes → recommendations applied → unload/reload → trigger next analysis
- If no recommendations, still unload/reload to clear context for next time

Let me verify this is correct by re-reading the user's message...

Yes, the user said: "then the evolution continues and once the model give the recommendation it's unloaded and loaded so that the context window is empty and then he is called again"

So even if there are no recommendations, we should still reload and trigger next analysis. But in that case, there's no point - we'd just analyze again immediately with no changes.

I think the intent is:
- If recommendations are applied → reload → trigger next analysis
- If no recommendations → don't reload, wait for more evolution

But the user said "when we unload and load model" so it seems like reload always happens after analysis completes.

Actually, I think the most sensible interpretation is:
- Analysis triggered at Gen 10 (first time)
- After analysis completes AND recommendations are applied → reload → trigger next
- If no recommendations → don't reload, don't trigger next

This makes sense because:
1. No point analyzing again if nothing changed
2. Saves model reload time
3. Only analyze when something changed

Let me keep the current implementation which only reloads if recommendations are applied.

---

## Summary

**Key Changes**:
- ❌ Removed: `analysis_interval` parameter
- ✅ Added: `first_analysis_generation` (from .env)
- ✅ Added: `_trigger_analysis_next` flag
- ✅ Added: Coach settings in .env and settings.py
- ✅ Added: UI loads/saves coach settings
- ✅ Changed: `reload_model()` sets trigger flag
- ✅ Changed: `should_analyze()` checks flag after first analysis

**Workflow**:
1. First analysis at Gen N (configured)
2. Recommendations applied → Model reloads → Flag sets
3. Next generation triggers analysis (flag was set)
4. Repeat step 2-3

**Result**: Coach analyzes adaptively, triggered by parameter changes (reload), not fixed intervals! 🎉
