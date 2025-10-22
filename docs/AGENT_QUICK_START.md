# Evolution Coach Agent: Quick Start Guide

## Quick Setup (30 seconds)

### Step 1: Configure

Edit `.env`:
```bash
COACH_ENABLED=true
COACH_AGENT_MAX_ITERATIONS=10      # Max tool calls per session
COACH_ANALYSIS_INTERVAL=10         # Analyze every 10 generations
COACH_DEBUG_PAYLOADS=true          # Optional: see full LLM traffic
```

### Step 2: Run

```bash
python examples/evolution_with_coach.py
```

### Step 3: Watch

Look for these logs:
```
✓ 🤖 Evolution Coach Manager initialized: agent_mode=ENABLED, interval=10 gens
[GEN 10] 🤖 Evolution Coach Agent starting...
[AGENT ] 🔄 Iteration 1/10
[AGENT ] 🔧 Executing: analyze_population(...)
[AGENT ] ✅ analyze_population succeeded
[AGENT ] 💭 100% below gates, need to act
[AGENT ] 🔧 Executing: update_fitness_gates(...)
[PARAMS] ✅ fitness.min_trades: 20 → 5
[AGENT ] 🔧 Executing: finish_analysis(...)
[COACH ] ✅ Agent analysis complete
[GEN 11] Evolution resumed...
```

---

## What To Expect

### Generation 10 (First Analysis)

**Agent Will:**
1. Call `analyze_population()` to understand state
2. Call `get_param_distribution()` if investigating parameters
3. Take 2-4 corrective actions:
   - Lower `min_trades` if gate too strict
   - Inject immigrants if diversity low
   - Mutate top performers
   - Expand bounds if clustering
4. Call `finish_analysis()` to complete

**Typical Pattern:**
- **3-5 iterations**
- **5-7 tool calls**
- **2-3 minutes** (depends on LLM speed)

### Generation 20 (Second Analysis)

**Agent Will:**
1. Check impact of previous changes
2. Adapt strategy based on results
3. Either:
   - Continue current approach (if working)
   - Escalate interventions (if not improving)
   - Fine-tune parameters (if near optimal)

---

## Common Agent Behaviors

### Scenario 1: Gate Crisis
```
Problem: 100% below min_trades=20

Agent Actions:
✅ analyze_population()
✅ update_fitness_gates(min_trades=5)
✅ update_ga_params(immigrant_fraction=0.15)
✅ finish_analysis()

Impact: 55% now passing gates, diversity up
```

### Scenario 2: Boundary Clustering
```
Problem: 7/12 individuals at ema_fast=48 (minimum)

Agent Actions:
✅ analyze_population()
✅ get_param_distribution("ema_fast")
✅ get_param_bounds()
✅ [Observes: Top performers use ema_fast=48]
✅ [TODO: update_param_bounds(ema_fast, new_min=24)]
✅ mutate_individual(id=9, parameter="ema_fast", new_value=36)
✅ finish_analysis()

Impact: Exploration of faster EMA region
```

### Scenario 3: Premature Convergence
```
Problem: diversity=0.08 (very low), early convergence

Agent Actions:
✅ analyze_population()
✅ get_param_distribution("vol_z")
✅ update_ga_params(mutation_rate=0.6, immigrant_fraction=0.25)
✅ insert_individual(strategy="random")
✅ insert_individual(strategy="random")
✅ finish_analysis()

Impact: Diversity increases, exploration resumes
```

---

## Debugging

### Enable Full Logging

```bash
export COACH_DEBUG_PAYLOADS=true
```

This shows:
- **Full system prompt** sent to LLM
- **Full user message** with population data
- **Full LLM response** with tool calls
- **Tool execution details**

### Check Logs

```bash
# Watch evolution log
tail -f logs/evolution_*.log

# Search for agent activity
grep "AGENT" logs/evolution_*.log

# Search for tool executions
grep "Executing:" logs/evolution_*.log
```

---

## Troubleshooting

### Agent Not Triggering

**Check:**
```bash
grep "should_analyze" logs/evolution_*.log
```

**Fix:**
- Verify `COACH_ENABLED=true`
- Check `COACH_ANALYSIS_INTERVAL` (should trigger at gen 10, 20, 30...)

### LLM Timeout

**Symptoms:**
```
[COACH ] ⏱️ LLM timeout after 3600s
```

**Fix:**
- Check LM Studio is running: `lms ps`
- Increase timeout: `COACH_RESPONSE_TIMEOUT=7200`
- Use faster model: `COACH_MODEL=gemma-2-9b-it`

### Tool Execution Failures

**Symptoms:**
```
[AGENT ] ❌ mutate_individual failed: Invalid individual_id
```

**Fix:**
- Check tool arguments in debug log
- Verify individual IDs are valid (0-11 for pop of 12)
- Check parameter names match exactly

### Agent Not Calling Tools

**Symptoms:**
```
[AGENT ] ⚠️ No tool calls in response
```

**Fix:**
- LLM may not understand tool format
- Check system prompt was loaded
- Enable debug: `COACH_DEBUG_PAYLOADS=true`
- Verify LLM response contains JSON

## Configuration Options

### Agent Behavior

```bash
COACH_AGENT_MAX_ITERATIONS=10      # Max tool calls (5-15 typical)
COACH_ANALYSIS_INTERVAL=10         # Analyze every N gens
COACH_POPULATION_WINDOW=10         # Context size (generations)
```

### LLM Settings

```bash
COACH_MODEL=google/gemma-3-12b     # Model name
COACH_RESPONSE_TIMEOUT=3600        # Timeout (seconds)
COACH_CONTEXT_LENGTH=5000          # Context window
COACH_GPU=0.6                      # GPU offload ratio
```

---

## Performance Expectations

### Agent Session Timing

- **Iteration 1**: 10-30 seconds (analyze_population)
- **Iteration 2**: 10-30 seconds (get_param_distribution)
- **Iteration 3**: 10-30 seconds (actions + finish)

**Total: 30-90 seconds per analysis** (depends on LLM speed)

### Resource Usage

- **GPU**: 60% of VRAM (Gemma 3 12B)
- **CPU**: Minimal during coach (evolution paused)
- **Memory**: ~2GB for LLM + population data

---

## Success Indicators

### Good Signs ✅
- Agent completes in 3-5 iterations
- Uses 5-7 tools per session
- Fitness improves within 5 generations
- Diversity trends toward 0.2-0.4
- Gate pass rate increases

### Warning Signs ⚠️
- Agent hits max iterations (10)
- Many tool execution failures
- Fitness declines after intervention
- Diversity drops to <0.05

---

## Advanced Usage

### Manual Intervention

If agent makes bad decisions, you can:
1. Stop optimization (Ctrl+C)
2. Manually adjust parameters in UI
3. Resume optimization
4. Agent will adapt to your changes

### Hybrid Mode

Use both agent and manual control:
- Let agent handle routine interventions
- Manually intervene for strategic decisions
- Agent learns from your changes

### Multi-Session Learning

Agent can learn across sessions:
- Track what worked/failed
- Build confidence in strategies
- Refine playbooks automatically

---

## Common Questions

### Q: How many tools does agent typically use?
**A:** 5-7 tools per session (3-5 iterations)

### Q: Can I disable the agent?
**A:** No. Agent mode is now the default and legacy JSON mode has been removed.

### Q: Does agent modify population permanently?
**A:** Yes, changes persist in current evolution run

### Q: Can agent break evolution?
**A:** Unlikely - has sanity checks, but monitor first few sessions

### Q: How do I know if agent is helping?
**A:** Compare generations 1-50 with vs without agent

---

## Quick Commands

```bash
# Enable debug
export COACH_DEBUG_PAYLOADS=true

# Adjust agent iterations
export COACH_AGENT_MAX_ITERATIONS=12

# Test agent
python examples/evolution_with_coach.py

# Check if working
grep "AGENT" logs/evolution_*.log | tail -20
```

---

## Summary

**Enable:** Set `COACH_ENABLED=true` and choose agent iterations in `.env`  
**Run:** `python examples/evolution_with_coach.py`  
**Watch:** Agent analyzes every 10 generations  
**Verify:** Fitness improves, diversity maintained  

**That's it! The agent is ready to steer your evolution.** 🚀
