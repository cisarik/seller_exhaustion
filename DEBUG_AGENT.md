# Debugging the Evolution Coach Agent

## Quick Debug Tools

### 1. Run with LM Studio Logs (RECOMMENDED)

See the **exact prompt** sent to the model:

```bash
./scripts/debug_agent_with_logs.sh
```

This will:
- ✅ Start `lms log stream` in background
- ✅ Run `test_agent.py`
- ✅ Show LM Studio's input/output in real-time
- ✅ Kill log stream when done

**What to look for:**
```
timestamp: 5/2/2024, 9:49:47 PM
type: llm.prediction.input
modelIdentifier: google/gemma-3-12b
input: "POPULATION STATE - Generation 25
OVERVIEW:
- Population size: 12
...
```

### 2. Run with Maximum Python Logging

See detailed Python-side logs:

```bash
./scripts/test_agent_verbose.sh
```

This will:
- ✅ Enable `COACH_DEBUG_PAYLOADS=true`
- ✅ Set Python logging to DEBUG
- ✅ Save full log to `agent_test_verbose.log`

### 3. Manual LM Studio Log Streaming

In one terminal:
```bash
lms log stream
```

In another terminal:
```bash
poetry run python test_agent.py
```

**Filtering logs:**
```bash
# Show only LLM input/output
lms log stream --filter llm.prediction

# Show all types
lms log stream
```

---

## Common Issues & Fixes

### Issue 1: LLM Not Returning Tool Calls

**Symptoms:**
```
⚠️ No tool calls in response - agent may be done
```

**Debug:**
1. Check `lms log stream` output
2. Look for `llm.prediction.output`
3. See what model actually returned

**Possible causes:**
- System prompt not loaded properly
- Model doesn't understand JSON format
- Temperature too high (too creative)
- Context window filled

**Fixes:**
```python
# Lower temperature for more deterministic output
llm_client = GemmaCoachClient(
    temperature=0.1,  # Was 0.3, try lower
    ...
)

# Simplify system prompt
# Edit coach_prompts/agent01.txt
# Add more JSON examples
```

### Issue 2: JSON Parse Errors

**Symptoms:**
```
❌ JSON parse error: Expecting ',' delimiter: line 11 column 8
```

**Debug:**
1. Check `lms log stream` for raw response
2. Look at what JSON was malformed
3. Check if model is outputting code fences

**Fixes:**
```python
# In coach_agent_executor.py, update regex:
# Strip code fences before parsing
response_text = re.sub(r'```json\s*|\s*```', '', response_text)
json_match = re.search(r'\{.*?"tool_calls".*?\}', response_text, re.DOTALL)
```

### Issue 3: Agent Not Calling finish_analysis

**Symptoms:**
```
Agent hits max iterations (10) without finishing
```

**Debug:**
1. Check conversation history length
2. See if context window is full
3. Verify finish_analysis is in tool list

**Fixes:**
```python
# Reduce max iterations
agent = AgentExecutor(
    max_iterations=7,  # Was 10
    ...
)

# Or increase context window
lms load google/gemma-3-12b --context-length 8000
```

### Issue 4: Model Taking Too Long

**Symptoms:**
```
✅ Agent completed in 120.5 seconds  # Too slow!
```

**Debug:**
1. Check GPU offload ratio
2. Monitor `lms ps` during inference
3. Check system resources

**Fixes:**
```bash
# Increase GPU offload (if you have VRAM)
lms load google/gemma-3-12b --gpu 0.8  # Was 0.6

# Or use smaller model
lms load google/gemma-2-9b-it --gpu 0.6

# Or reduce context
lms load google/gemma-3-12b --context-length 3000
```

---

## Debugging Workflow

### Step 1: Verify LM Studio Connection

```bash
# Check server
lms server status

# Check loaded models
lms ps

# Test basic inference
curl http://localhost:1234/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-12b",
    "prompt": "Hello, world!",
    "max_tokens": 50
  }'
```

### Step 2: Check System Prompt Loading

```bash
# Verify prompt file exists
ls -lh coach_prompts/agent01.txt

# Check it loads correctly
poetry run python -c "
from backtest.coach_protocol import load_coach_prompt
prompt = load_coach_prompt('agent01')
print(f'Loaded {len(prompt)} chars')
print(prompt[:200])
"
```

### Step 3: Test Tool Execution

```bash
# Test individual tools
poetry run python -c "
import asyncio
from backtest.coach_tools import CoachToolkit
# ... create toolkit ...
result = asyncio.run(toolkit.analyze_population())
print(result)
"
```

### Step 4: Run with Full Logging

```bash
# See everything
export COACH_DEBUG_PAYLOADS=true
export PYTHONUNBUFFERED=1

poetry run python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
" test_agent.py 2>&1 | tee full_debug.log
```

### Step 5: Analyze LM Studio Logs

```bash
# In another terminal while test runs
lms log stream | grep -A 20 "llm.prediction.input"

# Save for later analysis
lms log stream > lms_logs.txt
```

---

## Log Levels

### Python Logging

```python
import logging

# Set level in code
logging.basicConfig(level=logging.DEBUG)

# Or via environment
export PYTHONLOG=DEBUG
```

**Levels:**
- `DEBUG`: Everything (prompts, responses, tool calls)
- `INFO`: Important events (tool execution, iterations)
- `WARNING`: Issues (parse errors, timeouts)
- `ERROR`: Failures (tool errors, LLM errors)

### LM Studio Logs

**Log types:**
- `llm.prediction.input`: What was sent to model
- `llm.prediction.output`: What model returned
- `llm.prediction.progress`: Streaming updates
- `model.load`: Model loading events
- `model.unload`: Model unloading events

**Filter by type:**
```bash
lms log stream --filter llm.prediction.input
lms log stream --filter llm.prediction.output
```

---

## Example Debug Session

```bash
# Terminal 1: Start LM Studio logs
$ lms log stream --filter llm.prediction
I Streaming logs from LM Studio

# Terminal 2: Run test with verbose logging
$ export COACH_DEBUG_PAYLOADS=true
$ poetry run python test_agent.py

# What to look for:

# 1. System prompt was loaded
✓ 📋 Loaded coach prompt: agent01

# 2. Model loaded successfully
✅ Model loaded successfully

# 3. Observation sent to agent
📤 Sending observation to agent:
POPULATION STATE - Generation 25
...

# 4. LLM received prompt (Terminal 1)
type: llm.prediction.input
input: "POPULATION STATE - Generation 25
OVERVIEW:
...

# 5. LLM returned response (Terminal 1)
type: llm.prediction.output
output: {
  "thinking": "I see the population is stagnant...",
  "tool_calls": [
    {
      "name": "analyze_population",
      "arguments": {}
    }
  ]
}

# 6. Tool executed (Terminal 2)
🔧 Executing: analyze_population(group_by='fitness')
✅ analyze_population succeeded

# 7. Agent completed (Terminal 2)
✅ Agent analysis complete
   Iterations: 5
   Tool calls: 5
   Actions: 4
```

---

## Performance Benchmarks

**Expected timings:**
- Model load: 5-15 seconds
- First iteration: 30-60 seconds (includes model warmup)
- Subsequent iterations: 10-30 seconds each
- Tool execution: <1 second each
- Total session: 60-120 seconds

**If slower:**
- Increase GPU offload
- Reduce context length
- Use faster model (gemma-2-9b-it)
- Check CPU/GPU usage

**If faster than expected:**
- Verify model is using GPU (check `nvidia-smi`)
- Check context length is adequate

---

## Checklist for Debugging

Before filing an issue or asking for help:

- [ ] Ran `lms server status` - server is running
- [ ] Ran `lms ps` - model is loaded
- [ ] Checked `lms log stream` - saw actual prompt/response
- [ ] Enabled `COACH_DEBUG_PAYLOADS=true`
- [ ] Reviewed `agent_test_verbose.log`
- [ ] Verified system prompt loads (`coach_prompts/agent01.txt`)
- [ ] Checked Python version (3.10+)
- [ ] Confirmed LM Studio version is recent
- [ ] Tested with different temperature/model
- [ ] Saved full logs for review

---

## Advanced: Prompt Engineering

If agent isn't working well, tune the system prompt:

### 1. Add More Examples

Edit `coach_prompts/agent01.txt`:

```
## Example 1: Gate Crisis

OBSERVATION: 100% below min_trades=20

RESPONSE:
```json
{
  "thinking": "All individuals failing gate. Need to lower threshold.",
  "tool_calls": [
    {"name": "analyze_population", "arguments": {"group_by": "fitness"}},
    {"name": "update_fitness_gates", "arguments": {"min_trades": 5, "reason": "Gate too strict"}},
    {"name": "finish_analysis", "arguments": {"summary": "Lowered gate"}}
  ]
}
```

### 2. Simplify Tool Descriptions

Make tools more concise:

```
1. analyze_population() - Get stats (ALWAYS call first)
2. update_fitness_gates(min_trades=5) - Lower gate
3. finish_analysis() - Complete (ALWAYS call last)
```

### 3. Add Constraints

```
RULES:
- MUST call analyze_population() first
- MUST call finish_analysis() last
- Use 5-7 tools maximum
- Respond ONLY with JSON, no explanation
```

---

## Getting Help

If still stuck:

1. **Save full logs:**
   ```bash
   ./scripts/test_agent_verbose.sh
   # Saves to agent_test_verbose.log
   ```

2. **Get LM Studio logs:**
   ```bash
   lms log stream > lms_full_logs.txt
   # Run test in another terminal
   ```

3. **Check versions:**
   ```bash
   lms --version
   poetry run python --version
   poetry run python -c "import lmstudio; print(lmstudio.__version__)"
   ```

4. **Share relevant sections:**
   - System info
   - Error messages
   - LLM input/output from logs
   - Tool execution results

---

## Summary

**For 99% of issues:**
```bash
# Just run this and watch the logs
./scripts/debug_agent_with_logs.sh
```

**The logs will show you:**
- ✅ Exact prompt sent to model
- ✅ Exact response from model
- ✅ Why JSON parsing failed
- ✅ Why tools weren't called
- ✅ Where the issue is

**Most common fix:**
- Temperature too high → Lower to 0.1
- Prompt too complex → Simplify and add examples
- Context too small → Increase to 8000
- Model too slow → Use gemma-2-9b-it

Happy debugging! 🐛🔧
