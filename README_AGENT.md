# Evolution Coach Agent - Implementation Complete! 🤖

## What Was Implemented

A **full agent-based Evolution Coach** that can steer genetic algorithm evolution through 27 tools with complete GA control.

### Key Components Created

1. **Shell Scripts** (`scripts/`)
   - `lms_load_model.sh` - Load models into LM Studio
   - `lms_unload_model.sh` - Unload models
   - `lms_status.sh` - Check server and model status
   - `lms_server_start.sh` - Start LM Studio server

2. **Coach Tools** (`backtest/coach_tools.py`)
   - 27 tools across 5 categories:
     - **Observability**: `analyze_population`, `get_param_distribution`, `get_param_bounds`
     - **Individual Control**: `mutate_individual`, `drop_individual`, `insert_individual`
     - **GA Steering**: `update_ga_params`, `update_diversity_settings`
     - **Fitness Function**: `update_fitness_gates`, `update_fitness_weights`
     - **Control Flow**: `finish_analysis`

3. **Agent Executor** (`backtest/coach_agent_executor.py`)
   - Multi-step reasoning loop
   - Tool call parsing from LLM responses
   - Conversation history management
   - Iterative observation → action → result cycle

4. **Agent Prompt** (`coach_prompts/agent01.txt`)
   - Complete tool specifications
   - Strategic playbooks for common scenarios
   - Decision frameworks
   - Example reasoning chains

5. **Integration** (`backtest/coach_manager_blocking.py`)
   - `analyze_and_apply_with_agent()` method
   - Agent initialization and lifecycle
   - Observation builder
   - Results tracking

6. **Test Suite** (`test_agent.py`)
   - Complete end-to-end test
   - Model loading/unloading
   - Stagnating population scenario
   - Agent validation

---

## Quick Start

### 1. Install Dependencies

```bash
cd /home/agile/seller_exhaustion
poetry install
```

This adds:
- `openai ^1.54.0`
- `lmstudio` (from GitHub)

### 2. Start LM Studio Server

```bash
./scripts/lms_server_start.sh
```

Or manually:
```bash
lms server start
```

### 3. Run Test

```bash
python test_agent.py
```

This will:
1. ✅ Check if LM Studio server is running
2. ✅ Check if google/gemma-3-12b is loaded
3. ✅ Load model if needed (GPU 60%, 5000 context)
4. ✅ Create stagnating population scenario
5. ✅ Call agent and wait for response
6. ✅ Verify agent took actions
7. ✅ Unload model

**Expected output:**
```
╔════════════════════════════════════════════════════════════════════╗
║                    EVOLUTION COACH AGENT TEST                      ║
╚════════════════════════════════════════════════════════════════════╝

Model: google/gemma-3-12b
GPU Offload: 0.6
Context Length: 5000

1️⃣  Checking LM Studio server...
   ✅ Server is running

2️⃣  Managing model...
🔍 Checking if google/gemma-3-12b is loaded...
   ✅ Model is loaded

3️⃣  Creating test scenario...
🧬 Creating stagnating population scenario...
   Population size: 12
   Generation: 25
   Diversity: 0.087 (VERY LOW)
   Top fitness: 0.3500
   Bottom 9 fitness: 0.0000
   Top 3 trades: [18, 19, 20]
   Bottom 9 trades: [0, 0, 0]

   📊 Stagnation characteristics:
      ✓ Low diversity: 0.087 < 0.15
      ✓ Gate crisis: 100% below min_trades=20
      ✓ Signal failure: 75% with 0 trades
      ✓ Boundary clustering: ema_fast near 48 (min bound)
      ✓ Parameter clustering: vol_z 1.2-2.0 (poor range)

4️⃣  Creating configurations...
   Fitness: min_trades=20, min_win_rate=0.4
   GA: mutation_rate=0.55, immigrant_fraction=0.0

5️⃣  Running agent analysis...
   This will take 30-90 seconds depending on LLM speed...

🤖 Evolution Coach Agent starting...
   Max iterations: 10

🔄 Iteration 1/10
🔧 Executing: analyze_population(group_by='fitness', top_n=5, bottom_n=3)
✅ analyze_population succeeded

🔄 Iteration 2/10
🔧 Executing: update_fitness_gates(min_trades=5)
✅ update_fitness_gates succeeded

🔄 Iteration 3/10
🔧 Executing: update_ga_params(immigrant_fraction=0.15, mutation_rate=0.35)
✅ update_ga_params succeeded

🔄 Iteration 4/10
🔧 Executing: finish_analysis(summary='Reduced min_trades to 5, injected diversity')
✅ finish_analysis succeeded
✅ Agent called finish_analysis - session complete

----------------------------------------------------------------------
✅ Agent completed in 45.2 seconds
----------------------------------------------------------------------

6️⃣  Verifying agent actions...
   ✅ Success: True
   📊 Iterations: 4
   🔧 Tool calls: 4
   📝 Actions: 4

   📋 Actions taken by agent:
      1. analyze_population
         → Analyzed population state
      2. update_fitness_gates
         → min_trades: 20 → 5
      3. update_ga_params
         → immigrant_fraction: 0.0 → 0.15
         → mutation_rate: 0.55 → 0.35
      4. finish_analysis
         → Completed analysis

   🔍 Validation:
      ✅ Agent analyzed population (good start)
      ✅ Agent addressed gate crisis (expected)
      ✅ Agent adjusted GA parameters
      ✅ Agent called finish_analysis

======================================================================
✅ TEST PASSED - Agent successfully analyzed stagnating population!

Expected behavior:
  1. Agent called analyze_population() to understand state
  2. Agent diagnosed gate crisis (100% below min_trades)
  3. Agent lowered min_trades from 20 to 5-10
  4. Agent may have injected immigrants (diversity)
  5. Agent called finish_analysis() to complete
======================================================================
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│          LLM (Gemma 3 12B via LM Studio)                │
│                                                          │
│  System Prompt: agent01.txt                             │
│  - 27 tool specifications                               │
│  - 5 strategic playbooks                                │
│  - Decision frameworks                                  │
│                                                          │
│  Loop: Think → Tool Call → Observe → Think...         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│          AgentExecutor (Python)                         │
│                                                          │
│  • Manages conversation history                         │
│  • Parses tool calls from LLM                           │
│  • Executes tools via CoachToolkit                      │
│  • Max 10 iterations per session                        │
│  • Tracks actions and outcomes                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│          CoachToolkit (Python)                          │
│                                                          │
│  • Implements all 27 tools                              │
│  • Validates inputs                                     │
│  • Executes on live GA state                            │
│  • Returns structured results                           │
│  • Logs all actions                                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Live GA State                                          │
│                                                          │
│  • Population (individuals)                             │
│  • FitnessConfig (gates, weights)                       │
│  • OptimizationConfig (GA params)                       │
│  • Parameter bounds (search space)                      │
└─────────────────────────────────────────────────────────┘
```

---

## The 27 Agent Tools

### Observability (3 shown, 8 total)

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `analyze_population` | Get population stats | Start of every analysis |
| `get_param_distribution` | Analyze specific parameter | Before adjusting bounds |
| `get_param_bounds` | Query search space | Diagnosing clustering |

### Individual Control (3 tools)

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `mutate_individual` | Modify specific individual | Exploit success, test hypotheses |
| `drop_individual` | Remove underperformer | Zero fitness for 5+ gens |
| `insert_individual` | Add strategic individual | Test unexplored regions |

### GA Steering (2 shown, 6 total)

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `update_ga_params` | Evolution mechanics | Stagnation or refinement |
| `update_param_bounds` | Search space | Boundary clustering |

### Fitness Function (2 shown, 9 total)

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `update_fitness_gates` | Hard thresholds | Gate crisis (>80% failing) |
| `update_fitness_weights` | Optimization objectives | Change what we optimize |

### Control Flow (1 tool)

| Tool | Purpose |
|------|---------|
| `finish_analysis` | Complete session and return to GA |

---

## Agent Decision Process

### 1. OBSERVE
```python
# Agent starts by analyzing population
analyze_population(group_by="fitness", top_n=3, bottom_n=3)
```

### 2. DIAGNOSE
Identify the problem:
- Gate crisis? (100% below threshold)
- Premature convergence? (diversity < 0.15)
- Stagnation? (no improvement 10+ gens)
- Boundary clustering? (>30% at bounds)

### 3. ACT
Take strategic actions:
```python
# Gate crisis example
update_fitness_gates(min_trades=5, reason="Top performers at 18-19 trades")
update_ga_params(immigrant_fraction=0.15, reason="Inject diversity")
```

### 4. FINISH
```python
finish_analysis(
    summary="Reduced gates, injected diversity",
    overall_assessment="needs_adjustment"
)
```

---

## Strategic Playbooks

### Playbook 1: Gate Crisis
**Problem**: 100% below min_trades=20

**Actions**:
1. `analyze_population()` - confirm diagnosis
2. `update_fitness_gates(min_trades=5)` - quick relief
3. `get_param_distribution("vol_z")` - check signal generation
4. `update_ga_params(immigrant_fraction=0.15)` - inject diversity
5. `finish_analysis()`

### Playbook 2: Premature Convergence
**Problem**: diversity < 0.15, all similar

**Actions**:
1. `analyze_population()` - what converged on?
2. `get_param_distribution()` - check values
3. `update_ga_params(mutation_rate=0.7, immigrant_fraction=0.3)` - restart
4. `finish_analysis()`

### Playbook 3: Boundary Clustering
**Problem**: 8/12 at ema_fast=48 (minimum)

**Actions**:
1. `get_param_distribution("ema_fast", correlate_with="fitness")`
2. `update_param_bounds("ema_fast", new_min=24)` - expand downward
3. `mutate_individual(id=9, parameter="ema_fast", new_value=36)` - test
4. `update_ga_params(immigrant_fraction=0.15)` - populate new space
5. `finish_analysis()`

---

## Usage in Your Code

### Basic Integration

```python
from backtest.coach_manager_blocking import BlockingCoachManager

# Initialize coach with agent
coach = BlockingCoachManager(
    model="google/gemma-3-12b",
    prompt_version="agent01",
    analysis_interval=10,
    verbose=True
)

# During optimization loop
if coach.should_analyze(generation):
    success, mutations = await coach.analyze_and_apply_with_agent(
        population=population,
        fitness_config=fitness_config,
        ga_config=ga_config,
        current_data=data
    )
    # GA continues with modified population
```

### Advanced: Direct Agent Control

```python
from backtest.coach_tools import CoachToolkit
from backtest.coach_agent_executor import AgentExecutor
from backtest.llm_coach import GemmaCoachClient

# Create toolkit
toolkit = CoachToolkit(
    population=population,
    session=session,
    fitness_config=fitness_config,
    ga_config=ga_config,
    mutation_manager=mutation_manager
)

# Create agent
agent = AgentExecutor(
    llm_client=GemmaCoachClient(model="google/gemma-3-12b", prompt_version="agent01"),
    toolkit=toolkit,
    max_iterations=10
)

# Run analysis
result = await agent.run_analysis(initial_observation)
```

---

## Configuration

### Environment Variables (.env)

```bash
# Coach settings
COACH_ENABLED=true
COACH_MODEL=google/gemma-3-12b
COACH_ANALYSIS_INTERVAL=10
COACH_AGENT_MAX_ITERATIONS=10
COACH_DEBUG_PAYLOADS=false

# LM Studio settings
LM_STUDIO_BASE_URL=http://localhost:1234
LM_STUDIO_GPU_OFFLOAD=0.6
LM_STUDIO_CONTEXT_LENGTH=5000
```

### Model Management

```bash
# Check status
./scripts/lms_status.sh

# Load model
./scripts/lms_load_model.sh google/gemma-3-12b 0.6 5000

# Unload model
./scripts/lms_unload_model.sh google/gemma-3-12b
```

---

## Debugging 🔍

### Quick Debug (Recommended)

See the **exact prompt** sent to the model:

```bash
./scripts/debug_agent_with_logs.sh
```

This runs `lms log stream` alongside the test, showing you:
- ✅ Exact prompt sent to model
- ✅ Exact response from model  
- ✅ Why JSON parsing failed (if it did)
- ✅ Why tools weren't called (if they weren't)

**Example output:**
```
type: llm.prediction.input
input: "POPULATION STATE - Generation 25
OVERVIEW:
- Population size: 12
- Mean fitness: 0.0938
...
```

### Verbose Python Logging

```bash
./scripts/test_agent_verbose.sh
```

Saves full logs to `agent_test_verbose.log`.

### Manual Debugging

Terminal 1:
```bash
lms log stream
```

Terminal 2:
```bash
export COACH_DEBUG_PAYLOADS=true
poetry run python test_agent.py
```

**See [DEBUG_AGENT.md](DEBUG_AGENT.md) for complete debugging guide.**

---

## Troubleshooting

### Model Not Loading

```bash
# Check if model is downloaded
lms ls | grep gemma-3-12b

# Download if needed
lms get google/gemma-3-12b --yes
```

### Server Not Responding

```bash
# Check server status
lms server status

# Restart server
lms server stop
lms server start
```

### Agent Timeout

```bash
# Increase timeout in .env
COACH_RESPONSE_TIMEOUT=7200  # 2 hours

# Or use faster model
COACH_MODEL=google/gemma-2-9b-it
```

### Debug Mode

```bash
# Enable full payload logging
export COACH_DEBUG_PAYLOADS=true

# Run test
python test_agent.py
```

---

## Performance

### Expected Timings

- **Agent Analysis**: 30-90 seconds
- **Tool Execution**: <1 second each
- **Total per Session**: 1-2 minutes
- **Memory Usage**: ~8GB GPU for gemma-3-12b

### Optimization

- Use `gemma-2-9b-it` for faster responses
- Reduce `COACH_AGENT_MAX_ITERATIONS` to 7
- Increase `COACH_ANALYSIS_INTERVAL` to 15-20

---

## Success Metrics

The agent is successful if:

1. ✅ **Diagnostic Accuracy**: 90%+ correct problem identification
2. ✅ **Intervention Success**: 80%+ actions lead to improvement
3. ✅ **Performance Gain**: 20%+ better than single-shot approach
4. ✅ **Stability**: 95%+ interventions don't cause failures
5. ✅ **Efficiency**: <7 tool calls per session

---

## Next Steps

1. **Run the test**: `python test_agent.py`
2. **Review output**: Check agent's reasoning and actions
3. **Integrate into optimization**: Use in your GA loops
4. **Monitor results**: Track fitness improvements
5. **Tune playbooks**: Adjust strategies based on results

---

## Files Created

```
seller_exhaustion/
├── scripts/
│   ├── lms_load_model.sh          ✅ Load models
│   ├── lms_unload_model.sh        ✅ Unload models
│   ├── lms_status.sh              ✅ Check status
│   └── lms_server_start.sh        ✅ Start server
├── backtest/
│   ├── coach_tools.py             ✅ 27 agent tools
│   ├── coach_agent_executor.py    ✅ Agent loop
│   └── coach_manager_blocking.py  ✅ Integration (updated)
├── coach_prompts/
│   └── agent01.txt                ✅ Agent system prompt
├── test_agent.py                  ✅ End-to-end test
├── pyproject.toml                 ✅ Dependencies added
└── README_AGENT.md                ✅ This file
```

---

## Summary

✅ **Agent-based Evolution Coach is complete and ready to use!**

Key capabilities:
- 27 tools for full GA control
- Strategic playbooks for common scenarios
- Iterative reasoning with LLM
- Complete test suite
- Shell scripts for model management

**Next**: Run `python test_agent.py` to see it in action! 🚀
