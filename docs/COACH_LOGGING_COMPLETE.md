# Evolution Coach Logging System - Complete Implementation

## Overview

The Evolution Coach now has comprehensive logging to the dedicated "Evolution Coach" window, showing exactly what logs are trimmed and sent to the coach, plus all parameter changes when recommendations are applied.

## Log Categories

All logs in the Evolution Coach window are prefixed with categories for easy filtering:

### `[Gen XXX]` - Generation Logs
Every log line from the GA evolution is sent to the coach window with generation number:
```
[Gen  10] Best fitness: 0.4523
[Gen  10] Mean fitness: 0.2134 ± 0.0876
[Gen  11] Population initialized from seed
```

### `[TRIM   ]` - Log Trimming
When logs are trimmed to last N generations (default 25):
```
[TRIM   ] Trimmed 142 old log lines (keeping last 25 gens)
```

### `[COACH  ]` - Coach Operations
All coach analysis operations:
```
[COACH  ] 🤖 Triggering coach analysis for Gen 10
[COACH  ] Sending last 25 generations to coach
[COACH  ] Sending evolution state:
[COACH  ]   - Mean fitness: 0.2134 ± 0.0876
[COACH  ]   - Best fitness: 0.4523
[COACH  ]   - Below min_trades: 23.4%
[COACH  ]   - Diversity: 0.67
[COACH  ]   - Stagnant: False
[COACH  ]   - Log lines: 87
[COACH  ] 📤 Sending 4523 chars to LLM
[COACH  ] ⏳ Waiting for coach response...
[COACH  ] ✅ Coach response received:
[COACH  ]   - Assessment: PROGRESS
[COACH  ]   - Recommendations: 3
[COACH  ]   - Stagnation: No
[COACH  ]   - Diversity concern: No
[COACH  ]   1. mutation_rate: 0.3 → 0.25 (confidence: 80%)
[COACH  ]   2. min_trades: 10 → 15 (confidence: 90%)
[COACH  ]   3. mutation_sigma: 0.1 → 0.08 (confidence: 75%)
[COACH  ] 📥 Received 1234 chars from LLM
[COACH  ]   - Tokens: 456
[COACH  ]   - Time to first token: 3.45s
```

### `[PARAMS ]` - Parameter Changes
When coach recommendations are applied:
```
[PARAMS ] Current fitness config:
[PARAMS ]   - fitness_type: balanced
[PARAMS ]   - min_trades: 10
[PARAMS ]   - min_win_rate: 0.4
[PARAMS ] 🔧 Applying 3 recommendations...
[PARAMS ] ✅ ga.mutation_rate: 0.3 → 0.25
[PARAMS ] ✅ fitness.min_trades: 10 → 15
[PARAMS ] ✅ ga.mutation_sigma: 0.1 → 0.08
[PARAMS ] 📊 Applied 3/3 recommendations
```

### `[LMS    ]` - LM Studio CLI Operations
Model loading/unloading via `lms` CLI:
```
[LMS    ] 📦 Loading model: google/gemma-3-12b
[LMS    ]   - GPU: auto
[LMS    ]   - Context length: 131072
[LMS    ] ✅ Model loaded successfully
[LMS    ] 🗑️  Unloading model
[LMS    ] ✅ Model unloaded successfully
```

## LM Studio Integration

### Model Loading/Unloading (lms CLI)

Uses `lms` CLI commands for model management:

**Load Model:**
```bash
lms load google/gemma-3-12b --gpu=auto --context-length=131072
```

**Unload Model:**
```bash
lms unload
```

### Inference (lmstudio Python SDK)

Uses `lmstudio` Python SDK for actual LLM calls:

```python
import lmstudio as lms

# Connect to loaded model
client = lms.get_default_client("http://localhost:1234")
model = client.llm.model()

# Create chat
chat = lms.Chat(system_prompt)
chat.add_user_message(user_message)

# Generate response
response = model.respond(chat, config={
    "temperature": 0.3,
    "maxTokens": 4000
})

# Extract content
text = response.content
```

## Configuration

### Settings in .env

```bash
# Evolution Coach Parameters
COACH_MODEL=google/gemma-3-12b           # Model identifier
COACH_PROMPT_VERSION=async_coach_v1      # Prompt version
COACH_FIRST_ANALYSIS_GENERATION=10       # First analysis at Gen N
COACH_MAX_LOG_GENERATIONS=25             # Keep last N generations
COACH_AUTO_RELOAD_MODEL=true             # Auto reload after recommendations
COACH_CONTEXT_LENGTH=131072              # Gemma's maximum context
COACH_GPU=auto                           # GPU usage: max, auto, 0.0-1.0
```

### Context Window

- **Maximum**: 131072 tokens (Gemma 3's full context window)
- **Typical usage**: ~20-30% of context for evolution logs
- **Headroom**: Plenty of space for large populations and long runs

### GPU Configuration

- `max`: Use maximum GPU (100% offload)
- `auto`: Let LM Studio decide based on available VRAM
- `0.0` to `1.0`: Specific fraction (e.g., `0.75` = 75% GPU)

## UI Integration

### Load Model Button

1. **Initial State**: "Load Model" (green)
2. **Click**: Button changes to "Loading..." (orange, disabled)
3. **Loading**: Status bar shows "📦 Loading model: google/gemma-3-12b..."
4. **Success**: Button becomes "Unload Model" (red), status shows "✅ Model loaded"
5. **Error**: Shows dialog with error message

### Evolution Coach Window

Open via menu: **Tools → Evolution Coach**

The window shows:
- All generation logs with trimming
- Coach analysis triggers and responses
- Parameter changes applied
- LM Studio operations
- Real-time streaming as events occur

### Status Bar Indicators

Main window status bar shows coach state:
- "📦 Loading model..." - Model loading in progress
- "✅ Model loaded: google/gemma-3-12b" - Ready for analysis
- "🗑️ Model unloaded" - Model removed from memory
- "❌ Failed to load model" - Error occurred

## Workflow

### Complete Evolution Run with Coach

```
Gen 1-9:   Normal GA evolution
           Logs stream to Evolution Coach window with [Gen XXX] prefix

Gen 10:    COACH_FIRST_ANALYSIS_GENERATION reached
           [COACH  ] 🤖 Triggering coach analysis for Gen 10
           [COACH  ] Sending last 25 generations to coach
           Logs show exactly what's sent to coach
           
           [COACH  ] ⏳ Waiting for coach response...
           [COACH  ] ✅ Coach response received
           [COACH  ]   - Recommendations: 3
           
Gen 11-12: Evolution continues (non-blocking)

Gen 13:    Recommendations applied
           [PARAMS ] 🔧 Applying 3 recommendations...
           [PARAMS ] ✅ ga.mutation_rate: 0.3 → 0.25
           [PARAMS ] ✅ fitness.min_trades: 10 → 15
           [PARAMS ] ✅ ga.mutation_sigma: 0.1 → 0.08
           [PARAMS ] 📊 Applied 3/3 recommendations
           
           Model unload/reload
           [LMS    ] 🗑️  Unloading model
           [LMS    ] ✅ Model unloaded successfully
           [LMS    ] 📦 Loading model: google/gemma-3-12b
           [LMS    ] ✅ Model loaded successfully
           
           Trigger flag set for next analysis

Gen 14:    Next coach analysis triggered (because model reloaded)
           [COACH  ] 🤖 Triggering coach analysis for Gen 14
           Process repeats...
```

## Benefits

### 1. Full Transparency
See exactly what the coach sees:
- Which logs are sent
- How logs are trimmed
- Current parameter values

### 2. Parameter Tracking
Every parameter change is logged:
- Old value → New value
- Which category (fitness, GA, bounds)
- Success/failure status

### 3. Debug Support
Complete trace of coach operations:
- When analysis triggered
- What was sent
- What was received
- How long it took
- Token usage

### 4. Memory Management
LM Studio operations logged:
- Model load time
- GPU configuration
- Unload confirmation
- Error messages

## Files Modified

### Core Implementation
- `backtest/llm_coach.py` - LM Studio integration (lms CLI + Python SDK)
- `backtest/coach_manager.py` - Coach orchestration with logging
- `backtest/coach_integration.py` - Parameter application with logging
- `core/coach_logging.py` - Existing log manager (no changes needed)

### Configuration
- `.env.example` - Added COACH_CONTEXT_LENGTH, COACH_GPU
- `config/settings.py` - Added coach_context_length, coach_gpu fields

### UI
- `app/main.py` - Wired Load/Unload button signals
- `app/widgets/compact_params.py` - Coach config controls
- `app/widgets/evolution_coach.py` - Existing log window (no changes needed)

## Testing

### Manual Test Checklist

1. **Open Evolution Coach Window**
   - Menu → Tools → Evolution Coach
   - Window shows existing logs

2. **Load Model**
   - Click "Load Model" button
   - Watch [LMS    ] logs appear
   - Button changes to "Unload Model"

3. **Run Evolution**
   - Run backtest to populate data
   - Initialize GA population
   - Click "Step" repeatedly
   - Watch [Gen XXX] logs appear in coach window

4. **Trigger Analysis**
   - Step to Gen 10 (COACH_FIRST_ANALYSIS_GENERATION)
   - Watch [COACH  ] logs for analysis trigger
   - See evolution state being sent
   - Wait for response

5. **Apply Recommendations**
   - Continue stepping
   - When recommendations applied, watch [PARAMS ] logs
   - See exact parameter changes
   - Verify model reload [LMS    ] logs

6. **Verify Trimming**
   - Continue to Gen 40+
   - Watch [TRIM   ] logs when old generations removed
   - Verify only last 25 gens kept

### Expected Log Output

See the workflow section above for complete example of expected log output.

## Troubleshooting

### No [LMS    ] logs
- Check if `lms` CLI is in PATH
- Run `lms --version` in terminal
- Install LM Studio if missing

### No [COACH  ] logs
- Check if coach model loaded
- Verify COACH_FIRST_ANALYSIS_GENERATION in .env
- Check Evolution Coach window is open

### No [PARAMS ] logs
- Verify recommendations were received
- Check if COACH_AUTO_RELOAD_MODEL=true
- Look for error logs in coach window

### Context window overflow
- Increase COACH_MAX_LOG_GENERATIONS (more history)
- Or decrease it (less context, more recent focus)
- Monitor [TRIM   ] logs to see trimming frequency

## Future Enhancements

1. **Log Filtering**: Add UI controls to filter by category
2. **Export Logs**: Save coach logs to file for analysis
3. **Statistics**: Show coach performance metrics (response time, success rate)
4. **Visualization**: Chart parameter changes over time
5. **A/B Testing**: Compare coach recommendations vs no-coach runs
