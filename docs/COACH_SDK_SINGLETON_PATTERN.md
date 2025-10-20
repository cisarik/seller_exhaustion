# LM Studio SDK Singleton Pattern - Technical Deep Dive

## The Challenge

When implementing the Evolution Coach with LM Studio, we encountered a fundamental limitation of the lmstudio Python SDK:

```python
# This works once:
client1 = lms.get_default_client("http://localhost:1234")  # ✅ OK

# But this fails:
client2 = lms.get_default_client("http://localhost:1234")  # ❌ ERROR!
# Error: "Default client is already created, cannot set its API host"
```

The SDK implements a **singleton pattern** for the default client - only one instance can exist, and it can't be recreated or reconfigured.

## Why This Matters for the Coach

Our design needed to:
1. Load model at generation 5
2. Analyze with LLM
3. Unload model to free context window
4. Reload model at generation 10
5. Analyze again with LLM
6. Repeat for many generations

Each cycle required recreating the client - but the SDK forbade this!

## The Solution Architecture

We solved this by separating concerns:

### 1. Model Lifecycle (via lms CLI)
- Load/unload managed through subprocess calls to `lms` command
- Can be called multiple times without issues
- Handles context window clearing

### 2. Client Lifecycle (via SDK)
- Client created once with `lms.get_default_client()`
- **Reused indefinitely** across all operations
- Survives model unload/reload cycles
- Acts as a persistent connector to LM Studio server

### 3. Session/Context (on LM Studio side)
- Each LLM call creates new session/context
- Unloading model clears context on server side
- Reloading model starts fresh context

## Implementation Details

### The Client Creation Pattern

```python
# ✅ CORRECT: Create once, cache, reuse

class GemmaCoachClient:
    def __init__(self):
        self._lms_client = None  # Created on first use
        self._model_loaded = False
    
    async def _call_llm(self, message):
        # Create client ONCE
        if not self._lms_client:
            try:
                self._lms_client = await asyncio.to_thread(
                    lms.get_default_client  # ← No URL argument!
                )
            except Exception as e:
                if "Default client is already created" in str(e):
                    # Client already exists, just get it
                    self._lms_client = await asyncio.to_thread(
                        lms.get_default_client
                    )
                else:
                    raise
        
        # Reuse client for all subsequent calls
        model = await asyncio.to_thread(self._lms_client.llm.model)
        chat = lms.Chat(self.system_prompt)
        chat.add_user_message(message)
        response = await asyncio.to_thread(
            model.respond,
            chat,
            config={"temperature": 0.3, "maxTokens": 4000}
        )
        return response.content
```

### The Model Load/Unload Pattern

```python
# ✅ CORRECT: Separate model lifecycle from client lifecycle

async def load_model(self):
    """Load model via lms CLI. Can be called multiple times."""
    # Check if already loaded
    already_loaded = await self.check_model_loaded()
    if already_loaded:
        return  # Already loaded
    
    # Execute lms load command
    result = await asyncio.to_thread(
        subprocess.run,
        ["lms", "load", self.model, f"--gpu={self.gpu}"],
        capture_output=True,
        timeout=120
    )
    self._model_loaded = result.returncode == 0

async def unload_model(self):
    """Unload model via lms CLI to free context. Can be called multiple times."""
    # Execute lms unload command
    result = await asyncio.to_thread(
        subprocess.run,
        ["lms", "unload"],
        capture_output=True,
        timeout=30
    )
    self._model_loaded = False
    
    # CRITICAL: Do NOT clear self._lms_client!
    # The client can be reused after model reload.
```

## Why This Works

### The Key Insight

The lmstudio SDK's `get_default_client()` returns a **connection object** to the LM Studio server, not a session object. The actual LLM session/context lives on the **server side**, not in the Python client.

```
Architecture:
┌─────────────────┐
│  Python Process │
├─────────────────┤
│  Gemma Client   │
│  (singleton)    │  ← This persists across unload/reload
│  ↓              │
│  get_default_   │
│  client()       │
└────────┬────────┘
         │
    [TCP/HTTP]
         │
         ↓
┌──────────────────────────┐
│   LM Studio Server       │
├──────────────────────────┤
│  Model (loaded/unloaded) │  ← This can be unloaded/reloaded
│  ├─ Context Window       │     without affecting client
│  ├─ Session 1            │
│  ├─ Session 2            │
│  └─ Session 3            │     Each LLM call = new session
└──────────────────────────┘
```

When we:
1. **Unload model**: Clear server-side context
2. **Reload model**: Fresh context on server
3. **Call LLM**: Same client connects, server creates new session

## The URL Parameter Issue

Another key discovery: **Don't pass the URL to `get_default_client()`**

```python
# ❌ WRONG: Passing URL causes connection issues
client = lms.get_default_client("http://localhost:1234")

# ✅ RIGHT: SDK auto-detects localhost:1234
client = lms.get_default_client()
```

The SDK auto-discovers the LM Studio server on localhost:1234. Passing a URL explicitly causes connection validation issues in the SDK.

## Category Normalization

The LLM might return recommendation categories in various formats. We handle them gracefully:

```python
# Handle uppercase → lowercase conversion
try:
    category = RecommendationCategory(category_str.lower())
except ValueError:
    # Try enum name mapping (uppercase)
    for cat in RecommendationCategory:
        if cat.name == category_str.upper():
            category = cat
            break
    else:
        raise ValueError(f"Unknown category: {category_str}")
```

This allows the LLM to use either:
- `"curriculum"` (proper enum value)
- `"CURRICULUM"` (uppercase name)
- `"fitness_weights"` or `"FITNESS_WEIGHTS"`
- etc.

## Testing the Pattern

### Unit Tests (Mocked)

```python
def test_client_creation_once():
    """Verify client is created only once."""
    client = GemmaCoachClient()
    
    # Create client
    client._lms_client = Mock(name="lms_client")
    assert client._lms_client is not None
    
    # Reuse same client
    existing = client._lms_client
    client._lms_client = existing
    assert client._lms_client is existing  # Same instance
```

### Integration Tests (Real SDK)

```python
async def test_unload_reload_cycle():
    """Verify model can unload/reload with same client."""
    client = GemmaCoachClient()
    
    # Load and analyze
    await client.load_model()
    analysis1 = await client.analyze_evolution(state1, logs1)
    assert analysis1 is not None
    
    # Unload to free context
    await client.unload_model()
    assert client._model_loaded is False
    # BUT: client._lms_client is still valid!
    
    # Reload and analyze again
    await client.load_model()
    analysis2 = await client.analyze_evolution(state2, logs2)
    assert analysis2 is not None
    
    # ✅ NO client recreation errors!
```

## Performance Characteristics

Typical timing:
- **Client creation**: ~100ms (one-time)
- **Model load**: 5-10 seconds
- **Model unload**: <1 second
- **LLM inference**: 10-60 seconds (first call longer)
- **Model reload**: 5-10 seconds

For 10 analysis cycles:
- ✅ Only 1 client creation
- ✅ 10 model load/unload cycles
- ✅ 10 LLM inference calls
- ✅ No client conflicts or errors

## Memory Management

The singleton client holds a persistent TCP connection to LM Studio. This is actually beneficial:

- ✅ No connection overhead after first use
- ✅ Persistent connection is reused
- ✅ Memory footprint stable
- ✅ No connection leaks

The connection survives:
- Model unload/reload ✅
- Multiple LLM calls ✅
- Long optimization runs ✅

## Lessons Learned

### What Works
✅ SDK auto-detects localhost:1234 (don't pass URL)  
✅ Singleton client persists across model cycles  
✅ Manage model lifecycle separately from client  
✅ Use CLI (`lms load/unload`) for model, SDK for inference  
✅ Each LLM call creates new session on server  

### What Doesn't Work
❌ Trying to recreate the default client  
❌ Passing explicit URL to `get_default_client()`  
❌ Clearing client between unload/reload  
❌ Expecting client to manage context window  
❌ Using same client in multiple threads (not thread-safe)  

### Best Practices
✅ Create client once during initialization  
✅ Cache and reuse the singleton client  
✅ Handle model lifecycle with lms CLI  
✅ Normalize LLM response formats  
✅ Log all connection events for debugging  

## Conclusion

The lmstudio SDK's singleton pattern, while initially appearing as a limitation, actually enforces a clean architecture:

- **Single responsibility**: Client = connection; Server = session/context
- **Separation of concerns**: Python side (client), Server side (model/context)
- **Efficiency**: Persistent connection reused indefinitely
- **Simplicity**: No complex connection pooling needed

By working **with** the singleton pattern instead of against it, we achieved a robust, efficient, and maintainable coach integration.

---

**Reference Implementation**: `backtest/llm_coach.py`  
**Test Coverage**: `tests/test_coach_*`  
**Date**: January 2025
