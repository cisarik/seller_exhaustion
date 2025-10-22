# Agent-Based Evolution Coach Design

## Current Issues

1. **Validation Error**: LLM returns "MUTATIONS" category but validation rejects it
2. **Limited Logging**: Full prompt + payload not logged for debugging
3. **Limited Capabilities**: Need agent with tools to:
   - Mutate specific individuals
   - Drop individuals
   - Insert coach-designed individuals
   - Access population analytics
   - Query parameter distributions

## Proposed Architecture: Tool-Based Agent

Transform the coach from a single-shot JSON analyzer to an **agent with tools** that can:
- Query population data
- Analyze parameter distributions
- Mutate individuals
- Drop individuals
- Insert individuals
- Adjust GA hyperparameters

### Core Concept

```
┌──────────────────────────────────────────────────────────────┐
│                    Evolution Coach Agent                      │
│                                                               │
│  Input: Frozen Population State                              │
│  Output: Series of tool calls + reasoning                    │
│  Flow: Think → Tool Call → Observe → Think → Tool Call...   │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
           ┌────────────────────────────────┐
           │         Available Tools         │
           ├────────────────────────────────┤
           │ • analyze_population()          │
           │ • get_param_distribution()      │
           │ • mutate_individual()           │
           │ • drop_individual()             │
           │ • insert_individual()           │
           │ • update_ga_params()            │
           │ • update_fitness_gates()        │
           │ • finish_analysis()             │
           └────────────────────────────────┘
```

## Tool Definitions

### 1. `analyze_population`
**Purpose**: Get population-level statistics and insights

**Input**:
```python
{
    "group_by": "fitness" | "trade_count" | "win_rate",
    "top_n": 5,  # Show top N individuals
    "bottom_n": 3  # Show bottom N individuals
}
```

**Output**:
```python
{
    "mean_fitness": 0.0897,
    "std_fitness": 0.1559,
    "diversity": 0.12,
    "best_individual": {
        "id": 9,
        "fitness": 0.3972,
        "parameters": {...},
        "metrics": {...}
    },
    "worst_individual": {...},
    "below_gates": 12,  # Count below min_trades
    "clustering": {
        "ema_fast": {"mean": 52.3, "std": 8.1, "range": [48, 77]},
        "vol_z": {"mean": 1.35, "std": 0.18, "range": [1.0, 1.79]}
    }
}
```

### 2. `get_param_distribution`
**Purpose**: Analyze distribution of specific parameter

**Input**:
```python
{
    "parameter_name": "ema_fast",
    "bins": 5,  # Histogram bins
    "correlate_with": "fitness"  # Optional correlation analysis
}
```

**Output**:
```python
{
    "parameter": "ema_fast",
    "min": 48,
    "max": 77,
    "mean": 53.2,
    "std": 7.8,
    "median": 50,
    "histogram": {
        "48-54": 7,  # 7 individuals in this range
        "55-61": 3,
        "62-68": 1,
        "69-75": 0,
        "76-82": 1
    },
    "correlation_with_fitness": 0.12,  # Pearson correlation
    "top_performers_avg": 48.7,  # Avg value in top 3 individuals
    "bottom_performers_avg": 62.3
}
```

### 3. `mutate_individual`
**Purpose**: Mutate specific parameter of specific individual

**Input**:
```python
{
    "individual_id": 9,
    "parameter_name": "ema_fast",
    "new_value": 60,
    "reason": "Extend EMA for better trend detection"
}
```

**Output**:
```python
{
    "success": true,
    "individual_id": 9,
    "parameter": "ema_fast",
    "old_value": 48,
    "new_value": 60,
    "will_be_re_evaluated": true
}
```

### 4. `drop_individual`
**Purpose**: Remove individual from population

**Input**:
```python
{
    "individual_id": 3,
    "reason": "Zero trades, no signal generation"
}
```

**Output**:
```python
{
    "success": true,
    "individual_id": 3,
    "old_fitness": 0.0,
    "population_size_after": 11
}
```

### 5. `insert_individual`
**Purpose**: Insert new individual with specified parameters

**Input**:
```python
{
    "strategy": "coach_designed" | "random" | "clone_best",
    "parameters": {  # Only for coach_designed
        "ema_fast": 96,
        "ema_slow": 672,
        "vol_z": 2.5,
        "tr_z": 1.2,
        ...
    },
    "clone_from_id": 9,  # Only for clone_best
    "reason": "Explore high volume threshold region"
}
```

**Output**:
```python
{
    "success": true,
    "new_individual_id": 12,
    "parameters": {...},
    "population_size_after": 13
}
```

### 6. `update_ga_params`
**Purpose**: Update GA hyperparameters (mutation, selection, elitism)

**Input**:
```python
{
    "mutation_rate": 0.3,           # How much parameters mutate
    "mutation_probability": 0.8,    # Chance of mutation
    "sigma": 0.15,                  # Mutation step size
    "elite_fraction": 0.25,         # Top % preserved
    "tournament_size": 3,           # Selection pressure
    "immigrant_fraction": 0.1       # Random injection rate
}
```

**Output**:
```python
{
    "success": true,
    "changes_applied": {
        "mutation_rate": {"old": 0.55, "new": 0.3},
        "immigrant_fraction": {"old": 0.0, "new": 0.1}
    },
    "unchanged": ["elite_fraction", "tournament_size"]
}
```

### 7. `update_fitness_gates`
**Purpose**: Update fitness gate requirements (hard thresholds)

**Input**:
```python
{
    "min_trades": 5,
    "min_win_rate": 0.40
}
```

**Output**:
```python
{
    "success": true,
    "changes_applied": {
        "min_trades": {"old": 20, "new": 5}
    }
}
```

### 8. `update_fitness_weights`
**Purpose**: Adjust fitness function weights (what we optimize for)

**Input**:
```python
{
    "trade_count_weight": 0.1,
    "win_rate_weight": 0.2,
    "avg_r_weight": 0.3,
    "total_pnl_weight": 0.3,
    "max_drawdown_penalty": 0.1
}
```

**Output**:
```python
{
    "success": true,
    "changes_applied": {
        "trade_count_weight": {"old": 0.0, "new": 0.1}
    }
}
```

### 9. `update_fitness_penalties`
**Purpose**: Adjust soft penalty strengths (for soft_penalties mode)

**Input**:
```python
{
    "penalty_trades_strength": 0.5,  # 0.0-1.0
    "penalty_wr_strength": 0.3       # 0.0-1.0
}
```

**Output**:
```python
{
    "success": true,
    "changes_applied": {
        "penalty_trades_strength": {"old": 0.7, "new": 0.5}
    }
}
```

### 10. `switch_fitness_mode`
**Purpose**: Switch between hard_gates and soft_penalties

**Input**:
```python
{
    "mode": "soft_penalties" | "hard_gates"
}
```

**Output**:
```python
{
    "success": true,
    "old_mode": "hard_gates",
    "new_mode": "soft_penalties",
    "impact": "Individuals below gates will now receive penalty instead of -100"
}
```

### 11. `enable_curriculum`
**Purpose**: Enable/configure curriculum learning

**Input**:
```python
{
    "enabled": true,
    "start_min_trades": 5,
    "increase_per_gen": 2,
    "checkpoint_gens": 5
}
```

**Output**:
```python
{
    "success": true,
    "curriculum_enabled": true,
    "schedule": "Gen 0: min_trades=5, Gen 5: min_trades=15, Gen 10: min_trades=25..."
}
```

### 12. `update_diversity_settings`
**Purpose**: Adjust diversity preservation mechanisms

**Input**:
```python
{
    "track_diversity": true,
    "diversity_target": 0.3,
    "stagnation_threshold": 5,
    "stagnation_fitness_tolerance": 0.01
}
```

**Output**:
```python
{
    "success": true,
    "changes_applied": {
        "stagnation_threshold": {"old": 10, "new": 5}
    }
}
```

### 13. `get_param_bounds`
**Purpose**: Query current parameter bounds for random individual generation

**Input**:
```python
{
    "parameters": ["ema_fast", "ema_slow", "vol_z"]  # Optional: specific params, or null for all
}
```

**Output**:
```python
{
    "success": true,
    "bounds": {
        "ema_fast": {"min": 48, "max": 144, "type": "int"},
        "ema_slow": {"min": 336, "max": 1008, "type": "int"},
        "z_window": {"min": 336, "max": 1008, "type": "int"},
        "atr_window": {"min": 48, "max": 144, "type": "int"},
        "vol_z": {"min": 1.0, "max": 2.0, "type": "float"},
        "tr_z": {"min": 0.8, "max": 1.5, "type": "float"},
        "cloc_min": {"min": 0.3, "max": 0.6, "type": "float"},
        "fib_swing_lookback": {"min": 48, "max": 192, "type": "int"},
        "fib_swing_lookahead": {"min": 3, "max": 10, "type": "int"},
        "fib_target_level": {"values": [0.236, 0.382, 0.5, 0.618, 0.786, 1.0], "type": "discrete"},
        "atr_stop_mult": {"min": 0.5, "max": 1.0, "type": "float"},
        "reward_r": {"min": 1.5, "max": 3.0, "type": "float"},
        "max_hold": {"min": 48, "max": 192, "type": "int"},
        "fee_bp": {"min": 2.0, "max": 10.0, "type": "float"},
        "slippage_bp": {"min": 2.0, "max": 10.0, "type": "float"}
    },
    "population_clustering": {
        "ema_fast": {"at_min": 7, "at_max": 0, "in_middle": 5},
        "vol_z": {"at_min": 2, "at_max": 1, "in_middle": 9}
    }
}
```

### 14. `update_param_bounds`
**Purpose**: Update search space bounds for parameter generation

**Input**:
```python
{
    "parameter": "ema_fast",
    "new_min": 24,           # Optional: update min
    "new_max": 192,          # Optional: update max
    "reason": "Population clustering at lower bound, expand search space downward"
}
```

**Output**:
```python
{
    "success": true,
    "parameter": "ema_fast",
    "old_bounds": {"min": 48, "max": 144},
    "new_bounds": {"min": 24, "max": 192},
    "affected_operations": [
        "Random individual generation will use new bounds",
        "Mutations will respect new bounds",
        "7 existing individuals currently at old boundaries"
    ]
}
```

### 15. `update_multiple_bounds`
**Purpose**: Update bounds for multiple parameters at once

**Input**:
```python
{
    "bounds": {
        "ema_fast": {"min": 24, "max": 192},
        "ema_slow": {"min": 240, "max": 1200},
        "vol_z": {"min": 1.0, "max": 3.0}
    },
    "reason": "Expand search space globally - population converged too early"
}
```

**Output**:
```python
{
    "success": true,
    "changes": [
        {"param": "ema_fast", "old": [48, 144], "new": [24, 192]},
        {"param": "ema_slow", "old": [336, 1008], "new": [240, 1200]},
        {"param": "vol_z", "old": [1.0, 2.0], "new": [1.0, 3.0]}
    ],
    "total_changed": 3
}
```

### 16. `reset_param_bounds`
**Purpose**: Reset specific parameter bounds to default for timeframe

**Input**:
```python
{
    "parameters": ["ema_fast", "vol_z"],  # Or null for all parameters
    "reason": "Coach overrides didn't help, reverting to defaults"
}
```

**Output**:
```python
{
    "success": true,
    "reset": [
        {"param": "ema_fast", "reverted_to": [48, 144]},
        {"param": "vol_z", "reverted_to": [1.0, 2.0]}
    ]
}
```

### 17. `shift_param_bounds`
**Purpose**: Shift bounds window to different region (useful for exploration)

**Input**:
```python
{
    "parameter": "ema_fast",
    "shift_by": 48,          # Shift both min and max by this amount
    "preserve_width": true,  # Keep same range width
    "reason": "Explore faster EMA region (24-120 instead of 48-144)"
}
```

**Output**:
```python
{
    "success": true,
    "parameter": "ema_fast",
    "old_bounds": [48, 144],
    "new_bounds": [24, 120],
    "width_preserved": true,
    "shift_amount": -24
}
```

### 18. `finish_analysis`
**Purpose**: Complete analysis and return to GA

**Input**:
```python
{
    "summary": "Reduced min_trades to 5, mutated individuals 9 and 10, injected 10% immigrants",
    "overall_assessment": "needs_adjustment" | "positive" | "neutral",
    "stagnation_detected": false,
    "diversity_concern": true
}
```

**Output**:
```python
{
    "success": true,
    "total_actions_taken": 8,
    "breakdown": {
        "individuals_mutated": 2,
        "individuals_dropped": 0,
        "individuals_inserted": 0,
        "ga_params_changed": 2,
        "fitness_gates_changed": 1,
        "fitness_weights_changed": 0,
        "fitness_mode_switched": false,
        "curriculum_enabled": false,
        "diversity_settings_changed": 0,
        "bounds_overrides": 0
    }
}
```

## Tool Categories

All GA modifications go through tools:

### Population-Level Tools (Individuals)
- `analyze_population()` - Read
- `get_param_distribution()` - Read
- `mutate_individual()` - Write
- `drop_individual()` - Write
- `insert_individual()` - Write

### GA Algorithm Tools (Evolution Mechanics)
- `update_ga_params()` - mutation_rate, tournament_size, elite_fraction, immigrants
- `update_diversity_settings()` - stagnation detection, diversity targets

### Parameter Search Space Tools (Bounds Management)
- `get_param_bounds()` - Query current bounds and clustering
- `update_param_bounds()` - Update single parameter bounds
- `update_multiple_bounds()` - Update multiple bounds at once
- `reset_param_bounds()` - Revert to defaults
- `shift_param_bounds()` - Shift search window to new region

### Fitness Function Tools (What We Optimize)
- `update_fitness_gates()` - min_trades, min_win_rate (hard thresholds)
- `update_fitness_weights()` - trade_count_weight, pnl_weight, etc.
- `update_fitness_penalties()` - penalty strengths for soft mode
- `switch_fitness_mode()` - hard_gates ↔ soft_penalties
- `enable_curriculum()` - progressive gate relaxation

### Control Flow Tools
- `finish_analysis()` - Complete and return

## Agent Workflow

```python
# Phase 1: Analysis
response_1 = agent.think_and_act("""
You are analyzing a frozen population state.

Population has 12 individuals, mean fitness=0.09, best=0.397.
100% below min_trades=20, diversity=0.12.

Start by analyzing the population.
""")

# Agent returns tool call:
{
    "tool": "analyze_population",
    "args": {"group_by": "fitness", "top_n": 3, "bottom_n": 3}
}

# Phase 2: Parameter analysis
response_2 = agent.think_and_act("""
Analysis shows:
- Top 3 individuals: fitness 0.397, 0.340, 0.339
- All have ema_fast ~48-50
- Bottom 9 individuals: fitness 0.0, no trades

Check if ema_fast clustering causing issues.
""")

# Agent returns:
{
    "tool": "get_param_distribution",
    "args": {"parameter_name": "ema_fast", "correlate_with": "fitness"}
}

# Phase 3: Action
response_3 = agent.think_and_act("""
Distribution shows high fitness individuals use ema_fast 48-50.
Lower performers use 57-77 (slower).

Actions:
1. Lower min_trades from 20 to 5 (gate too strict)
2. Mutate individual #9 (best) ema_fast to 60 for exploration
3. Inject 10% immigrants for diversity
""")

# Agent returns multiple tool calls:
[
    {"tool": "update_fitness_gates", "args": {"min_trades": 5}},
    {"tool": "mutate_individual", "args": {"individual_id": 9, "parameter_name": "ema_fast", "new_value": 60}},
    {"tool": "update_ga_params", "args": {"immigrant_fraction": 0.1}},
    {"tool": "finish_analysis", "args": {"summary": "...", "overall_assessment": "needs_adjustment"}}
]
```

## Implementation Plan

### Step 1: Add Full Logging (IMMEDIATE)

**File**: `backtest/llm_coach.py`

```python
# In _call_llm() method:
if self.debug_payloads:
    logger.info("=" * 80)
    logger.info("LLM REQUEST (Full Prompt + Payload)")
    logger.info("=" * 80)
    logger.info("SYSTEM PROMPT (%s chars):\n%s", len(self.system_prompt), self.system_prompt)
    logger.info("-" * 80)
    logger.info("USER MESSAGE (%s chars):\n%s", len(user_message), user_message)
    logger.info("=" * 80)

# After receiving response:
if self.debug_payloads:
    logger.info("=" * 80)
    logger.info("LLM RESPONSE (Full Payload)")
    logger.info("=" * 80)
    logger.info("RESPONSE TEXT (%s chars):\n%s", len(response_text or ""), response_text)
    logger.info("=" * 80)
```

### Step 2: Fix Validation Error (IMMEDIATE)

**File**: `backtest/coach_protocol.py`

The prompt mentions MUTATIONS category but it's not in the enum. Two options:

**Option A**: Allow MUTATIONS in prompt, map to INDIVIDUAL_MUTATION
```python
class RecommendationCategory(str, Enum):
    # ... existing ...
    INDIVIDUAL_MUTATION = "individual_mutation"
    MUTATIONS = "individual_mutation"  # Alias for backwards compatibility
```

**Option B**: Update prompt to use correct category names
```python
# In blocking_coach_v1.txt, replace:
"category": "<FITNESS_GATES|GA_HYPERPARAMS|DIVERSITY|MUTATIONS>"
# With:
"category": "<FITNESS_GATES|GA_HYPERPARAMS|DIVERSITY|INDIVIDUAL_MUTATION|INDIVIDUAL_DROP|INDIVIDUAL_INSERT>"
```

### Step 3: Create Tool System (NEW)

**File**: `backtest/coach_tools.py`

```python
"""
Evolution Coach Tools

Provides tool-based interface for coach agent to:
- Query population analytics
- Mutate individuals
- Drop/insert individuals
- Update GA configuration
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from backtest.optimizer import Population
from backtest.coach_session import CoachAnalysisSession
from backtest.coach_mutations import CoachMutationManager
from core.models import FitnessConfig, OptimizationConfig
import numpy as np


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    tool_name: str
    result: Dict[str, Any]
    error: Optional[str] = None


class CoachToolkit:
    """
    Toolkit for Evolution Coach agent.
    
    Provides analytical and mutation tools for coach to use.
    """
    
    def __init__(
        self,
        population: Population,
        session: CoachAnalysisSession,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig,
        mutation_manager: CoachMutationManager
    ):
        self.population = population
        self.session = session
        self.fitness_config = fitness_config
        self.ga_config = ga_config
        self.mutation_manager = mutation_manager
        
        # Track actions taken
        self.actions_log: List[Dict[str, Any]] = []
    
    def analyze_population(
        self,
        group_by: str = "fitness",
        top_n: int = 5,
        bottom_n: int = 3
    ) -> ToolResult:
        """Get population-level statistics."""
        try:
            individuals = self.population.individuals
            
            # Sort by grouping criterion
            if group_by == "fitness":
                sorted_inds = sorted(individuals, key=lambda x: x.fitness, reverse=True)
            elif group_by == "trade_count":
                sorted_inds = sorted(individuals, key=lambda x: x.metrics.get('n', 0), reverse=True)
            elif group_by == "win_rate":
                sorted_inds = sorted(individuals, key=lambda x: x.metrics.get('win_rate', 0), reverse=True)
            else:
                sorted_inds = individuals
            
            # Get top and bottom
            top_individuals = [self._individual_summary(i, idx) for idx, i in enumerate(sorted_inds[:top_n])]
            bottom_individuals = [self._individual_summary(i, idx) for idx, i in enumerate(sorted_inds[-bottom_n:])]
            
            # Calculate clustering
            clustering = self._calculate_parameter_clustering()
            
            # Count below gates
            min_trades = self.fitness_config.get_effective_min_trades(self.population.generation)
            below_gates = sum(1 for ind in individuals if ind.metrics.get('n', 0) < min_trades)
            
            result = {
                "population_size": len(individuals),
                "mean_fitness": float(np.mean([i.fitness for i in individuals])),
                "std_fitness": float(np.std([i.fitness for i in individuals])),
                "diversity": float(self.population.get_diversity_metric()),
                "top_individuals": top_individuals,
                "bottom_individuals": bottom_individuals,
                "below_gates_count": below_gates,
                "below_gates_percent": 100 * below_gates / len(individuals),
                "parameter_clustering": clustering
            }
            
            self.actions_log.append({"action": "analyze_population", "result": "success"})
            return ToolResult(success=True, tool_name="analyze_population", result=result)
        
        except Exception as e:
            return ToolResult(success=False, tool_name="analyze_population", result={}, error=str(e))
    
    def get_param_distribution(
        self,
        parameter_name: str,
        bins: int = 5,
        correlate_with: Optional[str] = None
    ) -> ToolResult:
        """Analyze distribution of specific parameter."""
        # Implementation here
        pass
    
    def mutate_individual(
        self,
        individual_id: int,
        parameter_name: str,
        new_value: Any,
        reason: str = ""
    ) -> ToolResult:
        """Mutate specific individual parameter."""
        # Implementation here
        pass
    
    # ... other tools ...
```

### Step 4: Create Agent Executor (NEW)

**File**: `backtest/coach_agent.py`

```python
"""
Evolution Coach Agent

Agentic interface for evolution coach that can:
- Think through problems
- Call tools to gather information
- Make decisions based on observations
- Apply mutations strategically
"""

from typing import Dict, Any, List, Optional
import json

from backtest.coach_tools import CoachToolkit, ToolResult
from backtest.llm_coach import GemmaCoachClient


class CoachAgent:
    """
    Agent-based Evolution Coach.
    
    Uses tool-calling paradigm:
    1. Observe population state
    2. Think about what to do
    3. Call tools to gather info or take action
    4. Repeat until done
    """
    
    def __init__(
        self,
        llm_client: GemmaCoachClient,
        toolkit: CoachToolkit,
        max_iterations: int = 10
    ):
        self.llm_client = llm_client
        self.toolkit = toolkit
        self.max_iterations = max_iterations
        
        self.conversation_history: List[Dict[str, str]] = []
        self.tool_calls: List[Dict[str, Any]] = []
    
    async def run_analysis(self) -> Dict[str, Any]:
        """Run full agent analysis loop."""
        # Build initial observation
        initial_obs = self._build_initial_observation()
        
        # Start agent loop
        for iteration in range(self.max_iterations):
            # Think and decide next action
            response = await self._agent_step(initial_obs if iteration == 0 else None)
            
            # Parse tool calls from response
            tool_calls = self._parse_tool_calls(response)
            
            # Execute tools
            for tool_call in tool_calls:
                result = self._execute_tool(tool_call)
                self.tool_calls.append({
                    "tool": tool_call["tool"],
                    "args": tool_call["args"],
                    "result": result
                })
                
                # If finish_analysis called, done
                if tool_call["tool"] == "finish_analysis":
                    return self._build_final_summary()
            
            # Check if agent is done
            if not tool_calls or iteration >= self.max_iterations - 1:
                break
        
        # Max iterations reached
        return self._build_final_summary()
    
    async def _agent_step(self, initial_obs: Optional[str] = None) -> str:
        """Single agent reasoning step."""
        # Build prompt with conversation history and observations
        pass
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a tool call."""
        tool_name = tool_call["tool"]
        args = tool_call.get("args", {})
        
        # Map tool names to toolkit methods
        if tool_name == "analyze_population":
            return self.toolkit.analyze_population(**args)
        elif tool_name == "get_param_distribution":
            return self.toolkit.get_param_distribution(**args)
        # ... etc
        else:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                result={},
                error=f"Unknown tool: {tool_name}"
            )
```

### Step 5: Integrate Agent into Manager

**File**: `backtest/coach_manager_blocking.py`

```python
async def analyze_session_with_agent(
    self,
    session: CoachAnalysisSession
) -> Optional[CoachAnalysis]:
    """
    Analyze session using agent-based approach.
    
    Agent can call tools to:
    - Query population
    - Mutate individuals
    - Update GA params
    """
    from backtest.coach_tools import CoachToolkit
    from backtest.coach_agent import CoachAgent
    
    # Create toolkit
    toolkit = CoachToolkit(
        population=self.last_population,  # Need to track this
        session=session,
        fitness_config=self.last_fitness_config,
        ga_config=self.last_ga_config,
        mutation_manager=self.mutation_manager
    )
    
    # Create agent
    agent = CoachAgent(
        llm_client=self.coach_client,
        toolkit=toolkit,
        max_iterations=10
    )
    
    # Run agent analysis
    result = await agent.run_analysis()
    
    # Convert agent result to CoachAnalysis
    analysis = self._agent_result_to_analysis(result)
    
    return analysis
```

## Benefits

1. **More Intelligent**: Agent can explore data before making decisions
2. **More Targeted**: Mutations based on actual parameter analysis, not guesses
3. **More Flexible**: Can drop/insert/mutate in response to observations
4. **Better Debugging**: Full tool call trace shows reasoning
5. **Better Logging**: See exactly what agent is doing at each step

## Migration Path

1. ✅ **Phase 1** (IMMEDIATE): Fix validation + add full logging
2. ✅ **Phase 2** (Day 1): Implement tool system
3. ✅ **Phase 3** (Day 2): Implement agent executor
4. ✅ **Phase 4** (Day 3): Test agent vs. current approach
5. ✅ **Phase 5** (Day 4): Production rollout

## Example Agent Session

```
[AGENT] 🤖 Starting analysis of Gen 10 population...

[AGENT] 🔍 Tool: analyze_population(group_by="fitness", top_n=3, bottom_n=3)
[AGENT] ✅ Result: mean_fitness=0.09, diversity=0.12, 100% below gates

[AGENT] 💭 Observation: All individuals below min_trades=20. Only 3 have >0 fitness.
                         Gate is too strict for current parameter space.

[AGENT] 🔍 Tool: get_param_distribution(parameter_name="ema_fast", correlate_with="fitness")
[AGENT] ✅ Result: Top performers use ema_fast ~48-50, bottom use 57-77

[AGENT] 💭 Decision: Fast EMA in 48-50 range generates more signals.
                      Slower values missing entries.

[AGENT] 🔧 Action: update_fitness_gates(min_trades=5)
[AGENT] ✅ Gate reduced: min_trades 20 → 5

[AGENT] 🔧 Action: mutate_individual(id=9, param=ema_fast, value=60)
[AGENT] ✅ Mutated: Individual #9 ema_fast 48 → 60 (explore slightly slower)

[AGENT] 🔧 Action: update_ga_params(immigrant_fraction=0.1, mutation_rate=0.3)
[AGENT] ✅ GA params updated: immigrant_fraction 0.0 → 0.1

[AGENT] ✅ Analysis complete: 3 actions taken
[AGENT] 📋 Summary: Reduced min_trades gate, mutated best individual for exploration, injected immigrants
```
