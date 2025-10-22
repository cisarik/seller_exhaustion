# Integrating OpenAI Agents Framework into Evolution Coach

## Overview

OpenAI's Agents Python library (https://github.com/openai/openai-agents-python) provides a robust framework for building tool-calling agents. This aligns perfectly with our Evolution Coach Agent design.

---

## Why Use OpenAI Agents Framework?

### ✅ Benefits

1. **Proven Tool-Calling Pattern**
   - Handles tool call parsing automatically
   - Manages conversation history
   - Supports streaming responses
   - Built-in error handling

2. **Function Decorators**
   - Simple `@function_tool` decorator for tools
   - Automatic schema generation from docstrings
   - Type hints → JSON schema conversion

3. **Swarm-Compatible**
   - Can use Swarm patterns for multi-agent coordination
   - Handoffs between specialized agents
   - Shared context

4. **Production-Ready**
   - Used by OpenAI in production
   - Well-tested patterns
   - Active maintenance

### ❌ Considerations

1. **OpenAI API Dependency**
   - Framework designed for OpenAI API
   - We're using local LM Studio with Gemma
   - Need to adapt for local LLM

2. **Async Patterns**
   - Framework uses async/await
   - Good fit for our async architecture

---

## Adaptation Strategy

### Option 1: Use Framework with LM Studio Adapter (RECOMMENDED)

Create an adapter that makes LM Studio look like OpenAI API:

```python
# backtest/lmstudio_adapter.py

from typing import List, Dict, Any
import lmstudio as lms
from openai import OpenAI  # For type compatibility


class LMStudioAdapter:
    """
    Adapter to make LM Studio compatible with OpenAI Agents framework.
    
    Translates OpenAI-style tool calling to LM Studio's format.
    """
    
    def __init__(self, base_url: str = "http://localhost:1234", model: str = "gemma-2-9b-it"):
        self.base_url = base_url
        self.model = model
        self.client = lms.get_default_client()
        self.loaded_model = None
    
    async def load_model(self):
        """Ensure model is loaded"""
        if not self.loaded_model:
            # Use lms CLI to load
            import subprocess
            subprocess.run(["lms", "load", self.model], check=True)
            self.loaded_model = await asyncio.to_thread(self.client.llm.model)
    
    def create_completion(self, messages: List[Dict[str, Any]], tools: List[Dict] = None, **kwargs):
        """
        OpenAI-compatible completion that works with LM Studio.
        
        Translates tool definitions to LM Studio format.
        """
        # Build system prompt with tool descriptions
        system_content = self._build_system_prompt_with_tools(messages[0]["content"], tools)
        
        # Create chat
        chat = lms.Chat(system_content)
        
        # Add user messages
        for msg in messages[1:]:
            if msg["role"] == "user":
                chat.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                chat.add_assistant_message(msg["content"])
        
        # Generate response
        response = self.loaded_model.respond(chat)
        
        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response.content)
        
        # Return OpenAI-compatible format
        return self._format_response(response.content, tool_calls)
    
    def _build_system_prompt_with_tools(self, base_prompt: str, tools: List[Dict]) -> str:
        """Build system prompt including tool descriptions"""
        if not tools:
            return base_prompt
        
        tool_descriptions = "\n\n## Available Tools:\n\n"
        for tool in tools:
            func = tool["function"]
            tool_descriptions += f"### {func['name']}\n"
            tool_descriptions += f"{func.get('description', 'No description')}\n\n"
            tool_descriptions += f"**Parameters**: ```json\n{json.dumps(func['parameters'], indent=2)}\n```\n\n"
        
        tool_descriptions += """
## Tool Call Format

To call a tool, respond with JSON:

```json
{
  "tool_calls": [
    {
      "name": "tool_name",
      "arguments": {
        "param1": "value1",
        "param2": "value2"
      }
    }
  ]
}
```

You can call multiple tools in one response.
"""
        
        return base_prompt + tool_descriptions
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict]:
        """Extract tool calls from LLM response"""
        import json
        import re
        
        # Look for JSON with tool_calls
        json_match = re.search(r'\{.*"tool_calls".*\}', response_text, re.DOTALL)
        if not json_match:
            return []
        
        try:
            data = json.loads(json_match.group(0))
            return data.get("tool_calls", [])
        except json.JSONDecodeError:
            return []
    
    def _format_response(self, content: str, tool_calls: List[Dict]) -> Dict:
        """Format as OpenAI-compatible response"""
        message = {
            "role": "assistant",
            "content": content
        }
        
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"])
                    }
                }
                for i, tc in enumerate(tool_calls)
            ]
        
        return {
            "choices": [{"message": message}],
            "model": self.model
        }
```

### Option 2: Build Custom Agent Executor (FALLBACK)

If adapter is too complex, build our own using OpenAI Agents patterns:

```python
# backtest/coach_agent_executor.py

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import json


@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]


class AgentExecutor:
    """
    Agent executor inspired by OpenAI Agents patterns.
    
    Manages conversation history and tool execution loop.
    """
    
    def __init__(self, llm_client, tools: List[Tool], max_iterations: int = 10):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.conversation_history = []
    
    async def run(self, initial_message: str) -> Dict[str, Any]:
        """
        Run agent loop: think → call tools → observe → think...
        """
        self.conversation_history.append({
            "role": "user",
            "content": initial_message
        })
        
        for iteration in range(self.max_iterations):
            # LLM generates response with potential tool calls
            response = await self.llm_client.generate(
                messages=self.conversation_history,
                tools=self._get_tool_schemas()
            )
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response["content"],
                "tool_calls": response.get("tool_calls", [])
            })
            
            # Execute tool calls
            if response.get("tool_calls"):
                tool_results = await self._execute_tool_calls(response["tool_calls"])
                
                # Add tool results to conversation
                for result in tool_results:
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": result["call_id"],
                        "content": json.dumps(result["result"])
                    })
                
                # Check if finished
                if self._is_finished(response):
                    break
            else:
                # No tool calls, agent is done
                break
        
        return self._build_summary()
    
    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute all tool calls and return results"""
        results = []
        
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            arguments = json.loads(tc["function"]["arguments"])
            
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    result = await tool.function(**arguments)
                    results.append({
                        "call_id": tc["id"],
                        "tool": tool_name,
                        "result": result,
                        "success": True
                    })
                except Exception as e:
                    results.append({
                        "call_id": tc["id"],
                        "tool": tool_name,
                        "result": {"error": str(e)},
                        "success": False
                    })
            else:
                results.append({
                    "call_id": tc["id"],
                    "tool": tool_name,
                    "result": {"error": f"Unknown tool: {tool_name}"},
                    "success": False
                })
        
        return results
    
    def _get_tool_schemas(self) -> List[Dict]:
        """Get OpenAI-compatible tool schemas"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]
    
    def _is_finished(self, response: Dict) -> bool:
        """Check if agent called finish_analysis"""
        if not response.get("tool_calls"):
            return True
        
        for tc in response["tool_calls"]:
            if tc["function"]["name"] == "finish_analysis":
                return True
        
        return False
    
    def _build_summary(self) -> Dict[str, Any]:
        """Build final summary from conversation"""
        tool_calls_made = []
        for msg in self.conversation_history:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_calls_made.append({
                        "tool": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"])
                    })
        
        return {
            "success": True,
            "iterations": len([m for m in self.conversation_history if m["role"] == "assistant"]),
            "tool_calls": tool_calls_made,
            "final_message": self.conversation_history[-1].get("content", "")
        }
```

---

## Implementing Tools with Function Decorators

Using OpenAI Agents pattern, tools are simple decorated functions:

```python
# backtest/coach_tools.py

from typing import Dict, Any, List
from openai import OpenAI  # For type compatibility


class CoachTools:
    """
    Evolution Coach tools using OpenAI Agents patterns.
    
    Each tool is a method decorated with schema information.
    """
    
    def __init__(self, population, session, fitness_config, ga_config, mutation_manager):
        self.population = population
        self.session = session
        self.fitness_config = fitness_config
        self.ga_config = ga_config
        self.mutation_manager = mutation_manager
        self.actions_log = []
    
    # Tool definitions
    
    async def analyze_population(
        self,
        group_by: str = "fitness",
        top_n: int = 5,
        bottom_n: int = 3,
        include_params: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive population statistics and identify patterns.
        
        Use this tool to understand:
        - Current fitness distribution (mean, std, min, max)
        - Diversity level (0.0-1.0, where <0.15 is very low)
        - Top and bottom performers
        - Gate compliance (% below min_trades)
        - Stagnation status
        
        Args:
            group_by: Sort criterion - "fitness", "trade_count", "win_rate", "avg_r"
            top_n: Number of top individuals to show (default: 5)
            bottom_n: Number of bottom individuals to show (default: 3)
            include_params: Include full parameter sets (default: false)
        
        Returns:
            {
                "population_size": 12,
                "fitness": {"mean": 0.089, "std": 0.156, "max": 0.397},
                "diversity": {"metric": 0.12, "interpretation": "very_low"},
                "gates": {"below_min_trades_pct": 100.0, "passing_all_gates": 0},
                "top_individuals": [...],
                "bottom_individuals": [...],
                "stagnation": {"is_stagnant": false, "recent_improvement": 0.058}
            }
        
        When to use:
        - Start of every analysis (understand current state)
        - After making changes (verify impact)
        - Diagnosing performance issues
        """
        # Implementation
        stats = self.population.get_stats()
        diversity = self.population.get_diversity_metric()
        
        # ... full implementation from tool spec
        
        result = {
            "success": True,
            "population_size": len(self.population.individuals),
            "fitness": {
                "mean": float(stats['mean_fitness']),
                "std": float(stats['std_fitness']),
                "min": float(stats['min_fitness']),
                "max": float(stats['max_fitness']),
            },
            "diversity": {
                "metric": float(diversity),
                "interpretation": self._interpret_diversity(diversity)
            },
            # ... rest of result
        }
        
        self.actions_log.append({"action": "analyze_population", "result": "success"})
        return result
    
    async def get_param_distribution(
        self,
        parameter_name: str,
        bins: int = 5,
        correlate_with: str = None,
        show_by_fitness_quartile: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze how a specific parameter is distributed across population.
        
        Use this tool to:
        - See which parameter values lead to high fitness
        - Identify boundary clustering (population hitting bounds)
        - Find correlations with fitness/metrics
        - Compare top vs bottom performers
        
        Args:
            parameter_name: Parameter to analyze (e.g., "ema_fast", "vol_z")
            bins: Number of histogram bins (default: 5)
            correlate_with: Metric to correlate with - "fitness", "trade_count", "win_rate", "avg_r", or null
            show_by_fitness_quartile: Split analysis by fitness quartile (default: true)
        
        Returns:
            {
                "parameter": "ema_fast",
                "stats": {"mean": 53.2, "std": 7.8, "min": 48, "max": 77},
                "histogram": {"bins": [...], "counts": [...]},
                "correlation": {"with": "fitness", "pearson_r": 0.45, "interpretation": "moderate_positive"},
                "by_quartile": {
                    "top_25%": {"mean": 48.7, "interpretation": "Top performers use fast EMA"},
                    "bottom_25%": {"mean": 62.3, "interpretation": "Bottom performers use slow EMA"}
                },
                "boundary_analysis": {
                    "current_bounds": [48, 144],
                    "at_min_bound": 7,
                    "clustering_at_bounds": "min",
                    "interpretation": "Strong clustering at minimum - expand lower bound"
                },
                "insights": ["Top performers use ema_fast 48-50", "58% at minimum bound"]
            }
        
        When to use:
        - Before adjusting parameter bounds
        - Understanding what makes top performers successful
        - Identifying which parameters to focus mutations on
        """
        # Implementation
        # ... full implementation from tool spec
        pass
    
    async def mutate_individual(
        self,
        individual_id: int,
        parameter_name: str,
        new_value: Any,
        reason: str,
        respect_bounds: bool = True
    ) -> Dict[str, Any]:
        """
        Directly modify a specific parameter of a specific individual.
        
        Use this tool to:
        - Explore nearby regions around successful individuals
        - Test hypotheses about parameter effects
        - Repair obviously broken individuals
        - Create directed exploration
        
        Args:
            individual_id: 0-indexed individual ID (0-11 for population of 12)
            parameter_name: Parameter to mutate (e.g., "ema_fast", "vol_z")
            new_value: New value (must match parameter type)
            reason: Explanation for this mutation
            respect_bounds: Enforce parameter bounds (default: true)
        
        Returns:
            {
                "success": true,
                "individual_id": 9,
                "parameter": "ema_fast",
                "old_value": 48,
                "new_value": 60,
                "change": "+12",
                "individual_before": {"fitness": 0.397, "metrics": {...}},
                "impact": {"fitness_reset": true, "will_compete_in_next_gen": true}
            }
        
        Strategic use cases:
        1. Exploit best: Take top performer, mutate one param to explore nearby
        2. Test hypothesis: Verify if specific parameter change helps
        3. Repair bad: Fix obvious mistakes (e.g., too-slow EMA)
        4. Create gradient: Test multiple values along gradient
        
        When to use:
        - After identifying successful individuals (from analyze_population)
        - After seeing correlations (from get_param_distribution)
        - Testing specific parameter hypotheses
        """
        # Implementation
        success = self.mutation_manager.mutate_individual_parameter(
            self.population,
            self.session,
            individual_id,
            parameter_name,
            new_value,
            reason
        )
        
        if success:
            individual = self.population.individuals[individual_id]
            result = {
                "success": True,
                "individual_id": individual_id,
                "parameter": parameter_name,
                "new_value": new_value,
                "message": f"Mutated Individual #{individual_id}: {parameter_name} → {new_value}"
            }
            self.actions_log.append({"action": "mutate_individual", "result": "success"})
            return result
        else:
            return {"success": False, "error": "Mutation failed"}
    
    async def update_fitness_gates(
        self,
        min_trades: int = None,
        min_win_rate: float = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Update fitness gate requirements (hard thresholds).
        
        Gates are HARD thresholds - individuals below get fitness = -100 (in hard_gates mode).
        
        Use this tool to:
        - Lower gates when too many individuals fail (>80% below threshold)
        - Raise gates to increase selectivity (in late-stage refinement)
        - Balance signal frequency vs quality
        
        Args:
            min_trades: Minimum trades required (default: no change)
            min_win_rate: Minimum win rate required (default: no change)
            reason: Explanation for this change
        
        Returns:
            {
                "success": true,
                "changes_applied": {
                    "min_trades": {"old": 20, "new": 5, "interpretation": "Lowered threshold by 75%"}
                },
                "predicted_impact": {
                    "individuals_now_passing": 3,
                    "individuals_still_failing": 9,
                    "recommendation": "Good first step, but signal generation may still be issue"
                }
            }
        
        Strategic patterns:
        - Gate crisis (100% below threshold): Lower min_trades from 20 to 5
        - Many trades but poor quality: Raise min_win_rate from 0.40 to 0.50
        - Late-stage refinement: Raise both gates for selectivity
        
        When to use:
        - When >80% population below min_trades (gate too strict)
        - When fitness mode is "hard_gates" (gates matter)
        - As first intervention for gate crisis
        """
        # Implementation
        old_min_trades = self.fitness_config.min_trades
        old_min_wr = self.fitness_config.min_win_rate
        
        changes = {}
        if min_trades is not None:
            self.fitness_config.min_trades = min_trades
            changes["min_trades"] = {"old": old_min_trades, "new": min_trades}
        
        if min_win_rate is not None:
            self.fitness_config.min_win_rate = min_win_rate
            changes["min_win_rate"] = {"old": old_min_wr, "new": min_win_rate}
        
        result = {
            "success": True,
            "changes_applied": changes,
            "reason": reason
        }
        
        self.actions_log.append({"action": "update_fitness_gates", "changes": changes})
        return result
    
    async def update_param_bounds(
        self,
        parameter: str,
        new_min: Any = None,
        new_max: Any = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Update search space bounds for parameter generation.
        
        Bounds control:
        - Random individual generation
        - Mutation boundaries
        - Exploration space
        
        Use this tool to:
        - Expand bounds when population clusters at edges (>30% at bound)
        - Focus search on successful regions
        - Test hypotheses about parameter ranges
        
        Args:
            parameter: Parameter name (e.g., "ema_fast", "vol_z")
            new_min: New minimum (null = no change)
            new_max: New maximum (null = no change)
            reason: Explanation for this change
        
        Returns:
            {
                "success": true,
                "parameter": "ema_fast",
                "old_bounds": {"min": 48, "max": 144},
                "new_bounds": {"min": 24, "max": 144},
                "change": {"min": "-24 (-50%)", "max": "unchanged"},
                "population_impact": {
                    "individuals_at_old_min": 7,
                    "newly_explorable_region": "[24, 48)",
                    "interpretation": "7 individuals were stuck at old minimum"
                },
                "recommended_followup": [
                    "Insert individual with ema_fast=36 to test new region",
                    "Increase immigrant_fraction to 0.1 to populate new space"
                ]
            }
        
        Strategic patterns:
        - Boundary clustering: Expand that bound
        - Focus on success region: Contract bounds to narrow range
        - Hypothesis testing: Custom bounds for specific test
        
        When to use:
        - After get_param_bounds() shows clustering (>30% at bound)
        - After get_param_distribution() shows correlation outside bounds
        - To focus search on successful parameter ranges
        """
        # Implementation
        # ... implementation from tool spec
        pass
    
    async def update_ga_params(
        self,
        mutation_probability: float = None,
        mutation_rate: float = None,
        sigma: float = None,
        tournament_size: int = None,
        elite_fraction: float = None,
        immigrant_fraction: float = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Adjust genetic algorithm evolution mechanics.
        
        GA parameters control exploration vs exploitation:
        - High mutation_rate + low elite = aggressive exploration
        - Low mutation_rate + high elite = conservative exploitation
        - Immigrants inject diversity
        
        Use this tool to:
        - Increase exploration when stagnant or converged early
        - Decrease exploration when refining good solutions
        - Inject diversity when low (<0.15)
        
        Args:
            mutation_probability: Chance each individual mutates (0.0-1.0)
            mutation_rate: How much parameters change (0.0-1.0)
            sigma: Gaussian mutation std dev
            tournament_size: Selection pressure (2-8, higher = stronger)
            elite_fraction: Top % preserved unchanged (0.0-0.4)
            immigrant_fraction: Random injection rate (0.0-0.3)
            reason: Explanation for changes
        
        Returns:
            {
                "success": true,
                "changes_applied": {
                    "mutation_rate": {
                        "old": 0.55,
                        "new": 0.30,
                        "interpretation": "Reduced mutation by 45% - smaller parameter changes"
                    },
                    "immigrant_fraction": {
                        "old": 0.0,
                        "new": 0.1,
                        "interpretation": "Injecting 10% random individuals for diversity"
                    }
                },
                "predicted_impact": {
                    "exploration_vs_exploitation": "More exploitation (lower mutation)",
                    "diversity_change": "Increased (immigrants)",
                    "convergence_speed": "Slower (less aggressive)",
                    "recommendation": "Good for preserving good solutions while exploring"
                }
            }
        
        Strategic patterns:
        - Stagnation: mutation_rate=0.7, immigrant_fraction=0.25
        - Convergence too early: mutation_rate=0.6, immigrant_fraction=0.2
        - Good progress: mutation_rate=0.3, elite_fraction=0.3
        - Late refinement: mutation_rate=0.2, elite_fraction=0.35
        
        When to use:
        - After analyze_population() shows low diversity (<0.15)
        - When stagnation detected (no improvement for 10 gens)
        - When good solutions found (preserve with high elite)
        """
        # Implementation
        # ... implementation from tool spec
        pass
    
    async def finish_analysis(
        self,
        summary: str,
        overall_assessment: str = "neutral",
        stagnation_detected: bool = False,
        diversity_concern: bool = False
    ) -> Dict[str, Any]:
        """
        Complete analysis session and return control to GA.
        
        Call this tool when:
        - You've made all necessary interventions
        - Ready to let evolution run with your changes
        - Max ~5-7 tool calls made
        
        Args:
            summary: 1-2 sentence summary of actions taken
            overall_assessment: "positive" | "neutral" | "needs_adjustment"
            stagnation_detected: Is evolution stagnant?
            diversity_concern: Is diversity too low?
        
        Returns:
            {
                "success": true,
                "total_actions_taken": 6,
                "breakdown": {
                    "individuals_mutated": 2,
                    "ga_params_changed": 2,
                    "fitness_gates_changed": 1,
                    "bounds_expanded": 1
                }
            }
        
        IMPORTANT: This ends the analysis session. Evolution will resume.
        """
        summary_data = {
            "success": True,
            "summary": summary,
            "overall_assessment": overall_assessment,
            "stagnation_detected": stagnation_detected,
            "diversity_concern": diversity_concern,
            "total_actions": len(self.actions_log),
            "actions_log": self.actions_log
        }
        
        return summary_data


# Tool registration for OpenAI Agents framework

def get_tool_schemas(tools_instance: CoachTools) -> List[Dict]:
    """
    Extract tool schemas from CoachTools methods.
    
    Uses docstrings and type hints to generate OpenAI function schemas.
    """
    import inspect
    
    tools = []
    
    for name, method in inspect.getmembers(tools_instance, predicate=inspect.ismethod):
        if name.startswith("_") or name == "finish_analysis":
            continue  # Skip private methods, handle finish_analysis separately
        
        # Get signature
        sig = inspect.signature(method)
        
        # Build parameters schema from type hints
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Get type and default
            param_type = param.annotation
            has_default = param.default != inspect.Parameter.empty
            
            # Map Python types to JSON schema types
            json_type = "string"
            if param_type == int:
                json_type = "integer"
            elif param_type == float:
                json_type = "number"
            elif param_type == bool:
                json_type = "boolean"
            
            parameters["properties"][param_name] = {"type": json_type}
            
            if not has_default:
                parameters["required"].append(param_name)
        
        # Extract description from docstring
        docstring = inspect.getdoc(method) or "No description"
        description = docstring.split("\n\n")[0]  # First paragraph
        
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        })
    
    # Add finish_analysis separately (always last)
    tools.append({
        "type": "function",
        "function": {
            "name": "finish_analysis",
            "description": "Complete analysis session and return control to GA",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "overall_assessment": {"type": "string", "enum": ["positive", "neutral", "needs_adjustment"]},
                    "stagnation_detected": {"type": "boolean"},
                    "diversity_concern": {"type": "boolean"}
                },
                "required": ["summary"]
            }
        }
    })
    
    return tools
```

---

## Integration with Coach Manager

```python
# backtest/coach_manager_blocking.py

from backtest.coach_tools import CoachTools, get_tool_schemas
from backtest.coach_agent_executor import AgentExecutor
from backtest.lmstudio_adapter import LMStudioAdapter


class BlockingCoachManager:
    # ... existing code ...
    
    async def analyze_session_with_agent(
        self,
        session: CoachAnalysisSession
    ) -> Optional[CoachAnalysis]:
        """
        Analyze session using OpenAI Agents-style tool-calling agent.
        """
        coach_log_manager.append(
            f"[COACH  ] 🤖 Starting agent analysis for Gen {session.generation}"
        )
        
        # Create tools instance
        tools_instance = CoachTools(
            population=self.current_population,
            session=session,
            fitness_config=self.current_fitness_config,
            ga_config=self.current_ga_config,
            mutation_manager=self.mutation_manager
        )
        
        # Get tool schemas
        tool_schemas = get_tool_schemas(tools_instance)
        
        # Create LM Studio adapter
        llm_adapter = LMStudioAdapter(
            base_url=self.base_url,
            model=self.model
        )
        await llm_adapter.load_model()
        
        # Create agent executor
        agent = AgentExecutor(
            llm_client=llm_adapter,
            tools=self._tools_instance_to_list(tools_instance),
            max_iterations=10
        )
        
        # Build initial message
        initial_message = self._build_analysis_prompt(session)
        
        # Run agent
        result = await agent.run(initial_message)
        
        # Convert result to CoachAnalysis
        analysis = self._agent_result_to_analysis(result, session.generation)
        
        return analysis
```

---

## System Prompt for Agent

The system prompt is loaded from `coach_prompts/agent.txt` (see [coach_protocol.py](coach_protocol.py) `load_coach_prompt()` function).
```

---

## Conclusion

Using OpenAI Agents patterns provides:

✅ **Proven architecture** for tool-calling agents
✅ **Clean abstractions** (function decorators, tool schemas)
✅ **Conversation management** built-in
✅ **Error handling** patterns
✅ **Extensibility** (easy to add new tools)

Next steps:
1. Implement LMStudioAdapter for local LLM compatibility
2. Implement CoachTools with all 27 tools
3. Create AgentExecutor with OpenAI Agents patterns
4. Test on real evolution runs

The framework significantly simplifies our implementation while maintaining all the power and flexibility of our original design.
