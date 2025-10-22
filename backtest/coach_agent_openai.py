"""
OpenAI Agents Framework Agent Implementation

Uses proper OpenAI Agents framework with Runner.run() for tool-calling agent.
Integrates with OpenRouter API using the SDK's built-in OpenAI client replacement.
"""

import asyncio
from typing import Dict, Any, Optional
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_default_openai_client, set_default_openai_api, set_tracing_disabled, function_tool
from agents.tool import FunctionTool
from backtest.coach_tools_agents import CoachToolsAgents
from backtest.coach_protocol import load_coach_prompt
from backtest.optimizer import Population
from backtest.coach_session import CoachAnalysisSession
from core.models import FitnessConfig, OptimizationConfig
import logging

logger = logging.getLogger(__name__)


class CoachAgentOpenAI:
    """
    OpenAI Agents framework implementation of Evolution Coach Agent.

    Uses the SDK's built-in OpenAI client replacement for OpenRouter integration.
    """

    def __init__(
        self,
        population: Population,
        session: CoachAnalysisSession,
        fitness_config: FitnessConfig,
        ga_config: OptimizationConfig,
        mutation_manager,
        islands_registry: Optional[Dict[int, Population]] = None,
        island_policy_reference: Optional[Dict[str, Any]] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "anthropic/claude-3.5-sonnet",
        verbose: bool = True
    ):
        self.tool_history = []  # To collect tool calls for debugger
        """
        Initialize OpenAI Agents coach agent.

        Args:
            population: Current population
            session: Analysis session
            fitness_config: Fitness configuration
            ga_config: GA configuration
            mutation_manager: Mutation manager
            islands_registry: Island populations registry
            island_policy_reference: Island policy reference
            openrouter_api_key: OpenRouter API key
            openrouter_model: OpenRouter model name
            verbose: Enable verbose logging
        """
        self.population = population
        self.session = session
        self.fitness_config = fitness_config
        self.ga_config = ga_config
        self.mutation_manager = mutation_manager
        self.islands_registry = islands_registry
        self.island_policy_reference = island_policy_reference
        self.verbose = verbose

        # Load system prompt
        self.system_prompt = load_coach_prompt("agent")

        # Create tools wrapper
        self.tools_wrapper = CoachToolsAgents(
            population=population,
            session=session,
            fitness_config=fitness_config,
            ga_config=ga_config,
            mutation_manager=mutation_manager,
            islands_registry=islands_registry,
            island_policy_reference=island_policy_reference
        )

        # Get API credentials - support both OpenAI and OpenRouter
        from config.settings import settings
        
        # Check for OpenAI API key first (preferred)
        openai_api_key = getattr(settings, 'openai_api_key', '')
        if not openai_api_key:
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            openai_api_key = line.split('=', 1)[1].strip()
                            break
            except Exception:
                pass

        # Fallback to OpenRouter if OpenAI key not available
        if not openai_api_key:
            if openrouter_api_key is None:
                openrouter_api_key = getattr(settings, 'openrouter_api_key', '')
                if not openrouter_api_key:
                    try:
                        with open('.env', 'r') as f:
                            for line in f:
                                if line.startswith('OPENROUTER_API_KEY='):
                                    openrouter_api_key = line.split('=', 1)[1].strip()
                                    break
                    except Exception:
                        pass

        if not openai_api_key and not openrouter_api_key:
            raise ValueError("Either OPENAI_API_KEY or OPENROUTER_API_KEY required")

        # Configure OpenAI Agents SDK
        from openai import AsyncOpenAI

        if openai_api_key:
            # Use OpenAI API directly
            openai_client = AsyncOpenAI(api_key=openai_api_key)
            model_name = "gpt-4o"  # Use GPT-4o for OpenAI API
            if self.verbose:
                print(f"🤖 Using OpenAI API with model: {model_name}")
        else:
            # Use OpenRouter API
            openai_client = AsyncOpenAI(
                base_url="https://api.openai.com/v1/",
                api_key=openrouter_api_key
            )
            model_name = openrouter_model
            if self.verbose:
                print(f"🤖 Using OpenRouter API with model: {model_name}")

        # Configure SDK to use our client
        set_default_openai_client(openai_client)
        set_default_openai_api("chat_completions")  # Use Chat Completions API
        set_tracing_disabled(True)  # Disable OpenAI tracing

        # Create model using OpenAI Chat Completions model
        self.model = OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=openai_client
        )

        # Create tools manually using FunctionTool to avoid schema issues
        tools = []
        
        # List of available tools with their descriptions
        tool_definitions = [
            ('analyze_population', 'Analyze current population state and provide detailed statistics'),
            ('get_correlation_matrix', 'Get correlation matrix between parameters in the population'),
            ('get_param_distribution', 'Get parameter distribution statistics'),
            ('get_param_bounds', 'Get current parameter bounds for optimization'),
            ('get_generation_history', 'Get fitness evolution history across generations'),
            ('mutate_individual', 'Apply mutations to a specific individual'),
            ('insert_llm_individual', 'Insert a new individual created by LLM'),
            ('create_islands', 'Create island populations for parallel evolution'),
            ('migrate_between_islands', 'Migrate individuals between island populations'),
            ('configure_island_scheduler', 'Configure island migration scheduler'),
            ('inject_immigrants', 'Inject new individuals into population'),
            ('export_population', 'Export current population to file'),
            ('import_population', 'Import population from file'),
            ('drop_individual', 'Remove worst individual from population'),
            ('bulk_update_param', 'Update parameter for multiple individuals'),
            ('update_param_bounds', 'Update bounds for a specific parameter'),
            ('update_bounds_multi', 'Update bounds for multiple parameters'),
            ('reseed_population', 'Reseed population with new random individuals'),
            ('insert_individual', 'Insert a new individual into population'),
            ('update_fitness_gates', 'Update fitness gate thresholds'),
            ('update_ga_params', 'Update genetic algorithm parameters'),
            ('update_fitness_weights', 'Update fitness function weights'),
            ('set_fitness_function_type', 'Set fitness function type'),
            ('configure_curriculum', 'Configure curriculum learning parameters'),
            ('set_fitness_preset', 'Set fitness function preset'),
            ('set_exit_policy', 'Set exit policy for optimization'),
            ('set_costs', 'Set transaction costs for backtesting'),
            ('finish_analysis', 'Finish analysis and return final recommendations')
        ]
        
        # Create FunctionTool objects manually
        for tool_name, description in tool_definitions:
            if hasattr(self.tools_wrapper.toolkit, tool_name):
                # Create a simple wrapper function with correct signature
                def create_tool_wrapper(name):
                    async def wrapper(context_wrapper, arguments):
                        if self.verbose:
                            print(f"🔧 Coach calling tool: {name}")
                            if arguments:
                                print(f"   Arguments: {arguments}")
                        
                        toolkit_method = getattr(self.tools_wrapper.toolkit, name)
                        # Parse arguments if they are a string
                        if arguments:
                            if isinstance(arguments, str):
                                try:
                                    import json
                                    parsed_args = json.loads(arguments)
                                    result = await toolkit_method(**parsed_args)
                                except json.JSONDecodeError:
                                    # If not JSON, try calling without arguments
                                    result = await toolkit_method()
                            elif isinstance(arguments, dict):
                                result = await toolkit_method(**arguments)
                            else:
                                result = await toolkit_method()
                        else:
                            result = await toolkit_method()
                        
                        if self.verbose:
                            print(f"✅ Tool {name} completed")
                            if isinstance(result, dict) and 'success' in result:
                                print(f"   Success: {result['success']}")
                        
                        return result
                    return wrapper
                
                wrapper_func = create_tool_wrapper(tool_name)
                wrapper_func.__name__ = tool_name
                wrapper_func.__doc__ = description
                
                # Create FunctionTool manually with simple schema
                tool = FunctionTool(
                    name=tool_name,
                    description=description,
                    params_json_schema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                    on_invoke_tool=wrapper_func
                )
                tools.append(tool)

        if self.verbose:
            print(f"🤖 Collected {len(tools)} tools: {[t.name for t in tools]}")
            print(f"🤖 System prompt loaded: {len(self.system_prompt)} chars from coach_prompts/agent.txt")

            # Debug: Check if tools have proper schemas
            for tool in tools[:3]:  # Check first 3 tools
                print(f"   Tool '{tool.name}': params={bool(hasattr(tool, 'params_json_schema'))}, func={bool(hasattr(tool, 'func'))}")

        # Create agent without hooks to avoid import error
        self.agent = Agent(
            name="Evolution Coach Agent",
            instructions=self.system_prompt,
            model=self.model,
            tools=tools
        )

        if self.verbose:
            print(f"🤖 OpenAI Agents Evolution Coach initialized with {len(tools)} tools")

    async def run_analysis(self, initial_observation: str) -> Dict[str, Any]:
        """
        Run agent analysis using OpenAI Agents Runner.

        Args:
            initial_observation: Initial population state description

        Returns:
            Analysis results with actions taken
        """
        if self.verbose:
            print(f"🤖 Evolution Coach Agent starting analysis...")
            print(f"   Population size: {len(self.population.individuals)}")
            print(f"   Generation: {self.population.generation}")
            print(f"   Initial observation: {initial_observation[:1000]}{'...' if len(initial_observation) > 1000 else ''}")

        try:
            # Count tool calls for logging
            tool_call_count = 0

            if self.verbose:
                print(f"🚀 Starting agent analysis with OpenAI Agents Runner...")

            # Run agent with OpenAI Agents Runner
            result = await Runner.run(
                self.agent,
                input=initial_observation,
                max_turns=50  # Allow up to 50 tool calls
            )

            if self.verbose:
                print(f"🏁 Agent analysis completed")
                print(f"   Final output length: {len(str(result.final_output)) if result.final_output else 0} chars")

            # SDK should handle tool calls automatically, but let's log what happened
            if hasattr(result, 'new_items') and result.new_items:
                for item in result.new_items:
                    # Log tool calls for debugging
                    if hasattr(item, 'tool_calls') and item.tool_calls:
                        for tool_call in item.tool_calls:
                            tool_call_count += 1
                            if self.verbose:
                                print(f"🔧 SDK handled tool call #{tool_call_count}: {tool_call.name}")
                    elif isinstance(item, dict) and item.get('type') == 'tool_call_item':
                        tool_call_count += 1
                        if self.verbose:
                            print(f"🔧 SDK handled tool call #{tool_call_count}: {item.get('name')}")
                    elif getattr(item, 'type', None) == 'tool_call_item':
                        tool_call_count += 1
                        if self.verbose:
                            print(f"🔧 SDK handled tool call #{tool_call_count}: {getattr(item, 'name', 'unknown')}")

            # Additional parsing for JSON in content
            if result.final_output and isinstance(result.final_output, str):
                try:
                    import json
                    parsed = json.loads(result.final_output)
                    if isinstance(parsed, dict) and 'tool_calls' in parsed:
                        for tc in parsed['tool_calls']:
                            tool_call = type('ToolCall', (), {})()  # Mock tool call object
                            tool_call.name = tc['name']
                            tool_call.arguments = tc['arguments']
                            await custom_tool_handler(tool_call)
                except json.JSONDecodeError:
                    pass

            if self.verbose:
                print(f"🤖 Agent completed analysis")
                print(f"   Final output: {result.final_output[:500] if result.final_output else 'None'}...")
                print(f"   New items count: {len(result.new_items) if hasattr(result, 'new_items') else 0}")

                # Log all tool calls and responses with better formatting
                if hasattr(result, 'new_items') and result.new_items:
                    print(f"   Tool call details:")
                    tool_call_count = 0
                    for i, item in enumerate(result.new_items):
                        if hasattr(item, 'role'):
                            print(f"     [{i}] Role: {item.role}")
                            if hasattr(item, 'content') and item.content:
                                print(f"         Content: {item.content[:200]}...")
                            if hasattr(item, 'tool_calls') and item.tool_calls:
                                for tc in item.tool_calls:
                                    tool_call_count += 1
                                    tool_name = tc.get('name', 'unknown')
                                    args = tc.get('arguments', {})
                                    print(f"         🔧 Tool call #{tool_call_count}: {tool_name}")
                                    if args:
                                        # Pretty print arguments
                                        for arg_name, arg_value in args.items():
                                            print(f"            {arg_name}={arg_value}")
                        elif hasattr(item, 'type'):
                            print(f"     [{i}] Type: {item.type}")
                            if hasattr(item, 'content') and item.content:
                                # Try to parse as JSON for tool results
                                try:
                                    import json
                                    parsed = json.loads(item.content)
                                    if isinstance(parsed, dict) and 'success' in parsed:
                                        success = parsed.get('success', False)
                                        status = "✅" if success else "❌"
                                        print(f"         {status} Tool result: {parsed.get('message', 'completed')}")
                                        if not success and 'error' in parsed:
                                            print(f"            Error: {parsed['error']}")
                                    else:
                                        print(f"         Content: {item.content[:200]}...")
                                except json.JSONDecodeError:
                                    print(f"         Content: {item.content[:200]}...")

                # Log actions taken from toolkit
                if hasattr(self.tools_wrapper, 'toolkit') and hasattr(self.tools_wrapper.toolkit, 'actions_log'):
                    actions = self.tools_wrapper.toolkit.actions_log
                    if actions:
                        print(f"   📋 Actions taken by toolkit: {len(actions)}")
                        for j, action in enumerate(actions[-5:], 1):  # Show last 5 actions
                            action_type = action.get('action', 'unknown')
                            print(f"      {j}. {action_type}")
                    else:
                        print(f"   ⚠️  No actions logged by toolkit")

            # Extract actions from tools wrapper
            actions_taken = self.tools_wrapper.toolkit.actions_log

            # Count actual tool calls made by agent
            tool_calls_made = tool_call_count
            if tool_calls_made == 0 and hasattr(result, 'new_items') and result.new_items:
                # Fallback: count declared tool_calls lists even if we didn't execute
                for item in result.new_items:
                    if hasattr(item, 'tool_calls') and item.tool_calls:
                        tool_calls_made += len(item.tool_calls)

            # DEBUG: Check if the agent actually made tool calls in its response
            if self.verbose and tool_calls_made == 0:
                print("   ⚠️  Agent response contains NO tool calls!")
                if hasattr(result, 'final_output') and result.final_output:
                    print(f"   Agent final output: {result.final_output[:300]}...")
                    # Check if it contains tool call JSON
                    if '"tool_calls"' in result.final_output:
                        print("   📝 Agent output contains 'tool_calls' key - parsing issue?")
                    else:
                        print("   📝 Agent output has no 'tool_calls' key - agent chose not to use tools")

            if self.verbose:
                print(f"   Tool calls made: {tool_calls_made}")
                print(f"   Actions taken: {len(actions_taken)}")
                if actions_taken:
                    for i, action in enumerate(actions_taken[-3:], 1):  # Show last 3 actions
                        print(f"     [{len(actions_taken)-3+i}] {action}")

            # Parse final output for summary
            final_summary = self._extract_summary_from_output(result.final_output)

            # Show debugger if history collected (disabled to avoid Qt thread issues)
            # from app.widgets.coach_debugger import CoachDebugger
            # debugger = CoachDebugger(self.tool_history)
            # debugger.show()

            return {
                "success": True,
                "final_output": result.final_output,
                "actions_taken": actions_taken,
                "summary": final_summary,
                "tool_calls_count": tool_calls_made,
                "iterations": len([item for item in (result.new_items or []) if hasattr(item, 'role') and item.role == 'assistant']),
                "tool_history": self.tool_history
            }

        except Exception as e:
            logger.exception(f"Agent analysis failed: {e}")
            if self.verbose:
                print(f"❌ Agent analysis failed: {e}")
                import traceback
                traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "actions_taken": []
            }

    def _extract_summary_from_output(self, final_output: Optional[str]) -> Dict[str, Any]:
        """
        Extract structured summary from agent final output.

        Looks for JSON in the output or parses key information.
        """
        if not final_output:
            return {"summary": "No output generated"}

        import json
        import re

        # Try to find JSON in output
        json_match = re.search(r'\{[^{}]*"summary"[^{}]*\}', final_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: extract basic info
        return {
            "summary": final_output[:500] + ("..." if len(final_output) > 500 else ""),
            "assessment": "completed"
        }