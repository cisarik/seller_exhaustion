"""
OpenAI Agents Framework Agent Implementation

Uses proper OpenAI Agents framework with Runner.run() for tool-calling agent.
Integrates with OpenRouter API using the SDK's built-in OpenAI client replacement.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_default_openai_client, set_default_openai_api, set_tracing_disabled, function_tool, AgentHooks
from agents.tool import FunctionTool
from backtest.coach_tools_agents import CoachToolsAgents
from backtest.coach_protocol import load_coach_prompt
from backtest.optimizer import Population
from backtest.coach_session import CoachAnalysisSession
from core.models import FitnessConfig, OptimizationConfig, AdamConfig
import logging
import re

try:
    from app.signals import get_coach_signals
    SIGNALS_AVAILABLE = True
except ImportError:
    SIGNALS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CoachAgentHooks(AgentHooks):
    """Custom hooks for Evolution Coach Agent to handle status updates."""
    
    def __init__(self, status_callback=None, coach_window=None):
        super().__init__()
        self.status_callback = status_callback
        self.coach_window = coach_window
    
    async def on_tool_start(self, context, agent, tool):
        """Called when a tool starts executing."""
        if self.status_callback:
            try:
                self.status_callback(tool.name, "Starting tool execution")
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
        
        if self.coach_window:
            try:
                # Schedule UI update on main thread using QTimer
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.coach_window.add_tool_call(
                    tool.name, {}, {}, "Starting execution"
                ))
            except Exception as e:
                logger.error(f"Error updating coach window: {e}")
    
    async def on_tool_end(self, context, agent, tool, result):
        """Called when a tool completes execution."""
        if self.status_callback:
            try:
                self.status_callback(tool.name, "Tool completed")
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
        
        if self.coach_window:
            try:
                # Schedule UI update on main thread using QTimer
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.coach_window.update_last_tool_call(
                    {}, result, "Completed"
                ))
            except Exception as e:
                logger.error(f"Error updating coach window: {e}")
    
    async def on_start(self, context, agent):
        """Called when agent starts."""
        if self.status_callback:
            try:
                self.status_callback("Agent", "Starting analysis")
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    async def on_end(self, context, agent, output):
        """Called when agent completes."""
        if self.status_callback:
            try:
                self.status_callback("Agent", "Analysis completed")
            except Exception as e:
                logger.error(f"Error in status callback: {e}")


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
        adam_config: Optional[AdamConfig] = None,
        islands_registry: Optional[Dict[int, Population]] = None,
        island_policy_reference: Optional[Dict[str, Any]] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "anthropic/claude-3.5-sonnet",
        verbose: bool = True,
        status_callback: Optional[callable] = None,
        coach_window=None
    ):
        self.tool_history = []  # To collect tool calls for debugger
        self.status_callback = status_callback  # Callback for status updates
        self.coach_window = coach_window  # Coach window for UI updates
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
            adam_config=adam_config,
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

        # Get provider from settings to check which API key is needed
        from config.settings import settings
        provider = getattr(settings, 'agent_provider', 'novita')
        
        if provider == "openai" and not openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI provider")
        elif provider == "openrouter" and not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY required for OpenRouter provider")
        elif provider == "novita":
            novita_api_key = getattr(settings, 'novita_api_key', '')
            if not novita_api_key:
                raise ValueError("NOVITA_API_KEY required for Novita provider")

        # Configure OpenAI Agents SDK
        from openai import AsyncOpenAI
        
        if provider == "openai" and openai_api_key:
            # Use OpenAI API directly
            openai_base_url = getattr(settings, 'openai_base_url', 'https://api.openai.com/v1')
            openai_model = getattr(settings, 'openai_model', 'gpt-4o')
            
            self.openai_client = AsyncOpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url
            )
            model_name = openai_model
            if self.verbose:
                print(f"🤖 Using OpenAI API with model: {model_name} (base_url: {openai_base_url})")
                
        elif provider == "openrouter" and openrouter_api_key:
            # Use OpenRouter API
            openrouter_base_url = getattr(settings, 'openrouter_base_url', 'https://openrouter.ai/api/v1')
            openrouter_model = getattr(settings, 'openrouter_model', 'anthropic/claude-3.5-sonnet')
            
            self.openai_client = AsyncOpenAI(
                base_url=openrouter_base_url,
                api_key=openrouter_api_key
            )
            model_name = openrouter_model
            if self.verbose:
                print(f"🤖 Using OpenRouter API with model: {model_name} (base_url: {openrouter_base_url})")
                
        else:
            # Use Novita API (default)
            novita_base_url = getattr(settings, 'novita_base_url', 'https://api.novita.ai/openai')
            novita_model = getattr(settings, 'novita_model', 'deepseek/deepseek-r1')
            
            self.openai_client = AsyncOpenAI(
                base_url=novita_base_url,
                api_key=novita_api_key
            )
            model_name = novita_model
            if self.verbose:
                print(f"🤖 Using Novita API with model: {model_name} (base_url: {novita_base_url})")

        # Configure SDK to use our client
        set_default_openai_client(self.openai_client)
        set_default_openai_api("chat_completions")  # Use Chat Completions API
        set_tracing_disabled(True)  # Disable OpenAI tracing

        # Create model using OpenAI Chat Completions model
        self.model = OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=self.openai_client
        )

        # Use tools directly from CoachToolsAgents
        tools = []
        
        # Get all tools from CoachToolsAgents that are enabled
        for attr_name in dir(self.tools_wrapper):
            if not attr_name.startswith('_') and attr_name != 'toolkit':
                tool = getattr(self.tools_wrapper, attr_name)
                # Check if it's a FunctionTool
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    # Check if this tool is enabled in settings
                    if getattr(settings, f'coach_tool_{attr_name}', True):
                        # Check if islands management is disabled and this is an islands tool
                        if not settings.coach_islands_enabled and self._is_islands_tool(attr_name):
                            if self.verbose:
                                print(f"🚫 Skipped islands tool: {tool.name} (islands disabled)")
                            continue
                        
                        tools.append(tool)
                        if self.verbose:
                            print(f"✅ Added tool: {tool.name}")
        
        if self.verbose:
            print(f"🤖 System prompt loaded: {len(self.system_prompt)} chars")
            print(f"🤖 Tools available: {len(tools)} ({[t.name for t in tools[:5]]}{'...' if len(tools) > 5 else ''})")
            print(f"🤖 configure_island_scheduler in tools: {'configure_island_scheduler' in [t.name for t in tools]}")
            print(f"🤖 set_active_optimizer in tools: {'set_active_optimizer' in [t.name for t in tools]}")
            print(f"🤖 configure_ga_parameters in tools: {'configure_ga_parameters' in [t.name for t in tools]}")
            print(f"🤖 configure_adam_parameters in tools: {'configure_adam_parameters' in [t.name for t in tools]}")

        # Debug: Check if tools have proper schemas
        for tool in tools[:3]:  # Check first 3 tools
            print(f"   Tool '{tool.name}': params={bool(hasattr(tool, 'params_json_schema'))}, func={bool(hasattr(tool, 'func'))}")
        
        # Debug: List all tool names
        if self.verbose:
            print(f"🤖 All tool names: {[t.name for t in tools]}")

        # Create custom hooks for status updates
        self.hooks = CoachAgentHooks(
            status_callback=self.status_callback,
            coach_window=self.coach_window
        )

        # Create agent with hooks
        self.agent = Agent(
            name="Evolution Coach Agent",
            instructions=self.system_prompt,
            model=self.model,
            tools=tools,
            hooks=self.hooks
        )

        if self.verbose:
            print(f"✓ 🤖 Coach Agent initialized with {len(tools)} tools")
    
    def _is_islands_tool(self, attr_name: str) -> bool:
        """Check if a tool is related to islands management."""
        islands_tools = {
            'create_islands',
            'migrate_between_islands', 
            'configure_island_scheduler'
        }
        return attr_name in islands_tools

    async def run_analysis(self, initial_observation: str) -> Dict[str, Any]:
        """
        Run agent analysis using OpenAI Agents Runner.

        Args:
            initial_observation: Initial population state description

        Returns:
            Analysis results with actions taken
        """
        if self.verbose:
            logger.info(f"🤖 Starting analysis: Gen {self.population.generation}, Pop {len(self.population.individuals)}")
            logger.debug(f"   Initial observation: {initial_observation[:1000]}{'...' if len(initial_observation) > 1000 else ''}")

        # Report analysis start to status callback
        if self.status_callback:
            try:
                self.status_callback("Starting analysis", "Initializing agent")
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

        # Update coach window with request (thread-safe)
        if hasattr(self, 'coach_window') and self.coach_window:
            try:
                # Schedule UI update on main thread using QTimer
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.coach_window.set_agent_request(initial_observation))
            except Exception as e:
                logger.error(f"Error updating coach window with request: {e}")

        try:
            # Count tool calls for logging
            tool_call_count = 0

            if self.verbose:
                logger.info(f"🚀 Starting agent analysis with OpenAI Agents Runner...")
                logger.debug(f"   Agent model: {self.model}")
                logger.debug(f"   Tools count: {len(self.agent.tools) if hasattr(self.agent, 'tools') else 'unknown'}")
                logger.debug(f"   Max turns: 50")

            # Run agent with OpenAI Agents Runner
            # Each analysis starts with a fresh conversation history to prevent context window overflow
            # when running many analyses (e.g., 500+ analyses)
            result = await Runner.run(
                self.agent,
                input=initial_observation,
                max_turns=50  # Allow up to 50 tool calls
            )

            if self.verbose:
                logger.info(f"🏁 Agent analysis completed")
                logger.info(f"   Final output length: {len(str(result.final_output)) if result.final_output else 0} chars")
                logger.debug(f"   Final output: {result.final_output}")

            # Report analysis completion to status callback
            if self.status_callback:
                try:
                    self.status_callback("Analysis completed", f"Tool calls: {tool_call_count}")
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")

            # Update coach window with response (thread-safe)
            if hasattr(self, 'coach_window') and self.coach_window:
                try:
                    response_text = str(result.final_output) if result.final_output else "No response"
                    # Schedule UI update on main thread using QTimer
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self.coach_window.set_agent_response(response_text))
                except Exception as e:
                    logger.error(f"Error updating coach window with response: {e}")

            # SDK should handle tool calls automatically, but let's log what happened
            if hasattr(result, 'new_items') and result.new_items:
                for item in result.new_items:
                    # Log tool calls for debugging
                    if hasattr(item, 'tool_calls') and item.tool_calls:
                        for tool_call in item.tool_calls:
                            tool_call_count += 1
                            if self.verbose:
                                logger.info(f"🔧 SDK handled tool call #{tool_call_count}: {tool_call.name}")
                                logger.debug(f"   Tool call arguments: {tool_call.arguments}")

                            # Report tool call to status callback
                            if self.status_callback:
                                try:
                                    # Extract reason from tool call arguments if available
                                    reason = ""
                                    parameters = {}
                                    if hasattr(tool_call, 'arguments') and tool_call.arguments:
                                        args = tool_call.arguments
                                        if isinstance(args, dict):
                                            parameters = args
                                            if 'reason' in args:
                                                reason = args['reason']

                                    # Track tool call in history
                                    self.tool_history.append({
                                        'name': tool_call.name,
                                        'parameters': parameters,
                                        'response': {},  # Will be filled when response comes
                                        'reason': reason,
                                        'timestamp': datetime.now().isoformat()
                                    })

                                    logger.debug(f"   Tool call tracked: {tool_call.name} with {len(parameters)} parameters")
                                    # Status callback now handled by hooks
                                except Exception as e:
                                    logger.error(f"Error in status callback: {e}")

                    elif isinstance(item, dict) and item.get('type') == 'tool_call_item':
                        tool_call_count += 1
                        tool_name = item.get('name', 'unknown')
                        if self.verbose:
                            print(f"🔧 SDK handled tool call #{tool_call_count}: {tool_name}")

                        # Report tool call to status callback
                        if self.status_callback:
                            try:
                                reason = item.get('arguments', {}).get('reason', '')
                                self.status_callback(tool_name, reason)
                            except Exception as e:
                                print(f"Error in status callback: {e}")

                    elif getattr(item, 'type', None) == 'tool_call_item':
                        tool_call_count += 1
                        tool_name = getattr(item, 'name', 'unknown')
                        if self.verbose:
                            print(f"🔧 SDK handled tool call #{tool_call_count}: {tool_name}")

                        # Report tool call to status callback
                        if self.status_callback:
                            try:
                                reason = getattr(item, 'arguments', {}).get('reason', '')
                                self.status_callback(tool_name, reason)
                            except Exception as e:
                                print(f"Error in status callback: {e}")

            # Check if agent actually made tool calls
            if tool_call_count == 0:
                logger.warning("⚠️ Agent did not make any tool calls - this indicates the agent is not taking actions")
                if hasattr(result, 'final_output') and result.final_output:
                    logger.info(f"Agent output: {result.final_output[:500]}...")
                    if '"tool_calls"' in result.final_output:
                        logger.warning("Agent output contains 'tool_calls' key but no actual tool calls were executed")
                    else:
                        logger.warning("Agent output has no 'tool_calls' key - agent chose not to use tools")
                # CRITICAL: Force the agent to make at least one tool call by calling analyze_population
                logger.warning("🔄 Agent failed to make any tool calls - forcing analyze_population() call")
                try:
                    # Force a tool call to analyze_population to ensure the agent takes at least one action
                    await self.tools_wrapper.analyze_population(
                        group_by="fitness",
                        top_n=5,
                        bottom_n=3,
                        include_params=False,
                        reason="Forced analysis due to agent not making any tool calls"
                    )
                    tool_call_count += 1
                    logger.info("✅ Forced analyze_population() tool call executed")
                except Exception as force_error:
                    logger.error(f"❌ Failed to force analyze_population() call: {force_error}")

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
                            # await custom_tool_handler(tool_call)  # TODO: Implement custom tool handler
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
                            logger.debug(f"     [{i}] Type: {item.type}")
                            if hasattr(item, 'content') and item.content:
                                # Try to parse as JSON for tool results
                                try:
                                    import json
                                    parsed = json.loads(item.content)
                                    if isinstance(parsed, dict) and 'success' in parsed:
                                        success = parsed.get('success', False)
                                        status = "✅" if success else "❌"
                                        logger.info(f"         {status} Tool result: {parsed.get('message', 'completed')}")
                                        if not success and 'error' in parsed:
                                            logger.error(f"            Error: {parsed['error']}")
                                        logger.debug(f"         Full response: {json.dumps(parsed, indent=2)}")
                                    else:
                                        logger.debug(f"         Content: {item.content[:200]}...")
                                except json.JSONDecodeError:
                                    logger.debug(f"         Content: {item.content[:200]}...")

                # Log actions taken from toolkit
                if hasattr(self.tools_wrapper, 'toolkit') and hasattr(self.tools_wrapper.toolkit, 'actions_log'):
                    actions = self.tools_wrapper.toolkit.actions_log
                    if actions:
                        logger.info(f"   📋 Actions taken by toolkit: {len(actions)}")
                        for j, action in enumerate(actions[-5:], 1):  # Show last 5 actions
                            action_type = action.get('action', 'unknown')
                            logger.info(f"      {j}. {action_type}")
                            logger.debug(f"         Action details: {action}")
                    else:
                        logger.warning(f"   ⚠️  No actions logged by toolkit")

            # Extract actions from tools wrapper
            actions_taken = self.tools_wrapper.toolkit.actions_log

            # Count actual tool calls made by agent
            tool_calls_made = tool_call_count
            if tool_calls_made == 0 and hasattr(result, 'new_items') and result.new_items:
                # Fallback: count declared tool_calls lists even if we didn't execute
                for item in result.new_items:
                    if hasattr(item, 'tool_calls') and item.tool_calls:
                        tool_calls_made += len(item.tool_calls)

            # CRITICAL: If no tool calls were made, this is a failure
            if tool_calls_made == 0:
                logger.error("🚨 CRITICAL FAILURE: Agent made 0 tool calls - this violates the core requirement")
                logger.error("The Evolution Coach MUST use tools to take actions, not just describe them")
                logger.error("This indicates the agent is not properly integrated with the tool system")
                # We cannot force tool calls here, but we must report this as a failure
                # The system should be designed so this never happens

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
                # Force the agent to make at least one tool call by calling analyze_population if none were made
                if tool_calls_made == 0:
                    logger.warning("Agent failed to make any tool calls - this is a critical failure")
                    logger.warning("The agent should always start with analyze_population()")
                    # We can't force tool calls here, but we can log the issue

            if self.verbose:
                logger.info(f"   Tool calls made: {tool_calls_made}")
                logger.info(f"   Actions taken: {len(actions_taken)}")
                if actions_taken:
                    for i, action in enumerate(actions_taken[-3:], 1):  # Show last 3 actions
                        logger.info(f"     [{len(actions_taken)-3+i}] {action}")
                        logger.debug(f"         Action details: {action}")

            # Parse final output for summary
            final_summary = self._extract_summary_from_output(result.final_output)

            # Emit signals for real-time window updates (OpenAI Coach)
            if SIGNALS_AVAILABLE:
                try:
                    signals = get_coach_signals()
                    
                    # Emit coach message (shows in window)
                    signals.coach_message.emit(
                        f"✓ 🤖 OpenAI Agent Analysis: {tool_calls_made} tool calls, {len(actions_taken)} actions",
                        "blue"
                    )
                    
                    # Emit all tool calls to history table
                    for tool_call in self.tool_history:
                        signals.tool_call_complete.emit(
                            tool_call.get('name', 'unknown'),
                            tool_call.get('parameters', {}),
                            tool_call.get('response', {}),
                            tool_call.get('reason', '')
                        )
                except Exception as e:
                    logger.debug(f"Signals emission skipped: {e}")

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
                logger.error(f"❌ Agent analysis failed: {e}")
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "actions_taken": [],
                "tool_calls_count": 0,
                "iterations": 0,
                "tool_history": []
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