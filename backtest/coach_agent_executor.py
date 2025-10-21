"""
Evolution Coach Agent Executor

Manages agent loop for tool-calling coach:
1. Build observation from population state
2. LLM thinks and calls tools
3. Execute tools and return results
4. Repeat until finish_analysis() called or max iterations reached
"""

import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging

from backtest.coach_tools import CoachToolkit
from backtest.llm_coach import GemmaCoachClient

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message in agent conversation."""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class AgentExecutor:
    """
    Agent executor for Evolution Coach.
    
    Manages multi-step reasoning loop:
    - LLM observes population state
    - Thinks about what to do
    - Calls tools to gather info or take action
    - Observes results
    - Repeats until done
    """
    
    def __init__(
        self,
        llm_client: GemmaCoachClient,
        toolkit: CoachToolkit,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Initialize agent executor.
        
        Args:
            llm_client: LLM client (Gemma via LM Studio)
            toolkit: CoachToolkit with all 27 tools
            max_iterations: Max iterations per session (default: 10)
            verbose: Print detailed logs
        """
        self.llm_client = llm_client
        self.toolkit = toolkit
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.conversation_history: List[AgentMessage] = []
        self.tool_call_count = 0
    
    async def run_analysis(self, initial_observation: str) -> Dict[str, Any]:
        """
        Run full agent analysis loop.
        
        Args:
            initial_observation: Initial population state description
        
        Returns:
            Summary of actions taken and results
        """
        if self.verbose:
            print(f"🤖 Evolution Coach Agent starting...")
            print(f"   Max iterations: {self.max_iterations}")
        
        # Add initial user message
        self.conversation_history.append(AgentMessage(
            role="user",
            content=initial_observation
        ))
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            try:
                # Build prompt from conversation history
                prompt = self._build_prompt()
                
                # LLM generates response with potential tool calls
                if self.verbose:
                    print(f"   📤 Sending {len(prompt)} chars to LLM...")
                
                response_text = await self.llm_client._call_llm(prompt)
                
                if not response_text:
                    logger.warning("Empty LLM response")
                    break
                
                if self.verbose:
                    print(f"   📥 Received {len(response_text)} chars from LLM")
                    logger.debug("=" * 70)
                    logger.debug("LLM RESPONSE:")
                    logger.debug("=" * 70)
                    logger.debug(response_text)
                    logger.debug("=" * 70)
                
                # Parse tool calls from response
                tool_calls = self._parse_tool_calls(response_text)
                
                # Add assistant message
                self.conversation_history.append(AgentMessage(
                    role="assistant",
                    content=response_text,
                    tool_calls=tool_calls if tool_calls else None
                ))
                
                # Execute tool calls
                if tool_calls:
                    finish_called = False
                    for tool_call in tool_calls:
                        result = await self._execute_tool(tool_call)
                        
                        # Add tool result to conversation
                        self.conversation_history.append(AgentMessage(
                            role="tool",
                            content=json.dumps(result),
                            tool_call_id=tool_call.get("id", f"call_{self.tool_call_count}")
                        ))
                        
                        self.tool_call_count += 1
                        
                        # Check if finish_analysis called
                        if tool_call["name"] == "finish_analysis":
                            finish_called = True
                            if self.verbose:
                                print("✅ Agent called finish_analysis - session complete")
                            break
                    
                    if finish_called:
                        break
                else:
                    # No tool calls, agent is done
                    if self.verbose:
                        print("⚠️ No tool calls in response - agent may be done")
                    break
            
            except Exception as e:
                logger.exception(f"Error in agent iteration {iteration}")
                if self.verbose:
                    print(f"❌ Error in iteration: {e}")
                break
        
        # Build final summary
        summary = self._build_summary()
        
        if self.verbose:
            print(f"\n✅ Agent analysis complete")
            print(f"   Iterations: {summary['iterations']}")
            print(f"   Tool calls: {summary['tool_calls_count']}")
            print(f"   Actions taken: {len(self.toolkit.actions_log)}")
        
        return summary
    
    def _build_prompt(self) -> str:
        """Build prompt from conversation history."""
        parts = []
        
        for msg in self.conversation_history:
            if msg.role == "user":
                parts.append(f"OBSERVATION:\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"THINKING:\n{msg.content}")
            elif msg.role == "tool":
                parts.append(f"TOOL RESULT:\n{msg.content}")
        
        prompt = "\n\n".join(parts)
        
        # Log full prompt if verbose
        if self.verbose:
            logger.debug("=" * 70)
            logger.debug("AGENT PROMPT (%d chars):", len(prompt))
            logger.debug("=" * 70)
            logger.debug(prompt)
            logger.debug("=" * 70)
        
        return prompt
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response with robust error handling.
        
        Expected format:
        ```json
        {
          "thinking": "...",
          "tool_calls": [
            {
              "name": "tool_name",
              "arguments": {...}
            }
          ]
        }
        ```
        """
        # Clean response text
        cleaned_text = response_text.strip()
        
        # Remove markdown code fences if present
        cleaned_text = re.sub(r'^```json\s*', '', cleaned_text)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
        cleaned_text = re.sub(r'^```\s*', '', cleaned_text)
        
        # Fix common JSON errors
        # Remove trailing commas before closing brackets/braces
        cleaned_text = re.sub(r',(\s*[}\]])', r'\1', cleaned_text)
        
        try:
            # Strategy 1: Try to parse entire response as JSON
            try:
                data = json.loads(cleaned_text)
                if isinstance(data, dict):
                    if "tool_calls" in data and isinstance(data["tool_calls"], list):
                        if self.verbose:
                            print(f"   🔧 Parsed {len(data['tool_calls'])} tool calls (full JSON)")
                        return data["tool_calls"]
                    elif "name" in data and "arguments" in data:
                        if self.verbose:
                            print(f"   🔧 Parsed single tool call: {data.get('name')} (full JSON)")
                        return [data]
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Extract JSON object with tool_calls
            json_match = re.search(r'\{[^{}]*"tool_calls"[^{}]*:\s*\[[^\]]*\][^{}]*\}', cleaned_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if "tool_calls" in data:
                        if self.verbose:
                            print(f"   🔧 Parsed {len(data['tool_calls'])} tool calls (regex match)")
                        return data["tool_calls"]
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Find tool_calls array directly
            tool_calls_match = re.search(r'"tool_calls"\s*:\s*(\[[^\]]+\])', cleaned_text, re.DOTALL)
            if tool_calls_match:
                try:
                    tool_calls = json.loads(tool_calls_match.group(1))
                    if isinstance(tool_calls, list) and tool_calls:
                        if self.verbose:
                            print(f"   🔧 Parsed {len(tool_calls)} tool calls (array extraction)")
                        return tool_calls
                except json.JSONDecodeError:
                    pass
            
            # Strategy 4: Look for individual tool objects
            tool_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})\s*\}'
            tool_matches = re.finditer(tool_pattern, cleaned_text)
            tools_found = []
            for match in tool_matches:
                try:
                    tool_name = match.group(1)
                    args_str = match.group(2)
                    arguments = json.loads(args_str)
                    tools_found.append({
                        "name": tool_name,
                        "arguments": arguments
                    })
                except json.JSONDecodeError:
                    continue
            
            if tools_found:
                if self.verbose:
                    print(f"   🔧 Parsed {len(tools_found)} tool calls (pattern matching)")
                return tools_found
            
            # No tools found - show detailed error
            if self.verbose:
                print("   ⚠️  No valid tool calls found in response")
                logger.warning("Could not extract tool calls from response")
                logger.debug("Response preview (first 500 chars): %s", cleaned_text[:500])
                
                # Show full response for debugging (up to 2000 chars)
                if len(cleaned_text) < 2000:
                    logger.debug("Full response text:\n%s", cleaned_text)
                    print(f"   📄 Full LLM response ({len(cleaned_text)} chars):")
                    print("   " + "-" * 70)
                    for line in cleaned_text.split('\n')[:20]:  # First 20 lines
                        print(f"   {line}")
                    print("   " + "-" * 70)
                else:
                    logger.debug("Response too long (%d chars), showing first 2000", len(cleaned_text))
                    print(f"   📄 LLM response preview ({len(cleaned_text)} chars total):")
                    print("   " + "-" * 70)
                    for line in cleaned_text[:2000].split('\n')[:20]:
                        print(f"   {line}")
                    print("   " + "-" * 70)
            
            return []
        
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Unexpected error parsing tool calls: {e}")
            logger.exception("Unexpected error in _parse_tool_calls")
            logger.debug(f"Response text: {cleaned_text[:500]}")
            return []
    
    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return result."""
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        if self.verbose:
            print(f"🔧 Executing: {tool_name}({', '.join(f'{k}={v}' for k, v in arguments.items())})")
        
        # Map tool names to toolkit methods
        tool_map = {
            "analyze_population": self.toolkit.analyze_population,
            "get_param_distribution": self.toolkit.get_param_distribution,
            "get_param_bounds": self.toolkit.get_param_bounds,
            "mutate_individual": self.toolkit.mutate_individual,
            "update_fitness_gates": self.toolkit.update_fitness_gates,
            "update_ga_params": self.toolkit.update_ga_params,
            "finish_analysis": self.toolkit.finish_analysis
        }
        
        if tool_name not in tool_map:
            result = {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
            if self.verbose:
                print(f"❌ Unknown tool: {tool_name}")
            return result
        
        try:
            # Execute tool
            tool_func = tool_map[tool_name]
            result = await tool_func(**arguments)
            
            if result.get("success"):
                if self.verbose:
                    print(f"✅ {tool_name} succeeded")
            else:
                if self.verbose:
                    print(f"❌ {tool_name} failed: {result.get('error', 'Unknown error')}")
            
            return result
        
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            result = {
                "success": False,
                "error": str(e)
            }
            if self.verbose:
                print(f"❌ {tool_name} error: {e}")
            return result
    
    def _build_summary(self) -> Dict[str, Any]:
        """Build final summary from conversation and actions."""
        # Count iterations
        iterations = sum(1 for msg in self.conversation_history if msg.role == "assistant")
        
        # Extract tool calls
        tool_calls_made = []
        for msg in self.conversation_history:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_made.append({
                        "tool": tc.get("name"),
                        "arguments": tc.get("arguments")
                    })
        
        # Get actions from toolkit
        actions = self.toolkit.actions_log
        
        # Extract final summary from finish_analysis call if present
        final_message = None
        for msg in reversed(self.conversation_history):
            if msg.role == "tool":
                try:
                    tool_result = json.loads(msg.content)
                    if tool_result.get("summary"):
                        final_message = tool_result["summary"]
                        break
                except:
                    pass
        
        return {
            "success": True,
            "iterations": iterations,
            "tool_calls_count": len(tool_calls_made),
            "tool_calls": tool_calls_made,
            "actions_taken": actions,
            "final_summary": final_message or "Agent completed analysis",
            "conversation_length": len(self.conversation_history)
        }
