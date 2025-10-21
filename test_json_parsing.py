"""
Test JSON parsing from various LLM response formats

Tests the robust JSON parser against common LLM output patterns.
"""

from backtest.coach_agent_executor import AgentExecutor
from backtest.coach_tools import CoachToolkit
from backtest.llm_coach import GemmaCoachClient


def test_json_parsing():
    """Test various JSON formats that LLMs might return."""
    
    test_cases = [
        # Case 1: Perfect JSON
        (
            "Perfect JSON",
            '''{"thinking": "I need to analyze", "tool_calls": [{"name": "analyze_population", "arguments": {}}]}'''
        ),
        
        # Case 2: With markdown code fences
        (
            "With code fences",
            '''```json
{"thinking": "I need to analyze", "tool_calls": [{"name": "analyze_population", "arguments": {}}]}
```'''
        ),
        
        # Case 3: With explanation before
        (
            "With explanation",
            '''Let me analyze the population.

{"thinking": "I need to analyze", "tool_calls": [{"name": "analyze_population", "arguments": {}}]}'''
        ),
        
        # Case 4: With explanation after
        (
            "With trailing text",
            '''{"thinking": "I need to analyze", "tool_calls": [{"name": "analyze_population", "arguments": {}}]}

This should help us understand the population.'''
        ),
        
        # Case 5: Malformed but recoverable
        (
            "Malformed but recoverable",
            '''{"thinking": "I need to analyze",
  "tool_calls": [
    {
      "name": "analyze_population",
      "arguments": {}
    }
  ]
}'''
        ),
        
        # Case 6: Single tool at top level
        (
            "Single tool",
            '''{"name": "analyze_population", "arguments": {"group_by": "fitness"}}'''
        ),
        
        # Case 7: Multiple tools
        (
            "Multiple tools",
            '''{
  "thinking": "First analyze, then act",
  "tool_calls": [
    {"name": "analyze_population", "arguments": {"group_by": "fitness"}},
    {"name": "update_fitness_gates", "arguments": {"min_trades": 5}}
  ]
}'''
        ),
        
        # Case 8: With trailing comma (common error)
        (
            "Trailing comma",
            '''{"thinking": "test", "tool_calls": [{"name": "analyze_population", "arguments": {},}]}'''
        ),
    ]
    
    # Create mock agent executor just for parsing
    class MockLLM:
        async def _call_llm(self, prompt):
            return ""
    
    class MockToolkit:
        actions_log = []
    
    executor = AgentExecutor(
        llm_client=MockLLM(),
        toolkit=MockToolkit(),
        verbose=False
    )
    
    print("🧪 Testing JSON Parsing Robustness")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    for name, response_text in test_cases:
        print(f"Test: {name}")
        print(f"Input: {response_text[:60]}...")
        
        try:
            tool_calls = executor._parse_tool_calls(response_text)
            if tool_calls:
                print(f"✅ PASS - Parsed {len(tool_calls)} tool(s)")
                for tc in tool_calls:
                    print(f"   → {tc['name']}")
                passed += 1
            else:
                print(f"⚠️  WARN - No tools parsed")
                failed += 1
        except Exception as e:
            print(f"❌ FAIL - {e}")
            failed += 1
        
        print()
    
    print("=" * 70)
    print(f"Results: {passed}/{len(test_cases)} passed, {failed} failed")
    print()
    
    if failed == 0:
        print("✅ All tests passed! Parser is robust.")
    else:
        print(f"⚠️  {failed} tests failed. Parser needs improvement.")
    
    return failed == 0


if __name__ == "__main__":
    success = test_json_parsing()
    exit(0 if success else 1)
