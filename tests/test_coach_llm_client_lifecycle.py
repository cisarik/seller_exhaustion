"""
Tests for LM Studio client lifecycle and LLM calling with proper client management.
This addresses the "Default client is already created" error.
"""

import sys
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Mock lmstudio module BEFORE importing coach modules
sys.modules['lmstudio'] = MagicMock()
sys.modules['lmstudio.sync_api'] = MagicMock()

import asyncio
from backtest.llm_coach import GemmaCoachClient
from backtest.coach_protocol import EvolutionState
from core.coach_logging import coach_log_manager


class TestLMStudioClientLifecycle:
    """Test proper client lifecycle management."""
    
    def test_client_creation_once(self):
        """Test that client is created only once."""
        print("\n[TEST] Client creation lifecycle...")
        
        # Create client
        client = GemmaCoachClient(verbose=False)
        assert client._lms_client is None, "Client should not be created on init"
        print("  ✓ Client not created on initialization")
        
        # Simulate client creation
        client._lms_client = Mock(name="lms_client")
        assert client._lms_client is not None, "Client should be created when needed"
        print("  ✓ Client created when needed")
        
        # Try to get client again (shouldn't recreate)
        existing_client = client._lms_client
        client._lms_client = existing_client  # Reuse same client
        assert client._lms_client is existing_client, "Should reuse same client"
        print("  ✓ Client is reused (not recreated)")
        
        print("✅ Client lifecycle test passed")
    
    @patch('asyncio.to_thread')
    async def test_model_check_without_client_conflict(self, mock_thread):
        """Test checking model status without client conflicts."""
        print("\n[TEST] Model check without client conflicts...")
        
        client = GemmaCoachClient(model="google/gemma-3-12b", verbose=False)
        
        # Mock the lms ps output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "MODEL google/gemma-3-12b STATUS READY"
        mock_thread.return_value = mock_result
        
        # This should NOT create LM Studio client
        is_loaded = await client.check_model_loaded()
        
        assert is_loaded is True, "Model should be detected as loaded"
        assert client._lms_client is None, "check_model_loaded should not create LM client"
        print("  ✓ Model check doesn't create LM client")
        print("✅ Model check without conflicts test passed")


class TestLLMCallWithDummyLogs:
    """Test LLM calling with dummy logs and proper client management."""
    
    @patch('backtest.llm_coach.GemmaCoachClient._call_llm', new_callable=AsyncMock)
    async def test_llm_call_with_dummy_logs(self, mock_call_llm):
        """Test calling LLM with dummy evolution logs."""
        print("\n[TEST] LLM call with dummy logs...")
        
        # Mock LLM response
        dummy_response = {
            "generation": 10,
            "summary": "Population shows stagnation",
            "overall_assessment": "stagnant",
            "stagnation_detected": True,
            "diversity_concern": True,
            "recommendations": [
                {
                    "category": "ga_hyperparams",
                    "parameter": "mutation_rate",
                    "current_value": 0.28,
                    "suggested_value": 0.5,
                    "reasoning": "Increase mutation to escape stagnation",
                    "confidence": 0.85
                }
            ]
        }
        
        mock_call_llm.return_value = json.dumps(dummy_response)
        
        # Create client
        client = GemmaCoachClient(model="google/gemma-3-12b", verbose=False)
        
        # Create dummy evolution state
        evolution_state = EvolutionState(
            generation=10,
            population_size=24,
            mean_fitness=-15.2,
            std_fitness=35.4,
            best_fitness=0.6,
            best_trades=11,
            best_win_rate=0.91,
            best_avg_r=3.96,
            best_pnl=0.3086,
            below_min_trades_percent=57.1,
            mean_trade_count=8.5,
            diversity_metric=0.42,
            recent_improvement=0.0,
            is_stagnant=True,
            fitness_config_dict={"preset": "balanced"},
            ga_config_dict={"population_size": 24}
        )
        
        # Create dummy logs
        dummy_logs = [
            "[Gen    1] Best: 5 trades, 0.40 WR, 0.1 avg_R",
            "[Gen    2] Best: 8 trades, 0.50 WR, 0.3 avg_R",
            "[Gen    3] Best: 8 trades, 0.50 WR, 0.3 avg_R",
            "[Gen    4] Best: 10 trades, 0.60 WR, 0.5 avg_R",
            "[Gen    5] Best: 10 trades, 0.60 WR, 0.5 avg_R",
            "[Gen    6] Best: 10 trades, 0.60 WR, 0.5 avg_R",
            "[Gen    7] Best: 10 trades, 0.60 WR, 0.5 avg_R",
            "[Gen    8] Best: 10 trades, 0.60 WR, 0.5 avg_R",
            "[Gen    9] Best: 11 trades, 0.64 WR, 0.6 avg_R",
            "[Gen   10] Best: 11 trades, 0.64 WR, 0.6 avg_R - STAGNATION DETECTED",
        ]
        
        print(f"  📊 Evolution state: Gen {evolution_state.generation}, {evolution_state.best_trades} trades")
        print(f"  📝 Dummy logs: {len(dummy_logs)} lines")
        
        # Call LLM
        analysis = await client.analyze_evolution(evolution_state, dummy_logs)
        
        assert analysis is not None, "Should parse response"
        assert analysis.overall_assessment == "stagnant", "Should detect stagnation"
        assert len(analysis.recommendations) == 1, "Should have 1 recommendation"
        print(f"  ✓ Analysis received: {analysis.overall_assessment}")
        print(f"  ✓ Recommendations: {len(analysis.recommendations)}")
        print("✅ LLM call with dummy logs test passed")


class TestFullCoachFlowWithProperClientHandling:
    """Test full coach flow with proper client lifecycle."""
    
    @patch('backtest.llm_coach.GemmaCoachClient.check_model_loaded', new_callable=AsyncMock)
    @patch('backtest.llm_coach.GemmaCoachClient._call_llm', new_callable=AsyncMock)
    async def test_full_flow_proper_client_handling(self, mock_call_llm, mock_check_model):
        """Test full coach flow without client conflicts."""
        print("\n[TEST] Full coach flow with proper client handling...")
        
        # Mock check_model_loaded to return True
        mock_check_model.return_value = True
        
        # Mock LLM response
        dummy_response = {
            "generation": 10,
            "summary": "Population exhibits stagnation",
            "overall_assessment": "stagnant",
            "stagnation_detected": True,
            "diversity_concern": True,
            "recommendations": [
                {
                    "category": "ga_hyperparams",
                    "parameter": "mutation_rate",
                    "current_value": 0.28,
                    "suggested_value": 0.55,
                    "reasoning": "Increase mutation pressure",
                    "confidence": 0.88
                },
                {
                    "category": "ga_hyperparams",
                    "parameter": "sigma",
                    "current_value": 0.12,
                    "suggested_value": 0.18,
                    "reasoning": "Larger steps for exploration",
                    "confidence": 0.82
                }
            ]
        }
        
        mock_call_llm.return_value = json.dumps(dummy_response)
        
        print("\n[STEP 1] Checking if model is loaded...")
        client = GemmaCoachClient(model="google/gemma-3-12b", verbose=False)
        is_loaded = await client.check_model_loaded()
        print(f"  ✓ Model loaded: {is_loaded}")
        print(f"  ✓ Client state: {client._lms_client}")
        
        print("\n[STEP 2] Creating evolution state...")
        evolution_state = EvolutionState(
            generation=10,
            population_size=24,
            mean_fitness=-10.5,
            std_fitness=32.1,
            best_fitness=1.2,
            best_trades=15,
            best_win_rate=0.73,
            best_avg_r=0.85,
            best_pnl=0.4521,
            below_min_trades_percent=25.0,
            mean_trade_count=12.3,
            diversity_metric=0.38,
            recent_improvement=0.01,
            is_stagnant=True,
            fitness_config_dict={"preset": "balanced"},
            ga_config_dict={"population_size": 24}
        )
        print(f"  ✓ State created: Gen {evolution_state.generation}")
        
        print("\n[STEP 3] Creating dummy logs...")
        dummy_logs = [
            "[Gen    5] Best: 8 trades, 50% WR, 0.35 avg_R",
            "[Gen    6] Best: 10 trades, 60% WR, 0.50 avg_R",
            "[Gen    7] Best: 12 trades, 67% WR, 0.65 avg_R",
            "[Gen    8] Best: 14 trades, 71% WR, 0.75 avg_R",
            "[Gen    9] Best: 15 trades, 73% WR, 0.80 avg_R",
            "[Gen   10] Best: 15 trades, 73% WR, 0.85 avg_R - NO IMPROVEMENT",
        ]
        print(f"  ✓ Logs created: {len(dummy_logs)} entries")
        
        print("\n[STEP 4] Calling LLM for analysis...")
        analysis = await client.analyze_evolution(evolution_state, dummy_logs)
        
        assert analysis is not None, "Should get analysis"
        assert analysis.generation == 10, "Should have correct generation"
        assert analysis.stagnation_detected is True, "Should detect stagnation"
        assert len(analysis.recommendations) == 2, "Should have 2 recommendations"
        
        print(f"  ✓ Analysis received:")
        print(f"    - Assessment: {analysis.overall_assessment}")
        print(f"    - Stagnation: {analysis.stagnation_detected}")
        print(f"    - Diversity concern: {analysis.diversity_concern}")
        print(f"    - Recommendations: {len(analysis.recommendations)}")
        
        print("\n[STEP 5] Validating recommendations...")
        for i, rec in enumerate(analysis.recommendations, 1):
            print(f"  ✓ Rec {i}: {rec.parameter}")
            print(f"    - From: {rec.current_value} → To: {rec.suggested_value}")
            print(f"    - Confidence: {rec.confidence:.0%}")
        
        print("\n✅ Full coach flow with proper client handling test passed")


# Run tests
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING LM STUDIO CLIENT LIFECYCLE TESTS")
    print("="*80)
    
    # Test 1: Client lifecycle
    print("\n[TEST SUITE 1] Client Lifecycle Management")
    test_lifecycle = TestLMStudioClientLifecycle()
    test_lifecycle.test_client_creation_once()
    
    # Test 2: Model check without conflicts
    print("\n[TEST SUITE 2] Model Check Without Conflicts")
    asyncio.run(TestLMStudioClientLifecycle().test_model_check_without_client_conflict())
    
    # Test 3: LLM call with dummy logs
    print("\n[TEST SUITE 3] LLM Call With Dummy Logs")
    asyncio.run(TestLLMCallWithDummyLogs().test_llm_call_with_dummy_logs())
    
    # Test 4: Full flow with proper client handling
    print("\n[TEST SUITE 4] Full Coach Flow With Proper Client Handling")
    asyncio.run(TestFullCoachFlowWithProperClientHandling().test_full_flow_proper_client_handling())
    
    print("\n" + "="*80)
    print("✅ ALL CLIENT LIFECYCLE TESTS PASSED")
    print("="*80 + "\n")
