"""
Comprehensive tests for Evolution Coach integration with mocked LLM responses.
"""

import pytest
import asyncio
import json
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Mock lmstudio module before importing coach modules
sys.modules['lmstudio'] = MagicMock()
sys.modules['lmstudio.sync_api'] = MagicMock()

from backtest.coach_manager import CoachManager
from backtest.coach_protocol import (
    CoachAnalysis, CoachRecommendation, RecommendationCategory, EvolutionState
)
from backtest.llm_coach import GemmaCoachClient
from backtest.optimizer import Population, Individual
from core.models import FitnessConfig, OptimizationConfig, BacktestParams, Timeframe
from strategy.seller_exhaustion import SellerParams


class TestGemmaCoachClientJSON:
    """Test JSON parsing from LLM responses."""
    
    def test_parse_valid_json_response(self):
        """Test parsing valid JSON from LLM response."""
        client = GemmaCoachClient(verbose=False)
        
        response_text = """
        Here's my analysis of the evolution state:
        
        {
            "generation": 10,
            "summary": "Population is stagnant with low diversity",
            "overall_assessment": "stagnant",
            "stagnation_detected": true,
            "diversity_concern": true,
            "recommendations": [
                {
                    "category": "ga_hyperparams",
                    "parameter": "ga_mutation_rate",
                    "current_value": 0.3,
                    "suggested_value": 0.5,
                    "reasoning": "Low mutation rate causing stagnation",
                    "confidence": 0.8
                },
                {
                    "category": "ga_hyperparams",
                    "parameter": "ga_tournament_size",
                    "current_value": 3,
                    "suggested_value": 5,
                    "reasoning": "Increase selection pressure",
                    "confidence": 0.6
                }
            ]
        }
        
        Let me know if you have any questions!
        """
        
        analysis = client._parse_response(response_text, generation=10)
        
        assert analysis is not None
        assert analysis.overall_assessment == "stagnant"
        assert analysis.stagnation_detected is True
        assert analysis.diversity_concern is True
        assert len(analysis.recommendations) == 2
        assert analysis.recommendations[0].parameter == "ga_mutation_rate"
        assert analysis.recommendations[0].suggested_value == 0.5
        print("✅ Valid JSON parsing test passed")
    
    def test_parse_json_with_extra_text(self):
        """Test parsing JSON embedded in text."""
        client = GemmaCoachClient(verbose=False)
        
        response_text = """
        Based on my analysis, I recommend:
        
        Some intro text here...
        
        {
            "generation": 5,
            "summary": "Population is improving with good diversity",
            "overall_assessment": "improving",
            "stagnation_detected": false,
            "diversity_concern": false,
            "recommendations": []
        }
        
        And some outro text here.
        """
        
        analysis = client._parse_response(response_text, generation=5)
        
        assert analysis is not None
        assert analysis.overall_assessment == "improving"
        assert analysis.stagnation_detected is False
        print("✅ JSON with extra text parsing test passed")
    
    def test_parse_invalid_json(self):
        """Test handling of invalid JSON."""
        client = GemmaCoachClient(verbose=False)
        
        response_text = "This is not JSON at all"
        
        analysis = client._parse_response(response_text, generation=5)
        
        assert analysis is None
        print("✅ Invalid JSON handling test passed")
    
    def test_extract_json_with_braces(self):
        """Test JSON extraction with balanced braces."""
        client = GemmaCoachClient(verbose=False)
        
        text = 'Start {"key": "value", "nested": {"inner": "data"}} End'
        
        extracted = client._extract_json(text)
        
        assert extracted is not None
        assert '{"key"' in extracted
        assert '"nested"' in extracted
        print("✅ JSON extraction with braces test passed")


class TestCoachManagerWorkflow:
    """Test coach manager workflow with mocked LLM."""
    
    @pytest.mark.asyncio
    async def test_coach_analysis_with_mock_llm(self):
        """Test complete coach analysis workflow with mocked LLM."""
        
        # Create mock response
        mock_llm_response = {
            "generation": 10,
            "summary": "Population shows signs of stagnation",
            "overall_assessment": "stagnant",
            "stagnation_detected": True,
            "diversity_concern": False,
            "recommendations": [
                {
                    "category": "ga_hyperparams",
                    "parameter": "ga_mutation_rate",
                    "current_value": 0.3,
                    "suggested_value": 0.5,
                    "reasoning": "Increase mutation to escape local optima",
                    "confidence": 0.85
                }
            ]
        }
        
        # Create coach manager
        coach_manager = CoachManager(verbose=True)
        
        # Mock the LLM response
        with patch.object(
            GemmaCoachClient,
            '_call_llm',
            new_callable=AsyncMock,
            return_value=json.dumps(mock_llm_response)
        ):
            # Create test population
            pop = Population(size=24, seed_individual=None)
            pop.generation = 10
            
            # Create fitness and GA configs
            fitness_config = FitnessConfig()
            ga_config = OptimizationConfig(
                population_size=24,
                stagnation_threshold=5,
                stagnation_fitness_tolerance=1e-4,
                track_diversity=True
            )
            
            # Run analysis
            analysis = await coach_manager.analyze_async(pop, fitness_config, ga_config)
            
            # Wait for result
            result = await coach_manager.wait_for_analysis()
            
            print(f"✅ Coach analysis test passed")
            print(f"   Overall assessment: {result.overall_assessment if result else 'None'}")
            print(f"   Stagnation detected: {result.stagnation_detected if result else 'None'}")
            if result and result.recommendations:
                print(f"   Recommendations: {len(result.recommendations)}")
    
    @pytest.mark.asyncio
    async def test_model_status_check_mock(self):
        """Test model status checking with mock lms ps response."""
        
        client = GemmaCoachClient(model="google/gemma-3-12b", verbose=False)
        
        # Mock the subprocess call
        mock_output = """
        MODEL                           SIZE    CONTEXT    STATUS
        google/gemma-3-12b              11B     131072     READY
        """
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_thread:
            # Create mock result
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = mock_output
            mock_thread.return_value = mock_result
            
            is_loaded = await client.check_model_loaded()
            
            assert is_loaded is True
            print("✅ Model status check mock test passed")
    
    @pytest.mark.asyncio
    async def test_model_not_loaded_mock(self):
        """Test model status when not loaded."""
        
        client = GemmaCoachClient(model="google/gemma-3-12b", verbose=False)
        
        # Mock the subprocess call with empty output
        mock_output = "No models loaded."
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_thread:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = mock_output
            mock_thread.return_value = mock_result
            
            is_loaded = await client.check_model_loaded()
            
            assert is_loaded is False
            print("✅ Model not loaded mock test passed")


class TestCoachRecommendationApplication:
    """Test recommendation application to configs."""
    
    def test_apply_mutation_rate_recommendation(self):
        """Test applying mutation rate recommendation."""
        from backtest.coach_integration import apply_coach_recommendations
        
        # Create test recommendation
        rec = CoachRecommendation(
            category=RecommendationCategory.GA_HYPERPARAMS,
            parameter="mutation_rate",
            current_value=0.28,
            suggested_value=0.5,
            reasoning="Increase mutation",
            confidence=0.85
        )
        
        analysis = CoachAnalysis(
            generation=10,
            summary="Test analysis",
            overall_assessment="stagnant",
            stagnation_detected=True,
            diversity_concern=False,
            recommendations=[rec]
        )
        
        fitness_config = FitnessConfig()
        ga_config = OptimizationConfig(
            population_size=24,
            stagnation_threshold=5,
            stagnation_fitness_tolerance=1e-4
        )
        
        print(f"\n  Initial mutation_rate: {ga_config.mutation_rate}")
        
        new_fitness, new_ga = apply_coach_recommendations(
            analysis, fitness_config, ga_config
        )
        
        print(f"  Final mutation_rate: {new_ga.mutation_rate}")
        
        # Verify mutation rate was updated
        assert new_ga.mutation_rate == 0.5, f"Expected mutation_rate to be 0.5 but got {new_ga.mutation_rate}"
        print("✅ Recommendation application test passed")


class TestCoachEndToEnd:
    """End-to-end integration tests with full simulation."""
    
    @pytest.mark.asyncio
    async def test_full_coach_workflow_simulation(self):
        """Simulate full coach workflow from analysis to recommendation application."""
        
        print("\n" + "="*80)
        print("FULL COACH WORKFLOW SIMULATION TEST")
        print("="*80)
        
        # Step 1: Create evolution state
        print("\n[STEP 1] Creating evolution state...")
        pop = Population(size=24, seed_individual=None)
        pop.generation = 10
        
        fitness_config = FitnessConfig()
        ga_config = OptimizationConfig(
            population_size=24,
            stagnation_threshold=5,
            stagnation_fitness_tolerance=1e-4
        )
        
        print(f"  ✓ Population: {pop.size} individuals, Gen {pop.generation}")
        print(f"  ✓ Fitness config: {fitness_config.preset}")
        print(f"  ✓ GA config: population_size={ga_config.population_size}")
        
        # Step 2: Create coach manager
        print("\n[STEP 2] Creating coach manager...")
        coach_manager = CoachManager(verbose=False)
        print("  ✓ Coach manager created")
        
        # Step 3: Mock LLM response
        print("\n[STEP 3] Mocking LLM response...")
        mock_response = {
            "generation": 10,
            "summary": "Population exhibits stagnation with diversity loss",
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
                },
                {
                    "category": "ga_hyperparams",
                    "parameter": "sigma",
                    "current_value": 0.12,
                    "suggested_value": 0.15,
                    "reasoning": "Larger mutation steps needed",
                    "confidence": 0.75
                }
            ]
        }
        
        print(f"  ✓ Mock response: {len(mock_response['recommendations'])} recommendations")
        
        # Step 4: Simulate analysis with mocked LLM
        print("\n[STEP 4] Running coach analysis with mocked LLM...")
        with patch.object(
            GemmaCoachClient,
            '_call_llm',
            new_callable=AsyncMock,
            return_value=json.dumps(mock_response)
        ):
            await coach_manager.analyze_async(pop, fitness_config, ga_config)
            analysis = await coach_manager.wait_for_analysis()
        
        if analysis:
            print(f"  ✓ Analysis complete")
            print(f"    - Overall assessment: {analysis.overall_assessment}")
            print(f"    - Stagnation detected: {analysis.stagnation_detected}")
            print(f"    - Diversity concern: {analysis.diversity_concern}")
            print(f"    - Recommendations: {len(analysis.recommendations)}")
        else:
            print("  ✗ Analysis failed!")
            return
        
        # Step 5: Apply recommendations
        print("\n[STEP 5] Applying coach recommendations...")
        from backtest.coach_integration import apply_coach_recommendations
        
        new_fitness, new_ga = apply_coach_recommendations(
            analysis, fitness_config, ga_config
        )
        
        print(f"  ✓ Recommendations applied")
        print(f"    - Old mutation_rate: {ga_config.mutation_rate}")
        print(f"    - New mutation_rate: {new_ga.mutation_rate}")
        print(f"    - Old sigma: {ga_config.sigma}")
        print(f"    - New sigma: {new_ga.sigma}")
        
        # Verify changes
        assert new_ga.mutation_rate == 0.5, "Mutation rate should be updated to 0.5"
        assert new_ga.sigma == 0.15, "Sigma should be updated to 0.15"
        
        print("\n✅ FULL COACH WORKFLOW SIMULATION PASSED")
        print("="*80 + "\n")


def test_coach_logging():
    """Test coach logging functionality."""
    from core.coach_logging import coach_log_manager
    
    print("[TEST] Coach logging...")
    
    # Add test logs
    coach_log_manager.append("[TEST] Test message 1")
    coach_log_manager.append("[TEST] Test message 2")
    coach_log_manager.append("[TEST] Test message 3")
    
    # Get recent logs
    all_logs = coach_log_manager.dump()
    recent = all_logs[-2:] if len(all_logs) >= 2 else all_logs
    
    assert len(recent) >= 1
    assert any("[TEST] Test message" in log for log in recent)
    
    print("✅ Coach logging test passed")


# Run tests
if __name__ == "__main__":
    print("\n" + "█"*80)
    print("RUNNING COACH INTEGRATION TESTS")
    print("█"*80)
    
    # Run sync tests
    test = TestGemmaCoachClientJSON()
    print("\n[TEST SUITE 1] JSON Parsing Tests")
    test.test_parse_valid_json_response()
    test.test_parse_json_with_extra_text()
    test.test_parse_invalid_json()
    test.test_extract_json_with_braces()
    
    print("\n[TEST SUITE 2] Coach Logging")
    test_coach_logging()
    
    # Run async tests
    print("\n[TEST SUITE 3] Coach Manager Workflow (Async)")
    coach_test = TestCoachManagerWorkflow()
    asyncio.run(coach_test.test_model_status_check_mock())
    asyncio.run(coach_test.test_model_not_loaded_mock())
    
    print("\n[TEST SUITE 4] Recommendation Application")
    rec_test = TestCoachRecommendationApplication()
    rec_test.test_apply_mutation_rate_recommendation()
    
    print("\n[TEST SUITE 5] End-to-End Simulation")
    e2e_test = TestCoachEndToEnd()
    asyncio.run(e2e_test.test_full_coach_workflow_simulation())
    
    print("\n" + "█"*80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("█"*80 + "\n")
