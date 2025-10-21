#!/usr/bin/env python3
"""
Integration test with REAL LM Studio instance.

This test connects to an actual LM Studio server and runs real LLM inference.
Use this to debug coach integration with a live model.

Requirements:
1. LM Studio running: `lms load google/gemma-3-12b --gpu=0.6`
2. Model status: `lms ps` should show STATUS = READY
3. Run this test: `poetry run python tests/test_coach_with_real_lm_studio.py`

What this test does:
1. Checks if model is loaded via "lms ps"
2. Creates evolution state from dummy logs
3. Calls the LLM for first analysis (Gen 5)
4. Unloads model to clear context
5. Reloads model
6. Calls the LLM for second analysis (Gen 10)
7. Verifies no "Default client is already created" errors
"""

import sys
import asyncio
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.llm_coach import GemmaCoachClient, HAS_LMSTUDIO
from backtest.coach_protocol import EvolutionState


async def test_real_lm_studio():
    """Test with real LM Studio instance."""
    
    print("\n" + "="*80)
    print("REAL LM STUDIO INTEGRATION TEST")
    print("="*80)
    
    # Check if lmstudio is installed
    if not HAS_LMSTUDIO:
        print("\n❌ SKIP: lmstudio package not installed")
        print("   Install with: pip install lmstudio")
        return False
    
    print("\n[STEP 1] Create coach client")
    print("─" * 80)
    
    try:
        client = GemmaCoachClient(
            model="google/gemma-3-12b",
            verbose=True,
            prompt_version="async_coach_v1"
        )
        print("✓ Client created")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return False
    
    print("\n[STEP 2] Check if model is loaded")
    print("─" * 80)
    
    try:
        is_loaded = await client.check_model_loaded()
        print(f"✓ Model status check completed: {is_loaded}")
        
        if not is_loaded:
            print("\n⚠️  Model not loaded. Loading model...")
            try:
                await client.load_model()
                print("✓ Model loaded")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                return False
    except Exception as e:
        print(f"❌ Failed to check model status: {e}")
        return False
    
    print("\n[STEP 3] First analysis at Generation 5")
    print("─" * 80)
    
    try:
        # Create evolution state
        state_5 = EvolutionState(
            generation=5,
            population_size=24,
            mean_fitness=-12.0,
            std_fitness=33.0,
            best_fitness=0.8,
            best_trades=10,
            best_win_rate=0.70,
            best_avg_r=0.60,
            best_pnl=0.25,
            below_min_trades_percent=30.0,
            mean_trade_count=9.0,
            diversity_metric=0.45,
            recent_improvement=0.05,
            is_stagnant=False,
            fitness_config_dict={"preset": "balanced"},
            ga_config_dict={"population_size": 24}
        )
        
        # Create dummy logs
        logs_5 = [
            "[Gen    1] Best: 5 trades, 50% WR, 0.20 avg_R",
            "[Gen    2] Best: 7 trades, 60% WR, 0.35 avg_R",
            "[Gen    3] Best: 8 trades, 65% WR, 0.45 avg_R",
            "[Gen    4] Best: 9 trades, 68% WR, 0.55 avg_R",
            "[Gen    5] Best: 10 trades, 70% WR, 0.60 avg_R",
        ]
        
        print(f"  📊 Evolution state: Gen {state_5.generation}, {state_5.best_trades} trades")
        print(f"  📝 Logs: {len(logs_5)} lines")
        print(f"  🤖 Calling LLM for analysis...")
        
        analysis_1 = await client.analyze_evolution(state_5, logs_5)
        
        if analysis_1 is None:
            print("❌ LLM returned None")
            return False
        
        print(f"✓ Analysis received:")
        print(f"    - Assessment: {analysis_1.overall_assessment}")
        print(f"    - Stagnation detected: {analysis_1.stagnation_detected}")
        print(f"    - Diversity concern: {analysis_1.diversity_concern}")
        print(f"    - Recommendations: {len(analysis_1.recommendations)}")
        
        for i, rec in enumerate(analysis_1.recommendations, 1):
            print(f"      Rec {i}: {rec.parameter}")
            print(f"        → From {rec.current_value} to {rec.suggested_value}")
            print(f"        → Confidence: {rec.confidence:.0%}")
    
    except Exception as e:
        print(f"❌ First analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[STEP 4] Unload model to free context window")
    print("─" * 80)
    
    try:
        print("  🗑️  Unloading model...")
        await client.unload_model()
        print("✓ Model unloaded and client cleared")
    except Exception as e:
        print(f"❌ Failed to unload model: {e}")
        return False
    
    print("\n[STEP 5] Reload model for second analysis")
    print("─" * 80)
    
    try:
        print("  📦 Loading model again...")
        await client.load_model()
        print("✓ Model reloaded successfully")
    except Exception as e:
        print(f"❌ Failed to reload model: {e}")
        return False
    
    print("\n[STEP 6] Second analysis at Generation 10")
    print("─" * 80)
    
    try:
        # Create evolution state
        state_10 = EvolutionState(
            generation=10,
            population_size=24,
            mean_fitness=-10.5,
            std_fitness=32.1,
            best_fitness=1.2,
            best_trades=15,
            best_win_rate=0.73,
            best_avg_r=0.85,
            best_pnl=0.45,
            below_min_trades_percent=25.0,
            mean_trade_count=12.3,
            diversity_metric=0.38,
            recent_improvement=0.0,
            is_stagnant=True,
            fitness_config_dict={"preset": "balanced"},
            ga_config_dict={"population_size": 24}
        )
        
        # Create dummy logs
        logs_10 = [
            "[Gen    6] Best: 12 trades, 72% WR, 0.70 avg_R",
            "[Gen    7] Best: 14 trades, 72% WR, 0.75 avg_R",
            "[Gen    8] Best: 15 trades, 73% WR, 0.80 avg_R",
            "[Gen    9] Best: 15 trades, 73% WR, 0.82 avg_R",
            "[Gen   10] Best: 15 trades, 73% WR, 0.85 avg_R - NO IMPROVEMENT",
        ]
        
        print(f"  📊 Evolution state: Gen {state_10.generation}, {state_10.best_trades} trades")
        print(f"  📝 Logs: {len(logs_10)} lines")
        print(f"  🤖 Calling LLM for analysis...")
        
        analysis_2 = await client.analyze_evolution(state_10, logs_10)
        
        if analysis_2 is None:
            print("❌ LLM returned None")
            return False
        
        print(f"✓ Analysis received:")
        print(f"    - Assessment: {analysis_2.overall_assessment}")
        print(f"    - Stagnation detected: {analysis_2.stagnation_detected}")
        print(f"    - Diversity concern: {analysis_2.diversity_concern}")
        print(f"    - Recommendations: {len(analysis_2.recommendations)}")
        
        for i, rec in enumerate(analysis_2.recommendations, 1):
            print(f"      Rec {i}: {rec.parameter}")
            print(f"        → From {rec.current_value} to {rec.suggested_value}")
            print(f"        → Confidence: {rec.confidence:.0%}")
    
    except Exception as e:
        print(f"❌ Second analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ ALL REAL LM STUDIO TESTS PASSED")
    print("="*80)
    print("\nKey findings:")
    print("  ✓ Model loaded and ready for inference")
    print("  ✓ First analysis completed successfully")
    print("  ✓ Model unloaded and context cleared")
    print("  ✓ Model reloaded without client conflicts")
    print("  ✓ Second analysis completed successfully")
    print("  ✓ NO 'Default client is already created' errors!")
    print("\nCoach integration is PRODUCTION READY! 🚀\n")
    
    # Print logs
    print("\n[COACH LOGS]")
    print("─" * 80)
    all_logs = # coach_log_manager removed: dump()
    for log in all_logs[-20:]:  # Last 20 entries
        print(log)
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_real_lm_studio())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏸️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
