"""
Evolution Coach Agent Demo

Demonstrates agent-based coaching with tool-calling LLM.

Usage:
    # With agent-based coach
    python examples/agent_coach_demo.py
    
    # Watch agent work:
    # 1. Agent calls analyze_population() to understand state
    # 2. Agent diagnoses problems (gate crisis, low diversity, etc.)
    # 3. Agent takes strategic actions (lower gates, mutate individuals, etc.)
    # 4. Agent calls finish_analysis() when done
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.optimizer import Population
from backtest.coach_manager_blocking import BlockingCoachManager
from core.models import FitnessConfig, OptimizationConfig, Timeframe
from strategy.seller_exhaustion import SellerParams
from backtest.engine import BacktestParams
import pandas as pd
import numpy as np


def create_test_data(n_bars: int = 1440) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')
    
    # Random walk with some trend
    close_prices = np.cumsum(np.random.randn(n_bars) * 0.01) + 0.5
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(n_bars) * 0.005),
        'high': close_prices * (1 + abs(np.random.randn(n_bars)) * 0.01),
        'low': close_prices * (1 - abs(np.random.randn(n_bars)) * 0.01),
        'close': close_prices,
        'volume': 1000 + np.random.randn(n_bars) * 100
    }, index=dates)
    
    return df


async def main():
    """Run agent-based coach demo."""
    print("🤖 Evolution Coach Agent Demo")
    print("=" * 60)
    print()
    
    # 1. Create test data
    print("📊 Creating test data...")
    data = create_test_data(n_bars=1440)  # 1 day of 15m bars
    print(f"   Created {len(data)} bars")
    print()
    
    # 2. Initialize population
    print("🧬 Initializing population...")
    population = Population(
        size=12,
        seed_individual=None,
        timeframe=Timeframe.m15
    )
    print(f"   Population size: {len(population.individuals)}")
    print()
    
    # 3. Create configs
    fitness_config = FitnessConfig(
        preset="balanced",
        min_trades=20,  # Intentionally high to trigger gate crisis
        min_win_rate=0.40
    )
    
    ga_config = OptimizationConfig(
        population_size=12,
        mutation_probability=0.9,
        mutation_rate=0.55,
        sigma=0.15,
        elite_fraction=0.1,
        tournament_size=3,
        immigrant_fraction=0.0  # Start with no immigrants
    )
    
    print("⚙️  Fitness config:")
    print(f"   min_trades: {fitness_config.min_trades}")
    print(f"   min_win_rate: {fitness_config.min_win_rate}")
    print()
    
    print("⚙️  GA config:")
    print(f"   mutation_rate: {ga_config.mutation_rate}")
    print(f"   elite_fraction: {ga_config.elite_fraction}")
    print(f"   immigrant_fraction: {ga_config.immigrant_fraction}")
    print()
    
    # 4. Initialize coach manager
    print("🤖 Initializing coach manager...")
    coach_manager = BlockingCoachManager(
        base_url="http://localhost:1234",
        model="google/gemma-3-12b",
        prompt_version="agent01",
        analysis_interval=10,
        auto_apply=True,
        verbose=True
    )
    print("   Coach manager ready")
    print()
    
    # 5. Check if we should analyze
    generation = 10  # Simulate generation 10
    should_analyze = coach_manager.should_analyze(generation)
    
    if should_analyze:
        print(f"✅ Generation {generation} - Time for coach analysis!")
        print()
        print("🔄 Starting agent-based analysis...")
        print("=" * 60)
        print()
        
        try:
            # Use agent-based analysis
            success, mutations = await coach_manager.analyze_and_apply_with_agent(
                population=population,
                fitness_config=fitness_config,
                ga_config=ga_config,
                current_data=data
            )
            
            if success:
                print()
                print("=" * 60)
                print("✅ Agent analysis completed successfully!")
                print()
                print("📋 Actions taken:")
                
                # Show what agent did
                if coach_manager.last_session:
                    toolkit = coach_manager.last_session
                    if hasattr(toolkit, 'actions_log'):
                        for i, action in enumerate(toolkit.actions_log, 1):
                            action_name = action.get("action", "unknown")
                            print(f"   {i}. {action_name}")
                            
                            if action_name == "mutate_individual":
                                print(f"      → Individual #{action['individual_id']}: "
                                      f"{action['parameter']} {action['old_value']} → {action['new_value']}")
                            elif action_name == "update_fitness_gates":
                                for param, change in action.get("changes", {}).items():
                                    print(f"      → {param}: {change['old']} → {change['new']}")
                            elif action_name == "update_ga_params":
                                for param, change in action.get("changes", {}).items():
                                    print(f"      → {param}: {change['old']} → {change['new']}")
                
                print()
                print("💡 Next steps:")
                print("   1. Evolution will resume with modified population")
                print("   2. Agent changes will affect next generations")
                print("   3. Coach will analyze again at generation 20")
            else:
                print("❌ Agent analysis failed")
                print("   Check LM Studio is running: lms ps")
        
        except Exception as e:
            print(f"❌ Error during agent analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⏭️  Generation {generation} - Not time for coach yet")
        print(f"   Next analysis at generation: {generation + coach_manager.analysis_interval}")
    
    print()
    print("=" * 60)
    print("Demo complete! 🎉")
    print()
    print("To use in your optimization:")
    print("  1. Enable coach in .env: COACH_ENABLED=true")
    print("  2. Set interval: COACH_ANALYSIS_INTERVAL=10")
    print("  3. Run optimization: python cli.py ui")
    print("  4. Watch agent work at generation 10, 20, 30...")


if __name__ == "__main__":
    asyncio.run(main())
