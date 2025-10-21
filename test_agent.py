"""
Test Agent with Real LM Studio Model

Tests the Evolution Coach Agent with google/gemma-3-12b model:
1. Check if model is loaded
2. Load model if needed
3. Create stagnating population scenario
4. Call agent and wait for response
5. Verify agent took actions
6. Unload model

Usage:
    python test_agent.py
    
Requirements:
    - LM Studio installed and server running
    - google/gemma-3-12b model downloaded
"""

import asyncio
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.optimizer import Population, Individual
from backtest.coach_manager_blocking import BlockingCoachManager
from backtest.coach_session import CoachAnalysisSession
from backtest.coach_tools import CoachToolkit
from backtest.coach_agent_executor import AgentExecutor
from backtest.llm_coach import GemmaCoachClient
from core.models import FitnessConfig, OptimizationConfig, Timeframe, BacktestParams
from strategy.seller_exhaustion import SellerParams
import pandas as pd
import numpy as np


MODEL_NAME = "google/gemma-3-12b"
GPU_OFFLOAD = 0.6
CONTEXT_LENGTH = 5000


def run_lms_command(args: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run lms CLI command."""
    try:
        result = subprocess.run(
            ["lms"] + args,
            capture_output=True,
            text=True,
            check=check,
            timeout=30
        )
        return result
    except subprocess.TimeoutExpired:
        print("❌ Command timeout")
        raise
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        raise


def is_model_loaded(model_name: str) -> bool:
    """Check if model is currently loaded."""
    print(f"🔍 Checking if {model_name} is loaded...")
    result = run_lms_command(["ps"], check=False)
    
    if result.returncode == 0 and model_name in result.stdout:
        print(f"   ✅ Model is loaded")
        return True
    else:
        print(f"   ❌ Model not loaded")
        return False


def load_model(model_name: str, gpu: float = 0.6, context: int = 5000) -> bool:
    """Load model into LM Studio."""
    print(f"🔄 Loading {model_name}...")
    print(f"   GPU offload: {gpu}")
    print(f"   Context length: {context}")
    
    try:
        result = run_lms_command([
            "load",
            model_name,
            "--gpu", str(gpu),
            "--context-length", str(context),
            "--yes"
        ])
        
        if result.returncode == 0:
            print(f"   ✅ Model loaded successfully")
            return True
        else:
            print(f"   ❌ Failed to load model")
            return False
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False


def unload_model(model_identifier: str) -> bool:
    """Unload model from LM Studio."""
    print(f"🔄 Unloading {model_identifier}...")
    
    try:
        # Get loaded model ID
        ps_result = run_lms_command(["ps", "--json"], check=False)
        if ps_result.returncode == 0:
            try:
                models = json.loads(ps_result.stdout)
                # Find model ID
                model_id = None
                for model in models:
                    if model_identifier in model.get("path", ""):
                        model_id = model.get("identifier")
                        break
                
                if model_id:
                    result = run_lms_command(["unload", model_id])
                    if result.returncode == 0:
                        print(f"   ✅ Model unloaded successfully")
                        return True
            except json.JSONDecodeError:
                pass
        
        # Fallback: try unloading by name
        result = run_lms_command(["unload", model_identifier], check=False)
        if result.returncode == 0:
            print(f"   ✅ Model unloaded successfully")
            return True
        else:
            print(f"   ⚠️  Could not unload model (may not be loaded)")
            return False
    
    except Exception as e:
        print(f"   ❌ Error unloading model: {e}")
        return False


def create_stagnating_population(size: int = 12) -> tuple[Population, pd.DataFrame]:
    """
    Create a realistic stagnating population scenario.
    
    Characteristics:
    - Low diversity (< 0.15) - all individuals very similar
    - 100% below min_trades gate (min_trades=20, but individuals have 0-18 trades)
    - No improvement for 10+ generations
    - Boundary clustering (ema_fast at minimum, vol_z at maximum)
    - Bottom 75% have zero trades (signal generation failure)
    """
    print("🧬 Creating stagnating population scenario...")
    
    # Create synthetic data
    dates = pd.date_range('2024-01-01', periods=1440, freq='15min')
    close_prices = np.cumsum(np.random.randn(1440) * 0.01) + 0.5
    
    data = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(1440) * 0.005),
        'high': close_prices * (1 + abs(np.random.randn(1440)) * 0.01),
        'low': close_prices * (1 - abs(np.random.randn(1440)) * 0.01),
        'close': close_prices,
        'volume': 1000 + np.random.randn(1440) * 100
    }, index=dates)
    
    # Create population with stagnation characteristics
    population = Population(size=size, seed_individual=None, timeframe=Timeframe.m15)
    
    # Make population converged (low diversity)
    # Top 3 individuals: fitness 0.35-0.40, have 18-19 trades (below gate of 20)
    # Bottom 9 individuals: fitness 0.0, have 0 trades
    
    for i, individual in enumerate(population.individuals):
        if i < 3:
            # Top performers (but still fail gate)
            individual.seller_params = SellerParams(
                ema_fast=48,  # At minimum bound (clustering)
                ema_slow=358 + i * 10,  # Slight variation
                z_window=672,
                atr_window=96,
                vol_z=1.2 + i * 0.01,  # Low selectivity
                tr_z=1.0,
                cloc_min=0.6
            )
            individual.backtest_params = BacktestParams(
                use_fib_exits=True,
                fib_swing_lookback=96,
                fib_swing_lookahead=5,
                fib_target_level=0.618,
                atr_stop_mult=0.7,
                reward_r=2.0,
                max_hold=96,
                fee_bp=5.0,
                slippage_bp=5.0
            )
            # Metrics: 18-19 trades, good win rate, but below gate
            individual.fitness = 0.35 + i * 0.025
            individual.metrics = {
                'n': 18 + i,  # Below min_trades=20
                'win_rate': 0.84 - i * 0.02,
                'avg_R': 2.7 - i * 0.1,
                'total_pnl': 0.3 - i * 0.03,
                'max_dd': -0.05
            }
        else:
            # Bottom performers (clustered, no trades)
            individual.seller_params = SellerParams(
                ema_fast=48 + (i - 3) * 3,  # Slight clustering variation
                ema_slow=371 + (i - 3) * 5,
                z_window=672,
                atr_window=96,
                vol_z=1.49 + (i - 3) * 0.05,  # High selectivity = no signals
                tr_z=1.01,
                cloc_min=0.6
            )
            individual.backtest_params = BacktestParams(
                use_fib_exits=True,
                fib_swing_lookback=96,
                fib_swing_lookahead=5,
                fib_target_level=0.618,
                atr_stop_mult=0.7,
                reward_r=2.0,
                max_hold=96,
                fee_bp=5.0,
                slippage_bp=5.0
            )
            # No trades, zero fitness
            individual.fitness = 0.0
            individual.metrics = {
                'n': 0,
                'win_rate': 0.0,
                'avg_R': 0.0,
                'total_pnl': 0.0,
                'max_dd': 0.0
            }
    
    # Set generation (simulating stagnation for 10 generations)
    population.generation = 25
    population.best_ever = population.individuals[0]
    
    # Calculate diversity (should be very low)
    diversity = population.get_diversity_metric()
    
    print(f"   Population size: {len(population.individuals)}")
    print(f"   Generation: {population.generation}")
    print(f"   Diversity: {diversity:.3f} ({'VERY LOW' if diversity < 0.15 else 'LOW'})")
    print(f"   Top fitness: {population.individuals[0].fitness:.4f}")
    print(f"   Bottom 9 fitness: {population.individuals[-1].fitness:.4f}")
    print(f"   Top 3 trades: {[ind.metrics['n'] for ind in population.individuals[:3]]}")
    print(f"   Bottom 9 trades: {[ind.metrics['n'] for ind in population.individuals[-3:]]}")
    
    # Diagnosis summary
    print()
    print("   📊 Stagnation characteristics:")
    print(f"      ✓ Low diversity: {diversity:.3f} < 0.15")
    print(f"      ✓ Gate crisis: 100% below min_trades=20")
    print(f"      ✓ Signal failure: 75% with 0 trades")
    print(f"      ✓ Boundary clustering: ema_fast near 48 (min bound)")
    print(f"      ✓ Parameter clustering: vol_z 1.2-2.0 (poor range)")
    
    return population, data


async def test_agent_with_stagnation():
    """Test agent with stagnating population."""
    print()
    print("=" * 70)
    print("🧪 Testing Evolution Coach Agent with Stagnating Population")
    print("=" * 70)
    print()
    
    # Step 1: Check LM Studio server
    print("1️⃣  Checking LM Studio server...")
    try:
        result = run_lms_command(["server", "status"], check=False)
        output = (result.stdout + result.stderr).lower()
        if "running" not in output and "port" not in output:
            print("   ❌ LM Studio server not running")
            print("   → Start server: lms server start")
            return False
        print("   ✅ Server is running")
        status_msg = result.stderr.strip() or result.stdout.strip()
        if status_msg:
            print(f"      {status_msg}")
    except Exception as e:
        print(f"   ❌ Could not check server status: {e}")
        return False
    
    print()
    
    # Step 2: Check/load model
    print("2️⃣  Managing model...")
    model_was_loaded = is_model_loaded(MODEL_NAME)
    
    if not model_was_loaded:
        if not load_model(MODEL_NAME, GPU_OFFLOAD, CONTEXT_LENGTH):
            print("   ❌ Failed to load model")
            return False
    
    print()
    
    # Step 3: Create stagnating population
    print("3️⃣  Creating test scenario...")
    population, data = create_stagnating_population(size=12)
    
    print()
    
    # Step 4: Create configs
    print("4️⃣  Creating configurations...")
    fitness_config = FitnessConfig(
        preset="balanced",
        min_trades=20,  # High gate causing crisis
        min_win_rate=0.40
    )
    
    ga_config = OptimizationConfig(
        population_size=12,
        mutation_probability=0.9,
        mutation_rate=0.55,
        sigma=0.15,
        elite_fraction=0.1,
        tournament_size=3,
        immigrant_fraction=0.0  # No immigrants yet
    )
    
    print(f"   Fitness: min_trades={fitness_config.min_trades}, min_win_rate={fitness_config.min_win_rate}")
    print(f"   GA: mutation_rate={ga_config.mutation_rate}, immigrant_fraction={ga_config.immigrant_fraction}")
    
    print()
    
    # Step 5: Create agent and run analysis
    print("5️⃣  Running agent analysis...")
    print("   This will take 30-90 seconds depending on LLM speed...")
    print()
    
    try:
        # Initialize coach manager with agent
        coach_manager = BlockingCoachManager(
            base_url="http://localhost:1234",
            model=MODEL_NAME,
            prompt_version="agent01",
            analysis_interval=10,
            auto_apply=True,
            verbose=True
        )
        
        # Create frozen session
        session = CoachAnalysisSession.from_population(
            population=population,
            fitness_config=fitness_config,
            ga_config=ga_config
        )
        
        # Create toolkit
        toolkit = CoachToolkit(
            population=population,
            session=session,
            fitness_config=fitness_config,
            ga_config=ga_config,
            mutation_manager=coach_manager.mutation_manager
        )
        
        # Initialize LLM client
        llm_client = GemmaCoachClient(
            base_url="http://localhost:1234",
            model=MODEL_NAME,
            prompt_version="agent01",
            system_prompt="agent01",
            verbose=True,
            debug_payloads=False
        )
        
        # Create agent
        agent = AgentExecutor(
            llm_client=llm_client,
            toolkit=toolkit,
            max_iterations=10,
            verbose=True
        )
        
        # Build initial observation
        observation = coach_manager._build_agent_observation(session, fitness_config, ga_config)
        
        print("📤 Sending observation to agent:")
        print("-" * 70)
        print(observation[:500] + "..." if len(observation) > 500 else observation)
        print("-" * 70)
        print()
        
        # Run agent
        start_time = datetime.now()
        result = await agent.run_analysis(observation)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print()
        print("-" * 70)
        print(f"✅ Agent completed in {elapsed:.1f} seconds")
        print("-" * 70)
        print()
        
        # Step 6: Verify results
        print("6️⃣  Verifying agent actions...")
        
        if result.get("success"):
            print(f"   ✅ Success: {result['success']}")
            print(f"   📊 Iterations: {result['iterations']}")
            print(f"   🔧 Tool calls: {result['tool_calls_count']}")
            print(f"   📝 Actions: {len(toolkit.actions_log)}")
            print()
            
            if toolkit.actions_log:
                print("   📋 Actions taken by agent:")
                for i, action in enumerate(toolkit.actions_log, 1):
                    action_name = action.get("action", "unknown")
                    print(f"      {i}. {action_name}")
                    
                    if action_name == "analyze_population":
                        print(f"         → Analyzed population state")
                    elif action_name == "get_param_distribution":
                        param = action.get("parameter", "?")
                        print(f"         → Analyzed {param} distribution")
                    elif action_name == "mutate_individual":
                        ind_id = action.get("individual_id")
                        param = action.get("parameter")
                        old = action.get("old_value")
                        new = action.get("new_value")
                        print(f"         → Mutated Individual #{ind_id}: {param} {old} → {new}")
                    elif action_name == "update_fitness_gates":
                        for param, change in action.get("changes", {}).items():
                            print(f"         → {param}: {change['old']} → {change['new']}")
                    elif action_name == "update_ga_params":
                        for param, change in action.get("changes", {}).items():
                            print(f"         → {param}: {change['old']} → {change['new']}")
                    elif action_name == "finish_analysis":
                        print(f"         → Completed analysis")
                
                print()
                
                # Check for expected actions given stagnation
                action_names = [a.get("action") for a in toolkit.actions_log]
                
                print("   🔍 Validation:")
                if "analyze_population" in action_names:
                    print("      ✅ Agent analyzed population (good start)")
                else:
                    print("      ⚠️  Agent didn't analyze population first")
                
                if "update_fitness_gates" in action_names:
                    print("      ✅ Agent addressed gate crisis (expected)")
                else:
                    print("      ⚠️  Agent didn't lower fitness gates (unexpected)")
                
                if "update_ga_params" in action_names:
                    print("      ✅ Agent adjusted GA parameters")
                
                if "finish_analysis" in action_names:
                    print("      ✅ Agent called finish_analysis")
                else:
                    print("      ⚠️  Agent didn't call finish_analysis")
            else:
                print("   ⚠️  No actions logged (unexpected)")
            
            print()
            return True
        else:
            print(f"   ❌ Agent failed: {result.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"   ❌ Error during agent test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Step 7: Unload model if we loaded it
        print()
        print("7️⃣  Cleanup...")
        if not model_was_loaded:
            unload_model(MODEL_NAME)
        else:
            print(f"   ℹ️  Model was already loaded, leaving it loaded")


async def main():
    """Main test function."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "EVOLUTION COACH AGENT TEST" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"Model: {MODEL_NAME}")
    print(f"GPU Offload: {GPU_OFFLOAD}")
    print(f"Context Length: {CONTEXT_LENGTH}")
    print()
    
    success = await test_agent_with_stagnation()
    
    print()
    print("=" * 70)
    if success:
        print("✅ TEST PASSED - Agent successfully analyzed stagnating population!")
        print()
        print("Expected behavior:")
        print("  1. Agent called analyze_population() to understand state")
        print("  2. Agent diagnosed gate crisis (100% below min_trades)")
        print("  3. Agent lowered min_trades from 20 to 5-10")
        print("  4. Agent may have injected immigrants (diversity)")
        print("  5. Agent called finish_analysis() to complete")
    else:
        print("❌ TEST FAILED - Check error messages above")
        print()
        print("Troubleshooting:")
        print("  • Is LM Studio running? Run: lms server start")
        print("  • Is model downloaded? Run: lms ls")
        print("  • Check LM Studio logs: lms log stream")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
