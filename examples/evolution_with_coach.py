#!/usr/bin/env python3
"""
Example: Running GA Evolution with Async Coach

Demonstrates:
- CoachManager setup
- Non-blocking coach analysis
- Automatic recommendation application
- Log tracking
- Model management
"""

import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.coach_manager import CoachManager
from backtest.optimizer import Population, evolution_step
from core.models import FitnessConfig, OptimizationConfig, Timeframe
from data.provider import DataProvider
from strategy.seller_exhaustion import build_features


async def run_evolution_with_coach(
    ticker: str = "X:ADAUSD",
    from_date: str = "2024-12-01",
    to_date: str = "2024-12-31",
    timeframe: Timeframe = Timeframe.m15,
    n_generations: int = 50,
    population_size: int = 56,
    use_coach: bool = True
):
    """
    Run GA evolution with Evolution Coach.
    
    CRITICAL WORKFLOW:
    - First analysis at Gen N (configured in .env: COACH_FIRST_ANALYSIS_GENERATION)
    - After recommendations applied → model reloads → triggers next analysis
    - NOT on fixed interval - analysis triggered by model reload
    
    Args:
        ticker: Trading pair
        from_date: Start date
        to_date: End date
        timeframe: Timeframe for data
        n_generations: Number of generations to run
        population_size: Population size
        use_coach: Enable/disable coach
    """
    print("=" * 80)
    print("EVOLUTION WITH COACH")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Period: {from_date} to {to_date}")
    print(f"Timeframe: {timeframe.value}")
    print(f"Generations: {n_generations}")
    print(f"Population: {population_size}")
    print(f"Coach: {'Enabled' if use_coach else 'Disabled'}")
    print("=" * 80)
    print()
    
    # Fetch data
    print("📊 Fetching data...")
    dp = DataProvider()
    data = await dp.fetch_15m(ticker, from_date, to_date)
    print(f"✅ Loaded {len(data)} bars\n")
    
    # Initialize coach
    coach = None
    if use_coach:
        from config.settings import settings
        print("🤖 Initializing Evolution Coach...")
        coach = CoachManager(
            base_url="http://localhost:1234",
            # All coach params loaded from settings (.env)
            verbose=True
        )
        print("✅ Coach initialized")
        print(f"   Model: {coach.model}")
        print(f"   Prompt: {coach.prompt_version}")
        print(f"   First analysis: Generation {coach.first_analysis_generation}")
        print(f"   Auto-reload model: {coach.auto_reload_model} (clears context window)\n")
    
    # Initialize GA configs
    print("⚙️  Initializing GA configuration...")
    fitness_cfg = FitnessConfig(
        preset="balanced",
        curriculum_enabled=False,  # Coach may enable this
        fitness_function_type="hard_gates"  # Coach may switch to soft_penalties
    )
    
    ga_cfg = OptimizationConfig(
        population_size=population_size,
        tournament_size=6,
        elite_fraction=0.25,
        mutation_probability=0.90,
        mutation_rate=0.70,
        sigma=0.10,
        immigrant_fraction=0.0,  # Coach may increase this
        track_diversity=True,
        track_stagnation=True,
        stagnation_threshold=5
    )
    print("✅ Configuration ready\n")
    
    # Initialize population
    print("🧬 Initializing population...")
    population = Population(size=population_size, seed_individual=None, timeframe=timeframe)
    print(f"✅ Population initialized: {population_size} individuals\n")
    
    # Evolution loop
    print("🔄 Starting evolution...")
    print("=" * 80)
    
    start_time = datetime.now()
    
    for gen in range(n_generations):
        gen_start = datetime.now()
        
        # Log generation header
        print(f"\n{'='*80}")
        print(f"GENERATION {gen}")
        print(f"{'='*80}")
        
        if coach:
            coach.add_log(gen, f"=== Generation {gen} ===")
        
        # Run evolution step
        population = evolution_step(
            population=population,
            data=data,
            tf=timeframe,
            fitness_config=fitness_cfg,
            ga_config=ga_cfg
        )
        
        # Get statistics
        stats = population.get_stats()
        best = population.best_ever or population.get_best()
        
        # Log statistics
        print(f"\n📊 Population Statistics:")
        print(f"   Mean Fitness: {stats['mean_fitness']:7.4f} ± {stats['std_fitness']:.4f}")
        print(f"   Best Fitness: {stats['max_fitness']:7.4f}")
        print(f"   Best Ever:    {best.fitness:7.4f}")
        
        if best.metrics:
            print(f"\n🏆 Best Individual:")
            print(f"   Trades:   {best.metrics.get('n', 0):3d}")
            print(f"   Win Rate: {best.metrics.get('win_rate', 0.0):5.1%}")
            print(f"   Avg R:    {best.metrics.get('avg_R', 0.0):6.2f}")
            print(f"   Total PnL: ${best.metrics.get('total_pnl', 0.0):7.4f}")
        
        # Log to coach
        if coach:
            coach.add_log(gen, f"Mean fitness: {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
            coach.add_log(gen, f"Best fitness: {stats['max_fitness']:.4f}")
            if best.metrics:
                coach.add_log(gen, f"Best: {best.metrics.get('n', 0)} trades, WR {best.metrics.get('win_rate', 0.0):.1%}")
            
            # Diversity and stagnation
            if 'diversity' in population.history[-1]:
                diversity = population.history[-1]['diversity']
                print(f"   Diversity: {diversity:.2f}")
                coach.add_log(gen, f"Diversity: {diversity:.2f}")
            
            if population.history:
                recent_gens = population.history[-5:]
                if len(recent_gens) >= 2:
                    best_fits = [h['best_fitness'] for h in recent_gens]
                    improvement = max(best_fits) - min(best_fits)
                    print(f"   Recent improvement: {improvement:.4f}")
                    coach.add_log(gen, f"Recent improvement: {improvement:.4f}")
        
        gen_time = (datetime.now() - gen_start).total_seconds()
        print(f"\n⏱️  Generation time: {gen_time:.2f}s")
        
        # Coach analysis (non-blocking, every 10 gens by default)
        if coach and coach.should_analyze(gen):
            print(f"\n{'='*80}")
            print(f"🤖 TRIGGERING COACH ANALYSIS (Gen {gen})")
            print(f"{'='*80}")
            
            # Start analysis in background (evolution continues)
            await coach.analyze_async(population, fitness_cfg, ga_cfg)
            print("✅ Coach analysis started (running in background)")
            print("   Evolution will continue while coach analyzes...")
        
        # Check if previous analysis finished
        if coach and coach.pending_analysis and coach.pending_analysis.done():
            print(f"\n{'='*80}")
            print("📥 COACH ANALYSIS COMPLETE")
            print(f"{'='*80}")
            
            try:
                analysis = await coach.wait_for_analysis()
                
                if analysis:
                    print(f"\n📋 SUMMARY:")
                    print(analysis.summary)
                    
                    print(f"\n⚠️  FLAGS:")
                    print(f"   Stagnation: {'Yes' if analysis.stagnation_detected else 'No'}")
                    print(f"   Diversity concern: {'Yes' if analysis.diversity_concern else 'No'}")
                    print(f"   Assessment: {analysis.overall_assessment.upper()}")
                    
                    if analysis.recommendations:
                        print(f"\n📌 RECOMMENDATIONS ({len(analysis.recommendations)}):")
                        for i, rec in enumerate(analysis.recommendations, 1):
                            print(f"   {i}. {rec.parameter.upper()}")
                            print(f"      Current: {rec.current_value}")
                            print(f"      Suggested: {rec.suggested_value}")
                            print(f"      Confidence: {rec.confidence:.0%}")
                            print(f"      Reason: {rec.reasoning}")
                        
                        # Apply recommendations
                        if coach.auto_apply:
                            print(f"\n{'='*80}")
                            print("✅ APPLYING RECOMMENDATIONS")
                            print(f"{'='*80}")
                            
                            fitness_cfg, ga_cfg = coach.apply_recommendations(
                                analysis, fitness_cfg, ga_cfg
                            )
                            
                            print(f"✅ Applied {len(analysis.recommendations)} recommendations at Gen {gen}")
                            
                            # Reload model to clear context window (CRITICAL)
                            if coach.auto_reload_model:
                                print(f"\n{'='*80}")
                                print("🔄 RELOADING MODEL (clearing context window)")
                                print(f"{'='*80}")
                                await coach.reload_model()
                                print("✅ Model reloaded - ready for next analysis")
                            
                            print(f"   Next generation will use optimized configuration")
                    else:
                        print("\n✅ No recommendations - configuration looks good!")
                    
                    if analysis.next_steps:
                        print(f"\n🎯 NEXT STEPS:")
                        for step in analysis.next_steps:
                            print(f"   • {step}")
                    
                    print(f"{'='*80}\n")
            
            except Exception as e:
                print(f"❌ Error processing coach analysis: {e}")
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*80}")
    print("EVOLUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/n_generations:.2f}s/gen)")
    print(f"Final best fitness: {population.best_ever.fitness:.4f}")
    
    if coach:
        coach_stats = coach.get_stats()
        print(f"\n🤖 Coach Statistics:")
        print(f"   Total analyses: {coach_stats['total_analyses']}")
        print(f"   Total recommendations applied: {coach_stats['total_recommendations_applied']}")
        print(f"   Last analysis: Gen {coach_stats['last_analysis_generation']}")
    
    # Cleanup
    await dp.close()
    if coach and coach.coach_client:
        await coach.coach_client.unload_model()
        print("\n✅ Coach model unloaded")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GA evolution with coach")
    parser.add_argument("--from", dest="from_date", default="2024-12-01", help="Start date")
    parser.add_argument("--to", dest="to_date", default="2024-12-31", help="End date")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--population", type=int, default=56, help="Population size")
    parser.add_argument("--no-coach", action="store_true", help="Disable coach")
    parser.add_argument("--tf", default="15m", choices=["1m", "3m", "5m", "10m", "15m"], help="Timeframe")
    
    args = parser.parse_args()
    
    # Parse timeframe
    tf_map = {"1m": Timeframe.m1, "3m": Timeframe.m3, "5m": Timeframe.m5, "10m": Timeframe.m10, "15m": Timeframe.m15}
    timeframe = tf_map[args.tf]
    
    # Run evolution
    asyncio.run(run_evolution_with_coach(
        from_date=args.from_date,
        to_date=args.to_date,
        timeframe=timeframe,
        n_generations=args.generations,
        population_size=args.population,
        use_coach=not args.no_coach
    ))


if __name__ == "__main__":
    main()
