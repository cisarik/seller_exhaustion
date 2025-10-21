# Ultra-Detailed Tool Specifications for Evolution Coach Agent

## Philosophy: Agent as Evolution Director

The Evolution Coach Agent is not just an observer—it's the **director of evolution**. It has complete authority to:
- **Inspect** any aspect of the population or GA configuration
- **Modify** individuals, parameters, bounds, fitness function
- **Insert** strategically designed individuals
- **Drop** underperforming individuals
- **Steer** the search toward promising regions
- **Recover** from stagnation or convergence failures

---

## Tool Category 1: OBSERVABILITY (Deep Inspection)

### Tool 1.1: `analyze_population`

**Purpose**: Get comprehensive population statistics and identify patterns

**When to Use**:
- Start of every analysis session (understand current state)
- After making changes (verify impact)
- When diagnosing stagnation or poor performance

**Input Schema**:
```python
{
    "group_by": "fitness" | "trade_count" | "win_rate" | "avg_r",  # Sort criterion
    "top_n": int,          # Show top N individuals (default: 5)
    "bottom_n": int,       # Show bottom N individuals (default: 3)
    "include_params": bool # Include full parameters (default: false)
}
```

**Output Schema**:
```python
{
    "success": true,
    "population_size": 12,
    "generation": 25,
    
    # Fitness statistics
    "fitness": {
        "mean": 0.089,
        "std": 0.156,
        "min": 0.0,
        "max": 0.397,
        "median": 0.0,
        "quartiles": [0.0, 0.0, 0.168]
    },
    
    # Performance metrics
    "metrics": {
        "mean_trades": 4.6,
        "std_trades": 5.8,
        "mean_win_rate": 0.63,
        "mean_avg_r": 1.42,
        "mean_pnl": 0.084
    },
    
    # Gate compliance
    "gates": {
        "min_trades": 20,
        "min_win_rate": 0.40,
        "below_min_trades": 12,      # Count
        "below_min_trades_pct": 100.0,  # Percentage
        "below_min_wr": 0,
        "passing_all_gates": 0
    },
    
    # Diversity analysis
    "diversity": {
        "metric": 0.12,              # 0.0-1.0 (low = converged)
        "interpretation": "very_low", # very_low | low | moderate | high
        "parameter_variance": {
            "ema_fast": 8.2,
            "vol_z": 0.18,
            "tr_z": 0.09
        }
    },
    
    # Top performers
    "top_individuals": [
        {
            "id": 9,
            "fitness": 0.397,
            "metrics": {"n": 19, "win_rate": 0.842, "avg_r": 2.74, "pnl": 0.337},
            "key_params": {
                "ema_fast": 48,
                "ema_slow": 358,
                "vol_z": 1.21,
                "tr_z": 1.00
            },
            "full_params": {...}  # If include_params=true
        },
        ...
    ],
    
    # Bottom performers
    "bottom_individuals": [
        {
            "id": 0,
            "fitness": 0.0,
            "metrics": {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "pnl": 0.0},
            "reason": "No trades generated",
            "key_params": {
                "ema_fast": 49,
                "ema_slow": 371,
                "vol_z": 1.49,
                "tr_z": 1.01
            }
        },
        ...
    ],
    
    # Stagnation analysis
    "stagnation": {
        "best_fitness_flat_gens": 3,
        "stagnation_threshold": 5,
        "is_stagnant": false,
        "recent_improvement": 0.058
    }
}
```

**Decision Framework**:
- **diversity < 0.15**: Population converged too early → inject immigrants, increase mutation
- **below_min_trades_pct > 80%**: Gates too strict → lower min_trades or switch to soft_penalties
- **bottom performers have 0 trades**: Signal generation problem → check vol_z/tr_z thresholds
- **top performers cluster on specific params**: Promising region found → mutate explorers nearby
- **is_stagnant = true**: Evolution stuck → aggressive intervention needed

---

### Tool 1.2: `get_param_distribution`

**Purpose**: Analyze how a specific parameter is distributed across the population and its correlation with fitness

**When to Use**:
- Investigating which parameter values lead to success
- Identifying parameter clustering or boundary effects
- Before adjusting parameter bounds
- When designing new individuals

**Input Schema**:
```python
{
    "parameter_name": str,           # e.g., "ema_fast", "vol_z"
    "bins": int,                     # Histogram bins (default: 5)
    "correlate_with": str | null,    # "fitness", "trade_count", "win_rate", "avg_r", null
    "show_by_fitness_quartile": bool # Split by fitness quartile (default: true)
}
```

**Output Schema**:
```python
{
    "success": true,
    "parameter": "ema_fast",
    "type": "int",  # or "float", "discrete"
    
    # Distribution statistics
    "stats": {
        "min": 48,
        "max": 77,
        "mean": 53.2,
        "std": 7.8,
        "median": 50,
        "mode": 48
    },
    
    # Histogram
    "histogram": {
        "bins": ["48-54", "55-61", "62-68", "69-75", "76-82"],
        "counts": [7, 3, 1, 0, 1],
        "percentages": [58.3, 25.0, 8.3, 0.0, 8.3]
    },
    
    # Correlation analysis
    "correlation": {
        "with": "fitness",
        "pearson_r": 0.45,
        "interpretation": "moderate_positive",  # strong_positive | moderate_positive | weak | negative
        "p_value": 0.02,
        "significant": true
    },
    
    # By fitness quartile
    "by_quartile": {
        "top_25%": {
            "mean": 48.7,
            "std": 1.2,
            "range": [48, 50],
            "interpretation": "Top performers use fast EMA (48-50)"
        },
        "q2": {
            "mean": 52.3,
            "std": 3.1,
            "range": [49, 57]
        },
        "q3": {
            "mean": 60.1,
            "std": 8.4,
            "range": [52, 70]
        },
        "bottom_25%": {
            "mean": 62.3,
            "std": 9.7,
            "range": [49, 77],
            "interpretation": "Bottom performers use slower EMA (60+)"
        }
    },
    
    # Boundary effects
    "boundary_analysis": {
        "current_bounds": [48, 144],
        "at_min_bound": 7,     # 7 individuals at lower bound
        "at_max_bound": 0,     # 0 at upper bound
        "clustering_at_bounds": "min",
        "interpretation": "Strong clustering at minimum (48) - population hitting lower bound. Consider expanding downward to 24-48 range."
    },
    
    # Strategic recommendations
    "insights": [
        "Top performers (Q1) use ema_fast 48-50 (mean=48.7)",
        "Bottom performers (Q4) use ema_fast 60+ (mean=62.3)",
        "Positive correlation with fitness (r=0.45, p<0.05)",
        "58% of population at minimum bound (48)",
        "Suggest: Expand lower bound to explore 24-48 region"
    ]
}
```

**Decision Framework**:
- **Strong correlation (|r| > 0.5)**: Parameter is critical → focus mutations here
- **Top quartile mean ≠ population mean**: Success is in specific region → steer search there
- **Clustering at bounds**: Population wants to explore beyond → expand bounds
- **Weak correlation (|r| < 0.2)**: Parameter doesn't matter much → ignore for now
- **High std in top quartile**: Multiple strategies work → preserve diversity

---

### Tool 1.3: `get_param_bounds`

**Purpose**: Query current search space bounds and identify boundary clustering

**When to Use**:
- Before deciding to expand/shift bounds
- When designing new individuals
- Diagnosing why population can't find better solutions

**Input Schema**:
```python
{
    "parameters": list[str] | null,  # Specific params or null for all
    "include_clustering": bool       # Show boundary clustering (default: true)
}
```

**Output Schema**:
```python
{
    "success": true,
    "bounds": {
        "ema_fast": {
            "min": 48,
            "max": 144,
            "type": "int",
            "default": [48, 144],
            "overridden": false,
            "clustering": {
                "at_min": 7,         # 7 individuals at exactly 48
                "at_max": 0,         # 0 at exactly 144
                "near_min": 9,       # 9 within 10% of min (48-57)
                "near_max": 0,       # 0 within 10% of max (135-144)
                "in_middle": 3,      # 3 in middle region
                "interpretation": "Heavy clustering at minimum bound"
            }
        },
        "vol_z": {
            "min": 1.0,
            "max": 2.0,
            "type": "float",
            "default": [1.0, 2.0],
            "overridden": false,
            "clustering": {
                "at_min": 1,
                "at_max": 6,         # 6 individuals at exactly 2.0
                "near_min": 2,
                "near_max": 8,       # 8 within 10% of max (1.9-2.0)
                "in_middle": 2,
                "interpretation": "Heavy clustering at maximum bound"
            }
        },
        "fib_target_level": {
            "type": "discrete",
            "values": [0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
            "distribution": {
                "0.382": 8,          # 8 individuals use 0.382
                "0.618": 3,
                "0.5": 1
            },
            "interpretation": "Strong preference for 0.382 level"
        }
    },
    
    # Global boundary pressure analysis
    "global_analysis": {
        "parameters_at_bounds": ["ema_fast", "vol_z"],
        "pressure_score": 0.72,  # 0.0-1.0 (high = many at bounds)
        "interpretation": "High boundary pressure - population wants to explore beyond current bounds",
        "recommendations": [
            "Expand ema_fast lower bound (7/12 at minimum)",
            "Expand vol_z upper bound (6/12 at maximum)"
        ]
    }
}
```

**Decision Framework**:
- **at_min > 30% population**: Expand lower bound (successful param values are below minimum)
- **at_max > 30% population**: Expand upper bound (successful param values are above maximum)
- **near_min + near_max < 20%**: Good middle exploration, bounds OK
- **pressure_score > 0.6**: Global bounds too tight → expand multiple parameters
- **All in middle**: Bounds may be too wide → consider shifting to focus search

---

### Tool 1.4: `get_correlation_matrix`

**Purpose**: Analyze correlations between all parameters and fitness/metrics

**When to Use**:
- Understanding which parameters matter most
- Identifying parameter interactions
- Prioritizing which parameters to focus mutations on

**Input Schema**:
```python
{
    "include_params": list[str] | null,  # Specific params or null for all
    "correlate_with": list[str]          # e.g., ["fitness", "trade_count", "win_rate"]
}
```

**Output Schema**:
```python
{
    "success": true,
    "correlations": {
        "fitness": {
            "ema_fast": {"r": 0.45, "p": 0.02, "sig": true},
            "ema_slow": {"r": 0.12, "p": 0.34, "sig": false},
            "vol_z": {"r": 0.67, "p": 0.001, "sig": true},
            "tr_z": {"r": -0.23, "p": 0.18, "sig": false},
            ...
        },
        "trade_count": {
            "vol_z": {"r": -0.54, "p": 0.01, "sig": true},  # Higher vol_z = fewer trades
            ...
        }
    },
    
    # Ranked by importance
    "ranked_by_importance": [
        {"param": "vol_z", "r": 0.67, "importance": "critical"},
        {"param": "ema_fast", "r": 0.45, "importance": "high"},
        {"param": "fib_target_level", "r": 0.31, "importance": "moderate"},
        {"param": "ema_slow", "r": 0.12, "importance": "low"},
        ...
    ],
    
    # Strategic insights
    "insights": [
        "vol_z is CRITICAL (r=0.67 with fitness) - focus optimization here",
        "ema_fast has moderate positive correlation (r=0.45) - explore faster EMAs",
        "tr_z shows weak negative correlation - may be hurting performance",
        "vol_z negatively correlates with trade_count (r=-0.54) - tradeoff between selectivity and frequency"
    ]
}
```

**Decision Framework**:
- **|r| > 0.6**: Critical parameter → prioritize mutations here
- **0.4 < |r| < 0.6**: Important parameter → include in focused search
- **|r| < 0.2**: Low impact → can ignore or use default values
- **Negative correlation**: Higher values hurt performance → shift bounds downward

---

## Tool Category 2: INDIVIDUAL MANIPULATION (Direct Control)

### Tool 2.1: `mutate_individual`

**Purpose**: Directly modify a specific parameter of a specific individual

**When to Use**:
- Exploring nearby regions around successful individuals
- Testing hypotheses about parameter effects
- Creating directed exploration (not random mutation)
- Fine-tuning promising individuals

**Input Schema**:
```python
{
    "individual_id": int,     # 0-indexed individual ID
    "parameter_name": str,    # e.g., "ema_fast", "vol_z"
    "new_value": any,         # New value (must respect type and bounds)
    "reason": str,            # Explanation for mutation
    "respect_bounds": bool    # Enforce bounds (default: true)
}
```

**Output Schema**:
```python
{
    "success": true,
    "individual_id": 9,
    "parameter": "ema_fast",
    "old_value": 48,
    "new_value": 60,
    "change": "+12",
    "change_pct": "+25%",
    
    # Individual context
    "individual_before": {
        "fitness": 0.397,
        "metrics": {"n": 19, "win_rate": 0.842},
        "generation": 9
    },
    
    # Mutation validation
    "validation": {
        "within_bounds": true,
        "bounds": [48, 144],
        "type_correct": true
    },
    
    # Impact tracking
    "impact": {
        "fitness_reset": true,     # Individual will be re-evaluated
        "evaluation_pending": true,
        "will_compete_in_next_gen": true
    },
    
    "message": "Mutated Individual #9: ema_fast 48 → 60 (Explore slightly slower EMA for trend detection)"
}
```

**Strategic Use Cases**:
1. **Exploit Best Individual**: Take top performer, mutate one param to explore nearby
   ```python
   # Best individual has ema_fast=48, vol_z=1.21
   mutate_individual(id=9, parameter="vol_z", new_value=1.5, 
                     reason="Test if higher selectivity improves top performer")
   ```

2. **Test Hypothesis**: Verify if specific parameter change helps
   ```python
   # Hypothesis: Faster EMA improves signal generation
   mutate_individual(id=5, parameter="ema_fast", new_value=36,
                     reason="Test hypothesis: faster EMA (36 vs 48) generates more trades")
   ```

3. **Repair Bad Individual**: Fix obvious mistakes
   ```python
   # Individual has ema_fast=120 (too slow), fitness=0
   mutate_individual(id=3, parameter="ema_fast", new_value=48,
                     reason="Repair: EMA too slow, no signals. Reset to successful range")
   ```

4. **Create Gradient**: Test multiple values along gradient
   ```python
   # Create individuals testing vol_z 1.0, 1.5, 2.0, 2.5
   mutate_individual(id=7, parameter="vol_z", new_value=1.5, reason="Gradient test point 1")
   mutate_individual(id=8, parameter="vol_z", new_value=2.5, reason="Gradient test point 2")
   ```

**Decision Framework**:
- **Mutate top performer**: Explore local neighborhood of best solution
- **Mutate similar to top**: Steer mediocre individuals toward success patterns
- **Mutate extremes**: Test boundary cases (very fast EMA, very high vol_z)
- **Mutate multiple params**: Change 2-3 correlated params together
- **Don't mutate**: Elite individuals if elite_fraction preserves them

---

### Tool 2.2: `drop_individual`

**Purpose**: Remove underperforming or redundant individuals from population

**When to Use**:
- Individual has zero fitness for multiple generations
- Duplicate or near-duplicate individuals
- Population size too large
- Making room for coach-designed individuals

**Input Schema**:
```python
{
    "individual_id": int,
    "reason": str
}
```

**Output Schema**:
```python
{
    "success": true,
    "dropped_individual": {
        "id": 3,
        "fitness": 0.0,
        "metrics": {"n": 0, "win_rate": 0.0},
        "generation": 8,
        "key_params": {"ema_fast": 70, "vol_z": 1.8}
    },
    "population_size_before": 12,
    "population_size_after": 11,
    "impact": "Removed zero-fitness individual. Population will regenerate or receive immigrants.",
    "message": "Dropped Individual #3: Zero trades for 10 generations (ema_fast=70, vol_z=1.8 too restrictive)"
}
```

**Strategic Use Cases**:
1. **Remove Dead Weight**: Zero fitness for many generations
   ```python
   drop_individual(id=3, reason="Zero trades for 10 generations, no signal generation")
   ```

2. **Remove Duplicates**: Very similar parameter sets
   ```python
   drop_individual(id=7, reason="Near-duplicate of individual #9 (99% similarity)")
   ```

3. **Make Room**: Create space for coach-designed explorers
   ```python
   drop_individual(id=11, reason="Make room for coach-designed high-vol_z explorer")
   ```

**Decision Framework**:
- **fitness = 0 for > 3 gens**: Drop (can't generate signals)
- **Duplicate of better individual**: Drop the worse one
- **Population > target size**: Drop worst performers
- **Never drop**: Top 25% performers (elite)

---

### Tool 2.3: `insert_individual`

**Purpose**: Add new strategically designed individual to population

**When to Use**:
- Testing unexplored parameter combinations
- Injecting diversity
- Exploring regions outside current bounds
- Implementing coach hypotheses

**Input Schema**:
```python
{
    "strategy": "coach_designed" | "random" | "clone_best" | "hybrid",
    
    # For coach_designed
    "parameters": dict | null,  # Full parameter set
    
    # For clone_best
    "clone_from_id": int | null,
    "mutations": dict | null,    # Params to change from clone
    
    # For hybrid
    "parent_ids": list[int],     # Parents to blend
    "blend_strategy": "average" | "best_of_each",
    
    "reason": str,
    "position": int | null       # Where to insert (null = append)
}
```

**Output Schema**:
```python
{
    "success": true,
    "new_individual_id": 12,
    "position": 12,
    "strategy": "coach_designed",
    "parameters": {
        "ema_fast": 36,
        "ema_slow": 480,
        "vol_z": 2.5,
        ...
    },
    "reasoning": "Explore high-selectivity region (vol_z=2.5) beyond current bound (2.0)",
    "population_size_after": 13,
    "impact": {
        "fitness": 0.0,           # Not yet evaluated
        "evaluation_pending": true,
        "will_compete_next_gen": true
    }
}
```

**Strategic Use Cases**:

1. **Test Unexplored Region**:
   ```python
   insert_individual(
       strategy="coach_designed",
       parameters={
           "ema_fast": 36,        # Below current bound
           "ema_slow": 480,
           "vol_z": 2.5,          # Above current bound
           "tr_z": 1.0,
           "cloc_min": 0.5,
           # ... complete param set
       },
       reason="Test high-selectivity strategy beyond current bounds (vol_z=2.5)"
   )
   ```

2. **Clone and Mutate Best**:
   ```python
   insert_individual(
       strategy="clone_best",
       clone_from_id=9,           # Best individual
       mutations={
           "vol_z": 1.5,          # Test higher selectivity
           "fib_target_level": 0.618  # Different exit
       },
       reason="Clone best (#9) with higher vol_z to test selectivity hypothesis"
   )
   ```

3. **Hybrid Strategy**:
   ```python
   insert_individual(
       strategy="hybrid",
       parent_ids=[9, 10],        # Top 2 individuals
       blend_strategy="best_of_each",
       reason="Combine best parameters from top 2 performers"
   )
   ```

4. **Random with Constraints**:
   ```python
   insert_individual(
       strategy="coach_designed",
       parameters={
           "ema_fast": random(24, 48),    # Force fast EMA
           "vol_z": random(2.0, 3.0),     # Force high vol_z
           # ... other params random
       },
       reason="Random individual constrained to fast-EMA + high-vol_z region"
   )
   ```

**Decision Framework**:
- **Bounds too tight**: Insert individual beyond bounds to test
- **Unexplored region**: Insert individual in undersampled area
- **Hypothesis test**: Insert individual with specific param combination
- **Diversity boost**: Insert random individuals
- **Exploit success**: Clone and mutate best performers

---

## Tool Category 3: GA ALGORITHM STEERING

### Tool 3.1: `update_ga_params`

**Purpose**: Adjust genetic algorithm evolution mechanics

**When to Use**:
- Population converged too early → increase mutation, immigrants
- Evolution too chaotic → decrease mutation, increase elitism
- Stagnation → inject diversity
- Good progress → preserve current strategy

**Input Schema**:
```python
{
    # Mutation control
    "mutation_probability": float | null,  # 0.0-1.0 (chance each individual mutates)
    "mutation_rate": float | null,        # 0.0-1.0 (how much parameters change)
    "sigma": float | null,                # Gaussian mutation std dev
    
    # Selection control
    "tournament_size": int | null,        # 2-8 (higher = stronger selection pressure)
    "elite_fraction": float | null,       # 0.0-0.4 (top % preserved unchanged)
    
    # Diversity control
    "immigrant_fraction": float | null,   # 0.0-0.3 (random injection rate)
    "immigrant_strategy": str | null,     # "worst_replacement" | "random_replacement"
    
    "reason": str
}
```

**Output Schema**:
```python
{
    "success": true,
    "changes_applied": {
        "mutation_rate": {
            "old": 0.55,
            "new": 0.30,
            "change": "-0.25",
            "interpretation": "Reduced mutation by 45% - will make smaller parameter changes"
        },
        "immigrant_fraction": {
            "old": 0.0,
            "new": 0.1,
            "change": "+0.1",
            "interpretation": "Injecting 10% random individuals per generation for diversity"
        }
    },
    "unchanged": ["elite_fraction", "tournament_size", "sigma"],
    
    # Impact prediction
    "predicted_impact": {
        "exploration_vs_exploitation": "More exploitation (lower mutation)",
        "diversity_change": "Increased (immigrants)",
        "convergence_speed": "Slower (less aggressive search)",
        "recommendation": "Good for preserving current good solutions while exploring with immigrants"
    },
    
    "effective_next_gen": 26
}
```

**Strategic Patterns**:

1. **Convergence Too Early** (diversity < 0.15):
   ```python
   update_ga_params(
       mutation_rate=0.6,           # Increase from 0.55
       mutation_probability=0.9,     # Keep high
       immigrant_fraction=0.2,       # Increase from 0.0
       reason="Low diversity (0.12) - inject variation"
   )
   ```

2. **Stagnation** (no improvement for 10 gens):
   ```python
   update_ga_params(
       mutation_rate=0.7,           # Aggressive
       immigrant_fraction=0.25,     # High
       elite_fraction=0.15,         # Reduce from 0.25
       reason="Stagnation for 10 generations - aggressive exploration"
   )
   ```

3. **Good Progress** (fitness improving steadily):
   ```python
   update_ga_params(
       mutation_rate=0.3,           # Reduce from 0.55
       elite_fraction=0.3,          # Increase from 0.25
       immigrant_fraction=0.05,     # Small diversity injection
       reason="Good progress - preserve good solutions, fine-tune"
   )
   ```

4. **Initial Exploration** (early generations):
   ```python
   update_ga_params(
       mutation_rate=0.7,           # High
       mutation_probability=0.95,   # Very high
       tournament_size=3,           # Lower pressure
       reason="Early exploration - cast wide net"
   )
   ```

5. **Final Refinement** (late generations):
   ```python
   update_ga_params(
       mutation_rate=0.2,           # Low
       elite_fraction=0.35,         # High preservation
       tournament_size=5,           # Higher pressure
       reason="Late-stage refinement - fine-tune best solutions"
   )
   ```

**Decision Framework**:
- **diversity < 0.15**: Increase mutation_rate + immigrant_fraction
- **stagnant**: Aggressive mutation (0.6-0.7) + immigrants (0.2-0.3)
- **best_fitness improving**: Reduce mutation (0.2-0.3), increase elitism
- **early gens (< 20)**: High mutation (0.6-0.7), low elitism (0.1-0.15)
- **late gens (> 80)**: Low mutation (0.2-0.3), high elitism (0.3-0.4)

---

### Tool 3.2: `update_param_bounds`

**Purpose**: Expand, contract, or shift parameter search space

**When to Use**:
- Population clustering at bounds (can't explore beyond)
- Correlation analysis suggests better values outside bounds
- Want to focus search on specific region
- Testing hypothesis about parameter ranges

**Input Schema**:
```python
{
    "parameter": str,
    "new_min": any | null,   # Update minimum (null = no change)
    "new_max": any | null,   # Update maximum (null = no change)
    "reason": str,
    "retroactive": bool      # Apply to existing individuals? (default: false)
}
```

**Output Schema**:
```python
{
    "success": true,
    "parameter": "ema_fast",
    "old_bounds": {"min": 48, "max": 144},
    "new_bounds": {"min": 24, "max": 144},
    "change": {
        "min": "-24 (-50%)",
        "max": "unchanged",
        "width": "96 → 120 (+25%)"
    },
    
    # Impact on population
    "population_impact": {
        "individuals_at_old_min": 7,   # Were at 48
        "individuals_now_in_bounds": 7, # Now at new min (24)
        "newly_explorable_region": "[24, 48)",
        "interpretation": "7 individuals were stuck at old minimum. New region [24, 48) now explorable."
    },
    
    # Operational impact
    "operational_impact": {
        "random_individuals": "Will use new bounds [24, 144]",
        "mutations": "Will respect new bounds [24, 144]",
        "existing_individuals": "Unchanged (retroactive=false)"
    },
    
    "recommended_followup": [
        "Insert coach-designed individual with ema_fast=36 to test new region",
        "Increase immigrant_fraction to 0.1 to populate new space"
    ]
}
```

**Strategic Patterns**:

1. **Expand Beyond Clustering**:
   ```python
   # 7/12 at ema_fast=48 (minimum)
   update_param_bounds(
       parameter="ema_fast",
       new_min=24,              # Expand downward
       new_max=144,             # Keep
       reason="7/12 individuals at minimum (48) - expand to test faster EMAs"
   )
   ```

2. **Focus on Successful Region**:
   ```python
   # Top performers use vol_z 1.8-2.0
   update_param_bounds(
       parameter="vol_z",
       new_min=1.5,             # Narrow from 1.0
       new_max=2.5,             # Expand from 2.0
       reason="Top performers cluster at 1.8-2.0. Focus search on high-selectivity region."
   )
   ```

3. **Test Hypothesis**:
   ```python
   # Hypothesis: Very fast EMA works better
   update_param_bounds(
       parameter="ema_fast",
       new_min=12,              # Very fast
       new_max=96,              # Shorter range
       reason="Test hypothesis: ultra-fast EMAs (12-48) capture micro-trends"
   )
   ```

4. **Shift Search Window**:
   ```python
   # Shift from [336, 1008] to [576, 1248]
   update_param_bounds(
       parameter="ema_slow",
       new_min=576,
       new_max=1248,
       reason="Population prefers upper range. Shift window to explore slower EMAs."
   )
   ```

**Decision Framework**:
- **> 30% at bound**: Expand that bound
- **Correlation shows optimum outside**: Expand bounds
- **Low diversity**: Expand bounds for more exploration
- **Top performers in narrow range**: Contract bounds to focus
- **Hypothesis testing**: Custom bounds for specific test

---

[Continue with remaining 15 tools in similar ultra-detailed format...]

Would you like me to continue with the remaining tools (fitness function, curriculum, exit strategies, etc.) in the same level of detail?
