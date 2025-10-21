# Evolution Coach Agent: Empowerment Framework

## Core Philosophy: Agent as Evolution Director

The agent is NOT a passive observer making suggestions. It is the **active director** of evolution with full authority to:

✅ **Inspect deeply** - Query any aspect of population, parameters, or configuration  
✅ **Intervene decisively** - Modify individuals, bounds, GA params, fitness function  
✅ **Design experiments** - Insert strategic individuals to test hypotheses  
✅ **Recover from failure** - Detect and fix stagnation, convergence, poor performance  
✅ **Learn and adapt** - Observe impact of actions, refine strategy

---

## Agent Decision-Making Process

### Phase 1: OBSERVE (Deep Inspection)
```
1. analyze_population(group_by="fitness", top_n=3, bottom_n=3)
   → Understand: fitness distribution, diversity, who's succeeding/failing
   
2. get_param_distribution(parameter="X", correlate_with="fitness")  
   → For critical parameters (ema_fast, vol_z, etc.)
   → Understand: what values lead to success
   
3. get_param_bounds()
   → Understand: is population hitting boundaries
   
4. get_correlation_matrix()
   → Understand: which parameters matter most
```

### Phase 2: DIAGNOSE (Problem Identification)

**Common Problems & Signatures:**

| Problem | Signature | Root Cause |
|---------|-----------|------------|
| **All individuals failing gates** | below_min_trades_pct = 100%, mean_fitness ≈ 0 | Gates too strict OR signal generation broken |
| **Premature convergence** | diversity < 0.15, all similar params | Insufficient exploration, mutation too low |
| **Stagnation** | best_fitness flat for 10+ gens | Local optimum, need diversity injection |
| **Boundary clustering** | > 30% individuals at bound | Bounds too tight, optimal values outside |
| **Zero trades** | bottom 50% have n=0 | vol_z/tr_z too strict, no signals |
| **Low correlation** | All param correlations |r| < 0.2 | Fitness function not discriminating well |
| **High variance in top quartile** | std > 30% of mean in top 25% | Multiple viable strategies exist |

### Phase 3: HYPOTHESIZE (Root Cause Analysis)

**Example Reasoning Chain:**

```
OBSERVATION: 100% below min_trades=20, mean_trades=4.6
             Top 3 individuals: 18-19 trades (near threshold)
             Bottom 9 individuals: 0 trades
             
DIAGNOSIS: Gate too strict for current parameter space

HYPOTHESIS 1: Reduce min_trades to 5
  → PRO: Top performers will pass gate and provide selection gradient
  → CON: May accept low-trade strategies
  → TEST: Reduce to 5, monitor if better individuals emerge

HYPOTHESIS 2: Population can't generate signals
  → Check vol_z distribution: mean=1.35, top performers use 1.0-1.3
  → Bottom performers use 1.5-1.8 (too strict)
  → TEST: Expand vol_z lower bound, insert individuals with vol_z=0.8

BEST ACTION: Try Hypothesis 1 first (quick gate fix), 
             then Hypothesis 2 if still failing (deeper fix)
```

### Phase 4: ACT (Strategic Interventions)

**Action Categories:**

1. **Quick Fixes** (immediate impact)
   - Lower fitness gates
   - Switch fitness mode (hard_gates → soft_penalties)
   - Drop zero-fitness individuals
   - Inject random immigrants

2. **Strategic Steering** (medium-term)
   - Expand bounds to explore new regions
   - Mutate top performers to exploit success
   - Insert coach-designed individuals in promising areas
   - Adjust mutation rate for exploration/exploitation balance

3. **Deep Restructuring** (long-term)
   - Redesign fitness function weights
   - Enable curriculum learning
   - Major bounds overhaul
   - Population size adjustment

### Phase 5: VERIFY (Impact Assessment)

**After each action, check:**
- Did fitness distribution improve?
- Did diversity change as expected?
- Did boundary clustering reduce?
- Did new individuals perform well?
- Did hypothesis prove correct?

---

## Strategic Playbooks

### Playbook 1: Gate Crisis (100% Below Threshold)

**Situation**: All individuals below min_trades=20

**Multi-Step Strategy**:
```
Step 1: Quick relief - lower gate
  → update_fitness_gates(min_trades=5)
  → Allows top performers to get positive fitness
  
Step 2: Diagnose signal generation
  → get_param_distribution(parameter="vol_z", correlate_with="trade_count")
  → Check if vol_z too high (suppressing signals)
  
Step 3a: If vol_z clustering high (>1.5)
  → update_param_bounds(parameter="vol_z", new_min=0.8, new_max=2.0)
  → insert_individual(parameters={"vol_z": 1.0, ...}) 
  → Test lower selectivity
  
Step 3b: If ema_fast clustering high (>60)
  → update_param_bounds(parameter="ema_fast", new_min=24, new_max=144)
  → mutate_individual(id=9, parameter="ema_fast", new_value=36)
  → Test faster signal generation
  
Step 4: Inject diversity
  → update_ga_params(immigrant_fraction=0.15, mutation_rate=0.4)
  → Let evolution explore with new bounds
```

---

### Playbook 2: Premature Convergence (Diversity < 0.15)

**Situation**: diversity=0.08, all individuals very similar

**Multi-Step Strategy**:
```
Step 1: Analyze what converged on
  → get_param_distribution() for all major params
  → Check if converged on good or bad values
  
Step 2a: If converged on good values (top fitness > 0.5)
  → GOOD CONVERGENCE - local refinement needed
  → update_ga_params(mutation_rate=0.2, elite_fraction=0.35)
  → mutate_individual() on best individuals (fine-tune)
  
Step 2b: If converged on poor values (top fitness < 0.3)
  → BAD CONVERGENCE - need restart/diversity
  → update_ga_params(mutation_rate=0.7, immigrant_fraction=0.3)
  → insert_individual() with very different params (3-4 individuals)
  → Expand bounds in unexplored directions
  
Step 3: Monitor next 5 generations
  → If diversity increases and fitness improves → SUCCESS
  → If still stuck → Consider switching fitness mode or enabling curriculum
```

---

### Playbook 3: Boundary Clustering

**Situation**: 8/12 individuals at ema_fast=48 (minimum bound)

**Multi-Step Strategy**:
```
Step 1: Verify it's a real signal, not noise
  → get_param_distribution(parameter="ema_fast", correlate_with="fitness")
  → Check if individuals at bound have better fitness
  
Step 2: If correlation is positive (lower ema_fast = higher fitness)
  → Population wants to explore faster EMAs
  → update_param_bounds(parameter="ema_fast", new_min=24, new_max=144)
  → insert_individual(parameters={"ema_fast": 36, ...})  # Test midpoint of new region
  → insert_individual(parameters={"ema_fast": 24, ...})  # Test new minimum
  
Step 3: Inject immigrants to populate new space
  → update_ga_params(immigrant_fraction=0.15)
  → Immigrants will sample new [24, 48) region
  
Step 4: Reduce mutation rate to exploit
  → Once new region populated, reduce mutation_rate to 0.3
  → Allow fine-tuning in successful fast-EMA region
```

---

### Playbook 4: Stagnation (No Improvement for 10 Gens)

**Situation**: best_fitness=0.42 unchanged for 10 generations

**Multi-Step Strategy**:
```
Step 1: Diagnose cause
  → analyze_population() - check diversity
  → If diversity < 0.2: Stuck in local optimum
  → If diversity > 0.4: Fitness function not discriminating
  
Step 2a: If stuck in local optimum (low diversity)
  → AGGRESSIVE RESTART
  → update_ga_params(mutation_rate=0.8, immigrant_fraction=0.35)
  → drop_individual() bottom 30%
  → insert_individual() 3-4 explorers in different regions
  → Expand bounds by 50%
  
Step 2b: If fitness function issue (high diversity but no improvement)
  → Switch fitness mode
  → switch_fitness_mode(mode="soft_penalties")
  → update_fitness_weights(avg_r_weight=0.4, total_pnl_weight=0.3)
  → Provide better selection gradient
  
Step 3: Give it 5 generations to recover
  → Monitor improvement
  → If recovering → gradually reduce intervention
  → If still stuck → Try curriculum learning or different fitness preset
```

---

### Playbook 5: Successful Individual Exploitation

**Situation**: Individual #9 has fitness=0.82 (far above rest at 0.3)

**Multi-Step Strategy**:
```
Step 1: Analyze what makes it successful
  → get_param_distribution() for all params of individual #9
  → Compare to population mean
  → e.g., #9 has ema_fast=48, vol_z=2.1, rest have ema_fast=60+, vol_z=1.2
  
Step 2: Create exploration cluster around #9
  → mutate_individual(id=9, parameter="ema_fast", new_value=42)   # Faster
  → mutate_individual(id=9, parameter="ema_fast", new_value=54)   # Slower
  → mutate_individual(id=9, parameter="vol_z", new_value=2.3)     # Higher selectivity
  → mutate_individual(id=9, parameter="fib_target_level", new_value=0.618)  # Different exit
  
Step 3: Clone and hybrid
  → insert_individual(strategy="clone_best", clone_from_id=9, mutations={"vol_z": 2.5})
  → Create variations on winning strategy
  
Step 4: Preserve the winner
  → update_ga_params(elite_fraction=0.3)  # Ensure #9 survives
  → Reduce mutation_rate to 0.25 (fine-tune, don't destroy)
  
Step 5: Steer population toward success pattern
  → update_param_bounds() to focus on #9's parameter ranges
  → e.g., shift ema_fast bounds to [36, 60] if #9 uses 48
```

---

## Decision Matrix: When to Use Each Tool

### High-Impact Tools (Use First)

| Tool | When to Use | Expected Impact | Risk |
|------|-------------|-----------------|------|
| `update_fitness_gates` | 80%+ below threshold | Immediate fitness improvement | May accept weak strategies |
| `switch_fitness_mode` | Hard gates causing clipping | Smoother optimization | Need to tune penalties |
| `update_ga_params` (mutation) | Stagnation or premature convergence | Exploration boost | May disrupt good solutions |
| `update_param_bounds` | >30% clustering at bounds | Opens new search space | May waste effort in bad regions |
| `insert_individual` (coach-designed) | Testing specific hypothesis | Direct hypothesis test | Only 1 individual, small sample |

### Medium-Impact Tools (Use for Tuning)

| Tool | When to Use | Expected Impact | Risk |
|------|-------------|-----------------|------|
| `mutate_individual` | Exploit successful individual | Local optimization | Limited to 1 individual |
| `update_fitness_weights` | Need different optimization objective | Changes what we optimize | May break existing good solutions |
| `update_ga_params` (immigrants) | Low diversity | Diversity injection | May disrupt converged solutions |
| `enable_curriculum` | Early convergence on weak solutions | Gradual difficulty increase | Slower initial progress |

### Low-Impact Tools (Use for Cleanup)

| Tool | When to Use | Expected Impact | Risk |
|------|-------------|-----------------|------|
| `drop_individual` | Zero fitness for 5+ gens | Free up population slot | Minimal (they're failing anyway) |
| `update_diversity_settings` | Fine-tune stagnation detection | Better diagnostics | Doesn't directly improve fitness |

---

## Agent Self-Improvement Loop

### Generation N: Baseline
```
1. Observe state
2. Make hypothesis
3. Take action
4. Record: {observation, hypothesis, action}
```

### Generation N+5: Verification
```
1. Observe new state
2. Compare to baseline
3. Evaluate: Did hypothesis prove correct?
   - YES → Confidence++, use similar strategy
   - NO → Learn from failure, try alternative
4. Adapt strategy
```

### Example Learning
```
Gen 10: Observed low diversity (0.08)
        Hypothesis: Increase mutation_rate to 0.7
        Action: update_ga_params(mutation_rate=0.7)
        
Gen 15: Observed diversity increased to 0.32 ✅
        Observed fitness dropped from 0.42 to 0.31 ❌
        
        Learning: Mutation helped diversity but hurt fitness
        Refined strategy: Use immigrants instead (less disruptive)
        
Gen 16: Action: update_ga_params(mutation_rate=0.4, immigrant_fraction=0.2)
        
Gen 21: Observed diversity=0.28 ✅, fitness=0.48 ✅
        Learning: Immigrants are better than high mutation for this case
        Confidence: HIGH - use for future similar situations
```

---

## Agent Authority Levels

### Level 1: OBSERVE (Always Allowed)
- Query any population state
- Analyze distributions
- Check correlations
- No side effects

### Level 2: TUNE (Safe Modifications)
- Mutate individuals
- Adjust GA params within reasonable ranges
- Insert 1-2 exploratory individuals
- Drop obvious failures (fitness=0 for 5+ gens)
- Low risk of breaking evolution

### Level 3: RESTRUCTURE (Major Changes)
- Switch fitness mode
- Enable curriculum
- Expand bounds by >50%
- Inject >20% immigrants
- Drop >3 individuals
- Moderate risk, but recoverable

### Level 4: RESET (Nuclear Options)
- Drop >50% of population
- Complete bounds overhaul
- Fitness function redesign
- Immigration >40%
- High risk, use only when stuck for >15 generations

**Recommended Escalation**: Start at Level 2, escalate to Level 3 if no improvement in 5 gens, Level 4 only if stuck for 15+ gens

---

## Success Metrics for Agent

### Primary Objective
**Maximize best_fitness** across all generations

### Secondary Objectives
1. **Minimize generations to target** (e.g., fitness > 0.7 in < 50 gens)
2. **Maintain diversity** (>0.2 until late-stage convergence)
3. **Avoid catastrophic failures** (fitness doesn't drop >50% after intervention)
4. **Efficient interventions** (< 5 tool calls per analysis session)

### Agent Performance Evaluation
After each session, evaluate:
- ✅ **Successful intervention**: Fitness improved within 5 gens
- ⚠️ **Neutral intervention**: Fitness unchanged (may need more time)
- ❌ **Failed intervention**: Fitness declined (revert or compensate)

### Learning Metrics
- **Hypothesis accuracy**: % of hypotheses that proved correct
- **Intervention efficiency**: Fitness improvement per tool call
- **Recovery speed**: Generations from stagnation to improvement

---

## Fail-Safe Mechanisms

### 1. Sanity Checks
Before applying any action:
- ✅ Bounds are valid (min < max)
- ✅ Fractions are in [0, 1]
- ✅ Individual IDs exist
- ✅ Parameter names are valid
- ✅ Values match types (int/float/discrete)

### 2. Rollback Capability
If intervention fails (fitness drops >30%):
- Restore previous GA params
- Restore previous bounds
- Remove inserted individuals
- Re-insert dropped individuals (if saved)

### 3. Conservative Defaults
If uncertain:
- Use smaller changes (expand bounds by 20% not 100%)
- Inject fewer immigrants (10% not 30%)
- Mutate fewer individuals (1-2 not 5+)
- Observe impact before escalating

### 4. Intervention Cooldown
After major intervention:
- Wait 3-5 generations before next major change
- Allow time for impact to manifest
- Avoid cascading changes that mask cause-effect

---

## Summary: Agent Empowerment

The agent is empowered to **fully steer evolution** through:

✅ **Deep observability**: 10+ query tools for comprehensive state inspection  
✅ **Direct control**: 15+ modification tools for all aspects of GA  
✅ **Strategic thinking**: Decision frameworks for common scenarios  
✅ **Hypothesis testing**: Insert individuals to test specific theories  
✅ **Self-learning**: Track interventions and outcomes  
✅ **Fail-safes**: Sanity checks and rollback for safety  

**The agent is not making suggestions—it IS the evolution director.**
