# Phase 6: Evolutionary Engine

## Objective

Implement the core evolutionary loop with MAS (Minimum Acceptable Score) threshold. This is the heart of the ProFiT system where strategies are mutated, evaluated, and selected.

From the design document:

> Orchestrates the evolutionary loop (selection, mutation via LLMs, code compilation/testing, and survivor selection) as described in Algorithm 1 of the paper.

---

## Algorithm Overview

```
Algorithm 1: ProFiT Evolutionary Loop
─────────────────────────────────────
1. Initialize population with seed strategy S0
2. Compute baseline performance P0 on validation
3. Set MAS (Minimum Acceptable Score) = P0
4. FOR generation = 1 to N:
   5. Select parent strategy (S_t, P_t) from population
   6. LLM A: Generate improvement proposal Δ
   7. LLM B: Generate modified code Ŝ_t
   8. FOR attempt = 1 to 10:
      9. Try to compile and run Ŝ_t
      10. IF success: BREAK
      11. ELSE: LLM B fixes code with error traceback
   12. IF success:
      13. Evaluate fitness P(Ŝ_t) on validation
      14. IF P(Ŝ_t) >= MAS:
         15. Add (Ŝ_t, P(Ŝ_t)) to population
16. Return best strategy from population
```

---

## Method: evolve_strategy()

### Signature

```python
def evolve_strategy(self, strategy_class, train_data: pd.DataFrame,
                    val_data: pd.DataFrame, max_iters=15):
    """
    Evolves the given strategy on the train/validation data using LLM mutations.
    Returns the best evolved strategy class and its performance on validation.
    """
```

### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_class` | class | Seed strategy to evolve |
| `train_data` | DataFrame | Training data (for context) |
| `val_data` | DataFrame | Validation data (for fitness) |
| `max_iters` | int | Max generations (default 15) |

### Returns
- `best_strategy_class`: The evolved strategy class with highest fitness
- `best_perf`: Performance (annualized return) on validation

---

## Step 1-3: Initialization

```python
# 1. Compute baseline performance P0 on validation set
_, base_result = self.run_backtest(strategy_class, val_data)
P0 = base_result['Return (Ann.) [%]']
print(f"Initial strategy {strategy_class.__name__} baseline annualized return on validation: {P0:.2f}%")

# 2. Set MAS = P0
MAS = P0  # minimum acceptable score (annual return)

# Archive of viable strategies (as tuples of class and performance)
population = [(strategy_class, P0)]
best_perf = P0
best_strategy_class = strategy_class
```

### MAS Threshold

> A Minimum Acceptable Score (MAS) threshold (initially set to the seed strategy's performance) governs whether new strategies are kept.

- New strategies must meet or exceed MAS to be accepted
- This ensures evolutionary progress (no regression)

---

## Step 4-5: Selection

```python
for gen in range(1, max_iters+1):
    print(f"\nGeneration {gen}: Current population size = {len(population)}. Selecting a strategy to mutate...")

    # 5. Select a strategy from population (random selection for diversity)
    parent_class, parent_perf = population[self._random_index(len(population))]
    print(f"Selected parent strategy '{parent_class.__name__}' with validation return {parent_perf:.2f}% for mutation.")
```

### Selection Strategy

> We use a simple random selection among viable strategies to maintain diversity (the paper keeps an archive of all above-threshold strategies to avoid premature convergence).

```python
def _random_index(self, n):
    import random
    return random.randrange(n)
```

---

## Step 6-7: LLM Mutation

```python
# Get source code of parent strategy
import inspect
parent_code = inspect.getsource(parent_class)

# 6. Prompt LLM A for improvement proposal
improvement = self.llm.generate_improvement(
    parent_code,
    f"AnnReturn={parent_perf:.2f}%, Sharpe={parent_perf:.2f}"
)
print(f"LLM suggested improvement: {improvement}")

# 7. Prompt LLM B to synthesize modified strategy code
new_code = self.llm.generate_strategy_code(parent_code, improvement)

# Give the new strategy a unique name by generation
new_class_name = f"{parent_class.__name__}_Gen{gen}"

# Replace class name in code
if new_code.startswith("class"):
    new_code = new_code.replace(parent_class.__name__, new_class_name, 1)
```

### Class Naming
- Each mutation gets a unique name: `StrategyName_Gen{N}`
- Prevents namespace collisions
- Enables tracking lineage

---

## Step 8-11: Compilation and Repair Loop

```python
success = False
for attempt in range(1, 11):  # up to 10 repair attempts
    try:
        # Dynamically define the new strategy class from code
        namespace = {}
        exec(new_code, globals(), namespace)
        NewStrategyClass = namespace[new_class_name]

        # Run backtest on validation data to get performance
        metrics, res = self.run_backtest(NewStrategyClass, val_data)
        success = True
        break

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"Attempt {attempt}: Strategy code failed with error: {error_msg}")

        if attempt < 10:
            # 11. Prompt LLM B with traceback to fix code
            new_code = self.llm.fix_code(new_code, tb)

            # Ensure the class name persists in the corrected code
            if new_class_name not in new_code:
                new_code = new_code.replace(parent_class.__name__, new_class_name, 1)
        else:
            print("Max repair attempts reached. Discarding this mutation.")
            success = False
```

### Dynamic Code Execution

```python
namespace = {}
exec(new_code, globals(), namespace)
NewStrategyClass = namespace[new_class_name]
```

- `globals()` provides access to imports (Strategy, pd, np, etc.)
- `namespace` captures the new class definition
- Class is retrieved by its unique name

---

## Step 12-15: Fitness Evaluation and Selection

```python
if not success:
    continue  # Move to next generation

# 14. Compute fitness of new strategy
P_new = res['Return (Ann.) [%]']
print(f"New strategy variant '{new_class_name}' achieved validation annual return {P_new:.2f}%")

# 15-18. Check against MAS threshold
if P_new is not None and P_new >= MAS:
    # Accept new strategy into population
    population.append((NewStrategyClass, P_new))
    print(f"Accepted new strategy (>= MAS={MAS:.2f}%). Population size now {len(population)}.")

    # Update best if this is highest so far
    if P_new > best_perf:
        best_perf = P_new
        best_strategy_class = NewStrategyClass
else:
    print(f"Discarded new strategy (did not meet MAS={MAS:.2f}%).")
```

### Acceptance Criteria
- `P_new >= MAS`: Strategy must perform at least as well as seed
- If accepted, added to population archive
- Best strategy tracked separately

---

## Step 16: Return Best

```python
print(f"\nEvolution complete. Best strategy '{best_strategy_class.__name__}' validation return = {best_perf:.2f}%.")
return best_strategy_class, best_perf
```

---

## Complete evolve_strategy() Implementation

```python
def evolve_strategy(self, strategy_class, train_data: pd.DataFrame,
                    val_data: pd.DataFrame, max_iters=15):
    """
    Evolves the given strategy on the train/validation data using LLM mutations.
    Returns the best evolved strategy class and its performance on validation.
    """
    import inspect
    import traceback

    # 1. Compute baseline performance P0 on validation set
    _, base_result = self.run_backtest(strategy_class, val_data)
    P0 = base_result['Return (Ann.) [%]']
    print(f"Initial strategy {strategy_class.__name__} baseline annualized return on validation: {P0:.2f}%")

    # 2. Set MAS = P0
    MAS = P0

    # Archive of viable strategies
    population = [(strategy_class, P0)]
    best_perf = P0
    best_strategy_class = strategy_class

    # 4. Evolution loop
    for gen in range(1, max_iters+1):
        print(f"\nGeneration {gen}: Current population size = {len(population)}. Selecting a strategy to mutate...")

        # 5. Select a strategy from population
        parent_class, parent_perf = population[self._random_index(len(population))]
        print(f"Selected parent strategy '{parent_class.__name__}' with validation return {parent_perf:.2f}% for mutation.")

        # Get source code
        parent_code = inspect.getsource(parent_class)

        # 6. Prompt LLM A for improvement proposal
        improvement = self.llm.generate_improvement(
            parent_code,
            f"AnnReturn={parent_perf:.2f}%, Sharpe={parent_perf:.2f}"
        )
        print(f"LLM suggested improvement: {improvement}")

        # 7. Prompt LLM B to synthesize modified strategy code
        new_code = self.llm.generate_strategy_code(parent_code, improvement)

        # Unique class name
        new_class_name = f"{parent_class.__name__}_Gen{gen}"
        if new_code.startswith("class"):
            new_code = new_code.replace(parent_class.__name__, new_class_name, 1)

        # 8. Try to compile and backtest
        success = False
        for attempt in range(1, 11):
            try:
                namespace = {}
                exec(new_code, globals(), namespace)
                NewStrategyClass = namespace[new_class_name]
                metrics, res = self.run_backtest(NewStrategyClass, val_data)
                success = True
                break
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Attempt {attempt}: Strategy code failed with error: {str(e)}")
                if attempt < 10:
                    new_code = self.llm.fix_code(new_code, tb)
                    if new_class_name not in new_code:
                        new_code = new_code.replace(parent_class.__name__, new_class_name, 1)
                else:
                    print("Max repair attempts reached. Discarding this mutation.")

        if not success:
            continue

        # 14. Compute fitness
        P_new = res['Return (Ann.) [%]']
        print(f"New strategy variant '{new_class_name}' achieved validation annual return {P_new:.2f}%")

        # 15-18. Check against MAS threshold
        if P_new is not None and P_new >= MAS:
            population.append((NewStrategyClass, P_new))
            print(f"Accepted new strategy (>= MAS={MAS:.2f}%). Population size now {len(population)}.")
            if P_new > best_perf:
                best_perf = P_new
                best_strategy_class = NewStrategyClass
        else:
            print(f"Discarded new strategy (did not meet MAS={MAS:.2f}%).")

    print(f"\nEvolution complete. Best strategy '{best_strategy_class.__name__}' validation return = {best_perf:.2f}%.")
    return best_strategy_class, best_perf
```

---

## Deliverables

- [ ] `evolve_strategy()` method
- [ ] MAS threshold initialization and checking
- [ ] Population management (archive of strategies)
- [ ] Random selection from population
- [ ] Dynamic code compilation with `exec()`
- [ ] Repair loop (up to 10 attempts)
- [ ] Unique class naming per generation
- [ ] Best strategy tracking
- [ ] `_random_index()` helper method
