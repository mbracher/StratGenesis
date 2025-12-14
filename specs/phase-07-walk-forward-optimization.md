# Phase 7: Walk-Forward Optimization

## Objective

Implement multi-fold walk-forward optimization with baseline comparison and results aggregation.

From the design document:

> The `walk_forward_optimize()` method runs the above evolution for each fold, then evaluates the chosen strategy on that fold's test set. It collects performance metrics across all test folds for final comparison to baselines.

---

## Method: walk_forward_optimize()

### Signature

```python
def walk_forward_optimize(self, full_data: pd.DataFrame, strategy_class, n_folds=5):
    """
    Perform walk-forward optimization: evolve strategy on each training set,
    evaluate on its test set. Returns a summary of performance across folds.
    """
```

### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `full_data` | DataFrame | Complete historical data |
| `strategy_class` | class | Seed strategy to evolve |
| `n_folds` | int | Number of walk-forward folds (default 5) |

### Returns
- `results`: List of per-fold result dictionaries

---

## Implementation

```python
def walk_forward_optimize(self, full_data: pd.DataFrame, strategy_class, n_folds=5):
    """
    Perform walk-forward optimization: evolve strategy on each training set,
    evaluate on its test set. Returns a summary of performance across folds.
    """
    folds = self.prepare_folds(full_data, n_folds=n_folds)
    results = []

    for i, (train, val, test) in enumerate(folds, start=1):
        print(f"\n=== Fold {i} ===")
        print(f"Training period: {train.index[0]} to {train.index[-1]}")
        print(f"Validation period: {val.index[0]} to {val.index[-1]}")
        print(f"Test period: {test.index[0]} to {test.index[-1]}")

        # Evolve strategy on this fold's data
        best_strat, _ = self.evolve_strategy(strategy_class, train, val)

        # Evaluate best strategy on test set
        metrics, res = self.run_backtest(best_strat, test)
        ann_return = metrics["AnnReturn%"]
        sharpe = metrics["Sharpe"]
        expectancy = metrics["Expectancy%"]
        print(f"Fold {i} Test Performance - Annualized Return: {ann_return:.2f}%, Sharpe: {sharpe:.2f}, Expectancy: {expectancy:.2f}%")

        # Also evaluate baselines on the test set for comparison
        _, res_rand = self.run_backtest(RandomStrategy, test)
        _, res_bh = self.run_backtest(BuyAndHoldStrategy, test)
        rand_return = res_rand['Return (Ann.) [%]']
        bh_return = res_bh['Return (Ann.) [%]']
        print(f"Fold {i} Baselines - Random Strat Return: {rand_return:.2f}%, Buy&Hold Return: {bh_return:.2f}%")

        results.append({
            "fold": i,
            "strategy": best_strat,
            "ann_return": ann_return,
            "sharpe": sharpe,
            "expectancy": expectancy,
            "random_return": rand_return,
            "buy_hold_return": bh_return
        })

    # Summarize across folds
    avg_ret = np.mean([r["ann_return"] for r in results])
    avg_bh = np.mean([r["buy_hold_return"] for r in results])
    avg_rand = np.mean([r["random_return"] for r in results])

    print(f"\nAverage Annualized Return over {n_folds} folds: {avg_ret:.2f}%")
    print(f"Average Buy-and-Hold Return over {n_folds} folds: {avg_bh:.2f}%")
    print(f"Average Random Strategy Return over {n_folds} folds: {avg_rand:.2f}%")

    return results
```

---

## Walk-Forward Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Full Data                                │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────┤
│  Fold 1 │  Fold 1 │  Fold 1 │  Fold 2 │  Fold 2 │  Fold 2 │ ... │
│  Train  │   Val   │  Test   │  Train  │   Val   │  Test   │     │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────┘
     │         │         │
     │         │         │
     ▼         ▼         ▼
  Evolve   Fitness   Final
  Here     Eval      Eval
```

---

## Per-Fold Process

### Step 1: Data Splitting
```python
folds = self.prepare_folds(full_data, n_folds=n_folds)
```

### Step 2: Evolution
```python
best_strat, _ = self.evolve_strategy(strategy_class, train, val)
```
- Uses training data for context
- Uses validation data for fitness evaluation
- Returns best evolved strategy

### Step 3: Test Evaluation
```python
metrics, res = self.run_backtest(best_strat, test)
```
- Evaluates on completely held-out test data
- No information leakage

### Step 4: Baseline Comparison
```python
_, res_rand = self.run_backtest(RandomStrategy, test)
_, res_bh = self.run_backtest(BuyAndHoldStrategy, test)
```
- Same test data for fair comparison
- Random and Buy-and-Hold baselines

---

## Results Structure

### Per-Fold Result Dictionary

```python
{
    "fold": int,           # Fold number (1-indexed)
    "strategy": class,     # Best evolved strategy class
    "ann_return": float,   # Annualized return on test (%)
    "sharpe": float,       # Sharpe ratio on test
    "expectancy": float,   # Expectancy on test (%)
    "random_return": float,    # Random baseline return (%)
    "buy_hold_return": float   # Buy-and-hold baseline return (%)
}
```

### Aggregated Metrics

```python
avg_ret = np.mean([r["ann_return"] for r in results])
avg_bh = np.mean([r["buy_hold_return"] for r in results])
avg_rand = np.mean([r["random_return"] for r in results])
```

---

## Expected Output Format

From the design document:

```
=== Fold 1 ===
Training period: 2008-01-02 00:00:00 to 2010-06-30 23:00:00
...
Initial strategy EMACrossover baseline annualized return on validation: X.XX%
Generation 1: ...
LLM suggested improvement: *Increase the short EMA period to respond faster to changes.*
New strategy variant 'EMACrossover_Gen1' achieved validation annual return Y.YY% ...
Accepted new strategy ...
...
Evolution complete. Best strategy 'EMACrossover_Gen5' validation return = Z.ZZ%.
Fold 1 Test Performance - Annualized Return: Z1.ZZ%, Sharpe: S1.SS, Expectancy: E1.EE%
Fold 1 Baselines - Random Strat Return: R1.RR%, Buy&Hold Return: B1.BB%

=== Fold 2 ===
... (similar logs for each fold) ...

Average Annualized Return over 5 folds: M.ZZ%
Average Buy-and-Hold Return over 5 folds: N.NN%
Average Random Strategy Return over 5 folds: O.OO%
```

---

## Performance Expectations

From the ProFiT paper:

> Evolved strategies beat buy-and-hold in ~77% and random in 100% of cases.

### Success Criteria
- Evolved strategy should outperform Random baseline consistently
- Evolved strategy should outperform Buy-and-Hold in majority of folds

---

## Note on Data Usage

From the design document:

> In this code, the evolutionary loop uses validation data performance as the fitness score to decide if a new strategy is viable. In practice, one might evolve strategies on the training set and use validation purely to evaluate fitness, ensuring we don't leak test information. The key is that test data remains completely held-out until final evaluation.

### Data Separation
| Data | Used For |
|------|----------|
| Train | Strategy can "see" market conditions |
| Validation | Fitness evaluation during evolution |
| Test | Final out-of-sample evaluation only |

---

## Complete evolver.py Structure (After Phase 7)

```python
# File: evolver.py
from backtesting import Backtest
import pandas as pd
import numpy as np
from importlib import import_module
import types
import traceback

from strategies import (
    BollingerMeanReversion, CCIStrategy, EMACrossover,
    MACDStrategy, WilliamsRStrategy,
    RandomStrategy, BuyAndHoldStrategy
)
from llm_interface import LLMClient

class ProfitEvolver:
    def __init__(self, llm_client: LLMClient, initial_capital=10000,
                 commission=0.002, exclusive_orders=True): ...

    def prepare_folds(self, full_data: pd.DataFrame, n_folds=5): ...

    def run_backtest(self, strategy_class, data: pd.DataFrame): ...

    def evolve_strategy(self, strategy_class, train_data: pd.DataFrame,
                        val_data: pd.DataFrame, max_iters=15): ...

    def walk_forward_optimize(self, full_data: pd.DataFrame,
                              strategy_class, n_folds=5): ...

    def _random_index(self, n): ...
```

---

## Deliverables

- [x] `walk_forward_optimize()` method
- [x] Per-fold evolution execution
- [x] Test set evaluation for evolved strategies
- [x] Baseline comparison (Random, Buy-and-Hold)
- [x] Results collection and aggregation
- [x] Summary statistics printing
- [x] Complete `evolver.py` module
