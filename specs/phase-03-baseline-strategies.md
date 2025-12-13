# Phase 3: Baseline Strategies

## Objective

Implement two baseline strategies for performance comparison. These baselines represent naive approaches that evolved strategies must outperform to demonstrate value.

From the design document:

> We also define baseline strategies (Random and BuyAndHold) for performance comparison.

---

## Strategy 1: RandomStrategy

> Implements R0 as described in the paper. When not in a position, it randomly chooses to go long, short, or stay out (each 1/3 chance). If holding a position, it exits with 50% probability on each step.

### Behavior
- **When flat:** Randomly choose action with equal probability:
  - 1/3 chance: Go long
  - 1/3 chance: Go short
  - 1/3 chance: Do nothing
- **When holding:** Exit with 50% probability each time step

### Implementation

```python
class RandomStrategy(Strategy):
    """Random trading strategy (baseline R0).
    When flat, randomly choose to go long, short, or do nothing.
    If holding a position, exit with 50% probability each time step.
    """
    def init(self):
        # Use numpy RandomState for reproducibility (set a seed if desired)
        self.rs = np.random.RandomState(42)  # fixed seed for repeatable random behavior

    def next(self):
        if not self.position:
            # Choose action uniformly: 0=do nothing, 1=buy, 2=sell(short)
            choice = self.rs.randint(0, 3)
            if choice == 1:
                self.buy()
            elif choice == 2:
                self.sell()
        else:
            # If currently in a trade, decide to exit with 50% probability
            if self.rs.rand() < 0.5:
                self.position.close()
```

### Notes
- Uses `np.random.RandomState(42)` for reproducibility
- Seed can be parameterized for different random runs
- Represents a "no-skill" baseline

---

## Strategy 2: BuyAndHoldStrategy

> Buys the maximum quantity at the first bar and holds until the end, representing a passive long benchmark.

### Behavior
- Buy with all available capital at the first data point
- Hold position until end of backtest (no exit)
- Represents passive market exposure

### Implementation

```python
class BuyAndHoldStrategy(Strategy):
    """Buy-and-Hold strategy (baseline B&H).
    Buys with all capital at the first bar and holds the position until the end.
    """
    def init(self):
        self.bought = False

    def next(self):
        if not self.bought:
            # Enter long with all available cash at the first data point
            self.buy()
            self.bought = True
        # No exit until the end (the backtesting engine will mark-to-market the position)
```

### Notes
- Simple flag to track if initial buy occurred
- Position is marked-to-market by backtesting engine
- Represents "market return" baseline

---

## Performance Comparison Context

From the design document:

> After evolution, the system compares the best evolved strategy against baselines (Random and Buy-and-Hold) on test data, in terms of annualized return, Sharpe ratio, and expectancy.

### Metrics for Comparison
1. **Annualized Return (%)** - Primary fitness metric
2. **Sharpe Ratio** - Risk-adjusted return
3. **Expectancy (%)** - Expected return per trade

### Expected Results

From the ProFiT paper results:

> Evolved strategies beat buy-and-hold in ~77% and random in 100% of cases.

---

## Usage in Evolution

Baselines are evaluated alongside evolved strategies:

```python
# Evaluate best evolved strategy
metrics_evolved, _ = self.run_backtest(best_strat, test)

# Evaluate baselines
_, res_rand = self.run_backtest(RandomStrategy, test)
_, res_bh = self.run_backtest(BuyAndHoldStrategy, test)

# Compare
print(f"Evolved: {metrics_evolved['AnnReturn%']:.2f}%")
print(f"Random: {res_rand['Return (Ann.) [%]']:.2f}%")
print(f"Buy&Hold: {res_bh['Return (Ann.) [%]']:.2f}%")
```

---

## Complete strategies.py Structure

After Phase 2 and 3, the file should contain:

```python
# File: strategies.py
from backtesting import Strategy
import pandas as pd
import numpy as np

# Seed strategies
class BollingerMeanReversion(Strategy): ...
class CCIStrategy(Strategy): ...
class EMACrossover(Strategy): ...
class MACDStrategy(Strategy): ...
class WilliamsRStrategy(Strategy): ...

# Baseline strategies
class RandomStrategy(Strategy): ...
class BuyAndHoldStrategy(Strategy): ...
```

---

## Deliverables

- [ ] `RandomStrategy` class with seeded randomness
- [ ] `BuyAndHoldStrategy` class
- [ ] Both strategies testable with backtesting.py
- [ ] Verification that baselines produce expected behavior
