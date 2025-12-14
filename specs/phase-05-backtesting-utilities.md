# Phase 5: Backtesting Utilities

## Objective

Implement core backtesting functions and walk-forward data splitting utilities. These form the foundation for strategy evaluation in the evolutionary loop.

---

## ProfitEvolver Class Setup

From the design document:

```python
class ProfitEvolver:
    def __init__(self, llm_client: LLMClient, initial_capital=10000, commission=0.002, exclusive_orders=True):
        """
        llm_client: an instance of LLMClient for generating strategy mutations.
        initial_capital: starting cash for backtests.
        commission: per-trade commission rate (e.g., 0.002 = 0.2%).
        exclusive_orders: if True, no overlapping long/short positions.
        """
        self.llm = llm_client
        self.initial_capital = initial_capital
        self.commission = commission
        self.exclusive_orders = exclusive_orders
```

### Configuration Defaults
| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | $10,000 | Starting cash |
| `commission` | 0.002 (0.2%) | Per-trade commission |
| `exclusive_orders` | True | No overlapping positions |

---

## Method 1: run_backtest()

> Takes a strategy class and data subset, runs the `backtesting.Backtest`, and returns performance metrics. We extract key metrics (annualized return, Sharpe ratio, expectancy) from the result.

### Implementation

```python
def run_backtest(self, strategy_class, data: pd.DataFrame):
    """
    Helper to run a backtest on given data with specified strategy class.
    Returns a dict of performance metrics and the full result Series.
    """
    bt = Backtest(
        data,
        strategy_class,
        cash=self.initial_capital,
        commission=self.commission,
        exclusive_orders=self.exclusive_orders
    )
    result = bt.run()

    # Extract key metrics
    metrics = {
        "AnnReturn%": result.get('Return (Ann.) [%]', None),
        "Sharpe": result.get('Sharpe Ratio', None),
        "Expectancy%": result.get('Expectancy [%]', None),
        "Trades": result.get('# Trades', None),
    }
    return metrics, result
```

### Return Values
- `metrics` (dict): Extracted key performance metrics
- `result` (pd.Series): Full backtesting.py result object

### Key Metrics Extracted

| Metric | Key in Result | Description |
|--------|---------------|-------------|
| Annualized Return | `'Return (Ann.) [%]'` | Primary fitness metric |
| Sharpe Ratio | `'Sharpe Ratio'` | Risk-adjusted return |
| Expectancy | `'Expectancy [%]'` | Expected return per trade |
| Number of Trades | `'# Trades'` | Trade frequency |

---

## Method 2: prepare_folds()

> Supports splitting historical data into five sequential folds (each with train, validation, test segments) to mimic the paper's robust evaluation regime. Strategies are evolved on training sets and evaluated on validation sets during evolution, with final performance assessed on out-of-sample test sets.

### Walk-Forward Validation Structure

From the design document:

> By default, it uses 5 folds as in the paper, each with ~2.5 years training, 6 months validation, 6 months test, and a 10-day gap between periods.

```
Fold 1:
├── Train:      2.5 years
├── [Gap]:      10 days
├── Validation: 6 months
├── [Gap]:      10 days
└── Test:       6 months

Fold 2:
├── Train:      2.5 years (starts after Fold 1 test)
├── [Gap]:      10 days
├── Validation: 6 months
...
```

### Implementation

```python
def prepare_folds(self, full_data: pd.DataFrame, n_folds=5):
    """
    Split the full_data DataFrame into train/validation/test folds.
    Returns a list of (train_df, val_df, test_df) tuples.
    Assumes full_data index is datetime and spans enough period for n_folds.
    """
    fold_splits = []
    data_index = full_data.index
    start_date = data_index.min()

    for fold in range(n_folds):
        # Train period: 2.5 years from current start
        train_end = start_date + pd.DateOffset(years=2) + pd.DateOffset(months=6)

        # Validation: 6 months after train
        val_end = train_end + pd.DateOffset(months=6)

        # Test: 6 months after validation
        test_end = val_end + pd.DateOffset(months=6)

        # Add 10-day gaps between periods
        val_start = train_end + pd.DateOffset(days=10)
        test_start = val_end + pd.DateOffset(days=10)

        # Slice data
        train = full_data[start_date:train_end]
        val = full_data[val_start:val_end]
        test = full_data[test_start:test_end]

        if len(test) == 0:
            break  # in case we run out of data

        fold_splits.append((train, val, test))

        # Move start_date to end of this test for next fold
        start_date = test_end

    return fold_splits
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `full_data` | - | Complete DataFrame with datetime index |
| `n_folds` | 5 | Number of walk-forward folds |

### Return Value
List of tuples: `[(train_df, val_df, test_df), ...]`

---

## Data Requirements

### DataFrame Format
```python
# Required columns
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Required index
data.index = pd.DatetimeIndex  # Must be datetime
```

### Minimum Data Duration
For 5 folds with default settings:
- Per fold: 2.5y + 6mo + 6mo = 3.5 years
- Total minimum: ~17.5 years of data

### Example Data Loading
```python
data = pd.read_csv("data/ES_hourly.csv", parse_dates=True, index_col="Datetime")
```

---

## Time Period Configuration

### Default Periods (from paper)
| Period | Duration |
|--------|----------|
| Training | 2.5 years (30 months) |
| Validation | 6 months |
| Test | 6 months |
| Gap | 10 days |

### Gap Purpose
> The 10-day gap (dormant period) between periods prevents look-ahead bias and simulates real-world deployment lag.

---

## Usage Example

```python
from evolver import ProfitEvolver
from llm_interface import LLMClient
import pandas as pd

# Load data
data = pd.read_csv("data/ES_hourly.csv", parse_dates=True, index_col="Datetime")

# Initialize
llm = LLMClient(provider="openai", model="gpt-4")
evolver = ProfitEvolver(llm)

# Prepare folds
folds = evolver.prepare_folds(data, n_folds=5)

for i, (train, val, test) in enumerate(folds):
    print(f"Fold {i+1}:")
    print(f"  Train: {train.index[0]} to {train.index[-1]}")
    print(f"  Val:   {val.index[0]} to {val.index[-1]}")
    print(f"  Test:  {test.index[0]} to {test.index[-1]}")
```

---

## File Structure (evolver.py - Part 1)

```python
# File: evolver.py
from backtesting import Backtest
import pandas as pd
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

    # Phase 6-7 methods will be added later
```

---

## Deliverables

- [x] `ProfitEvolver` class with constructor
- [x] `run_backtest()` method returning metrics dict and result
- [x] `prepare_folds()` method with configurable fold count
- [x] Proper datetime handling for data splitting
- [x] 10-day gap implementation between periods
