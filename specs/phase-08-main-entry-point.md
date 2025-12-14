# Phase 8: Main Entry Point

## Objective

Create a usable entry point for running the ProFiT system, including data loading, configuration, and results output.

From the design document:

> Finally, we demonstrate how to use these modules together. In the example below, we assume you have historical hourly data for a given asset (one of the seven futures like ES, 6E, etc.) loaded into a pandas DataFrame.

---

## Example Usage (from design document)

```python
# File: main.py (example usage)
import pandas as pd
from evolver import ProfitEvolver
from strategies import EMACrossover  # using EMA Crossover as an example seed
from llm_interface import LLMClient

# Load your 1-hour historical data for an asset (e.g., ES).
# For example, from a CSV:
data = pd.read_csv("ES_hourly.csv", parse_dates=True, index_col="Datetime")

# Instantiate the LLM client (OpenAI GPT example; ensure OPENAI_API_KEY is set in env)
llm_client = LLMClient(provider="openai", model="gpt-4")

# Initialize the evolver with the LLM client
evolver = ProfitEvolver(llm_client)

# Run walk-forward evolution and testing for the chosen strategy
results = evolver.walk_forward_optimize(data, strategy_class=EMACrossover, n_folds=5)

# Print summary of results
for res in results:
    print(f"Fold {res['fold']}: Best Strategy = {res['strategy'].__name__}, "
          f"Return = {res['ann_return']:.2f}%, Sharpe = {res['sharpe']:.2f}, "
          f"Expectancy = {res['expectancy']:.2f}%, "
          f"RandomBaseline = {res['random_return']:.2f}%, "
          f"BuyHoldBaseline = {res['buy_hold_return']:.2f}%")
```

---

## Data Requirements

### DataFrame Format

```python
# Required columns
['Open', 'High', 'Low', 'Close', 'Volume']

# Required index
pd.DatetimeIndex
```

### Data Loading Example

```python
data = pd.read_csv("ES_hourly.csv", parse_dates=True, index_col="Datetime")
```

### Supported Assets (from paper)
Seven liquid futures assets tested:
- ES (S&P 500 E-mini)
- 6E (Euro FX)
- And 5 others

### Data Resolution
- Hourly (1-hour bars) as used in experiments
- Must span enough time for 5 folds (~17.5 years minimum)

---

## Configuration Options

### LLM Configuration

```python
# OpenAI
llm_client = LLMClient(provider="openai", model="gpt-4")

# Anthropic
llm_client = LLMClient(provider="anthropic", model="claude-2")

# With explicit API keys
llm_client = LLMClient(
    provider="openai",
    model="gpt-4",
    openai_api_key="sk-..."
)
```

### Evolution Configuration

```python
evolver = ProfitEvolver(
    llm_client,
    initial_capital=10000,   # Starting cash
    commission=0.002,        # 0.2% per trade
    exclusive_orders=True    # No overlapping positions
)
```

### Walk-Forward Configuration

```python
results = evolver.walk_forward_optimize(
    data,
    strategy_class=EMACrossover,
    n_folds=5
)
```

---

## Seed Strategy Options

Available seed strategies:

```python
from strategies import (
    BollingerMeanReversion,
    CCIStrategy,
    EMACrossover,
    MACDStrategy,
    WilliamsRStrategy
)

# Choose one as the starting point
results = evolver.walk_forward_optimize(data, strategy_class=MACDStrategy, n_folds=5)
```

---

## Results Output

### Per-Fold Output

```python
for res in results:
    print(f"Fold {res['fold']}: Best Strategy = {res['strategy'].__name__}, "
          f"Return = {res['ann_return']:.2f}%, Sharpe = {res['sharpe']:.2f}, "
          f"Expectancy = {res['expectancy']:.2f}%, "
          f"RandomBaseline = {res['random_return']:.2f}%, "
          f"BuyHoldBaseline = {res['buy_hold_return']:.2f}%")
```

### Expected Console Output

```
=== Fold 1 ===
Training period: 2008-01-02 00:00:00 to 2010-06-30 23:00:00
Validation period: 2010-07-10 00:00:00 to 2011-01-09 23:00:00
Test period: 2011-01-19 00:00:00 to 2011-07-18 23:00:00
Initial strategy EMACrossover baseline annualized return on validation: X.XX%

Generation 1: Current population size = 1. Selecting a strategy to mutate...
Selected parent strategy 'EMACrossover' with validation return X.XX% for mutation.
LLM suggested improvement: *Increase the short EMA period to respond faster to changes.*
New strategy variant 'EMACrossover_Gen1' achieved validation annual return Y.YY%
Accepted new strategy (>= MAS=X.XX%). Population size now 2.

...

Evolution complete. Best strategy 'EMACrossover_Gen5' validation return = Z.ZZ%.
Fold 1 Test Performance - Annualized Return: Z1.ZZ%, Sharpe: S1.SS, Expectancy: E1.EE%
Fold 1 Baselines - Random Strat Return: R1.RR%, Buy&Hold Return: B1.BB%

=== Fold 2 ===
...

Average Annualized Return over 5 folds: M.ZZ%
Average Buy-and-Hold Return over 5 folds: N.NN%
Average Random Strategy Return over 5 folds: O.OO%
```

---

## Enhanced Main Script

### Full Implementation

```python
#!/usr/bin/env python
"""
ProFiT: LLM-Driven Evolutionary Trading System

Usage:
    python main.py --data ES_hourly.csv --strategy EMACrossover --provider openai --model gpt-4
"""

import argparse
import pandas as pd
import sys

from evolver import ProfitEvolver
from llm_interface import LLMClient
from strategies import (
    BollingerMeanReversion,
    CCIStrategy,
    EMACrossover,
    MACDStrategy,
    WilliamsRStrategy
)

# Map of strategy names to classes
STRATEGIES = {
    "BollingerMeanReversion": BollingerMeanReversion,
    "CCIStrategy": CCIStrategy,
    "EMACrossover": EMACrossover,
    "MACDStrategy": MACDStrategy,
    "WilliamsRStrategy": WilliamsRStrategy,
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from CSV file."""
    data = pd.read_csv(filepath, parse_dates=True, index_col=0)

    # Validate required columns
    required = ['Open', 'High', 'Low', 'Close']
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be datetime")

    return data


def print_results(results: list):
    """Print formatted results summary."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for res in results:
        print(f"\nFold {res['fold']}:")
        print(f"  Best Strategy: {res['strategy'].__name__}")
        print(f"  Annualized Return: {res['ann_return']:.2f}%")
        print(f"  Sharpe Ratio: {res['sharpe']:.2f}")
        print(f"  Expectancy: {res['expectancy']:.2f}%")
        print(f"  vs Random: {res['ann_return'] - res['random_return']:+.2f}%")
        print(f"  vs Buy&Hold: {res['ann_return'] - res['buy_hold_return']:+.2f}%")

    # Aggregate stats
    import numpy as np
    avg_ret = np.mean([r["ann_return"] for r in results])
    avg_bh = np.mean([r["buy_hold_return"] for r in results])
    avg_rand = np.mean([r["random_return"] for r in results])

    print("\n" + "-" * 60)
    print("AVERAGES ACROSS FOLDS:")
    print(f"  Evolved Strategy: {avg_ret:.2f}%")
    print(f"  Buy-and-Hold: {avg_bh:.2f}%")
    print(f"  Random: {avg_rand:.2f}%")
    print(f"  Improvement over B&H: {avg_ret - avg_bh:+.2f}%")
    print(f"  Improvement over Random: {avg_ret - avg_rand:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="ProFiT: LLM-Driven Evolutionary Trading System")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--strategy", default="EMACrossover", choices=STRATEGIES.keys(),
                        help="Seed strategy to evolve")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LLM provider")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.002, help="Commission rate")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    print(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    # Initialize LLM client
    print(f"Initializing {args.provider} LLM client...")
    llm_client = LLMClient(provider=args.provider, model=args.model)

    # Initialize evolver
    evolver = ProfitEvolver(
        llm_client,
        initial_capital=args.capital,
        commission=args.commission
    )

    # Get strategy class
    strategy_class = STRATEGIES[args.strategy]
    print(f"Using seed strategy: {strategy_class.__name__}")

    # Run walk-forward optimization
    print(f"\nStarting walk-forward optimization with {args.folds} folds...")
    results = evolver.walk_forward_optimize(data, strategy_class, n_folds=args.folds)

    # Print results
    print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## CLI Usage Examples

```bash
# Basic usage
uv run python main.py --data ES_hourly.csv

# Specify strategy and LLM
uv run python main.py --data ES_hourly.csv --strategy MACDStrategy --provider anthropic --model claude-3-opus

# Custom configuration
uv run python main.py --data ES_hourly.csv --folds 3 --capital 50000 --commission 0.001
```

---

## Deliverables

- [x] `main.py` with data loading
- [x] Command-line argument parsing
- [x] Strategy selection by name
- [x] LLM provider configuration
- [x] Results formatting and output
- [x] Data validation
- [x] Error handling
