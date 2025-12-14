# Usage Guide

## Basic Usage

### Using the CLI

```bash
# Run with default settings
uv run python -m profit.main --data data/ES_hourly.csv

# Specify strategy
uv run python -m profit.main --data data/ES_hourly.csv --strategy MACDStrategy

# Use Anthropic Claude
uv run python -m profit.main --data data/ES_hourly.csv --provider anthropic --model claude-3-opus

# Custom configuration
uv run python -m profit.main \
    --data data/ES_hourly.csv \
    --strategy EMACrossover \
    --folds 3 \
    --capital 50000 \
    --commission 0.001
```

### Using the Python API

```python
from profit.evolver import ProfitEvolver
from profit.llm_interface import LLMClient
from profit.strategies import EMACrossover
import pandas as pd

# Load data
data = pd.read_csv("data/ES_hourly.csv", parse_dates=True, index_col=0)

# Initialize
llm = LLMClient(provider="openai", model="gpt-4")
evolver = ProfitEvolver(llm)

# Run evolution
results = evolver.walk_forward_optimize(data, EMACrossover, n_folds=5)
```

## Data Format

Your data must be a CSV with:
- DateTime index column
- OHLCV columns: Open, High, Low, Close, Volume

Example:
```csv
Datetime,Open,High,Low,Close,Volume
2020-01-01 00:00:00,100.0,101.5,99.5,101.0,10000
2020-01-01 01:00:00,101.0,102.0,100.5,101.5,12000
...
```

### Data Requirements

- **Minimum data**: At least 5 years of data for 5-fold walk-forward optimization
- **Frequency**: Hourly or daily data recommended
- **Quality**: Ensure no missing values in OHLCV columns

## Available Strategies

### Seed Strategies

| Strategy | Description |
|----------|-------------|
| BollingerMeanReversion | Mean reversion using Bollinger Bands (20-period) |
| CCIStrategy | CCI overbought/oversold signals |
| EMACrossover | EMA 50/200 crossover |
| MACDStrategy | MACD (12-26-9) crossover signals |
| WilliamsRStrategy | Williams %R oscillator (14-period) |

### Baselines

| Strategy | Description |
|----------|-------------|
| RandomStrategy | Random trading (1/3 long, 1/3 short, 1/3 flat) |
| BuyAndHoldStrategy | Buy at start, hold until end |

## Understanding Output

The system outputs:

1. **Per-generation progress**: Shows improvements proposed, acceptance/rejection decisions
2. **Per-fold test results**: Return, Sharpe ratio, expectancy for each fold
3. **Baseline comparisons**: Performance of Random and Buy-and-Hold strategies
4. **Aggregate statistics**: Mean and standard deviation across all folds

### Example Output

```
=== Fold 1/5 ===
Generation 1: Improvement accepted (Return: 12.5%, Sharpe: 1.2)
Generation 2: Improvement rejected (Return: 8.3%, below MAS)
...
Test Performance: Return=15.2%, Sharpe=1.4, Expectancy=0.25

=== Results Summary ===
Evolved Strategy: Mean Return=14.3% (+/- 3.2%)
Buy-and-Hold:     Mean Return=8.1% (+/- 5.6%)
Random:           Mean Return=-2.4% (+/- 4.1%)
```

## Advanced Usage

### Custom Seed Strategy

```python
from backtesting import Strategy
from profit.evolver import ProfitEvolver
from profit.llm_interface import LLMClient

class MyCustomStrategy(Strategy):
    def init(self):
        # Initialize indicators
        pass

    def next(self):
        # Trading logic
        pass

llm = LLMClient(provider="openai", model="gpt-4")
evolver = ProfitEvolver(llm)
results = evolver.walk_forward_optimize(data, MyCustomStrategy)
```

### Accessing Evolved Strategy Code

```python
# After evolution, get the best strategy code
results = evolver.walk_forward_optimize(data, EMACrossover)
for fold_result in results['fold_results']:
    print(f"Fold {fold_result['fold']}: Best strategy code")
    print(fold_result['best_strategy_code'])
```
