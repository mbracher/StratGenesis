# Phase 10: Documentation & Extensions

## Objective

Complete documentation for users and contributors, plus guidance on extending the system with alternative data sources.

---

## Documentation Structure

```
docs/
├── README.md            # Main documentation (or project root)
├── installation.md      # Installation guide
├── usage.md             # Usage examples
├── configuration.md     # Configuration reference
├── alternative-data.md  # Alternative data integration
└── api-reference.md     # API documentation
```

---

## README.md Updates

### Project Overview

```markdown
# ProFiT: LLM-Driven Evolutionary Trading System

ProFiT (Program Search for Financial Trading) is a framework for automated discovery
and continual improvement of trading strategies using large language models (LLMs)
within an evolutionary loop.

## Features

- **LLM-Guided Evolution**: Uses GPT-4 or Claude to suggest strategy improvements
- **Walk-Forward Validation**: Robust out-of-sample testing across multiple folds
- **Technical Indicators**: 5 seed strategies using common technical indicators
- **Baseline Comparison**: Automatic comparison against Random and Buy-and-Hold
- **Modular Design**: Easy to extend with new strategies and data sources

## Quick Start

```bash
# Install dependencies
uv sync

# Set API keys
export OPENAI_API_KEY="your-key-here"

# Run evolution
uv run python -m profit.main --data data/ES_hourly.csv --strategy EMACrossover
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   LLM A     │────▶│   LLM B     │────▶│  Backtest   │
│  (Analyst)  │     │   (Coder)   │     │   Engine    │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
  Improvement         New Code           Fitness
   Proposal                              Score
```

## License

MIT
```

---

## Installation Guide

### installation.md

```markdown
# Installation

## Requirements

- Python 3.12+
- uv package manager
- OpenAI API key and/or Anthropic API key

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-repo/profit.git
cd profit
```

2. Install dependencies:
```bash
uv sync
```

3. Configure API keys:
```bash
# Option 1: Environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Option 2: .env file
cp .env.example .env
# Edit .env with your keys
```

4. Verify installation:
```bash
uv run python -c "from profit.strategies import EMACrossover; print('OK')"
```

## Development Setup

Install with dev dependencies:
```bash
uv sync --all-extras
```

Run tests:
```bash
uv run pytest
```
```

---

## Usage Guide

### usage.md

```markdown
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

## Available Strategies

| Strategy | Description |
|----------|-------------|
| BollingerMeanReversion | Mean reversion using Bollinger Bands |
| CCIStrategy | CCI overbought/oversold signals |
| EMACrossover | EMA 50/200 crossover |
| MACDStrategy | MACD crossover signals |
| WilliamsRStrategy | Williams %R oscillator |

## Understanding Output

The system outputs:
1. Per-generation progress (improvements, acceptance/rejection)
2. Per-fold test results (return, Sharpe, expectancy)
3. Baseline comparisons (Random, Buy-and-Hold)
4. Aggregate statistics across all folds
```

---

## Configuration Reference

### configuration.md

```markdown
# Configuration Reference

## LLMClient Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | "openai" | LLM provider: "openai" or "anthropic" |
| `model` | Provider default | Model name (e.g., "gpt-4", "claude-3-opus") |
| `openai_api_key` | From env | OpenAI API key |
| `anthropic_api_key` | From env | Anthropic API key |

## ProfitEvolver Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 10000 | Starting cash for backtests |
| `commission` | 0.002 | Per-trade commission (0.2%) |
| `exclusive_orders` | True | Prevent overlapping positions |

## Evolution Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iters` | 15 | Max generations per fold |
| `n_folds` | 5 | Number of walk-forward folds |

## Walk-Forward Periods

| Period | Duration |
|--------|----------|
| Training | 2.5 years |
| Validation | 6 months |
| Test | 6 months |
| Gap | 10 days |

## Environment Variables

```bash
OPENAI_API_KEY      # Required for OpenAI provider
ANTHROPIC_API_KEY   # Required for Anthropic provider
```
```

---

## Alternative Data Integration

From the design document:

> While the above implementation focuses on price-based technical signals, the framework can be extended to include non-price data.

### alternative-data.md

```markdown
# Alternative Data Integration

ProFiT can be extended to incorporate non-price data such as macroeconomic
indicators, sentiment scores, or other alternative signals.

## Data Integration

### Adding Extra Columns

Add alternative data columns to your DataFrame:

```python
# Load price data
data = pd.read_csv("ES_hourly.csv", parse_dates=True, index_col=0)

# Load and merge sentiment data
sentiment = pd.read_csv("sentiment.csv", parse_dates=True, index_col=0)
data = data.join(sentiment, how='left')
data['Sentiment'] = data['Sentiment'].ffill()  # Forward-fill missing values

# Load and merge macro data
macro = pd.read_csv("macro.csv", parse_dates=True, index_col=0)
data = data.join(macro, how='left')
data['GDP_Growth'] = data['GDP_Growth'].ffill()
```

### Strategy Access

Strategies access additional data via `self.data.<ColumnName>`:

```python
class MacroEMACrossover(Strategy):
    def init(self):
        # Initialize EMAs (like EMACrossover)
        price = self.data.Close
        self.ema_fast = self.I(pd.Series.ewm, price, span=50).mean()
        self.ema_slow = self.I(pd.Series.ewm, price, span=200).mean()

    def next(self):
        # Only trade if macro condition is met
        if self.data.GDP_Growth[-1] > 0:
            # Apply EMA crossover logic as before
            if self.ema_fast[-1] > self.ema_slow[-1] and not self.position:
                self.buy()
            elif self.ema_fast[-1] < self.ema_slow[-1] and self.position:
                self.position.close()
```

## LLM Prompt Customization

### Informing LLM of Available Data

Modify the improvement prompt to mention available data:

```python
def generate_improvement(self, strategy_code, metrics_summary, available_data=None):
    data_note = ""
    if available_data:
        data_note = f"\n\nAlternate data available: {', '.join(available_data)}. Consider using these in the strategy."

    prompt = (
        "You are an expert trading strategy coach..."
        f"{data_note}\n\n"
        "Strategy Code:\n..."
    )
```

### Example with Sentiment

```python
# In evolver.py
improvement = self.llm.generate_improvement(
    parent_code,
    f"AnnReturn={parent_perf:.2f}%, Sharpe={sharpe:.2f}",
    available_data=['Sentiment', 'VIX', 'GDP_Growth']
)
```

The LLM might then suggest:
- "Incorporate Sentiment: only buy when sentiment is above 0.5"
- "Use VIX to adjust position sizing during high volatility"
- "Filter trades based on GDP_Growth trend"

## Example: Sentiment-Filtered Strategy

```python
class SentimentEMACrossover(Strategy):
    """EMA Crossover with sentiment filter."""
    fast_ema = 50
    slow_ema = 200
    sentiment_threshold = 0.5

    def init(self):
        price = self.data.Close
        self.ema_fast = self.I(pd.Series.ewm, price, span=self.fast_ema).mean()
        self.ema_slow = self.I(pd.Series.ewm, price, span=self.slow_ema).mean()

    def next(self):
        sentiment = self.data.Sentiment[-1]

        # Only consider long positions with positive sentiment
        if sentiment > self.sentiment_threshold:
            if self.ema_fast[-1] > self.ema_slow[-1] and not self.position:
                self.buy()

        # Exit logic unchanged
        if self.position.is_long and self.ema_fast[-1] < self.ema_slow[-1]:
            self.position.close()
```

## Regime-Aware Strategies

Use macro indicators to detect market regimes:

```python
class RegimeAwareStrategy(Strategy):
    def init(self):
        # Technical indicators
        self.setup_technical_indicators()

    def next(self):
        vix = self.data.VIX[-1]

        if vix < 20:
            # Low volatility regime: trend following
            self.trend_follow_logic()
        else:
            # High volatility regime: mean reversion
            self.mean_reversion_logic()
```

## Best Practices

1. **Data Alignment**: Ensure alternative data is properly aligned with price data timestamps
2. **Forward Fill**: Use forward-filling for sparse data to avoid look-ahead bias
3. **Feature Selection**: Start with few features and let evolution discover useful combinations
4. **Prompt Engineering**: Clearly describe available data and its meaning to the LLM
```

---

## Deliverables

- [ ] Updated README.md with project overview
- [ ] `docs/installation.md` - Installation guide
- [ ] `docs/usage.md` - Usage examples
- [ ] `docs/configuration.md` - Configuration reference
- [ ] `docs/alternative-data.md` - Alternative data integration guide
- [ ] API reference documentation (optional, can use docstrings)
- [ ] Example data files or instructions for obtaining data
