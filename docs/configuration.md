# Configuration Reference

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | Required | Path to OHLCV CSV file |
| `--strategy` | EMACrossover | Seed strategy to evolve |
| `--provider` | openai | LLM provider: "openai" or "anthropic" |
| `--model` | gpt-4 | LLM model name |
| `--folds` | 5 | Number of walk-forward folds |
| `--capital` | 10000 | Initial capital |
| `--commission` | 0.002 | Commission rate (0.2%) |

## LLMClient Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | "openai" | LLM provider: "openai" or "anthropic" |
| `model` | Provider default | Model name (e.g., "gpt-4", "claude-3-opus") |
| `openai_api_key` | From env | OpenAI API key |
| `anthropic_api_key` | From env | Anthropic API key |

### Supported Models

**OpenAI:**
- gpt-4
- gpt-4-turbo
- gpt-3.5-turbo

**Anthropic:**
- claude-3-opus
- claude-3-sonnet
- claude-3-haiku

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

| Period | Duration | Description |
|--------|----------|-------------|
| Training | 2.5 years | Data used for strategy development |
| Validation | 6 months | Data used for fitness evaluation during evolution |
| Test | 6 months | Out-of-sample evaluation (held out) |
| Gap | 10 days | Buffer between periods to prevent look-ahead |

### Walk-Forward Timeline

```
Fold 1: |----Train----|--Val--|--Test--|
Fold 2:        |----Train----|--Val--|--Test--|
Fold 3:              |----Train----|--Val--|--Test--|
...
```

## Environment Variables

```bash
# Required for OpenAI provider
OPENAI_API_KEY="sk-..."

# Required for Anthropic provider
ANTHROPIC_API_KEY="sk-ant-..."
```

## Default Configuration Values

| Parameter | Value |
|-----------|-------|
| Initial Capital | $10,000 |
| Commission | 0.2% |
| Walk-Forward Folds | 5 |
| Training Period | 2.5 years |
| Validation Period | 6 months |
| Test Period | 6 months |
| Gap Between Periods | 10 days |
| Max Evolution Iterations | 15 |
| Max Code Repair Attempts | 10 |

## Programmatic Configuration

```python
from profit.evolver import ProfitEvolver
from profit.llm_interface import LLMClient

# Configure LLM client
llm = LLMClient(
    provider="openai",
    model="gpt-4",
    openai_api_key="sk-..."  # Or use env var
)

# Configure evolver
evolver = ProfitEvolver(
    llm_client=llm,
    initial_capital=50000,
    commission=0.001
)

# Run with custom fold count
results = evolver.walk_forward_optimize(
    data,
    strategy_class,
    n_folds=3,
    max_iters=20
)
```
