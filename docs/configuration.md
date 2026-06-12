# Configuration Reference

## CLI Arguments

### Core

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | Required | Path to OHLCV CSV file (not needed with `--export-strategy`) |
| `--strategy` | EMACrossover | Seed strategy to evolve |
| `--folds` | 5 | Number of walk-forward folds |
| `--capital` | 10000 | Initial capital |
| `--commission` | 0.002 | Commission rate (0.2%) |
| `--no-finalize-trades` | off | Don't auto-close open trades at backtest end |
| `--output-dir` | None | **Deprecated.** Legacy file persistence directory; emits a `DeprecationWarning` when used. The program database is the system of record; use `--export-strategy` instead |

### LLM Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--provider` | openai | Default LLM provider for both roles: "openai" or "anthropic" |
| `--model` | Provider default | Default model for both roles |
| `--analyst-provider` | None | Provider for analysis/improvements (overrides `--provider`) |
| `--analyst-model` | None | Model for analysis/improvements (overrides `--model`) |
| `--coder-provider` | None | Provider for code generation (overrides `--provider`) |
| `--coder-model` | None | Model for code generation (overrides `--model`) |

### Program Database

| Argument | Default | Description |
|----------|---------|-------------|
| `--db-backend` | json | Program database backend: "json" or "sqlite" |
| `--db-path` | program_db | Path for the program database |
| `--no-inspirations` | off | Disable inspiration sampling from the program database |
| `--export-strategy` | — | Export a strategy from the program database to a .py file by its ID, then exit |
| `--export-dir` | exported_strategies | Directory for exported strategies |

The program database is created on every run. Each generation samples 3 inspiration strategies (mode "mixed", excluding the parent, filtered by the fold's evaluation context) and feeds them to the analyst LLM prompt. `--no-inspirations` disables only this sampling — strategies are still recorded to the database. Every record carries an `eval_context_id` (dataset, date ranges, capital, commission) and, when a cascade ran, a `cascade_result` annotation.

### Diff Mutations

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-diffs` | off | Disable diff-based mutations (use full rewrites only) |
| `--diff-mode` | adaptive | When to use diffs: "always", "never", or "adaptive" |
| `--diff-match` | tolerant | Diff matching mode: "strict" (literal) or "tolerant" (normalized) |
| `--exploration-gens` | 5 | In adaptive mode, use full rewrites for the first N generations |

### Selection Policy

| Argument | Default | Description |
|----------|---------|-------------|
| `--selection-policy` | gated | Acceptance policy: "weighted", "gated", or "pareto" |
| `--min-return` | 0.0 | Minimum annualized return threshold (gated) |
| `--min-sharpe` | 0.0 | Minimum Sharpe ratio threshold (gated) |
| `--max-drawdown` | -50.0 | Maximum drawdown threshold, negative (gated) |
| `--min-trades` | 1 | Minimum number of trades required (gated) |
| `--min-consistency` | 0.0 | Minimum % of folds with positive return (gated) |
| `--min-worst-fold` | -100.0 | Minimum return for worst fold (gated) |
| `--max-stability` | 100.0 | Maximum cross-fold std dev (gated) |
| `--w-return` | 0.5 | Weight for return (weighted) |
| `--w-sharpe` | 0.3 | Weight for Sharpe (weighted) |
| `--w-drawdown` | 0.2 | Weight for drawdown (weighted) |
| `--pareto-objectives` | ann_return sharpe max_drawdown | Objectives for Pareto selection |
| `--debug-policy` | off | Debug logging for selection policy decisions |

The legacy plain-MAS acceptance path is no longer reachable from the CLI; it is available programmatically only via `selection_policy=None`.

### Evaluation Cascade and Promotion Gate

| Argument | Default | Description |
|----------|---------|-------------|
| `--skip-cascade` | off | Skip the evaluation cascade (direct backtest only) |
| `--quick-eval` | off | Quick evaluation: syntax check + smoke test only |
| `--smoke-months` | 3 | Months of data for smoke test |
| `--risk-free-rate` | 0.0 | Annual risk-free rate for Sharpe/Sortino |
| `--gate-min-trades` | 1 | Promotion gate: minimum trades |
| `--gate-max-drawdown` | -80.0 | Promotion gate: max drawdown limit |
| `--gate-min-sharpe` | None | Promotion gate: minimum Sharpe ratio (disabled by default) |
| `--gate-min-win-rate` | None | Promotion gate: minimum win rate % (disabled by default) |

The cascade runs by default in full mode with four stages: syntax_check, smoke_test, single_fold, full_walkforward. `--quick-eval` reduces it to syntax + smoke test; `--skip-cascade` disables it entirely. The full walk-forward stage receives the walk-forward folds automatically.

## LLMClient Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | "openai" | LLM provider: "openai" or "anthropic" |
| `model` | Provider default | Model name (e.g., "gpt-5.2", "claude-sonnet-4-6") |
| `openai_api_key` | From env | OpenAI API key |
| `anthropic_api_key` | From env | Anthropic API key |

### Default Models

| Provider | Default Model |
|----------|---------------|
| openai | gpt-5.2 |
| anthropic | claude-sonnet-4-6 |

Model IDs are exact strings — do not append date suffixes.

### Recommended Configurations

| Goal | Analyst | Coder |
|------|---------|-------|
| Best quality | claude-opus-4-8 | claude-sonnet-4-6 |
| Balanced | gpt-5.2 | claude-sonnet-4-6 |
| Cost-effective | claude-haiku-4-5 | claude-haiku-4-5 |

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
    model="gpt-5.2",
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
