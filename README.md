# StratGenesis

StratGenesis is a research-oriented codebase for **evolving algorithmic trading strategies** using LLMs inside an automated evolutionary loop.

This repository is an implementation that **combines ideas from two papers**:

- **ProFiT — “Program Search for Financial Trading”** (LLM-driven evolutionary program search for trading strategies, with walk-forward validation):  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5889762  
  (alt mirror: https://www.researchgate.net/publication/398248186_ProFiT_Program_Search_for_Financial_Trading)

- **AlphaEvolve — “A coding agent for scientific and algorithmic discovery”** (iterative code improvement via evaluator feedback and evolutionary search):  
  https://arxiv.org/abs/2506.13131  
  (blog + pdf: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

On top of these foundations, StratGenesis adds autonomous **Research** and **Data Collection** agents to support strategy idea discovery and dataset acquisition/curation for evaluation.

> **AI-generated code notice:** This repository (architecture, implementation, and documentation) was produced end-to-end by **ChatGPT** and **Claude Code**. Treat it as experimental research software and review it carefully before relying on it.

> ⚠️ **Security warning — run at your own risk:** StratGenesis can prompt LLMs to **generate/modify code** (e.g., strategy implementations and patches) and may **execute that generated code** as part of the evaluation loop.  
> Only run it in an isolated environment (VM/container), keep secrets/credentials off the machine, and assume generated code may be incorrect, insecure, or harmful. You are responsible for any outcomes.

## Features

- **LLM-Guided Evolution**: Uses GPT-4 or Claude to suggest and implement strategy improvements
- **Diff-Based Mutations**: Surgical code changes via SEARCH/REPLACE diffs instead of full rewrites
- **Walk-Forward Validation**: Robust out-of-sample testing across multiple time folds
- **Technical Indicators**: 5 seed strategies using common technical indicators
- **Baseline Comparison**: Automatic comparison against Random and Buy-and-Hold strategies
- **Program Database**: AlphaEvolve-style strategy archive with lineage tracking and inspiration sampling
- **Multi-Objective Selection**: Pareto-based and weighted sampling for diverse strategy exploration
- **Evaluation Cascade**: Fast rejection with staged evaluation (syntax, smoke test, single-fold, full walk-forward)
- **Selection Policies**: WeightedSum, GatedMAS, and Pareto policies for multi-metric acceptance
- **Modular Design**: Easy to extend with new strategies and data sources

## Architecture

```
┌─────────────┐     Δ (proposal)     ┌─────────────┐     code      ┌─────────────┐
│   LLM A     │─────────────────────▶│   LLM B     │──────────────▶│  Backtest   │
│  (Analyst)  │                      │   (Coder)   │               │   Engine    │
└─────────────┘                      └─────────────┘               └─────────────┘
      ▲                                                                   │
      │                                                                   │
      └───────────────────── fitness score ◀──────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
uv sync

# Download sample data (S&P 500, 5 years daily)
uv run scripts/download_sample.py

# Set API keys
export OPENAI_API_KEY="your-key-here"

# Run evolution
uv run python -m profit.main --data data/ES_daily.csv --strategy EMACrossover
```

## Download Data

Self-contained scripts to download market data (no manual dependency setup needed):

```bash
# Quick start - download sample S&P 500 data
uv run scripts/download_sample.py

# Yahoo Finance (free, no API key)
uv run scripts/download_yahoo.py --ticker SPY --period 10y

# Alpha Vantage (requires free API key)
uv run scripts/download_alphavantage.py --ticker AAPL --api-key YOUR_KEY

# FRED macroeconomic data
uv run scripts/download_fred.py --series "VIXCLS,FEDFUNDS"
```

See [docs/data-sources.md](docs/data-sources.md) for full documentation.

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

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to OHLCV CSV file |
| `--strategy` | EMACrossover | Seed strategy to evolve |
| `--provider` | openai | Default LLM provider (openai/anthropic) |
| `--model` | gpt-4 | Default LLM model name |
| `--analyst-provider` | (from --provider) | LLM provider for analysis/improvements |
| `--analyst-model` | (from --model) | LLM model for analysis/improvements |
| `--coder-provider` | (from --provider) | LLM provider for code generation |
| `--coder-model` | (from --model) | LLM model for code generation |
| `--folds` | 5 | Number of walk-forward folds |
| `--capital` | 10000 | Initial capital |
| `--commission` | 0.002 | Commission rate (0.2%) |
| `--output-dir` | evolved_strategies | Directory to save evolved strategies |
| `--db-backend` | json | Program database backend (json/sqlite) |
| `--db-path` | program_db | Path for program database |
| `--no-inspirations` | False | Disable inspiration sampling from database |
| `--no-diffs` | False | Disable diff-based mutations (use full rewrites) |
| `--diff-mode` | adaptive | When to use diffs: always, never, or adaptive |
| `--diff-match` | tolerant | Diff matching: strict (literal) or tolerant |
| `--exploration-gens` | 5 | In adaptive mode, use rewrites for first N gens |
| `--selection-policy` | (none) | Selection policy: weighted, gated, or pareto |
| `--min-return` | 0.0 | Minimum annualized return threshold (gated policy) |
| `--min-sharpe` | 0.0 | Minimum Sharpe ratio threshold (gated policy) |
| `--max-drawdown` | -50.0 | Maximum drawdown threshold (gated policy) |
| `--min-trades` | 1 | Minimum number of trades (gated policy) |
| `--gate-min-trades` | 1 | Promotion gate: minimum trades before policy check |
| `--gate-max-drawdown` | -80.0 | Promotion gate: max drawdown limit |
| `--gate-min-sharpe` | (none) | Promotion gate: minimum Sharpe ratio |
| `--gate-min-win-rate` | (none) | Promotion gate: minimum win rate (%) |

### Dual-Model Configuration

Use different LLMs for analysis vs coding to optimize each role:

```bash
# Use GPT-4 for analysis, Claude Sonnet for coding
uv run python -m profit.main --data data/ES_daily.csv --strategy EMACrossover \
    --analyst-provider openai --analyst-model gpt-4 \
    --coder-provider anthropic --coder-model claude-sonnet-4-20250514
```

### Diff-Based Mutations

By default, ProFiT uses an adaptive approach to code mutations:
- **Early generations (1-5)**: Full code rewrites to explore new structures
- **Later generations (6+)**: Surgical diffs to fine-tune working strategies

Strategies include EVOLVE-BLOCK markers that define safe mutation regions:
- `indicator_params` - Tunable parameters (periods, thresholds)
- `signal_generation` - Indicator calculations
- `entry_logic` - Entry conditions
- `exit_logic` - Exit conditions

```bash
# Always use diff-based mutations
uv run python -m profit.main --data data/ES_daily.csv --diff-mode always

# Disable diffs (full rewrites only)
uv run python -m profit.main --data data/ES_daily.csv --no-diffs

# Custom exploration period (10 generations before switching to diffs)
uv run python -m profit.main --data data/ES_daily.csv --exploration-gens 10
```

### Multi-Metric Evaluation

ProFiT uses a two-stage evaluation process:

1. **Evaluation Cascade** - Fast rejection of invalid strategies:
   - Syntax check (~1ms) - Parse and compile code
   - Smoke test (~1s) - Quick backtest on 3 months of data
   - Single-fold evaluation (~10s) - Full backtest + **promotion gate**

2. **Selection Policy** - Multi-objective acceptance for strategies that pass the cascade

The **promotion gate** (`--gate-*` arguments) filters out junk strategies early, before the selection policy runs. This saves compute and prevents strategies with 0 trades or extreme drawdowns from polluting the population.

**Selection Policies:**

- **GatedMAS** (`--selection-policy gated`): Multi-gate acceptance requiring strategies to pass all thresholds (return, Sharpe, drawdown, trades) AND beat the baseline
- **WeightedSum** (`--selection-policy weighted`): Weighted combination of metrics with baseline-relative normalization
- **Pareto** (`--selection-policy pareto`): Accept non-dominated strategies for diverse Pareto frontier exploration

```bash
# Use Pareto policy with promotion gate thresholds
uv run python -m profit.main --data data/ES_daily.csv \
    --selection-policy pareto \
    --gate-min-sharpe -2.0 --gate-min-win-rate 20.0

# Use gated policy with custom thresholds
uv run python -m profit.main --data data/ES_daily.csv \
    --selection-policy gated --min-sharpe 0.5 --max-drawdown -30

# Use weighted policy with custom weights
uv run python -m profit.main --data data/ES_daily.csv \
    --selection-policy weighted --w-return 0.4 --w-sharpe 0.4 --w-drawdown 0.2

# Use Pareto policy for diverse exploration
uv run python -m profit.main --data data/ES_daily.csv \
    --selection-policy pareto --pareto-objectives ann_return sharpe max_drawdown
```

## Data Format

CSV with datetime index and OHLCV columns:

```csv
Datetime,Open,High,Low,Close,Volume
2020-01-01 00:00:00,100.0,101.5,99.5,101.0,10000
2020-01-01 01:00:00,101.0,102.0,100.5,101.5,12000
```

## Documentation

- **[docs/installation.md](docs/installation.md)** - Installation guide
- **[docs/usage.md](docs/usage.md)** - Usage examples and API reference
- **[docs/configuration.md](docs/configuration.md)** - Configuration reference
- **[docs/data-sources.md](docs/data-sources.md)** - Where to download example data
- **[docs/alternative-data.md](docs/alternative-data.md)** - Alternative data integration
- **[ROADMAP.md](ROADMAP.md)** - Implementation roadmap with phase overview
- **[specs/](specs/)** - Detailed specifications for each implementation phase
- **[CLAUDE.md](CLAUDE.md)** - Guidance for Claude Code

## Project Structure

```
profit/
├── src/profit/
│   ├── strategies.py      # Seed and baseline strategies (with EVOLVE markers)
│   ├── llm_interface.py   # LLM client for mutations and diffs
│   ├── evolver.py         # Evolutionary engine with adaptive diff support
│   ├── program_db.py      # Program database for strategy storage
│   ├── diff_utils.py      # Diff parsing, application, and validation
│   ├── evaluation.py      # Multi-metric evaluation cascade and policies
│   └── main.py            # CLI entry point
├── scripts/               # Data download and utility scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
├── specs/                 # Implementation specifications
├── ROADMAP.md             # Implementation roadmap
└── README.md              # This file
```

## Requirements

- Python 3.12+
- uv package manager
- OpenAI API key and/or Anthropic API key

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Contributing

Contributions are welcome — issues and pull requests are appreciated.

By submitting a contribution (including code, documentation, tests, or other materials) to this repository, you agree that your contribution will be licensed under the **MIT License**, and may be redistributed and modified under the same terms as the rest of the project.

If you do not have the right to submit the work under the MIT License (for example, due to employer or third‑party restrictions), please do not contribute that material.

