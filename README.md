# ProFiT: LLM-Driven Evolutionary Trading System

ProFiT (Program Search for Financial Trading) is a framework for automated discovery and continual improvement of trading strategies using large language models (LLMs) within an evolutionary loop.

## Features

- **LLM-Guided Evolution**: Uses GPT-4 or Claude to suggest and implement strategy improvements
- **Diff-Based Mutations**: Surgical code changes via SEARCH/REPLACE diffs instead of full rewrites
- **Walk-Forward Validation**: Robust out-of-sample testing across multiple time folds
- **Technical Indicators**: 5 seed strategies using common technical indicators
- **Baseline Comparison**: Automatic comparison against Random and Buy-and-Hold strategies
- **Program Database**: AlphaEvolve-style strategy archive with lineage tracking and inspiration sampling
- **Multi-Objective Selection**: Pareto-based and weighted sampling for diverse strategy exploration
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

MIT
