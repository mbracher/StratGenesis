# ProFiT: LLM-Driven Evolutionary Trading System

ProFiT (Program Search for Financial Trading) is a framework for automated discovery and continual improvement of trading strategies using large language models (LLMs) within an evolutionary loop.

## Features

- **LLM-Guided Evolution**: Uses GPT-4 or Claude to suggest and implement strategy improvements
- **Walk-Forward Validation**: Robust out-of-sample testing across multiple time folds
- **Technical Indicators**: 5 seed strategies using common technical indicators
- **Baseline Comparison**: Automatic comparison against Random and Buy-and-Hold strategies
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

# Set API keys
export OPENAI_API_KEY="your-key-here"

# Run evolution
uv run python -m profit.main --data data/ES_hourly.csv --strategy EMACrossover
```

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
| `--provider` | openai | LLM provider (openai/anthropic) |
| `--model` | gpt-4 | LLM model name |
| `--folds` | 5 | Number of walk-forward folds |
| `--capital` | 10000 | Initial capital |
| `--commission` | 0.002 | Commission rate (0.2%) |

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
│   ├── strategies.py      # Seed and baseline strategies
│   ├── llm_interface.py   # LLM client for mutations
│   ├── evolver.py         # Evolutionary engine
│   └── main.py            # CLI entry point
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
