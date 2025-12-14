# Phase 1: Project Setup & Dependencies

## Objective

Configure the development environment, project structure, and install all required dependencies for the ProFiT framework.

## Dependencies

From the design document:

> Installation requirements (via pip): `backtesting`, `pandas`, `numpy`, `openai`, and `anthropic` (for Claude API). You will need API keys for OpenAI and Anthropic (if using Claude).

### Required Packages

| Package | Purpose |
|---------|---------|
| `backtesting` | Strategy simulation engine |
| `pandas` | Data manipulation and time series |
| `numpy` | Numerical operations |
| `openai` | OpenAI GPT API client |
| `anthropic` | Anthropic Claude API client |

## Project Structure

```
profit/
├── src/
│   └── profit/
│       ├── __init__.py
│       ├── strategies.py      # Phase 2-3
│       ├── llm_interface.py   # Phase 4
│       ├── evolver.py         # Phase 5-7
│       └── main.py            # Phase 8
├── tests/
│   └── ...                    # Phase 9
├── specs/
│   └── ...                    # Specifications
├── doc/
│   └── ...                    # Design documents
├── pyproject.toml
├── CLAUDE.md
├── ROADMAP.md
└── README.md
```

## pyproject.toml Configuration

```toml
[project]
name = "profit"
version = "0.1.0"
description = "ProFiT: LLM-Driven Evolutionary Trading System"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "backtesting",
    "pandas",
    "numpy",
    "openai",
    "anthropic",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
]
```

## Environment Configuration

API keys should be loaded from environment variables:

```python
# From llm_interface.py design
self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
```

### .env.example

```
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

## Backtesting Configuration Defaults

From the design document:

> We ensure these strategies use a consistent initial capital (e.g. $10,000) and commission (0.2%) as per the paper's setup.

```python
initial_capital = 10000
commission = 0.002  # 0.2%
exclusive_orders = True  # No overlapping long/short positions
```

## Verification

After setup, verify with:

```bash
uv sync
uv run python -c "from backtesting import Strategy; print('Setup complete')"
```

## Deliverables

- [x] Updated `pyproject.toml` with all dependencies
- [x] Source directory structure created
- [x] `.env.example` file for API configuration
- [x] Basic `__init__.py` files in place
- [x] Successful `uv sync`
