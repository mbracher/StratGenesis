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

## Troubleshooting

### Common Issues

**ImportError: No module named 'profit'**
- Ensure you're running with `uv run python` and not plain `python`
- Verify `uv sync` completed successfully

**API Key Errors**
- Check that environment variables are set: `echo $OPENAI_API_KEY`
- Verify key format (OpenAI starts with `sk-`, Anthropic with `sk-ant-`)

**backtesting module errors**
- Run `uv sync` to ensure all dependencies are installed
- Check Python version: `python --version` (must be 3.12+)
