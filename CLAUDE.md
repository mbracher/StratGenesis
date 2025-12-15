# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProFiT (Program Search for Financial Trading) is a framework for automated discovery and continual improvement of trading strategies using LLMs within an evolutionary loop. The system uses LLM-guided code mutation and walk-forward validation to evolve strategies that adapt to changing market conditions.

## Build and Run Commands

This project uses `uv` as the package manager (Python 3.12).

```bash
# Install dependencies
uv sync

# Run Python scripts
uv run python <script.py>

# Run the main entry point
uv run python -m profit.main --data <data.csv> --strategy EMACrossover

# Run tests
uv run pytest

# Add dependencies
uv add <package>
```

## Project Structure

```
profit/
├── src/profit/
│   ├── __init__.py
│   ├── strategies.py      # Seed and baseline strategies
│   ├── llm_interface.py   # LLM client for mutations
│   ├── evolver.py         # Evolutionary engine
│   └── main.py            # CLI entry point
├── scripts/               # Data download utilities (PEP 723 inline deps)
├── tests/                 # Test suite
├── specs/                 # Phase specifications (implementation details)
├── ROADMAP.md             # Implementation roadmap
└── README.md              # Project documentation
```

## Specifications

Detailed implementation specs are in the `specs/` directory:

| Spec | Description |
|------|-------------|
| `phase-01-project-setup.md` | Dependencies and project structure |
| `phase-02-seed-strategies.md` | 5 seed strategy implementations |
| `phase-03-baseline-strategies.md` | Random and Buy-and-Hold baselines |
| `phase-04-llm-interface.md` | LLMClient class and prompts |
| `phase-05-backtesting-utilities.md` | Backtest and data splitting |
| `phase-06-evolutionary-engine.md` | Core evolution loop with MAS |
| `phase-07-walk-forward-optimization.md` | Multi-fold optimization |
| `phase-08-main-entry-point.md` | CLI implementation |
| `phase-09-testing.md` | Test structure and cases |
| `phase-10-documentation.md` | Documentation and extensions |
| `phase-11-strategy-persistence.md` | Strategy persistence to disk |
| `phase-12-dual-model-llm.md` | Dual-model LLM configuration |
| `phase-13-program-database.md` | AlphaEvolve-style program database |

**Always consult the relevant spec file before implementing a phase.**

## Architecture

### Core Modules

- **strategies.py** - Trading strategy classes as `backtesting.Strategy` subclasses
  - Five seed strategies: BollingerMeanReversion, CCIStrategy, EMACrossover, MACDStrategy, WilliamsRStrategy
  - Two baselines: RandomStrategy, BuyAndHoldStrategy

- **llm_interface.py** - LLM integration for strategy mutation
  - `LLMClient` class supporting OpenAI GPT and Anthropic Claude
  - `generate_improvement()` - LLM A: analyzes strategy and proposes improvements
  - `generate_strategy_code()` - LLM B: applies improvements to code
  - `fix_code()` - iterative code repair (up to 10 attempts)

- **evolver.py** - Evolutionary search engine
  - `ProfitEvolver` class orchestrating the evolution loop
  - `prepare_folds()` - walk-forward data splitting (5 folds: 2.5yr train, 6mo validation, 6mo test)
  - `run_backtest()` - backtesting utility using backtesting.py
  - `evolve_strategy()` - evolutionary loop with MAS (Minimum Acceptable Score) threshold
  - `walk_forward_optimize()` - full walk-forward optimization across folds

- **program_db.py** - AlphaEvolve-style program database
  - `ProgramDatabase` class for strategy storage and inspiration sampling
  - `StrategyRecord` dataclass with full metadata and lineage
  - `EvaluationContext` for apples-to-apples comparison
  - `JsonFileBackend` for file-based storage (default)
  - `SqliteBackend` for larger databases with efficient queries
  - `sample_inspirations()` - exploitation, exploration, trajectory, map_elites, cross_island, pareto, weighted, and mixed modes
  - `SelectionObjective` dataclass for multi-objective selection
  - `compute_pareto_ranks()`, `compute_weighted_score()`, `passes_thresholds()` helper functions
  - `generate_improvement_with_inspirations()` in LLMClient for richer prompts

- **diff_utils.py** - Diff-based code mutation utilities
  - `DiffBlock`, `EvolveBlock`, `ValidationResult` dataclasses
  - `extract_evolve_blocks()` - Parse EVOLVE-BLOCK markers with character offsets
  - `parse_diff_response()` - Parse LLM SEARCH/REPLACE diff format
  - `apply_diff()` - Apply diffs with single-replace and ambiguity detection
  - `validate_modified_code()` - Security validation (imports, dangerous patterns)
  - `MatchMode` enum - STRICT or TOLERANT matching
  - `generate_diff()`, `fix_diff()`, `generate_strategy_code_with_fallback()` in LLMClient

- **main.py** - Entry point and CLI

### Key Dependencies

- `backtesting` - Strategy simulation engine
- `pandas`, `numpy` - Data handling
- `openai`, `anthropic` - LLM API clients

### Evolutionary Loop Algorithm

```
1. Initialize population with seed strategy, compute baseline performance P0
2. Set MAS threshold = P0
3. For each generation:
   - Select parent strategy from population
   - LLM A proposes improvement
   - LLM B generates modified code
   - Repair loop if code fails (up to 10 attempts)
   - Evaluate on validation data
   - Accept if performance >= MAS
4. Return best strategy from population
```

### Configuration Defaults

| Parameter | Value |
|-----------|-------|
| Initial capital | $10,000 |
| Commission | 0.2% |
| Walk-forward folds | 5 |
| Training period | 2.5 years |
| Validation period | 6 months |
| Test period | 6 months |
| Gap between periods | 10 days |
| Max evolution iterations | 15 |
| Max code repair attempts | 10 |
