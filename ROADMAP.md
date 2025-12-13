# ProFiT Implementation Roadmap

This roadmap outlines incremental steps for implementing the ProFiT (Program Search for Financial Trading) framework.

**Detailed specifications for each phase are in the `specs/` directory.**

---

## Phase Overview

| Phase | Goal | Spec File | Status |
|-------|------|-----------|--------|
| 1 | Project Setup & Dependencies | [`specs/phase-01-project-setup.md`](specs/phase-01-project-setup.md) | ✅ |
| 2 | Seed Strategies | [`specs/phase-02-seed-strategies.md`](specs/phase-02-seed-strategies.md) | ✅ |
| 3 | Baseline Strategies | [`specs/phase-03-baseline-strategies.md`](specs/phase-03-baseline-strategies.md) | ✅ |
| 4 | LLM Interface | [`specs/phase-04-llm-interface.md`](specs/phase-04-llm-interface.md) | ✅ |
| 5 | Backtesting Utilities | [`specs/phase-05-backtesting-utilities.md`](specs/phase-05-backtesting-utilities.md) | ✅ |
| 6 | Evolutionary Engine | [`specs/phase-06-evolutionary-engine.md`](specs/phase-06-evolutionary-engine.md) | ⬜ |
| 7 | Walk-Forward Optimization | [`specs/phase-07-walk-forward-optimization.md`](specs/phase-07-walk-forward-optimization.md) | ⬜ |
| 8 | Main Entry Point | [`specs/phase-08-main-entry-point.md`](specs/phase-08-main-entry-point.md) | ⬜ |
| 9 | Testing & Validation | [`specs/phase-09-testing.md`](specs/phase-09-testing.md) | ⬜ |
| 10 | Documentation & Extensions | [`specs/phase-10-documentation.md`](specs/phase-10-documentation.md) | ⬜ |

---

## Phase 1: Project Setup & Dependencies

**Spec:** [`specs/phase-01-project-setup.md`](specs/phase-01-project-setup.md)

- [x] Update `pyproject.toml` with dependencies (backtesting, pandas, numpy, openai, anthropic)
- [x] Create source directory structure (`src/profit/`)
- [x] Add `.env.example` for API key configuration
- [x] Verify setup with `uv sync`

---

## Phase 2: Seed Strategies

**Spec:** [`specs/phase-02-seed-strategies.md`](specs/phase-02-seed-strategies.md)

**File:** `src/profit/strategies.py`

- [x] BollingerMeanReversion (20-period Bollinger Bands)
- [x] CCIStrategy (Commodity Channel Index)
- [x] EMACrossover (50/200 EMA crossover)
- [x] MACDStrategy (12-26-9 MACD)
- [x] WilliamsRStrategy (14-period Williams %R)

---

## Phase 3: Baseline Strategies

**Spec:** [`specs/phase-03-baseline-strategies.md`](specs/phase-03-baseline-strategies.md)

**File:** `src/profit/strategies.py`

- [x] RandomStrategy (random trading baseline)
- [x] BuyAndHoldStrategy (passive long benchmark)

---

## Phase 4: LLM Interface

**Spec:** [`specs/phase-04-llm-interface.md`](specs/phase-04-llm-interface.md)

**File:** `src/profit/llm_interface.py`

- [x] LLMClient class (OpenAI/Anthropic support)
- [x] `generate_improvement()` method
- [x] `generate_strategy_code()` method
- [x] `fix_code()` method

---

## Phase 5: Backtesting Utilities

**Spec:** [`specs/phase-05-backtesting-utilities.md`](specs/phase-05-backtesting-utilities.md)

**File:** `src/profit/evolver.py`

- [x] ProfitEvolver class initialization
- [x] `run_backtest()` method
- [x] `prepare_folds()` method (walk-forward splitting)

---

## Phase 6: Evolutionary Engine

**Spec:** [`specs/phase-06-evolutionary-engine.md`](specs/phase-06-evolutionary-engine.md)

**File:** `src/profit/evolver.py`

- [ ] `evolve_strategy()` method
- [ ] MAS threshold logic
- [ ] Population management
- [ ] Code compilation and repair loop

---

## Phase 7: Walk-Forward Optimization

**Spec:** [`specs/phase-07-walk-forward-optimization.md`](specs/phase-07-walk-forward-optimization.md)

**File:** `src/profit/evolver.py`

- [ ] `walk_forward_optimize()` method
- [ ] Baseline comparison
- [ ] Results aggregation

---

## Phase 8: Main Entry Point

**Spec:** [`specs/phase-08-main-entry-point.md`](specs/phase-08-main-entry-point.md)

**File:** `src/profit/main.py`

- [ ] Data loading
- [ ] CLI argument parsing
- [ ] Results output

---

## Phase 9: Testing & Validation

**Spec:** [`specs/phase-09-testing.md`](specs/phase-09-testing.md)

**Directory:** `tests/`

- [ ] Strategy unit tests
- [ ] LLM interface tests (mocked)
- [ ] Evolver unit tests
- [ ] Integration tests
- [ ] Sample data fixtures

---

## Phase 10: Documentation & Extensions

**Spec:** [`specs/phase-10-documentation.md`](specs/phase-10-documentation.md)

- [ ] Installation guide
- [ ] Usage documentation
- [ ] Alternative data integration guide

---

## Quick Reference

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| Strategies | `src/profit/strategies.py` | 5 seed + 2 baseline strategies |
| LLM Client | `src/profit/llm_interface.py` | `LLMClient` |
| Evolution | `src/profit/evolver.py` | `ProfitEvolver` |
| CLI | `src/profit/main.py` | Entry point |
| Tests | `tests/` | pytest suite |

## Configuration Defaults

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
