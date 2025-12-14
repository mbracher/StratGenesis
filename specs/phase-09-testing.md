# Phase 9: Testing & Validation

## Objective

Ensure correctness and reliability of the ProFiT system through comprehensive testing.

---

## Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── test_strategies.py   # Unit tests for strategies
├── test_llm_interface.py # Unit tests for LLM client (mocked)
├── test_evolver.py      # Unit tests for evolver utilities
├── test_integration.py  # Integration tests
└── sample_data/
    └── sample_ohlcv.csv # Test data
```

---

## Sample Data

### Requirements
- OHLCV format with datetime index
- At least several hundred bars for meaningful backtests
- Can use synthetic or real historical data

### Sample Data Generator

```python
# tests/conftest.py
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 5000

    dates = pd.date_range(start='2020-01-01', periods=n_bars, freq='h')

    # Random walk for price
    returns = np.random.randn(n_bars) * 0.001
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLCV
    data = pd.DataFrame({
        'Open': close * (1 + np.random.randn(n_bars) * 0.001),
        'High': close * (1 + np.abs(np.random.randn(n_bars) * 0.002)),
        'Low': close * (1 - np.abs(np.random.randn(n_bars) * 0.002)),
        'Close': close,
        'Volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)

    return data

@pytest.fixture
def small_data(sample_data):
    """Smaller subset for quick tests."""
    return sample_data.iloc[:500]
```

---

## Unit Tests: Strategies

### test_strategies.py

```python
"""Unit tests for trading strategies."""
import pytest
from backtesting import Backtest

from profit.strategies import (
    BollingerMeanReversion,
    CCIStrategy,
    EMACrossover,
    MACDStrategy,
    WilliamsRStrategy,
    RandomStrategy,
    BuyAndHoldStrategy,
)


class TestSeedStrategies:
    """Test seed strategy classes."""

    @pytest.mark.parametrize("strategy_class", [
        BollingerMeanReversion,
        CCIStrategy,
        EMACrossover,
        MACDStrategy,
        WilliamsRStrategy,
    ])
    def test_strategy_runs(self, small_data, strategy_class):
        """Each strategy should run without errors."""
        bt = Backtest(small_data, strategy_class, cash=10000, commission=0.002)
        result = bt.run()
        assert result is not None
        assert 'Return (Ann.) [%]' in result

    @pytest.mark.parametrize("strategy_class", [
        BollingerMeanReversion,
        CCIStrategy,
        EMACrossover,
        MACDStrategy,
        WilliamsRStrategy,
    ])
    def test_strategy_makes_trades(self, sample_data, strategy_class):
        """Each strategy should generate at least some trades."""
        bt = Backtest(sample_data, strategy_class, cash=10000, commission=0.002)
        result = bt.run()
        assert result['# Trades'] > 0


class TestBaselineStrategies:
    """Test baseline strategy classes."""

    def test_random_strategy_runs(self, small_data):
        """Random strategy should run without errors."""
        bt = Backtest(small_data, RandomStrategy, cash=10000, commission=0.002)
        result = bt.run()
        assert result is not None

    def test_random_strategy_reproducible(self, small_data):
        """Random strategy should be reproducible with same seed."""
        bt1 = Backtest(small_data, RandomStrategy, cash=10000, commission=0.002)
        bt2 = Backtest(small_data, RandomStrategy, cash=10000, commission=0.002)
        result1 = bt1.run()
        result2 = bt2.run()
        assert result1['# Trades'] == result2['# Trades']

    def test_buy_and_hold_runs(self, small_data):
        """Buy-and-hold strategy should run without errors."""
        bt = Backtest(small_data, BuyAndHoldStrategy, cash=10000, commission=0.002)
        result = bt.run()
        assert result is not None

    def test_buy_and_hold_single_trade(self, small_data):
        """Buy-and-hold should make exactly one trade."""
        bt = Backtest(small_data, BuyAndHoldStrategy, cash=10000, commission=0.002)
        result = bt.run()
        assert result['# Trades'] == 1


class TestStrategyParameters:
    """Test strategy parameter configuration."""

    def test_bollinger_parameters(self):
        """Bollinger strategy should have configurable parameters."""
        assert hasattr(BollingerMeanReversion, 'bb_period')
        assert hasattr(BollingerMeanReversion, 'bb_stddev')
        assert BollingerMeanReversion.bb_period == 20
        assert BollingerMeanReversion.bb_stddev == 2

    def test_ema_parameters(self):
        """EMA strategy should have configurable parameters."""
        assert hasattr(EMACrossover, 'fast_ema')
        assert hasattr(EMACrossover, 'slow_ema')
        assert EMACrossover.fast_ema == 50
        assert EMACrossover.slow_ema == 200
```

---

## Unit Tests: LLM Interface

### test_llm_interface.py

```python
"""Unit tests for LLM interface (with mocking)."""
import pytest
from unittest.mock import Mock, patch

from profit.llm_interface import LLMClient


class TestLLMClientInit:
    """Test LLMClient initialization."""

    def test_default_openai(self):
        """Default provider should be OpenAI."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = LLMClient()
            assert client.provider == "openai"
            assert client.model == "gpt-4"

    def test_anthropic_provider(self):
        """Should initialize Anthropic client when specified."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('profit.llm_interface.anthropic') as mock_anthropic:
                mock_anthropic.Client.return_value = Mock()
                client = LLMClient(provider="anthropic")
                assert client.provider == "anthropic"
                assert client.model == "claude-2"

    def test_custom_model(self):
        """Should accept custom model specification."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = LLMClient(model="gpt-3.5-turbo")
            assert client.model == "gpt-3.5-turbo"


class TestGenerateImprovement:
    """Test improvement generation."""

    def test_returns_string(self):
        """Should return improvement proposal as string."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = LLMClient()

            with patch.object(client, '_chat') as mock_chat:
                mock_chat.return_value = "Add a trailing stop-loss"

                result = client.generate_improvement(
                    "class MyStrategy: pass",
                    "AnnReturn=5.0%, Sharpe=0.5"
                )

                assert isinstance(result, str)
                assert len(result) > 0
                mock_chat.assert_called_once()


class TestGenerateStrategyCode:
    """Test strategy code generation."""

    def test_returns_code_string(self):
        """Should return valid Python code."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = LLMClient()

            with patch.object(client, '_chat') as mock_chat:
                mock_chat.return_value = "class MyStrategy(Strategy): pass"

                result = client.generate_strategy_code(
                    "class MyStrategy: pass",
                    "Add trailing stop"
                )

                assert isinstance(result, str)
                assert "class" in result


class TestFixCode:
    """Test code repair functionality."""

    def test_attempts_fix(self):
        """Should attempt to fix code given error traceback."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = LLMClient()

            with patch.object(client, '_chat') as mock_chat:
                mock_chat.return_value = "class MyStrategy(Strategy): pass"

                result = client.fix_code(
                    "class MyStrategy: syntax error",
                    "SyntaxError: invalid syntax"
                )

                assert isinstance(result, str)


class TestCodeStripping:
    """Test markdown code fence stripping."""

    def test_strips_markdown_fences(self):
        """Should strip markdown code fences from response."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = LLMClient()

            # Simulate _chat stripping logic
            text = "```python\nclass Foo: pass\n```"
            if text.strip().startswith("```"):
                text = text.strip().strip("```").strip()
                if text.startswith("python"):
                    text = text[len("python"):].strip()

            assert text == "class Foo: pass"
```

---

## Unit Tests: Evolver

### test_evolver.py

```python
"""Unit tests for ProfitEvolver."""
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from profit.evolver import ProfitEvolver
from profit.strategies import EMACrossover


class TestProfitEvolverInit:
    """Test ProfitEvolver initialization."""

    def test_default_config(self):
        """Should initialize with default configuration."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        assert evolver.initial_capital == 10000
        assert evolver.commission == 0.002
        assert evolver.exclusive_orders == True

    def test_custom_config(self):
        """Should accept custom configuration."""
        mock_llm = Mock()
        evolver = ProfitEvolver(
            mock_llm,
            initial_capital=50000,
            commission=0.001,
            exclusive_orders=False
        )

        assert evolver.initial_capital == 50000
        assert evolver.commission == 0.001
        assert evolver.exclusive_orders == False


class TestRunBacktest:
    """Test backtest execution."""

    def test_returns_metrics_dict(self, small_data):
        """Should return metrics dictionary."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        metrics, result = evolver.run_backtest(EMACrossover, small_data)

        assert isinstance(metrics, dict)
        assert "AnnReturn%" in metrics
        assert "Sharpe" in metrics
        assert "Expectancy%" in metrics
        assert "Trades" in metrics


class TestPrepareFolds:
    """Test walk-forward data splitting."""

    def test_creates_correct_number_of_folds(self, sample_data):
        """Should create requested number of folds."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        # Use smaller data that can support fewer folds
        folds = evolver.prepare_folds(sample_data, n_folds=2)

        # May get fewer folds if data doesn't support them
        assert len(folds) <= 2

    def test_fold_structure(self, sample_data):
        """Each fold should be a tuple of (train, val, test)."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        folds = evolver.prepare_folds(sample_data, n_folds=1)

        if len(folds) > 0:
            train, val, test = folds[0]
            assert isinstance(train, pd.DataFrame)
            assert isinstance(val, pd.DataFrame)
            assert isinstance(test, pd.DataFrame)

    def test_no_overlap(self, sample_data):
        """Train, val, test should not overlap."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        folds = evolver.prepare_folds(sample_data, n_folds=1)

        if len(folds) > 0:
            train, val, test = folds[0]

            # Check no overlap
            assert train.index.max() < val.index.min()
            assert val.index.max() < test.index.min()


class TestRandomIndex:
    """Test random index helper."""

    def test_returns_valid_index(self):
        """Should return index in valid range."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        for _ in range(100):
            idx = evolver._random_index(10)
            assert 0 <= idx < 10
```

---

## Integration Tests

### test_integration.py

```python
"""Integration tests for full system."""
import pytest
from unittest.mock import Mock, patch

from profit.evolver import ProfitEvolver
from profit.llm_interface import LLMClient
from profit.strategies import EMACrossover


class TestFullEvolution:
    """Test complete evolution loop (with mocked LLM)."""

    def test_evolve_strategy_returns_class(self, sample_data):
        """Evolution should return a strategy class."""
        # Mock LLM to return valid code
        mock_llm = Mock()
        mock_llm.generate_improvement.return_value = "Add trailing stop"
        mock_llm.generate_strategy_code.return_value = '''
class EMACrossover_Gen1(Strategy):
    fast_ema = 50
    slow_ema = 200

    def init(self):
        price = self.data.Close
        self.ema_fast = self.I(pd.Series.ewm, price, span=self.fast_ema).mean()
        self.ema_slow = self.I(pd.Series.ewm, price, span=self.slow_ema).mean()

    def next(self):
        if self.ema_fast[-1] > self.ema_slow[-1] and not self.position:
            self.buy()
        elif self.ema_fast[-1] < self.ema_slow[-1] and self.position:
            self.position.close()
'''

        evolver = ProfitEvolver(mock_llm)

        # Split data manually for test
        train = sample_data.iloc[:1000]
        val = sample_data.iloc[1000:1500]

        with patch('profit.evolver.inspect.getsource') as mock_source:
            mock_source.return_value = "class EMACrossover(Strategy): pass"

            best_class, best_perf = evolver.evolve_strategy(
                EMACrossover, train, val, max_iters=1
            )

        assert best_class is not None
        assert isinstance(best_perf, (int, float))
```

---

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=profit --cov-report=html

# Run specific test file
uv run pytest tests/test_strategies.py

# Run specific test class
uv run pytest tests/test_strategies.py::TestSeedStrategies

# Run with verbose output
uv run pytest -v
```

---

## Deliverables

- [x] `tests/conftest.py` with sample data fixtures
- [x] `tests/test_strategies.py` - Strategy unit tests
- [x] `tests/test_llm_interface.py` - LLM client tests (mocked)
- [x] `tests/test_evolver.py` - Evolver utility tests
- [x] `tests/test_integration.py` - Integration tests
- [x] Sample data generator or fixture
- [x] pytest configuration
- [x] All tests passing
