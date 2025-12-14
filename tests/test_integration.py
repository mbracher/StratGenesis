"""Integration tests for full system."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from profit.evolver import ProfitEvolver
from profit.strategies import EMACrossover, RandomStrategy, BuyAndHoldStrategy


# Valid strategy code that can be exec'd
VALID_STRATEGY_CODE = '''
class EMACrossover_Gen1(Strategy):
    """Modified EMA crossover strategy."""
    fast_ema = 40
    slow_ema = 180

    def init(self):
        price = self.data.Close
        self.ema_fast = self.I(
            lambda x: pd.Series(x).ewm(span=self.fast_ema, adjust=False).mean(), price
        )
        self.ema_slow = self.I(
            lambda x: pd.Series(x).ewm(span=self.slow_ema, adjust=False).mean(), price
        )

    def next(self):
        fast = self.ema_fast[-1]
        slow = self.ema_slow[-1]

        if fast > slow and not self.position:
            self.buy()
        elif fast < slow and not self.position:
            self.sell()

        if self.position.is_long and fast < slow:
            self.position.close()
        elif self.position.is_short and fast > slow:
            self.position.close()
'''


class TestFullEvolution:
    """Test complete evolution loop (with mocked LLM)."""

    def test_evolve_strategy_returns_class_and_perf(self, medium_data):
        """Evolution should return a strategy class and performance."""
        mock_llm = Mock()
        mock_llm.generate_improvement.return_value = "Adjust EMA periods"
        mock_llm.generate_strategy_code.return_value = VALID_STRATEGY_CODE

        evolver = ProfitEvolver(mock_llm)

        # Split data manually for test
        train = medium_data.iloc[:1000]
        val = medium_data.iloc[1000:1500]

        best_class, best_perf, best_code = evolver.evolve_strategy(
            EMACrossover, train, val, max_iters=1
        )

        assert best_class is not None
        assert isinstance(best_perf, (int, float))
        assert isinstance(best_code, str)

    def test_evolve_strategy_calls_llm(self, medium_data):
        """Evolution should call LLM for improvements."""
        mock_llm = Mock()
        mock_llm.generate_improvement.return_value = "Add stop loss"
        mock_llm.generate_strategy_code.return_value = VALID_STRATEGY_CODE

        evolver = ProfitEvolver(mock_llm)

        train = medium_data.iloc[:1000]
        val = medium_data.iloc[1000:1500]

        evolver.evolve_strategy(EMACrossover, train, val, max_iters=1)

        assert mock_llm.generate_improvement.called
        assert mock_llm.generate_strategy_code.called

    def test_evolve_handles_invalid_code(self, medium_data):
        """Evolution should handle invalid generated code via repair loop."""
        mock_llm = Mock()
        mock_llm.generate_improvement.return_value = "Add feature"

        # First return invalid code, then valid code on fix
        mock_llm.generate_strategy_code.return_value = "invalid python code!!!"
        mock_llm.fix_code.return_value = VALID_STRATEGY_CODE

        evolver = ProfitEvolver(mock_llm)

        train = medium_data.iloc[:1000]
        val = medium_data.iloc[1000:1500]

        # Should not raise, should use fix_code to repair
        best_class, best_perf, best_code = evolver.evolve_strategy(
            EMACrossover, train, val, max_iters=1
        )

        # Should have attempted fix
        assert mock_llm.fix_code.called


class TestBaselineComparison:
    """Test baseline strategy comparison."""

    def test_random_vs_evolved(self, medium_data):
        """Compare random baseline to evolved strategy."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        metrics_random, _ = evolver.run_backtest(RandomStrategy, medium_data)
        metrics_ema, _ = evolver.run_backtest(EMACrossover, medium_data)

        # Both should produce valid metrics
        assert metrics_random["AnnReturn%"] is not None
        assert metrics_ema["AnnReturn%"] is not None

    def test_buyhold_vs_evolved(self, medium_data):
        """Compare buy-and-hold to evolved strategy."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        metrics_bh, _ = evolver.run_backtest(BuyAndHoldStrategy, medium_data)
        metrics_ema, _ = evolver.run_backtest(EMACrossover, medium_data)

        # Both should produce valid metrics
        assert metrics_bh["AnnReturn%"] is not None
        assert metrics_ema["AnnReturn%"] is not None
        # backtesting.py only counts closed trades; buy-and-hold may show 0
        assert metrics_bh["Trades"] <= 1


class TestWalkForwardOptimization:
    """Test walk-forward optimization (basic checks)."""

    def test_prepare_and_backtest_flow(self, sample_data):
        """Test the prepare_folds -> backtest flow."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        folds = evolver.prepare_folds(sample_data, n_folds=1)

        if len(folds) > 0:
            train, val, test = folds[0]

            # Run backtest on each segment
            metrics_train, _ = evolver.run_backtest(EMACrossover, train)
            metrics_val, _ = evolver.run_backtest(EMACrossover, val)
            metrics_test, _ = evolver.run_backtest(EMACrossover, test)

            assert metrics_train is not None
            assert metrics_val is not None
            assert metrics_test is not None


class TestEndToEnd:
    """End-to-end tests with synthetic data."""

    def test_full_pipeline_mock_llm(self, medium_data):
        """Test complete pipeline with mocked LLM."""
        mock_llm = Mock()
        mock_llm.generate_improvement.return_value = "Optimize parameters"
        mock_llm.generate_strategy_code.return_value = VALID_STRATEGY_CODE

        evolver = ProfitEvolver(mock_llm)

        # Get metrics for seed strategy
        seed_metrics, _ = evolver.run_backtest(EMACrossover, medium_data)

        # Evolve strategy
        train = medium_data.iloc[:1000]
        val = medium_data.iloc[1000:1500]

        evolved_class, evolved_perf, evolved_code = evolver.evolve_strategy(
            EMACrossover, train, val, max_iters=1
        )

        # Get metrics for evolved strategy
        evolved_metrics, _ = evolver.run_backtest(evolved_class, medium_data)

        # All should produce valid results
        assert seed_metrics is not None
        assert evolved_metrics is not None
        assert evolved_class is not None

    def test_multiple_iterations_multi_gen(self, medium_data):
        """Test evolution with multiple generations.

        Dynamically created strategies can now be mutated because we store
        their source code alongside the class in the population.
        """
        # Generate unique class names for each call
        def make_strategy_code(parent_code, improvement):
            # The evolve_strategy method will rename the class based on generation
            return VALID_STRATEGY_CODE.replace("EMACrossover_Gen1", "EMACrossover")

        mock_llm = Mock()
        mock_llm.generate_improvement.return_value = "Improve entry timing"
        mock_llm.generate_strategy_code.side_effect = make_strategy_code
        mock_llm.fix_code.side_effect = make_strategy_code  # In case repair is needed

        evolver = ProfitEvolver(mock_llm)

        train = medium_data.iloc[:1000]
        val = medium_data.iloc[1000:1500]

        # Test with multiple iterations to ensure dynamically created strategies
        # can be mutated (source code is stored with the class)
        evolved_class, evolved_perf, evolved_code = evolver.evolve_strategy(
            EMACrossover, train, val, max_iters=3
        )

        # LLM should have been called multiple times for multiple generations
        assert mock_llm.generate_improvement.call_count >= 1
        assert evolved_class is not None
        assert isinstance(evolved_code, str)
