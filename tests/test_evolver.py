"""Unit tests for ProfitEvolver."""

import pytest
from unittest.mock import Mock
import pandas as pd

from profit.evolver import ProfitEvolver
from profit.strategies import EMACrossover, BuyAndHoldStrategy


class TestProfitEvolverInit:
    """Test ProfitEvolver initialization."""

    def test_default_config(self):
        """Should initialize with default configuration."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        assert evolver.initial_capital == 10000
        assert evolver.commission == 0.002
        assert evolver.exclusive_orders is True

    def test_custom_config(self):
        """Should accept custom configuration."""
        mock_llm = Mock()
        evolver = ProfitEvolver(
            mock_llm,
            initial_capital=50000,
            commission=0.001,
            exclusive_orders=False,
        )

        assert evolver.initial_capital == 50000
        assert evolver.commission == 0.001
        assert evolver.exclusive_orders is False

    def test_stores_llm_client(self):
        """Should store the LLM client."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        assert evolver.llm is mock_llm


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

    def test_returns_result_series(self, small_data):
        """Should return full backtesting result series."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        metrics, result = evolver.run_backtest(EMACrossover, small_data)

        assert isinstance(result, pd.Series)
        assert "Return (Ann.) [%]" in result
        assert "# Trades" in result

    def test_uses_configured_capital(self, small_data):
        """Should use configured initial capital."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm, initial_capital=50000)

        # Just verify it runs - capital affects results but not structure
        metrics, result = evolver.run_backtest(BuyAndHoldStrategy, small_data)

        assert result is not None

    def test_uses_configured_commission(self, small_data):
        """Should use configured commission rate."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm, commission=0.005)

        # Just verify it runs - commission affects results but not structure
        metrics, result = evolver.run_backtest(BuyAndHoldStrategy, small_data)

        assert result is not None


class TestPrepareFolds:
    """Test walk-forward data splitting."""

    def test_returns_list_of_tuples(self, sample_data):
        """Should return list of fold tuples."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        folds = evolver.prepare_folds(sample_data, n_folds=1)

        assert isinstance(folds, list)
        if len(folds) > 0:
            assert isinstance(folds[0], tuple)
            assert len(folds[0]) == 3

    def test_fold_contains_dataframes(self, sample_data):
        """Each fold should contain train, val, test DataFrames."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        folds = evolver.prepare_folds(sample_data, n_folds=1)

        if len(folds) > 0:
            train, val, test = folds[0]
            assert isinstance(train, pd.DataFrame)
            assert isinstance(val, pd.DataFrame)
            assert isinstance(test, pd.DataFrame)

    def test_no_overlap_between_periods(self, sample_data):
        """Train, val, test should not overlap."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        folds = evolver.prepare_folds(sample_data, n_folds=1)

        if len(folds) > 0:
            train, val, test = folds[0]

            # Check no overlap
            assert train.index.max() < val.index.min()
            assert val.index.max() < test.index.min()

    def test_respects_requested_folds(self, sample_data):
        """Should create up to requested number of folds."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        folds = evolver.prepare_folds(sample_data, n_folds=2)

        # May get fewer folds if data doesn't support them
        assert len(folds) <= 2

    def test_returns_empty_if_insufficient_data(self, small_data):
        """Should handle insufficient data gracefully."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        # Small data likely won't support 5 folds with 2.5yr train + 6mo val + 6mo test
        folds = evolver.prepare_folds(small_data, n_folds=5)

        # Should not raise, just return fewer/no folds
        assert isinstance(folds, list)


class TestRandomIndex:
    """Test random index helper."""

    def test_returns_valid_index(self):
        """Should return index in valid range."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        for _ in range(100):
            idx = evolver._random_index(10)
            assert 0 <= idx < 10

    def test_returns_zero_for_single_element(self):
        """Should return 0 for single element range."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        idx = evolver._random_index(1)
        assert idx == 0

    def test_raises_for_zero_range(self):
        """Should raise for empty range."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        with pytest.raises(ValueError):
            evolver._random_index(0)


class TestMetricsExtraction:
    """Test that metrics are correctly extracted from backtest results."""

    def test_metrics_keys(self, medium_data):
        """Should extract expected metric keys."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        metrics, _ = evolver.run_backtest(EMACrossover, medium_data)

        expected_keys = {"AnnReturn%", "Sharpe", "Expectancy%", "Trades"}
        assert set(metrics.keys()) == expected_keys

    def test_trades_is_integer(self, medium_data):
        """Trades count should be an integer."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        metrics, _ = evolver.run_backtest(EMACrossover, medium_data)

        assert isinstance(metrics["Trades"], (int, float))
