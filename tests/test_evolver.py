"""Unit tests for ProfitEvolver."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import pandas as pd

from profit.evolver import ProfitEvolver, StrategyPersister, load_strategy
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

    def test_output_dir_deprecation_warning(self):
        """Should emit deprecation warning when output_dir is provided."""
        import warnings

        mock_llm = Mock()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evolver = ProfitEvolver(mock_llm, output_dir="test_output")

            # Check that a deprecation warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "program_db" in str(w[0].message).lower()

    def test_no_deprecation_warning_without_output_dir(self):
        """Should not emit warning when output_dir is None (default)."""
        import warnings

        mock_llm = Mock()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evolver = ProfitEvolver(mock_llm)

            # No deprecation warning should be raised
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0


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


class TestExtractStandardMetrics:
    """Test _extract_standard_metrics helper method."""

    def test_extracts_all_standard_metrics(self, medium_data):
        """Should extract all standard metrics from backtest result."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        _, result = evolver.run_backtest(EMACrossover, medium_data)
        metrics = evolver._extract_standard_metrics(result)

        # Check that key standard metrics are present
        expected_keys = {
            "ann_return", "total_return", "sharpe", "sortino", "calmar",
            "max_drawdown", "volatility", "trade_count", "exposure_time",
        }
        # These should always be present (even with few trades)
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_filters_nan_values(self, medium_data):
        """Should filter out NaN values from metrics."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        _, result = evolver.run_backtest(EMACrossover, medium_data)
        metrics = evolver._extract_standard_metrics(result)

        # No values should be NaN
        import math
        for key, value in metrics.items():
            assert not (isinstance(value, float) and math.isnan(value)), \
                f"Metric {key} is NaN"

    def test_converts_duration_to_days(self, medium_data):
        """Should convert avg_holding_period from timedelta to days."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        _, result = evolver.run_backtest(EMACrossover, medium_data)
        metrics = evolver._extract_standard_metrics(result)

        # If avg_holding_period is present, it should be numeric (days)
        if "avg_holding_period" in metrics:
            assert isinstance(metrics["avg_holding_period"], (int, float))


class TestStrategyPersister:
    """Test StrategyPersister class."""

    def test_init_sets_output_dir(self):
        """Should store output directory."""
        persister = StrategyPersister("custom_dir")
        assert persister.output_dir == Path("custom_dir")

    def test_start_run_creates_directory(self):
        """Should create run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StrategyPersister(tmpdir)
            run_dir = persister.start_run(
                "TestStrategy",
                analyst_provider="openai",
                analyst_model="gpt-4",
                coder_provider="openai",
                coder_model="gpt-4",
            )

            assert run_dir.exists()
            assert run_dir.parent == Path(tmpdir)
            assert "run_" in run_dir.name

    def test_start_run_creates_summary_file(self):
        """Should create initial run summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StrategyPersister(tmpdir)
            run_dir = persister.start_run(
                "TestStrategy",
                analyst_provider="openai",
                analyst_model="gpt-4",
                coder_provider="anthropic",
                coder_model="claude-sonnet-4-20250514",
            )

            summary_path = run_dir / "run_summary.json"
            assert summary_path.exists()

            summary = json.loads(summary_path.read_text())
            assert summary["seed_strategy"] == "TestStrategy"
            assert summary["llm_config"]["analyst"]["provider"] == "openai"
            assert summary["llm_config"]["analyst"]["model"] == "gpt-4"
            assert summary["llm_config"]["coder"]["provider"] == "anthropic"
            assert summary["llm_config"]["coder"]["model"] == "claude-sonnet-4-20250514"
            assert summary["folds"] == []

    def test_save_strategy_creates_files(self):
        """Should create .py and .json files for strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StrategyPersister(tmpdir)
            persister.start_run(
                "TestStrategy",
                analyst_provider="openai",
                analyst_model="gpt-4",
                coder_provider="openai",
                coder_model="gpt-4",
            )

            metrics = {"AnnReturn%": 15.5, "Sharpe": 1.2}
            strategy_path = persister.save_strategy(
                strategy_class=EMACrossover,
                source_code="class TestClass(Strategy):\n    pass",
                fold=1,
                generation=3,
                metrics=metrics,
                parent_name="ParentStrategy",
                improvement_proposal="Add RSI filter",
            )

            assert strategy_path.exists()
            assert strategy_path.suffix == ".py"

            # Check metadata file exists
            metadata_path = strategy_path.with_suffix(".json")
            assert metadata_path.exists()

            # Check metadata contents
            metadata = json.loads(metadata_path.read_text())
            assert metadata["class_name"] == "EMACrossover"
            assert metadata["fold"] == 1
            assert metadata["generation"] == 3
            assert metadata["parent_strategy"] == "ParentStrategy"
            assert metadata["improvement_proposal"] == "Add RSI filter"
            assert metadata["metrics"]["AnnReturn%"] == 15.5

    def test_save_fold_best_creates_file(self):
        """Should create best_strategy.py file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StrategyPersister(tmpdir)
            persister.start_run(
                "TestStrategy",
                analyst_provider="openai",
                analyst_model="gpt-4",
                coder_provider="openai",
                coder_model="gpt-4",
            )

            metrics = {"AnnReturn%": 20.0, "Sharpe": 1.5}
            best_path = persister.save_fold_best(
                fold=1,
                strategy_class=EMACrossover,
                source_code="class TestClass(Strategy):\n    pass",
                metrics=metrics,
            )

            assert best_path.exists()
            assert best_path.name == "best_strategy.py"

    def test_finalize_run_updates_summary(self):
        """Should update summary with fold results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StrategyPersister(tmpdir)
            persister.start_run(
                "TestStrategy",
                analyst_provider="openai",
                analyst_model="gpt-4",
                coder_provider="openai",
                coder_model="gpt-4",
            )

            # Create a fold directory with best_strategy.py
            fold_dir = persister.run_dir / "fold_1"
            fold_dir.mkdir()
            (fold_dir / "best_strategy.py").write_text("# test")

            results = [{
                "fold": 1,
                "strategy": EMACrossover,
                "ann_return": 15.0,
                "sharpe": 1.2,
                "expectancy": 2.5,
                "random_return": 5.0,
                "buy_hold_return": 10.0,
            }]

            summary_path = persister.finalize_run(results)

            summary = json.loads(summary_path.read_text())
            assert summary["completed_at"] is not None
            assert len(summary["folds"]) == 1
            assert summary["folds"][0]["ann_return"] == 15.0
            assert summary["avg_ann_return"] == 15.0
            assert summary["best_fold"] == 1

    def test_finalize_run_copies_best_overall(self):
        """Should copy best strategy to root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persister = StrategyPersister(tmpdir)
            persister.start_run(
                "TestStrategy",
                analyst_provider="openai",
                analyst_model="gpt-4",
                coder_provider="openai",
                coder_model="gpt-4",
            )

            # Create fold directories with best strategies
            fold1_dir = persister.run_dir / "fold_1"
            fold1_dir.mkdir()
            (fold1_dir / "best_strategy.py").write_text("# fold 1 best")

            fold2_dir = persister.run_dir / "fold_2"
            fold2_dir.mkdir()
            (fold2_dir / "best_strategy.py").write_text("# fold 2 best")

            results = [
                {"fold": 1, "strategy": EMACrossover, "ann_return": 10.0,
                 "sharpe": 1.0, "expectancy": 2.0, "random_return": 5.0, "buy_hold_return": 8.0},
                {"fold": 2, "strategy": EMACrossover, "ann_return": 20.0,
                 "sharpe": 1.5, "expectancy": 3.0, "random_return": 5.0, "buy_hold_return": 8.0},
            ]

            persister.finalize_run(results)

            best_overall = persister.run_dir / "best_overall.py"
            assert best_overall.exists()
            assert best_overall.read_text() == "# fold 2 best"


class TestLoadStrategy:
    """Test load_strategy function."""

    def test_load_strategy_returns_class(self):
        """Should return a Strategy subclass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_code = '''
from backtesting import Strategy

class TestLoadedStrategy(Strategy):
    def init(self):
        pass
    def next(self):
        pass
'''
            strategy_path = Path(tmpdir) / "test_strategy.py"
            strategy_path.write_text(strategy_code)

            loaded_class = load_strategy(str(strategy_path))

            assert loaded_class is not None
            assert loaded_class.__name__ == "TestLoadedStrategy"

    def test_load_strategy_raises_if_no_strategy(self):
        """Should raise if no Strategy subclass found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = "x = 1"
            path = Path(tmpdir) / "no_strategy.py"
            path.write_text(code)

            with pytest.raises(ValueError, match="No Strategy subclass found"):
                load_strategy(str(path))


class TestProfitEvolverPersistence:
    """Test ProfitEvolver persistence integration."""

    def test_output_dir_creates_persister(self):
        """Should create persister when output_dir is set."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm, output_dir="test_output")

        assert evolver.persister is not None
        assert evolver.persister.output_dir == Path("test_output")

    def test_output_dir_none_disables_persister(self):
        """Should not create persister when output_dir is None."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm, output_dir=None)

        assert evolver.persister is None

    def test_default_output_dir(self):
        """Should have no persister by default (deprecated)."""
        mock_llm = Mock()
        evolver = ProfitEvolver(mock_llm)

        # Default is now None (StrategyPersister is deprecated)
        assert evolver.persister is None
