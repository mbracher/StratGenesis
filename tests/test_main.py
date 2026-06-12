"""Unit tests for the CLI entry point (load_data, print_results, parser)."""

import pandas as pd
import pytest

from profit.main import build_parser, load_data, print_results
from profit.strategies import EMACrossover


class TestLoadData:
    """Tests for load_data CSV loading and validation."""

    @staticmethod
    def _write_csv(path, columns=("Open", "High", "Low", "Close", "Volume")):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        data = pd.DataFrame({col: range(10) for col in columns}, index=dates)
        data.index.name = "Date"
        data.to_csv(path)
        return path

    def test_loads_valid_csv(self, tmp_path):
        path = self._write_csv(tmp_path / "ohlcv.csv")
        data = load_data(str(path))

        assert isinstance(data.index, pd.DatetimeIndex)
        assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(data) == 10

    def test_missing_column_raises(self, tmp_path):
        path = self._write_csv(tmp_path / "bad.csv", columns=("Open", "High", "Low"))

        with pytest.raises(ValueError, match="Close"):
            load_data(str(path))

    def test_non_datetime_index_raises(self, tmp_path):
        path = tmp_path / "noindex.csv"
        pd.DataFrame(
            {
                "Open": range(5),
                "High": range(5),
                "Low": range(5),
                "Close": range(5),
            },
            index=["a", "b", "c", "d", "e"],
        ).to_csv(path)

        with pytest.raises(ValueError, match="datetime"):
            load_data(str(path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "does_not_exist.csv"))


class TestPrintResults:
    """Smoke tests for the results formatter."""

    def test_prints_folds_and_averages(self, capsys):
        class TaggedStrategy(EMACrossover):
            pass

        TaggedStrategy._db_id = "abc12345"

        # Plain class (no shared-strategy base) guarantees no _db_id,
        # regardless of what other tests did to the strategy classes
        class UntaggedStrategy:
            pass

        results = [
            {
                "fold": 1,
                "strategy": TaggedStrategy,
                "ann_return": 12.0,
                "sharpe": 1.1,
                "expectancy": 0.5,
                "random_return": -2.0,
                "buy_hold_return": 8.0,
            },
            {
                "fold": 2,
                "strategy": UntaggedStrategy,
                "ann_return": 6.0,
                "sharpe": 0.7,
                "expectancy": 0.2,
                "random_return": 1.0,
                "buy_hold_return": 4.0,
            },
        ]

        print_results(results)
        out = capsys.readouterr().out

        assert "RESULTS SUMMARY" in out
        assert "Fold 1:" in out
        assert "Fold 2:" in out
        assert "[abc12345] TaggedStrategy" in out
        # Strategies without a DB id render without the bracketed prefix
        assert "Best Strategy: UntaggedStrategy" in out
        assert "AVERAGES ACROSS FOLDS:" in out
        assert "Evolved Strategy: 9.00%" in out
        assert "Improvement over B&H: +3.00%" in out


class TestParserDefaults:
    """Spec defaults for the CLI parser (Phases 12, 13, 14, 15)."""

    def test_defaults(self):
        args = build_parser().parse_args([])

        # Phase 15: gated policy and full cascade are the defaults
        assert args.selection_policy == "gated"
        assert args.skip_cascade is False
        assert args.quick_eval is False
        # Phase 13: program database always on, inspirations enabled
        assert args.db_backend == "json"
        assert args.db_path == "program_db"
        assert args.no_inspirations is False
        # Phase 14: diff mutations on, adaptive
        assert args.no_diffs is False
        assert args.diff_mode == "adaptive"
        assert args.diff_match == "tolerant"
        assert args.exploration_gens == 5
        # Phase 12: provider defaults (models resolved by LLMClient)
        assert args.provider == "openai"
        assert args.model is None
        assert args.analyst_provider is None
        assert args.coder_provider is None
        # Phase 11: file persistence deprecated, off by default
        assert args.output_dir is None
        # Core run parameters
        assert args.folds == 5
        assert args.capital == 10000
        assert args.commission == 0.002

    def test_llm_configuration_group(self):
        parser = build_parser()
        group_titles = [group.title for group in parser._action_groups]
        assert "LLM Configuration" in group_titles

        llm_group = next(
            group
            for group in parser._action_groups
            if group.title == "LLM Configuration"
        )
        flags = {
            option
            for action in llm_group._group_actions
            for option in action.option_strings
        }
        assert {
            "--provider",
            "--model",
            "--analyst-provider",
            "--analyst-model",
            "--coder-provider",
            "--coder-model",
        } <= flags

    def test_selection_policy_choices(self):
        parser = build_parser()
        args = parser.parse_args(["--selection-policy", "pareto"])
        assert args.selection_policy == "pareto"

        with pytest.raises(SystemExit):
            parser.parse_args(["--selection-policy", "bogus"])
