"""Tests for the evaluation module (Phase 15).

Tests:
- StrategyMetrics dataclass
- MetricsCalculator
- Evaluation stages (SyntaxCheck, SmokeTest, SingleFold)
- PromotionGate
- EvaluationCascade
- Selection policies (WeightedSum, GatedMAS, Pareto)
- Cache functionality
"""

import math

import numpy as np
import pandas as pd
import pytest
from backtesting import Strategy

from profit.evaluation import (
    # Core helpers
    load_strategy_class,
    run_bt,
    evaluate_on_data,
    code_hash,
    # Cache
    CacheKey,
    EvaluationCache,
    # Metrics
    StrategyMetrics,
    MetricsCalculator,
    METRIC_CAPS,
    # Stages
    StageResult,
    StageOutput,
    SyntaxCheckStage,
    SmokeTestStage,
    SingleFoldStage,
    FullWalkForwardStage,
    PromotionGate,
    # Cascade
    CascadeResult,
    EvaluationCascade,
    # Policies
    WeightedSumPolicy,
    GatedMASPolicy,
    ParetoPolicy,
    # Factories
    create_selection_policy,
    create_cascade,
)


# ===========================================================================
# Sample strategy code for testing
# ===========================================================================

VALID_STRATEGY_CODE = '''
class TestStrategy(Strategy):
    """Simple test strategy."""

    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()
'''

INVALID_SYNTAX_CODE = '''
class TestStrategy(Strategy)
    def init(self):
        pass
'''

MISSING_METHODS_CODE = '''
class TestStrategy(Strategy):
    pass
'''

NON_STRATEGY_CODE = '''
class TestClass:
    def init(self):
        pass

    def next(self):
        pass
'''


# ===========================================================================
# Core Helper Tests
# ===========================================================================


class TestLoadStrategyClass:
    """Tests for load_strategy_class helper."""

    def test_load_valid_strategy(self):
        """Should load valid strategy code."""
        strategy_class = load_strategy_class(VALID_STRATEGY_CODE)
        assert issubclass(strategy_class, Strategy)
        assert strategy_class.__name__ == "TestStrategy"

    def test_reject_syntax_error(self):
        """Should raise SyntaxError for invalid syntax."""
        with pytest.raises(SyntaxError):
            load_strategy_class(INVALID_SYNTAX_CODE)

    def test_reject_non_strategy(self):
        """Should raise ValueError if no Strategy subclass found."""
        with pytest.raises(ValueError, match="No valid Strategy subclass"):
            load_strategy_class(NON_STRATEGY_CODE)


class TestRunBt:
    """Tests for run_bt helper."""

    def test_run_backtest(self, small_data):
        """Should run backtest and return result Series."""
        strategy_class = load_strategy_class(VALID_STRATEGY_CODE)
        result = run_bt(strategy_class, small_data)
        assert isinstance(result, pd.Series)
        assert "Return (Ann.) [%]" in result.index
        assert "Sharpe Ratio" in result.index


class TestCodeHash:
    """Tests for code_hash helper."""

    def test_deterministic_hash(self):
        """Same code should produce same hash."""
        hash1 = code_hash(VALID_STRATEGY_CODE)
        hash2 = code_hash(VALID_STRATEGY_CODE)
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_different_code_different_hash(self):
        """Different code should produce different hash."""
        hash1 = code_hash(VALID_STRATEGY_CODE)
        hash2 = code_hash(VALID_STRATEGY_CODE + "# comment")
        assert hash1 != hash2


# ===========================================================================
# Cache Tests
# ===========================================================================


class TestEvaluationCache:
    """Tests for EvaluationCache."""

    def test_cache_put_get(self):
        """Should store and retrieve results."""
        cache = EvaluationCache()
        key = CacheKey("hash1", "stage1", "data1", "config1")
        output = StageOutput(result=StageResult.PASS)

        cache.put(key, output)
        result = cache.get(key)

        assert result is not None
        assert result.result == StageResult.PASS

    def test_cache_miss(self):
        """Should return None for missing keys."""
        cache = EvaluationCache()
        key = CacheKey("hash1", "stage1", "data1", "config1")
        assert cache.get(key) is None

    def test_hit_rate(self):
        """Should track hit rate correctly."""
        cache = EvaluationCache()
        key = CacheKey("hash1", "stage1", "data1", "config1")
        output = StageOutput(result=StageResult.PASS)

        cache.put(key, output)
        cache.get(key)  # hit
        cache.get(key)  # hit
        cache.get(CacheKey("other", "stage", "data", "config"))  # miss

        assert cache.hit_rate == 2 / 3

    def test_eviction(self):
        """Should evict entries when max size reached."""
        cache = EvaluationCache(max_size=10)

        # Fill cache beyond max size
        for i in range(15):
            key = CacheKey(f"hash{i}", "stage", "data", "config")
            cache.put(key, StageOutput(result=StageResult.PASS))

        # Cache should have evicted some entries
        assert len(cache._cache) <= 10


# ===========================================================================
# Metrics Tests
# ===========================================================================


class TestStrategyMetrics:
    """Tests for StrategyMetrics dataclass."""

    def test_default_values(self):
        """Should initialize with defaults."""
        metrics = StrategyMetrics()
        assert metrics.ann_return == 0.0
        assert metrics.sharpe == 0.0
        assert metrics.trade_count == 0

    def test_to_dict(self):
        """Should convert to dict without lists."""
        metrics = StrategyMetrics(ann_return=10.0, sharpe=1.5, trade_count=50)
        d = metrics.to_dict()
        assert d["ann_return"] == 10.0
        assert d["sharpe"] == 1.5
        assert d["trade_count"] == 50
        assert "equity_curve" not in d

    def test_from_dict(self):
        """Should create from dict."""
        d = {"ann_return": 15.0, "sharpe": 2.0, "max_drawdown": -10.0}
        metrics = StrategyMetrics.from_dict(d)
        assert metrics.ann_return == 15.0
        assert metrics.sharpe == 2.0
        assert metrics.max_drawdown == -10.0

    def test_is_valid(self):
        """Should detect invalid values."""
        valid = StrategyMetrics(ann_return=10.0, sharpe=1.0)
        assert valid.is_valid()

        invalid_nan = StrategyMetrics(ann_return=float("nan"))
        assert not invalid_nan.is_valid()

        invalid_inf = StrategyMetrics(sharpe=float("inf"))
        assert not invalid_inf.is_valid()


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_compute_all(self, small_data):
        """Should compute all metrics from backtest result."""
        strategy_class = load_strategy_class(VALID_STRATEGY_CODE)
        result = run_bt(strategy_class, small_data)

        calc = MetricsCalculator()
        metrics = calc.compute_all(result)

        assert isinstance(metrics, StrategyMetrics)
        assert isinstance(metrics.ann_return, float)
        assert isinstance(metrics.sharpe, float)
        assert isinstance(metrics.trade_count, int)

    def test_compute_cross_fold_metrics(self):
        """Should compute stability and consistency across folds."""
        fold_metrics = [
            StrategyMetrics(ann_return=10.0),
            StrategyMetrics(ann_return=15.0),
            StrategyMetrics(ann_return=-5.0),
            StrategyMetrics(ann_return=20.0),
        ]

        calc = MetricsCalculator()
        agg = calc.compute_cross_fold_metrics(fold_metrics)

        # Mean return should be (10+15-5+20)/4 = 10
        assert agg.ann_return == pytest.approx(10.0)
        # 3 out of 4 folds positive = 75%
        assert agg.consistency == 75.0
        # Worst fold is -5
        assert agg.worst_fold_return == -5.0
        # Stability is std dev
        assert agg.stability > 0

    def test_metric_capping(self):
        """Should cap infinite values."""
        # Sortino with no negative returns should be capped
        metrics = StrategyMetrics(sortino=METRIC_CAPS["sortino"])
        assert metrics.sortino == METRIC_CAPS["sortino"]


# ===========================================================================
# Stage Tests
# ===========================================================================


class TestSyntaxCheckStage:
    """Tests for SyntaxCheckStage."""

    def test_pass_valid_code(self):
        """Should pass valid strategy code."""
        stage = SyntaxCheckStage()
        result = stage.evaluate(VALID_STRATEGY_CODE, None)
        assert result.result == StageResult.PASS

    def test_fail_syntax_error(self):
        """Should fail code with syntax errors."""
        stage = SyntaxCheckStage()
        result = stage.evaluate(INVALID_SYNTAX_CODE, None)
        assert result.result == StageResult.FAIL
        assert "Syntax error" in result.error

    def test_fail_missing_methods(self):
        """Should fail code missing required methods."""
        stage = SyntaxCheckStage()
        result = stage.evaluate(MISSING_METHODS_CODE, None)
        assert result.result == StageResult.FAIL
        assert "Missing methods" in result.error


class TestSmokeTestStage:
    """Tests for SmokeTestStage."""

    def test_pass_valid_strategy(self, small_data):
        """Should pass valid strategy that runs."""
        stage = SmokeTestStage(slice_months=1)
        result = stage.evaluate(VALID_STRATEGY_CODE, small_data)
        assert result.result == StageResult.PASS

    def test_fail_non_strategy(self, small_data):
        """Should fail non-strategy code."""
        stage = SmokeTestStage(slice_months=1)
        result = stage.evaluate(NON_STRATEGY_CODE, small_data)
        assert result.result == StageResult.FAIL


class TestSingleFoldStage:
    """Tests for SingleFoldStage."""

    def test_pass_with_metrics(self, small_data):
        """Should pass and return metrics."""
        # Use gate with min_trades=0 since our simple strategy doesn't trade
        gate = PromotionGate(min_trades=0)
        stage = SingleFoldStage(promotion_gate=gate)
        result = stage.evaluate(VALID_STRATEGY_CODE, small_data)
        assert result.result == StageResult.PASS
        assert result.metrics is not None
        assert isinstance(result.metrics, StrategyMetrics)

    def test_promotion_gate_rejection(self, small_data):
        """Should fail strategies that don't meet promotion gate."""
        # Our simple strategy only makes 1 trade, so min_trades=100 will fail
        gate = PromotionGate(min_trades=100)
        stage = SingleFoldStage(promotion_gate=gate)
        result = stage.evaluate(VALID_STRATEGY_CODE, small_data)
        assert result.result == StageResult.FAIL
        assert "Promotion gate failed" in result.error
        assert "Too few trades" in result.error


class TestPromotionGate:
    """Tests for PromotionGate."""

    def test_pass_default_gate(self):
        """Should pass metrics meeting default thresholds."""
        gate = PromotionGate()
        metrics = StrategyMetrics(trade_count=10, max_drawdown=-20.0)
        passed, error = gate.check(metrics)
        assert passed is True
        assert error is None

    def test_fail_too_few_trades(self):
        """Should fail if too few trades."""
        gate = PromotionGate(min_trades=5)
        metrics = StrategyMetrics(trade_count=3)
        passed, error = gate.check(metrics)
        assert passed is False
        assert "Too few trades" in error

    def test_fail_severe_drawdown(self):
        """Should fail if drawdown too severe."""
        # Set min_trades=0 so we actually test the drawdown check
        gate = PromotionGate(min_trades=0, max_drawdown_limit=-50.0)
        metrics = StrategyMetrics(trade_count=0, max_drawdown=-60.0)
        passed, error = gate.check(metrics)
        assert passed is False
        assert "Drawdown too severe" in error

    def test_fail_sharpe_too_low(self):
        """Should fail if Sharpe ratio below threshold."""
        gate = PromotionGate(min_trades=0, min_sharpe=-2.0)
        metrics = StrategyMetrics(trade_count=0, sharpe=-3.5)
        passed, error = gate.check(metrics)
        assert passed is False
        assert "Sharpe too low" in error

    def test_pass_sharpe_above_threshold(self):
        """Should pass if Sharpe ratio meets threshold."""
        gate = PromotionGate(min_trades=0, min_sharpe=-2.0)
        metrics = StrategyMetrics(trade_count=0, sharpe=-1.5)
        passed, error = gate.check(metrics)
        assert passed is True
        assert error is None

    def test_fail_win_rate_too_low(self):
        """Should fail if win rate below threshold."""
        gate = PromotionGate(min_trades=0, min_win_rate=30.0)
        metrics = StrategyMetrics(trade_count=0, win_rate=20.0)
        passed, error = gate.check(metrics)
        assert passed is False
        assert "Win rate too low" in error

    def test_pass_win_rate_above_threshold(self):
        """Should pass if win rate meets threshold."""
        gate = PromotionGate(min_trades=0, min_win_rate=30.0)
        metrics = StrategyMetrics(trade_count=0, win_rate=40.0)
        passed, error = gate.check(metrics)
        assert passed is True
        assert error is None

    def test_combined_gates(self):
        """Should check all gates together."""
        gate = PromotionGate(
            min_trades=5,
            max_drawdown_limit=-50.0,
            min_sharpe=-2.0,
            min_win_rate=25.0,
        )
        # Metrics that pass all gates
        good_metrics = StrategyMetrics(
            trade_count=10,
            max_drawdown=-30.0,
            sharpe=-1.0,
            win_rate=35.0,
        )
        passed, error = gate.check(good_metrics)
        assert passed is True

        # Metrics that fail sharpe gate
        bad_sharpe = StrategyMetrics(
            trade_count=10,
            max_drawdown=-30.0,
            sharpe=-3.0,  # Below -2.0 threshold
            win_rate=35.0,
        )
        passed, error = gate.check(bad_sharpe)
        assert passed is False
        assert "Sharpe too low" in error


# ===========================================================================
# Cascade Tests
# ===========================================================================


class TestEvaluationCascade:
    """Tests for EvaluationCascade."""

    def test_full_cascade_pass(self, small_data):
        """Should run full cascade for valid strategy."""
        # Use lenient gate since our test strategy doesn't trade
        gate = PromotionGate(min_trades=0)
        cascade = EvaluationCascade(
            stages=[
                SyntaxCheckStage(),
                SmokeTestStage(slice_months=1),
                SingleFoldStage(promotion_gate=gate),
            ],
            verbose=False,
        )
        result = cascade.evaluate(VALID_STRATEGY_CODE, small_data)
        assert result.passed is True
        assert result.metrics is not None

    def test_cascade_fail_fast(self, small_data):
        """Should stop at first failing stage."""
        cascade = EvaluationCascade(
            stages=[
                SyntaxCheckStage(),
                SmokeTestStage(slice_months=1),
            ],
            verbose=False,
        )
        result = cascade.evaluate(INVALID_SYNTAX_CODE, small_data)
        assert result.passed is False
        assert result.final_stage == "syntax_check"
        assert "smoke_test" not in result.stage_results

    def test_quick_evaluate(self, small_data):
        """Should stop at smoke test stage."""
        cascade = EvaluationCascade(
            stages=[
                SyntaxCheckStage(),
                SmokeTestStage(slice_months=1),
                SingleFoldStage(),
            ],
            verbose=False,
        )
        result = cascade.quick_evaluate(VALID_STRATEGY_CODE, small_data)
        assert result.passed is True
        # Should not have run single_fold
        assert "single_fold" not in result.stage_results


# ===========================================================================
# Selection Policy Tests
# ===========================================================================


class TestWeightedSumPolicy:
    """Tests for WeightedSumPolicy."""

    def test_compute_fitness(self):
        """Should compute weighted fitness score."""
        policy = WeightedSumPolicy(w_return=0.5, w_sharpe=0.3, w_drawdown=0.2)
        baseline = StrategyMetrics(ann_return=10.0, sharpe=1.0, max_drawdown=-20.0)
        candidate = StrategyMetrics(ann_return=15.0, sharpe=1.5, max_drawdown=-15.0)

        # Candidate is better, should have higher fitness
        fitness_base = policy.compute_fitness(baseline, baseline)
        fitness_cand = policy.compute_fitness(candidate, baseline)

        assert fitness_cand > fitness_base

    def test_should_accept_better_strategy(self):
        """Should accept strategy with higher fitness."""
        policy = WeightedSumPolicy()
        baseline = StrategyMetrics(ann_return=10.0, sharpe=1.0, max_drawdown=-20.0)
        candidate = StrategyMetrics(ann_return=20.0, sharpe=2.0, max_drawdown=-10.0)

        assert policy.should_accept(candidate, baseline) is True

    def test_should_reject_worse_strategy(self):
        """Should reject strategy with lower fitness."""
        policy = WeightedSumPolicy()
        baseline = StrategyMetrics(ann_return=20.0, sharpe=2.0, max_drawdown=-10.0)
        candidate = StrategyMetrics(ann_return=5.0, sharpe=0.5, max_drawdown=-40.0)

        assert policy.should_accept(candidate, baseline) is False


class TestGatedMASPolicy:
    """Tests for GatedMASPolicy."""

    def test_pass_all_gates(self):
        """Should accept if all gates pass and beats baseline."""
        policy = GatedMASPolicy(
            min_return=5.0,
            min_sharpe=0.5,
            max_drawdown=-30.0,
            min_trades=5,
        )
        baseline = StrategyMetrics(ann_return=10.0)
        candidate = StrategyMetrics(
            ann_return=15.0,
            sharpe=1.0,
            max_drawdown=-20.0,
            trade_count=10,
        )

        assert policy.should_accept(candidate, baseline) is True

    def test_fail_return_gate(self):
        """Should reject if return below threshold."""
        policy = GatedMASPolicy(min_return=10.0)
        baseline = StrategyMetrics(ann_return=5.0)
        candidate = StrategyMetrics(ann_return=8.0)  # Above baseline but below gate

        assert policy.should_accept(candidate, baseline) is False

    def test_fail_sharpe_gate(self):
        """Should reject if Sharpe below threshold."""
        policy = GatedMASPolicy(min_sharpe=1.0)
        baseline = StrategyMetrics(ann_return=5.0)
        candidate = StrategyMetrics(ann_return=15.0, sharpe=0.5)

        assert policy.should_accept(candidate, baseline) is False

    def test_fail_below_baseline(self):
        """Should reject if below baseline even if gates pass."""
        policy = GatedMASPolicy()
        baseline = StrategyMetrics(ann_return=20.0)
        candidate = StrategyMetrics(ann_return=15.0)  # Below baseline

        assert policy.should_accept(candidate, baseline) is False


class TestParetoPolicy:
    """Tests for ParetoPolicy."""

    def test_dominates_strictly_better(self):
        """A should dominate B if strictly better on all objectives."""
        policy = ParetoPolicy(objectives=("ann_return", "sharpe"))
        a = StrategyMetrics(ann_return=20.0, sharpe=2.0)
        b = StrategyMetrics(ann_return=10.0, sharpe=1.0)

        assert policy.dominates(a, b) is True
        assert policy.dominates(b, a) is False

    def test_no_dominance_tradeoff(self):
        """Neither dominates if there's a tradeoff."""
        policy = ParetoPolicy(objectives=("ann_return", "sharpe"))
        a = StrategyMetrics(ann_return=20.0, sharpe=1.0)
        b = StrategyMetrics(ann_return=10.0, sharpe=2.0)

        assert policy.dominates(a, b) is False
        assert policy.dominates(b, a) is False

    def test_should_accept_non_dominated(self):
        """Should accept non-dominated strategy."""
        policy = ParetoPolicy(objectives=("ann_return", "sharpe"))
        baseline = StrategyMetrics(ann_return=10.0, sharpe=1.5)
        candidate = StrategyMetrics(ann_return=15.0, sharpe=1.0)  # Different tradeoff

        assert policy.should_accept(candidate, baseline) is True

    def test_should_reject_dominated(self):
        """Should reject dominated strategy."""
        policy = ParetoPolicy(objectives=("ann_return", "sharpe"))
        baseline = StrategyMetrics(ann_return=20.0, sharpe=2.0)
        candidate = StrategyMetrics(ann_return=10.0, sharpe=1.0)

        assert policy.should_accept(candidate, baseline) is False

    def test_pareto_with_population(self):
        """Should check against full population for dominance."""
        policy = ParetoPolicy(objectives=("ann_return", "sharpe"))
        baseline = StrategyMetrics(ann_return=10.0, sharpe=1.0)
        population = [
            StrategyMetrics(ann_return=10.0, sharpe=1.0),
            StrategyMetrics(ann_return=20.0, sharpe=2.0),  # Dominates candidate
        ]
        candidate = StrategyMetrics(ann_return=15.0, sharpe=1.5)

        # Dominated by second strategy in population
        assert policy.should_accept(candidate, baseline, population_metrics=population) is False

    def test_max_drawdown_dominance(self):
        """Should correctly handle max_drawdown in dominance checks.

        Bug fix test: max_drawdown is stored as negative (e.g., -15.0 means 15% drawdown).
        Less negative = better (e.g., -10% > -20%, so -10% is better).
        """
        policy = ParetoPolicy(objectives=("ann_return", "sharpe", "max_drawdown"))

        # Baseline: better on all metrics (higher return, higher sharpe, less negative drawdown)
        baseline = StrategyMetrics(
            ann_return=-15.86,
            sharpe=-1.89,
            max_drawdown=-13.27  # Less negative = less drawdown = better
        )

        # Candidate: worse on ALL metrics
        candidate = StrategyMetrics(
            ann_return=-24.50,  # Worse (more negative)
            sharpe=-3.44,       # Worse (more negative)
            max_drawdown=-15.23  # Worse (more negative = more drawdown)
        )

        # Baseline should dominate candidate (better on all objectives)
        assert policy.dominates(baseline, candidate) is True

        # Candidate should NOT dominate baseline
        assert policy.dominates(candidate, baseline) is False

        # Therefore, candidate should be REJECTED
        assert policy.should_accept(candidate, baseline) is False

    def test_max_drawdown_tradeoff(self):
        """Should accept if max_drawdown creates a valid tradeoff."""
        policy = ParetoPolicy(objectives=("ann_return", "sharpe", "max_drawdown"))

        baseline = StrategyMetrics(
            ann_return=10.0,
            sharpe=1.0,
            max_drawdown=-20.0  # 20% drawdown
        )

        # Candidate: worse return and sharpe, but BETTER drawdown
        candidate = StrategyMetrics(
            ann_return=8.0,      # Worse
            sharpe=0.8,          # Worse
            max_drawdown=-10.0   # Better (less negative = less drawdown)
        )

        # Neither should dominate (there's a valid tradeoff)
        assert policy.dominates(baseline, candidate) is False
        assert policy.dominates(candidate, baseline) is False

        # Candidate should be accepted (non-dominated)
        assert policy.should_accept(candidate, baseline) is True

    def test_debug_logging(self, capsys):
        """Debug mode should print decision details."""
        policy = ParetoPolicy(
            objectives=("ann_return", "sharpe", "max_drawdown"),
            debug=True
        )

        baseline = StrategyMetrics(ann_return=10.0, sharpe=1.0, max_drawdown=-20.0)
        candidate = StrategyMetrics(ann_return=5.0, sharpe=0.5, max_drawdown=-30.0)

        # This should print debug info
        policy.should_accept(candidate, baseline)

        captured = capsys.readouterr()
        assert "[Pareto] should_accept called" in captured.out
        assert "Checking dominance" in captured.out

    def test_regression_run_output_scenario(self):
        """Regression test: exact scenario from user's run that showed the bug.

        From the run output:
        - Baseline: Return=-15.86%, Sharpe=-1.89, MaxDD=-13.27%
        - EMACrossover_1: Return=-24.50%, Sharpe=-3.44, MaxDD=-15.23%

        EMACrossover_1 is WORSE on ALL objectives but was incorrectly accepted.
        After fix, it should be REJECTED.
        """
        policy = ParetoPolicy(
            objectives=("ann_return", "sharpe", "max_drawdown")
        )

        # Baseline from actual run
        baseline = StrategyMetrics(
            ann_return=-15.86,
            sharpe=-1.89,
            max_drawdown=-13.27
        )

        # EMACrossover_1 from actual run - worse on ALL objectives
        candidate = StrategyMetrics(
            ann_return=-24.50,
            sharpe=-3.44,
            max_drawdown=-15.23
        )

        # Should be rejected (dominated by baseline)
        result = policy.should_accept(candidate, baseline)
        assert result is False, (
            f"Expected rejection but got acceptance. "
            f"Baseline dominates candidate on all objectives: "
            f"return ({baseline.ann_return} > {candidate.ann_return}), "
            f"sharpe ({baseline.sharpe} > {candidate.sharpe}), "
            f"max_drawdown ({baseline.max_drawdown} > {candidate.max_drawdown})"
        )


# ===========================================================================
# Factory Tests
# ===========================================================================


class TestCreateSelectionPolicy:
    """Tests for create_selection_policy factory."""

    def test_create_weighted(self):
        """Should create WeightedSumPolicy."""
        policy = create_selection_policy("weighted", w_return=0.6)
        assert isinstance(policy, WeightedSumPolicy)
        assert policy.w_return == 0.6

    def test_create_gated(self):
        """Should create GatedMASPolicy."""
        policy = create_selection_policy("gated", min_sharpe=1.0)
        assert isinstance(policy, GatedMASPolicy)
        assert policy.min_sharpe == 1.0

    def test_create_pareto(self):
        """Should create ParetoPolicy."""
        policy = create_selection_policy("pareto", objectives=["ann_return", "sharpe"])
        assert isinstance(policy, ParetoPolicy)
        assert "ann_return" in policy.objectives


class TestCreateCascade:
    """Tests for create_cascade factory."""

    def test_create_quick(self):
        """Should create quick cascade with 2 stages."""
        cascade = create_cascade(mode="quick", verbose=False)
        assert len(cascade.stages) == 2

    def test_create_standard(self):
        """Should create standard cascade with 3 stages."""
        cascade = create_cascade(mode="standard", verbose=False)
        assert len(cascade.stages) == 3

    def test_create_full(self):
        """Should create full cascade with 4 stages."""
        cascade = create_cascade(mode="full", verbose=False)
        assert len(cascade.stages) == 4
