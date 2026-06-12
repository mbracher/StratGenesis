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
    SEED_STRATEGIES,
    BASELINE_STRATEGIES,
    ALL_STRATEGIES,
)


class TestSeedStrategies:
    """Test seed strategy classes."""

    @pytest.mark.parametrize(
        "strategy_class",
        [
            BollingerMeanReversion,
            CCIStrategy,
            EMACrossover,
            MACDStrategy,
            WilliamsRStrategy,
        ],
    )
    def test_strategy_runs(self, small_data, strategy_class):
        """Each strategy should run without errors."""
        bt = Backtest(small_data, strategy_class, cash=10000, commission=0.002, finalize_trades=True)
        result = bt.run()
        assert result is not None
        assert "Return (Ann.) [%]" in result

    @pytest.mark.parametrize(
        "strategy_class",
        [
            BollingerMeanReversion,
            CCIStrategy,
            EMACrossover,
            MACDStrategy,
            WilliamsRStrategy,
        ],
    )
    def test_strategy_makes_trades(self, sample_data, strategy_class):
        """Each strategy should generate at least some trades."""
        bt = Backtest(sample_data, strategy_class, cash=10000, commission=0.002, finalize_trades=True)
        result = bt.run()
        assert result["# Trades"] > 0


class TestBaselineStrategies:
    """Test baseline strategy classes."""

    def test_random_strategy_runs(self, small_data):
        """Random strategy should run without errors."""
        bt = Backtest(small_data, RandomStrategy, cash=10000, commission=0.002, finalize_trades=True)
        result = bt.run()
        assert result is not None

    def test_random_strategy_reproducible(self, small_data):
        """Random strategy should be reproducible with same seed."""
        bt1 = Backtest(small_data, RandomStrategy, cash=10000, commission=0.002, finalize_trades=True)
        bt2 = Backtest(small_data, RandomStrategy, cash=10000, commission=0.002, finalize_trades=True)
        result1 = bt1.run()
        result2 = bt2.run()
        assert result1["# Trades"] == result2["# Trades"]

    def test_random_strategy_seed_parameter(self, small_data):
        """Seed should be a backtesting.py parameter, overridable per run."""
        assert RandomStrategy.seed == 42

        bt = Backtest(small_data, RandomStrategy, cash=10000, commission=0.002, finalize_trades=True)
        default_result = bt.run()
        same_seed_result = bt.run(seed=42)
        other_seed_result = bt.run(seed=7)

        # Explicit seed=42 must match the default run exactly
        assert same_seed_result["# Trades"] == default_result["# Trades"]
        assert same_seed_result["Return [%]"] == default_result["Return [%]"]
        # A different seed must change the random trade stream (deterministic
        # on the seeded fixture; catches implementations that ignore the param)
        assert other_seed_result["Return [%]"] != default_result["Return [%]"]

    def test_random_strategy_seed_none(self, small_data):
        """seed=None should draw fresh entropy and still run."""
        bt = Backtest(small_data, RandomStrategy, cash=10000, commission=0.002, finalize_trades=True)
        result = bt.run(seed=None)
        assert result is not None

    def test_buy_and_hold_runs(self, small_data):
        """Buy-and-hold strategy should run without errors."""
        bt = Backtest(small_data, BuyAndHoldStrategy, cash=10000, commission=0.002, finalize_trades=True)
        result = bt.run()
        assert result is not None

    def test_buy_and_hold_single_trade(self, small_data):
        """Buy-and-hold should make exactly one trade."""
        bt = Backtest(small_data, BuyAndHoldStrategy, cash=10000, commission=0.002, finalize_trades=True)
        result = bt.run()
        # finalize_trades=True closes the held position at the end of the
        # backtest, so the single buy is always counted as one trade
        assert result["# Trades"] == 1


class TestStrategyParameters:
    """Test strategy parameter configuration."""

    def test_bollinger_parameters(self):
        """Bollinger strategy should have configurable parameters."""
        assert hasattr(BollingerMeanReversion, "bb_period")
        assert hasattr(BollingerMeanReversion, "bb_stddev")
        assert BollingerMeanReversion.bb_period == 20
        assert BollingerMeanReversion.bb_stddev == 2

    def test_ema_parameters(self):
        """EMA strategy should have configurable parameters."""
        assert hasattr(EMACrossover, "fast_ema")
        assert hasattr(EMACrossover, "slow_ema")
        assert EMACrossover.fast_ema == 50
        assert EMACrossover.slow_ema == 200

    def test_cci_parameters(self):
        """CCI strategy should have configurable parameters."""
        assert hasattr(CCIStrategy, "cci_period")
        assert CCIStrategy.cci_period == 20

    def test_macd_parameters(self):
        """MACD strategy should have configurable parameters."""
        assert hasattr(MACDStrategy, "fast")
        assert hasattr(MACDStrategy, "slow")
        assert hasattr(MACDStrategy, "signal")
        assert MACDStrategy.fast == 12
        assert MACDStrategy.slow == 26
        assert MACDStrategy.signal == 9

    def test_williams_r_parameters(self):
        """Williams %R strategy should have configurable parameters."""
        assert hasattr(WilliamsRStrategy, "lookback")
        assert WilliamsRStrategy.lookback == 14


class TestStrategyRegistries:
    """Test strategy registry dictionaries."""

    def test_seed_strategies_count(self):
        """Should have 5 seed strategies."""
        assert len(SEED_STRATEGIES) == 5

    def test_baseline_strategies_count(self):
        """Should have 2 baseline strategies."""
        assert len(BASELINE_STRATEGIES) == 2

    def test_all_strategies_combined(self):
        """ALL_STRATEGIES should contain all seed and baseline strategies."""
        assert len(ALL_STRATEGIES) == 7
        for name in SEED_STRATEGIES:
            assert name in ALL_STRATEGIES
        for name in BASELINE_STRATEGIES:
            assert name in ALL_STRATEGIES

    def test_strategy_lookup_by_name(self):
        """Should be able to look up strategies by name."""
        assert ALL_STRATEGIES["EMACrossover"] == EMACrossover
        assert ALL_STRATEGIES["RandomStrategy"] == RandomStrategy
