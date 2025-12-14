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

    def test_buy_and_hold_runs(self, small_data):
        """Buy-and-hold strategy should run without errors."""
        bt = Backtest(small_data, BuyAndHoldStrategy, cash=10000, commission=0.002, finalize_trades=True)
        result = bt.run()
        assert result is not None

    def test_buy_and_hold_single_trade(self, small_data):
        """Buy-and-hold should make at most one trade (may be 0 if position never closed)."""
        bt = Backtest(small_data, BuyAndHoldStrategy, cash=10000, commission=0.002, finalize_trades=True)
        result = bt.run()
        # backtesting.py only counts a "trade" when position is closed
        # Buy-and-hold may show 0 trades if position is held until end
        assert result["# Trades"] <= 1


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
