"""Evolutionary engine for strategy optimization.

This module contains the ProfitEvolver class which orchestrates the
evolution loop for trading strategy optimization using LLMs.
"""

import pandas as pd
from backtesting import Backtest

from profit.llm_interface import LLMClient
from profit.strategies import (
    BollingerMeanReversion,
    CCIStrategy,
    EMACrossover,
    MACDStrategy,
    WilliamsRStrategy,
    RandomStrategy,
    BuyAndHoldStrategy,
)


class ProfitEvolver:
    """Evolutionary search engine for trading strategy optimization.

    Uses LLM-guided code mutation and walk-forward validation to evolve
    strategies that adapt to changing market conditions.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        initial_capital: float = 10000,
        commission: float = 0.002,
        exclusive_orders: bool = True,
    ):
        """Initialize the ProfitEvolver.

        Args:
            llm_client: An instance of LLMClient for generating strategy mutations.
            initial_capital: Starting cash for backtests (default: $10,000).
            commission: Per-trade commission rate (default: 0.002 = 0.2%).
            exclusive_orders: If True, no overlapping long/short positions.
        """
        self.llm = llm_client
        self.initial_capital = initial_capital
        self.commission = commission
        self.exclusive_orders = exclusive_orders

    def run_backtest(self, strategy_class, data: pd.DataFrame) -> tuple[dict, pd.Series]:
        """Run a backtest on given data with specified strategy class.

        Args:
            strategy_class: A backtesting.Strategy subclass to evaluate.
            data: DataFrame with OHLCV data and datetime index.

        Returns:
            A tuple of (metrics, result) where:
            - metrics: dict with key performance metrics (AnnReturn%, Sharpe, Expectancy%, Trades)
            - result: Full backtesting.py result Series
        """
        bt = Backtest(
            data,
            strategy_class,
            cash=self.initial_capital,
            commission=self.commission,
            exclusive_orders=self.exclusive_orders,
        )
        result = bt.run()

        # Extract key metrics
        metrics = {
            "AnnReturn%": result.get("Return (Ann.) [%]", None),
            "Sharpe": result.get("Sharpe Ratio", None),
            "Expectancy%": result.get("Expectancy [%]", None),
            "Trades": result.get("# Trades", None),
        }
        return metrics, result

    def prepare_folds(
        self, full_data: pd.DataFrame, n_folds: int = 5
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Split the full_data DataFrame into train/validation/test folds.

        Implements walk-forward validation with:
        - Training period: 2.5 years (30 months)
        - Validation period: 6 months
        - Test period: 6 months
        - Gap between periods: 10 days (prevents look-ahead bias)

        Args:
            full_data: Complete DataFrame with datetime index and OHLCV columns.
            n_folds: Number of walk-forward folds (default: 5).

        Returns:
            List of (train_df, val_df, test_df) tuples for each fold.
        """
        fold_splits = []
        data_index = full_data.index
        start_date = data_index.min()

        for fold in range(n_folds):
            # Train period: 2.5 years (2 years + 6 months) from current start
            train_end = start_date + pd.DateOffset(years=2) + pd.DateOffset(months=6)

            # Validation start: 10 days after train end (gap)
            val_start = train_end + pd.DateOffset(days=10)
            # Validation end: 6 months after train end
            val_end = train_end + pd.DateOffset(months=6)

            # Test start: 10 days after validation end (gap)
            test_start = val_end + pd.DateOffset(days=10)
            # Test end: 6 months after validation end
            test_end = val_end + pd.DateOffset(months=6)

            # Slice data for each period
            train = full_data[start_date:train_end]
            val = full_data[val_start:val_end]
            test = full_data[test_start:test_end]

            # Stop if we run out of data
            if len(test) == 0:
                break

            fold_splits.append((train, val, test))

            # Move start_date to end of this test period for next fold
            start_date = test_end

        return fold_splits
