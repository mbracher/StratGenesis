"""Evolutionary engine for strategy optimization.

This module contains the ProfitEvolver class which orchestrates the
evolution loop for trading strategy optimization using LLMs.
"""

import inspect
import random
import traceback

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

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

    def _random_index(self, n: int) -> int:
        """Return a random index in range [0, n).

        Args:
            n: Upper bound (exclusive).

        Returns:
            Random integer in [0, n).
        """
        return random.randrange(n)

    def evolve_strategy(
        self,
        strategy_class,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        max_iters: int = 15,
    ):
        """Evolve a strategy using LLM-guided mutations.

        Implements the ProFiT evolutionary loop with MAS (Minimum Acceptable Score)
        threshold. New strategies must meet or exceed MAS to be accepted into the
        population.

        Args:
            strategy_class: Seed strategy class to evolve.
            train_data: Training data (for context, not directly used in fitness).
            val_data: Validation data for fitness evaluation.
            max_iters: Maximum number of evolution generations (default: 15).

        Returns:
            Tuple of (best_strategy_class, best_perf) where:
            - best_strategy_class: The evolved strategy class with highest fitness
            - best_perf: Performance (annualized return %) on validation
        """
        # 1. Compute baseline performance P0 on validation set
        _, base_result = self.run_backtest(strategy_class, val_data)
        P0 = base_result["Return (Ann.) [%]"]
        print(
            f"Initial strategy {strategy_class.__name__} baseline annualized return "
            f"on validation: {P0:.2f}%"
        )

        # 2. Set MAS = P0 (Minimum Acceptable Score)
        MAS = P0

        # Archive of viable strategies (as tuples of class and performance)
        population = [(strategy_class, P0)]
        best_perf = P0
        best_strategy_class = strategy_class

        # Build exec namespace with necessary imports for generated code
        exec_globals = {
            "Strategy": Strategy,
            "pd": pd,
            "np": np,
        }

        # 4. Evolution loop
        for gen in range(1, max_iters + 1):
            print(
                f"\nGeneration {gen}: Current population size = {len(population)}. "
                "Selecting a strategy to mutate..."
            )

            # 5. Select a strategy from population (random selection for diversity)
            parent_class, parent_perf = population[self._random_index(len(population))]
            print(
                f"Selected parent strategy '{parent_class.__name__}' with validation "
                f"return {parent_perf:.2f}% for mutation."
            )

            # Get source code of parent strategy
            parent_code = inspect.getsource(parent_class)

            # 6. Prompt LLM A for improvement proposal
            improvement = self.llm.generate_improvement(
                parent_code, f"AnnReturn={parent_perf:.2f}%"
            )
            print(f"LLM suggested improvement: {improvement}")

            # 7. Prompt LLM B to synthesize modified strategy code
            new_code = self.llm.generate_strategy_code(parent_code, improvement)

            # Give the new strategy a unique name by generation
            new_class_name = f"{parent_class.__name__}_Gen{gen}"

            # Replace class name in code to avoid collisions
            if new_code.startswith("class"):
                new_code = new_code.replace(parent_class.__name__, new_class_name, 1)

            # 8-11. Try to compile and backtest with repair loop
            success = False
            NewStrategyClass = None
            res = None

            for attempt in range(1, 11):  # up to 10 repair attempts
                try:
                    # Dynamically define the new strategy class from code
                    namespace = {}
                    exec(new_code, exec_globals, namespace)
                    NewStrategyClass = namespace[new_class_name]

                    # Run backtest on validation data to get performance
                    _, res = self.run_backtest(NewStrategyClass, val_data)
                    success = True
                    break

                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"Attempt {attempt}: Strategy code failed with error: {e}")

                    if attempt < 10:
                        # 11. Prompt LLM B with traceback to fix code
                        new_code = self.llm.fix_code(new_code, tb)

                        # Ensure the class name persists in the corrected code
                        if new_class_name not in new_code:
                            new_code = new_code.replace(
                                parent_class.__name__, new_class_name, 1
                            )
                    else:
                        print("Max repair attempts reached. Discarding this mutation.")

            if not success:
                continue  # Move to next generation

            # 14. Compute fitness of new strategy
            P_new = res["Return (Ann.) [%]"]
            print(
                f"New strategy variant '{new_class_name}' achieved validation "
                f"annual return {P_new:.2f}%"
            )

            # 15-18. Check against MAS threshold
            if P_new is not None and P_new >= MAS:
                # Accept new strategy into population
                population.append((NewStrategyClass, P_new))
                print(
                    f"Accepted new strategy (>= MAS={MAS:.2f}%). "
                    f"Population size now {len(population)}."
                )

                # Update best if this is highest so far
                if P_new > best_perf:
                    best_perf = P_new
                    best_strategy_class = NewStrategyClass
            else:
                print(f"Discarded new strategy (did not meet MAS={MAS:.2f}%).")

        print(
            f"\nEvolution complete. Best strategy '{best_strategy_class.__name__}' "
            f"validation return = {best_perf:.2f}%."
        )
        return best_strategy_class, best_perf

    def walk_forward_optimize(
        self, full_data: pd.DataFrame, strategy_class, n_folds: int = 5
    ) -> list[dict]:
        """Perform walk-forward optimization across multiple folds.

        Evolves the strategy on each training/validation set, then evaluates
        on the test set. Compares performance against baseline strategies.

        Args:
            full_data: Complete historical DataFrame with datetime index and OHLCV columns.
            strategy_class: Seed strategy class to evolve.
            n_folds: Number of walk-forward folds (default: 5).

        Returns:
            List of per-fold result dictionaries containing:
            - fold: Fold number (1-indexed)
            - strategy: Best evolved strategy class
            - ann_return: Annualized return on test (%)
            - sharpe: Sharpe ratio on test
            - expectancy: Expectancy on test (%)
            - random_return: Random baseline return (%)
            - buy_hold_return: Buy-and-hold baseline return (%)
        """
        folds = self.prepare_folds(full_data, n_folds=n_folds)
        results = []

        for i, (train, val, test) in enumerate(folds, start=1):
            print(f"\n=== Fold {i} ===")
            print(f"Training period: {train.index[0]} to {train.index[-1]}")
            print(f"Validation period: {val.index[0]} to {val.index[-1]}")
            print(f"Test period: {test.index[0]} to {test.index[-1]}")

            # Evolve strategy on this fold's data
            best_strat, _ = self.evolve_strategy(strategy_class, train, val)

            # Evaluate best strategy on test set
            metrics, res = self.run_backtest(best_strat, test)
            ann_return = metrics["AnnReturn%"]
            sharpe = metrics["Sharpe"]
            expectancy = metrics["Expectancy%"]
            print(
                f"Fold {i} Test Performance - Annualized Return: {ann_return:.2f}%, "
                f"Sharpe: {sharpe:.2f}, Expectancy: {expectancy:.2f}%"
            )

            # Also evaluate baselines on the test set for comparison
            _, res_rand = self.run_backtest(RandomStrategy, test)
            _, res_bh = self.run_backtest(BuyAndHoldStrategy, test)
            rand_return = res_rand["Return (Ann.) [%]"]
            bh_return = res_bh["Return (Ann.) [%]"]
            print(
                f"Fold {i} Baselines - Random Strat Return: {rand_return:.2f}%, "
                f"Buy&Hold Return: {bh_return:.2f}%"
            )

            results.append({
                "fold": i,
                "strategy": best_strat,
                "ann_return": ann_return,
                "sharpe": sharpe,
                "expectancy": expectancy,
                "random_return": rand_return,
                "buy_hold_return": bh_return,
            })

        # Summarize across folds
        avg_ret = np.mean([r["ann_return"] for r in results])
        avg_bh = np.mean([r["buy_hold_return"] for r in results])
        avg_rand = np.mean([r["random_return"] for r in results])

        print(f"\nAverage Annualized Return over {len(results)} folds: {avg_ret:.2f}%")
        print(f"Average Buy-and-Hold Return over {len(results)} folds: {avg_bh:.2f}%")
        print(f"Average Random Strategy Return over {len(results)} folds: {avg_rand:.2f}%")

        return results
