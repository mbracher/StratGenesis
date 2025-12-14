#!/usr/bin/env python
"""ProFiT: LLM-Driven Evolutionary Trading System

Usage:
    uv run python -m profit.main --data data/ES_hourly.csv --strategy EMACrossover --provider openai --model gpt-4
"""

import argparse
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from profit.evolver import ProfitEvolver
from profit.llm_interface import LLMClient
from profit.strategies import (
    BollingerMeanReversion,
    CCIStrategy,
    EMACrossover,
    MACDStrategy,
    WilliamsRStrategy,
)

# Map of strategy names to classes
STRATEGIES = {
    "BollingerMeanReversion": BollingerMeanReversion,
    "CCIStrategy": CCIStrategy,
    "EMACrossover": EMACrossover,
    "MACDStrategy": MACDStrategy,
    "WilliamsRStrategy": WilliamsRStrategy,
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from CSV file.

    Args:
        filepath: Path to CSV file with OHLCV data.

    Returns:
        DataFrame with datetime index and OHLCV columns.

    Raises:
        ValueError: If required columns are missing or index is not datetime.
        FileNotFoundError: If the file doesn't exist.
    """
    data = pd.read_csv(filepath, parse_dates=True, index_col=0)

    # Validate required columns
    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be datetime")

    return data


def print_results(results: list[dict]) -> None:
    """Print formatted results summary.

    Args:
        results: List of per-fold result dictionaries from walk_forward_optimize.
    """
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for res in results:
        print(f"\nFold {res['fold']}:")
        print(f"  Best Strategy: {res['strategy'].__name__}")
        print(f"  Annualized Return: {res['ann_return']:.2f}%")
        print(f"  Sharpe Ratio: {res['sharpe']:.2f}")
        print(f"  Expectancy: {res['expectancy']:.2f}%")
        print(f"  vs Random: {res['ann_return'] - res['random_return']:+.2f}%")
        print(f"  vs Buy&Hold: {res['ann_return'] - res['buy_hold_return']:+.2f}%")

    # Aggregate stats
    avg_ret = np.mean([r["ann_return"] for r in results])
    avg_bh = np.mean([r["buy_hold_return"] for r in results])
    avg_rand = np.mean([r["random_return"] for r in results])

    print("\n" + "-" * 60)
    print("AVERAGES ACROSS FOLDS:")
    print(f"  Evolved Strategy: {avg_ret:.2f}%")
    print(f"  Buy-and-Hold: {avg_bh:.2f}%")
    print(f"  Random: {avg_rand:.2f}%")
    print(f"  Improvement over B&H: {avg_ret - avg_bh:+.2f}%")
    print(f"  Improvement over Random: {avg_ret - avg_rand:+.2f}%")


def main() -> int:
    """Main entry point for ProFiT CLI.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="ProFiT: LLM-Driven Evolutionary Trading System"
    )
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    parser.add_argument(
        "--strategy",
        default="EMACrossover",
        choices=STRATEGIES.keys(),
        help="Seed strategy to evolve",
    )
    # LLM configuration - default provider/model
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="Default LLM provider for both roles (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Default model for both roles (uses provider default if not set)",
    )
    # Role-specific configuration
    parser.add_argument(
        "--analyst-provider",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider for analysis/improvements (overrides --provider)",
    )
    parser.add_argument(
        "--analyst-model",
        default=None,
        help="Model for analysis/improvements (overrides --model)",
    )
    parser.add_argument(
        "--coder-provider",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider for code generation (overrides --provider)",
    )
    parser.add_argument(
        "--coder-model",
        default=None,
        help="Model for code generation (overrides --model)",
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of walk-forward folds"
    )
    parser.add_argument(
        "--capital", type=float, default=10000, help="Initial capital"
    )
    parser.add_argument(
        "--commission", type=float, default=0.002, help="Commission rate"
    )
    parser.add_argument(
        "--output-dir",
        default="evolved_strategies",
        help="Directory to save evolved strategies (use 'none' to disable)",
    )
    parser.add_argument(
        "--no-finalize-trades",
        action="store_true",
        help="Don't auto-close open trades at backtest end (shows warnings)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    try:
        data = load_data(args.data)
    except FileNotFoundError:
        print(f"Error: File not found: {args.data}")
        return 1
    except ValueError as e:
        print(f"Error: Invalid data format: {e}")
        return 1

    print(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    # Initialize LLM client with dual-model support
    llm_client = LLMClient(
        provider=args.provider,
        model=args.model,
        analyst_provider=args.analyst_provider,
        analyst_model=args.analyst_model,
        coder_provider=args.coder_provider,
        coder_model=args.coder_model,
    )
    print(f"LLM Analyst: {llm_client.analyst_provider}/{llm_client.analyst_model}")
    print(f"LLM Coder: {llm_client.coder_provider}/{llm_client.coder_model}")

    # Initialize evolver
    output_dir = None if args.output_dir.lower() == "none" else args.output_dir
    evolver = ProfitEvolver(
        llm_client,
        initial_capital=args.capital,
        commission=args.commission,
        output_dir=output_dir,
        finalize_trades=not args.no_finalize_trades,
    )

    # Get strategy class
    strategy_class = STRATEGIES[args.strategy]
    print(f"Using seed strategy: {strategy_class.__name__}")

    # Run walk-forward optimization
    print(f"\nStarting walk-forward optimization with {args.folds} folds...")
    results = evolver.walk_forward_optimize(data, strategy_class, n_folds=args.folds)

    # Print results
    print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
