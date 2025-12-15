#!/usr/bin/env python
"""ProFiT: LLM-Driven Evolutionary Trading System

Usage:
    uv run python -m profit.main --data data/ES_hourly.csv --strategy EMACrossover --provider openai --model gpt-4
"""

import argparse
import sys
from pathlib import Path

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
        strat = res["strategy"]
        db_id = getattr(strat, "_db_id", None) or ""
        id_str = f"[{db_id}] " if db_id else ""
        print(f"\nFold {res['fold']}:")
        print(f"  Best Strategy: {id_str}{strat.__name__}")
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
    parser.add_argument("--data", help="Path to OHLCV CSV file (required unless using --export-strategy)")
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
        default=None,
        help="[DEPRECATED] Directory for legacy file persistence. Use program_db instead.",
    )
    parser.add_argument(
        "--no-finalize-trades",
        action="store_true",
        help="Don't auto-close open trades at backtest end (shows warnings)",
    )
    # Program database arguments (Phase 13C)
    parser.add_argument(
        "--db-backend",
        choices=["json", "sqlite"],
        default="json",
        help="Program database backend (default: json)",
    )
    parser.add_argument(
        "--db-path",
        default="program_db",
        help="Path for program database (default: program_db)",
    )
    parser.add_argument(
        "--no-inspirations",
        action="store_true",
        help="Disable inspiration sampling from program database",
    )
    # Phase 14: Diff-based mutations
    parser.add_argument(
        "--no-diffs",
        action="store_true",
        help="Disable diff-based mutations (use full rewrites only)",
    )
    parser.add_argument(
        "--diff-match",
        choices=["strict", "tolerant"],
        default="tolerant",
        help="Diff matching mode: strict (literal) or tolerant (normalized, default)",
    )
    parser.add_argument(
        "--diff-mode",
        choices=["always", "never", "adaptive"],
        default="adaptive",
        help="When to use diffs: always, never, or adaptive (default)",
    )
    parser.add_argument(
        "--exploration-gens",
        type=int,
        default=5,
        help="In adaptive mode, use full rewrites for first N generations (default: 5)",
    )

    # Phase 15: Selection Policy Arguments
    parser.add_argument(
        "--selection-policy",
        choices=["weighted", "gated", "pareto"],
        default=None,
        help="Selection policy for strategy acceptance (default: MAS threshold)",
    )
    # Primary thresholds (for gated policy)
    parser.add_argument(
        "--min-return",
        type=float,
        default=0.0,
        help="Minimum annualized return threshold (default: 0.0)",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Minimum Sharpe ratio threshold (default: 0.0)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=-50.0,
        help="Maximum drawdown threshold, negative (default: -50.0)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=1,
        help="Minimum number of trades required (default: 1)",
    )
    # Robustness thresholds (for gated policy)
    parser.add_argument(
        "--min-consistency",
        type=float,
        default=0.0,
        help="Minimum %% of folds with positive return (default: 0.0)",
    )
    parser.add_argument(
        "--min-worst-fold",
        type=float,
        default=-100.0,
        help="Minimum return for worst fold (default: -100.0)",
    )
    parser.add_argument(
        "--max-stability",
        type=float,
        default=100.0,
        help="Maximum cross-fold std dev (default: 100.0)",
    )
    # WeightedSum policy weights
    parser.add_argument(
        "--w-return",
        type=float,
        default=0.5,
        help="Weight for return in weighted policy (default: 0.5)",
    )
    parser.add_argument(
        "--w-sharpe",
        type=float,
        default=0.3,
        help="Weight for Sharpe in weighted policy (default: 0.3)",
    )
    parser.add_argument(
        "--w-drawdown",
        type=float,
        default=0.2,
        help="Weight for drawdown in weighted policy (default: 0.2)",
    )
    # Pareto objectives
    parser.add_argument(
        "--pareto-objectives",
        nargs="+",
        default=["ann_return", "sharpe", "max_drawdown"],
        help="Objectives for Pareto selection (default: ann_return sharpe max_drawdown)",
    )
    # Debug flag for selection policy
    parser.add_argument(
        "--debug-policy",
        action="store_true",
        help="Enable debug logging for selection policy decisions (Pareto dominance checks)",
    )
    # Cascade configuration
    parser.add_argument(
        "--skip-cascade",
        action="store_true",
        help="Skip evaluation cascade (use direct backtest only)",
    )
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        help="Use quick evaluation (syntax + smoke test only)",
    )
    parser.add_argument(
        "--smoke-months",
        type=int,
        default=3,
        help="Months of data for smoke test (default: 3)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annual risk-free rate for Sharpe/Sortino (default: 0.0)",
    )
    # Promotion gate thresholds
    parser.add_argument(
        "--gate-min-trades",
        type=int,
        default=1,
        help="Promotion gate: minimum trades (default: 1)",
    )
    parser.add_argument(
        "--gate-max-drawdown",
        type=float,
        default=-80.0,
        help="Promotion gate: max drawdown limit (default: -80.0)",
    )
    parser.add_argument(
        "--gate-min-sharpe",
        type=float,
        default=None,
        help="Promotion gate: minimum Sharpe ratio (default: None = disabled)",
    )
    parser.add_argument(
        "--gate-min-win-rate",
        type=float,
        default=None,
        help="Promotion gate: minimum win rate %% (default: None = disabled)",
    )

    # Export command (alternative to deprecated output_dir)
    parser.add_argument(
        "--export-strategy",
        metavar="ID",
        help="Export a strategy from program_db to a .py file by its ID (then exit)",
    )
    parser.add_argument(
        "--export-dir",
        default="exported_strategies",
        help="Directory for exported strategies (default: exported_strategies)",
    )

    args = parser.parse_args()

    # Handle export command (doesn't require data file)
    if args.export_strategy:
        from profit.program_db import JsonFileBackend, ProgramDatabase, SqliteBackend

        # Initialize database backend
        if args.db_backend == "sqlite":
            backend = SqliteBackend(args.db_path + ".sqlite")
        else:
            backend = JsonFileBackend(args.db_path)

        db = ProgramDatabase(backend)
        record = db.get_strategy(args.export_strategy)

        if not record:
            print(f"Error: Strategy '{args.export_strategy}' not found in database")
            return 1

        # Create export directory
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / f"{record.class_name}.py"

        # Build header with metadata
        header = f'''"""Exported Strategy: {record.class_name}

DB ID: {record.id}
Generation: {record.generation}
Status: {record.status.value}
Metrics: {record.metrics}
Parent IDs: {record.parent_ids}
Created: {record.created_at}
"""

import numpy as np
import pandas as pd
from backtesting import Strategy


'''
        export_path.write_text(header + record.code)
        print(f"Exported [{record.id}] {record.class_name} to {export_path}")
        return 0

    # Load data (required for evolution)
    if not args.data:
        print("Error: --data is required for evolution. Use --export-strategy for export mode.")
        return 1

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

    # Initialize program database (Phase 13C)
    program_db = None
    if not args.no_inspirations:
        from profit.program_db import ProgramDatabase, JsonFileBackend, SqliteBackend

        if args.db_backend == "sqlite":
            db_path = (
                args.db_path
                if args.db_path.endswith(".sqlite")
                else f"{args.db_path}.sqlite"
            )
            backend = SqliteBackend(db_path)
        else:
            backend = JsonFileBackend(args.db_path)

        program_db = ProgramDatabase(backend)
        print(f"Program database: {args.db_backend} backend at {args.db_path}")

    # Initialize evolver (output_dir is deprecated, use program_db instead)
    output_dir = None
    if args.output_dir is not None and args.output_dir.lower() != "none":
        output_dir = args.output_dir
    evolver = ProfitEvolver(
        llm_client,
        initial_capital=args.capital,
        commission=args.commission,
        output_dir=output_dir,
        finalize_trades=not args.no_finalize_trades,
        program_db=program_db,
        # Phase 14: Diff-based mutations
        prefer_diffs=not args.no_diffs,
        diff_mode=args.diff_mode,
        diff_match=args.diff_match,
        exploration_gens=args.exploration_gens,
    )

    # Log diff settings
    if not args.no_diffs:
        print(f"Diff mutations: mode={args.diff_mode}, match={args.diff_match}")
        if args.diff_mode == "adaptive":
            print(f"  Exploration generations: {args.exploration_gens}")
    else:
        print("Diff mutations: disabled (using full rewrites)")

    # Get strategy class
    strategy_class = STRATEGIES[args.strategy]
    print(f"Using seed strategy: {strategy_class.__name__}")

    # Phase 15: Build selection policy if specified
    selection_policy = None
    if args.selection_policy:
        from profit.evaluation import create_selection_policy

        selection_policy = create_selection_policy(
            args.selection_policy,
            # Gated policy parameters
            min_return=args.min_return,
            min_sharpe=args.min_sharpe,
            max_drawdown=args.max_drawdown,
            min_trades=args.min_trades,
            min_consistency=args.min_consistency,
            min_worst_fold=args.min_worst_fold,
            max_stability=args.max_stability,
            # Weighted policy parameters
            w_return=args.w_return,
            w_sharpe=args.w_sharpe,
            w_drawdown=args.w_drawdown,
            # Pareto parameters
            objectives=args.pareto_objectives,
            # Debug flag
            debug=args.debug_policy,
        )
        print(f"Selection policy: {args.selection_policy}")
        if args.debug_policy:
            print("  Debug logging: ENABLED")

    # Phase 15: Build evaluation cascade if not skipped
    cascade = None
    if not args.skip_cascade and selection_policy is not None:
        from profit.evaluation import (
            MetricsCalculator,
            PromotionGate,
            create_cascade,
        )

        metrics_calc = MetricsCalculator(risk_free_rate=args.risk_free_rate)
        promotion_gate = PromotionGate(
            min_trades=args.gate_min_trades,
            max_drawdown_limit=args.gate_max_drawdown,
            min_sharpe=args.gate_min_sharpe,
            min_win_rate=args.gate_min_win_rate,
        )

        cascade_mode = "quick" if args.quick_eval else "standard"
        cascade = create_cascade(
            mode=cascade_mode,
            metrics_calculator=metrics_calc,
            promotion_gate=promotion_gate,
            smoke_months=args.smoke_months,
            initial_capital=args.capital,
            commission=args.commission,
            verbose=True,
        )
        print(f"Evaluation cascade: {cascade_mode} mode")

        # Log promotion gate settings if non-default
        gate_settings = []
        if args.gate_min_trades != 1:
            gate_settings.append(f"min_trades={args.gate_min_trades}")
        if args.gate_max_drawdown != -80.0:
            gate_settings.append(f"max_dd={args.gate_max_drawdown}%")
        if args.gate_min_sharpe is not None:
            gate_settings.append(f"min_sharpe={args.gate_min_sharpe}")
        if args.gate_min_win_rate is not None:
            gate_settings.append(f"min_win_rate={args.gate_min_win_rate}%")
        if gate_settings:
            print(f"Promotion gate: {', '.join(gate_settings)}")

    # Run walk-forward optimization
    print(f"\nStarting walk-forward optimization with {args.folds} folds...")
    results = evolver.walk_forward_optimize(
        data,
        strategy_class,
        n_folds=args.folds,
        selection_policy=selection_policy,
        cascade=cascade,
        use_inspirations=not args.no_inspirations,
    )

    # Print results
    print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
