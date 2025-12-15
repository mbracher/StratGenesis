"""
Multi-Metric Scoring & Evaluation Cascade for ProFiT.

This module implements:
- StrategyMetrics dataclass with comprehensive metrics
- MetricsCalculator for computing metrics from backtest results
- Evaluation cascade with fast rejection stages
- Selection policies for multi-objective optimization

Based on AlphaEvolve's evaluation cascade pattern.
"""

from __future__ import annotations

import ast
import hashlib
import math
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, Type

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

if TYPE_CHECKING:
    pass


# =============================================================================
# Configurable caps for infinite values
# =============================================================================

METRIC_CAPS = {
    "sortino": 10.0,
    "calmar": 10.0,
    "profit_factor": 100.0,
}


# =============================================================================
# Core Helpers
# =============================================================================


def load_strategy_class(
    strategy_code: str,
    exec_globals: dict = None,
) -> Type[Strategy]:
    """
    Load and validate a strategy class from source code.

    Args:
        strategy_code: Python source code defining a Strategy subclass
        exec_globals: Global namespace for exec (defaults to Strategy, pd, np)

    Returns:
        The Strategy subclass (not an instance)

    Raises:
        ValueError: If no valid Strategy subclass found
        SyntaxError: If code has syntax errors
    """
    if exec_globals is None:
        exec_globals = {
            "Strategy": Strategy,
            "pd": pd,
            "np": np,
        }

    namespace = {}
    exec(strategy_code, exec_globals, namespace)

    # Find the Strategy subclass (strict check)
    strategy_class = None
    for name, obj in namespace.items():
        if (
            isinstance(obj, type)
            and issubclass(obj, Strategy)
            and obj is not Strategy
        ):
            strategy_class = obj
            break

    if strategy_class is None:
        raise ValueError(
            "No valid Strategy subclass found in code. "
            "Class must inherit from backtesting.Strategy."
        )

    return strategy_class


def run_bt(
    strategy_class: Type[Strategy],
    data: pd.DataFrame,
    cash: float = 10000,
    commission: float = 0.002,
    finalize_trades: bool = True,
) -> pd.Series:
    """
    Run backtest and return full result series.

    Args:
        strategy_class: Strategy subclass to backtest
        data: OHLCV DataFrame
        cash: Initial capital
        commission: Per-trade commission rate
        finalize_trades: If True, auto-close open trades at backtest end

    Returns:
        backtesting.py result Series with all metrics
    """
    bt = Backtest(
        data,
        strategy_class,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=finalize_trades,
    )
    return bt.run()


def code_hash(strategy_code: str) -> str:
    """Compute SHA256 hash of strategy code for caching."""
    return hashlib.sha256(strategy_code.encode()).hexdigest()[:16]


# =============================================================================
# Evaluation Cache
# =============================================================================


@dataclass(frozen=True)
class CacheKey:
    """Key for evaluation cache."""

    code_hash: str
    stage_name: str
    data_window_id: str  # e.g., "fold1_val" or hash of data slice
    config_hash: str  # hash of cash, commission, etc.


class EvaluationCache:
    """
    In-memory cache for evaluation results.

    Keyed by code hash + stage + data window + config.
    Prevents redundant evaluations during evolution loops.
    """

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[CacheKey, "StageOutput"] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: CacheKey) -> Optional["StageOutput"]:
        """Get cached result if exists."""
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, key: CacheKey, result: "StageOutput") -> None:
        """Store result in cache."""
        if len(self._cache) >= self._max_size:
            # Simple eviction: remove oldest entries
            oldest_keys = list(self._cache.keys())[: self._max_size // 4]
            for k in oldest_keys:
                del self._cache[k]
        self._cache[key] = result

    def make_key(
        self,
        strategy_code: str,
        stage_name: str,
        data: pd.DataFrame,
        cash: float,
        commission: float,
    ) -> CacheKey:
        """Create cache key from inputs."""
        code_h = code_hash(strategy_code)

        # Data window ID from shape + first/last dates
        data_id = f"{len(data)}_{data.index[0]}_{data.index[-1]}"
        data_h = hashlib.sha256(data_id.encode()).hexdigest()[:8]

        # Config hash
        config_str = f"{cash}_{commission}"
        config_h = hashlib.sha256(config_str.encode()).hexdigest()[:8]

        return CacheKey(code_h, stage_name, data_h, config_h)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# =============================================================================
# Metrics Model
# =============================================================================


def _clip_metric(value: float, metric_name: str) -> float:
    """Clip infinite values to configured caps."""
    if math.isinf(value):
        cap = METRIC_CAPS.get(metric_name, 100.0)
        return cap if value > 0 else -cap
    return value


@dataclass
class StrategyMetrics:
    """
    Comprehensive metrics for strategy evaluation.

    Primary metrics are used for fitness. Secondary metrics
    provide additional context for analysis.

    Note: max_drawdown is stored as a NEGATIVE value (e.g., -0.15 = 15% drawdown).
    Higher (closer to 0) is better.
    """

    # === Primary Metrics (used for fitness) ===
    ann_return: float = 0.0  # Annualized return (%)
    sharpe: float = 0.0  # Sharpe ratio (risk-adjusted return)
    max_drawdown: float = 0.0  # Maximum drawdown (%, NEGATIVE, e.g., -15.0)
    sortino: float = 0.0  # Sortino ratio (downside risk) - capped
    calmar: float = 0.0  # Calmar ratio (return / |max drawdown|) - capped

    # === Secondary Metrics (context/analysis) ===
    trade_count: int = 0  # Number of trades
    win_rate: float = 0.0  # Winning trades / total trades (%)
    profit_factor: float = 0.0  # Gross profit / gross loss - capped
    expectancy: float = 0.0  # Expected profit per trade (%)
    exposure_time: float = 0.0  # Time in market (%)

    # === Robustness Metrics (cross-fold) ===
    stability: float = 0.0  # Cross-fold std dev of returns (lower is better)
    consistency: float = 0.0  # % of folds with positive return
    worst_fold_return: float = 0.0  # Worst single fold performance

    # === Raw data for custom calculations ===
    equity_curve: List[float] = field(default_factory=list)
    trade_returns: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (excludes lists)."""
        return {
            "ann_return": self.ann_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "exposure_time": self.exposure_time,
            "stability": self.stability,
            "consistency": self.consistency,
            "worst_fold_return": self.worst_fold_return,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "StrategyMetrics":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_valid(self) -> bool:
        """Check if metrics contain valid (non-nan, non-inf) values."""
        for key, val in self.to_dict().items():
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return False
        return True


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""

    fold: int
    train_metrics: StrategyMetrics
    val_metrics: StrategyMetrics
    test_metrics: Optional[StrategyMetrics] = None


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all folds."""

    mean: StrategyMetrics
    std: StrategyMetrics
    min: StrategyMetrics
    max: StrategyMetrics
    fold_metrics: List[FoldMetrics] = field(default_factory=list)


# =============================================================================
# MetricsCalculator
# =============================================================================


class MetricsCalculator:
    """
    Calculates comprehensive strategy metrics from backtest results.

    Uses backtesting.py result structure but computes additional
    metrics for multi-objective optimization.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 0)
        """
        self.risk_free_rate = risk_free_rate

    def compute_all(self, backtest_result: pd.Series) -> StrategyMetrics:
        """
        Compute all metrics from a backtest result.

        Args:
            backtest_result: Result Series from backtesting.py Backtest.run()

        Returns:
            StrategyMetrics with all computed values (infinites capped)
        """
        # Extract raw data
        equity = backtest_result.get("_equity_curve", pd.DataFrame())
        trades = backtest_result.get("_trades", pd.DataFrame())

        # Basic metrics (already computed by backtesting.py)
        ann_return = float(backtest_result.get("Return (Ann.) [%]", 0.0))
        sharpe = float(backtest_result.get("Sharpe Ratio", 0.0))
        max_drawdown = float(backtest_result.get("Max. Drawdown [%]", 0.0))
        exposure_time = float(backtest_result.get("Exposure Time [%]", 0.0))

        # Trade-based metrics
        trade_count = int(backtest_result.get("# Trades", 0))
        win_rate = float(backtest_result.get("Win Rate [%]", 0.0))
        expectancy = float(backtest_result.get("Expectancy [%]", 0.0))

        # Compute additional metrics (with capping)
        sortino = self._compute_sortino(equity, ann_return)
        sortino = _clip_metric(sortino, "sortino")

        calmar = self._compute_calmar(ann_return, max_drawdown)
        calmar = _clip_metric(calmar, "calmar")

        profit_factor = self._compute_profit_factor(trades)
        profit_factor = _clip_metric(profit_factor, "profit_factor")

        # Handle NaN sharpe
        if np.isnan(sharpe):
            sharpe = 0.0

        # Extract equity curve and trade returns
        equity_curve = []
        trade_returns = []

        if not equity.empty and "Equity" in equity.columns:
            equity_curve = equity["Equity"].tolist()

        if not trades.empty and "ReturnPct" in trades.columns:
            trade_returns = trades["ReturnPct"].tolist()

        return StrategyMetrics(
            ann_return=ann_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,  # Already negative from backtesting.py
            sortino=sortino,
            calmar=calmar,
            trade_count=trade_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            exposure_time=exposure_time,
            equity_curve=equity_curve,
            trade_returns=trade_returns,
        )

    def _compute_sortino(
        self,
        equity: pd.DataFrame,
        ann_return: float,
    ) -> float:
        """Compute Sortino ratio (downside deviation only)."""
        if equity.empty or "Equity" not in equity.columns:
            return 0.0

        returns = equity["Equity"].pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        # Only consider negative returns for downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            # No negative returns - return capped positive value
            return METRIC_CAPS["sortino"] if ann_return > 0 else 0.0

        downside_std = negative_returns.std() * np.sqrt(252)  # Annualize

        if downside_std == 0:
            return 0.0

        return (ann_return / 100 - self.risk_free_rate) / downside_std

    def _compute_calmar(self, ann_return: float, max_drawdown: float) -> float:
        """Compute Calmar ratio (return / |max drawdown|)."""
        if max_drawdown == 0:
            return 0.0
        return ann_return / abs(max_drawdown)

    def _compute_profit_factor(self, trades: pd.DataFrame) -> float:
        """Compute profit factor (gross profit / gross loss)."""
        if trades.empty or "PnL" not in trades.columns:
            return 0.0

        profits = trades[trades["PnL"] > 0]["PnL"].sum()
        losses = abs(trades[trades["PnL"] < 0]["PnL"].sum())

        if losses == 0:
            # No losses - return capped value
            return METRIC_CAPS["profit_factor"] if profits > 0 else 0.0

        return profits / losses

    def compute_cross_fold_metrics(
        self,
        fold_metrics: List[StrategyMetrics],
    ) -> StrategyMetrics:
        """
        Compute stability and consistency metrics across folds.

        Args:
            fold_metrics: List of metrics from each fold

        Returns:
            StrategyMetrics with stability/consistency populated
        """
        if not fold_metrics:
            return StrategyMetrics()

        returns = [m.ann_return for m in fold_metrics]

        stability = float(np.std(returns)) if len(returns) > 1 else 0.0
        consistency = sum(1 for r in returns if r > 0) / len(returns) * 100
        worst_fold = min(returns)

        # Compute mean of all metrics
        mean_metrics = StrategyMetrics(
            ann_return=float(np.mean(returns)),
            sharpe=float(np.mean([m.sharpe for m in fold_metrics])),
            max_drawdown=float(np.mean([m.max_drawdown for m in fold_metrics])),
            sortino=float(np.mean([m.sortino for m in fold_metrics])),
            calmar=float(np.mean([m.calmar for m in fold_metrics])),
            trade_count=int(np.mean([m.trade_count for m in fold_metrics])),
            win_rate=float(np.mean([m.win_rate for m in fold_metrics])),
            profit_factor=float(np.mean([m.profit_factor for m in fold_metrics])),
            expectancy=float(np.mean([m.expectancy for m in fold_metrics])),
            exposure_time=float(np.mean([m.exposure_time for m in fold_metrics])),
            stability=stability,
            consistency=consistency,
            worst_fold_return=worst_fold,
        )

        return mean_metrics


def evaluate_on_data(
    strategy_code: str,
    data: pd.DataFrame,
    metrics_calculator: MetricsCalculator = None,
    cash: float = 10000,
    commission: float = 0.002,
) -> StrategyMetrics:
    """
    Full evaluation pipeline: load, backtest, compute metrics.

    Args:
        strategy_code: Strategy source code
        data: OHLCV DataFrame
        metrics_calculator: Optional calculator (creates default if None)
        cash: Initial capital
        commission: Per-trade commission

    Returns:
        StrategyMetrics with all computed values

    Raises:
        ValueError: If strategy invalid
        Exception: If backtest fails
    """
    strategy_class = load_strategy_class(strategy_code)
    result = run_bt(strategy_class, data, cash, commission)

    if metrics_calculator is None:
        metrics_calculator = MetricsCalculator()

    return metrics_calculator.compute_all(result)


# =============================================================================
# Evaluation Cascade
# =============================================================================


class StageResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class StageOutput:
    """Output from an evaluation stage."""

    result: StageResult
    metrics: Optional[StrategyMetrics] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize for Program DB storage."""
        return {
            "result": self.result.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class EvaluationStage(Protocol):
    """Protocol for evaluation stages."""

    name: str
    order: int  # Lower = earlier in cascade

    @abstractmethod
    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        cache: Optional[EvaluationCache] = None,
        **kwargs,
    ) -> StageOutput:
        """Run this evaluation stage."""
        ...


@dataclass
class PromotionGate:
    """
    Threshold gate for promoting strategies between stages.

    Applied after Stage 3 (SingleFoldStage) to prevent junk
    strategies from proceeding to expensive Stage 4.
    """

    min_trades: int = 1  # At least N trades required
    max_drawdown_limit: float = -80.0  # Max drawdown must be > this (e.g., -80%)
    min_sharpe: Optional[float] = None  # Optional min Sharpe
    min_win_rate: Optional[float] = None  # Optional min win rate (%)

    def check(self, metrics: StrategyMetrics) -> Tuple[bool, Optional[str]]:
        """
        Check if metrics pass the gate.

        Returns:
            (passed, error_message) - error_message is None if passed
        """
        if metrics.trade_count < self.min_trades:
            return False, f"Too few trades: {metrics.trade_count} < {self.min_trades}"

        if metrics.max_drawdown < self.max_drawdown_limit:
            return (
                False,
                f"Drawdown too severe: {metrics.max_drawdown}% < {self.max_drawdown_limit}%",
            )

        # Check for invalid values (NaN/Inf) only on critical metrics
        # Note: win_rate can be NaN when trade_count=0, which is expected
        for metric_name in ["ann_return", "sharpe", "max_drawdown"]:
            val = getattr(metrics, metric_name)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return False, f"Critical metric {metric_name} is NaN or Inf"

        if self.min_sharpe is not None and metrics.sharpe < self.min_sharpe:
            return False, f"Sharpe too low: {metrics.sharpe:.2f} < {self.min_sharpe}"

        if self.min_win_rate is not None:
            # Only check win_rate if it's not NaN (i.e., there were trades)
            if not math.isnan(metrics.win_rate) and metrics.win_rate < self.min_win_rate:
                return (
                    False,
                    f"Win rate too low: {metrics.win_rate:.1f}% < {self.min_win_rate}%",
                )

        return True, None


# =============================================================================
# Stage Implementations
# =============================================================================


class SyntaxCheckStage:
    """Stage 1: Parse and compile code (instant)."""

    name = "syntax_check"
    order = 1

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame = None,
        cache: Optional[EvaluationCache] = None,
        **kwargs,
    ) -> StageOutput:
        start = time.time()

        try:
            # Parse AST
            tree = ast.parse(strategy_code)

            # Check for class definition
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            if not classes:
                return StageOutput(
                    result=StageResult.FAIL,
                    error="No class definition found",
                    duration_ms=(time.time() - start) * 1000,
                )

            # Check for required methods
            methods = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    methods.add(node.name)

            required = {"init", "next"}
            missing = required - methods
            if missing:
                return StageOutput(
                    result=StageResult.FAIL,
                    error=f"Missing methods: {missing}",
                    duration_ms=(time.time() - start) * 1000,
                )

            # Try to compile
            compile(strategy_code, "<string>", "exec")

            return StageOutput(
                result=StageResult.PASS,
                duration_ms=(time.time() - start) * 1000,
            )

        except SyntaxError as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


class SmokeTestStage:
    """Stage 2: Run on small data slice (~1s)."""

    name = "smoke_test"
    order = 2

    def __init__(
        self,
        slice_months: int = 3,
        initial_capital: float = 10000,
        commission: float = 0.002,
    ):
        self.slice_months = slice_months
        self.initial_capital = initial_capital
        self.commission = commission

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        cache: Optional[EvaluationCache] = None,
        **kwargs,
    ) -> StageOutput:
        start = time.time()

        # Check cache
        if cache:
            key = cache.make_key(
                strategy_code,
                self.name,
                data,
                self.initial_capital,
                self.commission,
            )
            cached = cache.get(key)
            if cached:
                return cached

        try:
            # Take a small slice of data
            slice_days = self.slice_months * 21  # ~21 trading days per month
            if len(data) < slice_days:
                data_slice = data
            else:
                data_slice = data.iloc[:slice_days]

            # Use centralized helper (strict Strategy check)
            strategy_class = load_strategy_class(strategy_code)
            _ = run_bt(
                strategy_class,
                data_slice,
                self.initial_capital,
                self.commission,
            )

            result = StageOutput(
                result=StageResult.PASS,
                duration_ms=(time.time() - start) * 1000,
            )

        except ValueError as e:
            # From load_strategy_class - no valid Strategy found
            result = StageOutput(
                result=StageResult.FAIL,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            result = StageOutput(
                result=StageResult.FAIL,
                error=f"Smoke test failed: {str(e)}",
                duration_ms=(time.time() - start) * 1000,
            )

        # Store in cache
        if cache:
            cache.put(key, result)

        return result


class SingleFoldStage:
    """
    Stage 3: Run on single validation fold + promotion gate (~10s).

    This stage now includes a PromotionGate check after computing metrics.
    Strategies that pass backtest but fail the gate are REJECTED.
    """

    name = "single_fold"
    order = 3

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.002,
        metrics_calculator: MetricsCalculator = None,
        promotion_gate: PromotionGate = None,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.metrics_calc = metrics_calculator or MetricsCalculator()
        self.promotion_gate = promotion_gate or PromotionGate()

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        cache: Optional[EvaluationCache] = None,
        **kwargs,
    ) -> StageOutput:
        start = time.time()

        # Check cache
        if cache:
            key = cache.make_key(
                strategy_code,
                self.name,
                data,
                self.initial_capital,
                self.commission,
            )
            cached = cache.get(key)
            if cached:
                return cached

        try:
            # Use centralized helper
            strategy_class = load_strategy_class(strategy_code)
            bt_result = run_bt(
                strategy_class,
                data,
                self.initial_capital,
                self.commission,
            )

            # Compute metrics
            metrics = self.metrics_calc.compute_all(bt_result)

            # CHECK PROMOTION GATE
            gate_passed, gate_error = self.promotion_gate.check(metrics)
            if not gate_passed:
                result = StageOutput(
                    result=StageResult.FAIL,
                    metrics=metrics,
                    error=f"Promotion gate failed: {gate_error}",
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                result = StageOutput(
                    result=StageResult.PASS,
                    metrics=metrics,
                    duration_ms=(time.time() - start) * 1000,
                )

        except ValueError as e:
            result = StageOutput(
                result=StageResult.FAIL,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            result = StageOutput(
                result=StageResult.FAIL,
                error=f"Single fold failed: {str(e)}\n{traceback.format_exc()}",
                duration_ms=(time.time() - start) * 1000,
            )

        # Cache result
        if cache:
            cache.put(key, result)

        return result


class FullWalkForwardStage:
    """
    Stage 4: Run full walk-forward optimization (~60s).

    IMPORTANT: Requires folds parameter. Will FAIL if folds not provided.
    """

    name = "full_walkforward"
    order = 4

    def __init__(
        self,
        n_folds: int = 5,
        initial_capital: float = 10000,
        commission: float = 0.002,
        metrics_calculator: MetricsCalculator = None,
    ):
        self.n_folds = n_folds
        self.initial_capital = initial_capital
        self.commission = commission
        self.metrics_calc = metrics_calculator or MetricsCalculator()

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        folds: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None,
        cache: Optional[EvaluationCache] = None,
        **kwargs,
    ) -> StageOutput:
        start = time.time()

        # MUST HAVE FOLDS
        if folds is None or len(folds) == 0:
            return StageOutput(
                result=StageResult.FAIL,
                error="FullWalkForwardStage requires folds parameter with at least one fold",
                duration_ms=(time.time() - start) * 1000,
            )

        try:
            # Use centralized helper
            strategy_class = load_strategy_class(strategy_code)

            # Run across all folds
            fold_metrics = []

            for fold_num, (train, val, test) in enumerate(folds, 1):
                # Evaluate on validation data
                bt_result = run_bt(
                    strategy_class,
                    val,
                    self.initial_capital,
                    self.commission,
                )
                metrics = self.metrics_calc.compute_all(bt_result)
                fold_metrics.append(metrics)

            # Compute aggregate metrics (includes robustness)
            aggregate = self.metrics_calc.compute_cross_fold_metrics(fold_metrics)

            return StageOutput(
                result=StageResult.PASS,
                metrics=aggregate,
                duration_ms=(time.time() - start) * 1000,
            )

        except ValueError as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=f"Walk-forward failed: {str(e)}",
                duration_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# EvaluationCascade
# =============================================================================


@dataclass
class CascadeResult:
    """Result of running the evaluation cascade."""

    passed: bool
    final_stage: str
    metrics: Optional[StrategyMetrics]
    stage_results: Dict[str, StageOutput]
    total_duration_ms: float

    def to_dict(self) -> Dict:
        """Serialize for Program DB storage."""
        return {
            "passed": self.passed,
            "final_stage": self.final_stage,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "stage_results": {k: v.to_dict() for k, v in self.stage_results.items()},
            "total_duration_ms": self.total_duration_ms,
        }


class EvaluationCascade:
    """
    Runs evaluation stages in order, failing fast on early rejections.

    This implements the AlphaEvolve evaluation cascade pattern:
    cheap tests first, expensive tests only if earlier stages pass.
    """

    def __init__(
        self,
        stages: List[EvaluationStage] = None,
        cache: EvaluationCache = None,
        verbose: bool = True,
    ):
        """
        Args:
            stages: List of evaluation stages. Defaults to all stages.
            cache: Optional evaluation cache for speedup.
            verbose: Print progress messages.
        """
        if stages is None:
            stages = [
                SyntaxCheckStage(),
                SmokeTestStage(),
                SingleFoldStage(),
                FullWalkForwardStage(),
            ]

        # Sort by order
        self.stages = sorted(stages, key=lambda s: s.order)
        self.cache = cache or EvaluationCache()
        self.verbose = verbose

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        stop_at_stage: str = None,
        **kwargs,
    ) -> CascadeResult:
        """
        Run evaluation cascade on strategy code.

        Args:
            strategy_code: Strategy code to evaluate
            data: Market data for backtesting
            stop_at_stage: Optional stage name to stop at (for partial evaluation)
            **kwargs: Additional arguments passed to stages (e.g., folds)

        Returns:
            CascadeResult with pass/fail status and metrics
        """
        stage_results = {}
        total_duration = 0.0
        final_metrics = None
        passed = True
        final_stage = ""

        for stage in self.stages:
            final_stage = stage.name
            if self.verbose:
                print(f"  Running {stage.name}...", end=" ", flush=True)

            output = stage.evaluate(strategy_code, data, cache=self.cache, **kwargs)
            stage_results[stage.name] = output
            total_duration += output.duration_ms

            if output.result == StageResult.FAIL:
                if self.verbose:
                    print(f"FAIL ({output.duration_ms:.0f}ms)")
                    if output.error:
                        print(f"    Error: {output.error}")
                passed = False
                break
            else:
                if self.verbose:
                    print(f"PASS ({output.duration_ms:.0f}ms)")

            # Keep the most detailed metrics
            if output.metrics:
                final_metrics = output.metrics

            # Stop early if requested
            if stop_at_stage and stage.name == stop_at_stage:
                break

        return CascadeResult(
            passed=passed,
            final_stage=final_stage,
            metrics=final_metrics,
            stage_results=stage_results,
            total_duration_ms=total_duration,
        )

    def quick_evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
    ) -> CascadeResult:
        """Run only syntax and smoke test (for fast iteration)."""
        return self.evaluate(strategy_code, data, stop_at_stage="smoke_test")


# =============================================================================
# Selection Policies
# =============================================================================


class SelectionPolicy(Protocol):
    """
    Protocol for strategy acceptance policies.

    NOTE: should_accept receives optional population_metrics to support
    Pareto-based and archive-aware selection (AlphaEvolve style).
    """

    @abstractmethod
    def should_accept(
        self,
        candidate: StrategyMetrics,
        baseline: StrategyMetrics,
        population_metrics: Optional[List[StrategyMetrics]] = None,
        **kwargs,
    ) -> bool:
        """
        Determine if a strategy should be accepted.

        Args:
            candidate: Metrics of the new strategy
            baseline: Metrics of the seed/baseline strategy (MAS reference)
            population_metrics: Optional list of all current population metrics
                               (required for Pareto-based selection)
        """
        ...

    @abstractmethod
    def compute_fitness(self, metrics: StrategyMetrics) -> float:
        """Compute scalar fitness for ranking."""
        ...


@dataclass
class WeightedSumPolicy:
    """
    Accept based on weighted sum of metrics using baseline-relative normalization.

    IMPROVED: Uses baseline-relative scoring instead of fixed ranges.

    fitness = w_return * sigmoid((ret - baseline_ret) / scale)
            + w_sharpe * sigmoid((sharpe - baseline_sharpe) / scale)
            + w_drawdown * sigmoid((baseline_dd - dd) / scale)
    """

    w_return: float = 0.5
    w_sharpe: float = 0.3
    w_drawdown: float = 0.2
    scale: float = 10.0  # Scaling factor for sigmoid normalization

    # Robustness weight (optional)
    w_stability: float = 0.0  # Penalize high cross-fold variance
    w_consistency: float = 0.0  # Reward consistent positive returns

    _baseline: Optional[StrategyMetrics] = field(default=None, repr=False)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function mapping (-inf, inf) to (0, 1)."""
        return 1 / (1 + math.exp(-max(-500, min(500, x))))

    def compute_fitness(
        self,
        metrics: StrategyMetrics,
        baseline: StrategyMetrics = None,
    ) -> float:
        """
        Compute weighted fitness score using baseline-relative normalization.
        """
        baseline = baseline or self._baseline
        if baseline is None:
            # Fallback to absolute normalization if no baseline
            return self._compute_fitness_absolute(metrics)

        # Baseline-relative scoring
        return_score = self._sigmoid(
            (metrics.ann_return - baseline.ann_return) / self.scale
        )
        sharpe_score = self._sigmoid(
            (metrics.sharpe - baseline.sharpe) / (self.scale / 10)
        )
        # For drawdown: less negative is better, so (baseline - candidate) if candidate less negative
        dd_score = self._sigmoid(
            (baseline.max_drawdown - metrics.max_drawdown) / self.scale
        )

        fitness = (
            self.w_return * return_score
            + self.w_sharpe * sharpe_score
            + self.w_drawdown * dd_score
        )

        # Optional robustness penalties
        if self.w_stability > 0 and metrics.stability > 0:
            # Lower stability (std dev) is better
            stability_penalty = self._sigmoid(-metrics.stability / self.scale)
            fitness += self.w_stability * stability_penalty

        if self.w_consistency > 0:
            # Higher consistency (% positive folds) is better
            consistency_score = metrics.consistency / 100  # Already 0-100
            fitness += self.w_consistency * consistency_score

        return fitness

    def _compute_fitness_absolute(self, metrics: StrategyMetrics) -> float:
        """Fallback absolute scoring (less preferred)."""
        return_score = max(0, min(1, (metrics.ann_return + 50) / 100))
        sharpe_score = max(0, min(1, (metrics.sharpe + 1) / 3))
        dd_score = max(0, 1 - abs(metrics.max_drawdown) / 50)
        return (
            self.w_return * return_score
            + self.w_sharpe * sharpe_score
            + self.w_drawdown * dd_score
        )

    def should_accept(
        self,
        candidate: StrategyMetrics,
        baseline: StrategyMetrics,
        population_metrics: Optional[List[StrategyMetrics]] = None,
        **kwargs,
    ) -> bool:
        """Accept if fitness exceeds baseline fitness."""
        self._baseline = baseline
        return self.compute_fitness(candidate, baseline) >= self.compute_fitness(
            baseline, baseline
        )


@dataclass
class GatedMASPolicy:
    """
    Multi-gate acceptance: must pass all thresholds.

    IMPROVED: Includes robustness gates (stability, consistency, worst fold).
    """

    # Primary gates
    min_return: float = 0.0  # Minimum annualized return (%)
    min_sharpe: float = 0.0  # Minimum Sharpe ratio
    max_drawdown: float = -50.0  # Maximum drawdown (%, NEGATIVE, so > this)
    min_trades: int = 1  # Minimum trade count
    min_win_rate: float = 0.0  # Minimum win rate (%)

    # Robustness gates (NEW)
    min_consistency: float = 0.0  # Minimum % of folds with positive return
    min_worst_fold: float = -100.0  # Worst fold return must be > this
    max_stability: float = 100.0  # Max cross-fold std dev (lower = more stable)

    def compute_fitness(self, metrics: StrategyMetrics) -> float:
        """Fitness is simply annualized return for ranking."""
        return metrics.ann_return

    def should_accept(
        self,
        candidate: StrategyMetrics,
        baseline: StrategyMetrics,
        population_metrics: Optional[List[StrategyMetrics]] = None,
        **kwargs,
    ) -> bool:
        """Accept only if all gates pass AND beats baseline return."""
        # Check primary gates
        if candidate.ann_return < self.min_return:
            return False
        if candidate.sharpe < self.min_sharpe:
            return False
        if candidate.max_drawdown < self.max_drawdown:  # More negative = worse
            return False
        if candidate.trade_count < self.min_trades:
            return False
        if candidate.win_rate < self.min_win_rate:
            return False

        # Check robustness gates
        if candidate.consistency < self.min_consistency:
            return False
        if candidate.worst_fold_return < self.min_worst_fold:
            return False
        if candidate.stability > self.max_stability:
            return False

        # Also must beat baseline (original MAS behavior)
        return candidate.ann_return >= baseline.ann_return


@dataclass
class ParetoPolicy:
    """
    Pareto-optimal selection: accept if non-dominated.

    FIXED: Uses population_metrics for proper dominance checking.
    """

    objectives: tuple = ("ann_return", "sharpe", "max_drawdown")
    # For max_drawdown, less negative is better (maximize)
    # Note: max_drawdown is stored as negative (e.g., -15.0 for 15% drawdown)
    # Less negative = better, so -10% > -20% means -10% is better

    # Debug logging flag
    debug: bool = False

    def compute_fitness(self, metrics: StrategyMetrics) -> float:
        """Fitness is sum of objective values (for ranking)."""
        return sum(
            getattr(metrics, obj) * (1 if obj != "max_drawdown" else -1)
            for obj in self.objectives
        )

    def dominates(self, a: StrategyMetrics, b: StrategyMetrics) -> bool:
        """
        Check if strategy A dominates strategy B.

        A dominates B if:
        - A is at least as good as B on ALL objectives
        - A is strictly better than B on AT LEAST ONE objective

        For all objectives (including max_drawdown), higher values are better:
        - ann_return: higher % is better
        - sharpe: higher ratio is better
        - max_drawdown: less negative is better (e.g., -10% > -20%)
        """
        at_least_as_good_all = True  # A >= B on all objectives
        strictly_better_one = False  # A > B on at least one objective

        if self.debug:
            print(f"    [Pareto] Checking dominance: A vs B")

        for obj in self.objectives:
            val_a = getattr(a, obj)
            val_b = getattr(b, obj)

            # All objectives use "higher is better" semantics
            # max_drawdown: -10% > -20%, so -10% is better (no negation needed!)

            if self.debug:
                better_symbol = ">" if val_a > val_b else ("<" if val_a < val_b else "=")
                print(f"      {obj}: A={val_a:.2f} {better_symbol} B={val_b:.2f}")

            if val_a < val_b:
                at_least_as_good_all = False  # A is worse than B on this objective
            if val_a > val_b:
                strictly_better_one = True  # A is strictly better on this objective

        result = at_least_as_good_all and strictly_better_one

        if self.debug:
            print(
                f"      Result: at_least_as_good_all={at_least_as_good_all}, "
                f"strictly_better_one={strictly_better_one} => dominates={result}"
            )

        return result

    def should_accept(
        self,
        candidate: StrategyMetrics,
        baseline: StrategyMetrics,
        population_metrics: Optional[List[StrategyMetrics]] = None,
        **kwargs,
    ) -> bool:
        """
        Accept if not dominated by any existing strategy.

        FIXED: Uses population_metrics when provided.
        """
        # Use population if provided, otherwise just compare to baseline
        comparison_set = population_metrics if population_metrics else [baseline]

        if self.debug:
            print(f"  [Pareto] should_accept called:")
            print(
                f"    Candidate: return={candidate.ann_return:.2f}%, "
                f"sharpe={candidate.sharpe:.2f}, dd={candidate.max_drawdown:.2f}%"
            )
            print(f"    Population size: {len(comparison_set)}")

        for i, existing in enumerate(comparison_set):
            if self.debug:
                print(
                    f"    Comparing against population[{i}]: "
                    f"return={existing.ann_return:.2f}%, "
                    f"sharpe={existing.sharpe:.2f}, dd={existing.max_drawdown:.2f}%"
                )

            if self.dominates(existing, candidate):
                if self.debug:
                    print(f"    => REJECTED: dominated by population[{i}]")
                return False  # Dominated by existing strategy

        if self.debug:
            print(f"    => ACCEPTED: not dominated by any existing strategy")

        return True


# =============================================================================
# Factory Functions
# =============================================================================


def create_selection_policy(
    policy_type: str,
    **kwargs,
) -> SelectionPolicy:
    """
    Create a selection policy by name.

    Args:
        policy_type: One of 'weighted', 'gated', 'pareto'
        **kwargs: Policy-specific parameters
            - debug: bool - Enable debug logging for policy decisions

    Returns:
        SelectionPolicy instance
    """
    if policy_type == "weighted":
        return WeightedSumPolicy(
            w_return=kwargs.get("w_return", 0.5),
            w_sharpe=kwargs.get("w_sharpe", 0.3),
            w_drawdown=kwargs.get("w_drawdown", 0.2),
            scale=kwargs.get("scale", 10.0),
            w_stability=kwargs.get("w_stability", 0.0),
            w_consistency=kwargs.get("w_consistency", 0.0),
        )
    elif policy_type == "pareto":
        return ParetoPolicy(
            objectives=tuple(
                kwargs.get("objectives", ["ann_return", "sharpe", "max_drawdown"])
            ),
            debug=kwargs.get("debug", False),
        )
    else:  # default: gated
        return GatedMASPolicy(
            min_return=kwargs.get("min_return", 0.0),
            min_sharpe=kwargs.get("min_sharpe", 0.0),
            max_drawdown=kwargs.get("max_drawdown", -50.0),
            min_trades=kwargs.get("min_trades", 1),
            min_win_rate=kwargs.get("min_win_rate", 0.0),
            min_consistency=kwargs.get("min_consistency", 0.0),
            min_worst_fold=kwargs.get("min_worst_fold", -100.0),
            max_stability=kwargs.get("max_stability", 100.0),
        )


def create_cascade(
    mode: str = "full",
    metrics_calculator: MetricsCalculator = None,
    promotion_gate: PromotionGate = None,
    smoke_months: int = 3,
    initial_capital: float = 10000,
    commission: float = 0.002,
    verbose: bool = True,
) -> EvaluationCascade:
    """
    Create an evaluation cascade by mode.

    Args:
        mode: One of 'quick', 'standard', 'full'
        metrics_calculator: Optional metrics calculator
        promotion_gate: Optional promotion gate
        smoke_months: Months of data for smoke test
        initial_capital: Initial capital for backtests
        commission: Commission rate
        verbose: Print progress messages

    Returns:
        EvaluationCascade instance
    """
    metrics_calc = metrics_calculator or MetricsCalculator()
    gate = promotion_gate or PromotionGate()

    if mode == "quick":
        # Just syntax and smoke test
        stages = [
            SyntaxCheckStage(),
            SmokeTestStage(
                slice_months=smoke_months,
                initial_capital=initial_capital,
                commission=commission,
            ),
        ]
    elif mode == "standard":
        # Syntax, smoke, and single fold
        stages = [
            SyntaxCheckStage(),
            SmokeTestStage(
                slice_months=smoke_months,
                initial_capital=initial_capital,
                commission=commission,
            ),
            SingleFoldStage(
                initial_capital=initial_capital,
                commission=commission,
                metrics_calculator=metrics_calc,
                promotion_gate=gate,
            ),
        ]
    else:  # full
        stages = [
            SyntaxCheckStage(),
            SmokeTestStage(
                slice_months=smoke_months,
                initial_capital=initial_capital,
                commission=commission,
            ),
            SingleFoldStage(
                initial_capital=initial_capital,
                commission=commission,
                metrics_calculator=metrics_calc,
                promotion_gate=gate,
            ),
            FullWalkForwardStage(
                initial_capital=initial_capital,
                commission=commission,
                metrics_calculator=metrics_calc,
            ),
        ]

    return EvaluationCascade(stages=stages, verbose=verbose)
