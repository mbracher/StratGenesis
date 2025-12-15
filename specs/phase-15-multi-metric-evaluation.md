# Phase 15: Multi-Metric Scoring & Evaluation Cascade

## Objective

Replace single-metric fitness (annualized return) with multi-objective evaluation and implement a fast rejection cascade to improve iteration speed and reduce overfitting.

From the AlphaEvolve paper:

> AlphaEvolve explicitly supports multiple scores (a dict of metrics) and an evaluation cascade (cheap tests first, expensive tests later).

---

## Dependencies

- Phase 5 (Backtesting Utilities) - existing `run_backtest()` method
- Phase 6 (Evolutionary Engine) - existing MAS threshold logic
- Phase 13 (Program Database) - for storing full metrics

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                     Evaluation Cascade                              │
│                                                                     │
│  Stage 1         Stage 2         Stage 3         Stage 4           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│  │ Syntax  │───►│ Smoke   │───►│ 1-Fold  │───►│ Full WF │        │
│  │ Check   │    │ Test    │    │ Valid   │    │ 5 Folds │        │
│  │ <1ms    │    │ ~1s     │    │ ~10s    │    │ ~60s    │        │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘        │
│       │FAIL          │FAIL          │FAIL          │              │
│       ▼              ▼              ▼              ▼              │
│    REJECT         REJECT         REJECT      ┌─────────┐         │
│                                              │ Accept/ │         │
│                                              │ Reject  │         │
│                                              └─────────┘         │
└────────────────────────────────────────────────────────────────────┘
```

---

## Metrics Model

### StrategyMetrics Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import math


@dataclass
class StrategyMetrics:
    """
    Comprehensive metrics for strategy evaluation.

    Primary metrics are used for fitness. Secondary metrics
    provide additional context for analysis.
    """

    # === Primary Metrics (used for fitness) ===
    ann_return: float = 0.0          # Annualized return (%)
    sharpe: float = 0.0              # Sharpe ratio (risk-adjusted return)
    max_drawdown: float = 0.0        # Maximum drawdown (%, negative)
    sortino: float = 0.0             # Sortino ratio (downside risk)
    calmar: float = 0.0              # Calmar ratio (return / max drawdown)

    # === Secondary Metrics (context/analysis) ===
    trade_count: int = 0             # Number of trades
    win_rate: float = 0.0            # Winning trades / total trades
    profit_factor: float = 0.0       # Gross profit / gross loss
    expectancy: float = 0.0          # Expected profit per trade (%)
    exposure_time: float = 0.0       # Time in market (%)

    # === Robustness Metrics ===
    stability: float = 0.0           # Cross-fold std dev of returns
    consistency: float = 0.0         # % of folds with positive return
    worst_fold_return: float = 0.0   # Worst single fold performance

    # === Raw data for custom calculations ===
    equity_curve: List[float] = field(default_factory=list)
    trade_returns: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (excludes lists)."""
        return {
            'ann_return': self.ann_return,
            'sharpe': self.sharpe,
            'max_drawdown': self.max_drawdown,
            'sortino': self.sortino,
            'calmar': self.calmar,
            'trade_count': self.trade_count,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'exposure_time': self.exposure_time,
            'stability': self.stability,
            'consistency': self.consistency,
            'worst_fold_return': self.worst_fold_return,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'StrategyMetrics':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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
```

---

## MetricsCalculator

```python
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


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
            StrategyMetrics with all computed values
        """
        # Extract raw data
        equity = backtest_result.get('_equity_curve', pd.DataFrame())
        trades = backtest_result.get('_trades', pd.DataFrame())

        # Basic metrics (already computed by backtesting.py)
        ann_return = backtest_result.get('Return (Ann.) [%]', 0.0)
        sharpe = backtest_result.get('Sharpe Ratio', 0.0)
        max_drawdown = backtest_result.get('Max. Drawdown [%]', 0.0)
        exposure_time = backtest_result.get('Exposure Time [%]', 0.0)

        # Trade-based metrics
        trade_count = int(backtest_result.get('# Trades', 0))
        win_rate = backtest_result.get('Win Rate [%]', 0.0)
        expectancy = backtest_result.get('Expectancy [%]', 0.0)

        # Compute additional metrics
        sortino = self._compute_sortino(equity, ann_return)
        calmar = self._compute_calmar(ann_return, max_drawdown)
        profit_factor = self._compute_profit_factor(trades)

        # Extract equity curve and trade returns for later analysis
        equity_curve = []
        trade_returns = []

        if not equity.empty and 'Equity' in equity.columns:
            equity_curve = equity['Equity'].tolist()

        if not trades.empty and 'ReturnPct' in trades.columns:
            trade_returns = trades['ReturnPct'].tolist()

        return StrategyMetrics(
            ann_return=ann_return,
            sharpe=sharpe if not np.isnan(sharpe) else 0.0,
            max_drawdown=max_drawdown,
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
        ann_return: float
    ) -> float:
        """Compute Sortino ratio (downside deviation only)."""
        if equity.empty or 'Equity' not in equity.columns:
            return 0.0

        returns = equity['Equity'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        # Only consider negative returns for downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if ann_return > 0 else 0.0

        downside_std = negative_returns.std() * np.sqrt(252)  # Annualize

        if downside_std == 0:
            return 0.0

        return (ann_return / 100 - self.risk_free_rate) / downside_std

    def _compute_calmar(self, ann_return: float, max_drawdown: float) -> float:
        """Compute Calmar ratio (return / max drawdown)."""
        if max_drawdown == 0:
            return 0.0
        return ann_return / abs(max_drawdown)

    def _compute_profit_factor(self, trades: pd.DataFrame) -> float:
        """Compute profit factor (gross profit / gross loss)."""
        if trades.empty or 'PnL' not in trades.columns:
            return 0.0

        profits = trades[trades['PnL'] > 0]['PnL'].sum()
        losses = abs(trades[trades['PnL'] < 0]['PnL'].sum())

        if losses == 0:
            return float('inf') if profits > 0 else 0.0

        return profits / losses

    def compute_cross_fold_metrics(
        self,
        fold_metrics: List[StrategyMetrics]
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

        stability = np.std(returns) if len(returns) > 1 else 0.0
        consistency = sum(1 for r in returns if r > 0) / len(returns) * 100
        worst_fold = min(returns)

        # Compute mean of all metrics
        mean_metrics = StrategyMetrics(
            ann_return=np.mean(returns),
            sharpe=np.mean([m.sharpe for m in fold_metrics]),
            max_drawdown=np.mean([m.max_drawdown for m in fold_metrics]),
            sortino=np.mean([m.sortino for m in fold_metrics]),
            calmar=np.mean([m.calmar for m in fold_metrics]),
            trade_count=int(np.mean([m.trade_count for m in fold_metrics])),
            win_rate=np.mean([m.win_rate for m in fold_metrics]),
            profit_factor=np.mean([m.profit_factor for m in fold_metrics]),
            expectancy=np.mean([m.expectancy for m in fold_metrics]),
            exposure_time=np.mean([m.exposure_time for m in fold_metrics]),
            stability=stability,
            consistency=consistency,
            worst_fold_return=worst_fold,
        )

        return mean_metrics
```

---

## Evaluation Cascade

### EvaluationStage Protocol

```python
from typing import Protocol, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


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


class EvaluationStage(Protocol):
    """Protocol for evaluation stages."""

    name: str
    order: int  # Lower = earlier in cascade

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        **kwargs
    ) -> StageOutput:
        """Run this evaluation stage."""
        ...
```

### Stage Implementations

```python
import ast
import time
import traceback
from typing import Optional
import pandas as pd


class SyntaxCheckStage:
    """Stage 1: Parse and compile code (instant)."""

    name = "syntax_check"
    order = 1

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame = None,
        **kwargs
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
                    duration_ms=(time.time() - start) * 1000
                )

            # Check for required methods
            methods = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    methods.add(node.name)

            required = {'init', 'next'}
            missing = required - methods
            if missing:
                return StageOutput(
                    result=StageResult.FAIL,
                    error=f"Missing methods: {missing}",
                    duration_ms=(time.time() - start) * 1000
                )

            # Try to compile
            compile(strategy_code, '<string>', 'exec')

            return StageOutput(
                result=StageResult.PASS,
                duration_ms=(time.time() - start) * 1000
            )

        except SyntaxError as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                duration_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=str(e),
                duration_ms=(time.time() - start) * 1000
            )


class SmokeTestStage:
    """Stage 2: Run on small data slice (~1s)."""

    name = "smoke_test"
    order = 2

    def __init__(
        self,
        slice_months: int = 3,
        initial_capital: float = 10000,
        commission: float = 0.002
    ):
        self.slice_months = slice_months
        self.initial_capital = initial_capital
        self.commission = commission

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        **kwargs
    ) -> StageOutput:
        start = time.time()

        try:
            # Take a small slice of data
            slice_days = self.slice_months * 21  # ~21 trading days per month
            if len(data) < slice_days:
                data_slice = data
            else:
                data_slice = data.iloc[:slice_days]

            # Dynamic import and execution
            namespace = {}
            exec(strategy_code, globals(), namespace)

            # Find the strategy class
            strategy_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and hasattr(obj, 'init') and hasattr(obj, 'next'):
                    strategy_class = obj
                    break

            if not strategy_class:
                return StageOutput(
                    result=StageResult.FAIL,
                    error="Could not find valid strategy class",
                    duration_ms=(time.time() - start) * 1000
                )

            # Run quick backtest
            from backtesting import Backtest

            bt = Backtest(
                data_slice,
                strategy_class,
                cash=self.initial_capital,
                commission=self.commission,
            )
            result = bt.run()

            # Just check it ran without crashing
            return StageOutput(
                result=StageResult.PASS,
                duration_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=f"Smoke test failed: {str(e)}",
                duration_ms=(time.time() - start) * 1000
            )


class SingleFoldStage:
    """Stage 3: Run on single validation fold (~10s)."""

    name = "single_fold"
    order = 3

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.002,
        metrics_calculator: MetricsCalculator = None
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.metrics_calc = metrics_calculator or MetricsCalculator()

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        **kwargs
    ) -> StageOutput:
        start = time.time()

        try:
            # Execute code
            namespace = {}
            exec(strategy_code, globals(), namespace)

            strategy_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and hasattr(obj, 'init') and hasattr(obj, 'next'):
                    strategy_class = obj
                    break

            if not strategy_class:
                return StageOutput(
                    result=StageResult.FAIL,
                    error="Could not find valid strategy class",
                    duration_ms=(time.time() - start) * 1000
                )

            # Run backtest
            from backtesting import Backtest

            bt = Backtest(
                data,
                strategy_class,
                cash=self.initial_capital,
                commission=self.commission,
            )
            result = bt.run()

            # Compute metrics
            metrics = self.metrics_calc.compute_all(result)

            return StageOutput(
                result=StageResult.PASS,
                metrics=metrics,
                duration_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=f"Single fold failed: {str(e)}\n{traceback.format_exc()}",
                duration_ms=(time.time() - start) * 1000
            )


class FullWalkForwardStage:
    """Stage 4: Run full walk-forward optimization (~60s)."""

    name = "full_walkforward"
    order = 4

    def __init__(
        self,
        n_folds: int = 5,
        initial_capital: float = 10000,
        commission: float = 0.002,
        metrics_calculator: MetricsCalculator = None
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
        **kwargs
    ) -> StageOutput:
        start = time.time()

        try:
            # Execute code
            namespace = {}
            exec(strategy_code, globals(), namespace)

            strategy_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and hasattr(obj, 'init') and hasattr(obj, 'next'):
                    strategy_class = obj
                    break

            if not strategy_class:
                return StageOutput(
                    result=StageResult.FAIL,
                    error="Could not find valid strategy class",
                    duration_ms=(time.time() - start) * 1000
                )

            # Run across all folds
            fold_metrics = []

            for fold_num, (train, val, test) in enumerate(folds or [], 1):
                from backtesting import Backtest

                # Evaluate on validation data
                bt = Backtest(
                    val,
                    strategy_class,
                    cash=self.initial_capital,
                    commission=self.commission,
                )
                result = bt.run()
                metrics = self.metrics_calc.compute_all(result)
                fold_metrics.append(metrics)

            # Compute aggregate metrics
            aggregate = self.metrics_calc.compute_cross_fold_metrics(fold_metrics)

            return StageOutput(
                result=StageResult.PASS,
                metrics=aggregate,
                duration_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=f"Walk-forward failed: {str(e)}",
                duration_ms=(time.time() - start) * 1000
            )
```

### EvaluationCascade

```python
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CascadeResult:
    """Result of running the evaluation cascade."""
    passed: bool
    final_stage: str
    metrics: Optional[StrategyMetrics]
    stage_results: Dict[str, StageOutput]
    total_duration_ms: float


class EvaluationCascade:
    """
    Runs evaluation stages in order, failing fast on early rejections.

    This implements the AlphaEvolve evaluation cascade pattern:
    cheap tests first, expensive tests only if earlier stages pass.
    """

    def __init__(self, stages: List[EvaluationStage] = None):
        """
        Args:
            stages: List of evaluation stages. Defaults to all stages.
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

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        stop_at_stage: str = None,
        **kwargs
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

        for stage in self.stages:
            print(f"  Running {stage.name}...", end=" ", flush=True)

            output = stage.evaluate(strategy_code, data, **kwargs)
            stage_results[stage.name] = output
            total_duration += output.duration_ms

            if output.result == StageResult.FAIL:
                print(f"FAIL ({output.duration_ms:.0f}ms)")
                if output.error:
                    print(f"    Error: {output.error}")
                passed = False
                break
            else:
                print(f"PASS ({output.duration_ms:.0f}ms)")

            # Keep the most detailed metrics
            if output.metrics:
                final_metrics = output.metrics

            # Stop early if requested
            if stop_at_stage and stage.name == stop_at_stage:
                break

        return CascadeResult(
            passed=passed,
            final_stage=stage.name,
            metrics=final_metrics,
            stage_results=stage_results,
            total_duration_ms=total_duration
        )

    def quick_evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame
    ) -> CascadeResult:
        """Run only syntax and smoke test (for fast iteration)."""
        return self.evaluate(strategy_code, data, stop_at_stage="smoke_test")
```

---

## Selection Policies

### SelectionPolicy Protocol

```python
from typing import Protocol
from abc import abstractmethod


class SelectionPolicy(Protocol):
    """Protocol for strategy acceptance policies."""

    @abstractmethod
    def should_accept(self, metrics: StrategyMetrics, baseline: StrategyMetrics) -> bool:
        """Determine if a strategy should be accepted."""
        ...

    @abstractmethod
    def compute_fitness(self, metrics: StrategyMetrics) -> float:
        """Compute scalar fitness for ranking."""
        ...
```

### Policy Implementations

```python
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class WeightedSumPolicy:
    """
    Accept based on weighted sum of metrics.

    fitness = w_return * return + w_sharpe * sharpe + w_drawdown * (1 - |drawdown|/100)
    """

    w_return: float = 0.5
    w_sharpe: float = 0.3
    w_drawdown: float = 0.2
    min_fitness: float = 0.0  # Minimum fitness to accept

    def compute_fitness(self, metrics: StrategyMetrics) -> float:
        """Compute weighted fitness score."""
        # Normalize metrics to roughly 0-1 scale
        return_score = max(0, min(1, (metrics.ann_return + 50) / 100))  # -50% to 50% -> 0 to 1
        sharpe_score = max(0, min(1, (metrics.sharpe + 1) / 3))  # -1 to 2 -> 0 to 1
        dd_score = max(0, 1 - abs(metrics.max_drawdown) / 50)  # 0% to 50% DD -> 1 to 0

        return (
            self.w_return * return_score +
            self.w_sharpe * sharpe_score +
            self.w_drawdown * dd_score
        )

    def should_accept(
        self,
        metrics: StrategyMetrics,
        baseline: StrategyMetrics
    ) -> bool:
        """Accept if fitness exceeds baseline fitness."""
        return self.compute_fitness(metrics) >= self.compute_fitness(baseline)


@dataclass
class GatedMASPolicy:
    """
    Multi-gate acceptance: must pass all thresholds.

    More restrictive than single MAS - requires multiple criteria.
    """

    min_return: float = 0.0         # Minimum annualized return (%)
    min_sharpe: float = 0.0         # Minimum Sharpe ratio
    max_drawdown: float = -50.0     # Maximum drawdown (%, should be negative)
    min_trades: int = 1             # Minimum trade count
    min_win_rate: float = 0.0       # Minimum win rate (%)

    def compute_fitness(self, metrics: StrategyMetrics) -> float:
        """Fitness is simply annualized return for ranking."""
        return metrics.ann_return

    def should_accept(
        self,
        metrics: StrategyMetrics,
        baseline: StrategyMetrics
    ) -> bool:
        """Accept only if all gates pass AND beats baseline return."""
        # Check all gates
        if metrics.ann_return < self.min_return:
            return False
        if metrics.sharpe < self.min_sharpe:
            return False
        if metrics.max_drawdown < self.max_drawdown:  # More negative = worse
            return False
        if metrics.trade_count < self.min_trades:
            return False
        if metrics.win_rate < self.min_win_rate:
            return False

        # Also must beat baseline (original MAS behavior)
        return metrics.ann_return >= baseline.ann_return


@dataclass
class ParetoPolicy:
    """
    Pareto-optimal selection: accept if non-dominated.

    A strategy is non-dominated if no other strategy is better
    in ALL objectives simultaneously.
    """

    objectives: tuple = ('ann_return', 'sharpe', 'max_drawdown')
    # For max_drawdown, less negative is better (maximize)

    def compute_fitness(self, metrics: StrategyMetrics) -> float:
        """Fitness is sum of objective values (for ranking)."""
        return sum(
            getattr(metrics, obj) * (1 if obj != 'max_drawdown' else -1)
            for obj in self.objectives
        )

    def dominates(self, a: StrategyMetrics, b: StrategyMetrics) -> bool:
        """Check if strategy A dominates strategy B."""
        dominated = True
        strictly_better = False

        for obj in self.objectives:
            val_a = getattr(a, obj)
            val_b = getattr(b, obj)

            # For max_drawdown, less negative is better
            if obj == 'max_drawdown':
                val_a, val_b = -val_a, -val_b

            if val_a < val_b:
                dominated = False
            if val_a > val_b:
                strictly_better = True

        return dominated and strictly_better

    def should_accept(
        self,
        metrics: StrategyMetrics,
        baseline: StrategyMetrics,
        population: List[StrategyMetrics] = None
    ) -> bool:
        """Accept if not dominated by any existing strategy."""
        population = population or [baseline]

        for existing in population:
            if self.dominates(existing, metrics):
                return False  # Dominated by existing strategy

        return True
```

---

## Evolver Integration

### Updated evolve_strategy()

```python
def evolve_strategy(
    self,
    strategy_class,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    max_iters: int = 15,
    fold: int = 1,
    use_inspirations: bool = True,
    prefer_diffs: bool = True,
    selection_policy: SelectionPolicy = None,  # NEW
    cascade: EvaluationCascade = None,  # NEW
):
    """
    Evolve strategy with multi-metric evaluation and cascade.

    Args:
        strategy_class: Seed strategy to evolve
        train_data: Training data (for context)
        val_data: Validation data (for fitness)
        max_iters: Maximum generations
        fold: Current fold number
        use_inspirations: Use program DB inspirations
        prefer_diffs: Use diff-based mutations
        selection_policy: Policy for acceptance (default: GatedMASPolicy)
        cascade: Evaluation cascade (default: full cascade)
    """
    # Defaults
    if selection_policy is None:
        selection_policy = GatedMASPolicy()

    if cascade is None:
        cascade = EvaluationCascade([
            SyntaxCheckStage(),
            SmokeTestStage(),
            SingleFoldStage(metrics_calculator=MetricsCalculator()),
        ])

    # 1. Compute baseline metrics
    parent_code = inspect.getsource(strategy_class)
    baseline_result = cascade.evaluate(parent_code, val_data)

    if not baseline_result.passed:
        raise ValueError(f"Seed strategy failed evaluation: {baseline_result}")

    baseline_metrics = baseline_result.metrics
    print(f"Baseline: Return={baseline_metrics.ann_return:.2f}%, "
          f"Sharpe={baseline_metrics.sharpe:.2f}, "
          f"MaxDD={baseline_metrics.max_drawdown:.2f}%")

    # 2. Initialize population
    population = [(strategy_class, baseline_metrics, parent_code)]
    best_fitness = selection_policy.compute_fitness(baseline_metrics)
    best_strategy = (strategy_class, baseline_metrics, parent_code)

    # 3. Evolution loop
    for gen in range(1, max_iters + 1):
        print(f"\n=== Generation {gen} ===")

        # Select parent
        parent_class, parent_metrics, parent_code = random.choice(population)

        # Generate improvement
        improvement = self.llm.generate_improvement(
            parent_code,
            f"Return={parent_metrics.ann_return:.2f}%, "
            f"Sharpe={parent_metrics.sharpe:.2f}, "
            f"MaxDD={parent_metrics.max_drawdown:.2f}%"
        )
        print(f"Improvement: {improvement[:100]}...")

        # Generate new code
        if prefer_diffs:
            new_code, used_diff = self.llm.generate_strategy_code_with_fallback(
                parent_code, improvement
            )
        else:
            new_code = self.llm.generate_strategy_code(parent_code, improvement)

        # 4. Run evaluation cascade
        print("Evaluating...")
        result = cascade.evaluate(new_code, val_data)

        if not result.passed:
            print(f"  Rejected at {result.final_stage}")
            continue

        new_metrics = result.metrics
        print(f"  Return={new_metrics.ann_return:.2f}%, "
              f"Sharpe={new_metrics.sharpe:.2f}, "
              f"MaxDD={new_metrics.max_drawdown:.2f}%")

        # 5. Check acceptance
        if selection_policy.should_accept(new_metrics, baseline_metrics):
            # Create strategy class from code
            namespace = {}
            exec(new_code, globals(), namespace)
            new_class_name = f"{parent_class.__name__}_Gen{gen}"
            new_class = namespace.get(new_class_name) or list(namespace.values())[-1]

            population.append((new_class, new_metrics, new_code))
            print(f"  ACCEPTED (population size: {len(population)})")

            # Update best
            fitness = selection_policy.compute_fitness(new_metrics)
            if fitness > best_fitness:
                best_fitness = fitness
                best_strategy = (new_class, new_metrics, new_code)
                print(f"  NEW BEST! Fitness={fitness:.4f}")
        else:
            print(f"  Rejected by selection policy")

    best_class, best_metrics, best_code = best_strategy
    print(f"\nEvolution complete. Best fitness={best_fitness:.4f}")
    return best_class, best_metrics, best_code
```

---

## CLI Integration

Add to `src/profit/main.py`:

```python
# Selection policy arguments
parser.add_argument(
    '--selection-policy',
    choices=['weighted', 'gated', 'pareto'],
    default='gated',
    help='Selection policy for strategy acceptance'
)
parser.add_argument(
    '--min-return',
    type=float,
    default=0.0,
    help='Minimum annualized return threshold (for gated policy)'
)
parser.add_argument(
    '--min-sharpe',
    type=float,
    default=0.0,
    help='Minimum Sharpe ratio threshold (for gated policy)'
)
parser.add_argument(
    '--max-drawdown',
    type=float,
    default=-50.0,
    help='Maximum drawdown threshold (for gated policy, should be negative)'
)

# Cascade arguments
parser.add_argument(
    '--skip-cascade',
    action='store_true',
    help='Skip evaluation cascade (use direct backtest only)'
)
parser.add_argument(
    '--quick-eval',
    action='store_true',
    help='Use quick evaluation (syntax + smoke test only)'
)


# In main():
# Build selection policy
if args.selection_policy == 'weighted':
    policy = WeightedSumPolicy()
elif args.selection_policy == 'pareto':
    policy = ParetoPolicy()
else:
    policy = GatedMASPolicy(
        min_return=args.min_return,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown
    )

# Build cascade
if args.skip_cascade:
    cascade = None
elif args.quick_eval:
    cascade = EvaluationCascade([
        SyntaxCheckStage(),
        SmokeTestStage(),
    ])
else:
    cascade = EvaluationCascade()

# Pass to evolver
evolver.walk_forward_optimize(
    data,
    strategy_class,
    selection_policy=policy,
    cascade=cascade,
    ...
)
```

---

## File Structure

```
src/profit/
├── __init__.py
├── strategies.py
├── llm_interface.py
├── evolver.py            # Modified: use cascade and selection policies
├── main.py               # Modified: CLI arguments
├── program_db.py
├── diff_utils.py
└── evaluation.py         # NEW: metrics, cascade, policies
```

---

## Deliverables

- [ ] `StrategyMetrics` dataclass with all metrics
- [ ] `MetricsCalculator` class
  - [ ] `compute_all()` method
  - [ ] `compute_cross_fold_metrics()` method
  - [ ] Sortino, Calmar, profit factor calculations
- [ ] Evaluation stages:
  - [ ] `SyntaxCheckStage`
  - [ ] `SmokeTestStage`
  - [ ] `SingleFoldStage`
  - [ ] `FullWalkForwardStage`
- [ ] `EvaluationCascade` class
  - [ ] `evaluate()` method
  - [ ] `quick_evaluate()` method
- [ ] Selection policies:
  - [ ] `WeightedSumPolicy`
  - [ ] `GatedMASPolicy`
  - [ ] `ParetoPolicy`
- [ ] Evolver integration
- [ ] CLI arguments for policies and cascade
- [ ] Tests for:
  - [ ] Metrics calculation
  - [ ] Each evaluation stage
  - [ ] Cascade behavior
  - [ ] Selection policies
