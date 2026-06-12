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
┌────────────────────────────────────────────────────────────────────────────┐
│                     Evaluation Cascade                                      │
│                                                                             │
│  Stage 1         Stage 2         Stage 3              Stage 4              │
│  ┌─────────┐    ┌─────────┐    ┌─────────────┐      ┌─────────┐           │
│  │ Syntax  │───►│ Smoke   │───►│ 1-Fold      │─────►│ Full WF │           │
│  │ Check   │    │ Test    │    │ + Gate      │      │ 5 Folds │           │
│  │ <1ms    │    │ ~1s     │    │ ~10s        │      │ ~60s    │           │
│  └────┬────┘    └────┬────┘    └──────┬──────┘      └────┬────┘           │
│       │FAIL          │FAIL            │FAIL              │                 │
│       ▼              ▼                ▼                  ▼                 │
│    REJECT         REJECT           REJECT          ┌─────────┐            │
│                                 (metrics gate)     │ Accept/ │            │
│                                                    │ Reject  │            │
│                                                    └─────────┘            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Helpers (Centralized Logic)

### Strategy Loading Helper

```python
from backtesting import Strategy
from typing import Type, Optional
import hashlib


def load_strategy_class(
    strategy_code: str,
    exec_globals: dict = None
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
        import pandas as pd
        import numpy as np
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
        if (isinstance(obj, type)
            and issubclass(obj, Strategy)
            and obj is not Strategy):
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
) -> pd.Series:
    """
    Run backtest and return full result series.

    Args:
        strategy_class: Strategy subclass to backtest
        data: OHLCV DataFrame
        cash: Initial capital
        commission: Per-trade commission rate

    Returns:
        backtesting.py result Series with all metrics
    """
    from backtesting import Backtest

    bt = Backtest(
        data,
        strategy_class,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
    )
    return bt.run()


def evaluate_on_data(
    strategy_code: str,
    data: pd.DataFrame,
    metrics_calculator: "MetricsCalculator" = None,
    cash: float = 10000,
    commission: float = 0.002,
) -> "StrategyMetrics":
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


def code_hash(strategy_code: str) -> str:
    """Compute SHA256 hash of strategy code for caching."""
    return hashlib.sha256(strategy_code.encode()).hexdigest()[:16]
```

---

## Evaluation Cache

```python
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import hashlib


@dataclass
class CacheKey:
    """Key for evaluation cache."""
    code_hash: str
    stage_name: str
    data_window_id: str  # e.g., "fold1_val" or hash of data slice
    config_hash: str     # hash of cash, commission, etc.

    def __hash__(self):
        return hash((self.code_hash, self.stage_name,
                     self.data_window_id, self.config_hash))

    def __eq__(self, other):
        return (self.code_hash == other.code_hash and
                self.stage_name == other.stage_name and
                self.data_window_id == other.data_window_id and
                self.config_hash == other.config_hash)


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
            oldest_keys = list(self._cache.keys())[:self._max_size // 4]
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
```

---

## Metrics Model

### StrategyMetrics Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import math


# Configurable caps for infinite values
METRIC_CAPS = {
    "sortino": 10.0,
    "calmar": 10.0,
    "profit_factor": 100.0,
}


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
    ann_return: float = 0.0          # Annualized return (%)
    sharpe: float = 0.0              # Sharpe ratio (risk-adjusted return)
    max_drawdown: float = 0.0        # Maximum drawdown (%, NEGATIVE, e.g., -15.0)
    sortino: float = 0.0             # Sortino ratio (downside risk) - capped
    calmar: float = 0.0              # Calmar ratio (return / |max drawdown|) - capped

    # === Secondary Metrics (context/analysis) ===
    trade_count: int = 0             # Number of trades
    win_rate: float = 0.0            # Winning trades / total trades (%)
    profit_factor: float = 0.0       # Gross profit / gross loss - capped
    expectancy: float = 0.0          # Expected profit per trade (%)
    exposure_time: float = 0.0       # Time in market (%)

    # === Robustness Metrics (cross-fold) ===
    stability: float = 0.0           # Cross-fold std dev of returns (lower is better)
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
```

---

## MetricsCalculator

```python
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


def _clip_metric(value: float, metric_name: str) -> float:
    """Clip infinite values to configured caps."""
    if math.isinf(value):
        cap = METRIC_CAPS.get(metric_name, 100.0)
        return cap if value > 0 else -cap
    return value


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
        equity = backtest_result.get('_equity_curve', pd.DataFrame())
        trades = backtest_result.get('_trades', pd.DataFrame())

        # Basic metrics (already computed by backtesting.py)
        ann_return = float(backtest_result.get('Return (Ann.) [%]', 0.0))
        sharpe = float(backtest_result.get('Sharpe Ratio', 0.0))
        max_drawdown = float(backtest_result.get('Max. Drawdown [%]', 0.0))
        exposure_time = float(backtest_result.get('Exposure Time [%]', 0.0))

        # Trade-based metrics
        trade_count = int(backtest_result.get('# Trades', 0))
        win_rate = float(backtest_result.get('Win Rate [%]', 0.0))
        expectancy = float(backtest_result.get('Expectancy [%]', 0.0))

        # Compute additional metrics (with capping)
        sortino = self._compute_sortino(equity, ann_return)
        sortino = _clip_metric(sortino, 'sortino')

        calmar = self._compute_calmar(ann_return, max_drawdown)
        calmar = _clip_metric(calmar, 'calmar')

        profit_factor = self._compute_profit_factor(trades)
        profit_factor = _clip_metric(profit_factor, 'profit_factor')

        # Handle NaN sharpe
        if np.isnan(sharpe):
            sharpe = 0.0

        # Extract equity curve and trade returns
        equity_curve = []
        trade_returns = []

        if not equity.empty and 'Equity' in equity.columns:
            equity_curve = equity['Equity'].tolist()

        if not trades.empty and 'ReturnPct' in trades.columns:
            trade_returns = trades['ReturnPct'].tolist()

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
            # No negative returns - return capped positive value
            return METRIC_CAPS['sortino'] if ann_return > 0 else 0.0

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
        if trades.empty or 'PnL' not in trades.columns:
            return 0.0

        profits = trades[trades['PnL'] > 0]['PnL'].sum()
        losses = abs(trades[trades['PnL'] < 0]['PnL'].sum())

        if losses == 0:
            # No losses - return capped value
            return METRIC_CAPS['profit_factor'] if profits > 0 else 0.0

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
```

---

## Evaluation Cascade

### Stage Protocol and Output

```python
from typing import Protocol, Tuple, Optional, List
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

    def to_dict(self) -> Dict:
        """Serialize for Program DB storage."""
        return {
            'result': self.result.value,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'error': self.error,
            'duration_ms': self.duration_ms,
        }


class EvaluationStage(Protocol):
    """Protocol for evaluation stages."""

    name: str
    order: int  # Lower = earlier in cascade

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        cache: Optional[EvaluationCache] = None,
        **kwargs
    ) -> StageOutput:
        """Run this evaluation stage."""
        ...
```

### Promotion Gates (NEW)

```python
@dataclass
class PromotionGate:
    """
    Threshold gate for promoting strategies between stages.

    Applied after Stage 3 (SingleFoldStage) to prevent junk
    strategies from proceeding to expensive Stage 4.
    """
    min_trades: int = 1               # At least N trades required
    max_drawdown_limit: float = -80.0 # Max drawdown must be > this (e.g., -80%)
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
            return False, f"Drawdown too severe: {metrics.max_drawdown}% < {self.max_drawdown_limit}%"

        if not metrics.is_valid():
            return False, "Metrics contain NaN or Inf values"

        if self.min_sharpe is not None and metrics.sharpe < self.min_sharpe:
            return False, f"Sharpe too low: {metrics.sharpe:.2f} < {self.min_sharpe}"

        if self.min_win_rate is not None and metrics.win_rate < self.min_win_rate:
            return False, f"Win rate too low: {metrics.win_rate:.1f}% < {self.min_win_rate}%"

        return True, None
```

### Stage Implementations

```python
import ast
import time
import traceback
from typing import Optional
from backtesting import Strategy
import pandas as pd


class SyntaxCheckStage:
    """Stage 1: Parse and compile code (instant)."""

    name = "syntax_check"
    order = 1

    def evaluate(
        self,
        strategy_code: str,
        data: pd.DataFrame = None,
        cache: Optional[EvaluationCache] = None,
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
        cache: Optional[EvaluationCache] = None,
        **kwargs
    ) -> StageOutput:
        start = time.time()

        # Check cache
        if cache:
            key = cache.make_key(strategy_code, self.name, data,
                                 self.initial_capital, self.commission)
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
            _ = run_bt(strategy_class, data_slice,
                      self.initial_capital, self.commission)

            result = StageOutput(
                result=StageResult.PASS,
                duration_ms=(time.time() - start) * 1000
            )

        except ValueError as e:
            # From load_strategy_class - no valid Strategy found
            result = StageOutput(
                result=StageResult.FAIL,
                error=str(e),
                duration_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            result = StageOutput(
                result=StageResult.FAIL,
                error=f"Smoke test failed: {str(e)}",
                duration_ms=(time.time() - start) * 1000
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
        **kwargs
    ) -> StageOutput:
        start = time.time()

        # Check cache
        if cache:
            key = cache.make_key(strategy_code, self.name, data,
                                 self.initial_capital, self.commission)
            cached = cache.get(key)
            if cached:
                return cached

        try:
            # Use centralized helper
            strategy_class = load_strategy_class(strategy_code)
            bt_result = run_bt(strategy_class, data,
                              self.initial_capital, self.commission)

            # Compute metrics
            metrics = self.metrics_calc.compute_all(bt_result)

            # CHECK PROMOTION GATE
            gate_passed, gate_error = self.promotion_gate.check(metrics)
            if not gate_passed:
                result = StageOutput(
                    result=StageResult.FAIL,
                    metrics=metrics,
                    error=f"Promotion gate failed: {gate_error}",
                    duration_ms=(time.time() - start) * 1000
                )
            else:
                result = StageOutput(
                    result=StageResult.PASS,
                    metrics=metrics,
                    duration_ms=(time.time() - start) * 1000
                )

        except ValueError as e:
            result = StageOutput(
                result=StageResult.FAIL,
                error=str(e),
                duration_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            result = StageOutput(
                result=StageResult.FAIL,
                error=f"Single fold failed: {str(e)}\n{traceback.format_exc()}",
                duration_ms=(time.time() - start) * 1000
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
        cache: Optional[EvaluationCache] = None,
        **kwargs
    ) -> StageOutput:
        start = time.time()

        # MUST HAVE FOLDS
        if folds is None or len(folds) == 0:
            return StageOutput(
                result=StageResult.FAIL,
                error="FullWalkForwardStage requires folds parameter with at least one fold",
                duration_ms=(time.time() - start) * 1000
            )

        try:
            # Use centralized helper
            strategy_class = load_strategy_class(strategy_code)

            # Run across all folds
            fold_metrics = []

            for fold_num, (train, val, test) in enumerate(folds, 1):
                # Evaluate on validation data
                bt_result = run_bt(strategy_class, val,
                                  self.initial_capital, self.commission)
                metrics = self.metrics_calc.compute_all(bt_result)
                fold_metrics.append(metrics)

            # Compute aggregate metrics (includes robustness)
            aggregate = self.metrics_calc.compute_cross_fold_metrics(fold_metrics)

            return StageOutput(
                result=StageResult.PASS,
                metrics=aggregate,
                duration_ms=(time.time() - start) * 1000
            )

        except ValueError as e:
            return StageOutput(
                result=StageResult.FAIL,
                error=str(e),
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

    def to_dict(self) -> Dict:
        """Serialize for Program DB storage."""
        return {
            'passed': self.passed,
            'final_stage': self.final_stage,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'stage_results': {k: v.to_dict() for k, v in self.stage_results.items()},
            'total_duration_ms': self.total_duration_ms,
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
    ):
        """
        Args:
            stages: List of evaluation stages. Defaults to all stages.
            cache: Optional evaluation cache for speedup.
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

            output = stage.evaluate(strategy_code, data, cache=self.cache, **kwargs)
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

### SelectionPolicy Protocol (FIXED)

```python
from typing import Protocol, Optional, List
from abc import abstractmethod


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
        **kwargs
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
```

### Policy Implementations

```python
from dataclasses import dataclass
from typing import Dict, Optional, List


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
    w_stability: float = 0.0    # Penalize high cross-fold variance
    w_consistency: float = 0.0  # Reward consistent positive returns

    _baseline: Optional[StrategyMetrics] = None

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function mapping (-inf, inf) to (0, 1)."""
        import math
        return 1 / (1 + math.exp(-x))

    def compute_fitness(
        self,
        metrics: StrategyMetrics,
        baseline: StrategyMetrics = None
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
            self.w_return * return_score +
            self.w_sharpe * sharpe_score +
            self.w_drawdown * dd_score
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
            self.w_return * return_score +
            self.w_sharpe * sharpe_score +
            self.w_drawdown * dd_score
        )

    def should_accept(
        self,
        candidate: StrategyMetrics,
        baseline: StrategyMetrics,
        population_metrics: Optional[List[StrategyMetrics]] = None,
        **kwargs
    ) -> bool:
        """Accept if fitness exceeds baseline fitness."""
        self._baseline = baseline
        return self.compute_fitness(candidate, baseline) >= self.compute_fitness(baseline, baseline)


@dataclass
class GatedMASPolicy:
    """
    Multi-gate acceptance: must pass all thresholds.

    IMPROVED: Includes robustness gates (stability, consistency, worst fold).
    """

    # Primary gates
    min_return: float = 0.0         # Minimum annualized return (%)
    min_sharpe: float = 0.0         # Minimum Sharpe ratio
    max_drawdown: float = -50.0     # Maximum drawdown (%, NEGATIVE, so > this)
    min_trades: int = 1             # Minimum trade count
    min_win_rate: float = 0.0       # Minimum win rate (%)

    # Robustness gates (NEW)
    min_consistency: float = 0.0    # Minimum % of folds with positive return
    min_worst_fold: float = -100.0  # Worst fold return must be > this
    max_stability: float = 100.0    # Max cross-fold std dev (lower = more stable)

    def compute_fitness(self, metrics: StrategyMetrics) -> float:
        """Fitness is simply annualized return for ranking."""
        return metrics.ann_return

    def should_accept(
        self,
        candidate: StrategyMetrics,
        baseline: StrategyMetrics,
        population_metrics: Optional[List[StrategyMetrics]] = None,
        **kwargs
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
        candidate: StrategyMetrics,
        baseline: StrategyMetrics,
        population_metrics: Optional[List[StrategyMetrics]] = None,
        **kwargs
    ) -> bool:
        """
        Accept if not dominated by any existing strategy.

        FIXED: Uses population_metrics when provided.
        """
        # Use population if provided, otherwise just compare to baseline
        comparison_set = population_metrics if population_metrics else [baseline]

        for existing in comparison_set:
            if self.dominates(existing, candidate):
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
    selection_policy: SelectionPolicy = None,
    cascade: EvaluationCascade = None,
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
        cascade: Evaluation cascade (default: stages 1-3 for speed)
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

    # 2. Initialize population (track metrics for Pareto)
    population = [(strategy_class, baseline_metrics, parent_code)]
    population_metrics = [baseline_metrics]  # For Pareto selection

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

        # Generate new code (deterministic naming)
        new_class_name = f"{parent_class.__name__}_Gen{gen}"

        if prefer_diffs:
            new_code, used_diff = self.llm.generate_strategy_code_with_fallback(
                parent_code, improvement
            )
        else:
            new_code = self.llm.generate_strategy_code(parent_code, improvement)

        # Replace class name deterministically
        new_code = new_code.replace(
            f"class {parent_class.__name__}(",
            f"class {new_class_name}(",
            1  # Only replace first occurrence
        )

        # 4. Run evaluation cascade
        print("Evaluating...")
        result = cascade.evaluate(new_code, val_data)

        # Store cascade result in Program DB (NEW)
        if self.program_db:
            self._store_cascade_result(gen, fold, new_code, result)

        if not result.passed:
            print(f"  Rejected at {result.final_stage}")
            continue

        new_metrics = result.metrics
        print(f"  Return={new_metrics.ann_return:.2f}%, "
              f"Sharpe={new_metrics.sharpe:.2f}, "
              f"MaxDD={new_metrics.max_drawdown:.2f}%")

        # 5. Check acceptance (pass population for Pareto)
        if selection_policy.should_accept(
            new_metrics,
            baseline_metrics,
            population_metrics=population_metrics
        ):
            # Load strategy class using centralized helper (strict)
            new_class = load_strategy_class(new_code)
            # Rename to expected name
            new_class.__name__ = new_class_name

            population.append((new_class, new_metrics, new_code))
            population_metrics.append(new_metrics)
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


def _store_cascade_result(
    self,
    generation: int,
    fold: int,
    code: str,
    result: CascadeResult
) -> None:
    """Store cascade evaluation result in Program DB."""
    # This enables analysis of where strategies fail
    metadata = {
        'generation': generation,
        'fold': fold,
        'cascade_result': result.to_dict(),
    }
    # Store as annotation on strategy record if registered
    # Implementation depends on Program DB schema extensions
```

---

## CLI Integration

Add to `src/profit/main.py`:

```python
# === Selection Policy Arguments ===
parser.add_argument(
    '--selection-policy',
    choices=['weighted', 'gated', 'pareto'],
    default='gated',
    help='Selection policy for strategy acceptance'
)

# Primary thresholds
parser.add_argument('--min-return', type=float, default=0.0,
                    help='Minimum annualized return threshold')
parser.add_argument('--min-sharpe', type=float, default=0.0,
                    help='Minimum Sharpe ratio threshold')
parser.add_argument('--max-drawdown', type=float, default=-50.0,
                    help='Maximum drawdown threshold (negative, e.g., -50)')
parser.add_argument('--min-trades', type=int, default=1,
                    help='Minimum number of trades required')

# Robustness thresholds (NEW)
parser.add_argument('--min-consistency', type=float, default=0.0,
                    help='Minimum %% of folds with positive return')
parser.add_argument('--min-worst-fold', type=float, default=-100.0,
                    help='Minimum return for worst fold')
parser.add_argument('--max-stability', type=float, default=100.0,
                    help='Maximum cross-fold std dev (stability cap)')

# WeightedSum policy weights (NEW)
parser.add_argument('--w-return', type=float, default=0.5,
                    help='Weight for return in weighted policy')
parser.add_argument('--w-sharpe', type=float, default=0.3,
                    help='Weight for Sharpe in weighted policy')
parser.add_argument('--w-drawdown', type=float, default=0.2,
                    help='Weight for drawdown in weighted policy')

# Pareto objectives (NEW)
parser.add_argument('--pareto-objectives', nargs='+',
                    default=['ann_return', 'sharpe', 'max_drawdown'],
                    help='Objectives for Pareto selection')

# === Cascade Arguments ===
parser.add_argument('--skip-cascade', action='store_true',
                    help='Skip evaluation cascade (use direct backtest only)')
parser.add_argument('--quick-eval', action='store_true',
                    help='Use quick evaluation (syntax + smoke test only)')

# Smoke test config (NEW)
parser.add_argument('--smoke-months', type=int, default=3,
                    help='Months of data for smoke test')

# Metrics config (NEW)
parser.add_argument('--risk-free-rate', type=float, default=0.0,
                    help='Annual risk-free rate for Sharpe/Sortino')

# Promotion gate thresholds
parser.add_argument('--gate-min-trades', type=int, default=1,
                    help='Promotion gate: minimum trades')
parser.add_argument('--gate-max-drawdown', type=float, default=-80.0,
                    help='Promotion gate: max drawdown limit')


# In main():
# Build metrics calculator
metrics_calc = MetricsCalculator(risk_free_rate=args.risk_free_rate)

# Build promotion gate
promotion_gate = PromotionGate(
    min_trades=args.gate_min_trades,
    max_drawdown_limit=args.gate_max_drawdown,
)

# Build selection policy
if args.selection_policy == 'weighted':
    policy = WeightedSumPolicy(
        w_return=args.w_return,
        w_sharpe=args.w_sharpe,
        w_drawdown=args.w_drawdown,
    )
elif args.selection_policy == 'pareto':
    policy = ParetoPolicy(objectives=tuple(args.pareto_objectives))
else:
    policy = GatedMASPolicy(
        min_return=args.min_return,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        min_trades=args.min_trades,
        min_consistency=args.min_consistency,
        min_worst_fold=args.min_worst_fold,
        max_stability=args.max_stability,
    )

# Build cascade
if args.skip_cascade:
    cascade = None
elif args.quick_eval:
    cascade = EvaluationCascade([
        SyntaxCheckStage(),
        SmokeTestStage(slice_months=args.smoke_months),
    ])
else:
    cascade = EvaluationCascade([
        SyntaxCheckStage(),
        SmokeTestStage(slice_months=args.smoke_months),
        SingleFoldStage(
            metrics_calculator=metrics_calc,
            promotion_gate=promotion_gate,
        ),
        FullWalkForwardStage(metrics_calculator=metrics_calc),
    ])

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
└── evaluation.py         # NEW: metrics, cascade, policies, helpers, cache
```

---

## Deliverables

### Core Module (evaluation.py) ✅
- [x] Core helpers: `load_strategy_class()`, `run_bt()`, `evaluate_on_data()`, `code_hash()`
- [x] `EvaluationCache` class with LRU-style eviction
- [x] `StrategyMetrics` dataclass with capped infinites
- [x] `MetricsCalculator` class
  - [x] `compute_all()` method with metric capping
  - [x] `compute_cross_fold_metrics()` method
- [x] `PromotionGate` dataclass for stage 3 rejection
- [x] Evaluation stages:
  - [x] `SyntaxCheckStage`
  - [x] `SmokeTestStage` (with caching)
  - [x] `SingleFoldStage` (with promotion gate + caching)
  - [x] `FullWalkForwardStage` (fails if no folds)
- [x] `EvaluationCascade` class
  - [x] `evaluate()` method with cache support
  - [x] `quick_evaluate()` method
- [x] Selection policies:
  - [x] `WeightedSumPolicy` (baseline-relative normalization)
  - [x] `GatedMASPolicy` (with robustness gates)
  - [x] `ParetoPolicy` (uses population_metrics)

### Evolver Integration ✅
- [x] Updated `evolve_strategy()` passing population_metrics
- [x] Strict strategy class loading (no fallback)
- [x] Policy-based fitness ranking for best strategy

### CLI Integration ✅
- [x] Policy selection and threshold arguments
- [x] WeightedSum weight arguments
- [x] Pareto objectives argument
- [x] Smoke test config
- [x] Risk-free rate
- [x] Promotion gate thresholds

### Tests ✅
- [x] Metrics calculation accuracy
- [x] Infinite value capping
- [x] Each evaluation stage (pass/fail cases)
- [x] Promotion gate rejection
- [x] Cascade fail-fast behavior
- [x] Cache hit/miss
- [x] Selection policies with population context
- [x] Strict strategy class loading (rejects non-Strategy)

---

## Implementation Notes

- `ParetoPolicy.dominates` does not negate `max_drawdown`: the spec's negation double-counted
  the sign convention (metrics already store drawdown as negative). Fixed in commit `1fc9fec`
  with a regression test.
- If the seed strategy fails the cascade, the evolver warns and falls back to direct-backtest
  baseline metrics instead of raising, so a gate-tripping seed does not abort the run.
- When a cascade containing `FullWalkForwardStage` is invoked without folds, the evolver stops
  the cascade after `single_fold` rather than failing candidates on a missing input; folds are
  threaded automatically from `walk_forward_optimize()`.
- To prevent look-ahead bias, `walk_forward_optimize()` threads only folds up to and including
  the current one (`folds[:i]`) into fold *i*'s evolution loop: later folds' validation windows
  lie after fold *i*'s test period and must not influence candidate selection.
- Cascade stages receive the evolver's `expected_class_name`, so a candidate that defines
  helper classes is gated and scored on the named strategy class, not the first `Strategy`
  subclass in definition order.
