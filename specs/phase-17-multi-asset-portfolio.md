# Phase 17: Multi-Asset & Portfolio Support

## Objective

Enable strategy evaluation across multiple assets and support portfolio construction. This phase is split into two parts:

- **Phase 17A**: Multi-asset robustness testing (implement first)
- **Phase 17B**: Full portfolio construction layer (implement after 17A)

---

## Dependencies

- Phase 5 (Backtesting Utilities) - existing `run_backtest()` method
- Phase 15 (Multi-Metric Evaluation) - `StrategyMetrics`, `MetricsCalculator`

---

## Phase 17A: Multi-Asset Robustness

### Objective

Evaluate strategies across multiple assets to test generalization. A strategy that works on many assets is more robust than one optimized for a single instrument.

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    Multi-Asset Evaluation                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    UniverseManifest                          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │   │
│  │  │ SPY     │ │ QQQ     │ │ IWM     │ │ DIA     │           │   │
│  │  │ equity  │ │ equity  │ │ equity  │ │ equity  │           │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │   │
│  └───────┼──────────┼──────────┼──────────┼────────────────────┘   │
│          │          │          │          │                         │
│          ▼          ▼          ▼          ▼                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Same Strategy Applied to Each                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│          │          │          │          │                         │
│          ▼          ▼          ▼          ▼                         │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │         Cross-Asset Robustness Metrics                      │    │
│  │  • Mean return    • Median return   • Consistency          │    │
│  │  • Worst-case     • Std dev         • % profitable         │    │
│  └────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

### Data Structures

#### UniverseManifest

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import yaml


@dataclass
class AssetConfig:
    """Configuration for a single asset in the universe."""

    ticker: str                    # Asset identifier
    data_path: str                 # Path to OHLCV data file
    asset_class: str = "equity"    # equity, futures, crypto, forex
    sector: str = ""               # Technology, Healthcare, etc.
    market: str = ""               # US, EU, Asia, etc.

    # Optional metadata
    description: str = ""
    currency: str = "USD"
    multiplier: float = 1.0        # Contract multiplier for futures

    def load_data(self) -> pd.DataFrame:
        """Load OHLCV data for this asset."""
        df = pd.read_csv(self.data_path, parse_dates=True, index_col=0)
        return df


@dataclass
class UniverseManifest:
    """
    Defines a trading universe of multiple assets.

    Supports loading from YAML config file.
    """

    name: str = ""
    description: str = ""
    assets: List[AssetConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> 'UniverseManifest':
        """Load universe from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)

        assets = [
            AssetConfig(
                ticker=a['ticker'],
                data_path=a['data_path'],
                asset_class=a.get('asset_class', 'equity'),
                sector=a.get('sector', ''),
                market=a.get('market', ''),
                description=a.get('description', ''),
                currency=a.get('currency', 'USD'),
                multiplier=a.get('multiplier', 1.0)
            )
            for a in config.get('assets', [])
        ]

        return cls(
            name=config.get('name', ''),
            description=config.get('description', ''),
            assets=assets
        )

    def to_yaml(self, path: str):
        """Save universe to YAML file."""
        config = {
            'name': self.name,
            'description': self.description,
            'assets': [
                {
                    'ticker': a.ticker,
                    'data_path': a.data_path,
                    'asset_class': a.asset_class,
                    'sector': a.sector,
                    'market': a.market,
                }
                for a in self.assets
            ]
        }
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for all assets."""
        return {asset.ticker: asset.load_data() for asset in self.assets}

    def get_by_asset_class(self, asset_class: str) -> List[AssetConfig]:
        """Filter assets by class."""
        return [a for a in self.assets if a.asset_class == asset_class]

    def get_by_sector(self, sector: str) -> List[AssetConfig]:
        """Filter assets by sector."""
        return [a for a in self.assets if a.sector == sector]
```

#### Example Universe YAML

```yaml
# configs/universes/us_etfs.yaml
name: US ETF Universe
description: Major US equity ETFs for robustness testing

assets:
  - ticker: SPY
    data_path: data/SPY_daily.csv
    asset_class: equity
    sector: Broad Market
    market: US
    description: S&P 500 ETF

  - ticker: QQQ
    data_path: data/QQQ_daily.csv
    asset_class: equity
    sector: Technology
    market: US
    description: Nasdaq 100 ETF

  - ticker: IWM
    data_path: data/IWM_daily.csv
    asset_class: equity
    sector: Small Cap
    market: US
    description: Russell 2000 ETF

  - ticker: DIA
    data_path: data/DIA_daily.csv
    asset_class: equity
    sector: Blue Chip
    market: US
    description: Dow Jones Industrial Average ETF

  - ticker: XLF
    data_path: data/XLF_daily.csv
    asset_class: equity
    sector: Financials
    market: US
    description: Financial Select Sector SPDR

  - ticker: XLE
    data_path: data/XLE_daily.csv
    asset_class: equity
    sector: Energy
    market: US
    description: Energy Select Sector SPDR
```

### Cross-Asset Metrics

```python
from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class CrossAssetMetrics:
    """
    Metrics aggregated across multiple assets.

    Measures strategy robustness and generalization.
    """

    # Central tendency
    mean_return: float = 0.0       # Mean annualized return across assets
    median_return: float = 0.0     # Median return (robust to outliers)
    mean_sharpe: float = 0.0       # Mean Sharpe ratio

    # Dispersion
    return_std: float = 0.0        # Standard deviation of returns
    sharpe_std: float = 0.0        # Standard deviation of Sharpes

    # Extremes
    best_asset: str = ""           # Asset with highest return
    best_return: float = 0.0       # Highest return achieved
    worst_asset: str = ""          # Asset with lowest return
    worst_return: float = 0.0      # Lowest return (robustness indicator)

    # Consistency
    profitable_pct: float = 0.0    # % of assets with positive return
    consistent_pct: float = 0.0    # % of assets beating baseline

    # Raw data
    per_asset_metrics: Dict[str, StrategyMetrics] = None

    @classmethod
    def from_asset_metrics(
        cls,
        per_asset: Dict[str, StrategyMetrics],
        baseline_return: float = 0.0
    ) -> 'CrossAssetMetrics':
        """Compute cross-asset metrics from per-asset results."""
        if not per_asset:
            return cls()

        returns = [m.ann_return for m in per_asset.values()]
        sharpes = [m.sharpe for m in per_asset.values()]

        # Find best and worst
        best_asset = max(per_asset.keys(), key=lambda k: per_asset[k].ann_return)
        worst_asset = min(per_asset.keys(), key=lambda k: per_asset[k].ann_return)

        return cls(
            mean_return=np.mean(returns),
            median_return=np.median(returns),
            mean_sharpe=np.mean(sharpes),
            return_std=np.std(returns),
            sharpe_std=np.std(sharpes),
            best_asset=best_asset,
            best_return=per_asset[best_asset].ann_return,
            worst_asset=worst_asset,
            worst_return=per_asset[worst_asset].ann_return,
            profitable_pct=sum(1 for r in returns if r > 0) / len(returns) * 100,
            consistent_pct=sum(1 for r in returns if r > baseline_return) / len(returns) * 100,
            per_asset_metrics=per_asset
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary (for persistence)."""
        return {
            'mean_return': self.mean_return,
            'median_return': self.median_return,
            'mean_sharpe': self.mean_sharpe,
            'return_std': self.return_std,
            'sharpe_std': self.sharpe_std,
            'best_asset': self.best_asset,
            'best_return': self.best_return,
            'worst_asset': self.worst_asset,
            'worst_return': self.worst_return,
            'profitable_pct': self.profitable_pct,
            'consistent_pct': self.consistent_pct,
        }
```

### MultiAssetEvaluator

```python
from typing import List, Dict, Optional, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


class MultiAssetEvaluator:
    """
    Evaluates strategies across multiple assets.

    Runs the same strategy on each asset in a universe and
    aggregates results into cross-asset robustness metrics.
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.002,
        metrics_calculator: MetricsCalculator = None,
        parallel: bool = True,
        max_workers: int = 4
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.metrics_calc = metrics_calculator or MetricsCalculator()
        self.parallel = parallel
        self.max_workers = max_workers

    def evaluate(
        self,
        strategy_code: str,
        universe: UniverseManifest,
        data_override: Dict[str, pd.DataFrame] = None
    ) -> CrossAssetMetrics:
        """
        Evaluate strategy across all assets in universe.

        Args:
            strategy_code: Strategy source code
            universe: Universe manifest defining assets
            data_override: Optional pre-loaded data dict

        Returns:
            CrossAssetMetrics with aggregated results
        """
        # Load data if not provided
        data = data_override or universe.load_all_data()

        # Compile strategy
        namespace = {}
        exec(strategy_code, globals(), namespace)
        strategy_class = self._find_strategy_class(namespace)

        if not strategy_class:
            raise ValueError("No valid strategy class found in code")

        # Evaluate on each asset
        if self.parallel:
            per_asset = self._evaluate_parallel(strategy_class, data)
        else:
            per_asset = self._evaluate_sequential(strategy_class, data)

        return CrossAssetMetrics.from_asset_metrics(per_asset)

    def _find_strategy_class(self, namespace: dict):
        """Find the Strategy class in namespace."""
        for name, obj in namespace.items():
            if isinstance(obj, type) and hasattr(obj, 'init') and hasattr(obj, 'next'):
                return obj
        return None

    def _evaluate_single(
        self,
        strategy_class,
        ticker: str,
        data: pd.DataFrame
    ) -> Tuple[str, StrategyMetrics]:
        """Evaluate strategy on single asset."""
        try:
            from backtesting import Backtest

            bt = Backtest(
                data,
                strategy_class,
                cash=self.initial_capital,
                commission=self.commission,
            )
            result = bt.run()
            metrics = self.metrics_calc.compute_all(result)
            return ticker, metrics

        except Exception as e:
            print(f"Warning: Failed to evaluate {ticker}: {e}")
            return ticker, StrategyMetrics()

    def _evaluate_parallel(
        self,
        strategy_class,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, StrategyMetrics]:
        """Evaluate on all assets in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_single, strategy_class, ticker, df): ticker
                for ticker, df in data.items()
            }

            for future in as_completed(futures):
                ticker, metrics = future.result()
                results[ticker] = metrics

        return results

    def _evaluate_sequential(
        self,
        strategy_class,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, StrategyMetrics]:
        """Evaluate on all assets sequentially."""
        results = {}
        for ticker, df in data.items():
            ticker, metrics = self._evaluate_single(strategy_class, ticker, df)
            results[ticker] = metrics
        return results

    def evaluate_with_cascade(
        self,
        strategy_code: str,
        universe: UniverseManifest,
        cascade: EvaluationCascade = None
    ) -> CrossAssetMetrics:
        """
        Evaluate with cascade on each asset.

        Uses evaluation cascade for fast rejection across assets.
        """
        data = universe.load_all_data()
        per_asset = {}

        for ticker, df in data.items():
            print(f"Evaluating on {ticker}...")

            if cascade:
                result = cascade.evaluate(strategy_code, df)
                if result.passed and result.metrics:
                    per_asset[ticker] = result.metrics
                else:
                    print(f"  Failed cascade on {ticker}")
                    per_asset[ticker] = StrategyMetrics()
            else:
                _, metrics = self._evaluate_single(None, ticker, df)
                per_asset[ticker] = metrics

        return CrossAssetMetrics.from_asset_metrics(per_asset)
```

### Evolver Integration for Multi-Asset

```python
def evolve_strategy_multi_asset(
    self,
    strategy_class,
    universe: UniverseManifest,
    max_iters: int = 15,
    selection_policy: SelectionPolicy = None,
    cascade: EvaluationCascade = None,
    use_median: bool = True  # Use median for robustness
):
    """
    Evolve strategy optimizing for multi-asset robustness.

    Instead of optimizing for a single asset, evolves strategies
    that generalize across the entire universe.
    """
    evaluator = MultiAssetEvaluator(
        initial_capital=self.initial_capital,
        commission=self.commission
    )

    # Load all data once
    all_data = universe.load_all_data()

    # Baseline evaluation
    parent_code = inspect.getsource(strategy_class)
    baseline_cross = evaluator.evaluate(parent_code, universe, all_data)

    fitness_metric = 'median_return' if use_median else 'mean_return'
    baseline_fitness = getattr(baseline_cross, fitness_metric)

    print(f"Baseline cross-asset: mean={baseline_cross.mean_return:.2f}%, "
          f"median={baseline_cross.median_return:.2f}%, "
          f"worst={baseline_cross.worst_return:.2f}%")

    # Population
    population = [(strategy_class, baseline_cross, parent_code)]
    best = (strategy_class, baseline_cross, parent_code)

    for gen in range(1, max_iters + 1):
        print(f"\n=== Generation {gen} ===")

        # Select and mutate
        parent_class, parent_cross, parent_code = random.choice(population)

        improvement = self.llm.generate_improvement(
            parent_code,
            f"Cross-asset: mean={parent_cross.mean_return:.2f}%, "
            f"worst={parent_cross.worst_return:.2f}% on {parent_cross.worst_asset}"
        )

        new_code = self.llm.generate_strategy_code(parent_code, improvement)

        # Evaluate across universe
        try:
            new_cross = evaluator.evaluate(new_code, universe, all_data)
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            continue

        new_fitness = getattr(new_cross, fitness_metric)

        print(f"  New: mean={new_cross.mean_return:.2f}%, "
              f"worst={new_cross.worst_return:.2f}%")

        # Accept if better
        if new_fitness >= baseline_fitness:
            population.append((None, new_cross, new_code))
            print(f"  ACCEPTED")

            if new_fitness > getattr(best[1], fitness_metric):
                best = (None, new_cross, new_code)
                print(f"  NEW BEST!")

    return best
```

### CLI Integration (Phase 17A)

```python
# Add to main.py

parser.add_argument(
    '--universe',
    help='Path to universe YAML file for multi-asset evaluation'
)
parser.add_argument(
    '--multi-asset',
    action='store_true',
    help='Enable multi-asset robustness optimization'
)
parser.add_argument(
    '--use-median',
    action='store_true',
    help='Use median (instead of mean) for multi-asset fitness'
)


# In main():
if args.universe:
    universe = UniverseManifest.from_yaml(args.universe)
    print(f"Loaded universe '{universe.name}' with {len(universe.assets)} assets")

    if args.multi_asset:
        results = evolver.evolve_strategy_multi_asset(
            strategy_class,
            universe,
            use_median=args.use_median
        )
    else:
        # Just evaluate existing strategy across universe
        evaluator = MultiAssetEvaluator()
        cross_metrics = evaluator.evaluate(
            inspect.getsource(strategy_class),
            universe
        )
        print(f"\nCross-Asset Results:")
        print(f"  Mean Return: {cross_metrics.mean_return:.2f}%")
        print(f"  Median Return: {cross_metrics.median_return:.2f}%")
        print(f"  Worst: {cross_metrics.worst_return:.2f}% ({cross_metrics.worst_asset})")
        print(f"  % Profitable: {cross_metrics.profitable_pct:.0f}%")
```

---

## Phase 17B: Portfolio Construction Layer

### Objective

Enable evolution of portfolio strategies that output weights across a universe, with a proper portfolio simulator that handles capital allocation, rebalancing, and constraints.

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    Portfolio Strategy Flow                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    PortfolioStrategy                         │   │
│  │                                                              │   │
│  │  generate_weights(universe_data) → DataFrame[dates × assets]│   │
│  │                                                              │   │
│  │   Date       SPY    QQQ    IWM    DIA   Cash                │   │
│  │   2024-01-02 0.30   0.30   0.20   0.20  0.00                │   │
│  │   2024-01-03 0.25   0.35   0.20   0.15  0.05                │   │
│  │   ...                                                        │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  PortfolioSimulator                          │   │
│  │                                                              │   │
│  │  • Apply weights to capital                                 │   │
│  │  • Handle rebalancing costs                                 │   │
│  │  • Enforce constraints (leverage, position limits)          │   │
│  │  • Track portfolio equity curve                             │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  PortfolioResult                             │   │
│  │                                                              │   │
│  │  • Portfolio returns, Sharpe, drawdown                      │   │
│  │  • Turnover statistics                                      │   │
│  │  • Per-asset contribution                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### PortfolioStrategy Base Class

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class PortfolioStrategy(ABC):
    """
    Base class for cross-sectional portfolio strategies.

    Unlike backtesting.Strategy which handles single-asset timing,
    PortfolioStrategy outputs target weights for a universe of assets.
    """

    # Strategy parameters (can be evolved)
    # EVOLVE-BLOCK: params
    lookback: int = 20
    # END-EVOLVE-BLOCK

    @abstractmethod
    def generate_weights(
        self,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate target portfolio weights for each date.

        Args:
            universe_data: Dict mapping ticker -> OHLCV DataFrame
                          All DataFrames should have aligned datetime indices

        Returns:
            DataFrame with:
            - Index: dates
            - Columns: asset tickers + 'Cash'
            - Values: target weights (should sum to 1.0 or less)
        """
        pass


class EqualWeightStrategy(PortfolioStrategy):
    """Example: Equal weight across all assets."""

    def generate_weights(
        self,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        # Get common dates
        first_ticker = list(universe_data.keys())[0]
        dates = universe_data[first_ticker].index

        n_assets = len(universe_data)
        weight = 1.0 / n_assets

        weights = pd.DataFrame(index=dates)
        for ticker in universe_data:
            weights[ticker] = weight

        weights['Cash'] = 0.0
        return weights


class MomentumWeightStrategy(PortfolioStrategy):
    """Example: Weight by recent momentum."""

    # EVOLVE-BLOCK: params
    lookback: int = 20
    top_n: int = 5
    # END-EVOLVE-BLOCK

    def generate_weights(
        self,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        # EVOLVE-BLOCK: signal_generation
        # Compute momentum for each asset
        momentums = {}
        for ticker, df in universe_data.items():
            momentums[ticker] = df['Close'].pct_change(self.lookback)

        momentum_df = pd.DataFrame(momentums)
        # END-EVOLVE-BLOCK

        # EVOLVE-BLOCK: weight_calculation
        weights = pd.DataFrame(index=momentum_df.index)

        for date in momentum_df.index:
            row = momentum_df.loc[date].dropna()
            if len(row) == 0:
                for ticker in universe_data:
                    weights.loc[date, ticker] = 0.0
                weights.loc[date, 'Cash'] = 1.0
                continue

            # Top N by momentum
            top_assets = row.nlargest(self.top_n).index
            weight = 1.0 / len(top_assets)

            for ticker in universe_data:
                weights.loc[date, ticker] = weight if ticker in top_assets else 0.0

            weights.loc[date, 'Cash'] = 0.0
        # END-EVOLVE-BLOCK

        return weights
```

### PortfolioConstraints

```python
from dataclasses import dataclass


@dataclass
class PortfolioConstraints:
    """
    Constraints for portfolio construction.
    """

    # Position limits
    max_position_size: float = 0.20      # Max weight in single asset
    min_position_size: float = 0.0       # Min weight (if > 0, forces diversification)

    # Leverage
    max_leverage: float = 1.0            # Max gross exposure (1.0 = no leverage)
    allow_shorting: bool = False         # Allow negative weights

    # Diversification
    min_holdings: int = 1                # Minimum number of positions
    max_holdings: int = 100              # Maximum number of positions

    # Turnover
    max_turnover: float = 1.0            # Max daily turnover (as fraction of NAV)

    # Sector limits (optional)
    max_sector_weight: float = 1.0       # Max weight in single sector


def enforce_constraints(
    weights: pd.DataFrame,
    constraints: PortfolioConstraints,
    universe: UniverseManifest = None
) -> pd.DataFrame:
    """
    Enforce portfolio constraints on weights.

    Adjusts weights to satisfy all constraints.
    """
    adjusted = weights.copy()

    for date in adjusted.index:
        row = adjusted.loc[date].copy()
        cash_col = 'Cash' if 'Cash' in row.index else None

        # Separate cash from assets
        if cash_col:
            cash = row[cash_col]
            assets = row.drop(cash_col)
        else:
            cash = 0.0
            assets = row

        # 1. Remove shorts if not allowed
        if not constraints.allow_shorting:
            assets = assets.clip(lower=0)

        # 2. Enforce position limits
        assets = assets.clip(upper=constraints.max_position_size)
        if constraints.min_position_size > 0:
            assets = assets.where(
                (assets >= constraints.min_position_size) | (assets == 0),
                0
            )

        # 3. Enforce holdings limits
        nonzero = assets[assets != 0]
        if len(nonzero) > constraints.max_holdings:
            # Keep top N by absolute weight
            top = assets.abs().nlargest(constraints.max_holdings).index
            assets = assets.where(assets.index.isin(top), 0)

        if len(nonzero) < constraints.min_holdings:
            # Can't fix without more info - just warn
            pass

        # 4. Enforce leverage
        gross = assets.abs().sum()
        if gross > constraints.max_leverage:
            scale = constraints.max_leverage / gross
            assets = assets * scale

        # 5. Normalize and add remaining to cash
        total = assets.sum()
        if cash_col:
            adjusted.loc[date, cash_col] = max(0, 1.0 - total)

        # Write back
        for ticker in assets.index:
            adjusted.loc[date, ticker] = assets[ticker]

    return adjusted
```

### PortfolioSimulator

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class PortfolioResult:
    """Results from portfolio simulation."""

    # Performance
    total_return: float = 0.0
    ann_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0

    # Activity
    avg_turnover: float = 0.0            # Average daily turnover
    total_turnover: float = 0.0          # Sum of all turnover
    avg_positions: float = 0.0           # Average number of positions

    # Per-asset contribution
    asset_contributions: Dict[str, float] = field(default_factory=dict)

    # Time series
    equity_curve: pd.Series = None
    weights_history: pd.DataFrame = None
    returns: pd.Series = None


class PortfolioSimulator:
    """
    Simulates portfolio performance given weights and prices.

    Handles:
    - Capital allocation according to weights
    - Transaction costs on rebalancing
    - Constraint enforcement
    - Portfolio-level metrics
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.002,        # As fraction of trade value
        slippage: float = 0.0005,         # As fraction of price
        constraints: PortfolioConstraints = None
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.constraints = constraints or PortfolioConstraints()

    def run(
        self,
        weights: pd.DataFrame,
        prices: Dict[str, pd.DataFrame]
    ) -> PortfolioResult:
        """
        Simulate portfolio given target weights and price data.

        Args:
            weights: DataFrame of target weights (dates × assets)
            prices: Dict of price DataFrames per asset

        Returns:
            PortfolioResult with performance metrics
        """
        # Enforce constraints
        weights = enforce_constraints(weights, self.constraints)

        # Get common dates
        dates = weights.index
        assets = [c for c in weights.columns if c != 'Cash']

        # Build returns DataFrame
        returns_data = {}
        for asset in assets:
            if asset in prices:
                price_df = prices[asset]
                returns_data[asset] = price_df['Close'].pct_change()

        returns_df = pd.DataFrame(returns_data).loc[dates]

        # Initialize tracking
        equity = [self.initial_capital]
        portfolio_returns = []
        turnover_history = []
        prev_weights = pd.Series(0.0, index=weights.columns)

        for i, date in enumerate(dates[:-1]):
            target_weights = weights.loc[date]
            daily_returns = returns_df.loc[dates[i + 1]]

            # Calculate turnover
            turnover = (target_weights - prev_weights).abs().sum() / 2
            turnover_history.append(turnover)

            # Transaction costs
            trade_cost = turnover * equity[-1] * (self.commission + self.slippage)

            # Portfolio return
            asset_weights = target_weights.drop('Cash', errors='ignore')
            port_return = (asset_weights * daily_returns.fillna(0)).sum()

            # New equity
            new_equity = equity[-1] * (1 + port_return) - trade_cost
            equity.append(new_equity)
            portfolio_returns.append((new_equity - equity[-2]) / equity[-2])

            prev_weights = target_weights

        # Build equity curve
        equity_series = pd.Series(equity[1:], index=dates[1:])
        returns_series = pd.Series(portfolio_returns, index=dates[1:])

        # Compute metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        ann_return = self._annualize_return(total_return, len(dates))
        sharpe = self._compute_sharpe(returns_series)
        max_dd = self._compute_max_drawdown(equity_series)

        # Per-asset contribution
        contributions = {}
        for asset in assets:
            asset_contribution = (
                weights[asset] * returns_df[asset].fillna(0)
            ).sum()
            contributions[asset] = asset_contribution

        return PortfolioResult(
            total_return=total_return,
            ann_return=ann_return,
            sharpe=sharpe,
            max_drawdown=max_dd,
            avg_turnover=np.mean(turnover_history),
            total_turnover=sum(turnover_history),
            avg_positions=(weights != 0).sum(axis=1).mean(),
            asset_contributions=contributions,
            equity_curve=equity_series,
            weights_history=weights,
            returns=returns_series
        )

    def _annualize_return(self, total_return: float, n_days: int) -> float:
        """Annualize total return."""
        years = n_days / 252
        if years <= 0:
            return 0.0
        return ((1 + total_return / 100) ** (1 / years) - 1) * 100

    def _compute_sharpe(self, returns: pd.Series, risk_free: float = 0.0) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free / 252
        if excess.std() == 0:
            return 0.0
        return excess.mean() / excess.std() * np.sqrt(252)

    def _compute_max_drawdown(self, equity: pd.Series) -> float:
        """Compute maximum drawdown percentage."""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        return drawdown.min()
```

### Portfolio Evolution

```python
def evolve_portfolio_strategy(
    self,
    strategy_class: type,  # PortfolioStrategy subclass
    universe: UniverseManifest,
    max_iters: int = 15,
    constraints: PortfolioConstraints = None
):
    """
    Evolve a portfolio strategy.

    Optimizes weight generation logic while respecting constraints.
    """
    simulator = PortfolioSimulator(
        initial_capital=self.initial_capital,
        commission=self.commission,
        constraints=constraints or PortfolioConstraints()
    )

    # Load data
    all_data = universe.load_all_data()
    prices = {ticker: df for ticker, df in all_data.items()}

    # Baseline
    parent_code = inspect.getsource(strategy_class)

    # Execute to get weights
    namespace = {}
    exec(parent_code, globals(), namespace)
    strategy = list(namespace.values())[-1]()
    baseline_weights = strategy.generate_weights(all_data)
    baseline_result = simulator.run(baseline_weights, prices)

    print(f"Baseline: Return={baseline_result.ann_return:.2f}%, "
          f"Sharpe={baseline_result.sharpe:.2f}, "
          f"Turnover={baseline_result.avg_turnover:.2%}")

    # Evolution loop
    population = [(strategy_class, baseline_result, parent_code)]
    best = (strategy_class, baseline_result, parent_code)

    for gen in range(1, max_iters + 1):
        # ... similar to evolve_strategy but for portfolio ...
        pass

    return best
```

### CLI Integration (Phase 17B)

```python
parser.add_argument(
    '--portfolio',
    action='store_true',
    help='Evolve portfolio strategy (requires --universe)'
)
parser.add_argument(
    '--max-position',
    type=float,
    default=0.20,
    help='Maximum position size (default: 0.20)'
)
parser.add_argument(
    '--max-leverage',
    type=float,
    default=1.0,
    help='Maximum leverage (default: 1.0, no leverage)'
)
parser.add_argument(
    '--allow-shorts',
    action='store_true',
    help='Allow short positions'
)
```

---

## File Structure

```
src/profit/
├── __init__.py
├── strategies.py
├── llm_interface.py
├── evolver.py            # Modified: multi-asset and portfolio evolution
├── main.py               # Modified: --universe, --portfolio args
├── program_db.py
├── diff_utils.py
├── evaluation.py
├── sources.py
├── approval.py
├── universe.py           # NEW: UniverseManifest, AssetConfig
├── portfolio.py          # NEW: PortfolioStrategy, Simulator, Constraints
├── agents/
│   └── ...
└── data/
    └── ...

configs/
├── sources.yaml
├── pending_approvals.yaml
└── universes/            # NEW
    ├── us_etfs.yaml
    └── crypto.yaml
```

---

## Deliverables

### Phase 17A (Multi-Asset Robustness)
- [ ] `AssetConfig` dataclass
- [ ] `UniverseManifest` class with YAML loading
- [ ] `CrossAssetMetrics` dataclass
- [ ] `MultiAssetEvaluator` class
  - [ ] Parallel evaluation
  - [ ] Cascade integration
- [ ] `evolve_strategy_multi_asset()` method
- [ ] CLI `--universe` and `--multi-asset` arguments
- [ ] Example universe YAML files
- [ ] Tests for multi-asset evaluation

### Phase 17B (Portfolio Layer)
- [ ] `PortfolioStrategy` base class
- [ ] `PortfolioConstraints` dataclass
- [ ] `enforce_constraints()` function
- [ ] `PortfolioSimulator` class
  - [ ] Turnover calculation
  - [ ] Transaction costs
  - [ ] Equity curve tracking
- [ ] `PortfolioResult` dataclass
- [ ] `evolve_portfolio_strategy()` method
- [ ] Example portfolio strategies (EqualWeight, Momentum)
- [ ] CLI `--portfolio` and constraint arguments
- [ ] Tests for portfolio simulation
