# Phase 18: Production Loop & Monitoring

## Objective

Enable continuous monitoring of deployed strategies, detect performance drift, and implement a champion/challenger rotation system. This ensures strategies adapt to changing market conditions rather than assuming a once-trained strategy works forever.

---

## Dependencies

- Phase 13 (Program Database) - strategy storage and retrieval
- Phase 15 (Multi-Metric Evaluation) - `StrategyMetrics`

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                     Production Monitoring System                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Champion Strategy                           │   │
│  │                                                                  │   │
│  │  • Currently deployed                                            │   │
│  │  • Monitored continuously                                        │   │
│  │  • Receives live/paper trade signals                            │   │
│  └───────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                      │
│                                  ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      StrategyMonitor                             │   │
│  │                                                                  │   │
│  │  • Log daily performance                                        │   │
│  │  • Compare to backtest expectations                             │   │
│  │  • Detect drift / regime change                                 │   │
│  └───────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                      │
│                    ┌─────────────┴─────────────┐                       │
│                    │                           │                        │
│                    ▼                           ▼                        │
│  ┌───────────────────────────┐   ┌───────────────────────────┐        │
│  │  No Drift Detected       │   │  Drift Detected!          │        │
│  │  → Continue monitoring    │   │  → Trigger re-evolution   │        │
│  └───────────────────────────┘   │  → Promote challenger     │        │
│                                  └───────────────────────────┘        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Challenger Pool                               │   │
│  │                                                                  │   │
│  │  • Background evolution                                         │   │
│  │  • Paper trading evaluation                                     │   │
│  │  • Ready to promote when champion fails                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Performance Log

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, date
from enum import Enum
import json


class DriftType(Enum):
    NONE = "none"
    PERFORMANCE = "performance"      # Returns degraded
    VOLATILITY = "volatility"        # Vol regime changed
    CORRELATION = "correlation"      # Asset correlations shifted
    DRAWDOWN = "drawdown"            # Excessive drawdown
    TURNOVER = "turnover"            # Unusual trading activity


@dataclass
class DailyPerformanceLog:
    """Single day's performance record."""

    date: date
    strategy_id: str

    # Returns
    daily_return: float = 0.0
    cumulative_return: float = 0.0

    # Risk
    current_drawdown: float = 0.0
    rolling_volatility: float = 0.0

    # Activity
    positions_held: int = 0
    trades_today: int = 0

    # Comparison to backtest
    expected_return: float = 0.0     # What backtest predicted
    return_deviation: float = 0.0    # Actual - expected


@dataclass
class DriftReport:
    """Report of detected performance drift."""

    detected: bool = False
    drift_type: DriftType = DriftType.NONE
    severity: float = 0.0            # 0 to 1
    message: str = ""

    # Evidence
    rolling_sharpe: float = 0.0
    expected_sharpe: float = 0.0
    current_drawdown: float = 0.0
    max_allowed_drawdown: float = 0.0

    # Recommendation
    should_trigger_reevolution: bool = False
    should_demote_champion: bool = False


@dataclass
class MonitoringState:
    """Persistent state of the monitoring system."""

    champion_id: str = ""
    champion_deployed_at: datetime = None

    # Performance history
    performance_log: List[DailyPerformanceLog] = field(default_factory=list)

    # Drift detection
    last_drift_check: datetime = None
    consecutive_drift_days: int = 0
    total_drift_events: int = 0

    # Challengers
    challenger_ids: List[str] = field(default_factory=list)
    challenger_performances: Dict[str, float] = field(default_factory=dict)

    # Re-evolution triggers
    reevolution_in_progress: bool = False
    last_reevolution: datetime = None
```

---

## Strategy Monitor

### File: `src/profit/monitoring.py`

```python
"""
Strategy monitoring and drift detection.
"""

import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from profit.evaluation import StrategyMetrics


class StrategyMonitor:
    """
    Monitors a deployed strategy for performance drift.

    Compares actual performance to backtest expectations
    and detects when the strategy may be failing.
    """

    def __init__(
        self,
        strategy_id: str,
        expected_metrics: StrategyMetrics,
        config: 'MonitoringConfig' = None,
        state_path: str = None
    ):
        self.strategy_id = strategy_id
        self.expected = expected_metrics
        self.config = config or MonitoringConfig()
        self.state_path = Path(state_path or f"monitoring/{strategy_id}_state.json")

        # Load or initialize state
        self.state = self._load_state()

    def _load_state(self) -> MonitoringState:
        """Load monitoring state from disk."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                data = json.load(f)
            return MonitoringState(**data)
        return MonitoringState(
            champion_id=self.strategy_id,
            champion_deployed_at=datetime.now()
        )

    def _save_state(self):
        """Save monitoring state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(self.state.__dict__, f, default=str, indent=2)

    def log_performance(
        self,
        date: date,
        daily_return: float,
        positions: int = 0,
        trades: int = 0
    ):
        """
        Log a day's performance.

        Call this daily with actual strategy results.
        """
        # Calculate cumulative return
        if self.state.performance_log:
            prev_cum = self.state.performance_log[-1].cumulative_return
            cumulative = (1 + prev_cum / 100) * (1 + daily_return / 100) - 1
            cumulative *= 100
        else:
            cumulative = daily_return

        # Calculate current drawdown
        returns = [log.daily_return for log in self.state.performance_log]
        returns.append(daily_return)
        equity = np.cumprod(1 + np.array(returns) / 100) * 100
        peak = np.maximum.accumulate(equity)
        drawdown = (equity[-1] - peak[-1]) / peak[-1] * 100

        # Calculate rolling volatility
        if len(returns) >= 20:
            rolling_vol = np.std(returns[-20:]) * np.sqrt(252)
        else:
            rolling_vol = np.std(returns) * np.sqrt(252) if returns else 0

        log_entry = DailyPerformanceLog(
            date=date,
            strategy_id=self.strategy_id,
            daily_return=daily_return,
            cumulative_return=cumulative,
            current_drawdown=drawdown,
            rolling_volatility=rolling_vol,
            positions_held=positions,
            trades_today=trades,
            expected_return=self.expected.ann_return / 252,  # Daily expected
            return_deviation=daily_return - (self.expected.ann_return / 252)
        )

        self.state.performance_log.append(log_entry)
        self._save_state()

    def detect_drift(self) -> DriftReport:
        """
        Check for performance drift.

        Returns a DriftReport indicating if drift was detected
        and what type.
        """
        if len(self.state.performance_log) < self.config.min_days_for_drift:
            return DriftReport(detected=False, message="Insufficient data for drift detection")

        # Get recent performance
        recent = self.state.performance_log[-self.config.rolling_window:]
        returns = [log.daily_return for log in recent]

        # Calculate rolling Sharpe
        if np.std(returns) > 0:
            rolling_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            rolling_sharpe = 0

        # Check for performance drift
        sharpe_threshold = self.expected.sharpe * self.config.sharpe_decay_threshold
        if rolling_sharpe < sharpe_threshold:
            return DriftReport(
                detected=True,
                drift_type=DriftType.PERFORMANCE,
                severity=1 - (rolling_sharpe / self.expected.sharpe),
                message=f"Rolling Sharpe ({rolling_sharpe:.2f}) below threshold ({sharpe_threshold:.2f})",
                rolling_sharpe=rolling_sharpe,
                expected_sharpe=self.expected.sharpe,
                should_trigger_reevolution=True
            )

        # Check for drawdown breach
        current_dd = recent[-1].current_drawdown
        if current_dd < self.config.max_drawdown_threshold:
            return DriftReport(
                detected=True,
                drift_type=DriftType.DRAWDOWN,
                severity=abs(current_dd / self.config.max_drawdown_threshold),
                message=f"Drawdown ({current_dd:.1f}%) exceeds threshold ({self.config.max_drawdown_threshold:.1f}%)",
                current_drawdown=current_dd,
                max_allowed_drawdown=self.config.max_drawdown_threshold,
                should_demote_champion=abs(current_dd) > abs(self.config.emergency_drawdown)
            )

        # Check for volatility regime change
        recent_vol = recent[-1].rolling_volatility
        expected_vol = self.expected.max_drawdown / 3  # Rough approximation
        if recent_vol > expected_vol * self.config.volatility_spike_threshold:
            return DriftReport(
                detected=True,
                drift_type=DriftType.VOLATILITY,
                severity=recent_vol / expected_vol - 1,
                message=f"Volatility spike: {recent_vol:.1f}% vs expected {expected_vol:.1f}%"
            )

        self.state.consecutive_drift_days = 0
        return DriftReport(detected=False, message="No drift detected")

    def should_trigger_reevolution(self) -> bool:
        """Check if re-evolution should be triggered."""
        drift = self.detect_drift()

        if drift.should_trigger_reevolution:
            self.state.consecutive_drift_days += 1
        else:
            self.state.consecutive_drift_days = 0

        # Trigger after consecutive drift days
        return self.state.consecutive_drift_days >= self.config.drift_days_before_reevolve

    def get_summary(self) -> Dict:
        """Get monitoring summary."""
        if not self.state.performance_log:
            return {'status': 'No data'}

        recent = self.state.performance_log[-20:]
        returns = [log.daily_return for log in recent]

        return {
            'strategy_id': self.strategy_id,
            'days_monitored': len(self.state.performance_log),
            'cumulative_return': self.state.performance_log[-1].cumulative_return,
            'current_drawdown': self.state.performance_log[-1].current_drawdown,
            'rolling_sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'expected_sharpe': self.expected.sharpe,
            'consecutive_drift_days': self.state.consecutive_drift_days,
            'drift_status': self.detect_drift().drift_type.value
        }


@dataclass
class MonitoringConfig:
    """Configuration for monitoring behavior."""

    # Drift detection
    rolling_window: int = 20                    # Days for rolling calculations
    min_days_for_drift: int = 10                # Min days before checking drift
    sharpe_decay_threshold: float = 0.5         # Trigger if Sharpe falls below 50% of expected
    max_drawdown_threshold: float = -15.0       # Trigger if DD exceeds this
    emergency_drawdown: float = -25.0           # Immediate champion demotion
    volatility_spike_threshold: float = 2.0     # Trigger if vol is 2x expected

    # Re-evolution triggers
    drift_days_before_reevolve: int = 5         # Consecutive drift days to trigger

    # Challenger evaluation
    challenger_eval_period: int = 30            # Days to evaluate challengers
    min_challenger_outperformance: float = 0.05 # 5% better to promote

    @classmethod
    def from_yaml(cls, path: str) -> 'MonitoringConfig':
        """Load config from YAML."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get('monitoring', {}))
```

---

## Champion/Challenger System

### File: `src/profit/production.py`

```python
"""
Champion/Challenger management for production deployment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path

from profit.program_db import ProgramDatabase, StrategyRecord
from profit.monitoring import StrategyMonitor, MonitoringConfig
from profit.evaluation import StrategyMetrics


@dataclass
class ChallengerResult:
    """Result of evaluating a challenger strategy."""

    strategy_id: str
    paper_return: float              # Paper trading return
    paper_sharpe: float              # Paper trading Sharpe
    vs_champion_return: float        # Return difference vs champion
    vs_champion_sharpe: float        # Sharpe difference vs champion
    days_evaluated: int
    ready_for_promotion: bool


class ChampionChallenger:
    """
    Manages champion/challenger strategy rotation.

    - Champion: Currently deployed strategy
    - Challengers: Strategies being paper-traded for potential promotion
    """

    def __init__(
        self,
        program_db: ProgramDatabase,
        config_path: str = "configs/production.yaml",
        state_path: str = "monitoring/production_state.json"
    ):
        self.db = program_db
        self.config = ProductionConfig.from_yaml(config_path)
        self.state_path = Path(state_path)

        # Load state
        self.state = self._load_state()

        # Initialize champion monitor
        if self.state.champion_id:
            champion = self.db.get_strategy(self.state.champion_id)
            if champion:
                self.champion_monitor = StrategyMonitor(
                    strategy_id=self.state.champion_id,
                    expected_metrics=StrategyMetrics.from_dict(champion.metrics),
                    config=self.config.monitoring
                )

    def _load_state(self) -> 'ProductionState':
        """Load production state."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                data = json.load(f)
            return ProductionState(**data)
        return ProductionState()

    def _save_state(self):
        """Save production state."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(self.state.__dict__, f, default=str, indent=2)

    def set_champion(self, strategy_id: str):
        """Set the champion strategy."""
        strategy = self.db.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_id}")

        self.state.champion_id = strategy_id
        self.state.champion_deployed_at = datetime.now().isoformat()
        self.state.champion_metrics = strategy.metrics

        # Initialize monitor
        self.champion_monitor = StrategyMonitor(
            strategy_id=strategy_id,
            expected_metrics=StrategyMetrics.from_dict(strategy.metrics),
            config=self.config.monitoring
        )

        self._save_state()
        print(f"Champion set to {strategy_id}")

    def add_challenger(self, strategy_id: str):
        """Add a challenger for paper trading evaluation."""
        if strategy_id not in self.state.challenger_ids:
            self.state.challenger_ids.append(strategy_id)
            self.state.challenger_start_dates[strategy_id] = datetime.now().isoformat()
            self._save_state()
            print(f"Added challenger: {strategy_id}")

    def remove_challenger(self, strategy_id: str):
        """Remove a challenger."""
        if strategy_id in self.state.challenger_ids:
            self.state.challenger_ids.remove(strategy_id)
            self._save_state()

    def log_daily_performance(
        self,
        strategy_id: str,
        date: date,
        daily_return: float,
        positions: int = 0,
        trades: int = 0
    ):
        """Log daily performance for a strategy (champion or challenger)."""
        if strategy_id == self.state.champion_id:
            self.champion_monitor.log_performance(date, daily_return, positions, trades)
        elif strategy_id in self.state.challenger_ids:
            # Track challenger performance
            if strategy_id not in self.state.challenger_returns:
                self.state.challenger_returns[strategy_id] = []
            self.state.challenger_returns[strategy_id].append(daily_return)
            self._save_state()

    def evaluate_challengers(self) -> List[ChallengerResult]:
        """Evaluate all challengers against champion."""
        results = []

        champion_returns = [
            log.daily_return
            for log in self.champion_monitor.state.performance_log
        ]

        for challenger_id in self.state.challenger_ids:
            challenger_returns = self.state.challenger_returns.get(challenger_id, [])

            if len(challenger_returns) < self.config.monitoring.challenger_eval_period:
                continue  # Not enough data

            # Calculate metrics
            champ_mean = np.mean(champion_returns[-len(challenger_returns):])
            champ_std = np.std(champion_returns[-len(challenger_returns):])
            champ_sharpe = champ_mean / champ_std * np.sqrt(252) if champ_std > 0 else 0

            chal_mean = np.mean(challenger_returns)
            chal_std = np.std(challenger_returns)
            chal_sharpe = chal_mean / chal_std * np.sqrt(252) if chal_std > 0 else 0

            # Cumulative returns
            champ_cum = (np.prod(1 + np.array(champion_returns[-len(challenger_returns):]) / 100) - 1) * 100
            chal_cum = (np.prod(1 + np.array(challenger_returns) / 100) - 1) * 100

            ready = (chal_sharpe - champ_sharpe) > self.config.monitoring.min_challenger_outperformance

            results.append(ChallengerResult(
                strategy_id=challenger_id,
                paper_return=chal_cum,
                paper_sharpe=chal_sharpe,
                vs_champion_return=chal_cum - champ_cum,
                vs_champion_sharpe=chal_sharpe - champ_sharpe,
                days_evaluated=len(challenger_returns),
                ready_for_promotion=ready
            ))

        return results

    def should_promote(self, challenger_id: str) -> bool:
        """Check if a challenger should be promoted."""
        results = self.evaluate_challengers()
        for r in results:
            if r.strategy_id == challenger_id:
                return r.ready_for_promotion
        return False

    def promote(self, challenger_id: str, reviewer: str = "system"):
        """
        Promote a challenger to champion.

        Requires human approval in most cases.
        """
        if challenger_id not in self.state.challenger_ids:
            raise ValueError(f"Not a challenger: {challenger_id}")

        # Store old champion
        old_champion = self.state.champion_id
        self.state.promotion_history.append({
            'old_champion': old_champion,
            'new_champion': challenger_id,
            'promoted_at': datetime.now().isoformat(),
            'promoted_by': reviewer
        })

        # Swap
        self.set_champion(challenger_id)
        self.remove_challenger(challenger_id)

        # Add old champion as challenger
        if old_champion:
            self.add_challenger(old_champion)

        print(f"Promoted {challenger_id} to champion (demoted {old_champion})")

    def check_and_respond_to_drift(self) -> Optional[str]:
        """
        Check for drift and take appropriate action.

        Returns action taken, if any.
        """
        if not self.champion_monitor:
            return None

        drift = self.champion_monitor.detect_drift()

        if drift.should_demote_champion:
            # Emergency: find best challenger
            results = self.evaluate_challengers()
            if results:
                best = max(results, key=lambda r: r.paper_sharpe)
                if best.paper_sharpe > 0:
                    self.promote(best.strategy_id, reviewer="emergency_auto")
                    return f"Emergency promotion: {best.strategy_id}"

        if drift.should_trigger_reevolution:
            self.state.reevolution_requested = True
            self._save_state()
            return "Re-evolution requested"

        return None


@dataclass
class ProductionState:
    """Persistent state for production system."""

    champion_id: str = ""
    champion_deployed_at: str = ""
    champion_metrics: Dict = field(default_factory=dict)

    challenger_ids: List[str] = field(default_factory=list)
    challenger_start_dates: Dict[str, str] = field(default_factory=dict)
    challenger_returns: Dict[str, List[float]] = field(default_factory=dict)

    promotion_history: List[Dict] = field(default_factory=list)
    reevolution_requested: bool = False


@dataclass
class ProductionConfig:
    """Configuration for production system."""

    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Challenger evolution
    challenger_evolution_enabled: bool = True
    challenger_evolution_schedule: str = "weekly"
    max_concurrent_challengers: int = 3

    @classmethod
    def from_yaml(cls, path: str) -> 'ProductionConfig':
        """Load from YAML file."""
        import yaml
        config_path = Path(path)

        if not config_path.exists():
            # Create default
            config_path.parent.mkdir(parents=True, exist_ok=True)
            default = cls()
            with open(config_path, 'w') as f:
                yaml.dump({
                    'monitoring': default.monitoring.__dict__,
                    'challenger_evolution_enabled': default.challenger_evolution_enabled,
                    'challenger_evolution_schedule': default.challenger_evolution_schedule,
                    'max_concurrent_challengers': default.max_concurrent_challengers,
                }, f, default_flow_style=False)
            return default

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(
            monitoring=MonitoringConfig(**data.get('monitoring', {})),
            challenger_evolution_enabled=data.get('challenger_evolution_enabled', True),
            challenger_evolution_schedule=data.get('challenger_evolution_schedule', 'weekly'),
            max_concurrent_challengers=data.get('max_concurrent_challengers', 3)
        )
```

---

## Production Config File

### File: `configs/production.yaml`

```yaml
# Production monitoring configuration

# Champion strategy
champion:
  strategy_id: ""           # Set after evolution
  deployed_at: ""

# Monitoring parameters
monitoring:
  rolling_window: 20
  min_days_for_drift: 10
  sharpe_decay_threshold: 0.5
  max_drawdown_threshold: -15.0
  emergency_drawdown: -25.0
  volatility_spike_threshold: 2.0
  drift_days_before_reevolve: 5
  challenger_eval_period: 30
  min_challenger_outperformance: 0.05

# Challenger evolution
challenger_evolution_enabled: true
challenger_evolution_schedule: weekly
max_concurrent_challengers: 3

# Re-evolution triggers
reevolution:
  enabled: true
  trigger_on_drift: true
  trigger_on_drawdown: true
  drawdown_threshold: -15.0
  schedule: ""              # Optional: "monthly" for scheduled re-evolution
```

---

## CLI Commands

Add to `src/profit/main.py`:

```python
def create_parser():
    parser = argparse.ArgumentParser(prog='profit')
    subparsers = parser.add_subparsers(dest='command')

    # ... existing commands ...

    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor deployed strategy')
    monitor_parser.add_argument('--strategy', help='Strategy ID to monitor')
    monitor_parser.add_argument('--status', action='store_true', help='Show monitoring status')
    monitor_parser.add_argument('--check-drift', action='store_true', help='Check for drift')

    # Production commands
    prod_parser = subparsers.add_parser('production', help='Production management')
    prod_parser.add_argument('--set-champion', help='Set strategy as champion')
    prod_parser.add_argument('--add-challenger', help='Add challenger strategy')
    prod_parser.add_argument('--list-challengers', action='store_true')
    prod_parser.add_argument('--evaluate', action='store_true', help='Evaluate challengers')
    prod_parser.add_argument('--promote', help='Promote challenger to champion')

    # Re-evolution command
    reevolve_parser = subparsers.add_parser('trigger-reevolution', help='Trigger strategy re-evolution')

    return parser


def handle_monitor(args, program_db):
    """Handle monitor command."""
    if args.status:
        # Show all monitored strategies
        monitoring_dir = Path("monitoring")
        if monitoring_dir.exists():
            for state_file in monitoring_dir.glob("*_state.json"):
                strategy_id = state_file.stem.replace("_state", "")
                strategy = program_db.get_strategy(strategy_id)
                if strategy:
                    monitor = StrategyMonitor(
                        strategy_id=strategy_id,
                        expected_metrics=StrategyMetrics.from_dict(strategy.metrics)
                    )
                    summary = monitor.get_summary()
                    print(f"\n{strategy_id}:")
                    for k, v in summary.items():
                        print(f"  {k}: {v}")
        return

    if args.check_drift and args.strategy:
        strategy = program_db.get_strategy(args.strategy)
        if not strategy:
            print(f"Strategy not found: {args.strategy}")
            return

        monitor = StrategyMonitor(
            strategy_id=args.strategy,
            expected_metrics=StrategyMetrics.from_dict(strategy.metrics)
        )
        drift = monitor.detect_drift()

        print(f"\nDrift Check for {args.strategy}:")
        print(f"  Detected: {drift.detected}")
        print(f"  Type: {drift.drift_type.value}")
        print(f"  Severity: {drift.severity:.2f}")
        print(f"  Message: {drift.message}")
        if drift.should_trigger_reevolution:
            print(f"  RECOMMENDATION: Trigger re-evolution")


def handle_production(args, program_db):
    """Handle production command."""
    cc = ChampionChallenger(program_db)

    if args.set_champion:
        cc.set_champion(args.set_champion)

    elif args.add_challenger:
        cc.add_challenger(args.add_challenger)

    elif args.list_challengers:
        print(f"\nChampion: {cc.state.champion_id}")
        print(f"Challengers:")
        for cid in cc.state.challenger_ids:
            print(f"  - {cid}")

    elif args.evaluate:
        results = cc.evaluate_challengers()
        print("\nChallenger Evaluation:")
        for r in results:
            print(f"\n  {r.strategy_id}:")
            print(f"    Paper Return: {r.paper_return:.2f}%")
            print(f"    Paper Sharpe: {r.paper_sharpe:.2f}")
            print(f"    vs Champion: {r.vs_champion_return:+.2f}%")
            print(f"    Ready for Promotion: {r.ready_for_promotion}")

    elif args.promote:
        # Requires confirmation
        print(f"Promoting {args.promote} to champion...")
        print("This will demote the current champion.")
        confirm = input("Confirm? (yes/no): ")
        if confirm.lower() == 'yes':
            cc.promote(args.promote, reviewer="user")
        else:
            print("Cancelled.")


def handle_trigger_reevolution(args, program_db, evolver):
    """Trigger re-evolution of the champion strategy."""
    cc = ChampionChallenger(program_db)

    if not cc.state.champion_id:
        print("No champion set. Use 'production --set-champion' first.")
        return

    champion = program_db.get_strategy(cc.state.champion_id)
    if not champion:
        print(f"Champion strategy not found: {cc.state.champion_id}")
        return

    print(f"Re-evolving from champion: {cc.state.champion_id}")
    print("This will create new challenger strategies...")

    # Run evolution with champion as seed
    # (Implementation depends on how evolver is configured)
    # Result strategies become challengers
```

---

## Scheduled Monitoring Script

### File: `scripts/run_monitoring.py`

```python
#!/usr/bin/env python3
"""
Daily monitoring script.

Run this via cron or scheduler to:
1. Log daily performance
2. Check for drift
3. Evaluate challengers
4. Trigger re-evolution if needed
"""

import argparse
from datetime import date
from pathlib import Path

from profit.program_db import ProgramDatabase, JsonFileBackend
from profit.production import ChampionChallenger
from profit.monitoring import StrategyMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily-return', type=float, help='Champion daily return')
    parser.add_argument('--positions', type=int, default=0)
    parser.add_argument('--trades', type=int, default=0)
    args = parser.parse_args()

    # Initialize
    db = ProgramDatabase(JsonFileBackend())
    cc = ChampionChallenger(db)

    if not cc.state.champion_id:
        print("No champion configured. Exiting.")
        return

    # Log performance
    if args.daily_return is not None:
        cc.log_daily_performance(
            strategy_id=cc.state.champion_id,
            date=date.today(),
            daily_return=args.daily_return,
            positions=args.positions,
            trades=args.trades
        )
        print(f"Logged: {args.daily_return:.2f}%")

    # Check drift
    action = cc.check_and_respond_to_drift()
    if action:
        print(f"Action taken: {action}")

    # Summary
    summary = cc.champion_monitor.get_summary()
    print(f"\nMonitoring Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

---

## File Structure

```
src/profit/
├── __init__.py
├── strategies.py
├── llm_interface.py
├── evolver.py
├── main.py               # Modified: monitor, production CLI commands
├── program_db.py
├── diff_utils.py
├── evaluation.py
├── sources.py
├── approval.py
├── universe.py
├── portfolio.py
├── monitoring.py         # NEW: StrategyMonitor, DriftDetector
├── production.py         # NEW: ChampionChallenger, ProductionConfig
├── agents/
│   └── ...
└── data/
    └── ...

configs/
├── sources.yaml
├── pending_approvals.yaml
├── universes/
└── production.yaml       # NEW: production configuration

monitoring/               # NEW: runtime state
├── {strategy_id}_state.json
└── production_state.json

scripts/
└── run_monitoring.py     # NEW: scheduled monitoring
```

---

## Deliverables

- [ ] Data structures:
  - [ ] `DailyPerformanceLog` dataclass
  - [ ] `DriftReport` dataclass
  - [ ] `DriftType` enum
  - [ ] `MonitoringState` dataclass
  - [ ] `MonitoringConfig` dataclass
- [ ] `StrategyMonitor` class:
  - [ ] `log_performance()` method
  - [ ] `detect_drift()` method
  - [ ] `should_trigger_reevolution()` method
  - [ ] State persistence
- [ ] Drift detection:
  - [ ] Performance degradation (Sharpe decay)
  - [ ] Drawdown breach
  - [ ] Volatility regime change
- [ ] `ChampionChallenger` class:
  - [ ] `set_champion()` method
  - [ ] `add_challenger()` method
  - [ ] `evaluate_challengers()` method
  - [ ] `promote()` method
  - [ ] `check_and_respond_to_drift()` method
- [ ] `ProductionConfig` with YAML loading
- [ ] CLI commands:
  - [ ] `profit monitor --status`
  - [ ] `profit monitor --check-drift`
  - [ ] `profit production --set-champion`
  - [ ] `profit production --add-challenger`
  - [ ] `profit production --evaluate`
  - [ ] `profit production --promote`
  - [ ] `profit trigger-reevolution`
- [ ] `configs/production.yaml` template
- [ ] `scripts/run_monitoring.py` for scheduling
- [ ] Tests for:
  - [ ] Drift detection algorithms
  - [ ] Champion/challenger promotion
  - [ ] State persistence
