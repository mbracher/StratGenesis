"""Program Database for AlphaEvolve-style strategy storage.

This module implements an AlphaEvolve-style program database that stores
all evolved strategies (both accepted and rejected) with lineage tracking,
enabling inspiration sampling for richer LLM prompts.

Phase 13A: Minimal Working DB
- StrategyStatus enum
- StrategyRecord dataclass
- ProgramDatabaseBackend protocol
- JsonFileBackend implementation
- ProgramDatabase class with registration and queries

Phase 13B: Make It Useful
- EvaluationContext dataclass for apples-to-apples comparison
- eval_context_id filtering
- STANDARD_METRICS schema for consistent metric naming
- next_method_excerpt extraction for richer LLM prompts

Phase 13C: Scale & Quality
- SqliteBackend implementation with proper parent join table
- Atomic writes for JSON backend (already implemented)
- Append-only index with compaction (already implemented)

Phase 13D: Advanced Sampling
- MAP-Elites style sampling from behavior descriptor cells
- Cross-island sampling across tags/assets for cross-pollination
- Multi-objective selection support (Pareto front, weighted sum)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Protocol

if TYPE_CHECKING:
    pass


class StrategyStatus(str, Enum):
    """Status of a strategy in the database."""

    ACCEPTED = "accepted"  # Passed MAS threshold
    REJECTED = "rejected"  # Failed MAS threshold
    COMPILE_FAILED = "compile_failed"  # Code didn't compile
    EVAL_FAILED = "eval_failed"  # Evaluation crashed
    SEED = "seed"  # Initial seed strategy


# Standard metrics schema for consistent naming across the system (Phase 13B)
STANDARD_METRICS = {
    # Returns
    "ann_return": "Annualized return percentage",
    "total_return": "Total return percentage",
    # Risk-adjusted
    "sharpe": "Sharpe ratio (risk-free=0)",
    "sortino": "Sortino ratio",
    "calmar": "Calmar ratio (return/max_dd)",
    # Risk
    "max_drawdown": "Maximum drawdown (negative)",
    "volatility": "Annualized volatility",
    "var_95": "Value at Risk 95%",
    # Trading behavior
    "trade_count": "Number of trades",
    "win_rate": "Percentage of winning trades",
    "avg_trade_return": "Average return per trade",
    "avg_holding_period": "Average holding period in bars",
    "exposure_time": "Percentage of time in market",
    # Profit metrics
    "profit_factor": "Gross profit / gross loss",
    "expectancy": "Expected value per trade",
}


# Phase 13D: Multi-objective selection support


@dataclass
class SelectionObjective:
    """Defines an objective for multi-objective selection.

    Attributes:
        metric: Name of the metric to optimize.
        weight: Weight for weighted sum selection (default: 1.0).
        minimize: If True, lower values are better (default: False).
        threshold: Optional minimum (or maximum if minimize) acceptable value.
    """

    metric: str
    weight: float = 1.0
    minimize: bool = False
    threshold: Optional[float] = None


# Default objectives for common selection policies
# NOTE: max_drawdown is stored as negative (e.g., -0.05 = 5% drawdown).
# Since higher values (closer to 0) are better, we use minimize=False.
DEFAULT_OBJECTIVES = [
    SelectionObjective(metric="ann_return", weight=1.0),
    SelectionObjective(metric="sharpe", weight=0.5),
    SelectionObjective(metric="max_drawdown", weight=0.3, minimize=False),
]


def compute_pareto_ranks(
    records: List["StrategyRecord"], objectives: List[SelectionObjective]
) -> Dict[str, int]:
    """Compute Pareto ranks for a list of strategy records.

    Pareto rank 0 = non-dominated (Pareto front)
    Pareto rank 1 = dominated only by rank 0, etc.

    Args:
        records: List of StrategyRecord objects.
        objectives: List of SelectionObjective defining the optimization goals.

    Returns:
        Dict mapping strategy ID to Pareto rank (0 = best).
    """
    if not records or not objectives:
        return {}

    n = len(records)
    dominated_count = [0] * n  # How many solutions dominate this one
    dominates = [[] for _ in range(n)]  # Which solutions this one dominates

    def dominates_check(a_idx: int, b_idx: int) -> bool:
        """Check if record a dominates record b."""
        a_metrics = records[a_idx].metrics
        b_metrics = records[b_idx].metrics

        at_least_one_better = False
        for obj in objectives:
            a_val = a_metrics.get(obj.metric, float("-inf") if not obj.minimize else float("inf"))
            b_val = b_metrics.get(obj.metric, float("-inf") if not obj.minimize else float("inf"))

            if obj.minimize:
                if a_val > b_val:  # a is worse
                    return False
                if a_val < b_val:
                    at_least_one_better = True
            else:
                if a_val < b_val:  # a is worse
                    return False
                if a_val > b_val:
                    at_least_one_better = True

        return at_least_one_better

    # Count domination relationships
    for i in range(n):
        for j in range(n):
            if i != j and dominates_check(i, j):
                dominates[i].append(j)
                dominated_count[j] += 1

    # Assign Pareto ranks using fronts
    ranks = {}
    current_front = [i for i in range(n) if dominated_count[i] == 0]
    rank = 0

    while current_front:
        for idx in current_front:
            ranks[records[idx].id] = rank

        next_front = []
        for idx in current_front:
            for dominated_idx in dominates[idx]:
                dominated_count[dominated_idx] -= 1
                if dominated_count[dominated_idx] == 0:
                    next_front.append(dominated_idx)

        current_front = next_front
        rank += 1

    return ranks


def compute_weighted_score(
    record: "StrategyRecord", objectives: List[SelectionObjective]
) -> float:
    """Compute weighted sum score for a strategy.

    Args:
        record: StrategyRecord to score.
        objectives: List of SelectionObjective with weights.

    Returns:
        Weighted sum score (higher is better).
    """
    score = 0.0
    for obj in objectives:
        val = record.metrics.get(obj.metric, 0.0)
        if obj.minimize:
            # For minimization objectives, negate the value
            val = -val
        score += obj.weight * val
    return score


def passes_thresholds(
    record: "StrategyRecord", objectives: List[SelectionObjective]
) -> bool:
    """Check if a strategy passes all threshold requirements.

    Args:
        record: StrategyRecord to check.
        objectives: List of SelectionObjective with optional thresholds.

    Returns:
        True if all thresholds are met, False otherwise.
    """
    for obj in objectives:
        if obj.threshold is None:
            continue
        val = record.metrics.get(obj.metric)
        if val is None:
            return False
        if obj.minimize:
            if val > obj.threshold:  # Should be less than threshold
                return False
        else:
            if val < obj.threshold:  # Should be greater than threshold
                return False
    return True


@dataclass
class EvaluationContext:
    """Captures the evaluation environment for reproducibility.

    Strategies should only be compared within the same context.
    This is CRITICAL for apples-to-apples comparison in the program database.
    """

    dataset_id: str = ""  # Hash or identifier of dataset
    dataset_source: str = ""  # e.g., "yahoo_finance", "polygon"
    timeframe: str = ""  # e.g., "1D", "1H"
    train_start: str = ""
    train_end: str = ""
    val_start: str = ""
    val_end: str = ""
    test_start: str = ""
    test_end: str = ""
    initial_capital: float = 10000.0
    commission: float = 0.002
    slippage: float = 0.0
    backtest_engine: str = "backtesting.py"

    def context_id(self) -> str:
        """Generate a hash of the evaluation context.

        This ID is used to filter strategies for comparison - only strategies
        evaluated in the same context should be compared.
        """
        content = (
            f"{self.dataset_id}:{self.timeframe}:{self.train_start}:"
            f"{self.val_end}:{self.commission}"
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class StrategyRecord:
    """Complete record of an evolved strategy.

    Stores all information about a strategy including its code, lineage,
    performance metrics, and classification tags.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    code: str = ""
    class_name: str = ""

    # Status (CRITICAL: store ALL strategies, not just accepted)
    status: StrategyStatus = StrategyStatus.ACCEPTED

    # Lineage (uses DB IDs, not class names)
    parent_ids: List[str] = field(default_factory=list)
    mutation_text: str = ""  # The improvement proposal that created this
    generation: int = 0

    # Evaluation results (store FULL metric vector)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Evaluation context (Phase 13B: CRITICAL for apples-to-apples comparison)
    eval_context: Optional[EvaluationContext] = None
    eval_context_id: str = ""  # Cached context hash for fast filtering

    # Classification
    tags: List[str] = field(default_factory=list)

    # Behavior descriptor for MAP-Elites style diversity (Phase 13D)
    behavior_descriptor: Dict[str, float] = field(default_factory=dict)

    # Code analysis (Phase 13B: for richer LLM prompts)
    next_method_excerpt: str = ""  # First ~80 lines of next() method
    diff_from_parent: str = ""  # Git-style diff if available

    # Metadata
    fold: Optional[int] = None
    asset: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    # For inspiration sampling
    improvement_delta: float = 0.0  # Performance improvement over parent

    # Validation vs Test metrics (Phase 15: for overfitting detection)
    val_return: Optional[float] = None  # Validation annualized return
    test_return: Optional[float] = None  # Test annualized return (filled after fold completes)

    # Code repair tracking
    repair_attempts: int = 0  # Number of code fixes needed (0-10)


class ProgramDatabaseBackend(Protocol):
    """Abstract interface for strategy storage backends."""

    def save(self, record: StrategyRecord) -> str:
        """Save a strategy record. Returns the strategy ID."""
        ...

    def load(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Load a strategy by ID. Returns None if not found."""
        ...

    def query(self, filters: Dict) -> List[StrategyRecord]:
        """Query strategies matching filters.

        Supported filters:
        - tags: List[str] - match any of these tags
        - min_return: float - minimum annualized return
        - max_drawdown: float - maximum acceptable drawdown
        - generation: int - specific generation
        - parent_id: str - children of a specific parent
        - status: StrategyStatus - filter by status
        - eval_context_id: str - filter by evaluation context (Phase 13B)
        """
        ...

    def list_all(self) -> List[str]:
        """List all strategy IDs."""
        ...

    def delete(self, strategy_id: str) -> bool:
        """Delete a strategy. Returns True if deleted."""
        ...

    def count(self) -> int:
        """Return total number of strategies."""
        ...


class JsonFileBackend:
    """File-based backend using JSON for simplicity and portability."""

    def __init__(self, db_dir: str = "program_db"):
        """Initialize the JSON file backend.

        Args:
            db_dir: Directory for storing strategy files.
        """
        self.db_dir = Path(db_dir)
        self.strategies_dir = self.db_dir / "strategies"
        self.index_path = self.db_dir / "index.jsonl"
        self.lineage_path = self.db_dir / "lineage.json"

        # Create directories if they don't exist
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

        # Load index into memory for fast queries
        self._index: Dict[str, Dict] = {}
        self._load_index()

        # Load lineage graph
        self._lineage: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self._load_lineage()

    def _load_index(self) -> None:
        """Load index from disk into memory."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self._index[entry["id"]] = entry

    def _append_to_index(self, entry: Dict) -> None:
        """Append a single entry to the index file."""
        with open(self.index_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _rebuild_index(self) -> None:
        """Rebuild index from scratch (for compaction)."""
        self._atomic_write(
            self.index_path,
            "\n".join(json.dumps(e) for e in self._index.values()) + "\n",
        )

    def _atomic_write(self, path: Path, content: str) -> None:
        """Write file atomically using temp file + rename."""
        fd, temp_path = tempfile.mkstemp(dir=self.db_dir)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            shutil.move(temp_path, path)
        except Exception:
            os.unlink(temp_path)
            raise

    def _load_lineage(self) -> None:
        """Load lineage graph from disk."""
        if self.lineage_path.exists():
            with open(self.lineage_path, "r") as f:
                self._lineage = json.load(f)

    def _save_lineage(self) -> None:
        """Persist lineage graph atomically."""
        self._atomic_write(self.lineage_path, json.dumps(self._lineage, indent=2))

    def save(self, record: StrategyRecord) -> str:
        """Save strategy record to disk."""
        # Convert to dict for JSON serialization
        record_dict = asdict(record)
        record_dict["created_at"] = record.created_at.isoformat()
        record_dict["status"] = (
            record.status.value
            if isinstance(record.status, StrategyStatus)
            else record.status
        )
        # Handle EvaluationContext serialization (Phase 13B)
        if record.eval_context is not None:
            record_dict["eval_context"] = asdict(record.eval_context)
        else:
            record_dict["eval_context"] = None

        # Save full record atomically
        record_path = self.strategies_dir / f"{record.id}.json"
        self._atomic_write(record_path, json.dumps(record_dict, indent=2))

        # Update index (summary for fast queries)
        index_entry = {
            "id": record.id,
            "class_name": record.class_name,
            "status": (
                record.status.value
                if isinstance(record.status, StrategyStatus)
                else record.status
            ),
            "tags": record.tags,
            "metrics": record.metrics,
            "generation": record.generation,
            "parent_ids": record.parent_ids,
            "eval_context_id": record.eval_context_id,  # Phase 13B: for fast filtering
            "created_at": record_dict["created_at"],
            "improvement_delta": record.improvement_delta,
            "behavior_descriptor": record.behavior_descriptor,
        }

        is_new = record.id not in self._index
        self._index[record.id] = index_entry

        if is_new:
            self._append_to_index(index_entry)
        else:
            self._rebuild_index()  # Update requires rebuild

        # Update lineage graph
        for parent_id in record.parent_ids:
            if parent_id not in self._lineage:
                self._lineage[parent_id] = []
            if record.id not in self._lineage[parent_id]:
                self._lineage[parent_id].append(record.id)
        self._save_lineage()

        return record.id

    def load(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Load full strategy record from disk."""
        record_path = self.strategies_dir / f"{strategy_id}.json"
        if not record_path.exists():
            return None

        with open(record_path, "r") as f:
            data = json.load(f)

        # Convert datetime string back to datetime object
        data["created_at"] = datetime.fromisoformat(data["created_at"])

        # Convert status string to enum
        if "status" in data and isinstance(data["status"], str):
            data["status"] = StrategyStatus(data["status"])

        # Convert eval_context dict back to EvaluationContext object (Phase 13B)
        if data.get("eval_context") is not None:
            data["eval_context"] = EvaluationContext(**data["eval_context"])

        return StrategyRecord(**data)

    def query(self, filters: Dict) -> List[StrategyRecord]:
        """Query strategies matching filters."""
        results = []

        for entry in self._index.values():
            if self._matches_filters(entry, filters):
                record = self.load(entry["id"])
                if record:
                    results.append(record)

        return results

    def _matches_filters(self, entry: Dict, filters: Dict) -> bool:
        """Check if an index entry matches the given filters."""
        # Status filter
        if "status" in filters:
            status_val = (
                filters["status"].value
                if isinstance(filters["status"], StrategyStatus)
                else filters["status"]
            )
            if entry.get("status") != status_val:
                return False

        # Eval context filter (Phase 13B: CRITICAL for apples-to-apples comparison)
        if "eval_context_id" in filters:
            if entry.get("eval_context_id") != filters["eval_context_id"]:
                return False

        # Tag filter (match any)
        if "tags" in filters:
            if not any(tag in entry.get("tags", []) for tag in filters["tags"]):
                return False

        # Metric filters
        metrics = entry.get("metrics", {})

        if "min_return" in filters:
            if metrics.get("ann_return", float("-inf")) < filters["min_return"]:
                return False

        if "max_drawdown" in filters:
            if abs(metrics.get("max_drawdown", 0)) > abs(filters["max_drawdown"]):
                return False

        if "generation" in filters:
            if entry.get("generation") != filters["generation"]:
                return False

        if "parent_id" in filters:
            if filters["parent_id"] not in entry.get("parent_ids", []):
                return False

        return True

    def list_all(self) -> List[str]:
        """List all strategy IDs."""
        return list(self._index.keys())

    def delete(self, strategy_id: str) -> bool:
        """Delete a strategy record."""
        record_path = self.strategies_dir / f"{strategy_id}.json"
        if record_path.exists():
            record_path.unlink()
            self._index.pop(strategy_id, None)
            self._rebuild_index()
            return True
        return False

    def count(self) -> int:
        """Return total number of strategies."""
        return len(self._index)

    def get_children(self, strategy_id: str) -> List[str]:
        """Get IDs of all direct children of a strategy."""
        return self._lineage.get(strategy_id, [])

    def get_top_performers(
        self,
        n: int = 10,
        metric: str = "ann_return",
        eval_context_id: Optional[str] = None,
        status: StrategyStatus = StrategyStatus.ACCEPTED,
    ) -> List[str]:
        """Get IDs of top n performers by a metric.

        Args:
            n: Number of top performers to return.
            metric: Metric to sort by (default: ann_return).
            eval_context_id: Filter by evaluation context (Phase 13B).
            status: Filter by status (default: ACCEPTED).

        Returns:
            List of strategy IDs sorted by metric (descending).
        """
        filtered = [
            e
            for e in self._index.values()
            if e.get("status") == status.value
            and (eval_context_id is None or e.get("eval_context_id") == eval_context_id)
        ]
        sorted_entries = sorted(
            filtered,
            key=lambda e: e.get("metrics", {}).get(metric, float("-inf")),
            reverse=True,
        )
        return [e["id"] for e in sorted_entries[:n]]


class SqliteBackend:
    """SQLite backend for larger databases with efficient queries (Phase 13C).

    Uses a proper relational schema with:
    - strategy_parents join table for parent-child relationships
    - Separate tables for metrics, tags, and behavior descriptors
    - Indexes for fast queries
    """

    def __init__(self, db_path: str = "program_db.sqlite"):
        """Initialize the SQLite backend.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    id TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'accepted',
                    mutation_text TEXT,
                    generation INTEGER DEFAULT 0,
                    fold INTEGER,
                    asset TEXT,
                    eval_context_id TEXT,
                    improvement_delta REAL DEFAULT 0.0,
                    next_method_excerpt TEXT,
                    diff_from_parent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- CRITICAL: Proper join table for parent relationships (not JSON!)
                CREATE TABLE IF NOT EXISTS strategy_parents (
                    strategy_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    PRIMARY KEY (strategy_id, parent_id),
                    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
                    FOREIGN KEY (parent_id) REFERENCES strategies(id)
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
                    UNIQUE(strategy_id, metric_name)
                );

                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
                    UNIQUE(strategy_id, tag)
                );

                CREATE TABLE IF NOT EXISTS behavior_descriptors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    descriptor_name TEXT NOT NULL,
                    descriptor_value REAL NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
                    UNIQUE(strategy_id, descriptor_name)
                );

                CREATE TABLE IF NOT EXISTS eval_contexts (
                    context_id TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    dataset_source TEXT,
                    timeframe TEXT,
                    train_start TEXT,
                    train_end TEXT,
                    val_start TEXT,
                    val_end TEXT,
                    test_start TEXT,
                    test_end TEXT,
                    initial_capital REAL,
                    commission REAL,
                    slippage REAL,
                    backtest_engine TEXT
                );

                -- Indexes for fast queries
                CREATE INDEX IF NOT EXISTS idx_strategies_generation
                    ON strategies(generation);
                CREATE INDEX IF NOT EXISTS idx_strategies_created_at
                    ON strategies(created_at);
                CREATE INDEX IF NOT EXISTS idx_strategies_status
                    ON strategies(status);
                CREATE INDEX IF NOT EXISTS idx_strategies_eval_context
                    ON strategies(eval_context_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_strategy
                    ON metrics(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_name_value
                    ON metrics(metric_name, metric_value);
                CREATE INDEX IF NOT EXISTS idx_tags_tag
                    ON tags(tag);
                CREATE INDEX IF NOT EXISTS idx_parents_parent
                    ON strategy_parents(parent_id);
                CREATE INDEX IF NOT EXISTS idx_behavior_strategy
                    ON behavior_descriptors(strategy_id);
            """
            )

    def save(self, record: StrategyRecord) -> str:
        """Save strategy record to database."""
        with self._get_conn() as conn:
            # Insert main record
            conn.execute(
                """
                INSERT OR REPLACE INTO strategies
                (id, code, class_name, status, mutation_text, generation,
                 fold, asset, eval_context_id, improvement_delta,
                 next_method_excerpt, diff_from_parent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.id,
                    record.code,
                    record.class_name,
                    record.status.value
                    if isinstance(record.status, StrategyStatus)
                    else record.status,
                    record.mutation_text,
                    record.generation,
                    record.fold,
                    record.asset,
                    record.eval_context_id,
                    record.improvement_delta,
                    record.next_method_excerpt,
                    record.diff_from_parent,
                    record.created_at.isoformat(),
                ),
            )

            # Insert parent relationships (proper join table)
            conn.execute(
                "DELETE FROM strategy_parents WHERE strategy_id = ?", (record.id,)
            )
            for parent_id in record.parent_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO strategy_parents (strategy_id, parent_id)
                    VALUES (?, ?)
                """,
                    (record.id, parent_id),
                )

            # Insert metrics (delete old ones first for updates)
            conn.execute("DELETE FROM metrics WHERE strategy_id = ?", (record.id,))
            for name, value in record.metrics.items():
                if value is not None:
                    conn.execute(
                        """
                        INSERT INTO metrics (strategy_id, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    """,
                        (record.id, name, value),
                    )

            # Insert tags
            conn.execute("DELETE FROM tags WHERE strategy_id = ?", (record.id,))
            for tag in record.tags:
                conn.execute(
                    """
                    INSERT INTO tags (strategy_id, tag)
                    VALUES (?, ?)
                """,
                    (record.id, tag),
                )

            # Insert behavior descriptors
            conn.execute(
                "DELETE FROM behavior_descriptors WHERE strategy_id = ?", (record.id,)
            )
            for name, value in record.behavior_descriptor.items():
                conn.execute(
                    """
                    INSERT INTO behavior_descriptors
                    (strategy_id, descriptor_name, descriptor_value)
                    VALUES (?, ?, ?)
                """,
                    (record.id, name, value),
                )

            # Insert eval context if present
            if record.eval_context:
                ctx = record.eval_context
                conn.execute(
                    """
                    INSERT OR REPLACE INTO eval_contexts
                    (context_id, dataset_id, dataset_source, timeframe,
                     train_start, train_end, val_start, val_end, test_start, test_end,
                     initial_capital, commission, slippage, backtest_engine)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.eval_context_id,
                        ctx.dataset_id,
                        ctx.dataset_source,
                        ctx.timeframe,
                        ctx.train_start,
                        ctx.train_end,
                        ctx.val_start,
                        ctx.val_end,
                        ctx.test_start,
                        ctx.test_end,
                        ctx.initial_capital,
                        ctx.commission,
                        ctx.slippage,
                        ctx.backtest_engine,
                    ),
                )

        return record.id

    def load(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Load strategy record from database."""
        with self._get_conn() as conn:
            # Get main record
            row = conn.execute(
                "SELECT * FROM strategies WHERE id = ?", (strategy_id,)
            ).fetchone()

            if not row:
                return None

            # Get parent IDs from join table
            parent_ids = [
                p_row["parent_id"]
                for p_row in conn.execute(
                    "SELECT parent_id FROM strategy_parents WHERE strategy_id = ?",
                    (strategy_id,),
                )
            ]

            # Get metrics
            metrics = {}
            for m_row in conn.execute(
                "SELECT metric_name, metric_value FROM metrics WHERE strategy_id = ?",
                (strategy_id,),
            ):
                metrics[m_row["metric_name"]] = m_row["metric_value"]

            # Get tags
            tags = [
                t_row["tag"]
                for t_row in conn.execute(
                    "SELECT tag FROM tags WHERE strategy_id = ?", (strategy_id,)
                )
            ]

            # Get behavior descriptors
            behavior_descriptor = {}
            for b_row in conn.execute(
                "SELECT descriptor_name, descriptor_value FROM behavior_descriptors WHERE strategy_id = ?",
                (strategy_id,),
            ):
                behavior_descriptor[b_row["descriptor_name"]] = b_row["descriptor_value"]

            # Get eval context if present
            eval_context = None
            if row["eval_context_id"]:
                ctx_row = conn.execute(
                    "SELECT * FROM eval_contexts WHERE context_id = ?",
                    (row["eval_context_id"],),
                ).fetchone()
                if ctx_row:
                    eval_context = EvaluationContext(
                        dataset_id=ctx_row["dataset_id"] or "",
                        dataset_source=ctx_row["dataset_source"] or "",
                        timeframe=ctx_row["timeframe"] or "",
                        train_start=ctx_row["train_start"] or "",
                        train_end=ctx_row["train_end"] or "",
                        val_start=ctx_row["val_start"] or "",
                        val_end=ctx_row["val_end"] or "",
                        test_start=ctx_row["test_start"] or "",
                        test_end=ctx_row["test_end"] or "",
                        initial_capital=ctx_row["initial_capital"] or 10000.0,
                        commission=ctx_row["commission"] or 0.002,
                        slippage=ctx_row["slippage"] or 0.0,
                        backtest_engine=ctx_row["backtest_engine"] or "backtesting.py",
                    )

            return StrategyRecord(
                id=row["id"],
                code=row["code"],
                class_name=row["class_name"],
                status=StrategyStatus(row["status"]),
                parent_ids=parent_ids,
                mutation_text=row["mutation_text"] or "",
                generation=row["generation"],
                metrics=metrics,
                tags=tags,
                behavior_descriptor=behavior_descriptor,
                eval_context=eval_context,
                eval_context_id=row["eval_context_id"] or "",
                fold=row["fold"],
                asset=row["asset"],
                created_at=datetime.fromisoformat(row["created_at"]),
                improvement_delta=row["improvement_delta"],
                next_method_excerpt=row["next_method_excerpt"] or "",
                diff_from_parent=row["diff_from_parent"] or "",
            )

    def query(self, filters: Dict) -> List[StrategyRecord]:
        """Query strategies with filters."""
        conditions = []
        params: List = []

        # Build query dynamically
        base_query = "SELECT DISTINCT s.id FROM strategies s"
        joins = []

        if "status" in filters:
            status_val = (
                filters["status"].value
                if isinstance(filters["status"], StrategyStatus)
                else filters["status"]
            )
            conditions.append("s.status = ?")
            params.append(status_val)

        if "eval_context_id" in filters:
            conditions.append("s.eval_context_id = ?")
            params.append(filters["eval_context_id"])

        if "tags" in filters:
            joins.append("JOIN tags t ON s.id = t.strategy_id")
            placeholders = ",".join(["?" for _ in filters["tags"]])
            conditions.append(f"t.tag IN ({placeholders})")
            params.extend(filters["tags"])

        if "min_return" in filters:
            joins.append("JOIN metrics m_ret ON s.id = m_ret.strategy_id")
            conditions.append(
                "m_ret.metric_name = 'ann_return' AND m_ret.metric_value >= ?"
            )
            params.append(filters["min_return"])

        if "max_drawdown" in filters:
            joins.append("JOIN metrics m_dd ON s.id = m_dd.strategy_id")
            conditions.append(
                "m_dd.metric_name = 'max_drawdown' AND ABS(m_dd.metric_value) <= ?"
            )
            params.append(abs(filters["max_drawdown"]))

        if "generation" in filters:
            conditions.append("s.generation = ?")
            params.append(filters["generation"])

        if "parent_id" in filters:
            joins.append("JOIN strategy_parents sp ON s.id = sp.strategy_id")
            conditions.append("sp.parent_id = ?")
            params.append(filters["parent_id"])

        # Build final query
        query = base_query
        # Dedupe joins while preserving order
        seen_joins = set()
        for join in joins:
            if join not in seen_joins:
                query += f" {join}"
                seen_joins.add(join)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._get_conn() as conn:
            ids = [row["id"] for row in conn.execute(query, params)]

        # FIX: Load once, not twice!
        records = [self.load(id) for id in ids]
        return [r for r in records if r is not None]

    def list_all(self) -> List[str]:
        """List all strategy IDs."""
        with self._get_conn() as conn:
            return [row["id"] for row in conn.execute("SELECT id FROM strategies")]

    def delete(self, strategy_id: str) -> bool:
        """Delete a strategy and its related records."""
        with self._get_conn() as conn:
            # Check if exists first
            row = conn.execute(
                "SELECT id FROM strategies WHERE id = ?", (strategy_id,)
            ).fetchone()
            if not row:
                return False

            # Delete related records first (foreign key constraints)
            conn.execute("DELETE FROM tags WHERE strategy_id = ?", (strategy_id,))
            conn.execute("DELETE FROM metrics WHERE strategy_id = ?", (strategy_id,))
            conn.execute(
                "DELETE FROM strategy_parents WHERE strategy_id = ?", (strategy_id,)
            )
            conn.execute(
                "DELETE FROM behavior_descriptors WHERE strategy_id = ?", (strategy_id,)
            )
            conn.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
            return True

    def count(self) -> int:
        """Return total number of strategies."""
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]

    def get_children(self, strategy_id: str) -> List[str]:
        """Get IDs of all direct children of a strategy."""
        with self._get_conn() as conn:
            return [
                row["strategy_id"]
                for row in conn.execute(
                    "SELECT strategy_id FROM strategy_parents WHERE parent_id = ?",
                    (strategy_id,),
                )
            ]

    def get_top_performers(
        self,
        n: int = 10,
        metric: str = "ann_return",
        eval_context_id: Optional[str] = None,
        status: StrategyStatus = StrategyStatus.ACCEPTED,
    ) -> List[str]:
        """Get IDs of top n performers by a metric.

        Args:
            n: Number of top performers to return.
            metric: Metric to sort by (default: ann_return).
            eval_context_id: Filter by evaluation context.
            status: Filter by status (default: ACCEPTED).

        Returns:
            List of strategy IDs sorted by metric (descending).
        """
        query = """
            SELECT s.id, m.metric_value
            FROM strategies s
            JOIN metrics m ON s.id = m.strategy_id
            WHERE s.status = ?
            AND m.metric_name = ?
        """
        params: List = [status.value, metric]

        if eval_context_id is not None:
            query += " AND s.eval_context_id = ?"
            params.append(eval_context_id)

        query += " ORDER BY m.metric_value DESC LIMIT ?"
        params.append(n)

        with self._get_conn() as conn:
            return [row["id"] for row in conn.execute(query, params)]


class ProgramDatabase:
    """Strategy archive with lineage tracking and inspiration sampling.

    Provides AlphaEvolve-style program database functionality:
    - Store ALL evolved strategies with full metadata (accepted AND rejected)
    - Track lineage (parent-child relationships via DB IDs)
    - Sample inspirations for LLM prompts using various strategies
    - Query by tags, metrics, and behavior descriptors
    """

    def __init__(self, backend: Optional[ProgramDatabaseBackend] = None):
        """Initialize the program database.

        Args:
            backend: Storage backend. Defaults to JsonFileBackend if not specified.
        """
        self.backend = backend or JsonFileBackend()
        self._primary_metric = "ann_return"  # Configurable per context

    def register_strategy(
        self,
        code: str,
        class_name: str,
        parent_ids: List[str],
        mutation_text: str,
        metrics: Dict[str, float],
        tags: List[str],
        status: StrategyStatus = StrategyStatus.ACCEPTED,
        generation: int = 0,
        fold: Optional[int] = None,
        asset: Optional[str] = None,
        eval_context: Optional[EvaluationContext] = None,
        improvement_delta: float = 0.0,
        diff_from_parent: str = "",
        val_return: Optional[float] = None,
        repair_attempts: int = 0,
    ) -> str:
        """Register a strategy in the database (accepted or rejected).

        Args:
            code: Full Python source code of the strategy.
            class_name: Name of the strategy class.
            parent_ids: List of parent strategy DB IDs.
            mutation_text: The improvement proposal that created this strategy.
            metrics: Performance metrics dict.
            tags: Classification tags.
            status: Strategy status (default: ACCEPTED).
            generation: Evolution generation number.
            fold: Walk-forward fold number.
            asset: Asset/symbol this strategy was evolved on.
            eval_context: EvaluationContext for apples-to-apples comparison (Phase 13B).
            improvement_delta: Performance improvement over parent.
            diff_from_parent: Git-style diff showing changes from parent strategy.
            val_return: Validation set annualized return.
            repair_attempts: Number of code repair attempts needed (0-10).

        Returns:
            The assigned strategy ID.
        """
        # Auto-compute eval_context_id (Phase 13B)
        eval_context_id = eval_context.context_id() if eval_context else ""

        # Extract next() method for richer prompts (Phase 13B)
        next_excerpt = self._extract_next_method(code)

        # Compute behavior descriptor
        behavior_descriptor = self._compute_behavior_descriptor(metrics)

        record = StrategyRecord(
            code=code,
            class_name=class_name,
            parent_ids=parent_ids,
            mutation_text=mutation_text,
            metrics=metrics,
            tags=tags,
            status=status,
            generation=generation,
            fold=fold,
            asset=asset,
            eval_context=eval_context,
            eval_context_id=eval_context_id,
            next_method_excerpt=next_excerpt,
            improvement_delta=improvement_delta,
            behavior_descriptor=behavior_descriptor,
            diff_from_parent=diff_from_parent,
            val_return=val_return,
            repair_attempts=repair_attempts,
        )
        return self.backend.save(record)

    def _extract_next_method(self, code: str, max_lines: int = 80) -> str:
        """Extract the next() method from strategy code for LLM prompts.

        Phase 13B: This provides a focused code excerpt for inspiration sampling,
        showing just the signal/entry/exit logic without boilerplate.

        Args:
            code: Full strategy source code.
            max_lines: Maximum lines to extract (default: 80).

        Returns:
            The next() method body, or empty string if not found.
        """
        # Find next() method using regex
        match = re.search(
            r"def next\(self\):(.*?)(?=\n    def |\nclass |\Z)", code, re.DOTALL
        )
        if match:
            next_code = "def next(self):" + match.group(1)
            lines = next_code.split("\n")[:max_lines]
            return "\n".join(lines)
        return ""

    def _compute_behavior_descriptor(
        self, metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute behavior descriptor for MAP-Elites style diversity.

        Bins trading behavior into discrete cells for diversity sampling.
        """
        descriptor = {}

        # Trade frequency bin (trades per year equivalent)
        trade_count = metrics.get("trade_count", 0)
        if trade_count < 10:
            descriptor["trade_freq_bin"] = 0  # Low frequency
        elif trade_count < 50:
            descriptor["trade_freq_bin"] = 1  # Medium
        else:
            descriptor["trade_freq_bin"] = 2  # High frequency

        # Risk bin (based on max drawdown)
        max_dd = abs(metrics.get("max_drawdown", 0))
        if max_dd < 0.1:
            descriptor["risk_bin"] = 0  # Low risk
        elif max_dd < 0.25:
            descriptor["risk_bin"] = 1  # Medium
        else:
            descriptor["risk_bin"] = 2  # High risk

        # Win rate bin
        win_rate = metrics.get("win_rate", 0.5)
        if win_rate < 0.4:
            descriptor["win_bin"] = 0  # Low win rate
        elif win_rate < 0.6:
            descriptor["win_bin"] = 1  # Medium
        else:
            descriptor["win_bin"] = 2  # High win rate

        return descriptor

    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Retrieve a strategy by ID."""
        return self.backend.load(strategy_id)

    def update_test_metrics(self, strategy_id: str, test_return: float) -> bool:
        """Update test metrics for a strategy after fold completion.

        This is called after the test phase to record how the strategy
        performed on unseen data, enabling overfitting detection.

        Args:
            strategy_id: The strategy's database ID.
            test_return: Test set annualized return.

        Returns:
            True if updated successfully, False if strategy not found.
        """
        record = self.backend.load(strategy_id)
        if record is None:
            return False

        record.test_return = test_return
        self.backend.save(record)
        return True

    def query_by_tags(self, tags: List[str]) -> List[StrategyRecord]:
        """Find strategies with any of the given tags."""
        return self.backend.query({"tags": tags})

    def query_by_status(self, status: StrategyStatus) -> List[StrategyRecord]:
        """Find strategies with the given status."""
        return self.backend.query({"status": status})

    def get_lineage(self, strategy_id: str) -> List[StrategyRecord]:
        """Get the full ancestry of a strategy (parents, grandparents, etc).

        Returns list ordered from oldest ancestor to immediate parent.
        """
        lineage = []
        current = self.backend.load(strategy_id)

        while current and current.parent_ids:
            # Get the first parent (primary lineage)
            parent_id = current.parent_ids[0]
            parent = self.backend.load(parent_id)
            if parent:
                lineage.insert(0, parent)
                current = parent
            else:
                break

        return lineage

    def sample_inspirations(
        self,
        n: int = 3,
        mode: str = "mixed",
        exclude_ids: Optional[List[str]] = None,
        eval_context_id: Optional[str] = None,
        include_rejected: bool = False,
        objectives: Optional[List[SelectionObjective]] = None,
    ) -> List[StrategyRecord]:
        """Sample strategies for LLM inspiration.

        Modes:
        - "exploitation": Top performers by primary metric
        - "exploration": Random from diverse tags/behavior cells
        - "trajectory": Recently improved strategies
        - "map_elites": Sample from different behavior descriptor cells (Phase 13D)
        - "cross_island": Sample across different tags/assets (Phase 13D)
        - "pareto": Sample from Pareto front (Phase 13D)
        - "weighted": Sample by weighted multi-objective score (Phase 13D)
        - "mixed": Weighted combination of multiple modes (default)

        Args:
            n: Number of strategies to sample.
            mode: Sampling strategy.
            exclude_ids: Strategy IDs to exclude (e.g., current parent's DB ID).
            eval_context_id: Only sample from same evaluation context (Phase 13B).
            include_rejected: Include rejected strategies (for negative examples).
            objectives: Selection objectives for pareto/weighted modes (Phase 13D).

        Returns:
            List of StrategyRecord objects for inspiration.
        """
        exclude_ids = exclude_ids or []

        # Build base filters
        filters = {}
        if eval_context_id:
            filters["eval_context_id"] = eval_context_id
        if not include_rejected:
            filters["status"] = StrategyStatus.ACCEPTED

        # Get candidate IDs
        candidates = self.backend.query(filters)
        candidates = [c for c in candidates if c.id not in exclude_ids]

        if not candidates:
            return []

        if mode == "exploitation":
            return self._sample_top_performers(n, candidates)
        elif mode == "exploration":
            return self._sample_diverse(n, candidates)
        elif mode == "trajectory":
            return self._sample_improving(n, candidates)
        elif mode == "map_elites":
            return self._sample_map_elites(n, candidates)
        elif mode == "cross_island":
            return self._sample_cross_island(n, candidates)
        elif mode == "pareto":
            return self._sample_pareto(n, candidates, objectives or DEFAULT_OBJECTIVES)
        elif mode == "weighted":
            return self._sample_weighted(n, candidates, objectives or DEFAULT_OBJECTIVES)
        else:  # mixed
            return self._sample_mixed(n, candidates)

    def _sample_top_performers(
        self, n: int, candidates: List[StrategyRecord]
    ) -> List[StrategyRecord]:
        """Sample from top performing strategies."""
        sorted_candidates = sorted(
            candidates,
            key=lambda r: r.metrics.get(self._primary_metric, float("-inf")),
            reverse=True,
        )
        return sorted_candidates[:n]

    def _sample_diverse(
        self, n: int, candidates: List[StrategyRecord]
    ) -> List[StrategyRecord]:
        """Sample from different tag buckets for diversity."""
        # Group by primary tag
        tag_buckets: Dict[str, List[StrategyRecord]] = {}

        for record in candidates:
            primary_tag = record.tags[0] if record.tags else "untagged"
            if primary_tag not in tag_buckets:
                tag_buckets[primary_tag] = []
            tag_buckets[primary_tag].append(record)

        # Sample one from each bucket, cycling until we have n
        results = []
        bucket_keys = list(tag_buckets.keys())
        random.shuffle(bucket_keys)

        i = 0
        while len(results) < n and any(tag_buckets.values()):
            bucket = bucket_keys[i % len(bucket_keys)]
            if tag_buckets[bucket]:
                choice = random.choice(tag_buckets[bucket])
                results.append(choice)
                tag_buckets[bucket].remove(choice)
            i += 1
            if i > len(candidates):  # Safety break
                break

        return results

    def _sample_improving(
        self, n: int, candidates: List[StrategyRecord]
    ) -> List[StrategyRecord]:
        """Sample strategies that showed improvement."""
        improving = [c for c in candidates if c.improvement_delta > 0]

        # Sort by improvement delta
        sorted_candidates = sorted(
            improving,
            key=lambda r: r.improvement_delta,
            reverse=True,
        )

        return sorted_candidates[:n]

    def _sample_map_elites(
        self, n: int, candidates: List[StrategyRecord]
    ) -> List[StrategyRecord]:
        """Sample from different behavior descriptor cells (Phase 13D).

        This provides MAP-Elites style diversity by sampling the best
        strategy from different behavioral niches. Each cell is defined
        by the combination of behavior descriptor bins.

        Args:
            n: Number of strategies to sample.
            candidates: List of candidate strategies.

        Returns:
            List of elite strategies from different behavior cells.
        """
        # Group by behavior cell (combination of descriptor bins)
        cells: Dict[tuple, StrategyRecord] = {}

        for record in candidates:
            bd = record.behavior_descriptor
            cell_key = (
                bd.get("trade_freq_bin", 0),
                bd.get("risk_bin", 0),
                bd.get("win_bin", 0),
            )

            # Keep elite (best performer) per cell
            if cell_key not in cells:
                cells[cell_key] = record
            elif record.metrics.get(self._primary_metric, 0) > cells[
                cell_key
            ].metrics.get(self._primary_metric, 0):
                cells[cell_key] = record

        # Sample from different cells
        elites = list(cells.values())
        random.shuffle(elites)
        return elites[:n]

    def _sample_cross_island(
        self, n: int, candidates: List[StrategyRecord]
    ) -> List[StrategyRecord]:
        """Sample across different 'islands' (tags/assets) for cross-pollination (Phase 13D).

        This enables cross-pollination of ideas between strategies evolved
        on different assets or with different characteristics.

        Args:
            n: Number of strategies to sample.
            candidates: List of candidate strategies.

        Returns:
            List of strategies from different islands.
        """
        # Group by tag + asset combination
        islands: Dict[str, List[StrategyRecord]] = {}

        for record in candidates:
            tag = record.tags[0] if record.tags else "untagged"
            asset = record.asset or "default"
            island_key = f"{tag}:{asset}"

            if island_key not in islands:
                islands[island_key] = []
            islands[island_key].append(record)

        # Sample one from each island (the best performer from each)
        results = []
        island_keys = list(islands.keys())
        random.shuffle(island_keys)

        for island_key in island_keys:
            if len(results) >= n:
                break
            island_records = islands[island_key]
            if island_records:
                # Pick best from this island
                best = max(
                    island_records,
                    key=lambda r: r.metrics.get(self._primary_metric, 0),
                )
                results.append(best)

        return results[:n]

    def _sample_pareto(
        self,
        n: int,
        candidates: List[StrategyRecord],
        objectives: List[SelectionObjective],
    ) -> List[StrategyRecord]:
        """Sample from Pareto front (Phase 13D).

        Returns strategies that are non-dominated (Pareto optimal) or
        near the Pareto front, providing diverse trade-offs between
        multiple objectives.

        Args:
            n: Number of strategies to sample.
            candidates: List of candidate strategies.
            objectives: Selection objectives defining the optimization goals.

        Returns:
            List of Pareto-optimal or near-optimal strategies.
        """
        if not candidates:
            return []

        # Compute Pareto ranks
        ranks = compute_pareto_ranks(candidates, objectives)

        # Sort by Pareto rank (0 = front), then by primary metric
        sorted_candidates = sorted(
            candidates,
            key=lambda r: (
                ranks.get(r.id, float("inf")),
                -r.metrics.get(self._primary_metric, 0),
            ),
        )

        return sorted_candidates[:n]

    def _sample_weighted(
        self,
        n: int,
        candidates: List[StrategyRecord],
        objectives: List[SelectionObjective],
    ) -> List[StrategyRecord]:
        """Sample by weighted multi-objective score (Phase 13D).

        Computes a weighted sum of multiple metrics and samples the
        top scorers.

        Args:
            n: Number of strategies to sample.
            candidates: List of candidate strategies.
            objectives: Selection objectives with weights.

        Returns:
            List of top-scoring strategies by weighted sum.
        """
        if not candidates:
            return []

        # Sort by weighted score
        sorted_candidates = sorted(
            candidates,
            key=lambda r: compute_weighted_score(r, objectives),
            reverse=True,
        )

        return sorted_candidates[:n]

    def _sample_mixed(
        self, n: int, candidates: List[StrategyRecord]
    ) -> List[StrategyRecord]:
        """Mixed sampling: top performer + diverse + improving + map_elites (Phase 13D).

        Combines multiple sampling strategies for comprehensive coverage:
        - Exploitation: Top performers by primary metric
        - Exploration: Random from diverse tags
        - Trajectory: Strategies that showed improvement
        - MAP-Elites: Elites from different behavior cells
        """
        results = []
        used_ids: set = set()

        # Allocate slots (4 modes now)
        n_exploit = max(1, n // 4)
        n_explore = max(1, n // 4)
        n_trajectory = max(1, n // 4)
        n_map_elites = n - n_exploit - n_explore - n_trajectory

        # Sample from each mode
        exploit = self._sample_top_performers(n_exploit, candidates)
        for r in exploit:
            if r.id not in used_ids:
                results.append(r)
                used_ids.add(r.id)

        remaining = [c for c in candidates if c.id not in used_ids]
        explore = self._sample_diverse(n_explore, remaining)
        for r in explore:
            if r.id not in used_ids:
                results.append(r)
                used_ids.add(r.id)

        remaining = [c for c in candidates if c.id not in used_ids]
        trajectory = self._sample_improving(n_trajectory, remaining)
        for r in trajectory:
            if r.id not in used_ids:
                results.append(r)
                used_ids.add(r.id)

        remaining = [c for c in candidates if c.id not in used_ids]
        map_elites = self._sample_map_elites(n_map_elites, remaining)
        for r in map_elites:
            if r.id not in used_ids:
                results.append(r)
                used_ids.add(r.id)

        return results[:n]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        all_records = [self.backend.load(id) for id in self.backend.list_all()]
        all_records = [r for r in all_records if r]

        if not all_records:
            return {"count": 0}

        accepted = [r for r in all_records if r.status == StrategyStatus.ACCEPTED]
        returns = [r.metrics.get("ann_return", 0) for r in accepted]

        return {
            "count": len(all_records),
            "accepted": len(accepted),
            "rejected": len(
                [r for r in all_records if r.status == StrategyStatus.REJECTED]
            ),
            "failed": len(
                [
                    r
                    for r in all_records
                    if r.status
                    in [StrategyStatus.COMPILE_FAILED, StrategyStatus.EVAL_FAILED]
                ]
            ),
            "seeds": len(
                [r for r in all_records if r.status == StrategyStatus.SEED]
            ),
            "avg_return": sum(returns) / len(returns) if returns else 0,
            "max_return": max(returns) if returns else 0,
            "min_return": min(returns) if returns else 0,
            "generations": max((r.generation for r in all_records), default=0),
            "unique_tags": len(set(tag for r in all_records for tag in r.tags)),
            "behavior_cells": len(
                set(
                    tuple(r.behavior_descriptor.values())
                    for r in all_records
                    if r.behavior_descriptor
                )
            ),
        }
