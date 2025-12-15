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
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol

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
    ) -> List[StrategyRecord]:
        """Sample strategies for LLM inspiration.

        Modes:
        - "exploitation": Top performers by primary metric
        - "exploration": Random from diverse tags/behavior cells
        - "trajectory": Recently improved strategies
        - "mixed": Weighted combination (default)

        Args:
            n: Number of strategies to sample.
            mode: Sampling strategy.
            exclude_ids: Strategy IDs to exclude (e.g., current parent's DB ID).
            eval_context_id: Only sample from same evaluation context (Phase 13B).
            include_rejected: Include rejected strategies (for negative examples).

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

    def _sample_mixed(
        self, n: int, candidates: List[StrategyRecord]
    ) -> List[StrategyRecord]:
        """Mixed sampling: top performer + diverse + improving."""
        results = []
        used_ids = set()

        # Allocate slots
        n_exploit = max(1, n // 3)
        n_explore = max(1, n // 3)
        n_trajectory = n - n_exploit - n_explore

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
