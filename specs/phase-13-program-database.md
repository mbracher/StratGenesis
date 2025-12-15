# Phase 13: Program Database & Rich Evolution Context

## Objective

Implement an AlphaEvolve-style program database that stores all evolved strategies (both accepted and rejected) with lineage tracking, enabling inspiration sampling for richer LLM prompts. This replaces the simple population list with a persistent, queryable knowledge base.

From the AlphaEvolve paper:

> The key innovation is storing many candidates and sampling inspirations from the database to build richer prompts, enabling cross-pollination of ideas across different evolutionary branches.

---

## Dependencies

- Phase 11 (Strategy Persistence) - existing `StrategyPersister` class
- Phase 12 (Dual-Model LLM) - existing `LLMClient` class

---

## Implementation Phases

This phase is split into sub-phases to keep deliverables focused:

| Sub-Phase | Focus | Description |
|-----------|-------|-------------|
| **13A** | Minimal Working DB | StrategyRecord, JsonFileBackend, ProgramDatabase with correct ID tracking |
| **13B** | Make It Useful | eval_context_id, multi-metric normalization, improved prompt payloads |
| **13C** | Scale & Quality | Atomic writes, SQLite backend with proper parent join table |
| **13D** | Advanced Sampling | MAP-Elites/islands-inspired sampling, multi-objective selection |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      ProgramDatabase                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ProgramDatabaseBackend (Protocol)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│           ▲                              ▲                       │
│           │                              │                       │
│  ┌────────┴────────┐          ┌─────────┴─────────┐            │
│  │  JsonFileBackend │          │   SqliteBackend   │            │
│  │   (default)      │          │   (for scale)     │            │
│  └──────────────────┘          └───────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Evolver Integration                       │
│  • Register ALL strategies (accepted, rejected, failed)          │
│  • Track _db_id on strategy classes for correct lineage          │
│  • Sample inspirations for LLM prompts                          │
│  • Query by tags, context, behavior descriptors                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### StrategyRecord

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import uuid

class StrategyStatus(str, Enum):
    """Status of a strategy in the database."""
    ACCEPTED = "accepted"           # Passed MAS threshold
    REJECTED = "rejected"           # Failed MAS threshold
    COMPILE_FAILED = "compile_failed"  # Code didn't compile
    EVAL_FAILED = "eval_failed"     # Evaluation crashed
    SEED = "seed"                   # Initial seed strategy

@dataclass
class EvaluationContext:
    """
    Captures the evaluation environment for reproducibility.
    Strategies should only be compared within the same context.
    """
    dataset_id: str = ""            # Hash or identifier of dataset
    dataset_source: str = ""        # e.g., "yahoo_finance", "polygon"
    timeframe: str = ""             # e.g., "1D", "1H"
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
        """Generate a hash of the evaluation context."""
        import hashlib
        content = f"{self.dataset_id}:{self.timeframe}:{self.train_start}:{self.val_end}:{self.commission}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class StrategyRecord:
    """Complete record of an evolved strategy."""

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
    # Standard metrics: ann_return, sharpe, sortino, max_drawdown,
    # trade_count, win_rate, avg_holding_period, volatility, calmar

    # Evaluation context (CRITICAL: for apples-to-apples comparison)
    eval_context: Optional[EvaluationContext] = None
    eval_context_id: str = ""  # Cached context hash

    # Classification
    tags: List[str] = field(default_factory=list)
    # tags like: "trend", "mean-reversion", "momentum", "volatility"

    # Behavior descriptor for MAP-Elites style diversity (Phase 13D)
    behavior_descriptor: Dict[str, float] = field(default_factory=dict)
    # e.g., trade_frequency_bin, avg_hold_bin, volatility_exposure_bin

    # Code analysis (for richer LLM prompts)
    next_method_excerpt: str = ""  # First ~80 lines of next() method
    diff_from_parent: str = ""     # Git-style diff if available

    # Metadata
    fold: Optional[int] = None
    asset: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    # For inspiration sampling
    improvement_delta: float = 0.0  # Performance improvement over parent

    # Future-proofing: data dependencies & human approval
    data_requirements: List[str] = field(default_factory=list)  # Required datasets/features
    data_source_status: str = ""  # proposed, approved, rejected
    research_sources: List[str] = field(default_factory=list)  # Paper/note references

    # Future-proofing: multi-asset/portfolio fields
    universe_id: Optional[str] = None
    portfolio_metrics: Dict[str, float] = field(default_factory=dict)
```

### Standard Metrics Schema

To ensure consistent metric naming across the system:

```python
STANDARD_METRICS = {
    # Returns
    'ann_return': 'Annualized return percentage',
    'total_return': 'Total return percentage',

    # Risk-adjusted
    'sharpe': 'Sharpe ratio (risk-free=0)',
    'sortino': 'Sortino ratio',
    'calmar': 'Calmar ratio (return/max_dd)',

    # Risk
    'max_drawdown': 'Maximum drawdown (negative)',
    'volatility': 'Annualized volatility',
    'var_95': 'Value at Risk 95%',

    # Trading behavior
    'trade_count': 'Number of trades',
    'win_rate': 'Percentage of winning trades',
    'avg_trade_return': 'Average return per trade',
    'avg_holding_period': 'Average holding period in bars',
    'exposure_time': 'Percentage of time in market',

    # Profit metrics
    'profit_factor': 'Gross profit / gross loss',
    'expectancy': 'Expected value per trade',
}
```

---

## Backend Protocol

### ProgramDatabaseBackend

```python
from typing import Protocol, List, Dict, Optional

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
        - eval_context_id: str - filter by evaluation context
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
```

---

## JsonFileBackend Implementation (Phase 13A/13C)

### Directory Structure

```
program_db/
├── index.jsonl              # Append-only index (Phase 13C: atomic)
├── strategies/
│   ├── a1b2c3d4.json       # Full strategy records
│   ├── e5f6g7h8.json
│   └── ...
└── lineage.json            # Parent-child relationships
```

### Implementation

```python
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict
from datetime import datetime

class JsonFileBackend:
    """File-based backend using JSON for simplicity and portability."""

    def __init__(self, db_dir: str = "program_db"):
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

    def _load_index(self):
        """Load index from disk into memory."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self._index[entry['id']] = entry

    def _append_to_index(self, entry: Dict):
        """Append a single entry to the index file (Phase 13C improvement)."""
        with open(self.index_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _rebuild_index(self):
        """Rebuild index from scratch (for compaction)."""
        self._atomic_write(self.index_path,
            '\n'.join(json.dumps(e) for e in self._index.values()) + '\n')

    def _atomic_write(self, path: Path, content: str):
        """Write file atomically using temp file + rename."""
        fd, temp_path = tempfile.mkstemp(dir=self.db_dir)
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
            shutil.move(temp_path, path)
        except:
            os.unlink(temp_path)
            raise

    def _load_lineage(self):
        """Load lineage graph from disk."""
        if self.lineage_path.exists():
            with open(self.lineage_path, 'r') as f:
                self._lineage = json.load(f)

    def _save_lineage(self):
        """Persist lineage graph atomically."""
        self._atomic_write(self.lineage_path, json.dumps(self._lineage, indent=2))

    def save(self, record: StrategyRecord) -> str:
        """Save strategy record to disk."""
        # Convert to dict for JSON serialization
        record_dict = asdict(record)
        record_dict['created_at'] = record.created_at.isoformat()
        if record.eval_context:
            record_dict['eval_context'] = asdict(record.eval_context)

        # Save full record atomically
        record_path = self.strategies_dir / f"{record.id}.json"
        self._atomic_write(record_path, json.dumps(record_dict, indent=2))

        # Update index (summary for fast queries)
        index_entry = {
            'id': record.id,
            'class_name': record.class_name,
            'status': record.status.value if isinstance(record.status, StrategyStatus) else record.status,
            'tags': record.tags,
            'metrics': record.metrics,
            'generation': record.generation,
            'parent_ids': record.parent_ids,
            'eval_context_id': record.eval_context_id,
            'created_at': record_dict['created_at'],
            'improvement_delta': record.improvement_delta,
            'behavior_descriptor': record.behavior_descriptor,
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

        with open(record_path, 'r') as f:
            data = json.load(f)

        # Convert datetime string back to datetime object
        data['created_at'] = datetime.fromisoformat(data['created_at'])

        # Convert status string to enum
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = StrategyStatus(data['status'])

        # Convert eval_context dict back to object
        if data.get('eval_context'):
            data['eval_context'] = EvaluationContext(**data['eval_context'])

        return StrategyRecord(**data)

    def query(self, filters: Dict) -> List[StrategyRecord]:
        """Query strategies matching filters."""
        results = []

        for entry in self._index.values():
            if self._matches_filters(entry, filters):
                record = self.load(entry['id'])
                if record:
                    results.append(record)

        return results

    def _matches_filters(self, entry: Dict, filters: Dict) -> bool:
        """Check if an index entry matches the given filters."""
        # Status filter
        if 'status' in filters:
            status_val = filters['status'].value if isinstance(filters['status'], StrategyStatus) else filters['status']
            if entry.get('status') != status_val:
                return False

        # Eval context filter (CRITICAL for apples-to-apples comparison)
        if 'eval_context_id' in filters:
            if entry.get('eval_context_id') != filters['eval_context_id']:
                return False

        # Tag filter (match any)
        if 'tags' in filters:
            if not any(tag in entry.get('tags', []) for tag in filters['tags']):
                return False

        # Metric filters
        metrics = entry.get('metrics', {})

        if 'min_return' in filters:
            if metrics.get('ann_return', float('-inf')) < filters['min_return']:
                return False

        if 'max_drawdown' in filters:
            if abs(metrics.get('max_drawdown', 0)) > abs(filters['max_drawdown']):
                return False

        if 'generation' in filters:
            if entry.get('generation') != filters['generation']:
                return False

        if 'parent_id' in filters:
            if filters['parent_id'] not in entry.get('parent_ids', []):
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

    def get_top_performers(self, n: int = 10, metric: str = 'ann_return',
                          eval_context_id: Optional[str] = None,
                          status: StrategyStatus = StrategyStatus.ACCEPTED) -> List[str]:
        """Get IDs of top n performers by a metric within same eval context."""
        filtered = [
            e for e in self._index.values()
            if (eval_context_id is None or e.get('eval_context_id') == eval_context_id)
            and e.get('status') == status.value
        ]
        sorted_entries = sorted(
            filtered,
            key=lambda e: e.get('metrics', {}).get(metric, float('-inf')),
            reverse=True
        )
        return [e['id'] for e in sorted_entries[:n]]
```

---

## SqliteBackend Implementation (Phase 13C)

### Schema (with proper parent join table)

```sql
CREATE TABLE strategies (
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
CREATE TABLE strategy_parents (
    strategy_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    PRIMARY KEY (strategy_id, parent_id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
    FOREIGN KEY (parent_id) REFERENCES strategies(id)
);

CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
    UNIQUE(strategy_id, metric_name)
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
    UNIQUE(strategy_id, tag)
);

CREATE TABLE behavior_descriptors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    descriptor_name TEXT NOT NULL,
    descriptor_value REAL NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
    UNIQUE(strategy_id, descriptor_name)
);

CREATE TABLE eval_contexts (
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
CREATE INDEX idx_strategies_generation ON strategies(generation);
CREATE INDEX idx_strategies_created_at ON strategies(created_at);
CREATE INDEX idx_strategies_status ON strategies(status);
CREATE INDEX idx_strategies_eval_context ON strategies(eval_context_id);
CREATE INDEX idx_metrics_strategy ON metrics(strategy_id);
CREATE INDEX idx_metrics_name_value ON metrics(metric_name, metric_value);
CREATE INDEX idx_tags_tag ON tags(tag);
CREATE INDEX idx_parents_parent ON strategy_parents(parent_id);
CREATE INDEX idx_behavior_strategy ON behavior_descriptors(strategy_id);
```

### Implementation (with fixed query - no double load)

```python
import sqlite3
import json
from typing import List, Dict, Optional
from datetime import datetime
from contextlib import contextmanager

class SqliteBackend:
    """SQLite backend for larger databases with efficient queries."""

    def __init__(self, db_path: str = "program_db.sqlite"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
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
            """)

    def save(self, record: StrategyRecord) -> str:
        """Save strategy record to database."""
        with self._get_conn() as conn:
            # Insert main record
            conn.execute("""
                INSERT OR REPLACE INTO strategies
                (id, code, class_name, status, mutation_text, generation,
                 fold, asset, eval_context_id, improvement_delta,
                 next_method_excerpt, diff_from_parent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.code,
                record.class_name,
                record.status.value if isinstance(record.status, StrategyStatus) else record.status,
                record.mutation_text,
                record.generation,
                record.fold,
                record.asset,
                record.eval_context_id,
                record.improvement_delta,
                record.next_method_excerpt,
                record.diff_from_parent,
                record.created_at.isoformat()
            ))

            # Insert parent relationships (proper join table)
            conn.execute("DELETE FROM strategy_parents WHERE strategy_id = ?", (record.id,))
            for parent_id in record.parent_ids:
                conn.execute("""
                    INSERT OR IGNORE INTO strategy_parents (strategy_id, parent_id)
                    VALUES (?, ?)
                """, (record.id, parent_id))

            # Insert metrics
            for name, value in record.metrics.items():
                conn.execute("""
                    INSERT OR REPLACE INTO metrics (strategy_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """, (record.id, name, value))

            # Insert tags
            for tag in record.tags:
                conn.execute("""
                    INSERT OR IGNORE INTO tags (strategy_id, tag)
                    VALUES (?, ?)
                """, (record.id, tag))

            # Insert behavior descriptors
            for name, value in record.behavior_descriptor.items():
                conn.execute("""
                    INSERT OR REPLACE INTO behavior_descriptors
                    (strategy_id, descriptor_name, descriptor_value)
                    VALUES (?, ?, ?)
                """, (record.id, name, value))

            # Insert eval context if present
            if record.eval_context:
                ctx = record.eval_context
                conn.execute("""
                    INSERT OR REPLACE INTO eval_contexts
                    (context_id, dataset_id, dataset_source, timeframe,
                     train_start, train_end, val_start, val_end, test_start, test_end,
                     initial_capital, commission, slippage, backtest_engine)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.eval_context_id,
                    ctx.dataset_id, ctx.dataset_source, ctx.timeframe,
                    ctx.train_start, ctx.train_end, ctx.val_start, ctx.val_end,
                    ctx.test_start, ctx.test_end,
                    ctx.initial_capital, ctx.commission, ctx.slippage, ctx.backtest_engine
                ))

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
                p_row['parent_id'] for p_row in conn.execute(
                    "SELECT parent_id FROM strategy_parents WHERE strategy_id = ?",
                    (strategy_id,)
                )
            ]

            # Get metrics
            metrics = {}
            for m_row in conn.execute(
                "SELECT metric_name, metric_value FROM metrics WHERE strategy_id = ?",
                (strategy_id,)
            ):
                metrics[m_row['metric_name']] = m_row['metric_value']

            # Get tags
            tags = [
                t_row['tag'] for t_row in conn.execute(
                    "SELECT tag FROM tags WHERE strategy_id = ?", (strategy_id,)
                )
            ]

            # Get behavior descriptors
            behavior_descriptor = {}
            for b_row in conn.execute(
                "SELECT descriptor_name, descriptor_value FROM behavior_descriptors WHERE strategy_id = ?",
                (strategy_id,)
            ):
                behavior_descriptor[b_row['descriptor_name']] = b_row['descriptor_value']

            # Get eval context if present
            eval_context = None
            if row['eval_context_id']:
                ctx_row = conn.execute(
                    "SELECT * FROM eval_contexts WHERE context_id = ?",
                    (row['eval_context_id'],)
                ).fetchone()
                if ctx_row:
                    eval_context = EvaluationContext(
                        dataset_id=ctx_row['dataset_id'] or '',
                        dataset_source=ctx_row['dataset_source'] or '',
                        timeframe=ctx_row['timeframe'] or '',
                        train_start=ctx_row['train_start'] or '',
                        train_end=ctx_row['train_end'] or '',
                        val_start=ctx_row['val_start'] or '',
                        val_end=ctx_row['val_end'] or '',
                        test_start=ctx_row['test_start'] or '',
                        test_end=ctx_row['test_end'] or '',
                        initial_capital=ctx_row['initial_capital'] or 10000.0,
                        commission=ctx_row['commission'] or 0.002,
                        slippage=ctx_row['slippage'] or 0.0,
                        backtest_engine=ctx_row['backtest_engine'] or 'backtesting.py'
                    )

            return StrategyRecord(
                id=row['id'],
                code=row['code'],
                class_name=row['class_name'],
                status=StrategyStatus(row['status']),
                parent_ids=parent_ids,
                mutation_text=row['mutation_text'] or '',
                generation=row['generation'],
                metrics=metrics,
                tags=tags,
                behavior_descriptor=behavior_descriptor,
                eval_context=eval_context,
                eval_context_id=row['eval_context_id'] or '',
                fold=row['fold'],
                asset=row['asset'],
                created_at=datetime.fromisoformat(row['created_at']),
                improvement_delta=row['improvement_delta'],
                next_method_excerpt=row['next_method_excerpt'] or '',
                diff_from_parent=row['diff_from_parent'] or ''
            )

    def query(self, filters: Dict) -> List[StrategyRecord]:
        """Query strategies with filters."""
        conditions = []
        params = []

        # Build query dynamically
        base_query = "SELECT DISTINCT s.id FROM strategies s"
        joins = []

        if 'status' in filters:
            status_val = filters['status'].value if isinstance(filters['status'], StrategyStatus) else filters['status']
            conditions.append("s.status = ?")
            params.append(status_val)

        if 'eval_context_id' in filters:
            conditions.append("s.eval_context_id = ?")
            params.append(filters['eval_context_id'])

        if 'tags' in filters:
            joins.append("JOIN tags t ON s.id = t.strategy_id")
            placeholders = ','.join(['?' for _ in filters['tags']])
            conditions.append(f"t.tag IN ({placeholders})")
            params.extend(filters['tags'])

        if 'min_return' in filters:
            joins.append("JOIN metrics m_ret ON s.id = m_ret.strategy_id")
            conditions.append("m_ret.metric_name = 'ann_return' AND m_ret.metric_value >= ?")
            params.append(filters['min_return'])

        if 'max_drawdown' in filters:
            joins.append("JOIN metrics m_dd ON s.id = m_dd.strategy_id")
            conditions.append("m_dd.metric_name = 'max_drawdown' AND ABS(m_dd.metric_value) <= ?")
            params.append(abs(filters['max_drawdown']))

        if 'generation' in filters:
            conditions.append("s.generation = ?")
            params.append(filters['generation'])

        if 'parent_id' in filters:
            joins.append("JOIN strategy_parents sp ON s.id = sp.strategy_id")
            conditions.append("sp.parent_id = ?")
            params.append(filters['parent_id'])

        # Build final query
        query = base_query
        for join in set(joins):  # Dedupe joins
            query += f" {join}"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._get_conn() as conn:
            ids = [row['id'] for row in conn.execute(query, params)]

        # FIX: Load once, not twice!
        records = [self.load(id) for id in ids]
        return [r for r in records if r is not None]

    def list_all(self) -> List[str]:
        """List all strategy IDs."""
        with self._get_conn() as conn:
            return [row['id'] for row in conn.execute("SELECT id FROM strategies")]

    def delete(self, strategy_id: str) -> bool:
        """Delete a strategy and its related records."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM tags WHERE strategy_id = ?", (strategy_id,))
            conn.execute("DELETE FROM metrics WHERE strategy_id = ?", (strategy_id,))
            conn.execute("DELETE FROM strategy_parents WHERE strategy_id = ?", (strategy_id,))
            conn.execute("DELETE FROM behavior_descriptors WHERE strategy_id = ?", (strategy_id,))
            result = conn.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
            return result.rowcount > 0

    def count(self) -> int:
        """Return total number of strategies."""
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
```

---

## ProgramDatabase Class

The main interface that uses the backend abstraction.

```python
from typing import List, Optional, Union
import random
import re

class ProgramDatabase:
    """
    Strategy archive with lineage tracking and inspiration sampling.

    Provides AlphaEvolve-style program database functionality:
    - Store ALL evolved strategies with full metadata (accepted AND rejected)
    - Track lineage (parent-child relationships via DB IDs)
    - Sample inspirations for LLM prompts using various strategies
    - Query by tags, metrics, eval context, and behavior descriptors
    """

    def __init__(self, backend: Optional[ProgramDatabaseBackend] = None):
        """
        Initialize the program database.

        Args:
            backend: Storage backend. Defaults to JsonFileBackend if not specified.
        """
        self.backend = backend or JsonFileBackend()
        self._primary_metric = 'ann_return'  # Configurable per context

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
        improvement_delta: float = 0.0
    ) -> str:
        """
        Register a strategy in the database (accepted or rejected).

        Returns the assigned strategy ID.
        """
        # Auto-compute eval_context_id
        eval_context_id = eval_context.context_id() if eval_context else ""

        # Extract next() method for richer prompts
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
            improvement_delta=improvement_delta,
            next_method_excerpt=next_excerpt,
            behavior_descriptor=behavior_descriptor
        )
        return self.backend.save(record)

    def _extract_next_method(self, code: str, max_lines: int = 80) -> str:
        """Extract the next() method from strategy code for LLM prompts."""
        # Find next() method
        match = re.search(r'def next\(self\):(.*?)(?=\n    def |\nclass |\Z)',
                         code, re.DOTALL)
        if match:
            next_code = 'def next(self):' + match.group(1)
            lines = next_code.split('\n')[:max_lines]
            return '\n'.join(lines)
        return ""

    def _compute_behavior_descriptor(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compute behavior descriptor for MAP-Elites style diversity.

        Bins trading behavior into discrete cells for diversity sampling.
        """
        descriptor = {}

        # Trade frequency bin (trades per year equivalent)
        trade_count = metrics.get('trade_count', 0)
        if trade_count < 10:
            descriptor['trade_freq_bin'] = 0  # Low frequency
        elif trade_count < 50:
            descriptor['trade_freq_bin'] = 1  # Medium
        else:
            descriptor['trade_freq_bin'] = 2  # High frequency

        # Risk bin (based on max drawdown)
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd < 0.1:
            descriptor['risk_bin'] = 0  # Low risk
        elif max_dd < 0.25:
            descriptor['risk_bin'] = 1  # Medium
        else:
            descriptor['risk_bin'] = 2  # High risk

        # Win rate bin
        win_rate = metrics.get('win_rate', 0.5)
        if win_rate < 0.4:
            descriptor['win_bin'] = 0  # Low win rate
        elif win_rate < 0.6:
            descriptor['win_bin'] = 1  # Medium
        else:
            descriptor['win_bin'] = 2  # High win rate

        return descriptor

    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Retrieve a strategy by ID."""
        return self.backend.load(strategy_id)

    def query_by_tags(self, tags: List[str]) -> List[StrategyRecord]:
        """Find strategies with any of the given tags."""
        return self.backend.query({'tags': tags})

    def get_lineage(self, strategy_id: str) -> List[StrategyRecord]:
        """
        Get the full ancestry of a strategy (parents, grandparents, etc).

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
        include_rejected: bool = False
    ) -> List[StrategyRecord]:
        """
        Sample strategies for LLM inspiration.

        Modes:
        - "exploitation": Top performers by primary metric
        - "exploration": Random from diverse tags/behavior cells
        - "trajectory": Recently improved strategies
        - "map_elites": Sample from different behavior descriptor cells (Phase 13D)
        - "cross_island": Sample across different tags/assets (Phase 13D)
        - "mixed": Weighted combination (default)

        Args:
            n: Number of strategies to sample
            mode: Sampling strategy
            exclude_ids: Strategy IDs to exclude (e.g., current parent's DB ID)
            eval_context_id: Only sample from same evaluation context
            include_rejected: Include rejected strategies (for negative examples)

        Returns:
            List of StrategyRecord objects for inspiration
        """
        exclude_ids = exclude_ids or []

        # Build base filters
        filters = {}
        if eval_context_id:
            filters['eval_context_id'] = eval_context_id
        if not include_rejected:
            filters['status'] = StrategyStatus.ACCEPTED

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
        else:  # mixed
            return self._sample_mixed(n, candidates)

    def _sample_top_performers(self, n: int, candidates: List[StrategyRecord]) -> List[StrategyRecord]:
        """Sample from top performing strategies."""
        sorted_candidates = sorted(
            candidates,
            key=lambda r: r.metrics.get(self._primary_metric, float('-inf')),
            reverse=True
        )
        return sorted_candidates[:n]

    def _sample_diverse(self, n: int, candidates: List[StrategyRecord]) -> List[StrategyRecord]:
        """Sample from different tag buckets for diversity."""
        # Group by primary tag
        tag_buckets: Dict[str, List[StrategyRecord]] = {}

        for record in candidates:
            primary_tag = record.tags[0] if record.tags else 'untagged'
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

    def _sample_improving(self, n: int, candidates: List[StrategyRecord]) -> List[StrategyRecord]:
        """Sample strategies that showed improvement."""
        improving = [c for c in candidates if c.improvement_delta > 0]

        # Sort by improvement delta
        sorted_candidates = sorted(
            improving,
            key=lambda r: r.improvement_delta,
            reverse=True
        )

        return sorted_candidates[:n]

    def _sample_map_elites(self, n: int, candidates: List[StrategyRecord]) -> List[StrategyRecord]:
        """
        Sample from different behavior descriptor cells (Phase 13D).

        This provides MAP-Elites style diversity by sampling the best
        strategy from different behavioral niches.
        """
        # Group by behavior cell (combination of descriptor bins)
        cells: Dict[tuple, StrategyRecord] = {}

        for record in candidates:
            bd = record.behavior_descriptor
            cell_key = (
                bd.get('trade_freq_bin', 0),
                bd.get('risk_bin', 0),
                bd.get('win_bin', 0)
            )

            # Keep elite (best performer) per cell
            if cell_key not in cells:
                cells[cell_key] = record
            elif record.metrics.get(self._primary_metric, 0) > cells[cell_key].metrics.get(self._primary_metric, 0):
                cells[cell_key] = record

        # Sample from different cells
        elites = list(cells.values())
        random.shuffle(elites)
        return elites[:n]

    def _sample_cross_island(self, n: int, candidates: List[StrategyRecord]) -> List[StrategyRecord]:
        """
        Sample across different "islands" (tags/assets) for cross-pollination (Phase 13D).
        """
        # Group by tag + asset combination
        islands: Dict[str, List[StrategyRecord]] = {}

        for record in candidates:
            tag = record.tags[0] if record.tags else 'untagged'
            asset = record.asset or 'default'
            island_key = f"{tag}:{asset}"

            if island_key not in islands:
                islands[island_key] = []
            islands[island_key].append(record)

        # Sample one from each island
        results = []
        for island_records in islands.values():
            if island_records and len(results) < n:
                # Pick best from this island
                best = max(island_records,
                          key=lambda r: r.metrics.get(self._primary_metric, 0))
                results.append(best)

        return results[:n]

    def _sample_mixed(self, n: int, candidates: List[StrategyRecord]) -> List[StrategyRecord]:
        """Mixed sampling: top performer + diverse + improving + map_elites."""
        results = []
        used_ids = set()

        # Allocate slots
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
            return {'count': 0}

        accepted = [r for r in all_records if r.status == StrategyStatus.ACCEPTED]
        returns = [r.metrics.get('ann_return', 0) for r in accepted]

        return {
            'count': len(all_records),
            'accepted': len(accepted),
            'rejected': len([r for r in all_records if r.status == StrategyStatus.REJECTED]),
            'failed': len([r for r in all_records if r.status in [StrategyStatus.COMPILE_FAILED, StrategyStatus.EVAL_FAILED]]),
            'avg_return': sum(returns) / len(returns) if returns else 0,
            'max_return': max(returns) if returns else 0,
            'min_return': min(returns) if returns else 0,
            'generations': max((r.generation for r in all_records), default=0),
            'unique_tags': len(set(tag for r in all_records for tag in r.tags)),
            'behavior_cells': len(set(
                tuple(r.behavior_descriptor.values()) for r in all_records if r.behavior_descriptor
            )),
        }
```

---

## LLM Interface Integration (Phase 13B)

### Enhanced Method: generate_improvement_with_inspirations()

Add to `src/profit/llm_interface.py`:

```python
def generate_improvement_with_inspirations(
    self,
    strategy_code: str,
    metrics_summary: str,
    inspirations: List[StrategyRecord],
    max_tokens_per_inspiration: int = 500
) -> str:
    """
    Generate improvement proposal using inspiration from other strategies.

    This implements AlphaEvolve's key insight: providing examples of
    successful strategies helps the LLM generate better mutations.

    Includes:
    - Multiple metrics (not just return) to avoid overfitting
    - Code excerpts of signal/entry/exit logic
    - Diversity across inspirations
    """
    # Build inspiration context with code excerpts and multi-metric info
    inspiration_text = ""
    if inspirations:
        inspiration_text = "\n\nHere are successful strategies for inspiration:\n"
        for i, insp in enumerate(inspirations, 1):
            # Multi-metric summary
            metrics = insp.metrics
            metrics_str = (
                f"Return={metrics.get('ann_return', 'N/A'):.1f}%, "
                f"Sharpe={metrics.get('sharpe', 'N/A'):.2f}, "
                f"MaxDD={metrics.get('max_drawdown', 'N/A'):.1f}%, "
                f"Trades={metrics.get('trade_count', 'N/A')}, "
                f"WinRate={metrics.get('win_rate', 'N/A'):.0%}"
            )

            # Code excerpt (next() method or truncated code)
            code_excerpt = insp.next_method_excerpt or insp.code[:max_tokens_per_inspiration]

            inspiration_text += f"""
--- Inspiration {i}: {insp.class_name} ---
Performance: {metrics_str}
Tags: {', '.join(insp.tags)}
Key innovation: {insp.mutation_text}

Signal/Entry/Exit logic:
```python
{code_excerpt}
```
"""

    prompt = f"""You are a trading strategy improvement coach.
Analyze the following strategy and suggest ONE specific improvement.

Current Strategy:
```python
{strategy_code}
```

Current Performance: {metrics_summary}
{inspiration_text}

Based on the current strategy and the successful approaches above,
suggest ONE concrete improvement. Consider:
1. Signal generation patterns from the inspirations
2. Entry/exit timing approaches
3. Position sizing or risk management techniques
4. Indicator combinations that worked well

IMPORTANT:
- Be specific and actionable
- Your suggestion should be implementable as a code change
- Consider multiple metrics, not just returns (Sharpe, drawdown, etc.)
- Aim for robustness, not just higher returns
"""

    return self._call_llm(prompt, role="analyst")
```

---

## Evolver Integration (FIXED: Correct ID Tracking)

### Modifications to `src/profit/evolver.py`

```python
class ProfitEvolver:
    def __init__(
        self,
        llm_client,
        initial_capital=10000,
        commission=0.002,
        exclusive_orders=True,
        finalize_trades=True,
        output_dir="evolved_strategies",
        program_db: Optional[ProgramDatabase] = None
    ):
        # ... existing init ...
        self.program_db = program_db
        self._strategy_db_ids: Dict[str, str] = {}  # class_name -> db_id mapping

    def evolve_strategy(
        self,
        strategy_class,
        train_data,
        val_data,
        max_iters=15,
        fold=1,
        use_inspirations=True,
        eval_context: Optional[EvaluationContext] = None
    ):
        """Evolve strategy with optional inspiration sampling."""

        # ... existing initialization ...

        # Register seed strategy in program DB
        if self.program_db:
            seed_code = inspect.getsource(strategy_class)
            seed_id = self.program_db.register_strategy(
                code=seed_code,
                class_name=strategy_class.__name__,
                parent_ids=[],
                mutation_text="Seed strategy",
                metrics={'ann_return': P0},
                tags=self._infer_tags(seed_code),
                status=StrategyStatus.SEED,
                generation=0,
                fold=fold,
                eval_context=eval_context
            )
            # CRITICAL: Track the DB ID for this class
            self._strategy_db_ids[strategy_class.__name__] = seed_id
            strategy_class._db_id = seed_id

        for gen in range(1, max_iters + 1):
            # ... existing parent selection ...

            # Get parent's DB ID (not class name!)
            parent_db_id = getattr(parent_class, '_db_id', None) or \
                           self._strategy_db_ids.get(parent_class.__name__)

            # Get inspirations if enabled and DB available
            inspirations = []
            if use_inspirations and self.program_db and parent_db_id:
                # FIXED: Exclude by DB ID, not class name
                inspirations = self.program_db.sample_inspirations(
                    n=3,
                    mode="mixed",
                    exclude_ids=[parent_db_id],
                    eval_context_id=eval_context.context_id() if eval_context else None
                )

            # Use inspiration-enhanced improvement generation
            if inspirations:
                improvement = self.llm.generate_improvement_with_inspirations(
                    parent_code,
                    f"AnnReturn={parent_perf:.2f}%, Sharpe={parent_sharpe:.2f}",
                    inspirations
                )
            else:
                improvement = self.llm.generate_improvement(
                    parent_code,
                    f"AnnReturn={parent_perf:.2f}%"
                )

            # ... existing code generation ...

            # Try to generate and evaluate
            try:
                new_code = self.llm.generate_strategy_code(parent_code, improvement)
                new_class = self._compile_strategy(new_code, new_class_name)
                P_new, new_metrics = self.run_backtest(new_class, val_data)
                eval_status = StrategyStatus.ACCEPTED if P_new >= MAS else StrategyStatus.REJECTED
            except SyntaxError:
                eval_status = StrategyStatus.COMPILE_FAILED
                P_new = 0
                new_metrics = {}
            except Exception as e:
                eval_status = StrategyStatus.EVAL_FAILED
                P_new = 0
                new_metrics = {}

            # CRITICAL: Register ALL strategies (accepted AND rejected/failed)
            if self.program_db:
                child_id = self.program_db.register_strategy(
                    code=new_code if 'new_code' in locals() else "",
                    class_name=new_class_name,
                    parent_ids=[parent_db_id] if parent_db_id else [],
                    mutation_text=improvement,
                    metrics=new_metrics,
                    tags=self._infer_tags(new_code) if 'new_code' in locals() else [],
                    status=eval_status,
                    generation=gen,
                    fold=fold,
                    eval_context=eval_context,
                    improvement_delta=P_new - parent_perf if eval_status == StrategyStatus.ACCEPTED else 0
                )

                # CRITICAL: Track DB ID for accepted strategies
                if eval_status == StrategyStatus.ACCEPTED:
                    self._strategy_db_ids[new_class_name] = child_id
                    new_class._db_id = child_id

            if eval_status == StrategyStatus.ACCEPTED:
                # ... existing acceptance logic (add to population, etc.) ...
                pass

    def _infer_tags(self, code: str) -> List[str]:
        """Infer strategy tags from code content."""
        tags = []
        code_lower = code.lower()

        if 'bollinger' in code_lower or 'mean' in code_lower:
            tags.append('mean-reversion')
        if 'ema' in code_lower or 'sma' in code_lower or 'crossover' in code_lower:
            tags.append('trend')
        if 'rsi' in code_lower or 'cci' in code_lower or 'williams' in code_lower:
            tags.append('oscillator')
        if 'macd' in code_lower:
            tags.append('momentum')
        if 'atr' in code_lower or 'stop' in code_lower:
            tags.append('risk-management')

        return tags if tags else ['unclassified']
```

---

## CLI Integration

Add to `src/profit/main.py`:

```python
# New arguments
parser.add_argument(
    '--db-backend',
    choices=['json', 'sqlite'],
    default='json',
    help='Program database backend (default: json)'
)
parser.add_argument(
    '--db-path',
    default='program_db',
    help='Path for program database (default: program_db)'
)
parser.add_argument(
    '--no-inspirations',
    action='store_true',
    help='Disable inspiration sampling from program database'
)

# In main():
if args.db_backend == 'sqlite':
    from profit.program_db import SqliteBackend, ProgramDatabase
    db_path = args.db_path if args.db_path.endswith('.sqlite') else f"{args.db_path}.sqlite"
    backend = SqliteBackend(db_path)
else:
    from profit.program_db import JsonFileBackend, ProgramDatabase
    backend = JsonFileBackend(args.db_path)

program_db = ProgramDatabase(backend)
evolver = ProfitEvolver(
    llm_client,
    program_db=program_db,
    # ... other args ...
)
```

---

## Migration Script (FIXED: No false parent links)

Script to migrate existing `StrategyPersister` output to ProgramDatabase:

```python
#!/usr/bin/env python3
"""Migrate existing evolved strategies to ProgramDatabase.

NOTE: Parent lineage is NOT preserved during migration because existing
metadata stores class names, not DB IDs. Migrated strategies are treated
as independent roots. Future strategies evolved from them will have proper
lineage tracking.
"""

import json
from pathlib import Path
from profit.program_db import ProgramDatabase, JsonFileBackend, StrategyStatus

def migrate_run(run_dir: Path, db: ProgramDatabase):
    """Migrate a single run directory."""
    for fold_dir in run_dir.glob("fold_*"):
        fold_num = int(fold_dir.name.split("_")[1])

        for json_file in fold_dir.glob("*.json"):
            if json_file.name == "best_strategy.json":
                continue

            with open(json_file) as f:
                meta = json.load(f)

            # Load corresponding .py file
            py_file = json_file.with_suffix(".py")
            if py_file.exists():
                code = py_file.read_text()
            else:
                continue

            # FIXED: Don't set parent_ids - migrated strategies are roots
            # Existing metadata has class names, not DB IDs
            db.register_strategy(
                code=code,
                class_name=meta.get('class_name', ''),
                parent_ids=[],  # Treat as root - no reliable parent link
                mutation_text=meta.get('improvement_proposal', ''),
                metrics=meta.get('metrics', {}),
                tags=[],  # Will be inferred
                status=StrategyStatus.ACCEPTED,  # Assume migrated = accepted
                generation=meta.get('generation', 0),
                fold=fold_num
            )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='evolved_strategies', help='Source directory')
    parser.add_argument('--dest', default='program_db', help='Destination database')
    args = parser.parse_args()

    db = ProgramDatabase(JsonFileBackend(args.dest))
    source = Path(args.source)

    for run_dir in source.glob("run_*"):
        print(f"Migrating {run_dir.name}...")
        migrate_run(run_dir, db)

    stats = db.get_stats()
    print(f"Migration complete. {stats['count']} strategies imported.")
    print(f"NOTE: Migrated strategies have no parent links (treated as roots).")

if __name__ == "__main__":
    main()
```

---

## File Structure

```
src/profit/
├── __init__.py
├── strategies.py
├── llm_interface.py      # Modified: add generate_improvement_with_inspirations()
├── evolver.py            # Modified: integrate ProgramDatabase with correct ID tracking
├── main.py               # Modified: add CLI arguments
└── program_db.py         # NEW: ProgramDatabase, backends, StrategyRecord, StrategyStatus

scripts/
└── migrate_to_program_db.py  # NEW: migration script
```

---

## Deliverables by Sub-Phase

### Phase 13A (Minimal Working DB)
- [ ] `StrategyStatus` enum
- [ ] `StrategyRecord` dataclass (basic fields)
- [ ] `ProgramDatabaseBackend` protocol
- [ ] `JsonFileBackend` implementation
- [ ] `ProgramDatabase` class with:
  - [ ] Strategy registration (ALL strategies, not just accepted)
  - [ ] Correct `_db_id` tracking on strategy classes
  - [ ] Basic query by tags/status
- [ ] Evolver integration with correct ID lineage
- [ ] Basic tests

### Phase 13B (Make It Useful)
- [ ] `EvaluationContext` dataclass
- [ ] `eval_context_id` filtering
- [ ] Multi-metric normalization (STANDARD_METRICS)
- [ ] `next_method_excerpt` extraction
- [ ] `generate_improvement_with_inspirations()` with code excerpts
- [ ] Behavior descriptor computation
- [ ] Tests for context filtering

### Phase 13C (Scale & Quality)
- [ ] Atomic writes for JSON backend
- [ ] Append-only index with compaction
- [ ] `SqliteBackend` implementation with proper parent join table
- [ ] Fixed double-load bug in SQLite query
- [ ] CLI arguments (`--db-backend`, `--db-path`, `--no-inspirations`)
- [ ] Migration script (roots only)
- [ ] Performance tests

### Phase 13D (Advanced Sampling)
- [ ] MAP-Elites style `_sample_map_elites()`
- [ ] Cross-island sampling `_sample_cross_island()`
- [ ] Behavior descriptor cells
- [ ] Multi-objective selection support
- [ ] Tests for sampling diversity

---

## Tests

```python
# tests/test_program_db.py

import pytest
import tempfile
import shutil
from pathlib import Path
from profit.program_db import (
    ProgramDatabase, JsonFileBackend, SqliteBackend,
    StrategyRecord, StrategyStatus, EvaluationContext
)

class TestJsonFileBackend:
    @pytest.fixture
    def backend(self):
        tmpdir = tempfile.mkdtemp()
        yield JsonFileBackend(tmpdir)
        shutil.rmtree(tmpdir)

    def test_save_and_load(self, backend):
        record = StrategyRecord(
            code="class Test: pass",
            class_name="Test",
            status=StrategyStatus.ACCEPTED,
            metrics={'ann_return': 15.0}
        )
        id = backend.save(record)
        loaded = backend.load(id)
        assert loaded.class_name == "Test"
        assert loaded.metrics['ann_return'] == 15.0

    def test_query_by_status(self, backend):
        backend.save(StrategyRecord(class_name="A", status=StrategyStatus.ACCEPTED))
        backend.save(StrategyRecord(class_name="B", status=StrategyStatus.REJECTED))

        accepted = backend.query({'status': StrategyStatus.ACCEPTED})
        assert len(accepted) == 1
        assert accepted[0].class_name == "A"

    def test_atomic_write_on_save(self, backend):
        # Verify files are written atomically (no corruption on interrupt)
        record = StrategyRecord(class_name="Test")
        backend.save(record)
        # File should exist and be valid JSON
        assert (backend.strategies_dir / f"{record.id}.json").exists()


class TestSqliteBackend:
    @pytest.fixture
    def backend(self):
        tmpfile = tempfile.mktemp(suffix='.sqlite')
        yield SqliteBackend(tmpfile)
        Path(tmpfile).unlink(missing_ok=True)

    def test_parent_join_table(self, backend):
        parent = StrategyRecord(class_name="Parent")
        parent_id = backend.save(parent)

        child = StrategyRecord(class_name="Child", parent_ids=[parent_id])
        child_id = backend.save(child)

        # Query by parent should work correctly
        children = backend.query({'parent_id': parent_id})
        assert len(children) == 1
        assert children[0].id == child_id

    def test_query_loads_once(self, backend):
        # Regression test: query should not double-load
        backend.save(StrategyRecord(class_name="Test"))
        results = backend.query({})
        assert len(results) == 1


class TestProgramDatabase:
    @pytest.fixture
    def db(self):
        tmpdir = tempfile.mkdtemp()
        yield ProgramDatabase(JsonFileBackend(tmpdir))
        shutil.rmtree(tmpdir)

    def test_registers_all_statuses(self, db):
        db.register_strategy(
            code="", class_name="Accepted", parent_ids=[], mutation_text="",
            metrics={}, tags=[], status=StrategyStatus.ACCEPTED
        )
        db.register_strategy(
            code="", class_name="Rejected", parent_ids=[], mutation_text="",
            metrics={}, tags=[], status=StrategyStatus.REJECTED
        )

        stats = db.get_stats()
        assert stats['accepted'] == 1
        assert stats['rejected'] == 1

    def test_sample_excludes_by_db_id(self, db):
        id1 = db.register_strategy(
            code="", class_name="A", parent_ids=[], mutation_text="",
            metrics={'ann_return': 10}, tags=[]
        )
        db.register_strategy(
            code="", class_name="B", parent_ids=[], mutation_text="",
            metrics={'ann_return': 20}, tags=[]
        )

        # Exclude by DB ID (not class name)
        results = db.sample_inspirations(n=10, exclude_ids=[id1])
        assert all(r.id != id1 for r in results)

    def test_eval_context_filtering(self, db):
        ctx1 = EvaluationContext(dataset_id="d1", train_start="2020-01-01")
        ctx2 = EvaluationContext(dataset_id="d2", train_start="2021-01-01")

        db.register_strategy(
            code="", class_name="A", parent_ids=[], mutation_text="",
            metrics={'ann_return': 10}, tags=[], eval_context=ctx1
        )
        db.register_strategy(
            code="", class_name="B", parent_ids=[], mutation_text="",
            metrics={'ann_return': 20}, tags=[], eval_context=ctx2
        )

        # Should only get strategies from same context
        results = db.sample_inspirations(n=10, eval_context_id=ctx1.context_id())
        assert len(results) == 1
        assert results[0].class_name == "A"

    def test_map_elites_sampling_diversity(self, db):
        # Create strategies in different behavior cells
        for trade_count, max_dd in [(5, -0.05), (30, -0.15), (100, -0.30)]:
            db.register_strategy(
                code="", class_name=f"S_{trade_count}", parent_ids=[], mutation_text="",
                metrics={'ann_return': 10, 'trade_count': trade_count, 'max_drawdown': max_dd},
                tags=[]
            )

        results = db.sample_inspirations(n=3, mode="map_elites")
        # Should get strategies from different cells
        assert len(results) == 3
        cells = [tuple(r.behavior_descriptor.values()) for r in results]
        assert len(set(cells)) == 3  # All different cells

    def test_lineage_tracking(self, db):
        id1 = db.register_strategy(
            code="", class_name="Gen0", parent_ids=[], mutation_text="",
            metrics={}, tags=[], generation=0
        )
        id2 = db.register_strategy(
            code="", class_name="Gen1", parent_ids=[id1], mutation_text="",
            metrics={}, tags=[], generation=1
        )
        id3 = db.register_strategy(
            code="", class_name="Gen2", parent_ids=[id2], mutation_text="",
            metrics={}, tags=[], generation=2
        )

        lineage = db.get_lineage(id3)
        assert len(lineage) == 2
        assert lineage[0].id == id1
        assert lineage[1].id == id2
```

---

## Summary of Fixes Applied

| Issue | Fix |
|-------|-----|
| ID/lineage bug (exclude by class name vs DB ID) | Track `_db_id` on classes, use `_strategy_db_ids` mapping, exclude by actual DB IDs |
| Only storing accepted strategies | Added `StrategyStatus` enum, register ALL strategies with status |
| Migration parent lineage wrong | Treat migrated strategies as roots (`parent_ids=[]`) |
| SQLite double-load in query | Load once: `records = [self.load(i) for i in ids]; return [r for r in records if r]` |
| No eval context for comparison | Added `EvaluationContext` dataclass and `eval_context_id` filtering |
| Hardcoded metric names | Added `STANDARD_METRICS` schema, configurable `_primary_metric` |
| Basic sampling only | Added `_sample_map_elites()` and `_sample_cross_island()` for diversity |
| Poor LLM prompt content | Include `next_method_excerpt`, multi-metric summary, code excerpts |
| JSON index not append-only | Added `_append_to_index()` for new entries, `_rebuild_index()` for updates |
| JSON writes not atomic | Added `_atomic_write()` using temp file + rename |
| SQLite parent_ids as JSON | Created `strategy_parents` join table with proper indexing |
