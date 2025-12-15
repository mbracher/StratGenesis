"""Unit tests for ProgramDatabase and related classes."""

import pytest
import shutil
import tempfile
from pathlib import Path

from profit.program_db import (
    ProgramDatabase,
    JsonFileBackend,
    SqliteBackend,
    StrategyRecord,
    StrategyStatus,
    EvaluationContext,
    STANDARD_METRICS,
)


class TestStrategyStatus:
    """Test StrategyStatus enum."""

    def test_status_values(self):
        """Should have expected status values."""
        assert StrategyStatus.ACCEPTED.value == "accepted"
        assert StrategyStatus.REJECTED.value == "rejected"
        assert StrategyStatus.COMPILE_FAILED.value == "compile_failed"
        assert StrategyStatus.EVAL_FAILED.value == "eval_failed"
        assert StrategyStatus.SEED.value == "seed"

    def test_status_is_string(self):
        """Should be usable as string."""
        status = StrategyStatus.ACCEPTED
        assert isinstance(status.value, str)


class TestStrategyRecord:
    """Test StrategyRecord dataclass."""

    def test_default_values(self):
        """Should initialize with default values."""
        record = StrategyRecord()

        assert record.id is not None
        assert len(record.id) == 8
        assert record.code == ""
        assert record.class_name == ""
        assert record.status == StrategyStatus.ACCEPTED
        assert record.parent_ids == []
        assert record.metrics == {}
        assert record.tags == []
        assert record.generation == 0

    def test_custom_values(self):
        """Should accept custom values."""
        record = StrategyRecord(
            code="class Test: pass",
            class_name="TestStrategy",
            status=StrategyStatus.REJECTED,
            parent_ids=["abc123"],
            metrics={"ann_return": 15.0, "sharpe": 1.2},
            tags=["trend", "momentum"],
            generation=5,
            fold=2,
        )

        assert record.code == "class Test: pass"
        assert record.class_name == "TestStrategy"
        assert record.status == StrategyStatus.REJECTED
        assert record.parent_ids == ["abc123"]
        assert record.metrics["ann_return"] == 15.0
        assert "trend" in record.tags
        assert record.generation == 5
        assert record.fold == 2


class TestJsonFileBackend:
    """Test JsonFileBackend implementation."""

    @pytest.fixture
    def backend(self):
        """Create a temporary backend for testing."""
        tmpdir = tempfile.mkdtemp()
        yield JsonFileBackend(tmpdir)
        shutil.rmtree(tmpdir)

    def test_save_and_load(self, backend):
        """Should save and load a strategy record."""
        record = StrategyRecord(
            code="class Test: pass",
            class_name="Test",
            status=StrategyStatus.ACCEPTED,
            metrics={"ann_return": 15.0},
        )
        strategy_id = backend.save(record)
        loaded = backend.load(strategy_id)

        assert loaded is not None
        assert loaded.class_name == "Test"
        assert loaded.metrics["ann_return"] == 15.0
        assert loaded.status == StrategyStatus.ACCEPTED

    def test_load_nonexistent(self, backend):
        """Should return None for nonexistent ID."""
        result = backend.load("nonexistent")
        assert result is None

    def test_query_by_status(self, backend):
        """Should query by status."""
        backend.save(StrategyRecord(class_name="A", status=StrategyStatus.ACCEPTED))
        backend.save(StrategyRecord(class_name="B", status=StrategyStatus.REJECTED))
        backend.save(StrategyRecord(class_name="C", status=StrategyStatus.ACCEPTED))

        accepted = backend.query({"status": StrategyStatus.ACCEPTED})
        assert len(accepted) == 2
        assert all(r.status == StrategyStatus.ACCEPTED for r in accepted)

    def test_query_by_tags(self, backend):
        """Should query by tags."""
        backend.save(StrategyRecord(class_name="A", tags=["trend"]))
        backend.save(StrategyRecord(class_name="B", tags=["momentum"]))
        backend.save(StrategyRecord(class_name="C", tags=["trend", "oscillator"]))

        trend = backend.query({"tags": ["trend"]})
        assert len(trend) == 2

    def test_query_by_generation(self, backend):
        """Should query by generation."""
        backend.save(StrategyRecord(class_name="A", generation=0))
        backend.save(StrategyRecord(class_name="B", generation=1))
        backend.save(StrategyRecord(class_name="C", generation=1))

        gen1 = backend.query({"generation": 1})
        assert len(gen1) == 2

    def test_query_by_parent_id(self, backend):
        """Should query by parent ID."""
        parent = StrategyRecord(class_name="Parent")
        parent_id = backend.save(parent)

        backend.save(StrategyRecord(class_name="Child1", parent_ids=[parent_id]))
        backend.save(StrategyRecord(class_name="Child2", parent_ids=[parent_id]))
        backend.save(StrategyRecord(class_name="Other", parent_ids=["other_id"]))

        children = backend.query({"parent_id": parent_id})
        assert len(children) == 2

    def test_list_all(self, backend):
        """Should list all strategy IDs."""
        backend.save(StrategyRecord(class_name="A"))
        backend.save(StrategyRecord(class_name="B"))

        ids = backend.list_all()
        assert len(ids) == 2

    def test_delete(self, backend):
        """Should delete a strategy."""
        record = StrategyRecord(class_name="ToDelete")
        strategy_id = backend.save(record)

        assert backend.load(strategy_id) is not None
        result = backend.delete(strategy_id)
        assert result is True
        assert backend.load(strategy_id) is None

    def test_delete_nonexistent(self, backend):
        """Should return False for nonexistent delete."""
        result = backend.delete("nonexistent")
        assert result is False

    def test_count(self, backend):
        """Should return correct count."""
        assert backend.count() == 0
        backend.save(StrategyRecord(class_name="A"))
        assert backend.count() == 1
        backend.save(StrategyRecord(class_name="B"))
        assert backend.count() == 2

    def test_get_children(self, backend):
        """Should return children of a strategy."""
        parent = StrategyRecord(class_name="Parent")
        parent_id = backend.save(parent)

        child1 = StrategyRecord(class_name="Child1", parent_ids=[parent_id])
        child1_id = backend.save(child1)

        children = backend.get_children(parent_id)
        assert child1_id in children

    def test_get_top_performers(self, backend):
        """Should return top performers by metric."""
        backend.save(
            StrategyRecord(
                class_name="A",
                metrics={"ann_return": 10.0},
                status=StrategyStatus.ACCEPTED,
            )
        )
        backend.save(
            StrategyRecord(
                class_name="B",
                metrics={"ann_return": 20.0},
                status=StrategyStatus.ACCEPTED,
            )
        )
        backend.save(
            StrategyRecord(
                class_name="C",
                metrics={"ann_return": 15.0},
                status=StrategyStatus.ACCEPTED,
            )
        )

        top = backend.get_top_performers(n=2, metric="ann_return")
        assert len(top) == 2

        # Load to verify order
        records = [backend.load(id) for id in top]
        assert records[0].metrics["ann_return"] == 20.0
        assert records[1].metrics["ann_return"] == 15.0


class TestProgramDatabase:
    """Test ProgramDatabase class."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        tmpdir = tempfile.mkdtemp()
        yield ProgramDatabase(JsonFileBackend(tmpdir))
        shutil.rmtree(tmpdir)

    def test_register_strategy(self, db):
        """Should register a strategy and return ID."""
        strategy_id = db.register_strategy(
            code="class Test: pass",
            class_name="TestStrategy",
            parent_ids=[],
            mutation_text="Initial strategy",
            metrics={"ann_return": 15.0},
            tags=["trend"],
        )

        assert strategy_id is not None
        assert len(strategy_id) == 8

    def test_get_strategy(self, db):
        """Should retrieve a registered strategy."""
        strategy_id = db.register_strategy(
            code="class Test: pass",
            class_name="TestStrategy",
            parent_ids=[],
            mutation_text="Initial strategy",
            metrics={"ann_return": 15.0},
            tags=["trend"],
        )

        record = db.get_strategy(strategy_id)
        assert record is not None
        assert record.class_name == "TestStrategy"
        assert record.metrics["ann_return"] == 15.0

    def test_registers_all_statuses(self, db):
        """Should register strategies with all statuses."""
        db.register_strategy(
            code="",
            class_name="Accepted",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
            status=StrategyStatus.ACCEPTED,
        )
        db.register_strategy(
            code="",
            class_name="Rejected",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
            status=StrategyStatus.REJECTED,
        )
        db.register_strategy(
            code="",
            class_name="Failed",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
            status=StrategyStatus.COMPILE_FAILED,
        )

        stats = db.get_stats()
        assert stats["accepted"] == 1
        assert stats["rejected"] == 1
        assert stats["failed"] == 1

    def test_query_by_tags(self, db):
        """Should query by tags."""
        db.register_strategy(
            code="",
            class_name="A",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=["trend"],
        )
        db.register_strategy(
            code="",
            class_name="B",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=["momentum"],
        )

        trend = db.query_by_tags(["trend"])
        assert len(trend) == 1
        assert trend[0].class_name == "A"

    def test_query_by_status(self, db):
        """Should query by status."""
        db.register_strategy(
            code="",
            class_name="A",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
            status=StrategyStatus.ACCEPTED,
        )
        db.register_strategy(
            code="",
            class_name="B",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
            status=StrategyStatus.REJECTED,
        )

        accepted = db.query_by_status(StrategyStatus.ACCEPTED)
        assert len(accepted) == 1
        assert accepted[0].class_name == "A"

    def test_lineage_tracking(self, db):
        """Should track lineage through DB IDs."""
        id1 = db.register_strategy(
            code="",
            class_name="Gen0",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
            generation=0,
        )
        id2 = db.register_strategy(
            code="",
            class_name="Gen1",
            parent_ids=[id1],
            mutation_text="",
            metrics={},
            tags=[],
            generation=1,
        )
        id3 = db.register_strategy(
            code="",
            class_name="Gen2",
            parent_ids=[id2],
            mutation_text="",
            metrics={},
            tags=[],
            generation=2,
        )

        lineage = db.get_lineage(id3)
        assert len(lineage) == 2
        assert lineage[0].id == id1
        assert lineage[1].id == id2

    def test_sample_excludes_by_db_id(self, db):
        """Should exclude by DB ID when sampling."""
        id1 = db.register_strategy(
            code="",
            class_name="A",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 10},
            tags=[],
        )
        db.register_strategy(
            code="",
            class_name="B",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 20},
            tags=[],
        )

        # Exclude by DB ID
        results = db.sample_inspirations(n=10, exclude_ids=[id1])
        assert all(r.id != id1 for r in results)

    def test_sample_exploitation_mode(self, db):
        """Should sample top performers in exploitation mode."""
        db.register_strategy(
            code="",
            class_name="Low",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 5},
            tags=[],
        )
        db.register_strategy(
            code="",
            class_name="High",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 20},
            tags=[],
        )
        db.register_strategy(
            code="",
            class_name="Medium",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 10},
            tags=[],
        )

        results = db.sample_inspirations(n=2, mode="exploitation")
        assert len(results) == 2
        assert results[0].metrics["ann_return"] == 20

    def test_sample_trajectory_mode(self, db):
        """Should sample improving strategies in trajectory mode."""
        db.register_strategy(
            code="",
            class_name="NoImprovement",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 10},
            tags=[],
            improvement_delta=0,
        )
        db.register_strategy(
            code="",
            class_name="BigImprovement",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 20},
            tags=[],
            improvement_delta=15,
        )
        db.register_strategy(
            code="",
            class_name="SmallImprovement",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 15},
            tags=[],
            improvement_delta=5,
        )

        results = db.sample_inspirations(n=2, mode="trajectory")
        assert len(results) == 2
        # Should be ordered by improvement_delta
        assert results[0].improvement_delta == 15

    def test_sample_exploration_mode(self, db):
        """Should sample from diverse tags in exploration mode."""
        db.register_strategy(
            code="",
            class_name="Trend1",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 10},
            tags=["trend"],
        )
        db.register_strategy(
            code="",
            class_name="Trend2",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 15},
            tags=["trend"],
        )
        db.register_strategy(
            code="",
            class_name="Momentum1",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 12},
            tags=["momentum"],
        )

        # Should try to get from different tags
        results = db.sample_inspirations(n=2, mode="exploration")
        assert len(results) == 2

    def test_sample_mixed_mode(self, db):
        """Should use mixed sampling by default."""
        for i in range(5):
            db.register_strategy(
                code="",
                class_name=f"Strategy{i}",
                parent_ids=[],
                mutation_text="",
                metrics={"ann_return": i * 5},
                tags=["trend"] if i % 2 == 0 else ["momentum"],
                improvement_delta=i,
            )

        results = db.sample_inspirations(n=3, mode="mixed")
        assert len(results) == 3

    def test_sample_excludes_rejected_by_default(self, db):
        """Should exclude rejected strategies by default."""
        db.register_strategy(
            code="",
            class_name="Accepted",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 10},
            tags=[],
            status=StrategyStatus.ACCEPTED,
        )
        db.register_strategy(
            code="",
            class_name="Rejected",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 20},
            tags=[],
            status=StrategyStatus.REJECTED,
        )

        results = db.sample_inspirations(n=10)
        assert len(results) == 1
        assert results[0].status == StrategyStatus.ACCEPTED

    def test_sample_includes_rejected_when_requested(self, db):
        """Should include rejected strategies when requested."""
        db.register_strategy(
            code="",
            class_name="Accepted",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 10},
            tags=[],
            status=StrategyStatus.ACCEPTED,
        )
        db.register_strategy(
            code="",
            class_name="Rejected",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 20},
            tags=[],
            status=StrategyStatus.REJECTED,
        )

        results = db.sample_inspirations(n=10, include_rejected=True)
        assert len(results) == 2

    def test_behavior_descriptor_computed(self, db):
        """Should compute behavior descriptor from metrics."""
        strategy_id = db.register_strategy(
            code="",
            class_name="Test",
            parent_ids=[],
            mutation_text="",
            metrics={
                "ann_return": 15,
                "trade_count": 30,
                "max_drawdown": -0.15,
                "win_rate": 0.55,
            },
            tags=[],
        )

        record = db.get_strategy(strategy_id)
        assert "trade_freq_bin" in record.behavior_descriptor
        assert "risk_bin" in record.behavior_descriptor
        assert "win_bin" in record.behavior_descriptor

    def test_get_stats(self, db):
        """Should return database statistics."""
        db.register_strategy(
            code="",
            class_name="A",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 10},
            tags=["trend"],
            status=StrategyStatus.ACCEPTED,
        )
        db.register_strategy(
            code="",
            class_name="B",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 20},
            tags=["momentum"],
            status=StrategyStatus.ACCEPTED,
        )
        db.register_strategy(
            code="",
            class_name="C",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
            status=StrategyStatus.REJECTED,
        )

        stats = db.get_stats()
        assert stats["count"] == 3
        assert stats["accepted"] == 2
        assert stats["rejected"] == 1
        assert stats["avg_return"] == 15.0
        assert stats["max_return"] == 20.0
        assert stats["min_return"] == 10.0
        assert stats["unique_tags"] == 2

    def test_get_stats_empty_db(self, db):
        """Should handle empty database."""
        stats = db.get_stats()
        assert stats["count"] == 0


# Phase 13B Tests


class TestEvaluationContext:
    """Test EvaluationContext dataclass (Phase 13B)."""

    def test_default_values(self):
        """Should initialize with default values."""
        ctx = EvaluationContext()

        assert ctx.dataset_id == ""
        assert ctx.initial_capital == 10000.0
        assert ctx.commission == 0.002
        assert ctx.backtest_engine == "backtesting.py"

    def test_custom_values(self):
        """Should accept custom values."""
        ctx = EvaluationContext(
            dataset_id="SPY_2020_2023",
            dataset_source="yahoo_finance",
            timeframe="1D",
            train_start="2020-01-01",
            train_end="2022-06-30",
            val_start="2022-07-10",
            val_end="2022-12-31",
            initial_capital=50000.0,
            commission=0.001,
        )

        assert ctx.dataset_id == "SPY_2020_2023"
        assert ctx.dataset_source == "yahoo_finance"
        assert ctx.timeframe == "1D"
        assert ctx.initial_capital == 50000.0

    def test_context_id_generation(self):
        """Should generate consistent context ID hash."""
        ctx1 = EvaluationContext(
            dataset_id="SPY",
            timeframe="1D",
            train_start="2020-01-01",
            val_end="2022-12-31",
            commission=0.002,
        )
        ctx2 = EvaluationContext(
            dataset_id="SPY",
            timeframe="1D",
            train_start="2020-01-01",
            val_end="2022-12-31",
            commission=0.002,
        )
        ctx3 = EvaluationContext(
            dataset_id="AAPL",  # Different dataset
            timeframe="1D",
            train_start="2020-01-01",
            val_end="2022-12-31",
            commission=0.002,
        )

        # Same context should generate same ID
        assert ctx1.context_id() == ctx2.context_id()
        # Different context should generate different ID
        assert ctx1.context_id() != ctx3.context_id()
        # ID should be 12 characters
        assert len(ctx1.context_id()) == 12


class TestStandardMetrics:
    """Test STANDARD_METRICS schema (Phase 13B)."""

    def test_standard_metrics_defined(self):
        """Should have expected standard metrics."""
        assert "ann_return" in STANDARD_METRICS
        assert "sharpe" in STANDARD_METRICS
        assert "max_drawdown" in STANDARD_METRICS
        assert "trade_count" in STANDARD_METRICS
        assert "win_rate" in STANDARD_METRICS

    def test_standard_metrics_has_descriptions(self):
        """All metrics should have string descriptions."""
        for metric, description in STANDARD_METRICS.items():
            assert isinstance(metric, str)
            assert isinstance(description, str)
            assert len(description) > 0


class TestEvalContextFiltering:
    """Test eval_context_id filtering (Phase 13B)."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        tmpdir = tempfile.mkdtemp()
        yield ProgramDatabase(JsonFileBackend(tmpdir))
        shutil.rmtree(tmpdir)

    def test_register_with_eval_context(self, db):
        """Should register strategy with eval context."""
        ctx = EvaluationContext(
            dataset_id="SPY",
            timeframe="1D",
            train_start="2020-01-01",
            val_end="2022-12-31",
        )

        strategy_id = db.register_strategy(
            code="class Test: pass",
            class_name="TestStrategy",
            parent_ids=[],
            mutation_text="Test",
            metrics={"ann_return": 15.0},
            tags=["trend"],
            eval_context=ctx,
        )

        record = db.get_strategy(strategy_id)
        assert record is not None
        assert record.eval_context is not None
        assert record.eval_context.dataset_id == "SPY"
        assert record.eval_context_id == ctx.context_id()

    def test_query_by_eval_context_id(self, db):
        """Should filter strategies by eval_context_id."""
        ctx1 = EvaluationContext(dataset_id="SPY", train_start="2020-01-01")
        ctx2 = EvaluationContext(dataset_id="AAPL", train_start="2020-01-01")

        db.register_strategy(
            code="",
            class_name="SPY_Strategy",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 10},
            tags=[],
            eval_context=ctx1,
        )
        db.register_strategy(
            code="",
            class_name="AAPL_Strategy",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 20},
            tags=[],
            eval_context=ctx2,
        )

        # Should only get SPY strategy
        results = db.sample_inspirations(n=10, eval_context_id=ctx1.context_id())
        assert len(results) == 1
        assert results[0].class_name == "SPY_Strategy"

    def test_sample_inspirations_respects_context(self, db):
        """Sample inspirations should respect eval_context_id filter."""
        ctx1 = EvaluationContext(dataset_id="SPY", train_start="2020-01-01")
        ctx2 = EvaluationContext(dataset_id="AAPL", train_start="2020-01-01")

        # Register strategies in different contexts with improvement_delta for trajectory sampling
        for i in range(3):
            db.register_strategy(
                code="",
                class_name=f"SPY_Strategy_{i}",
                parent_ids=[],
                mutation_text="",
                metrics={"ann_return": 10 + i},
                tags=["trend"],
                eval_context=ctx1,
                improvement_delta=float(i + 1),  # For trajectory mode
            )

        for i in range(2):
            db.register_strategy(
                code="",
                class_name=f"AAPL_Strategy_{i}",
                parent_ids=[],
                mutation_text="",
                metrics={"ann_return": 20 + i},
                tags=["trend"],
                eval_context=ctx2,
                improvement_delta=float(i + 1),
            )

        # Sample from ctx1 only - should only return SPY strategies
        results = db.sample_inspirations(n=5, eval_context_id=ctx1.context_id())
        assert len(results) <= 3  # Can't get more than available
        assert len(results) >= 1  # Should get at least one
        assert all("SPY" in r.class_name for r in results)

        # Sample from ctx2 only - should only return AAPL strategies
        results = db.sample_inspirations(n=5, eval_context_id=ctx2.context_id())
        assert len(results) <= 2  # Can't get more than available
        assert len(results) >= 1  # Should get at least one
        assert all("AAPL" in r.class_name for r in results)


class TestNextMethodExtraction:
    """Test _extract_next_method functionality (Phase 13B)."""

    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        tmpdir = tempfile.mkdtemp()
        yield ProgramDatabase(JsonFileBackend(tmpdir))
        shutil.rmtree(tmpdir)

    def test_extracts_next_method(self, db):
        """Should extract next() method from strategy code."""
        code = '''class EMACrossover(Strategy):
    def init(self):
        self.ema_short = self.I(EMA, self.data.Close, 50)
        self.ema_long = self.I(EMA, self.data.Close, 200)

    def next(self):
        if crossover(self.ema_short, self.ema_long):
            self.buy()
        elif crossover(self.ema_long, self.ema_short):
            self.sell()
'''

        strategy_id = db.register_strategy(
            code=code,
            class_name="EMACrossover",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
        )

        record = db.get_strategy(strategy_id)
        assert record.next_method_excerpt != ""
        assert "def next(self):" in record.next_method_excerpt
        assert "crossover" in record.next_method_excerpt
        # Should not include init method
        assert "def init(self):" not in record.next_method_excerpt

    def test_handles_code_without_next_method(self, db):
        """Should handle code without next() method gracefully."""
        code = '''class NoNext(Strategy):
    def init(self):
        pass
'''

        strategy_id = db.register_strategy(
            code=code,
            class_name="NoNext",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
        )

        record = db.get_strategy(strategy_id)
        assert record.next_method_excerpt == ""

    def test_truncates_long_next_method(self, db):
        """Should truncate very long next() methods."""
        # Create a next() method with many lines
        lines = ["        pass"] * 100
        code = f'''class LongNext(Strategy):
    def init(self):
        pass

    def next(self):
{chr(10).join(lines)}
'''

        strategy_id = db.register_strategy(
            code=code,
            class_name="LongNext",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
        )

        record = db.get_strategy(strategy_id)
        # Should have excerpt, but limited to max_lines (default 80)
        excerpt_lines = record.next_method_excerpt.split("\n")
        assert len(excerpt_lines) <= 80


class TestStrategyRecordPhase13BFields:
    """Test new StrategyRecord fields added in Phase 13B."""

    def test_new_fields_have_defaults(self):
        """New Phase 13B fields should have sensible defaults."""
        record = StrategyRecord()

        assert record.eval_context is None
        assert record.eval_context_id == ""
        assert record.next_method_excerpt == ""
        assert record.diff_from_parent == ""

    def test_can_set_new_fields(self):
        """Should be able to set Phase 13B fields."""
        ctx = EvaluationContext(dataset_id="TEST")
        record = StrategyRecord(
            eval_context=ctx,
            eval_context_id="abc123",
            next_method_excerpt="def next(self): pass",
            diff_from_parent="- old\n+ new",
        )

        assert record.eval_context == ctx
        assert record.eval_context_id == "abc123"
        assert record.next_method_excerpt == "def next(self): pass"
        assert record.diff_from_parent == "- old\n+ new"


class TestJsonFileBackendPhase13B:
    """Test JsonFileBackend Phase 13B features."""

    @pytest.fixture
    def backend(self):
        """Create a temporary backend for testing."""
        tmpdir = tempfile.mkdtemp()
        yield JsonFileBackend(tmpdir)
        shutil.rmtree(tmpdir)

    def test_saves_and_loads_eval_context(self, backend):
        """Should correctly save and load EvaluationContext."""
        ctx = EvaluationContext(
            dataset_id="SPY",
            dataset_source="yahoo_finance",
            timeframe="1D",
            train_start="2020-01-01",
            train_end="2022-06-30",
            initial_capital=50000.0,
        )

        record = StrategyRecord(
            code="class Test: pass",
            class_name="Test",
            eval_context=ctx,
            eval_context_id=ctx.context_id(),
        )

        strategy_id = backend.save(record)
        loaded = backend.load(strategy_id)

        assert loaded is not None
        assert loaded.eval_context is not None
        assert loaded.eval_context.dataset_id == "SPY"
        assert loaded.eval_context.dataset_source == "yahoo_finance"
        assert loaded.eval_context.initial_capital == 50000.0
        assert loaded.eval_context_id == ctx.context_id()

    def test_query_filters_by_eval_context_id(self, backend):
        """Should filter by eval_context_id in queries."""
        ctx1_id = "ctx1_12char"
        ctx2_id = "ctx2_12char"

        backend.save(StrategyRecord(class_name="A", eval_context_id=ctx1_id))
        backend.save(StrategyRecord(class_name="B", eval_context_id=ctx1_id))
        backend.save(StrategyRecord(class_name="C", eval_context_id=ctx2_id))

        ctx1_results = backend.query({"eval_context_id": ctx1_id})
        assert len(ctx1_results) == 2
        assert all(r.class_name in ["A", "B"] for r in ctx1_results)

        ctx2_results = backend.query({"eval_context_id": ctx2_id})
        assert len(ctx2_results) == 1
        assert ctx2_results[0].class_name == "C"

    def test_get_top_performers_with_context(self, backend):
        """get_top_performers should support eval_context_id filter."""
        ctx1_id = "ctx1_12char"
        ctx2_id = "ctx2_12char"

        backend.save(
            StrategyRecord(
                class_name="A",
                metrics={"ann_return": 10},
                eval_context_id=ctx1_id,
                status=StrategyStatus.ACCEPTED,
            )
        )
        backend.save(
            StrategyRecord(
                class_name="B",
                metrics={"ann_return": 20},
                eval_context_id=ctx1_id,
                status=StrategyStatus.ACCEPTED,
            )
        )
        backend.save(
            StrategyRecord(
                class_name="C",
                metrics={"ann_return": 30},
                eval_context_id=ctx2_id,
                status=StrategyStatus.ACCEPTED,
            )
        )

        # Without context filter - should get C (highest return)
        top = backend.get_top_performers(n=1)
        assert len(top) == 1
        record = backend.load(top[0])
        assert record.class_name == "C"

        # With context filter - should get B (highest in ctx1)
        top_ctx1 = backend.get_top_performers(n=1, eval_context_id=ctx1_id)
        assert len(top_ctx1) == 1
        record = backend.load(top_ctx1[0])
        assert record.class_name == "B"


# Phase 13C Tests


class TestSqliteBackend:
    """Test SqliteBackend implementation (Phase 13C)."""

    @pytest.fixture
    def backend(self):
        """Create a temporary SQLite backend for testing."""
        tmpfile = tempfile.mktemp(suffix=".sqlite")
        yield SqliteBackend(tmpfile)
        Path(tmpfile).unlink(missing_ok=True)

    def test_save_and_load(self, backend):
        """Should save and load a strategy record."""
        record = StrategyRecord(
            code="class Test: pass",
            class_name="Test",
            status=StrategyStatus.ACCEPTED,
            metrics={"ann_return": 15.0, "sharpe": 1.2},
        )
        strategy_id = backend.save(record)
        loaded = backend.load(strategy_id)

        assert loaded is not None
        assert loaded.class_name == "Test"
        assert loaded.metrics["ann_return"] == 15.0
        assert loaded.metrics["sharpe"] == 1.2
        assert loaded.status == StrategyStatus.ACCEPTED

    def test_load_nonexistent(self, backend):
        """Should return None for nonexistent ID."""
        result = backend.load("nonexistent")
        assert result is None

    def test_parent_join_table(self, backend):
        """Should store parent relationships in proper join table."""
        parent = StrategyRecord(class_name="Parent")
        parent_id = backend.save(parent)

        child = StrategyRecord(class_name="Child", parent_ids=[parent_id])
        child_id = backend.save(child)

        # Query by parent should work correctly
        children = backend.query({"parent_id": parent_id})
        assert len(children) == 1
        assert children[0].id == child_id

        # Loaded child should have parent_ids
        loaded = backend.load(child_id)
        assert parent_id in loaded.parent_ids

    def test_multiple_parents(self, backend):
        """Should handle strategies with multiple parents."""
        parent1 = StrategyRecord(class_name="Parent1")
        parent1_id = backend.save(parent1)

        parent2 = StrategyRecord(class_name="Parent2")
        parent2_id = backend.save(parent2)

        child = StrategyRecord(class_name="Child", parent_ids=[parent1_id, parent2_id])
        child_id = backend.save(child)

        loaded = backend.load(child_id)
        assert parent1_id in loaded.parent_ids
        assert parent2_id in loaded.parent_ids

    def test_query_by_status(self, backend):
        """Should query by status."""
        backend.save(StrategyRecord(class_name="A", status=StrategyStatus.ACCEPTED))
        backend.save(StrategyRecord(class_name="B", status=StrategyStatus.REJECTED))
        backend.save(StrategyRecord(class_name="C", status=StrategyStatus.ACCEPTED))

        accepted = backend.query({"status": StrategyStatus.ACCEPTED})
        assert len(accepted) == 2
        assert all(r.status == StrategyStatus.ACCEPTED for r in accepted)

    def test_query_by_tags(self, backend):
        """Should query by tags."""
        backend.save(StrategyRecord(class_name="A", tags=["trend"]))
        backend.save(StrategyRecord(class_name="B", tags=["momentum"]))
        backend.save(StrategyRecord(class_name="C", tags=["trend", "oscillator"]))

        trend = backend.query({"tags": ["trend"]})
        assert len(trend) == 2

    def test_query_by_generation(self, backend):
        """Should query by generation."""
        backend.save(StrategyRecord(class_name="A", generation=0))
        backend.save(StrategyRecord(class_name="B", generation=1))
        backend.save(StrategyRecord(class_name="C", generation=1))

        gen1 = backend.query({"generation": 1})
        assert len(gen1) == 2

    def test_query_by_eval_context_id(self, backend):
        """Should query by eval_context_id."""
        backend.save(StrategyRecord(class_name="A", eval_context_id="ctx1"))
        backend.save(StrategyRecord(class_name="B", eval_context_id="ctx1"))
        backend.save(StrategyRecord(class_name="C", eval_context_id="ctx2"))

        ctx1 = backend.query({"eval_context_id": "ctx1"})
        assert len(ctx1) == 2

    def test_query_loads_once_not_twice(self, backend):
        """Regression test: query should not double-load strategies."""
        backend.save(StrategyRecord(class_name="Test"))
        results = backend.query({})
        assert len(results) == 1

    def test_list_all(self, backend):
        """Should list all strategy IDs."""
        backend.save(StrategyRecord(class_name="A"))
        backend.save(StrategyRecord(class_name="B"))

        ids = backend.list_all()
        assert len(ids) == 2

    def test_delete(self, backend):
        """Should delete a strategy and all related records."""
        record = StrategyRecord(
            class_name="ToDelete",
            tags=["test"],
            metrics={"ann_return": 10},
            behavior_descriptor={"risk_bin": 1},
        )
        strategy_id = backend.save(record)

        assert backend.load(strategy_id) is not None
        result = backend.delete(strategy_id)
        assert result is True
        assert backend.load(strategy_id) is None

    def test_delete_nonexistent(self, backend):
        """Should return False for nonexistent delete."""
        result = backend.delete("nonexistent")
        assert result is False

    def test_count(self, backend):
        """Should return correct count."""
        assert backend.count() == 0
        backend.save(StrategyRecord(class_name="A"))
        assert backend.count() == 1
        backend.save(StrategyRecord(class_name="B"))
        assert backend.count() == 2

    def test_get_children(self, backend):
        """Should return children of a strategy."""
        parent = StrategyRecord(class_name="Parent")
        parent_id = backend.save(parent)

        child1 = StrategyRecord(class_name="Child1", parent_ids=[parent_id])
        child1_id = backend.save(child1)

        child2 = StrategyRecord(class_name="Child2", parent_ids=[parent_id])
        child2_id = backend.save(child2)

        children = backend.get_children(parent_id)
        assert child1_id in children
        assert child2_id in children

    def test_get_top_performers(self, backend):
        """Should return top performers by metric."""
        backend.save(
            StrategyRecord(
                class_name="A",
                metrics={"ann_return": 10.0},
                status=StrategyStatus.ACCEPTED,
            )
        )
        backend.save(
            StrategyRecord(
                class_name="B",
                metrics={"ann_return": 20.0},
                status=StrategyStatus.ACCEPTED,
            )
        )
        backend.save(
            StrategyRecord(
                class_name="C",
                metrics={"ann_return": 15.0},
                status=StrategyStatus.ACCEPTED,
            )
        )

        top = backend.get_top_performers(n=2, metric="ann_return")
        assert len(top) == 2

        # Verify order
        records = [backend.load(id) for id in top]
        assert records[0].metrics["ann_return"] == 20.0
        assert records[1].metrics["ann_return"] == 15.0

    def test_saves_and_loads_eval_context(self, backend):
        """Should correctly save and load EvaluationContext."""
        ctx = EvaluationContext(
            dataset_id="SPY",
            dataset_source="yahoo_finance",
            timeframe="1D",
            train_start="2020-01-01",
            train_end="2022-06-30",
            initial_capital=50000.0,
        )

        record = StrategyRecord(
            code="class Test: pass",
            class_name="Test",
            eval_context=ctx,
            eval_context_id=ctx.context_id(),
        )

        strategy_id = backend.save(record)
        loaded = backend.load(strategy_id)

        assert loaded is not None
        assert loaded.eval_context is not None
        assert loaded.eval_context.dataset_id == "SPY"
        assert loaded.eval_context.dataset_source == "yahoo_finance"
        assert loaded.eval_context.initial_capital == 50000.0
        assert loaded.eval_context_id == ctx.context_id()

    def test_behavior_descriptors(self, backend):
        """Should save and load behavior descriptors."""
        record = StrategyRecord(
            class_name="Test",
            behavior_descriptor={"trade_freq_bin": 1, "risk_bin": 2, "win_bin": 0},
        )

        strategy_id = backend.save(record)
        loaded = backend.load(strategy_id)

        assert loaded.behavior_descriptor["trade_freq_bin"] == 1
        assert loaded.behavior_descriptor["risk_bin"] == 2
        assert loaded.behavior_descriptor["win_bin"] == 0

    def test_update_existing(self, backend):
        """Should update an existing strategy (INSERT OR REPLACE)."""
        record = StrategyRecord(
            code="class Test: pass",
            class_name="Test",
            metrics={"ann_return": 10.0},
        )
        strategy_id = backend.save(record)

        # Update the record
        record.metrics = {"ann_return": 20.0}
        record.tags = ["updated"]
        backend.save(record)

        # Verify update
        loaded = backend.load(strategy_id)
        assert loaded.metrics["ann_return"] == 20.0
        assert "updated" in loaded.tags


class TestProgramDatabaseWithSqliteBackend:
    """Test ProgramDatabase with SqliteBackend (Phase 13C)."""

    @pytest.fixture
    def db(self):
        """Create a temporary database with SQLite backend."""
        tmpfile = tempfile.mktemp(suffix=".sqlite")
        yield ProgramDatabase(SqliteBackend(tmpfile))
        Path(tmpfile).unlink(missing_ok=True)

    def test_register_and_get_strategy(self, db):
        """Should register and retrieve a strategy."""
        strategy_id = db.register_strategy(
            code="class Test: pass",
            class_name="TestStrategy",
            parent_ids=[],
            mutation_text="Initial strategy",
            metrics={"ann_return": 15.0},
            tags=["trend"],
        )

        record = db.get_strategy(strategy_id)
        assert record is not None
        assert record.class_name == "TestStrategy"
        assert record.metrics["ann_return"] == 15.0

    def test_lineage_tracking(self, db):
        """Should track lineage through DB IDs with SQLite backend."""
        id1 = db.register_strategy(
            code="",
            class_name="Gen0",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
            generation=0,
        )
        id2 = db.register_strategy(
            code="",
            class_name="Gen1",
            parent_ids=[id1],
            mutation_text="",
            metrics={},
            tags=[],
            generation=1,
        )
        id3 = db.register_strategy(
            code="",
            class_name="Gen2",
            parent_ids=[id2],
            mutation_text="",
            metrics={},
            tags=[],
            generation=2,
        )

        lineage = db.get_lineage(id3)
        assert len(lineage) == 2
        assert lineage[0].id == id1
        assert lineage[1].id == id2

    def test_sample_inspirations_with_sqlite(self, db):
        """Should sample inspirations correctly with SQLite backend."""
        for i in range(5):
            db.register_strategy(
                code="",
                class_name=f"Strategy{i}",
                parent_ids=[],
                mutation_text="",
                metrics={"ann_return": i * 5},
                tags=["trend"] if i % 2 == 0 else ["momentum"],
                improvement_delta=float(i),
            )

        results = db.sample_inspirations(n=3, mode="mixed")
        assert len(results) == 3

    def test_get_stats(self, db):
        """Should return correct stats with SQLite backend."""
        db.register_strategy(
            code="",
            class_name="A",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 10},
            tags=["trend"],
            status=StrategyStatus.ACCEPTED,
        )
        db.register_strategy(
            code="",
            class_name="B",
            parent_ids=[],
            mutation_text="",
            metrics={"ann_return": 20},
            tags=["momentum"],
            status=StrategyStatus.ACCEPTED,
        )
        db.register_strategy(
            code="",
            class_name="C",
            parent_ids=[],
            mutation_text="",
            metrics={},
            tags=[],
            status=StrategyStatus.REJECTED,
        )

        stats = db.get_stats()
        assert stats["count"] == 3
        assert stats["accepted"] == 2
        assert stats["rejected"] == 1


class TestBackendComparison:
    """Test that both backends produce equivalent results (Phase 13C)."""

    @pytest.fixture
    def json_backend(self):
        """Create JSON backend."""
        tmpdir = tempfile.mkdtemp()
        yield JsonFileBackend(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sqlite_backend(self):
        """Create SQLite backend."""
        tmpfile = tempfile.mktemp(suffix=".sqlite")
        yield SqliteBackend(tmpfile)
        Path(tmpfile).unlink(missing_ok=True)

    def test_equivalent_save_load(self, json_backend, sqlite_backend):
        """Both backends should produce equivalent save/load behavior."""
        record = StrategyRecord(
            code="class Test: pass",
            class_name="Test",
            status=StrategyStatus.ACCEPTED,
            metrics={"ann_return": 15.0, "sharpe": 1.2},
            tags=["trend", "momentum"],
            parent_ids=[],
            generation=3,
            fold=2,
            improvement_delta=5.0,
        )

        json_id = json_backend.save(record)
        # Create new record with same ID for sqlite
        record_sqlite = StrategyRecord(
            id=json_id,
            code="class Test: pass",
            class_name="Test",
            status=StrategyStatus.ACCEPTED,
            metrics={"ann_return": 15.0, "sharpe": 1.2},
            tags=["trend", "momentum"],
            parent_ids=[],
            generation=3,
            fold=2,
            improvement_delta=5.0,
        )
        sqlite_backend.save(record_sqlite)

        json_loaded = json_backend.load(json_id)
        sqlite_loaded = sqlite_backend.load(json_id)

        assert json_loaded.class_name == sqlite_loaded.class_name
        assert json_loaded.status == sqlite_loaded.status
        assert json_loaded.metrics == sqlite_loaded.metrics
        assert set(json_loaded.tags) == set(sqlite_loaded.tags)
        assert json_loaded.generation == sqlite_loaded.generation

    def test_equivalent_query_by_status(self, json_backend, sqlite_backend):
        """Both backends should produce equivalent query results."""
        records = [
            StrategyRecord(class_name="A", status=StrategyStatus.ACCEPTED),
            StrategyRecord(class_name="B", status=StrategyStatus.REJECTED),
            StrategyRecord(class_name="C", status=StrategyStatus.ACCEPTED),
        ]

        for r in records:
            json_backend.save(r)
            sqlite_backend.save(r)

        json_results = json_backend.query({"status": StrategyStatus.ACCEPTED})
        sqlite_results = sqlite_backend.query({"status": StrategyStatus.ACCEPTED})

        assert len(json_results) == len(sqlite_results) == 2


class TestPerformance:
    """Performance tests for backends (Phase 13C)."""

    def test_sqlite_handles_many_records(self):
        """SQLite should handle many records efficiently."""
        tmpfile = tempfile.mktemp(suffix=".sqlite")
        backend = SqliteBackend(tmpfile)

        try:
            # Insert 100 records
            for i in range(100):
                record = StrategyRecord(
                    class_name=f"Strategy{i}",
                    metrics={"ann_return": float(i)},
                    tags=[f"tag{i % 5}"],
                    generation=i % 10,
                )
                backend.save(record)

            # Verify count
            assert backend.count() == 100

            # Query should work
            gen5 = backend.query({"generation": 5})
            assert len(gen5) == 10

            # Top performers should work
            top = backend.get_top_performers(n=5)
            assert len(top) == 5

        finally:
            Path(tmpfile).unlink(missing_ok=True)

    def test_json_handles_many_records(self):
        """JSON backend should handle many records."""
        tmpdir = tempfile.mkdtemp()
        backend = JsonFileBackend(tmpdir)

        try:
            # Insert 100 records
            for i in range(100):
                record = StrategyRecord(
                    class_name=f"Strategy{i}",
                    metrics={"ann_return": float(i)},
                    tags=[f"tag{i % 5}"],
                    generation=i % 10,
                )
                backend.save(record)

            # Verify count
            assert backend.count() == 100

            # Query should work
            gen5 = backend.query({"generation": 5})
            assert len(gen5) == 10

        finally:
            shutil.rmtree(tmpdir)
