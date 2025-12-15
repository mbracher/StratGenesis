"""Unit tests for ProgramDatabase and related classes."""

import pytest
import shutil
import tempfile
from pathlib import Path

from profit.program_db import (
    ProgramDatabase,
    JsonFileBackend,
    StrategyRecord,
    StrategyStatus,
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
