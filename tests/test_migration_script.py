"""Tests for scripts/migrate_to_program_db.py."""

import importlib.util
import json
from pathlib import Path

import pytest

from profit.program_db import JsonFileBackend, ProgramDatabase, StrategyStatus

SCRIPT_PATH = (
    Path(__file__).parent.parent / "scripts" / "migrate_to_program_db.py"
)


@pytest.fixture
def migration():
    """Import the migration script as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("migrate_to_program_db", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_run_dir(tmp_path):
    """Build a StrategyPersister-style run directory."""
    run_dir = tmp_path / "run_20240101_000000"
    fold_dir = run_dir / "fold_1"
    fold_dir.mkdir(parents=True)

    (run_dir / "run_summary.json").write_text(json.dumps({"run_id": "test"}))

    strategy_code = (
        "class EMACrossover_1(Strategy):\n"
        "    def init(self):\n        pass\n"
        "    def next(self):\n        pass\n"
    )
    (fold_dir / "fold_1_gen_1.py").write_text(strategy_code)
    (fold_dir / "fold_1_gen_1.json").write_text(
        json.dumps(
            {
                "class_name": "EMACrossover_1",
                "generation": 1,
                "improvement_proposal": "Tighten the entry filter",
                "metrics": {"AnnReturn%": 12.5},
            }
        )
    )

    # Per-fold best + overall best must be skipped by the migration
    (fold_dir / "best_strategy.py").write_text(strategy_code)
    (fold_dir / "best_strategy.json").write_text(json.dumps({"class_name": "Best"}))
    (run_dir / "best_overall.py").write_text(strategy_code)

    return run_dir


class TestMigrateRun:
    def test_migrates_strategies_as_roots(self, migration, fake_run_dir, tmp_path):
        db = ProgramDatabase(backend=JsonFileBackend(str(tmp_path / "db")))

        migrated = migration.migrate_run(fake_run_dir, db)

        assert migrated == 1
        records = db.backend.query({})
        assert len(records) == 1

        record = records[0]
        assert record.class_name == "EMACrossover_1"
        assert record.parent_ids == []  # roots: no false parent links
        assert record.status == StrategyStatus.ACCEPTED
        assert record.generation == 1
        assert record.fold == 1
        assert record.mutation_text == "Tighten the entry filter"
        assert record.metrics == {"AnnReturn%": 12.5}

    def test_rerun_is_idempotent(self, migration, fake_run_dir, tmp_path):
        """Re-running the migration must not duplicate strategies."""
        db = ProgramDatabase(backend=JsonFileBackend(str(tmp_path / "db")))

        assert migration.migrate_run(fake_run_dir, db) == 1
        assert migration.migrate_run(fake_run_dir, db) == 0
        assert len(db.backend.query({})) == 1

    def test_skips_orphan_json(self, migration, fake_run_dir, tmp_path):
        """A metadata file without a matching .py file is skipped."""
        (fake_run_dir / "fold_1" / "fold_1_gen_2.json").write_text(
            json.dumps({"class_name": "Orphan"})
        )
        db = ProgramDatabase(backend=JsonFileBackend(str(tmp_path / "db")))

        migrated = migration.migrate_run(fake_run_dir, db)

        assert migrated == 1
        assert all(r.class_name != "Orphan" for r in db.backend.query({}))
