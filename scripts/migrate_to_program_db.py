#!/usr/bin/env python3
# No PEP 723 inline-deps block: this script imports the profit package, so it
# must run in the project environment (an isolated `uv run script` env would
# not have profit installed).
"""Migrate existing StrategyPersister output to the ProgramDatabase.

NOTE: Parent lineage is NOT preserved during migration because existing
metadata stores class names, not DB IDs. Migrated strategies are treated
as independent roots. Future strategies evolved from them will have proper
lineage tracking.

Usage:
    # Dry run (list what would be migrated)
    uv run python scripts/migrate_to_program_db.py --source evolved_strategies --dest program_db

    # Perform the migration
    uv run python scripts/migrate_to_program_db.py --source evolved_strategies --dest program_db --apply
"""

import argparse
import json
import sys
from pathlib import Path

from profit.program_db import (
    JsonFileBackend,
    ProgramDatabase,
    SqliteBackend,
    StrategyStatus,
)


def iter_strategy_files(run_dir: Path):
    """Yield (fold_num, json_path, py_path) for each migratable strategy."""
    for fold_dir in sorted(run_dir.glob("fold_*")):
        fold_num = int(fold_dir.name.split("_")[1])

        for json_file in sorted(fold_dir.glob("*.json")):
            # Per-fold/per-run summary files are not strategy records
            if json_file.name in ("best_strategy.json", "run_summary.json"):
                continue

            py_file = json_file.with_suffix(".py")
            if not py_file.exists():
                continue

            yield fold_num, json_file, py_file


def existing_code_set(db: ProgramDatabase) -> set:
    """Codes already in the database, for idempotent re-runs."""
    return {record.code for record in db.backend.query({})}


def migrate_run(run_dir: Path, db: ProgramDatabase, seen_codes: set | None = None) -> int:
    """Migrate a single run directory. Returns the number of records created.

    Strategies whose exact code is already in the database (or was migrated
    earlier in this invocation) are skipped, so re-running with --apply does
    not create duplicates.
    """
    if seen_codes is None:
        seen_codes = existing_code_set(db)

    migrated = 0
    for fold_num, json_file, py_file in iter_strategy_files(run_dir):
        with open(json_file) as f:
            meta = json.load(f)

        code = py_file.read_text()
        if code in seen_codes:
            continue
        seen_codes.add(code)

        # Don't set parent_ids - migrated strategies are roots; existing
        # metadata has class names, not DB IDs
        db.register_strategy(
            code=code,
            class_name=meta.get("class_name", ""),
            parent_ids=[],
            mutation_text=meta.get("improvement_proposal", ""),
            metrics=meta.get("metrics", {}),
            tags=[],
            status=StrategyStatus.ACCEPTED,  # Assume migrated = accepted
            generation=meta.get("generation", 0),
            fold=fold_num,
        )
        migrated += 1

    return migrated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate StrategyPersister runs into the program database."
    )
    parser.add_argument(
        "--source",
        default="evolved_strategies",
        help="Source directory containing run_* directories (default: evolved_strategies)",
    )
    parser.add_argument(
        "--dest",
        default="program_db",
        help="Destination database path (default: program_db)",
    )
    parser.add_argument(
        "--db-backend",
        choices=["json", "sqlite"],
        default="json",
        help="Program database backend (default: json)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration (default: dry run)",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: source directory not found: {source}")
        return 1

    run_dirs = sorted(source.glob("run_*"))
    if not run_dirs:
        print(f"No run_* directories found in {source}; nothing to migrate.")
        return 0

    if not args.apply:
        print(f"Dry run - would migrate from {source} to {args.dest}:")
        total = 0
        for run_dir in run_dirs:
            count = sum(1 for _ in iter_strategy_files(run_dir))
            total += count
            print(f"  {run_dir.name}: {count} strategies")
        print(f"Total: {total} strategies. Re-run with --apply to migrate.")
        return 0

    if args.db_backend == "sqlite":
        dest = args.dest if args.dest.endswith(".sqlite") else f"{args.dest}.sqlite"
        backend = SqliteBackend(dest)
    else:
        backend = JsonFileBackend(args.dest)
    db = ProgramDatabase(backend)

    total = 0
    seen_codes = existing_code_set(db)
    for run_dir in run_dirs:
        print(f"Migrating {run_dir.name}...")
        total += migrate_run(run_dir, db, seen_codes)

    print(f"Migration complete. {total} strategies imported ({db.backend.count()} in database).")
    print("NOTE: Migrated strategies have no parent links (treated as roots).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
