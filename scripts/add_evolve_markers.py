#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
AST-based EVOLVE marker insertion for strategy files.

This script adds EVOLVE-BLOCK markers to strategy files, enabling
diff-based mutations in the evolution loop.

More reliable than regex-based approach:
- Uses Python AST to identify code structures
- Identifies class-level assignments for indicator_params
- Identifies method bodies for signal_generation, entry_logic, exit_logic
- Respects Python structure and indentation

Usage:
    # Dry run (preview changes)
    uv run scripts/add_evolve_markers.py src/profit/strategies.py

    # Apply changes
    uv run scripts/add_evolve_markers.py src/profit/strategies.py --apply

    # Process multiple files
    uv run scripts/add_evolve_markers.py evolved_strategies/*.py --apply
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set


# Default configuration for block detection
DEFAULT_BLOCK_CONFIG = {
    # Class-level attributes to wrap as indicator_params
    "param_attributes": {
        "bb_period",
        "bb_stddev",
        "cci_period",
        "cci_oversold",
        "cci_overbought",
        "cci_exit_low",
        "cci_exit_high",
        "fast_ema",
        "slow_ema",
        "fast",
        "slow",
        "signal",
        "lookback",
        "oversold_threshold",
        "overbought_threshold",
        "exit_low",
        "exit_high",
    },
    # Pattern keywords that suggest entry logic
    "entry_patterns": {"self.buy(", "self.sell("},
    # Pattern keywords that suggest exit logic
    "exit_patterns": {"self.position.close(", "position.close()"},
}


class StrategyAnalyzer(ast.NodeVisitor):
    """AST visitor that identifies potential EVOLVE block locations."""

    def __init__(self, source_lines: List[str], config: Dict = None):
        self.source_lines = source_lines
        self.config = config or DEFAULT_BLOCK_CONFIG
        self.blocks: List[Tuple[str, int, int, str]] = []  # (name, start, end, indent)
        self.current_class = None

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions to find Strategy subclasses."""
        # Check if this class inherits from Strategy
        is_strategy = any(
            (isinstance(base, ast.Name) and base.id == "Strategy")
            or (isinstance(base, ast.Attribute) and base.attr == "Strategy")
            for base in node.bases
        )

        if is_strategy:
            self.current_class = node.name
            self._analyze_class(node)
            self.current_class = None

        self.generic_visit(node)

    def _analyze_class(self, node: ast.ClassDef):
        """Analyze a Strategy class for potential EVOLVE blocks."""
        # Find class-level assignments (indicator_params)
        param_assigns = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if target.id in self.config["param_attributes"]:
                            param_assigns.append(item)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id in self.config["param_attributes"]:
                    param_assigns.append(item)

        if param_assigns:
            # Group consecutive param assignments
            start_line = min(a.lineno for a in param_assigns)
            end_line = max(a.end_lineno or a.lineno for a in param_assigns)
            indent = self._get_indent(start_line)
            self.blocks.append(("indicator_params", start_line, end_line, indent))

        # Analyze methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "init":
                    self._analyze_init_method(item)
                elif item.name == "next":
                    self._analyze_next_method(item)

    def _analyze_init_method(self, node: ast.FunctionDef):
        """Analyze init() method for signal_generation block."""
        # Look for self.I() or self.indicator assignments
        indicator_stmts = []
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                # Check if assigning to self.something
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(
                        target.value, ast.Name
                    ):
                        if target.value.id == "self":
                            indicator_stmts.append(stmt)

        if indicator_stmts:
            start_line = min(s.lineno for s in indicator_stmts)
            end_line = max(s.end_lineno or s.lineno for s in indicator_stmts)
            indent = self._get_indent(start_line)
            self.blocks.append(("signal_generation", start_line, end_line, indent))

    def _analyze_next_method(self, node: ast.FunctionDef):
        """Analyze next() method for entry_logic and exit_logic blocks."""
        # Find entry conditions (self.buy(), self.sell())
        # Find exit conditions (self.position.close())
        entry_stmts = []
        exit_stmts = []

        for stmt in node.body:
            if isinstance(stmt, ast.If):
                stmt_code = self._get_stmt_code(stmt)
                if any(p in stmt_code for p in self.config["entry_patterns"]):
                    entry_stmts.append(stmt)
                if any(p in stmt_code for p in self.config["exit_patterns"]):
                    exit_stmts.append(stmt)

        if entry_stmts:
            start_line = min(s.lineno for s in entry_stmts)
            end_line = max(s.end_lineno or s.lineno for s in entry_stmts)
            indent = self._get_indent(start_line)
            self.blocks.append(("entry_logic", start_line, end_line, indent))

        if exit_stmts:
            start_line = min(s.lineno for s in exit_stmts)
            end_line = max(s.end_lineno or s.lineno for s in exit_stmts)
            indent = self._get_indent(start_line)
            self.blocks.append(("exit_logic", start_line, end_line, indent))

    def _get_indent(self, line_num: int) -> str:
        """Get the indentation of a source line."""
        line = self.source_lines[line_num - 1]  # 1-indexed
        return line[: len(line) - len(line.lstrip())]

    def _get_stmt_code(self, stmt: ast.stmt) -> str:
        """Get source code for a statement."""
        start = stmt.lineno - 1
        end = (stmt.end_lineno or stmt.lineno)
        return "\n".join(self.source_lines[start:end])


def has_evolve_markers(code: str) -> bool:
    """Check if code already has EVOLVE block markers."""
    return "# EVOLVE-BLOCK:" in code


def add_markers_to_code(code: str) -> Tuple[str, List[str]]:
    """
    Add EVOLVE markers to strategy code using AST analysis.

    Args:
        code: Strategy source code.

    Returns:
        Tuple of (modified_code, list_of_added_blocks).
    """
    if has_evolve_markers(code):
        return code, []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"  Syntax error: {e}")
        return code, []

    source_lines = code.split("\n")
    analyzer = StrategyAnalyzer(source_lines)
    analyzer.visit(tree)

    if not analyzer.blocks:
        return code, []

    # Sort blocks by line number (descending) to insert from bottom to top
    blocks = sorted(analyzer.blocks, key=lambda b: b[1], reverse=True)

    added_blocks = []
    result_lines = source_lines.copy()

    for block_name, start_line, end_line, indent in blocks:
        # Check if block spans multiple statements (avoid wrapping too much)
        start_idx = start_line - 1
        end_idx = end_line  # end_lineno is inclusive

        # Insert end marker
        result_lines.insert(end_idx, f"{indent}# END-EVOLVE-BLOCK")

        # Insert start marker
        result_lines.insert(start_idx, f"{indent}# EVOLVE-BLOCK: {block_name}")

        added_blocks.append(block_name)

    return "\n".join(result_lines), added_blocks


def process_file(filepath: Path, dry_run: bool = True) -> bool:
    """
    Process a single strategy file.

    Args:
        filepath: Path to the strategy file.
        dry_run: If True, don't write changes.

    Returns:
        True if changes were made (or would be made).
    """
    print(f"\nProcessing: {filepath}")

    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return False

    code = filepath.read_text()

    if has_evolve_markers(code):
        print("  Already has EVOLVE markers, skipping")
        return False

    modified_code, added_blocks = add_markers_to_code(code)

    if not added_blocks:
        print("  No blocks identified for marking")
        return False

    print(f"  Found blocks: {', '.join(added_blocks)}")

    if dry_run:
        print("  DRY RUN: Would add markers (use --apply to modify)")
    else:
        filepath.write_text(modified_code)
        print("  Added EVOLVE markers")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add EVOLVE-BLOCK markers to strategy files for diff-based mutations."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Strategy files to process (supports glob patterns)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files (default is dry run)",
    )

    args = parser.parse_args()

    # Expand glob patterns and collect files
    files = []
    for pattern in args.files:
        path = Path(pattern)
        if path.exists():
            files.append(path)
        else:
            # Try as glob pattern
            matched = list(Path(".").glob(pattern))
            if matched:
                files.extend(matched)
            else:
                print(f"Warning: No files match pattern: {pattern}")

    if not files:
        print("No files to process")
        return 1

    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Files: {len(files)}")

    modified_count = 0
    for filepath in files:
        if filepath.suffix == ".py":
            if process_file(filepath, dry_run=not args.apply):
                modified_count += 1

    print(f"\n{'Modified' if args.apply else 'Would modify'}: {modified_count} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
