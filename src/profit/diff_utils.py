"""
Diff-based code mutation utilities.

Provides EVOLVE block parsing, diff application, and code validation
for surgical strategy mutations.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class MatchMode(Enum):
    """Mode for matching SEARCH patterns in blocks."""

    STRICT = "strict"  # Literal substring match
    TOLERANT = "tolerant"  # Normalized whitespace, require uniqueness


@dataclass
class DiffBlock:
    """Represents a single SEARCH/REPLACE diff operation."""

    block_name: str
    search: str  # Preserved raw (no stripping)
    replace: str  # Preserved raw (no stripping)


@dataclass
class EvolveBlock:
    """Represents an EVOLVE block extracted from code."""

    name: str
    content: str  # Code inside the block (excluding markers)
    start_idx: int  # Character offset where block starts (including marker)
    end_idx: int  # Character offset where block ends (including marker)
    indentation: str  # Leading whitespace of the marker line
    full_text: str  # Including markers


@dataclass
class DiffApplicationResult:
    """Result of applying diffs to code."""

    success: bool
    modified_code: str
    blocks_modified: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    match_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of code validation."""

    valid: bool
    error: Optional[str] = None
    error_type: Optional[str] = None  # 'syntax', 'structure', 'security', 'ambiguous'


# Regex patterns for parsing
# Matches: # EVOLVE-BLOCK: name ... # END-EVOLVE-BLOCK
EVOLVE_BLOCK_PATTERN = re.compile(
    r"^([ \t]*)# EVOLVE-BLOCK:\s*(\w+)\s*\n"  # Start marker with name
    r"(.*?)"  # Content (non-greedy)
    r"^\1# END-EVOLVE-BLOCK\s*$",  # End marker with same indent
    re.MULTILINE | re.DOTALL,
)

# Matches LLM diff output format
DIFF_BLOCK_PATTERN = re.compile(
    r'<<<SEARCH\s+block_name="(\w+)">>>\s*\n'  # Start with block name
    r"(.*?)"  # Search content
    r"<<<REPLACE>>>\s*\n"  # Replace marker
    r"(.*?)"  # Replace content
    r"<<<END>>>",  # End marker
    re.DOTALL,
)

# Dangerous operations to block
DANGEROUS_PATTERNS = [
    "os.system",
    "subprocess",
    "eval(",
    "__import__",
    "open(",
    "exec(",
    "compile(",
    "os.popen",
    "os.spawn",
]

# Blocked import modules (network/file access)
BLOCKED_IMPORTS = {
    "requests",
    "urllib",
    "urllib3",
    "socket",
    "http",
    "http.client",
    "ftplib",
    "paramiko",
    "fabric",
    "asyncssh",
    "aiohttp",
    "httpx",
    "shutil",
    "tempfile",
    "subprocess",
}

# Allowed imports for strategy code
ALLOWED_IMPORTS = {
    "numpy",
    "np",
    "pandas",
    "pd",
    "backtesting",
    "Strategy",
    "math",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "typing",
    "dataclasses",
}


def extract_evolve_blocks(code: str) -> Dict[str, EvolveBlock]:
    """
    Extract all EVOLVE blocks from strategy code.

    Args:
        code: Full strategy source code

    Returns:
        Dict mapping block names to EvolveBlock objects with character offsets
    """
    blocks = {}

    for match in EVOLVE_BLOCK_PATTERN.finditer(code):
        indentation = match.group(1)
        name = match.group(2)
        content = match.group(3)

        blocks[name] = EvolveBlock(
            name=name,
            content=content,
            start_idx=match.start(),
            end_idx=match.end(),
            indentation=indentation,
            full_text=match.group(0),
        )

    return blocks


def parse_diff_response(llm_response: str) -> List[DiffBlock]:
    """
    Parse LLM response to extract diff blocks.

    Handles noise like markdown code fences by only extracting
    the specific diff format.

    Args:
        llm_response: Raw LLM output containing diff blocks

    Returns:
        List of DiffBlock objects (raw text preserved)
    """
    diffs = []

    for match in DIFF_BLOCK_PATTERN.finditer(llm_response):
        block_name = match.group(1)
        search = match.group(2)
        replace = match.group(3)

        # Raw text preserved - no stripping
        diffs.append(
            DiffBlock(
                block_name=block_name,
                search=search,
                replace=replace,
            )
        )

    return diffs


def normalize_code(code: str) -> str:
    """
    Normalize code for tolerant comparison.

    Strips leading/trailing whitespace per line and removes empty lines.
    """
    lines = code.split("\n")
    normalized = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            normalized.append(stripped)
    return "\n".join(normalized)


def compute_block_indentation(block_content: str) -> str:
    """
    Compute minimum indentation from non-empty lines in block.

    Works for both class-level and method-level blocks.
    """
    lines = block_content.split("\n")
    min_indent = None

    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if min_indent is None or indent < min_indent:
                min_indent = indent

    return " " * (min_indent or 0)


def reindent_replacement(replace: str, target_indent: str) -> str:
    """
    Dedent replacement code, then reindent to target indentation.

    Uses textwrap.dedent for clean normalization.
    """
    # Dedent the replacement to remove common leading whitespace
    dedented = textwrap.dedent(replace)

    # Reindent each line to the target indentation
    lines = dedented.split("\n")
    reindented = []
    for line in lines:
        if line.strip():
            reindented.append(target_indent + line)
        else:
            reindented.append("")

    return "\n".join(reindented)


def match_search_in_block(
    block_content: str,
    search: str,
    mode: MatchMode = MatchMode.TOLERANT,
) -> Tuple[int, Optional[str], Optional[int]]:
    """
    Find search pattern in block content.

    Args:
        block_content: The content of the EVOLVE block
        search: The SEARCH pattern to find
        mode: STRICT (literal) or TOLERANT (normalized)

    Returns:
        Tuple of (match_count, matched_substring, start_offset)
        - match_count: 0 (not found), 1 (unique), >1 (ambiguous)
        - matched_substring: The actual text that matched (for replacement)
        - start_offset: Character offset within block_content where match starts
    """
    if mode == MatchMode.STRICT:
        # Literal substring match
        count = block_content.count(search)
        if count == 0:
            return 0, None, None
        if count > 1:
            return count, search, None  # Ambiguous

        offset = block_content.find(search)
        return 1, search, offset

    else:  # TOLERANT mode
        # Normalize both for comparison
        search_normalized = normalize_code(search)
        search_lines = search_normalized.split("\n")

        if not search_lines:
            return 0, None, None

        block_lines = block_content.split("\n")
        matches = []

        # Sliding window search
        for i in range(len(block_lines) - len(search_lines) + 1):
            match = True
            for j, search_line in enumerate(search_lines):
                if block_lines[i + j].strip() != search_line:
                    match = False
                    break

            if match:
                # Calculate character offset
                offset = sum(len(line) + 1 for line in block_lines[:i])
                matched_text = "\n".join(block_lines[i : i + len(search_lines)])
                matches.append((matched_text, offset))

        if len(matches) == 0:
            return 0, None, None
        if len(matches) > 1:
            return len(matches), None, None  # Ambiguous

        return 1, matches[0][0], matches[0][1]


def apply_diff(
    code: str,
    diff_blocks: List[DiffBlock],
    match_mode: MatchMode = MatchMode.TOLERANT,
) -> DiffApplicationResult:
    """
    Apply diff blocks to strategy code.

    Uses offset-based splicing for precise replacement.
    Fails on ambiguous matches (>1) to prevent accidental damage.

    Args:
        code: Original strategy code
        diff_blocks: List of diffs to apply
        match_mode: STRICT or TOLERANT matching

    Returns:
        DiffApplicationResult with modified code and status
    """
    modified_code = code
    blocks_modified = []
    errors = []
    match_counts = {}

    # Extract current EVOLVE blocks
    evolve_blocks = extract_evolve_blocks(code)

    for diff in diff_blocks:
        # Check if target block exists
        if diff.block_name not in evolve_blocks:
            errors.append(f"Block '{diff.block_name}' not found in code")
            match_counts[diff.block_name] = 0
            continue

        evolve_block = evolve_blocks[diff.block_name]

        # Match search pattern
        count, matched_text, offset = match_search_in_block(
            evolve_block.content, diff.search, match_mode
        )

        match_counts[diff.block_name] = count

        if count == 0:
            errors.append(f"SEARCH pattern not found in block '{diff.block_name}'")
            continue

        if count > 1:
            errors.append(
                f"SEARCH matched {count} times in block '{diff.block_name}'; "
                "include more context lines for uniqueness"
            )
            continue

        # Compute target indentation from matched text
        target_indent = compute_block_indentation(matched_text)

        # Reindent the replacement
        reindented_replace = reindent_replacement(diff.replace, target_indent)

        # Check if replacement is actually different
        if matched_text.strip() == reindented_replace.strip():
            errors.append(
                f"Replacement in block '{diff.block_name}' is identical to original"
            )
            continue

        # Build new block content with single replacement
        new_content = evolve_block.content.replace(matched_text, reindented_replace, 1)

        # Build new full block text
        new_full_text = (
            f"{evolve_block.indentation}# EVOLVE-BLOCK: {diff.block_name}\n"
            f"{new_content}"
            f"{evolve_block.indentation}# END-EVOLVE-BLOCK"
        )

        # Use offset-based splicing for precise replacement
        modified_code = (
            modified_code[: evolve_block.start_idx]
            + new_full_text
            + modified_code[evolve_block.end_idx :]
        )

        blocks_modified.append(diff.block_name)

        # Re-extract blocks with updated offsets for subsequent diffs
        evolve_blocks = extract_evolve_blocks(modified_code)

    success = len(blocks_modified) > 0 and len(errors) == 0

    return DiffApplicationResult(
        success=success,
        modified_code=modified_code,
        blocks_modified=blocks_modified,
        errors=errors,
        match_counts=match_counts,
    )


def _find_strategy_class(tree: ast.Module) -> Optional[ast.ClassDef]:
    """Find the strategy class (inherits from Strategy) in AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                # Check for Strategy or backtesting.Strategy
                if isinstance(base, ast.Name) and base.id == "Strategy":
                    return node
                if (
                    isinstance(base, ast.Attribute)
                    and base.attr == "Strategy"
                ):
                    return node
    return None


def _get_class_methods(class_node: ast.ClassDef) -> Set[str]:
    """Get all method names defined in a class."""
    methods = set()
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            methods.add(node.name)
    return methods


def _check_imports(tree: ast.Module, allowed: Optional[Set[str]] = None) -> List[str]:
    """Check for blocked imports, return list of violations."""
    if allowed is None:
        allowed = ALLOWED_IMPORTS

    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_base = alias.name.split(".")[0]
                if module_base in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_base = node.module.split(".")[0]
                if module_base in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import from: {node.module}")

    return violations


def validate_modified_code(
    original: str,
    modified: str,
    allowed_imports: Optional[Set[str]] = None,
) -> ValidationResult:
    """
    Validate that modified code is syntactically correct and safe.

    Checks:
    1. Syntax validation (ast.parse)
    2. Strategy class structure (init/next methods IN the Strategy subclass)
    3. Dangerous operations (expanded list)
    4. Import validation (block network modules unless allowlisted)

    Args:
        original: Original strategy code
        modified: Modified strategy code
        allowed_imports: Optional set of allowed import modules

    Returns:
        ValidationResult indicating if code is valid
    """
    # Check 1: Syntax validation
    try:
        modified_tree = ast.parse(modified)
    except SyntaxError as e:
        return ValidationResult(
            valid=False,
            error=f"Syntax error: {e.msg} at line {e.lineno}",
            error_type="syntax",
        )

    # Check 2: Strategy class structure
    strategy_class = _find_strategy_class(modified_tree)

    if strategy_class is None:
        return ValidationResult(
            valid=False,
            error="No Strategy subclass found in modified code",
            error_type="structure",
        )

    # Verify init and next methods are IN the strategy class
    class_methods = _get_class_methods(strategy_class)
    required_methods = {"init", "next"}
    missing = required_methods - class_methods

    if missing:
        return ValidationResult(
            valid=False,
            error=f"Missing required methods in Strategy class: {missing}",
            error_type="structure",
        )

    # Check 3: Dangerous operations (only if newly introduced)
    for pattern in DANGEROUS_PATTERNS:
        if pattern in modified and pattern not in original:
            return ValidationResult(
                valid=False,
                error=f"Dangerous operation detected: {pattern}",
                error_type="security",
            )

    # Check 4: Import validation
    import_violations = _check_imports(modified_tree, allowed_imports)
    if import_violations:
        # Only flag if this is a new import not in original
        try:
            original_tree = ast.parse(original)
            original_violations = _check_imports(original_tree, allowed_imports)
            new_violations = [v for v in import_violations if v not in original_violations]
            if new_violations:
                return ValidationResult(
                    valid=False,
                    error=f"Security violation: {new_violations[0]}",
                    error_type="security",
                )
        except SyntaxError:
            # Original had syntax error, can't compare
            if import_violations:
                return ValidationResult(
                    valid=False,
                    error=f"Security violation: {import_violations[0]}",
                    error_type="security",
                )

    return ValidationResult(valid=True)


def has_evolve_blocks(code: str) -> bool:
    """Check if code has EVOLVE block markers."""
    return bool(EVOLVE_BLOCK_PATTERN.search(code))


def get_block_names(code: str) -> List[str]:
    """Get names of all EVOLVE blocks in code."""
    return list(extract_evolve_blocks(code).keys())


def has_nested_markers(code: str) -> bool:
    """
    Check if code has nested EVOLVE markers (invalid state).

    Returns True if there are nested markers that would cause issues.
    """
    blocks = extract_evolve_blocks(code)
    for block in blocks.values():
        # Check if content contains another EVOLVE-BLOCK marker
        if "# EVOLVE-BLOCK:" in block.content:
            return True
    return False
