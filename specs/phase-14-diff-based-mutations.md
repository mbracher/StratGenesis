# Phase 14: Diff-Based Edits & EVOLVE Blocks

## Objective

Implement surgical code mutations using SEARCH/REPLACE diffs instead of full rewrites. This reduces breakage during evolution by allowing targeted changes to specific code sections while preserving working logic.

From the AlphaEvolve paper:

> AlphaEvolve supports marking evolution targets using EVOLVE-BLOCK markers and asking the model to emit SEARCH/REPLACE diff blocks instead of rewriting everything.

---

## Dependencies

- Phase 4 (LLM Interface) - existing `LLMClient` class
- Phase 6 (Evolutionary Engine) - existing `evolve_strategy()` method

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                     Diff-Based Mutation Flow                        │
│                                                                     │
│  Original Code          LLM Output             Modified Code        │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐     │
│  │ # EVOLVE-BLOCK│      │<<<SEARCH>>>  │      │ # EVOLVE-BLOCK│     │
│  │ def signal(): │ ───► │old_code      │ ───► │ def signal(): │     │
│  │   old_logic   │      │<<<REPLACE>>> │      │   new_logic   │     │
│  │ # END-EVOLVE  │      │new_code      │      │ # END-EVOLVE  │     │
│  └──────────────┘      │<<<END>>>     │      └──────────────┘     │
│                        └──────────────┘                            │
│                                                                     │
│  extract_evolve_blocks() → generate_diff() → apply_diff()          │
│                                  │                                  │
│                                  ▼                                  │
│                          validate_code()                            │
│                                  │                                  │
│                          fallback_to_full_rewrite()                │
└────────────────────────────────────────────────────────────────────┘
```

---

## EVOLVE Block Convention

### Marker Syntax

```python
class MyStrategy(Strategy):
    # EVOLVE-BLOCK: signal_generation
    def calculate_signal(self):
        """Signal generation logic - safe to modify."""
        ema_fast = self.data.Close.rolling(self.fast_period).mean()
        ema_slow = self.data.Close.rolling(self.slow_period).mean()
        return ema_fast > ema_slow
    # END-EVOLVE-BLOCK

    # EVOLVE-BLOCK: entry_logic
    def should_enter(self):
        """Entry conditions - safe to modify."""
        return self.calculate_signal() and not self.position
    # END-EVOLVE-BLOCK

    # EVOLVE-BLOCK: exit_logic
    def should_exit(self):
        """Exit conditions - safe to modify."""
        return not self.calculate_signal() and self.position
    # END-EVOLVE-BLOCK

    # EVOLVE-BLOCK: position_sizing
    def calculate_size(self):
        """Position size calculation - safe to modify."""
        return 0.95  # 95% of available equity
    # END-EVOLVE-BLOCK

    def init(self):
        """Initialization - NOT marked for evolution."""
        self.fast_period = 50
        self.slow_period = 200

    def next(self):
        """Main loop - calls evolve blocks but structure is fixed."""
        if self.should_enter():
            self.buy(size=self.calculate_size())
        elif self.should_exit():
            self.position.close()
```

### Block Types

| Block Name | Purpose | Typical Contents |
|------------|---------|------------------|
| `signal_generation` | Core trading signal | Indicator calculations, signal logic |
| `entry_logic` | Entry conditions | When to open positions |
| `exit_logic` | Exit conditions | When to close positions |
| `position_sizing` | Size calculation | Risk management, position sizing |
| `indicator_params` | Parameters | Configurable values |
| `filters` | Trade filters | Additional conditions |

---

## Data Structures

### DiffBlock

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DiffBlock:
    """Represents a single SEARCH/REPLACE diff operation."""

    block_name: str  # Name of the EVOLVE block to modify
    search: str      # Code to find and replace
    replace: str     # Replacement code

    def __post_init__(self):
        # Normalize whitespace for matching
        self.search = self.search.strip()
        self.replace = self.replace.strip()


@dataclass
class EvolveBlock:
    """Represents an EVOLVE block extracted from code."""

    name: str           # Block identifier
    content: str        # Code inside the block
    start_line: int     # Line number where block starts
    end_line: int       # Line number where block ends
    indentation: str    # Leading whitespace for proper formatting


@dataclass
class ValidationResult:
    """Result of code validation after diff application."""

    valid: bool
    error: Optional[str] = None
    error_type: Optional[str] = None  # 'syntax', 'import', 'runtime'


@dataclass
class DiffApplicationResult:
    """Result of applying diffs to code."""

    success: bool
    modified_code: str
    blocks_modified: list[str]
    errors: list[str]
```

---

## Diff Format Specification

### LLM Output Format

The Coder LLM outputs diffs in this format:

```
<<<SEARCH block_name="signal_generation">>>
        ema_fast = self.data.Close.rolling(self.fast_period).mean()
        ema_slow = self.data.Close.rolling(self.slow_period).mean()
        return ema_fast > ema_slow
<<<REPLACE>>>
        ema_fast = self.data.Close.ewm(span=self.fast_period).mean()
        ema_slow = self.data.Close.ewm(span=self.slow_period).mean()
        # Add momentum confirmation
        momentum = self.data.Close.diff(5) > 0
        return (ema_fast > ema_slow) & momentum
<<<END>>>
```

### Multiple Diffs

Multiple blocks can be modified in one response:

```
<<<SEARCH block_name="signal_generation">>>
...old code...
<<<REPLACE>>>
...new code...
<<<END>>>

<<<SEARCH block_name="position_sizing">>>
...old code...
<<<REPLACE>>>
...new code...
<<<END>>>
```

---

## Implementation

### File: `src/profit/diff_utils.py`

```python
"""
Diff-based code mutation utilities.

Provides EVOLVE block parsing, diff application, and code validation
for surgical strategy mutations.
"""

import re
import ast
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DiffBlock:
    """Represents a single SEARCH/REPLACE diff operation."""
    block_name: str
    search: str
    replace: str

    def __post_init__(self):
        self.search = self.search.strip()
        self.replace = self.replace.strip()


@dataclass
class EvolveBlock:
    """Represents an EVOLVE block extracted from code."""
    name: str
    content: str
    start_line: int
    end_line: int
    indentation: str
    full_text: str  # Including markers


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    error: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class DiffApplicationResult:
    """Result of applying diffs to code."""
    success: bool
    modified_code: str
    blocks_modified: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# Regex patterns for parsing
EVOLVE_BLOCK_PATTERN = re.compile(
    r'^(\s*)# EVOLVE-BLOCK:\s*(\w+)\s*\n'  # Start marker with name
    r'(.*?)'                                # Content (non-greedy)
    r'^\1# END-EVOLVE-BLOCK\s*$',           # End marker with same indent
    re.MULTILINE | re.DOTALL
)

DIFF_BLOCK_PATTERN = re.compile(
    r'<<<SEARCH\s+block_name="(\w+)">>>\s*\n'  # Start with block name
    r'(.*?)'                                     # Search content
    r'<<<REPLACE>>>\s*\n'                        # Replace marker
    r'(.*?)'                                     # Replace content
    r'<<<END>>>',                                # End marker
    re.DOTALL
)


def extract_evolve_blocks(code: str) -> Dict[str, EvolveBlock]:
    """
    Extract all EVOLVE blocks from strategy code.

    Args:
        code: Full strategy source code

    Returns:
        Dict mapping block names to EvolveBlock objects
    """
    blocks = {}
    lines = code.split('\n')

    for match in EVOLVE_BLOCK_PATTERN.finditer(code):
        indentation = match.group(1)
        name = match.group(2)
        content = match.group(3)

        # Calculate line numbers
        start_pos = match.start()
        end_pos = match.end()
        start_line = code[:start_pos].count('\n')
        end_line = code[:end_pos].count('\n')

        blocks[name] = EvolveBlock(
            name=name,
            content=content.strip(),
            start_line=start_line,
            end_line=end_line,
            indentation=indentation,
            full_text=match.group(0)
        )

    return blocks


def parse_diff_response(llm_response: str) -> List[DiffBlock]:
    """
    Parse LLM response to extract diff blocks.

    Args:
        llm_response: Raw LLM output containing diff blocks

    Returns:
        List of DiffBlock objects
    """
    diffs = []

    for match in DIFF_BLOCK_PATTERN.finditer(llm_response):
        block_name = match.group(1)
        search = match.group(2)
        replace = match.group(3)

        diffs.append(DiffBlock(
            block_name=block_name,
            search=search,
            replace=replace
        ))

    return diffs


def apply_diff(code: str, diff_blocks: List[DiffBlock]) -> DiffApplicationResult:
    """
    Apply diff blocks to strategy code.

    Args:
        code: Original strategy code
        diff_blocks: List of diffs to apply

    Returns:
        DiffApplicationResult with modified code and status
    """
    modified_code = code
    blocks_modified = []
    errors = []

    # Extract current EVOLVE blocks
    evolve_blocks = extract_evolve_blocks(code)

    for diff in diff_blocks:
        # Check if target block exists
        if diff.block_name not in evolve_blocks:
            errors.append(f"Block '{diff.block_name}' not found in code")
            continue

        evolve_block = evolve_blocks[diff.block_name]

        # Normalize whitespace for matching
        search_normalized = normalize_code(diff.search)
        block_content_normalized = normalize_code(evolve_block.content)

        # Try to find and replace within the block
        if search_normalized in block_content_normalized:
            # Preserve indentation
            indented_replace = indent_code(diff.replace, evolve_block.indentation)

            # Replace in the full text
            old_full = evolve_block.full_text
            new_content = evolve_block.content.replace(
                find_matching_substring(evolve_block.content, diff.search),
                diff.replace
            )

            new_full = (
                f"{evolve_block.indentation}# EVOLVE-BLOCK: {diff.block_name}\n"
                f"{new_content}\n"
                f"{evolve_block.indentation}# END-EVOLVE-BLOCK"
            )

            modified_code = modified_code.replace(old_full, new_full)
            blocks_modified.append(diff.block_name)

            # Update evolve_blocks for subsequent diffs
            evolve_blocks = extract_evolve_blocks(modified_code)
        else:
            errors.append(
                f"Search pattern not found in block '{diff.block_name}'"
            )

    success = len(blocks_modified) > 0 and len(errors) == 0

    return DiffApplicationResult(
        success=success,
        modified_code=modified_code,
        blocks_modified=blocks_modified,
        errors=errors
    )


def normalize_code(code: str) -> str:
    """Normalize code for comparison (strip whitespace, normalize newlines)."""
    lines = code.strip().split('\n')
    normalized = []
    for line in lines:
        # Remove leading/trailing whitespace but preserve relative indentation
        stripped = line.strip()
        if stripped:
            normalized.append(stripped)
    return '\n'.join(normalized)


def find_matching_substring(haystack: str, needle: str) -> str:
    """
    Find the actual substring in haystack that matches the normalized needle.

    This handles whitespace differences between the search pattern and actual code.
    """
    needle_normalized = normalize_code(needle)
    needle_lines = needle_normalized.split('\n')

    haystack_lines = haystack.split('\n')

    for i in range(len(haystack_lines) - len(needle_lines) + 1):
        match = True
        for j, needle_line in enumerate(needle_lines):
            if haystack_lines[i + j].strip() != needle_line:
                match = False
                break

        if match:
            # Return the original (non-normalized) substring
            return '\n'.join(haystack_lines[i:i + len(needle_lines)])

    return needle  # Fallback to original if no match found


def indent_code(code: str, base_indent: str) -> str:
    """Add base indentation to code block."""
    lines = code.split('\n')
    indented = []
    for line in lines:
        if line.strip():
            indented.append(base_indent + '    ' + line)
        else:
            indented.append('')
    return '\n'.join(indented)


def validate_modified_code(
    original: str,
    modified: str
) -> ValidationResult:
    """
    Validate that modified code is syntactically correct and safe.

    Args:
        original: Original strategy code
        modified: Modified strategy code

    Returns:
        ValidationResult indicating if code is valid
    """
    # Check 1: Syntax validation
    try:
        ast.parse(modified)
    except SyntaxError as e:
        return ValidationResult(
            valid=False,
            error=f"Syntax error: {e.msg} at line {e.lineno}",
            error_type='syntax'
        )

    # Check 2: Class structure preserved
    try:
        original_tree = ast.parse(original)
        modified_tree = ast.parse(modified)

        # Find class definitions
        original_classes = [
            node.name for node in ast.walk(original_tree)
            if isinstance(node, ast.ClassDef)
        ]
        modified_classes = [
            node.name for node in ast.walk(modified_tree)
            if isinstance(node, ast.ClassDef)
        ]

        # Strategy class should still exist (possibly with new name)
        if not modified_classes:
            return ValidationResult(
                valid=False,
                error="No class definition found in modified code",
                error_type='structure'
            )

    except Exception as e:
        return ValidationResult(
            valid=False,
            error=f"AST parsing failed: {str(e)}",
            error_type='syntax'
        )

    # Check 3: Required methods present
    required_methods = {'init', 'next'}
    modified_methods = set()

    for node in ast.walk(modified_tree):
        if isinstance(node, ast.FunctionDef):
            modified_methods.add(node.name)

    missing = required_methods - modified_methods
    if missing:
        return ValidationResult(
            valid=False,
            error=f"Missing required methods: {missing}",
            error_type='structure'
        )

    # Check 4: No dangerous operations
    dangerous_patterns = [
        'os.system',
        'subprocess',
        'eval(',
        '__import__',
        'open(',
        'exec(',
    ]

    for pattern in dangerous_patterns:
        if pattern in modified and pattern not in original:
            return ValidationResult(
                valid=False,
                error=f"Dangerous operation detected: {pattern}",
                error_type='security'
            )

    return ValidationResult(valid=True)


def add_evolve_markers(code: str, block_definitions: Dict[str, Tuple[str, str]]) -> str:
    """
    Add EVOLVE block markers to existing code.

    Args:
        code: Strategy code without markers
        block_definitions: Dict of {block_name: (start_pattern, end_pattern)}
            where patterns identify the code section to wrap

    Returns:
        Code with EVOLVE markers added
    """
    modified = code

    for block_name, (start_pattern, end_pattern) in block_definitions.items():
        # Find the section
        start_match = re.search(start_pattern, modified)
        if not start_match:
            continue

        # Find the end of the section
        end_match = re.search(
            end_pattern,
            modified[start_match.end():]
        )
        if not end_match:
            continue

        # Calculate positions
        section_start = start_match.start()
        section_end = start_match.end() + end_match.end()

        # Detect indentation
        line_start = modified.rfind('\n', 0, section_start) + 1
        indent = ''
        for char in modified[line_start:]:
            if char in ' \t':
                indent += char
            else:
                break

        # Insert markers
        section = modified[section_start:section_end]
        wrapped = (
            f"{indent}# EVOLVE-BLOCK: {block_name}\n"
            f"{section}\n"
            f"{indent}# END-EVOLVE-BLOCK"
        )

        modified = modified[:section_start] + wrapped + modified[section_end:]

    return modified


def has_evolve_blocks(code: str) -> bool:
    """Check if code has EVOLVE block markers."""
    return bool(EVOLVE_BLOCK_PATTERN.search(code))


def get_block_names(code: str) -> List[str]:
    """Get names of all EVOLVE blocks in code."""
    return list(extract_evolve_blocks(code).keys())
```

---

## LLM Interface Updates

### New Method: generate_diff()

Add to `src/profit/llm_interface.py`:

```python
def generate_diff(
    self,
    strategy_code: str,
    improvement_proposal: str,
    available_blocks: List[str]
) -> str:
    """
    Generate diff blocks to implement an improvement proposal.

    Instead of rewriting the entire strategy, output surgical
    SEARCH/REPLACE diffs targeting specific EVOLVE blocks.

    Args:
        strategy_code: Current strategy code with EVOLVE markers
        improvement_proposal: The improvement to implement
        available_blocks: Names of EVOLVE blocks that can be modified

    Returns:
        LLM response containing diff blocks
    """
    prompt = f"""You are a precise code editor. Your task is to implement a strategy improvement
using surgical SEARCH/REPLACE diffs.

CURRENT STRATEGY CODE:
```python
{strategy_code}
```

IMPROVEMENT TO IMPLEMENT:
{improvement_proposal}

AVAILABLE BLOCKS FOR MODIFICATION:
{', '.join(available_blocks)}

OUTPUT FORMAT:
For each change, output a diff block in this exact format:

<<<SEARCH block_name="BLOCK_NAME">>>
exact code to find (copy from the strategy)
<<<REPLACE>>>
new code to replace it with
<<<END>>>

RULES:
1. Only modify code within the EVOLVE blocks listed above
2. The SEARCH section must EXACTLY match existing code (including whitespace)
3. Keep changes minimal - only change what's necessary
4. Preserve the overall structure and indentation
5. You can output multiple diff blocks for different blocks
6. If no changes are needed to a block, don't include it

OUTPUT YOUR DIFFS:
"""

    return self._call_llm(prompt, role="coder")


def generate_strategy_code_with_fallback(
    self,
    base_code: str,
    improvement_proposal: str
) -> Tuple[str, bool]:
    """
    Generate modified strategy code, preferring diffs but falling back to full rewrite.

    Args:
        base_code: Original strategy code
        improvement_proposal: The improvement to implement

    Returns:
        Tuple of (modified_code, used_diff_mode)
    """
    from profit.diff_utils import (
        has_evolve_blocks,
        get_block_names,
        parse_diff_response,
        apply_diff,
        validate_modified_code
    )

    # Check if code has EVOLVE blocks
    if has_evolve_blocks(base_code):
        # Try diff-based approach
        available_blocks = get_block_names(base_code)

        diff_response = self.generate_diff(
            base_code,
            improvement_proposal,
            available_blocks
        )

        diff_blocks = parse_diff_response(diff_response)

        if diff_blocks:
            result = apply_diff(base_code, diff_blocks)

            if result.success:
                validation = validate_modified_code(base_code, result.modified_code)

                if validation.valid:
                    return result.modified_code, True
                else:
                    print(f"Diff validation failed: {validation.error}")
            else:
                print(f"Diff application failed: {result.errors}")

    # Fallback to full rewrite
    print("Falling back to full code rewrite...")
    full_code = self.generate_strategy_code(base_code, improvement_proposal)
    return full_code, False
```

---

## Updated Seed Strategies

Add EVOLVE markers to existing strategies in `src/profit/strategies.py`:

### EMACrossover with EVOLVE Blocks

```python
class EMACrossover(Strategy):
    """EMA Crossover strategy with EVOLVE block markers."""

    # EVOLVE-BLOCK: indicator_params
    fast_ema = 50
    slow_ema = 200
    # END-EVOLVE-BLOCK

    def init(self):
        close = pd.Series(self.data.Close)
        # EVOLVE-BLOCK: signal_generation
        self.fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        self.slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        self.signal = self.fast > self.slow
        # END-EVOLVE-BLOCK

    def next(self):
        # EVOLVE-BLOCK: entry_logic
        if self.signal[-1] and not self.position:
            self.buy()
        # END-EVOLVE-BLOCK

        # EVOLVE-BLOCK: exit_logic
        elif not self.signal[-1] and self.position:
            self.position.close()
        # END-EVOLVE-BLOCK
```

### BollingerMeanReversion with EVOLVE Blocks

```python
class BollingerMeanReversion(Strategy):
    """Bollinger Bands mean reversion with EVOLVE blocks."""

    # EVOLVE-BLOCK: indicator_params
    period = 20
    num_std = 2.0
    # END-EVOLVE-BLOCK

    def init(self):
        close = pd.Series(self.data.Close)
        # EVOLVE-BLOCK: signal_generation
        self.sma = close.rolling(self.period).mean()
        self.std = close.rolling(self.period).std()
        self.upper = self.sma + self.num_std * self.std
        self.lower = self.sma - self.num_std * self.std
        # END-EVOLVE-BLOCK

    def next(self):
        price = self.data.Close[-1]

        # EVOLVE-BLOCK: entry_logic
        if price < self.lower[-1] and not self.position:
            self.buy()
        # END-EVOLVE-BLOCK

        # EVOLVE-BLOCK: exit_logic
        elif price > self.upper[-1] and self.position:
            self.position.close()
        # END-EVOLVE-BLOCK
```

---

## Evolver Integration

### Modified evolve_strategy()

Update `src/profit/evolver.py`:

```python
def evolve_strategy(
    self,
    strategy_class,
    train_data,
    val_data,
    max_iters=15,
    fold=1,
    use_inspirations=True,
    prefer_diffs=True  # NEW
):
    """Evolve strategy with optional diff-based mutations."""

    # ... existing initialization ...

    for gen in range(1, max_iters + 1):
        # ... parent selection and improvement generation ...

        # NEW: Try diff-based code generation first
        if prefer_diffs:
            new_code, used_diff = self.llm.generate_strategy_code_with_fallback(
                parent_code,
                improvement
            )
            if used_diff:
                print(f"Applied diff-based mutation")
            else:
                print(f"Used full code rewrite")
        else:
            new_code = self.llm.generate_strategy_code(parent_code, improvement)

        # ... rest of existing code (naming, compilation, evaluation) ...
```

---

## CLI Integration

Add to `src/profit/main.py`:

```python
parser.add_argument(
    '--no-diffs',
    action='store_true',
    help='Disable diff-based mutations (use full rewrites)'
)

# In main():
evolver = ProfitEvolver(
    llm_client,
    # ... other args ...
)

results = evolver.walk_forward_optimize(
    data,
    strategy_class,
    n_folds=args.folds,
    prefer_diffs=not args.no_diffs  # NEW
)
```

---

## Utility Script: Add EVOLVE Markers

Script to add EVOLVE markers to existing strategies:

```python
#!/usr/bin/env python3
"""Add EVOLVE block markers to existing strategy files."""

import re
import sys
from pathlib import Path

# Default block patterns for common strategy structures
DEFAULT_BLOCKS = {
    'signal_generation': (
        r'(self\.\w+\s*=.*(?:rolling|ewm|diff).*)',  # Indicator assignment
        r'\n\s*\n|\n\s*def '  # End at blank line or next method
    ),
    'entry_logic': (
        r'if.*not self\.position.*:.*\n.*self\.buy',
        r'\n'
    ),
    'exit_logic': (
        r'(?:elif|if).*self\.position.*:.*\n.*(?:close|sell)',
        r'\n'
    ),
}


def add_markers_to_file(filepath: Path, dry_run: bool = True):
    """Add EVOLVE markers to a strategy file."""
    content = filepath.read_text()

    # Skip if already has markers
    if '# EVOLVE-BLOCK:' in content:
        print(f"  {filepath.name}: Already has EVOLVE markers, skipping")
        return

    from profit.diff_utils import add_evolve_markers

    modified = add_evolve_markers(content, DEFAULT_BLOCKS)

    if modified != content:
        if dry_run:
            print(f"  {filepath.name}: Would add markers (dry run)")
        else:
            filepath.write_text(modified)
            print(f"  {filepath.name}: Added EVOLVE markers")
    else:
        print(f"  {filepath.name}: No patterns matched")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='Strategy files to process')
    parser.add_argument('--apply', action='store_true', help='Actually modify files')
    args = parser.parse_args()

    for filepath in args.files:
        path = Path(filepath)
        if path.exists():
            add_markers_to_file(path, dry_run=not args.apply)
        else:
            print(f"File not found: {filepath}")


if __name__ == "__main__":
    main()
```

---

## File Structure

```
src/profit/
├── __init__.py
├── strategies.py         # Modified: add EVOLVE markers to all strategies
├── llm_interface.py      # Modified: add generate_diff(), generate_strategy_code_with_fallback()
├── evolver.py            # Modified: prefer_diffs option
├── main.py               # Modified: --no-diffs argument
└── diff_utils.py         # NEW: all diff-related utilities

scripts/
└── add_evolve_markers.py # NEW: utility script
```

---

## Test Cases

### test_diff_utils.py

```python
import pytest
from profit.diff_utils import (
    extract_evolve_blocks,
    parse_diff_response,
    apply_diff,
    validate_modified_code,
    DiffBlock,
)


class TestExtractEvolveBlocks:
    def test_extracts_single_block(self):
        code = '''
class Strategy:
    # EVOLVE-BLOCK: signal
    def signal(self):
        return True
    # END-EVOLVE-BLOCK
'''
        blocks = extract_evolve_blocks(code)
        assert 'signal' in blocks
        assert 'def signal' in blocks['signal'].content

    def test_extracts_multiple_blocks(self):
        code = '''
class Strategy:
    # EVOLVE-BLOCK: entry
    def enter(self): pass
    # END-EVOLVE-BLOCK

    # EVOLVE-BLOCK: exit
    def exit(self): pass
    # END-EVOLVE-BLOCK
'''
        blocks = extract_evolve_blocks(code)
        assert len(blocks) == 2
        assert 'entry' in blocks
        assert 'exit' in blocks


class TestParseDiffResponse:
    def test_parses_single_diff(self):
        response = '''
<<<SEARCH block_name="signal">>>
return True
<<<REPLACE>>>
return False
<<<END>>>
'''
        diffs = parse_diff_response(response)
        assert len(diffs) == 1
        assert diffs[0].block_name == 'signal'
        assert 'True' in diffs[0].search
        assert 'False' in diffs[0].replace

    def test_parses_multiple_diffs(self):
        response = '''
<<<SEARCH block_name="entry">>>
buy()
<<<REPLACE>>>
buy(size=0.5)
<<<END>>>

<<<SEARCH block_name="exit">>>
sell()
<<<REPLACE>>>
close()
<<<END>>>
'''
        diffs = parse_diff_response(response)
        assert len(diffs) == 2


class TestApplyDiff:
    def test_applies_simple_diff(self):
        code = '''
class Strategy:
    # EVOLVE-BLOCK: signal
    def signal(self):
        return True
    # END-EVOLVE-BLOCK
'''
        diffs = [DiffBlock('signal', 'return True', 'return False')]
        result = apply_diff(code, diffs)

        assert result.success
        assert 'return False' in result.modified_code
        assert 'signal' in result.blocks_modified


class TestValidateModifiedCode:
    def test_valid_code_passes(self):
        original = 'class S:\n    def init(self): pass\n    def next(self): pass'
        modified = 'class S:\n    def init(self): pass\n    def next(self): return 1'

        result = validate_modified_code(original, modified)
        assert result.valid

    def test_syntax_error_fails(self):
        original = 'class S: pass'
        modified = 'class S: def broken('

        result = validate_modified_code(original, modified)
        assert not result.valid
        assert result.error_type == 'syntax'

    def test_missing_method_fails(self):
        original = 'class S:\n    def init(self): pass\n    def next(self): pass'
        modified = 'class S:\n    def init(self): pass'

        result = validate_modified_code(original, modified)
        assert not result.valid
        assert 'next' in result.error
```

---

## Deliverables

- [ ] `DiffBlock`, `EvolveBlock`, `ValidationResult` dataclasses
- [ ] `extract_evolve_blocks()` function
- [ ] `parse_diff_response()` function
- [ ] `apply_diff()` function
- [ ] `validate_modified_code()` function
- [ ] `has_evolve_blocks()` and `get_block_names()` helpers
- [ ] `generate_diff()` method in LLMClient
- [ ] `generate_strategy_code_with_fallback()` method in LLMClient
- [ ] Updated seed strategies with EVOLVE markers
- [ ] Evolver integration with `prefer_diffs` option
- [ ] CLI `--no-diffs` argument
- [ ] `add_evolve_markers.py` utility script
- [ ] Tests for all diff utilities
