"""Tests for diff-based mutation utilities."""

import pytest

from profit.diff_utils import (
    DiffApplicationResult,
    DiffBlock,
    EvolveBlock,
    MatchMode,
    ValidationResult,
    apply_diff,
    compute_block_indentation,
    extract_evolve_blocks,
    get_block_names,
    has_evolve_blocks,
    has_nested_markers,
    match_search_in_block,
    normalize_code,
    parse_diff_response,
    reindent_replacement,
    validate_modified_code,
)


class TestExtractEvolveBlocks:
    """Tests for extract_evolve_blocks function."""

    def test_extracts_single_block(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: signal
    def signal(self):
        return True
    # END-EVOLVE-BLOCK
"""
        blocks = extract_evolve_blocks(code)
        assert "signal" in blocks
        assert "def signal" in blocks["signal"].content

    def test_extracts_multiple_blocks(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: entry
    def enter(self):
        pass
    # END-EVOLVE-BLOCK

    # EVOLVE-BLOCK: exit
    def exit(self):
        pass
    # END-EVOLVE-BLOCK
"""
        blocks = extract_evolve_blocks(code)
        assert len(blocks) == 2
        assert "entry" in blocks
        assert "exit" in blocks

    def test_preserves_character_offsets(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: params
    fast = 50
    slow = 200
    # END-EVOLVE-BLOCK
"""
        blocks = extract_evolve_blocks(code)
        block = blocks["params"]

        # Verify we can use offsets for splicing
        assert code[block.start_idx : block.end_idx] == block.full_text

    def test_handles_nested_indentation(self):
        code = """\
class Strategy:
    def init(self):
        # EVOLVE-BLOCK: signal
        self.fast = 50
        self.slow = 200
        # END-EVOLVE-BLOCK
"""
        blocks = extract_evolve_blocks(code)
        assert "signal" in blocks
        assert blocks["signal"].indentation == "        "

    def test_empty_code_returns_empty_dict(self):
        blocks = extract_evolve_blocks("")
        assert blocks == {}

    def test_code_without_markers_returns_empty_dict(self):
        code = "class Strategy:\n    pass"
        blocks = extract_evolve_blocks(code)
        assert blocks == {}

    def test_class_level_block_indentation(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: params
    fast_ema = 50
    slow_ema = 200
    # END-EVOLVE-BLOCK
"""
        blocks = extract_evolve_blocks(code)
        assert blocks["params"].indentation == "    "


class TestParseDiffResponse:
    """Tests for parse_diff_response function."""

    def test_parses_single_diff(self):
        response = """\
<<<SEARCH block_name="signal">>>
return True
<<<REPLACE>>>
return False
<<<END>>>
"""
        diffs = parse_diff_response(response)
        assert len(diffs) == 1
        assert diffs[0].block_name == "signal"
        assert "True" in diffs[0].search
        assert "False" in diffs[0].replace

    def test_parses_multiple_diffs(self):
        response = """\
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
"""
        diffs = parse_diff_response(response)
        assert len(diffs) == 2
        assert diffs[0].block_name == "entry"
        assert diffs[1].block_name == "exit"

    def test_preserves_raw_text(self):
        response = """\
<<<SEARCH block_name="code">>>
    indented line
<<<REPLACE>>>
    still indented
<<<END>>>
"""
        diffs = parse_diff_response(response)
        # Raw text should be preserved (not stripped)
        assert "    indented line\n" in diffs[0].search

    def test_ignores_code_fences(self):
        response = """\
Here's the diff:
```
<<<SEARCH block_name="signal">>>
old code
<<<REPLACE>>>
new code
<<<END>>>
```
"""
        diffs = parse_diff_response(response)
        assert len(diffs) == 1

    def test_empty_response_returns_empty_list(self):
        diffs = parse_diff_response("")
        assert diffs == []

    def test_response_without_diffs_returns_empty_list(self):
        response = "I don't know how to improve this strategy."
        diffs = parse_diff_response(response)
        assert diffs == []


class TestMatchModes:
    """Tests for match_search_in_block with different modes."""

    def test_strict_requires_exact_match(self):
        block = "    return True"
        search = "return True"

        # Strict matches literal substrings - "return True" IS in "    return True"
        count, matched, _ = match_search_in_block(block, search, MatchMode.STRICT)
        assert count == 1
        assert matched == "return True"

        # Full string also matches
        count, matched, _ = match_search_in_block(
            block, "    return True", MatchMode.STRICT
        )
        assert count == 1
        assert matched == "    return True"

        # Non-existent substring does not match
        count, _, _ = match_search_in_block(block, "return False", MatchMode.STRICT)
        assert count == 0

    def test_tolerant_ignores_whitespace_differences(self):
        block = "    return True\n"
        search = "return True"

        count, matched, offset = match_search_in_block(
            block, search, MatchMode.TOLERANT
        )
        assert count == 1
        assert "return True" in matched

    def test_tolerant_requires_uniqueness(self):
        block = """\
        if condition:
            return True
        else:
            return True
"""
        search = "return True"

        count, _, _ = match_search_in_block(block, search, MatchMode.TOLERANT)
        assert count == 2  # Ambiguous

    def test_no_match_returns_zero(self):
        block = "return False"
        search = "return True"

        count, matched, offset = match_search_in_block(
            block, search, MatchMode.TOLERANT
        )
        assert count == 0
        assert matched is None
        assert offset is None


class TestApplyDiff:
    """Tests for apply_diff function."""

    def test_applies_simple_diff(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: signal
    def signal(self):
        return True
    # END-EVOLVE-BLOCK
"""
        diffs = [DiffBlock("signal", "return True", "return False")]
        result = apply_diff(code, diffs)

        assert result.success
        assert "return False" in result.modified_code
        assert "signal" in result.blocks_modified

    def test_fails_on_ambiguous_match(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: logic
    if x:
        return True
    else:
        return True
    # END-EVOLVE-BLOCK
"""
        diffs = [DiffBlock("logic", "return True", "return False")]
        result = apply_diff(code, diffs)

        assert not result.success
        assert result.match_counts.get("logic", 0) == 2
        assert any("matched 2 times" in err for err in result.errors)

    def test_fails_on_no_match(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: signal
    return False
    # END-EVOLVE-BLOCK
"""
        diffs = [DiffBlock("signal", "return True", "return Maybe")]
        result = apply_diff(code, diffs)

        assert not result.success
        assert any("not found" in err for err in result.errors)

    def test_single_replacement_only(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: logic
    x = 1
    y = 1
    z = 1
    # END-EVOLVE-BLOCK
"""
        # This will match multiple times in TOLERANT mode
        diffs = [DiffBlock("logic", "= 1", "= 2")]
        result = apply_diff(code, diffs, MatchMode.TOLERANT)

        # Should fail due to ambiguity
        assert not result.success

    def test_preserves_surrounding_code(self):
        code = """\
# Header comment
class Strategy:
    # EVOLVE-BLOCK: signal
    return True
    # END-EVOLVE-BLOCK

    def other(self):
        pass
"""
        diffs = [DiffBlock("signal", "return True", "return False")]
        result = apply_diff(code, diffs)

        assert "# Header comment" in result.modified_code
        assert "def other(self):" in result.modified_code

    def test_handles_multiple_diffs(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: entry
    self.buy()
    # END-EVOLVE-BLOCK

    # EVOLVE-BLOCK: exit
    self.sell()
    # END-EVOLVE-BLOCK
"""
        diffs = [
            DiffBlock("entry", "self.buy()", "self.buy(size=0.5)"),
            DiffBlock("exit", "self.sell()", "self.position.close()"),
        ]
        result = apply_diff(code, diffs)

        assert result.success
        assert "self.buy(size=0.5)" in result.modified_code
        assert "self.position.close()" in result.modified_code
        assert len(result.blocks_modified) == 2

    def test_block_not_found_error(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: signal
    return True
    # END-EVOLVE-BLOCK
"""
        diffs = [DiffBlock("nonexistent", "x", "y")]
        result = apply_diff(code, diffs)

        assert not result.success
        assert any("not found" in err for err in result.errors)

    def test_no_change_is_failure(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: signal
    return True
    # END-EVOLVE-BLOCK
"""
        diffs = [DiffBlock("signal", "return True", "return True")]
        result = apply_diff(code, diffs)

        assert not result.success
        assert any("identical" in err for err in result.errors)


class TestIndentation:
    """Tests for indentation handling."""

    def test_compute_block_indentation_method_level(self):
        content = """\
        self.fast = 50
        self.slow = 200
"""
        indent = compute_block_indentation(content)
        assert indent == "        "  # 8 spaces

    def test_compute_block_indentation_class_level(self):
        content = """\
    fast_ema = 50
    slow_ema = 200
"""
        indent = compute_block_indentation(content)
        assert indent == "    "  # 4 spaces

    def test_compute_block_indentation_empty_content(self):
        indent = compute_block_indentation("")
        assert indent == ""

    def test_reindent_replacement(self):
        replace = "return True"
        target = "        "

        result = reindent_replacement(replace, target)
        assert result == "        return True"

    def test_reindent_multiline(self):
        replace = """\
if condition:
    return True
else:
    return False
"""
        target = "    "
        result = reindent_replacement(replace, target)

        lines = result.split("\n")
        for line in lines:
            if line.strip():
                assert line.startswith("    ")


class TestValidation:
    """Tests for validate_modified_code function."""

    def test_valid_code_passes(self):
        original = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        modified = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        self.x = 1
    def next(self):
        return 1
"""
        result = validate_modified_code(original, modified)
        assert result.valid

    def test_syntax_error_fails(self):
        original = "class S: pass"
        modified = "class S: def broken("

        result = validate_modified_code(original, modified)
        assert not result.valid
        assert result.error_type == "syntax"

    def test_missing_init_method_fails(self):
        original = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        modified = """\
from backtesting import Strategy

class S(Strategy):
    def next(self):
        pass
"""
        result = validate_modified_code(original, modified)
        assert not result.valid
        assert "init" in result.error

    def test_missing_next_method_fails(self):
        original = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        modified = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        pass
"""
        result = validate_modified_code(original, modified)
        assert not result.valid
        assert "next" in result.error

    def test_methods_must_be_in_strategy_class(self):
        original = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        # Methods exist but not in Strategy class
        modified = """\
from backtesting import Strategy

class S(Strategy):
    pass

def init():
    pass

def next():
    pass
"""
        result = validate_modified_code(original, modified)
        assert not result.valid
        assert result.error_type == "structure"

    def test_blocks_dangerous_patterns(self):
        original = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        modified = """\
from backtesting import Strategy
import os

class S(Strategy):
    def init(self):
        os.system('rm -rf /')
    def next(self):
        pass
"""
        result = validate_modified_code(original, modified)
        assert not result.valid
        assert result.error_type == "security"

    def test_blocks_network_imports(self):
        original = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        modified = """\
from backtesting import Strategy
import requests

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        result = validate_modified_code(original, modified)
        assert not result.valid
        assert result.error_type == "security"
        assert "requests" in result.error

    def test_blocks_file_write_imports(self):
        original = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        modified = """\
from backtesting import Strategy
import shutil

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        result = validate_modified_code(original, modified)
        assert not result.valid
        assert result.error_type == "security"

    def test_allows_numpy_pandas(self):
        original = """\
from backtesting import Strategy

class S(Strategy):
    def init(self):
        pass
    def next(self):
        pass
"""
        modified = """\
from backtesting import Strategy
import numpy as np
import pandas as pd

class S(Strategy):
    def init(self):
        self.data = np.array([1,2,3])
    def next(self):
        pass
"""
        result = validate_modified_code(original, modified)
        assert result.valid

    def test_no_strategy_class_fails(self):
        original = "class S: pass"
        modified = "class NotStrategy:\n    def init(self): pass\n    def next(self): pass"

        result = validate_modified_code(original, modified)
        assert not result.valid
        assert result.error_type == "structure"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_nested_markers_detected(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: outer
    # EVOLVE-BLOCK: inner
    x = 1
    # END-EVOLVE-BLOCK
    # END-EVOLVE-BLOCK
"""
        assert has_nested_markers(code)

    def test_no_nested_markers_clean(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: one
    x = 1
    # END-EVOLVE-BLOCK
    # EVOLVE-BLOCK: two
    y = 2
    # END-EVOLVE-BLOCK
"""
        assert not has_nested_markers(code)

    def test_has_evolve_blocks_true(self):
        code = "# EVOLVE-BLOCK: test\ncode\n# END-EVOLVE-BLOCK"
        assert has_evolve_blocks(code)

    def test_has_evolve_blocks_false(self):
        code = "class Strategy: pass"
        assert not has_evolve_blocks(code)

    def test_get_block_names(self):
        code = """\
class Strategy:
    # EVOLVE-BLOCK: alpha
    x = 1
    # END-EVOLVE-BLOCK
    # EVOLVE-BLOCK: beta
    y = 2
    # END-EVOLVE-BLOCK
"""
        names = get_block_names(code)
        assert set(names) == {"alpha", "beta"}


class TestNormalizeCode:
    """Tests for normalize_code function."""

    def test_strips_whitespace(self):
        code = "    return True    "
        assert normalize_code(code) == "return True"

    def test_removes_empty_lines(self):
        code = "line1\n\nline2"
        assert normalize_code(code) == "line1\nline2"

    def test_preserves_line_order(self):
        code = "a\nb\nc"
        assert normalize_code(code) == "a\nb\nc"
