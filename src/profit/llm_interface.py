"""LLM interface for strategy mutation.

This module contains the LLMClient class for OpenAI/Anthropic integration.
It implements the two-LLM architecture from the ProFiT paper:
- LLM A (Analyst): Analyzes strategy and proposes improvements
- LLM B (Coder): Applies improvements to generate new code

Supports using different providers/models for each role (dual-model configuration).

Phase 13B additions:
- generate_improvement_with_inspirations(): Uses inspiration strategies for richer prompts

Phase 14 additions:
- generate_diff(): Generates SEARCH/REPLACE diff blocks for targeted mutations
- fix_diff(): Re-prompts LLM to fix failed diff operations
- generate_strategy_code_with_fallback(): Tries diffs first, falls back to full rewrite
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from profit.program_db import StrategyRecord

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4",
    "anthropic": "claude-sonnet-4-20250514",
}


class LLMClient:
    """Interface to LLMs for generating and modifying strategy code.

    Supports using different providers/models for the analyst and coder roles.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        analyst_provider: str | None = None,
        analyst_model: str | None = None,
        coder_provider: str | None = None,
        coder_model: str | None = None,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ):
        """Initialize the LLM client.

        Args:
            provider: Default provider for both roles ("openai" or "anthropic").
                      Used if role-specific providers not set. Defaults to "openai".
            model: Default model for both roles. Used if role-specific models not set.
            analyst_provider: Provider for LLM A (analysis/improvement suggestions).
            analyst_model: Model for LLM A.
            coder_provider: Provider for LLM B (code generation/fixing).
            coder_model: Model for LLM B.
            openai_api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            anthropic_api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

        Examples:
            # Single provider (backward compatible)
            client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")

            # Dual-model: OpenAI for analysis, Anthropic for coding
            client = LLMClient(
                analyst_provider="openai",
                analyst_model="gpt-4",
                coder_provider="anthropic",
                coder_model="claude-sonnet-4-20250514",
            )
        """
        # Store API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        # Resolve default provider
        default_provider = provider or "openai"

        # Resolve analyst configuration
        self.analyst_provider = analyst_provider or default_provider
        self.analyst_model = (
            analyst_model or model or DEFAULT_MODELS.get(self.analyst_provider, "gpt-4")
        )

        # Resolve coder configuration
        self.coder_provider = coder_provider or default_provider
        self.coder_model = (
            coder_model or model or DEFAULT_MODELS.get(self.coder_provider, "gpt-4")
        )

        # Legacy attributes for backward compatibility
        self.provider = default_provider
        self.model = model or DEFAULT_MODELS.get(default_provider, "gpt-4")

        # Initialize API clients for providers in use
        self._clients: dict = {}
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize API clients for providers in use."""
        providers_needed = {self.analyst_provider, self.coder_provider}

        if "openai" in providers_needed:
            if openai is None:
                raise ImportError(
                    "openai SDK not installed. Install via `uv add openai`"
                )
            self._clients["openai"] = openai.OpenAI(api_key=self.openai_api_key)

        if "anthropic" in providers_needed:
            if anthropic is None:
                raise ImportError(
                    "anthropic SDK not installed. Install via `uv add anthropic`"
                )
            self._clients["anthropic"] = anthropic.Anthropic(
                api_key=self.anthropic_api_key
            )

    def generate_improvement(self, strategy_code: str, metrics_summary: str) -> str:
        """Ask the LLM to analyze the strategy code and performance metrics,
        and suggest an improvement (natural language description).

        This is LLM A in the ProFiT loop - the analyst role.
        Uses analyst_provider and analyst_model configuration.

        Args:
            strategy_code: The current strategy's Python code
            metrics_summary: Performance metrics (e.g., "Return: 15%, Sharpe: 0.8")

        Returns:
            Natural language improvement proposal
        """
        prompt = (
            "You are an expert trading strategy coach. I will provide you with the code of a trading strategy "
            "and its recent performance metrics. Identify the strategy's weaknesses or areas for improvement, "
            "and propose a specific change or addition to the strategy logic that could improve its performance.\n\n"
            "Strategy Code:\n"
            "```python\n" + strategy_code + "\n```\n"
            f"Performance Summary: {metrics_summary}\n\n"
            "Please suggest one concrete improvement (in a brief bullet or sentence)."
        )
        return self._chat(
            prompt,
            provider=self.analyst_provider,
            model=self.analyst_model,
            expect_code=False,
        )

    def generate_improvement_with_inspirations(
        self,
        strategy_code: str,
        metrics_summary: str,
        inspirations: List["StrategyRecord"],
        max_tokens_per_inspiration: int = 500,
    ) -> str:
        """Generate improvement proposal using inspiration from other strategies.

        This is a Phase 13B enhancement that implements AlphaEvolve's key insight:
        providing examples of successful strategies helps the LLM generate better mutations.

        Includes:
        - Multiple metrics (not just return) to avoid overfitting
        - Code excerpts of signal/entry/exit logic (next() method)
        - Diversity across inspirations

        Args:
            strategy_code: The current strategy's Python code.
            metrics_summary: Performance metrics of current strategy.
            inspirations: List of StrategyRecord objects for inspiration.
            max_tokens_per_inspiration: Max code excerpt length per inspiration.

        Returns:
            Natural language improvement proposal.
        """
        # Build inspiration context with code excerpts and multi-metric info
        inspiration_text = ""
        if inspirations:
            inspiration_text = "\n\nHere are successful strategies for inspiration:\n"
            for i, insp in enumerate(inspirations, 1):
                # Multi-metric summary
                metrics = insp.metrics
                ann_return = metrics.get("ann_return", "N/A")
                sharpe = metrics.get("sharpe", "N/A")
                max_dd = metrics.get("max_drawdown", "N/A")
                trade_count = metrics.get("trade_count", "N/A")
                win_rate = metrics.get("win_rate", "N/A")

                # Format metrics with type checking
                ann_str = f"{ann_return:.1f}%" if isinstance(ann_return, (int, float)) else "N/A"
                sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "N/A"
                dd_str = f"{max_dd:.1f}%" if isinstance(max_dd, (int, float)) else "N/A"
                trades_str = str(int(trade_count)) if isinstance(trade_count, (int, float)) else "N/A"
                wr_str = f"{win_rate:.0%}" if isinstance(win_rate, (int, float)) else "N/A"

                metrics_str = (
                    f"Return={ann_str}, Sharpe={sharpe_str}, MaxDD={dd_str}, "
                    f"Trades={trades_str}, WinRate={wr_str}"
                )

                # Code excerpt (next() method or truncated code)
                code_excerpt = (
                    insp.next_method_excerpt
                    if insp.next_method_excerpt
                    else insp.code[:max_tokens_per_inspiration]
                )

                # Tags for context
                tags_str = ", ".join(insp.tags) if insp.tags else "unclassified"

                inspiration_text += f"""
--- Inspiration {i}: {insp.class_name} ---
Performance: {metrics_str}
Tags: {tags_str}
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

        return self._chat(
            prompt,
            provider=self.analyst_provider,
            model=self.analyst_model,
            expect_code=False,
        )

    def generate_strategy_code(self, base_code: str, improvement_proposal: str) -> str:
        """Ask the LLM to apply the proposed improvement to the base strategy code.

        This is LLM B in the ProFiT loop - the coder role.
        Uses coder_provider and coder_model configuration.

        Args:
            base_code: The original strategy Python code
            improvement_proposal: Natural language description of improvement to apply

        Returns:
            Modified strategy code as a string
        """
        prompt = (
            "You are a coding assistant specialized in trading strategies. I will provide a base strategy code "
            "and an improvement idea. Your task is to modify the strategy code to implement the improvement. "
            "The code must remain a valid subclass of backtesting.Strategy and retain previous functionality unless changed. "
            "Provide only the full modified code without any explanations.\n\n"
            "Base Strategy Code:\n"
            "```python\n" + base_code + "\n```\n"
            "Improvement to implement: " + improvement_proposal + "\n\n"
            "Now output the full updated strategy code (only code, no comments or explanation)."
        )
        return self._chat(
            prompt,
            provider=self.coder_provider,
            model=self.coder_model,
            expect_code=True,
        )

    def fix_code(self, base_code: str, error_trace: str) -> str:
        """If the generated code fails, prompt the LLM to fix the error given the traceback.

        Uses coder_provider and coder_model configuration.

        Args:
            base_code: The code that failed to compile/run
            error_trace: The error traceback from the failed execution

        Returns:
            Fixed strategy code as a string
        """
        prompt = (
            "The following trading strategy code did not compile or run correctly. Error trace:\n"
            f"{error_trace}\n"
            "Please fix the code accordingly. Provide only the corrected code.\n\n"
            "Code with error:\n"
            "```python\n" + base_code + "\n```\n"
            "Now output the fixed code."
        )
        return self._chat(
            prompt,
            provider=self.coder_provider,
            model=self.coder_model,
            expect_code=True,
        )

    def _chat(
        self,
        prompt: str,
        provider: str,
        model: str,
        expect_code: bool = False,
    ) -> str:
        """Internal helper to send chat prompt to the specified provider/model.

        Args:
            prompt: The prompt to send to the LLM
            provider: Which provider to use ("openai" or "anthropic")
            model: Which model to use
            expect_code: If True, strip markdown code fences from response

        Returns:
            The LLM's response text
        """
        if provider == "openai":
            client = self._clients["openai"]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
        elif provider == "anthropic":
            client = self._clients["anthropic"]
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # If expecting code, strip markdown fences
        if expect_code:
            text = self._strip_markdown_fences(text)

        return text

    def _strip_markdown_fences(self, text: str) -> str:
        """Strip markdown code fences from text.

        Args:
            text: Text potentially containing markdown code fences

        Returns:
            Text with code fences removed
        """
        text = text.strip()

        # Match ```python or ``` at start and ``` at end
        pattern = r"^```(?:python)?\s*\n?(.*?)\n?```$"
        match = re.match(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Also handle case where there's text before/after code block
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return text

    # =========================================================================
    # Phase 14: Diff-based mutations
    # =========================================================================

    def generate_diff(
        self,
        strategy_code: str,
        improvement_proposal: str,
        available_blocks: List[str],
    ) -> str:
        """Generate SEARCH/REPLACE diff blocks for targeted mutations.

        Instead of rewriting the entire strategy, outputs surgical diffs
        targeting specific EVOLVE blocks.

        Args:
            strategy_code: Current strategy code with EVOLVE markers.
            improvement_proposal: The improvement to implement.
            available_blocks: Names of EVOLVE blocks that can be modified.

        Returns:
            LLM response containing diff blocks in the specified format.
        """
        blocks_list = ", ".join(available_blocks)

        prompt = f"""You are a precise code editor. Your task is to implement a strategy improvement
using surgical SEARCH/REPLACE diffs.

CURRENT STRATEGY CODE:
```python
{strategy_code}
```

IMPROVEMENT TO IMPLEMENT:
{improvement_proposal}

AVAILABLE BLOCKS FOR MODIFICATION:
{blocks_list}

OUTPUT FORMAT:
For each change, output a diff block in this exact format:

<<<SEARCH block_name="BLOCK_NAME">>>
exact code to find (copy from the strategy, be specific enough to be unique)
<<<REPLACE>>>
new code to replace it with
<<<END>>>

RULES:
1. Only modify code within the EVOLVE blocks listed above
2. The SEARCH section should contain enough context to be unique within the block
3. Include multiple lines in SEARCH if needed for uniqueness
4. Keep changes minimal - only change what's necessary for the improvement
5. Preserve the overall structure and indentation style
6. You can output multiple diff blocks for different blocks
7. If no changes are needed to a block, don't include it
8. Match tolerant - whitespace is flexible but content must be unique

OUTPUT YOUR DIFFS:
"""

        return self._chat(
            prompt,
            provider=self.coder_provider,
            model=self.coder_model,
            expect_code=False,  # Don't strip fences, we parse diff format
        )

    def fix_diff(
        self,
        original_code: str,
        failed_diff: str,
        error_message: str,
        block_content: str,
        block_name: str,
    ) -> str:
        """Re-prompt LLM to fix a failed diff operation.

        Called when diff parsing or application fails.

        Args:
            original_code: The full strategy code.
            failed_diff: The diff that failed to apply.
            error_message: Description of what went wrong.
            block_content: The exact content of the target block.
            block_name: Name of the target block.

        Returns:
            New diff response with corrected SEARCH/REPLACE.
        """
        prompt = f"""Your previous diff failed to apply. Here's what went wrong:

ERROR: {error_message}

The SEARCH pattern must exactly match code in the target block.

TARGET BLOCK '{block_name}' CONTENT:
```
{block_content}
```

YOUR FAILED DIFF:
{failed_diff}

Please try again. Output a corrected diff that will match the exact code in the target block.
Include enough context lines (2-3 lines before and after) to ensure uniqueness.

CORRECTED DIFF:
"""

        return self._chat(
            prompt,
            provider=self.coder_provider,
            model=self.coder_model,
            expect_code=False,
        )

    def generate_strategy_code_with_fallback(
        self,
        base_code: str,
        improvement_proposal: str,
        max_diff_attempts: int = 3,
        match_mode: str = "tolerant",
    ) -> Tuple[str, bool, Optional[str]]:
        """Generate modified strategy code, preferring diffs but falling back to full rewrite.

        Implements the diff-first approach with retry loop before fallback.

        Args:
            base_code: Original strategy code.
            improvement_proposal: The improvement to implement.
            max_diff_attempts: Number of diff attempts before fallback.
            match_mode: "strict" or "tolerant" for diff matching.

        Returns:
            Tuple of (modified_code, used_diff, raw_diff_text):
            - modified_code: The new strategy code
            - used_diff: True if diff was used, False if full rewrite
            - raw_diff_text: The raw diff response (or None if full rewrite)
        """
        from profit.diff_utils import (
            MatchMode,
            apply_diff,
            extract_evolve_blocks,
            get_block_names,
            has_evolve_blocks,
            parse_diff_response,
            validate_modified_code,
        )

        # Check if code has EVOLVE blocks
        if not has_evolve_blocks(base_code):
            # No EVOLVE blocks - fall back to full rewrite
            new_code = self.generate_strategy_code(base_code, improvement_proposal)
            return new_code, False, None

        available_blocks = get_block_names(base_code)
        evolve_blocks = extract_evolve_blocks(base_code)

        # Convert match_mode string to enum
        mode = MatchMode.STRICT if match_mode == "strict" else MatchMode.TOLERANT

        last_diff_response = None
        last_error = None

        for attempt in range(1, max_diff_attempts + 1):
            # Generate diff (or fix previous failed diff)
            if attempt == 1:
                diff_response = self.generate_diff(
                    base_code, improvement_proposal, available_blocks
                )
            else:
                # Find the block that failed and retry with fix_diff
                if last_error and last_error.get("block_name"):
                    block_name = last_error["block_name"]
                    block_content = evolve_blocks.get(block_name, {})
                    if hasattr(block_content, "content"):
                        block_content = block_content.content
                    else:
                        block_content = str(block_content)

                    diff_response = self.fix_diff(
                        base_code,
                        last_diff_response or "",
                        last_error.get("message", "Diff failed"),
                        block_content,
                        block_name,
                    )
                else:
                    # Generic retry
                    diff_response = self.generate_diff(
                        base_code, improvement_proposal, available_blocks
                    )

            last_diff_response = diff_response

            # Parse diff blocks
            diff_blocks = parse_diff_response(diff_response)

            if not diff_blocks:
                last_error = {"message": "No valid diff blocks found in response"}
                continue

            # Apply diffs
            result = apply_diff(base_code, diff_blocks, mode)

            if not result.success:
                # Record error for retry
                error_msg = result.errors[0] if result.errors else "Unknown error"
                # Try to identify which block failed
                failed_block = None
                for block_name, count in result.match_counts.items():
                    if count != 1:
                        failed_block = block_name
                        break

                last_error = {
                    "message": error_msg,
                    "block_name": failed_block,
                }
                continue

            # Validate the modified code
            validation = validate_modified_code(base_code, result.modified_code)

            if not validation.valid:
                last_error = {
                    "message": f"Validation failed: {validation.error}",
                    "block_name": result.blocks_modified[0] if result.blocks_modified else None,
                }
                continue

            # Success
            return result.modified_code, True, diff_response

        # All attempts failed - fall back to full rewrite
        new_code = self.generate_strategy_code(base_code, improvement_proposal)
        return new_code, False, None
