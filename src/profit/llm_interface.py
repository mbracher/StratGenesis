"""LLM interface for strategy mutation.

This module contains the LLMClient class for OpenAI/Anthropic integration.
It implements the two-LLM architecture from the ProFiT paper:
- LLM A (Analyst): Analyzes strategy and proposes improvements
- LLM B (Coder): Applies improvements to generate new code

Supports using different providers/models for each role (dual-model configuration).

Phase 13B additions:
- generate_improvement_with_inspirations(): Uses inspiration strategies for richer prompts
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, List

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
