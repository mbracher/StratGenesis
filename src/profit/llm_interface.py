"""LLM interface for strategy mutation.

This module contains the LLMClient class for OpenAI/Anthropic integration.
It implements the two-LLM architecture from the ProFiT paper:
- LLM A (Analyst): Analyzes strategy and proposes improvements
- LLM B (Coder): Applies improvements to generate new code
"""

import os
import re

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


class LLMClient:
    """Interface to LLMs (OpenAI GPT or Anthropic Claude) for generating and modifying strategy code."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ):
        """Initialize the LLM client.

        Args:
            provider: "openai" or "anthropic"
            model: Model name (e.g., "gpt-4" or "claude-3-opus-20240229").
                   If None, uses a default for each provider.
            openai_api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            anthropic_api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        """
        self.provider = provider

        # Use API keys from environment if not provided
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        # Default models
        if model:
            self.model = model
        else:
            self.model = "gpt-4" if provider == "openai" else "claude-3-5-sonnet-20241022"

        # Initialize clients
        if provider == "anthropic":
            if anthropic is None:
                raise ImportError(
                    "anthropic SDK not installed. Install via `uv add anthropic`"
                )
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            self.openai_client = None
        elif provider == "openai":
            if openai is None:
                raise ImportError(
                    "openai SDK not installed. Install via `uv add openai`"
                )
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            self.anthropic_client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_improvement(self, strategy_code: str, metrics_summary: str) -> str:
        """Ask the LLM to analyze the strategy code and performance metrics,
        and suggest an improvement (natural language description).

        This is LLM A in the ProFiT loop - the analyst role.

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
        return self._chat(prompt, expect_code=False)

    def generate_strategy_code(self, base_code: str, improvement_proposal: str) -> str:
        """Ask the LLM to apply the proposed improvement to the base strategy code.

        This is LLM B in the ProFiT loop - the coder role.

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
        return self._chat(prompt, expect_code=True)

    def fix_code(self, base_code: str, error_trace: str) -> str:
        """If the generated code fails, prompt the LLM to fix the error given the traceback.

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
        return self._chat(prompt, expect_code=True)

    def _chat(self, prompt: str, expect_code: bool = False) -> str:
        """Internal helper to send chat prompt to the chosen provider and return response text.

        Args:
            prompt: The prompt to send to the LLM
            expect_code: If True, strip markdown code fences from response

        Returns:
            The LLM's response text
        """
        if self.provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
        elif self.provider == "anthropic":
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

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
