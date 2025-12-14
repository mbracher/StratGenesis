"""Unit tests for LLM interface (with mocking)."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from profit.llm_interface import LLMClient


class TestLLMClientInit:
    """Test LLMClient initialization."""

    def test_default_openai(self):
        """Default provider should be OpenAI."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()
                assert client.provider == "openai"
                assert client.model == "gpt-4"

    def test_anthropic_provider(self):
        """Should initialize Anthropic client when specified."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("profit.llm_interface.anthropic") as mock_anthropic:
                mock_anthropic.Anthropic.return_value = Mock()
                client = LLMClient(provider="anthropic")
                assert client.provider == "anthropic"
                assert client.model == "claude-3-5-sonnet-20241022"

    def test_custom_model(self):
        """Should accept custom model specification."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient(model="gpt-3.5-turbo")
                assert client.model == "gpt-3.5-turbo"

    def test_unsupported_provider_raises(self):
        """Should raise ValueError for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClient(provider="invalid")

    def test_api_key_from_env(self):
        """Should read API key from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()
                assert client.openai_api_key == "env-key"

    def test_api_key_from_param(self):
        """Should prefer API key from parameter over env."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient(openai_api_key="param-key")
                assert client.openai_api_key == "param-key"


class TestGenerateImprovement:
    """Test improvement generation."""

    def test_returns_string(self):
        """Should return improvement proposal as string."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                with patch.object(client, "_chat") as mock_chat:
                    mock_chat.return_value = "Add a trailing stop-loss"

                    result = client.generate_improvement(
                        "class MyStrategy: pass", "AnnReturn=5.0%, Sharpe=0.5"
                    )

                    assert isinstance(result, str)
                    assert len(result) > 0
                    mock_chat.assert_called_once()

    def test_prompt_contains_code_and_metrics(self):
        """Prompt should include both strategy code and metrics."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                with patch.object(client, "_chat") as mock_chat:
                    mock_chat.return_value = "Improvement suggestion"

                    client.generate_improvement(
                        "class TestStrategy: pass", "Return=10%"
                    )

                    # Check the prompt argument
                    call_args = mock_chat.call_args
                    prompt = call_args[0][0]
                    assert "class TestStrategy: pass" in prompt
                    assert "Return=10%" in prompt


class TestGenerateStrategyCode:
    """Test strategy code generation."""

    def test_returns_code_string(self):
        """Should return valid Python code."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                with patch.object(client, "_chat") as mock_chat:
                    mock_chat.return_value = "class MyStrategy(Strategy): pass"

                    result = client.generate_strategy_code(
                        "class MyStrategy: pass", "Add trailing stop"
                    )

                    assert isinstance(result, str)
                    assert "class" in result

    def test_expects_code_flag(self):
        """Should pass expect_code=True to _chat."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                with patch.object(client, "_chat") as mock_chat:
                    mock_chat.return_value = "class MyStrategy(Strategy): pass"

                    client.generate_strategy_code("code", "improvement")

                    mock_chat.assert_called_once()
                    assert mock_chat.call_args[1].get("expect_code") is True


class TestFixCode:
    """Test code repair functionality."""

    def test_attempts_fix(self):
        """Should attempt to fix code given error traceback."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                with patch.object(client, "_chat") as mock_chat:
                    mock_chat.return_value = "class MyStrategy(Strategy): pass"

                    result = client.fix_code(
                        "class MyStrategy: syntax error",
                        "SyntaxError: invalid syntax",
                    )

                    assert isinstance(result, str)

    def test_prompt_contains_error_trace(self):
        """Prompt should include the error traceback."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                with patch.object(client, "_chat") as mock_chat:
                    mock_chat.return_value = "fixed code"

                    client.fix_code("broken code", "NameError: undefined")

                    call_args = mock_chat.call_args
                    prompt = call_args[0][0]
                    assert "NameError: undefined" in prompt


class TestCodeStripping:
    """Test markdown code fence stripping."""

    def test_strips_python_fences(self):
        """Should strip ```python ... ``` fences."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                text = "```python\nclass Foo: pass\n```"
                result = client._strip_markdown_fences(text)

                assert result == "class Foo: pass"

    def test_strips_plain_fences(self):
        """Should strip ``` ... ``` fences without language."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                text = "```\nclass Bar: pass\n```"
                result = client._strip_markdown_fences(text)

                assert result == "class Bar: pass"

    def test_handles_no_fences(self):
        """Should return text unchanged if no fences present."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                text = "class Baz: pass"
                result = client._strip_markdown_fences(text)

                assert result == "class Baz: pass"

    def test_strips_embedded_fences(self):
        """Should extract code from fences embedded in text."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                client = LLMClient()

                text = "Here is the code:\n```python\nclass Qux: pass\n```\nDone."
                result = client._strip_markdown_fences(text)

                assert result == "class Qux: pass"


class TestChatOpenAI:
    """Test _chat method with OpenAI."""

    def test_openai_chat_call(self):
        """Should call OpenAI API correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("profit.llm_interface.openai") as mock_openai:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "response text"
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.OpenAI.return_value = mock_client

                client = LLMClient()
                result = client._chat("test prompt")

                assert result == "response text"
                mock_client.chat.completions.create.assert_called_once()


class TestChatAnthropic:
    """Test _chat method with Anthropic."""

    def test_anthropic_chat_call(self):
        """Should call Anthropic API correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("profit.llm_interface.anthropic") as mock_anthropic:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "claude response"
                mock_client.messages.create.return_value = mock_response
                mock_anthropic.Anthropic.return_value = mock_client

                client = LLMClient(provider="anthropic")
                result = client._chat("test prompt")

                assert result == "claude response"
                mock_client.messages.create.assert_called_once()
