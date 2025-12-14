# Phase 12: Dual-Model LLM Configuration

## Objective

Enable using different LLM providers and models for the Analyst (LLM A) and Coder (LLM B) roles in the evolutionary loop. This allows optimizing each role independently—using models better suited for analysis versus code generation.

From user requirements:

> The evolving strategies take a long time to compile correctly. Prefer using a strong coding model (e.g., Claude Sonnet 4.5) for code generation and a reasoning-focused model (e.g., GPT-5.2) for analysis and improvement suggestions.

---

## Available Models (December 2025)

### OpenAI Models

| Model ID | Description | Best For |
|----------|-------------|----------|
| `gpt-5.2` | Latest flagship (Instant/Thinking/Pro) | Analysis, reasoning |
| `gpt-5.2-thinking` | Extended reasoning variant | Complex analysis |
| `gpt-5.1` | Balanced intelligence and speed | General purpose |
| `gpt-5` | Standard/Mini/Nano variants | Cost-effective tasks |
| `gpt-5-codex` | Optimized for agentic coding | Code generation |
| `gpt-4.1` | Previous gen, 1M context window | Long context tasks |

### Anthropic Models

| Model ID | Description | Best For |
|----------|-------------|----------|
| `claude-opus-4.5` | Latest flagship (Nov 2025) | Complex reasoning |
| `claude-sonnet-4.5` | Best coding/agents (Sep 2025) | Code generation |
| `claude-haiku-4.5` | Fast/cheap (Oct 2025) | Quick iterations |
| `claude-opus-4.1` | Agentic tasks focused | Multi-step tasks |
| `claude-sonnet-4` | Previous gen Sonnet | Balanced tasks |

### Recommended Configurations

| Use Case | Analyst (LLM A) | Coder (LLM B) |
|----------|-----------------|---------------|
| Best quality | `gpt-5.2-thinking` | `claude-sonnet-4.5` |
| Balanced | `gpt-5.1` | `claude-sonnet-4.5` |
| Cost-effective | `gpt-5` | `claude-haiku-4.5` |
| OpenAI only | `gpt-5.2` | `gpt-5-codex` |
| Anthropic only | `claude-opus-4.5` | `claude-sonnet-4.5` |

---

## LLMClient Class Updates

### New Constructor Signature

```python
class LLMClient:
    """Interface to LLMs for generating and modifying strategy code.

    Supports using different providers/models for the analyst and coder roles.
    """

    def __init__(
        self,
        # Legacy single-provider mode (backward compatible)
        provider: str | None = None,
        model: str | None = None,
        # New dual-model configuration
        analyst_provider: str | None = None,
        analyst_model: str | None = None,
        coder_provider: str | None = None,
        coder_model: str | None = None,
        # API keys
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ):
        """Initialize the LLM client.

        Args:
            provider: Default provider for both roles ("openai" or "anthropic").
                      Used if role-specific providers not set.
            model: Default model for both roles. Used if role-specific models not set.
            analyst_provider: Provider for LLM A (analysis/improvement suggestions).
            analyst_model: Model for LLM A.
            coder_provider: Provider for LLM B (code generation/fixing).
            coder_model: Model for LLM B.
            openai_api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            anthropic_api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

        Examples:
            # Single provider (backward compatible)
            client = LLMClient(provider="anthropic", model="claude-sonnet-4.5")

            # Dual-model: OpenAI for analysis, Anthropic for coding
            client = LLMClient(
                analyst_provider="openai",
                analyst_model="gpt-5.2",
                coder_provider="anthropic",
                coder_model="claude-sonnet-4.5",
            )
        """
```

### Internal Configuration

```python
def __init__(self, ...):
    # Store API keys
    self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    # Resolve analyst configuration
    self.analyst_provider = analyst_provider or provider or "openai"
    self.analyst_model = analyst_model or model or self._default_model(self.analyst_provider)

    # Resolve coder configuration
    self.coder_provider = coder_provider or provider or "openai"
    self.coder_model = coder_model or model or self._default_model(self.coder_provider)

    # Initialize clients for each provider in use
    self._clients = {}
    self._init_clients()

def _default_model(self, provider: str) -> str:
    """Return default model for a provider."""
    defaults = {
        "openai": "gpt-5.1",
        "anthropic": "claude-sonnet-4.5",
    }
    return defaults.get(provider, "gpt-5.1")

def _init_clients(self):
    """Initialize API clients for providers in use."""
    providers_needed = {self.analyst_provider, self.coder_provider}

    if "openai" in providers_needed:
        if openai is None:
            raise ImportError("openai SDK not installed. Install via `uv add openai`")
        self._clients["openai"] = openai.OpenAI(api_key=self.openai_api_key)

    if "anthropic" in providers_needed:
        if anthropic is None:
            raise ImportError("anthropic SDK not installed. Install via `uv add anthropic`")
        self._clients["anthropic"] = anthropic.Anthropic(api_key=self.anthropic_api_key)
```

---

## Updated Methods

### generate_improvement() - Uses Analyst Config

```python
def generate_improvement(self, strategy_code: str, metrics_summary: str) -> str:
    """Ask the LLM to analyze the strategy and suggest an improvement.

    This is LLM A in the ProFiT loop - the analyst role.
    Uses analyst_provider and analyst_model configuration.
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
```

### generate_strategy_code() - Uses Coder Config

```python
def generate_strategy_code(self, base_code: str, improvement_proposal: str) -> str:
    """Ask the LLM to apply the proposed improvement to the base strategy code.

    This is LLM B in the ProFiT loop - the coder role.
    Uses coder_provider and coder_model configuration.
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
```

### fix_code() - Uses Coder Config

```python
def fix_code(self, base_code: str, error_trace: str) -> str:
    """Prompt the LLM to fix code errors given the traceback.

    Uses coder_provider and coder_model configuration.
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
```

---

## Updated _chat() Method

```python
def _chat(
    self,
    prompt: str,
    provider: str,
    model: str,
    expect_code: bool = False,
) -> str:
    """Send chat prompt to the specified provider/model and return response.

    Args:
        prompt: The prompt to send to the LLM.
        provider: Which provider to use ("openai" or "anthropic").
        model: Which model to use.
        expect_code: If True, strip markdown code fences from response.

    Returns:
        The LLM's response text.
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

    if expect_code:
        text = self._strip_markdown_fences(text)

    return text
```

---

## CLI Updates (main.py)

### New Arguments

```python
# LLM configuration group
llm_group = parser.add_argument_group("LLM Configuration")

llm_group.add_argument(
    "--provider",
    choices=["openai", "anthropic"],
    default="openai",
    help="Default LLM provider for both roles (default: openai)",
)
llm_group.add_argument(
    "--model",
    default=None,
    help="Default model for both roles (uses provider default if not set)",
)

# Role-specific configuration
llm_group.add_argument(
    "--analyst-provider",
    choices=["openai", "anthropic"],
    default=None,
    help="LLM provider for analysis/improvements (overrides --provider)",
)
llm_group.add_argument(
    "--analyst-model",
    default=None,
    help="Model for analysis/improvements (overrides --model)",
)
llm_group.add_argument(
    "--coder-provider",
    choices=["openai", "anthropic"],
    default=None,
    help="LLM provider for code generation (overrides --provider)",
)
llm_group.add_argument(
    "--coder-model",
    default=None,
    help="Model for code generation (overrides --model)",
)
```

### Updated Client Initialization

```python
# Create LLM client with dual-model support
llm_client = LLMClient(
    provider=args.provider,
    model=args.model,
    analyst_provider=args.analyst_provider,
    analyst_model=args.analyst_model,
    coder_provider=args.coder_provider,
    coder_model=args.coder_model,
)

# Log configuration
print(f"Analyst: {llm_client.analyst_provider}/{llm_client.analyst_model}")
print(f"Coder: {llm_client.coder_provider}/{llm_client.coder_model}")
```

---

## Example Usage

### CLI Examples

```bash
# Single provider (backward compatible)
uv run python -m profit.main --data btc.csv --strategy EMACrossover --provider anthropic

# Dual-model: GPT-5.2 for analysis, Claude Sonnet 4.5 for coding
uv run python -m profit.main --data btc.csv --strategy EMACrossover \
    --analyst-provider openai --analyst-model gpt-5.2 \
    --coder-provider anthropic --coder-model claude-sonnet-4.5

# Cost-effective: GPT-5 for analysis, Claude Haiku for coding
uv run python -m profit.main --data btc.csv --strategy EMACrossover \
    --analyst-provider openai --analyst-model gpt-5 \
    --coder-provider anthropic --coder-model claude-haiku-4.5
```

### Programmatic Usage

```python
from profit.llm_interface import LLMClient
from profit.evolver import ProfitEvolver

# Dual-model configuration
llm = LLMClient(
    analyst_provider="openai",
    analyst_model="gpt-5.2",
    coder_provider="anthropic",
    coder_model="claude-sonnet-4.5",
)

evolver = ProfitEvolver(llm_client=llm)
results = evolver.walk_forward_optimize(data, EMACrossover)
```

---

## Persistence Updates

Update `StrategyPersister.start_run()` to record dual-model configuration:

```python
def start_run(self, seed_strategy_name: str, llm_client: LLMClient) -> Path:
    """Initialize a new run directory.

    Args:
        seed_strategy_name: Name of the seed strategy being evolved.
        llm_client: The LLMClient instance (contains provider/model info).
    """
    self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.run_dir = self.output_dir / f"run_{self.run_id}"
    self.run_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "run_id": self.run_id,
        "started_at": datetime.now().isoformat(),
        "seed_strategy": seed_strategy_name,
        "llm_config": {
            "analyst": {
                "provider": llm_client.analyst_provider,
                "model": llm_client.analyst_model,
            },
            "coder": {
                "provider": llm_client.coder_provider,
                "model": llm_client.coder_model,
            },
        },
        "folds": [],
    }
    self._write_json(self.run_dir / "run_summary.json", run_info)

    return self.run_dir
```

---

## Backward Compatibility

The implementation maintains full backward compatibility:

```python
# These are equivalent:
LLMClient(provider="openai", model="gpt-5.1")
LLMClient(
    analyst_provider="openai", analyst_model="gpt-5.1",
    coder_provider="openai", coder_model="gpt-5.1",
)

# Legacy single-provider still works
client = LLMClient(provider="anthropic")  # Uses default model for both roles
```

---

## Testing

### New Test Cases

```python
def test_dual_model_initialization():
    """Test LLMClient with different providers for each role."""
    client = LLMClient(
        analyst_provider="openai",
        analyst_model="gpt-5.1",
        coder_provider="anthropic",
        coder_model="claude-sonnet-4.5",
    )
    assert client.analyst_provider == "openai"
    assert client.analyst_model == "gpt-5.1"
    assert client.coder_provider == "anthropic"
    assert client.coder_model == "claude-sonnet-4.5"


def test_backward_compatibility():
    """Test that single provider/model still works."""
    client = LLMClient(provider="openai", model="gpt-5.1")
    assert client.analyst_provider == "openai"
    assert client.analyst_model == "gpt-5.1"
    assert client.coder_provider == "openai"
    assert client.coder_model == "gpt-5.1"


def test_default_models():
    """Test default model selection per provider."""
    client = LLMClient(
        analyst_provider="openai",
        coder_provider="anthropic",
    )
    assert client.analyst_model == "gpt-5.1"  # OpenAI default
    assert client.coder_model == "claude-sonnet-4.5"  # Anthropic default
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/profit/llm_interface.py` | Update `LLMClient` class with dual-model support |
| `src/profit/main.py` | Add CLI arguments for role-specific configuration |
| `src/profit/evolver.py` | Update `start_run()` call to pass full llm_client |
| `tests/test_llm_interface.py` | Add tests for dual-model configuration |

---

## Deliverables

- [ ] Updated `LLMClient.__init__()` with dual-model parameters
- [ ] Updated `_chat()` to accept provider/model parameters
- [ ] Updated `generate_improvement()` to use analyst config
- [ ] Updated `generate_strategy_code()` to use coder config
- [ ] Updated `fix_code()` to use coder config
- [ ] New CLI arguments for role-specific configuration
- [ ] Updated persistence to record dual-model config
- [ ] Unit tests for new functionality
- [ ] Backward compatibility with existing single-provider usage

---

## Sources

- [OpenAI Models Documentation](https://platform.openai.com/docs/models)
- [Introducing GPT-5.2 | OpenAI](https://openai.com/index/introducing-gpt-5-2/)
- [Anthropic Models Overview](https://docs.anthropic.com/en/docs/about-claude/models/overview)
- [Introducing Claude Opus 4.5](https://www.anthropic.com/news/claude-opus-4-5)
