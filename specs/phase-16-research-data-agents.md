# Phase 16: Researcher & Data Collector Agents

## Objective

Add autonomous agents that discover new strategies and data sources, with a human approval workflow for untrusted sources. This enables the system to actively find new alpha opportunities rather than relying solely on the LLM's static training knowledge.

---

## Dependencies

- Phase 4 (LLM Interface) - existing `LLMClient` class
- Phase 13 (Program Database) - for storing discovered ideas
- Existing download scripts (`scripts/download_*.py`)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Agent Pipeline                                 │
│                                                                         │
│  ┌──────────────────┐          ┌──────────────────┐                   │
│  │  Researcher      │          │  Data Collector  │                   │
│  │  Agent           │          │  Agent           │                   │
│  │                  │          │                  │                   │
│  │  • Find papers   │          │  • Find datasets │                   │
│  │  • Market theses │          │  • Schema detect │                   │
│  │  • Strategy ideas│          │  • Fetch & cache │                   │
│  └────────┬─────────┘          └────────┬─────────┘                   │
│           │                             │                              │
│           ▼                             ▼                              │
│  ┌────────────────────────────────────────────────────────────┐       │
│  │                    Source Trust Filter                      │       │
│  │  ┌─────────────┐              ┌─────────────┐              │       │
│  │  │ Trusted     │              │ Untrusted   │              │       │
│  │  │ Sources     │              │ Sources     │              │       │
│  │  │             │              │             │              │       │
│  │  │ requires_   │              │ requires_   │              │       │
│  │  │ review:false│              │ review:true │              │       │
│  │  └──────┬──────┘              └──────┬──────┘              │       │
│  │         │                            │                      │       │
│  │         ▼                            ▼                      │       │
│  │    ┌─────────┐              ┌─────────────────┐            │       │
│  │    │ Direct  │              │ Pending Queue   │            │       │
│  │    │ to      │              │ (human review)  │            │       │
│  │    │Evolution│              └────────┬────────┘            │       │
│  │    └─────────┘                       │                      │       │
│  │                                      ▼                      │       │
│  │                               ┌─────────────┐              │       │
│  │                               │ CLI Approve │              │       │
│  │                               │ / Reject    │              │       │
│  │                               └─────────────┘              │       │
│  └────────────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Data Models

### IdeaCard

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum
import uuid


class IdeaStatus(Enum):
    PENDING = "pending"      # Awaiting human review
    APPROVED = "approved"    # Ready for evolution
    REJECTED = "rejected"    # Declined by human
    IMPLEMENTED = "implemented"  # Strategy created


@dataclass
class IdeaCard:
    """
    A strategy idea discovered by the Researcher Agent.

    Contains all information needed to implement a trading strategy
    without requiring additional research.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Core idea
    hypothesis: str = ""          # What market behavior does this exploit?
    rationale: str = ""           # Why might this work?

    # Requirements
    required_datasets: List[str] = field(default_factory=list)  # e.g., ["OHLCV", "VIX", "FRED:GDP"]
    required_indicators: List[str] = field(default_factory=list)  # e.g., ["RSI", "MACD", "ATR"]

    # Implementation guidance
    implementation_outline: str = ""  # Pseudocode or description
    entry_conditions: str = ""        # When to enter
    exit_conditions: str = ""         # When to exit
    position_sizing: str = ""         # How to size positions

    # Expected behavior
    expected_behavior: str = ""       # How it should perform
    expected_market_conditions: str = ""  # When it works best
    failure_modes: str = ""           # When it might fail

    # Validation
    evaluation_checklist: List[str] = field(default_factory=list)  # Tests to run

    # Provenance
    source: str = ""                  # Where this idea came from
    source_references: List[str] = field(default_factory=list)  # URLs, paper titles
    confidence: float = 0.5           # Agent's confidence (0-1)

    # Metadata
    status: IdeaStatus = IdeaStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_notes: str = ""

    # Tags for categorization
    tags: List[str] = field(default_factory=list)  # e.g., ["momentum", "mean-reversion"]


@dataclass
class DataProposal:
    """
    A data source proposal from the Data Collector Agent.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Dataset info
    dataset_name: str = ""        # Human-readable name
    description: str = ""         # What this data represents

    # Source info
    source: str = ""              # Connector name (yahoo, alphavantage, fred, etc.)
    source_url: str = ""          # Original data URL
    connector_params: dict = field(default_factory=dict)  # Parameters for connector

    # Schema
    schema: dict = field(default_factory=dict)  # Column names and types
    frequency: str = ""           # daily, hourly, etc.
    history_start: str = ""       # Earliest available date
    update_schedule: str = ""     # How often to refresh

    # Licensing
    licensing_notes: str = ""     # Terms, attribution requirements
    is_free: bool = True

    # Status
    status: str = "pending"       # pending, approved, rejected, active
    created_at: datetime = field(default_factory=datetime.now)
```

---

## Source Configuration

### Config File: `configs/sources.yaml`

```yaml
# Research sources for the Researcher Agent
research_sources:
  # Trusted academic sources (no review needed)
  - name: arXiv
    type: academic
    url: https://arxiv.org
    search_categories: ["q-fin.TR", "q-fin.PM", "stat.ML"]
    requires_review: false

  - name: SSRN
    type: academic
    url: https://ssrn.com
    search_categories: ["Finance", "Quantitative Finance"]
    requires_review: false

  - name: Journal of Portfolio Management
    type: academic
    url: https://jpm.pm-research.com
    requires_review: false

  # Trusted curated sources
  - name: Quantocracy
    type: aggregator
    url: https://quantocracy.com
    requires_review: false

  - name: Alpha Architect
    type: blog
    url: https://alphaarchitect.com
    requires_review: false

  - name: QuantStart
    type: blog
    url: https://www.quantstart.com
    requires_review: false

  # Untrusted sources (require human review)
  - name: Reddit r/algotrading
    type: forum
    url: https://reddit.com/r/algotrading
    requires_review: true

  - name: TradingView Ideas
    type: community
    url: https://www.tradingview.com/ideas
    requires_review: true

  - name: Seeking Alpha
    type: news
    url: https://seekingalpha.com
    requires_review: true

# Data sources for the Data Collector Agent
data_sources:
  # Trusted standard sources (no review needed)
  - name: yahoo
    connector: YahooConnector
    description: Yahoo Finance price data
    requires_review: false
    default_params:
      interval: "1d"

  - name: alphavantage
    connector: AlphaVantageConnector
    description: AlphaVantage market data
    requires_review: false
    requires_api_key: true
    env_var: ALPHAVANTAGE_API_KEY

  - name: fred
    connector: FREDConnector
    description: Federal Reserve Economic Data
    requires_review: false
    default_params:
      series: ["GDP", "UNRATE", "CPIAUCSL"]

  # Untrusted/new sources (require review)
  - name: custom_api
    connector: CustomAPIConnector
    description: Custom REST API connector
    requires_review: true

  - name: csv_file
    connector: CSVFileConnector
    description: Local CSV file import
    requires_review: true
```

### Config File: `configs/pending_approvals.yaml`

```yaml
# Items awaiting human review
pending_ideas: []
  # - id: abc123
  #   type: idea
  #   submitted_at: 2024-01-15T10:00:00
  #   source: Reddit r/algotrading

pending_data: []
  # - id: def456
  #   type: data
  #   submitted_at: 2024-01-15T11:00:00
  #   source: custom_api

# Review history
review_history: []
  # - id: abc123
  #   action: approved
  #   reviewed_at: 2024-01-15T12:00:00
  #   reviewed_by: user
  #   notes: "Looks promising"
```

---

## Source Registry

### File: `src/profit/sources.py`

```python
"""Source configuration and trust management."""

import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ResearchSource:
    """Configuration for a research source."""
    name: str
    type: str
    url: str
    requires_review: bool
    search_categories: List[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'ResearchSource':
        return cls(
            name=d['name'],
            type=d.get('type', 'unknown'),
            url=d.get('url', ''),
            requires_review=d.get('requires_review', True),
            search_categories=d.get('search_categories', [])
        )


@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    connector: str
    description: str
    requires_review: bool
    requires_api_key: bool = False
    env_var: str = None
    default_params: dict = None

    @classmethod
    def from_dict(cls, d: dict) -> 'DataSource':
        return cls(
            name=d['name'],
            connector=d['connector'],
            description=d.get('description', ''),
            requires_review=d.get('requires_review', True),
            requires_api_key=d.get('requires_api_key', False),
            env_var=d.get('env_var'),
            default_params=d.get('default_params', {})
        )


class SourceRegistry:
    """
    Registry for research and data sources with trust levels.
    """

    def __init__(self, config_path: str = "configs/sources.yaml"):
        self.config_path = Path(config_path)
        self.research_sources: Dict[str, ResearchSource] = {}
        self.data_sources: Dict[str, DataSource] = {}
        self._load_config()

    def _load_config(self):
        """Load sources from config file."""
        if not self.config_path.exists():
            # Create default config
            self._create_default_config()

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        for src in config.get('research_sources', []):
            rs = ResearchSource.from_dict(src)
            self.research_sources[rs.name] = rs

        for src in config.get('data_sources', []):
            ds = DataSource.from_dict(src)
            self.data_sources[ds.name] = ds

    def _create_default_config(self):
        """Create default config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            'research_sources': [
                {'name': 'arXiv', 'type': 'academic', 'url': 'https://arxiv.org', 'requires_review': False},
                {'name': 'SSRN', 'type': 'academic', 'url': 'https://ssrn.com', 'requires_review': False},
            ],
            'data_sources': [
                {'name': 'yahoo', 'connector': 'YahooConnector', 'description': 'Yahoo Finance', 'requires_review': False},
                {'name': 'fred', 'connector': 'FREDConnector', 'description': 'FRED Economic Data', 'requires_review': False},
            ]
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

    def get_trusted_research_sources(self) -> List[ResearchSource]:
        """Get sources that don't require review."""
        return [s for s in self.research_sources.values() if not s.requires_review]

    def get_trusted_data_sources(self) -> List[DataSource]:
        """Get data sources that don't require review."""
        return [s for s in self.data_sources.values() if not s.requires_review]

    def is_trusted_source(self, source_name: str) -> bool:
        """Check if a source is trusted (no review needed)."""
        if source_name in self.research_sources:
            return not self.research_sources[source_name].requires_review
        if source_name in self.data_sources:
            return not self.data_sources[source_name].requires_review
        return False

    def trust_source(self, source_name: str):
        """Mark a source as trusted (set requires_review=False)."""
        if source_name in self.research_sources:
            self.research_sources[source_name].requires_review = False
        elif source_name in self.data_sources:
            self.data_sources[source_name].requires_review = False
        self._save_config()

    def _save_config(self):
        """Save current config to file."""
        config = {
            'research_sources': [
                {
                    'name': s.name,
                    'type': s.type,
                    'url': s.url,
                    'requires_review': s.requires_review,
                    'search_categories': s.search_categories or []
                }
                for s in self.research_sources.values()
            ],
            'data_sources': [
                {
                    'name': s.name,
                    'connector': s.connector,
                    'description': s.description,
                    'requires_review': s.requires_review,
                    'requires_api_key': s.requires_api_key,
                    'env_var': s.env_var,
                    'default_params': s.default_params or {}
                }
                for s in self.data_sources.values()
            ]
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
```

---

## Researcher Agent

### File: `src/profit/agents/researcher.py`

```python
"""
Researcher Agent - discovers new strategy ideas from approved sources.
"""

from typing import List, Optional
from dataclasses import asdict
import json

from profit.llm_interface import LLMClient
from profit.sources import SourceRegistry, ResearchSource


class ResearcherAgent:
    """
    Agent that discovers new trading strategy ideas.

    Uses LLM to analyze market conditions and research sources
    to generate implementation-ready strategy ideas.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        source_registry: SourceRegistry = None,
        max_ideas_per_run: int = 5
    ):
        self.llm = llm_client
        self.sources = source_registry or SourceRegistry()
        self.max_ideas = max_ideas_per_run

    def generate_ideas(
        self,
        market_context: str = "",
        focus_areas: List[str] = None,
        existing_strategies: List[str] = None
    ) -> List[IdeaCard]:
        """
        Generate new strategy ideas.

        Args:
            market_context: Current market conditions/themes
            focus_areas: Specific areas to explore (e.g., "volatility", "momentum")
            existing_strategies: Names of strategies already in the system (to avoid duplicates)

        Returns:
            List of IdeaCard objects
        """
        focus_areas = focus_areas or ["momentum", "mean-reversion", "volatility", "trend-following"]
        existing_strategies = existing_strategies or []

        # Build prompt with trusted sources
        trusted_sources = self.sources.get_trusted_research_sources()
        sources_context = "\n".join([
            f"- {s.name} ({s.type}): {s.url}"
            for s in trusted_sources
        ])

        prompt = f"""You are a quantitative research analyst. Generate {self.max_ideas} trading strategy ideas
based on established quantitative finance research.

TRUSTED RESEARCH SOURCES (prefer citing these):
{sources_context}

CURRENT MARKET CONTEXT:
{market_context or "No specific market context provided."}

FOCUS AREAS:
{', '.join(focus_areas)}

EXISTING STRATEGIES (avoid duplicating these):
{', '.join(existing_strategies) or "None"}

For each strategy idea, provide a complete specification in JSON format:

```json
[
  {{
    "hypothesis": "What market inefficiency or behavior does this exploit?",
    "rationale": "Why might this work based on academic research or market microstructure?",
    "required_datasets": ["OHLCV", "other required data"],
    "required_indicators": ["indicator names"],
    "implementation_outline": "Step-by-step implementation pseudocode",
    "entry_conditions": "Specific conditions for entering a position",
    "exit_conditions": "Specific conditions for exiting",
    "position_sizing": "How to size positions",
    "expected_behavior": "Expected win rate, risk/reward, holding period",
    "expected_market_conditions": "When this strategy works best",
    "failure_modes": "When and why this might fail",
    "evaluation_checklist": ["Test 1", "Test 2"],
    "source_references": ["Paper title or URL"],
    "confidence": 0.7,
    "tags": ["momentum", "equity"]
  }}
]
```

Generate diverse ideas across the focus areas. Each idea should be:
1. Grounded in academic research or well-established market principles
2. Specific enough to implement directly
3. Testable with clear success/failure criteria
"""

        response = self.llm._call_llm(prompt, role="analyst")

        # Parse response
        ideas = self._parse_ideas(response)

        # Mark source and review status
        for idea in ideas:
            idea.source = "ResearcherAgent"
            # Check if any referenced source requires review
            requires_review = False
            for ref in idea.source_references:
                for src_name in self.sources.research_sources:
                    if src_name.lower() in ref.lower():
                        if self.sources.research_sources[src_name].requires_review:
                            requires_review = True
                            break

            idea.status = IdeaStatus.PENDING if requires_review else IdeaStatus.APPROVED

        return ideas

    def _parse_ideas(self, response: str) -> List[IdeaCard]:
        """Parse LLM response into IdeaCard objects."""
        ideas = []

        # Extract JSON from response
        try:
            # Find JSON array in response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                for item in data:
                    idea = IdeaCard(
                        hypothesis=item.get('hypothesis', ''),
                        rationale=item.get('rationale', ''),
                        required_datasets=item.get('required_datasets', []),
                        required_indicators=item.get('required_indicators', []),
                        implementation_outline=item.get('implementation_outline', ''),
                        entry_conditions=item.get('entry_conditions', ''),
                        exit_conditions=item.get('exit_conditions', ''),
                        position_sizing=item.get('position_sizing', ''),
                        expected_behavior=item.get('expected_behavior', ''),
                        expected_market_conditions=item.get('expected_market_conditions', ''),
                        failure_modes=item.get('failure_modes', ''),
                        evaluation_checklist=item.get('evaluation_checklist', []),
                        source_references=item.get('source_references', []),
                        confidence=item.get('confidence', 0.5),
                        tags=item.get('tags', [])
                    )
                    ideas.append(idea)

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse LLM response as JSON: {e}")

        return ideas

    def refine_idea(self, idea: IdeaCard, feedback: str) -> IdeaCard:
        """
        Refine an existing idea based on feedback.

        Args:
            idea: The idea to refine
            feedback: Human or automated feedback

        Returns:
            Refined IdeaCard
        """
        prompt = f"""Refine this trading strategy idea based on the feedback provided.

ORIGINAL IDEA:
{json.dumps(asdict(idea), indent=2, default=str)}

FEEDBACK:
{feedback}

Provide the refined idea in the same JSON format, addressing the feedback while
maintaining the core hypothesis if it's sound, or pivoting if necessary.
"""

        response = self.llm._call_llm(prompt, role="analyst")
        refined_ideas = self._parse_ideas(f"[{response}]")

        if refined_ideas:
            refined = refined_ideas[0]
            refined.id = idea.id  # Keep same ID
            return refined

        return idea
```

---

## Data Collector Agent

### File: `src/profit/agents/data_collector.py`

```python
"""
Data Collector Agent - discovers and provisions data sources.
"""

from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import json

from profit.llm_interface import LLMClient
from profit.sources import SourceRegistry, DataSource
from profit.data.connector_registry import ConnectorRegistry


class DataCollectorAgent:
    """
    Agent that discovers, proposes, and fetches data sources.

    Works with the ConnectorRegistry to provision data for strategies.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        source_registry: SourceRegistry = None,
        connector_registry: ConnectorRegistry = None,
        data_dir: str = "data"
    ):
        self.llm = llm_client
        self.sources = source_registry or SourceRegistry()
        self.connectors = connector_registry or ConnectorRegistry()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def propose_data(
        self,
        requirements: List[str],
        idea_card: IdeaCard = None
    ) -> List[DataProposal]:
        """
        Propose data sources to fulfill requirements.

        Args:
            requirements: List of data requirements (e.g., ["OHLCV for SPY", "VIX index"])
            idea_card: Optional IdeaCard context

        Returns:
            List of DataProposal objects
        """
        # Get available connectors
        available = self.connectors.list_connectors()
        trusted_sources = self.sources.get_trusted_data_sources()

        prompt = f"""You are a data sourcing specialist. For each data requirement,
suggest how to obtain the data using available connectors.

REQUIREMENTS:
{chr(10).join(f'- {r}' for r in requirements)}

AVAILABLE CONNECTORS:
{chr(10).join(f'- {c}' for c in available)}

TRUSTED DATA SOURCES:
{chr(10).join(f'- {s.name}: {s.description}' for s in trusted_sources)}

For each requirement, provide a data proposal in JSON format:

```json
[
  {{
    "dataset_name": "Human readable name",
    "description": "What this data represents",
    "requirement_fulfilled": "Which requirement this fulfills",
    "source": "Connector name (yahoo, fred, etc.)",
    "connector_params": {{"ticker": "SPY", "interval": "1d"}},
    "schema": {{"Open": "float", "High": "float", "Low": "float", "Close": "float", "Volume": "int"}},
    "frequency": "daily",
    "history_start": "2010-01-01",
    "update_schedule": "daily at market close",
    "licensing_notes": "Free for personal use",
    "is_free": true
  }}
]
```

Prefer trusted sources when possible. If a requirement cannot be fulfilled
with available connectors, explain why.
"""

        response = self.llm._call_llm(prompt, role="analyst")
        proposals = self._parse_proposals(response)

        # Set review status based on source trust
        for proposal in proposals:
            if self.sources.is_trusted_source(proposal.source):
                proposal.status = "approved"
            else:
                proposal.status = "pending"

        return proposals

    def _parse_proposals(self, response: str) -> List[DataProposal]:
        """Parse LLM response into DataProposal objects."""
        proposals = []

        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                for item in data:
                    proposal = DataProposal(
                        dataset_name=item.get('dataset_name', ''),
                        description=item.get('description', ''),
                        source=item.get('source', ''),
                        connector_params=item.get('connector_params', {}),
                        schema=item.get('schema', {}),
                        frequency=item.get('frequency', ''),
                        history_start=item.get('history_start', ''),
                        update_schedule=item.get('update_schedule', ''),
                        licensing_notes=item.get('licensing_notes', ''),
                        is_free=item.get('is_free', True)
                    )
                    proposals.append(proposal)

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse response: {e}")

        return proposals

    def fetch_data(self, proposal: DataProposal) -> Optional[pd.DataFrame]:
        """
        Fetch data for an approved proposal.

        Args:
            proposal: Approved DataProposal

        Returns:
            DataFrame with the data, or None if fetch failed
        """
        if proposal.status != "approved":
            raise ValueError(f"Cannot fetch unapproved data: {proposal.dataset_name}")

        connector = self.connectors.get(proposal.source)
        if not connector:
            raise ValueError(f"Unknown connector: {proposal.source}")

        try:
            df = connector.fetch(**proposal.connector_params)

            # Save to disk
            filename = f"{proposal.dataset_name.replace(' ', '_').lower()}.csv"
            filepath = self.data_dir / filename
            df.to_csv(filepath)
            print(f"Saved data to {filepath}")

            return df

        except Exception as e:
            print(f"Failed to fetch data: {e}")
            return None

    def check_data_availability(
        self,
        requirements: List[str]
    ) -> Dict[str, bool]:
        """
        Check which requirements can be fulfilled with existing data.

        Args:
            requirements: List of data requirements

        Returns:
            Dict mapping requirements to availability status
        """
        availability = {}

        for req in requirements:
            # Check if we have cached data
            req_normalized = req.lower().replace(' ', '_')
            for f in self.data_dir.glob("*.csv"):
                if req_normalized in f.stem.lower():
                    availability[req] = True
                    break
            else:
                availability[req] = False

        return availability
```

---

## Connector Registry

### File: `src/profit/data/connector_registry.py`

```python
"""
Data connector registry and base implementations.
"""

from typing import Protocol, Dict, Optional, List
import pandas as pd
from abc import abstractmethod


class DataConnector(Protocol):
    """Protocol for data source connectors."""

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """Fetch data with given parameters."""
        ...

    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """Return expected column schema."""
        ...


class YahooConnector:
    """Yahoo Finance data connector."""

    def fetch(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        import yfinance as yf

        data = yf.download(ticker, start=start, end=end, interval=interval)
        return data

    def get_schema(self) -> Dict[str, str]:
        return {
            "Open": "float",
            "High": "float",
            "Low": "float",
            "Close": "float",
            "Adj Close": "float",
            "Volume": "int"
        }


class FREDConnector:
    """Federal Reserve Economic Data connector."""

    def fetch(
        self,
        series: str,
        start: str = None,
        end: str = None
    ) -> pd.DataFrame:
        """Fetch economic data from FRED."""
        import pandas_datareader as pdr

        data = pdr.get_data_fred(series, start=start, end=end)
        return data

    def get_schema(self) -> Dict[str, str]:
        return {"value": "float"}


class AlphaVantageConnector:
    """AlphaVantage data connector."""

    def __init__(self, api_key: str = None):
        import os
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")

    def fetch(
        self,
        ticker: str,
        interval: str = "daily",
        outputsize: str = "full"
    ) -> pd.DataFrame:
        """Fetch data from AlphaVantage."""
        from alpha_vantage.timeseries import TimeSeries

        ts = TimeSeries(key=self.api_key, output_format='pandas')

        if interval == "daily":
            data, _ = ts.get_daily(symbol=ticker, outputsize=outputsize)
        else:
            data, _ = ts.get_intraday(symbol=ticker, interval=interval, outputsize=outputsize)

        return data

    def get_schema(self) -> Dict[str, str]:
        return {
            "1. open": "float",
            "2. high": "float",
            "3. low": "float",
            "4. close": "float",
            "5. volume": "int"
        }


class CSVFileConnector:
    """Local CSV file connector."""

    def fetch(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(filepath, parse_dates=True, index_col=0, **kwargs)

    def get_schema(self) -> Dict[str, str]:
        return {}  # Schema depends on file


class ConnectorRegistry:
    """
    Registry of available data connectors.
    """

    def __init__(self):
        self._connectors: Dict[str, DataConnector] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default connectors."""
        self.register("yahoo", YahooConnector())
        self.register("fred", FREDConnector())
        self.register("csv", CSVFileConnector())

        # AlphaVantage only if API key available
        import os
        if os.getenv("ALPHAVANTAGE_API_KEY"):
            self.register("alphavantage", AlphaVantageConnector())

    def register(self, name: str, connector: DataConnector):
        """Register a new connector."""
        self._connectors[name] = connector

    def get(self, name: str) -> Optional[DataConnector]:
        """Get a connector by name."""
        return self._connectors.get(name)

    def list_connectors(self) -> List[str]:
        """List available connector names."""
        return list(self._connectors.keys())
```

---

## Approval Workflow

### File: `src/profit/approval.py`

```python
"""
Approval workflow for pending ideas and data proposals.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class ApprovalManager:
    """
    Manages the approval queue for ideas and data proposals.
    """

    def __init__(self, config_path: str = "configs/pending_approvals.yaml"):
        self.config_path = Path(config_path)
        self._ensure_config()

    def _ensure_config(self):
        """Ensure config file exists."""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._save({
                'pending_ideas': [],
                'pending_data': [],
                'review_history': []
            })

    def _load(self) -> Dict:
        """Load config from file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f) or {}

    def _save(self, config: Dict):
        """Save config to file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def add_pending_idea(self, idea: IdeaCard):
        """Add an idea to the pending queue."""
        config = self._load()
        config['pending_ideas'].append({
            'id': idea.id,
            'hypothesis': idea.hypothesis[:100] + '...',
            'source': idea.source,
            'submitted_at': datetime.now().isoformat()
        })
        self._save(config)

    def add_pending_data(self, proposal: DataProposal):
        """Add a data proposal to the pending queue."""
        config = self._load()
        config['pending_data'].append({
            'id': proposal.id,
            'dataset_name': proposal.dataset_name,
            'source': proposal.source,
            'submitted_at': datetime.now().isoformat()
        })
        self._save(config)

    def get_pending(self) -> Dict[str, List]:
        """Get all pending items."""
        config = self._load()
        return {
            'ideas': config.get('pending_ideas', []),
            'data': config.get('pending_data', [])
        }

    def approve(self, item_id: str, reviewer: str = "user", notes: str = "") -> bool:
        """Approve a pending item."""
        return self._review(item_id, "approved", reviewer, notes)

    def reject(self, item_id: str, reviewer: str = "user", notes: str = "") -> bool:
        """Reject a pending item."""
        return self._review(item_id, "rejected", reviewer, notes)

    def _review(self, item_id: str, action: str, reviewer: str, notes: str) -> bool:
        """Process a review action."""
        config = self._load()

        # Find and remove from pending
        found = False
        for queue in ['pending_ideas', 'pending_data']:
            for item in config[queue]:
                if item['id'] == item_id:
                    config[queue].remove(item)
                    found = True
                    break
            if found:
                break

        if not found:
            return False

        # Add to history
        config['review_history'].append({
            'id': item_id,
            'action': action,
            'reviewed_at': datetime.now().isoformat(),
            'reviewed_by': reviewer,
            'notes': notes
        })

        self._save(config)
        return True

    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get recent review history."""
        config = self._load()
        return config.get('review_history', [])[-limit:]
```

---

## CLI Commands

Add to `src/profit/main.py`:

```python
import argparse
from profit.agents.researcher import ResearcherAgent
from profit.agents.data_collector import DataCollectorAgent
from profit.approval import ApprovalManager
from profit.sources import SourceRegistry


def create_parser():
    parser = argparse.ArgumentParser(prog='profit')
    subparsers = parser.add_subparsers(dest='command')

    # ... existing subcommands ...

    # Research command
    research_parser = subparsers.add_parser('research', help='Generate strategy ideas')
    research_parser.add_argument('--context', help='Market context description')
    research_parser.add_argument('--focus', nargs='+', help='Focus areas')
    research_parser.add_argument('--count', type=int, default=5, help='Number of ideas')

    # Data command
    data_parser = subparsers.add_parser('data', help='Manage data sources')
    data_parser.add_argument('--propose', nargs='+', help='Propose data for requirements')
    data_parser.add_argument('--fetch', help='Fetch approved data by ID')
    data_parser.add_argument('--list', action='store_true', help='List available data')

    # Approval commands
    approve_parser = subparsers.add_parser('approve', help='Approve pending item')
    approve_parser.add_argument('item_id', help='ID of item to approve')
    approve_parser.add_argument('--notes', default='', help='Review notes')

    reject_parser = subparsers.add_parser('reject', help='Reject pending item')
    reject_parser.add_argument('item_id', help='ID of item to reject')
    reject_parser.add_argument('--notes', default='', help='Review notes')

    list_pending_parser = subparsers.add_parser('list-pending', help='List pending approvals')

    trust_parser = subparsers.add_parser('trust-source', help='Mark source as trusted')
    trust_parser.add_argument('source_name', help='Name of source to trust')

    return parser


def handle_research(args, llm_client):
    """Handle research command."""
    researcher = ResearcherAgent(llm_client, max_ideas_per_run=args.count)

    print(f"Generating {args.count} strategy ideas...")
    ideas = researcher.generate_ideas(
        market_context=args.context or "",
        focus_areas=args.focus
    )

    # Handle approval status
    approval = ApprovalManager()

    for idea in ideas:
        print(f"\n{'='*60}")
        print(f"ID: {idea.id}")
        print(f"Hypothesis: {idea.hypothesis}")
        print(f"Tags: {', '.join(idea.tags)}")
        print(f"Confidence: {idea.confidence:.0%}")
        print(f"Status: {idea.status.value}")

        if idea.status == IdeaStatus.PENDING:
            approval.add_pending_idea(idea)
            print("  → Added to pending queue for review")

    print(f"\nGenerated {len(ideas)} ideas. "
          f"Use 'profit list-pending' to see items awaiting review.")


def handle_list_pending(args):
    """Handle list-pending command."""
    approval = ApprovalManager()
    pending = approval.get_pending()

    print("\nPending Ideas:")
    for item in pending['ideas']:
        print(f"  [{item['id']}] {item['hypothesis']} (from {item['source']})")

    print("\nPending Data Proposals:")
    for item in pending['data']:
        print(f"  [{item['id']}] {item['dataset_name']} (from {item['source']})")

    if not pending['ideas'] and not pending['data']:
        print("  No pending items.")


def handle_approve(args):
    """Handle approve command."""
    approval = ApprovalManager()
    if approval.approve(args.item_id, notes=args.notes):
        print(f"Approved: {args.item_id}")
    else:
        print(f"Item not found: {args.item_id}")


def handle_reject(args):
    """Handle reject command."""
    approval = ApprovalManager()
    if approval.reject(args.item_id, notes=args.notes):
        print(f"Rejected: {args.item_id}")
    else:
        print(f"Item not found: {args.item_id}")


def handle_trust_source(args):
    """Handle trust-source command."""
    registry = SourceRegistry()
    registry.trust_source(args.source_name)
    print(f"Marked '{args.source_name}' as trusted (requires_review=false)")
```

---

## File Structure

```
src/profit/
├── __init__.py
├── strategies.py
├── llm_interface.py
├── evolver.py
├── main.py               # Modified: add CLI commands
├── program_db.py
├── diff_utils.py
├── evaluation.py
├── sources.py            # NEW: source registry
├── approval.py           # NEW: approval workflow
├── agents/
│   ├── __init__.py       # NEW
│   ├── researcher.py     # NEW
│   └── data_collector.py # NEW
└── data/
    ├── __init__.py       # NEW
    └── connector_registry.py  # NEW

configs/
├── sources.yaml          # NEW: source configuration
└── pending_approvals.yaml # NEW: approval queue
```

---

## Deliverables

- [ ] Data models:
  - [ ] `IdeaCard` dataclass
  - [ ] `DataProposal` dataclass
- [ ] Source management:
  - [ ] `SourceRegistry` class
  - [ ] `configs/sources.yaml` configuration
  - [ ] Per-source `requires_review` attribute
- [ ] Researcher Agent:
  - [ ] `ResearcherAgent` class
  - [ ] `generate_ideas()` method
  - [ ] `refine_idea()` method
  - [ ] JSON parsing for LLM responses
- [ ] Data Collector Agent:
  - [ ] `DataCollectorAgent` class
  - [ ] `propose_data()` method
  - [ ] `fetch_data()` method
- [ ] Connector Registry:
  - [ ] `ConnectorRegistry` class
  - [ ] `YahooConnector`
  - [ ] `FREDConnector`
  - [ ] `AlphaVantageConnector`
  - [ ] `CSVFileConnector`
- [ ] Approval Workflow:
  - [ ] `ApprovalManager` class
  - [ ] `configs/pending_approvals.yaml`
  - [ ] `approve()` / `reject()` methods
- [ ] CLI Commands:
  - [ ] `profit research`
  - [ ] `profit data --propose`
  - [ ] `profit approve <id>`
  - [ ] `profit reject <id>`
  - [ ] `profit list-pending`
  - [ ] `profit trust-source <name>`
- [ ] Tests for:
  - [ ] Researcher Agent idea generation
  - [ ] Data Collector proposals
  - [ ] Approval workflow
  - [ ] Source trust filtering
