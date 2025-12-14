# Alternative Data Integration

ProFiT can be extended to incorporate non-price data such as macroeconomic indicators, sentiment scores, or other alternative signals.

## Data Integration

### Adding Extra Columns

Add alternative data columns to your DataFrame:

```python
# Load price data
data = pd.read_csv("data/ES_hourly.csv", parse_dates=True, index_col=0)

# Load and merge sentiment data
sentiment = pd.read_csv("data/sentiment.csv", parse_dates=True, index_col=0)
data = data.join(sentiment, how='left')
data['Sentiment'] = data['Sentiment'].ffill()  # Forward-fill missing values

# Load and merge macro data
macro = pd.read_csv("data/macro.csv", parse_dates=True, index_col=0)
data = data.join(macro, how='left')
data['GDP_Growth'] = data['GDP_Growth'].ffill()
```

### Strategy Access

Strategies access additional data via `self.data.<ColumnName>`:

```python
class MacroEMACrossover(Strategy):
    def init(self):
        # Initialize EMAs (like EMACrossover)
        price = self.data.Close
        self.ema_fast = self.I(pd.Series.ewm, price, span=50).mean()
        self.ema_slow = self.I(pd.Series.ewm, price, span=200).mean()

    def next(self):
        # Only trade if macro condition is met
        if self.data.GDP_Growth[-1] > 0:
            # Apply EMA crossover logic as before
            if self.ema_fast[-1] > self.ema_slow[-1] and not self.position:
                self.buy()
            elif self.ema_fast[-1] < self.ema_slow[-1] and self.position:
                self.position.close()
```

## LLM Prompt Customization

### Informing LLM of Available Data

Modify the improvement prompt to mention available data:

```python
def generate_improvement(self, strategy_code, metrics_summary, available_data=None):
    data_note = ""
    if available_data:
        data_note = f"\n\nAlternate data available: {', '.join(available_data)}. Consider using these in the strategy."

    prompt = (
        "You are an expert trading strategy coach..."
        f"{data_note}\n\n"
        "Strategy Code:\n..."
    )
```

### Example with Sentiment

```python
# In evolver.py
improvement = self.llm.generate_improvement(
    parent_code,
    f"AnnReturn={parent_perf:.2f}%, Sharpe={sharpe:.2f}",
    available_data=['Sentiment', 'VIX', 'GDP_Growth']
)
```

The LLM might then suggest:
- "Incorporate Sentiment: only buy when sentiment is above 0.5"
- "Use VIX to adjust position sizing during high volatility"
- "Filter trades based on GDP_Growth trend"

## Example: Sentiment-Filtered Strategy

```python
class SentimentEMACrossover(Strategy):
    """EMA Crossover with sentiment filter."""
    fast_ema = 50
    slow_ema = 200
    sentiment_threshold = 0.5

    def init(self):
        price = self.data.Close
        self.ema_fast = self.I(pd.Series.ewm, price, span=self.fast_ema).mean()
        self.ema_slow = self.I(pd.Series.ewm, price, span=self.slow_ema).mean()

    def next(self):
        sentiment = self.data.Sentiment[-1]

        # Only consider long positions with positive sentiment
        if sentiment > self.sentiment_threshold:
            if self.ema_fast[-1] > self.ema_slow[-1] and not self.position:
                self.buy()

        # Exit logic unchanged
        if self.position.is_long and self.ema_fast[-1] < self.ema_slow[-1]:
            self.position.close()
```

## Regime-Aware Strategies

Use macro indicators to detect market regimes:

```python
class RegimeAwareStrategy(Strategy):
    def init(self):
        # Technical indicators
        self.setup_technical_indicators()

    def next(self):
        vix = self.data.VIX[-1]

        if vix < 20:
            # Low volatility regime: trend following
            self.trend_follow_logic()
        else:
            # High volatility regime: mean reversion
            self.mean_reversion_logic()
```

## Supported Alternative Data Types

| Data Type | Example Columns | Use Case |
|-----------|-----------------|----------|
| Sentiment | Sentiment, NewsScore | Filter trades by market sentiment |
| Volatility | VIX, ATR | Adjust position sizing, regime detection |
| Macro | GDP_Growth, Unemployment | Long-term trend filters |
| Technical | RSI, ADX | Additional signal confirmation |
| Volume | OBV, VWAP | Volume-based confirmation |

## Best Practices

1. **Data Alignment**: Ensure alternative data is properly aligned with price data timestamps
2. **Forward Fill**: Use forward-filling for sparse data to avoid look-ahead bias
3. **Feature Selection**: Start with few features and let evolution discover useful combinations
4. **Prompt Engineering**: Clearly describe available data and its meaning to the LLM
5. **Validation**: Always validate that alternative data doesn't introduce look-ahead bias

## Data Sources

Common sources for alternative data:

- **Sentiment**: News APIs, social media sentiment providers
- **Macro**: FRED (Federal Reserve Economic Data)
- **Volatility**: CBOE VIX data
- **Options**: Options flow data providers

### Example: Loading FRED Data

```python
import pandas_datareader as pdr

# Load GDP data from FRED
gdp = pdr.get_data_fred('GDP', start='2015-01-01')
gdp = gdp.resample('D').ffill()  # Resample to daily, forward fill

# Merge with price data
data = data.join(gdp, how='left')
data['GDP'] = data['GDP'].ffill()
```
