# Data Sources

This guide explains where to obtain the data files used in the examples and how to format them for use with ProFiT.

## Data Directory

All data files should be placed in the `data/` directory at the project root. This directory is in `.gitignore` and will not be committed to version control.

```bash
mkdir -p data
```

## Required Data Format

All OHLCV data files must be CSV format with:
- DateTime index column (first column)
- Columns: `Open`, `High`, `Low`, `Close`, `Volume` (case-sensitive)

Example structure:
```csv
Datetime,Open,High,Low,Close,Volume
2020-01-01 00:00:00,100.0,101.5,99.5,101.0,10000
2020-01-01 01:00:00,101.0,102.0,100.5,101.5,12000
```

**Minimum data requirement**: 5+ years for walk-forward optimization with 5 folds.

---

## Primary Data File

### ES_hourly.csv

**Description**: E-mini S&P 500 futures (ES) hourly OHLCV data. This is the primary example dataset used throughout the documentation.

**Minimum period**: 5 years of hourly data (approximately 31,000 bars)

#### Option 1: Yahoo Finance (Free)

Download S&P 500 index (^GSPC) or E-mini futures (ES=F) data:

```python
import yfinance as yf
import pandas as pd

# Download S&P 500 index data (proxy for ES futures)
ticker = yf.Ticker("^GSPC")
data = ticker.history(period="max", interval="1h")

# Format for ProFiT
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.index.name = 'Datetime'

# Save to data directory
data.to_csv("data/ES_hourly.csv")
print(f"Downloaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")
```

**Note**: Yahoo Finance hourly data may be limited to ~2 years. For longer history, use daily data or a paid data provider.

#### Option 2: Yahoo Finance Daily Data

For 5+ years of data, daily bars work well:

```python
import yfinance as yf

ticker = yf.Ticker("^GSPC")
data = ticker.history(period="10y", interval="1d")

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.index.name = 'Datetime'
data.to_csv("data/ES_daily.csv")
```

Then update your commands to use `--data data/ES_daily.csv`.

#### Option 3: Alpha Vantage (Free API Key)

1. Get a free API key at https://www.alphavantage.co/support/#api-key
2. Download data:

```python
import requests
import pandas as pd

API_KEY = "your_api_key_here"
symbol = "SPY"  # S&P 500 ETF

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}&datatype=csv"
data = pd.read_csv(url, index_col=0, parse_dates=True)

# Rename columns to match expected format
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data.sort_index()
data.index.name = 'Datetime'
data.to_csv("data/ES_daily.csv")
```

#### Option 4: Kaggle Datasets

Search Kaggle for "S&P 500 historical data" or "ES futures":
- https://www.kaggle.com/datasets

Download CSV and reformat if needed:

```python
import pandas as pd

# Load downloaded file
data = pd.read_csv("downloaded_file.csv", parse_dates=['Date'], index_col='Date')

# Ensure correct column names
data = data.rename(columns={
    'open': 'Open', 'high': 'High',
    'low': 'Low', 'close': 'Close', 'volume': 'Volume'
})
data.index.name = 'Datetime'
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.to_csv("data/ES_hourly.csv")
```

---

## Alternative Data Files (Optional)

These files are used in the [alternative data integration guide](alternative-data.md).

### sentiment.csv

**Description**: Market sentiment indicators (e.g., news sentiment, social media sentiment).

**Format**:
```csv
Datetime,Sentiment
2020-01-01,0.65
2020-01-02,0.72
```

#### Sources

**AAII Sentiment Survey** (Free):
- https://www.aaii.com/sentimentsurvey
- Weekly bull/bear/neutral percentages

**Fear & Greed Index**:
- CNN Fear & Greed historical data
- Can be scraped or obtained from financial APIs

**Example using pandas-datareader**:
```python
import pandas_datareader as pdr

# VIX as a volatility/fear proxy
vix = pdr.get_data_fred('VIXCLS', start='2015-01-01')
vix.columns = ['VIX']
vix.index.name = 'Datetime'
vix.to_csv("data/vix.csv")
```

### macro.csv

**Description**: Macroeconomic indicators (GDP growth, unemployment, interest rates).

**Format**:
```csv
Datetime,GDP_Growth,Unemployment,FedRate
2020-01-01,2.3,3.5,1.75
2020-04-01,-5.0,14.7,0.25
```

#### Source: FRED (Federal Reserve Economic Data)

FRED provides free access to thousands of economic indicators.

1. Browse data at https://fred.stlouisfed.org/
2. Download using pandas-datareader:

```python
import pandas_datareader as pdr
import pandas as pd

# Download multiple series
start = '2015-01-01'

gdp = pdr.get_data_fred('GDP', start=start)  # GDP
unrate = pdr.get_data_fred('UNRATE', start=start)  # Unemployment
fedfunds = pdr.get_data_fred('FEDFUNDS', start=start)  # Fed Funds Rate

# Combine into single DataFrame
macro = pd.concat([gdp, unrate, fedfunds], axis=1)
macro.columns = ['GDP', 'Unemployment', 'FedRate']

# Resample to daily and forward-fill (macro data is typically monthly/quarterly)
macro = macro.resample('D').ffill()
macro.index.name = 'Datetime'
macro.to_csv("data/macro.csv")
```

**Common FRED Series IDs**:
| Series ID | Description |
|-----------|-------------|
| GDP | Gross Domestic Product |
| UNRATE | Unemployment Rate |
| FEDFUNDS | Federal Funds Rate |
| CPIAUCSL | Consumer Price Index |
| T10Y2Y | 10Y-2Y Treasury Spread |
| VIXCLS | VIX Volatility Index |

---

## Data Validation

After downloading, verify your data:

```python
import pandas as pd

data = pd.read_csv("data/ES_hourly.csv", parse_dates=True, index_col=0)

print(f"Rows: {len(data)}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"Columns: {list(data.columns)}")
print(f"Missing values:\n{data.isnull().sum()}")

# Check required columns exist
required = ['Open', 'High', 'Low', 'Close']
missing = [col for col in required if col not in data.columns]
if missing:
    print(f"ERROR: Missing columns: {missing}")
else:
    print("All required columns present")
```

---

## Quick Start

Fastest way to get started with sample data:

```bash
# Create data directory
mkdir -p data

# Download with Python
python -c "
import yfinance as yf
data = yf.Ticker('^GSPC').history(period='5y', interval='1d')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.index.name = 'Datetime'
data.to_csv('data/ES_daily.csv')
print(f'Saved {len(data)} bars to data/ES_daily.csv')
"

# Run ProFiT
uv run python -m profit.main --data data/ES_daily.csv --strategy EMACrossover
```
