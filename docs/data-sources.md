# Data Sources

This guide explains where to obtain the data files used in the examples and how to format them for use with ProFiT.

## Quick Start

The fastest way to get started - run the download script directly:

```bash
# Download 5 years of S&P 500 daily data
uv run scripts/download_sample.py

# Run ProFiT
uv run python -m profit.main --data data/ES_daily.csv --strategy EMACrossover
```

---

## Download Scripts

ProFiT includes ready-to-use scripts for downloading data. Each script uses uv's inline dependencies - no manual setup required.

### Sample Data (Quick Start)

```bash
uv run scripts/download_sample.py           # 5 years daily data (default)
uv run scripts/download_sample.py --hourly  # ~2 years hourly data
```

### Yahoo Finance

Free, no API key needed:

```bash
uv run scripts/download_yahoo.py                              # S&P 500, 5 years daily
uv run scripts/download_yahoo.py --ticker SPY --period 10y    # SPY, 10 years
uv run scripts/download_yahoo.py --ticker AAPL --interval 1h  # Apple, hourly
uv run scripts/download_yahoo.py --ticker ES=F                # E-mini futures
```

Common tickers: `^GSPC` (S&P 500), `SPY` (S&P ETF), `ES=F` (E-mini futures), `^DJI` (Dow Jones)

### Alpha Vantage

Requires free API key from https://www.alphavantage.co/support/#api-key

```bash
# Using --api-key
uv run scripts/download_alphavantage.py --ticker SPY --api-key YOUR_KEY

# Or using environment variable
export ALPHA_VANTAGE_API_KEY=YOUR_KEY
uv run scripts/download_alphavantage.py --ticker AAPL
uv run scripts/download_alphavantage.py --ticker MSFT --interval 60min
```

### FRED (Macroeconomic Data)

Free access to economic indicators:

```bash
uv run scripts/download_fred.py                              # VIX (default)
uv run scripts/download_fred.py --series "VIXCLS,FEDFUNDS"   # Multiple series
uv run scripts/download_fred.py --series UNRATE --resample   # Daily resampling
```

Common series: `VIXCLS` (VIX), `FEDFUNDS` (Fed Rate), `UNRATE` (Unemployment), `GDP`, `CPIAUCSL` (CPI)

---

## Data Directory

All data files are saved to the `data/` directory at the project root. This directory is in `.gitignore` and will not be committed to version control.

```bash
mkdir -p data
```

---

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

### ES_hourly.csv / ES_daily.csv

**Description**: E-mini S&P 500 futures (ES) OHLCV data. This is the primary example dataset used throughout the documentation.

**Minimum period**: 5 years (approximately 1,250 daily bars or 31,000 hourly bars)

**Note**: Yahoo Finance hourly data is limited to ~2 years. For 5+ years needed for walk-forward optimization, use daily data.

---

## Alternative Data Files (Optional)

These files are used in the [alternative data integration guide](alternative-data.md).

### VIX / Sentiment Data

Download VIX as a volatility/fear proxy:

```bash
uv run scripts/download_fred.py --series VIXCLS --output data/vix.csv
```

### Macro Data

Download macroeconomic indicators:

```bash
uv run scripts/download_fred.py --series "GDP,UNRATE,FEDFUNDS" --resample --output data/macro.csv
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

## Manual Download (Alternative)

If you prefer to download data manually without using the scripts:

### Yahoo Finance with Python

```python
import yfinance as yf

ticker = yf.Ticker("^GSPC")
data = ticker.history(period="5y", interval="1d")

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.index.name = 'Datetime'
data.to_csv("data/ES_daily.csv")
```

### Kaggle Datasets

Search Kaggle for "S&P 500 historical data" or "ES futures":
- https://www.kaggle.com/datasets

Download CSV and reformat if needed:

```python
import pandas as pd

data = pd.read_csv("downloaded_file.csv", parse_dates=['Date'], index_col='Date')

# Ensure correct column names
data = data.rename(columns={
    'open': 'Open', 'high': 'High',
    'low': 'Low', 'close': 'Close', 'volume': 'Volume'
})
data.index.name = 'Datetime'
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.to_csv("data/ES_daily.csv")
```

---

## Data Validation

After downloading, verify your data:

```python
import pandas as pd

data = pd.read_csv("data/ES_daily.csv", parse_dates=True, index_col=0)

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
