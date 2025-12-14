#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "yfinance>=0.2.40",
#     "pandas>=2.0.0",
# ]
# ///
"""
Download OHLCV data from Yahoo Finance.

Usage:
    uv run scripts/download_yahoo.py
    uv run scripts/download_yahoo.py --ticker AAPL --period 10y
    uv run scripts/download_yahoo.py --ticker SPY --interval 1h --period 2y
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Args:
        ticker: Symbol to download (e.g., ^GSPC, SPY, AAPL)
        period: History period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {ticker} data (period={period}, interval={interval})...")

    t = yf.Ticker(ticker)
    data = t.history(period=period, interval=interval)

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    # Format for ProFiT: keep only OHLCV columns
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.index = data.index.tz_localize(None)  # Remove timezone for compatibility
    data.index.name = "Datetime"

    return data


def sanitize_filename(ticker: str) -> str:
    """Convert ticker to safe filename (replace ^ with _)."""
    return ticker.replace("^", "").replace("=", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Download OHLCV data from Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/download_yahoo.py                           # Download ^GSPC daily (5 years)
  uv run scripts/download_yahoo.py --ticker SPY              # Download SPY daily (5 years)
  uv run scripts/download_yahoo.py --ticker AAPL --period 10y
  uv run scripts/download_yahoo.py --ticker ES=F --interval 1h --period 2y

Common tickers:
  ^GSPC   S&P 500 Index
  SPY     S&P 500 ETF
  ES=F    E-mini S&P 500 Futures
  ^DJI    Dow Jones Industrial Average
  ^IXIC   NASDAQ Composite
  AAPL    Apple Inc.
        """,
    )
    parser.add_argument(
        "--ticker",
        default="^GSPC",
        help="Symbol to download (default: ^GSPC)",
    )
    parser.add_argument(
        "--period",
        default="5y",
        choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        help="History period (default: 5y)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        help="Data interval (default: 1d). Note: intraday data limited to 60 days for 1m, 730 days for 1h.",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: data/{ticker}_{interval}.csv)",
    )

    args = parser.parse_args()

    # Create data directory if needed
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        safe_ticker = sanitize_filename(args.ticker)
        output_path = data_dir / f"{safe_ticker}_{args.interval}.csv"

    try:
        data = download_data(args.ticker, args.period, args.interval)

        # Save to CSV
        data.to_csv(output_path)

        # Print summary
        print(f"\nSaved {len(data):,} bars to {output_path}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Columns: {list(data.columns)}")

        # Check data quality
        missing = data.isnull().sum().sum()
        if missing > 0:
            print(f"Warning: {missing} missing values")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
