#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.0.0",
#     "requests>=2.31.0",
# ]
# ///
"""
Download OHLCV data from Alpha Vantage.

Requires a free API key from: https://www.alphavantage.co/support/#api-key

Usage:
    uv run scripts/download_alphavantage.py --api-key YOUR_KEY
    uv run scripts/download_alphavantage.py --ticker AAPL --api-key YOUR_KEY

    # Or set environment variable:
    export ALPHA_VANTAGE_API_KEY=YOUR_KEY
    uv run scripts/download_alphavantage.py --ticker SPY
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests


def download_daily(ticker: str, api_key: str, outputsize: str = "full") -> pd.DataFrame:
    """Download daily OHLCV data from Alpha Vantage.

    Args:
        ticker: Symbol to download (e.g., SPY, AAPL)
        api_key: Alpha Vantage API key
        outputsize: 'compact' (100 days) or 'full' (20+ years)

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {ticker} daily data from Alpha Vantage...")

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}"
        f"&outputsize={outputsize}"
        f"&apikey={api_key}"
        f"&datatype=csv"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Check for API errors (returned as JSON even when requesting CSV)
    if response.text.startswith("{"):
        import json
        error = json.loads(response.text)
        if "Error Message" in error:
            raise ValueError(error["Error Message"])
        if "Note" in error:
            raise ValueError(f"API limit reached: {error['Note']}")
        raise ValueError(f"API error: {error}")

    # Parse CSV response
    from io import StringIO
    data = pd.read_csv(StringIO(response.text), parse_dates=["timestamp"], index_col="timestamp")

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    # Rename columns to match ProFiT format
    data.columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data.sort_index()  # Oldest first
    data.index.name = "Datetime"

    return data


def download_intraday(ticker: str, api_key: str, interval: str = "60min") -> pd.DataFrame:
    """Download intraday OHLCV data from Alpha Vantage.

    Args:
        ticker: Symbol to download
        api_key: Alpha Vantage API key
        interval: 1min, 5min, 15min, 30min, or 60min

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {ticker} intraday ({interval}) data from Alpha Vantage...")

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}"
        f"&interval={interval}"
        f"&outputsize=full"
        f"&apikey={api_key}"
        f"&datatype=csv"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Check for API errors
    if response.text.startswith("{"):
        import json
        error = json.loads(response.text)
        if "Error Message" in error:
            raise ValueError(error["Error Message"])
        if "Note" in error:
            raise ValueError(f"API limit reached: {error['Note']}")
        raise ValueError(f"API error: {error}")

    from io import StringIO
    data = pd.read_csv(StringIO(response.text), parse_dates=["timestamp"], index_col="timestamp")

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    data.columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data.sort_index()
    data.index.name = "Datetime"

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Download OHLCV data from Alpha Vantage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Get a free API key at: https://www.alphavantage.co/support/#api-key

Examples:
  uv run scripts/download_alphavantage.py --api-key YOUR_KEY
  uv run scripts/download_alphavantage.py --ticker AAPL --api-key YOUR_KEY
  uv run scripts/download_alphavantage.py --ticker SPY --interval 60min --api-key YOUR_KEY

  # Using environment variable:
  export ALPHA_VANTAGE_API_KEY=YOUR_KEY
  uv run scripts/download_alphavantage.py --ticker MSFT

Note: Free API allows 25 requests/day. Use --outputsize compact for testing.
        """,
    )
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Symbol to download (default: SPY)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ALPHA_VANTAGE_API_KEY"),
        help="Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env var)",
    )
    parser.add_argument(
        "--interval",
        default="daily",
        choices=["daily", "1min", "5min", "15min", "30min", "60min"],
        help="Data interval (default: daily)",
    )
    parser.add_argument(
        "--outputsize",
        default="full",
        choices=["compact", "full"],
        help="Output size: compact (100 days) or full (20+ years for daily)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: data/{ticker}_{interval}.csv)",
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: API key required. Use --api-key or set ALPHA_VANTAGE_API_KEY", file=sys.stderr)
        sys.exit(1)

    # Create data directory if needed
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / f"{args.ticker}_{args.interval}.csv"

    try:
        if args.interval == "daily":
            data = download_daily(args.ticker, args.api_key, args.outputsize)
        else:
            data = download_intraday(args.ticker, args.api_key, args.interval)

        # Save to CSV
        data.to_csv(output_path)

        # Print summary
        print(f"\nSaved {len(data):,} bars to {output_path}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Columns: {list(data.columns)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
