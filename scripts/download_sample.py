#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "yfinance>=0.2.40",
#     "pandas>=2.0.0",
# ]
# ///
"""
Quick-start script to download sample data for ProFiT.

Downloads S&P 500 data (^GSPC) from Yahoo Finance - no configuration needed.

Usage:
    uv run scripts/download_sample.py           # Download 5 years daily data
    uv run scripts/download_sample.py --hourly  # Download 2 years hourly data
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_daily() -> tuple[pd.DataFrame, str]:
    """Download 5 years of daily S&P 500 data."""
    print("Downloading S&P 500 (^GSPC) daily data for 5 years...")

    ticker = yf.Ticker("^GSPC")
    data = ticker.history(period="5y", interval="1d")

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.index = data.index.tz_localize(None)  # Remove timezone for compatibility
    data.index.name = "Datetime"

    return data, "ES_daily.csv"


def download_hourly() -> tuple[pd.DataFrame, str]:
    """Download hourly S&P 500 data (limited to ~2 years by Yahoo)."""
    print("Downloading S&P 500 (^GSPC) hourly data (max available)...")

    ticker = yf.Ticker("^GSPC")
    # Yahoo limits hourly data to ~730 days
    data = ticker.history(period="2y", interval="1h")

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.index = data.index.tz_localize(None)  # Remove timezone for compatibility
    data.index.name = "Datetime"

    return data, "ES_hourly.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Quick-start: download sample data for ProFiT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script downloads S&P 500 index data (^GSPC) from Yahoo Finance.
No API keys or configuration required.

Examples:
  uv run scripts/download_sample.py           # 5 years daily
  uv run scripts/download_sample.py --hourly  # ~2 years hourly

After downloading, run ProFiT:
  uv run python -m profit.main --data data/ES_daily.csv --strategy EMACrossover
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--daily",
        action="store_true",
        default=True,
        help="Download daily data (default, 5 years)",
    )
    group.add_argument(
        "--hourly",
        action="store_true",
        help="Download hourly data (~2 years, limited by Yahoo)",
    )

    args = parser.parse_args()

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    try:
        if args.hourly:
            data, filename = download_hourly()
        else:
            data, filename = download_daily()

        output_path = data_dir / filename

        if data.empty:
            print("Error: No data returned", file=sys.stderr)
            sys.exit(1)

        # Save to CSV
        data.to_csv(output_path)

        # Print summary
        print(f"\nSaved {len(data):,} bars to {output_path}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")

        # Validate for ProFiT
        years = (data.index[-1] - data.index[0]).days / 365
        if years < 5:
            print(f"\nNote: {years:.1f} years of data. Walk-forward optimization needs 5+ years.")
            print("Daily data recommended for full walk-forward analysis.")
        else:
            print(f"\nData spans {years:.1f} years - sufficient for walk-forward optimization.")

        print(f"\nNext steps:")
        print(f"  uv run python -m profit.main --data {output_path} --strategy EMACrossover")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
