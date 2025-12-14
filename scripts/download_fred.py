#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.0.0",
#     "pandas-datareader>=0.10.0",
# ]
# ///
"""
Download macroeconomic data from FRED (Federal Reserve Economic Data).

No API key required. Browse available series at: https://fred.stlouisfed.org/

Usage:
    uv run scripts/download_fred.py
    uv run scripts/download_fred.py --series "VIXCLS,FEDFUNDS,UNRATE"
    uv run scripts/download_fred.py --series GDP --start 2010-01-01
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pandas_datareader as pdr


# Common FRED series with descriptions
COMMON_SERIES = {
    "VIXCLS": "VIX Volatility Index",
    "FEDFUNDS": "Federal Funds Rate",
    "UNRATE": "Unemployment Rate",
    "GDP": "Gross Domestic Product",
    "CPIAUCSL": "Consumer Price Index",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "DGS10": "10-Year Treasury Rate",
    "DGS2": "2-Year Treasury Rate",
    "DCOILWTICO": "Crude Oil Price (WTI)",
    "DEXUSEU": "USD/EUR Exchange Rate",
}


def download_series(series_ids: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download one or more series from FRED.

    Args:
        series_ids: List of FRED series IDs
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), defaults to today

    Returns:
        DataFrame with series as columns
    """
    print(f"Downloading {len(series_ids)} series from FRED: {', '.join(series_ids)}")

    dfs = []
    for series_id in series_ids:
        try:
            print(f"  Fetching {series_id}...", end=" ")
            df = pdr.get_data_fred(series_id, start=start, end=end)
            dfs.append(df)
            print(f"OK ({len(df)} rows)")
        except Exception as e:
            print(f"FAILED: {e}")

    if not dfs:
        raise ValueError("No series could be downloaded")

    # Combine all series
    data = pd.concat(dfs, axis=1)
    data.index.name = "Datetime"

    return data


def resample_to_daily(data: pd.DataFrame) -> pd.DataFrame:
    """Resample data to daily frequency with forward-fill.

    Many FRED series are monthly/quarterly, this fills gaps for daily use.
    """
    data = data.resample("D").ffill()
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Download macroeconomic data from FRED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Browse available series at: https://fred.stlouisfed.org/

Examples:
  uv run scripts/download_fred.py                              # Download VIX (default)
  uv run scripts/download_fred.py --series "VIXCLS,FEDFUNDS"   # Multiple series
  uv run scripts/download_fred.py --series GDP --start 2010-01-01
  uv run scripts/download_fred.py --series UNRATE --resample   # Daily resampling

Common series IDs:
  VIXCLS     VIX Volatility Index
  FEDFUNDS   Federal Funds Rate
  UNRATE     Unemployment Rate
  GDP        Gross Domestic Product (quarterly)
  CPIAUCSL   Consumer Price Index (monthly)
  T10Y2Y     10Y-2Y Treasury Spread
  DGS10      10-Year Treasury Rate
  DCOILWTICO Crude Oil Price (WTI)
        """,
    )
    parser.add_argument(
        "--series",
        default="VIXCLS",
        help="FRED series ID(s), comma-separated (default: VIXCLS)",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date YYYY-MM-DD (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample to daily frequency with forward-fill",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: data/fred_{series}.csv or data/macro.csv for multiple)",
    )

    args = parser.parse_args()

    # Parse series list
    series_ids = [s.strip().upper() for s in args.series.split(",")]

    # Create data directory if needed
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif len(series_ids) == 1:
        output_path = data_dir / f"fred_{series_ids[0].lower()}.csv"
    else:
        output_path = data_dir / "macro.csv"

    try:
        data = download_series(series_ids, args.start, args.end)

        if args.resample:
            print("Resampling to daily frequency...")
            data = resample_to_daily(data)

        # Save to CSV
        data.to_csv(output_path)

        # Print summary
        print(f"\nSaved to {output_path}")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Rows: {len(data):,}")
        print(f"Columns: {list(data.columns)}")

        # Show sample
        print("\nFirst few rows:")
        print(data.head().to_string())

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
