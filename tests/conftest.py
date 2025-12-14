"""Shared fixtures for ProFiT tests."""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing.

    Creates 5000 hourly bars of synthetic price data using a random walk.
    """
    np.random.seed(42)
    n_bars = 5000

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="h")

    # Random walk for price
    returns = np.random.randn(n_bars) * 0.001
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLCV
    data = pd.DataFrame(
        {
            "Open": close * (1 + np.random.randn(n_bars) * 0.001),
            "High": close * (1 + np.abs(np.random.randn(n_bars) * 0.002)),
            "Low": close * (1 - np.abs(np.random.randn(n_bars) * 0.002)),
            "Close": close,
            "Volume": np.random.randint(1000, 10000, n_bars),
        },
        index=dates,
    )

    return data


@pytest.fixture
def small_data(sample_data):
    """Smaller subset for quick tests."""
    return sample_data.iloc[:500]


@pytest.fixture
def medium_data(sample_data):
    """Medium subset for tests requiring more data."""
    return sample_data.iloc[:2000]
