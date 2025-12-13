"""Trading strategies for ProFiT.

This module contains seed strategies and baseline strategies.
"""

from backtesting import Strategy
import pandas as pd
import numpy as np


class BollingerMeanReversion(Strategy):
    """Bollinger Bands mean-reversion strategy.

    Buys when price crosses below lower Bollinger Band and reverts,
    sells/shorts when above upper band.
    """

    bb_period = 20
    bb_stddev = 2

    def init(self):
        close = self.data.Close
        self.ma = self.I(lambda x: pd.Series(x).rolling(self.bb_period).mean(), close)
        self.std = self.I(
            lambda x: pd.Series(x).rolling(self.bb_period).std(ddof=0), close
        )
        self.upper_band = self.I(
            lambda ma, std: ma + self.bb_stddev * std, self.ma, self.std
        )
        self.lower_band = self.I(
            lambda ma, std: ma - self.bb_stddev * std, self.ma, self.std
        )

    def next(self):
        price = self.data.Close[-1]
        ma = self.ma[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]

        # Entry logic
        if price < lower and not self.position:
            self.buy()
        elif price > upper and not self.position:
            self.sell()

        # Exit logic - close when price reverts to mean
        if self.position.is_long and price >= ma:
            self.position.close()
        elif self.position.is_short and price <= ma:
            self.position.close()


class CCIStrategy(Strategy):
    """Commodity Channel Index (CCI) strategy.

    Buys when CCI indicates oversold (< -100),
    sells/shorts when CCI indicates overbought (> +100).
    """

    cci_period = 20

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # Typical Price
        tp = (high + low + close) / 3

        # SMA of Typical Price
        self.sma_tp = self.I(lambda x: pd.Series(x).rolling(self.cci_period).mean(), tp)

        # Mean Absolute Deviation
        def calc_mad(tp_arr, sma_arr):
            tp_series = pd.Series(tp_arr)
            sma_series = pd.Series(sma_arr)
            return (tp_series - sma_series).abs().rolling(self.cci_period).mean()

        self.mad = self.I(calc_mad, tp, self.sma_tp)

        # CCI = (TP - SMA(TP)) / (0.015 * MAD)
        def calc_cci(tp_arr, sma_arr, mad_arr):
            return (tp_arr - sma_arr) / (0.015 * mad_arr)

        self.cci = self.I(calc_cci, tp, self.sma_tp, self.mad)

    def next(self):
        cci = self.cci[-1]
        if np.isnan(cci):
            return

        if not self.position:
            if cci < -100:
                self.buy()
            elif cci > 100:
                self.sell()
        else:
            # Exit on mean reversion to neutral range
            if -50 < cci < 50:
                self.position.close()


class EMACrossover(Strategy):
    """Exponential Moving Average (EMA) Crossover strategy.

    Buys when a fast EMA crosses above a slow EMA,
    and sells/shorts when fast crosses below slow.
    """

    fast_ema = 50
    slow_ema = 200

    def init(self):
        price = self.data.Close
        self.ema_fast = self.I(
            lambda x: pd.Series(x).ewm(span=self.fast_ema, adjust=False).mean(), price
        )
        self.ema_slow = self.I(
            lambda x: pd.Series(x).ewm(span=self.slow_ema, adjust=False).mean(), price
        )

    def next(self):
        fast = self.ema_fast[-1]
        slow = self.ema_slow[-1]

        # Entry logic
        if fast > slow and not self.position:
            self.buy()
        elif fast < slow and not self.position:
            self.sell()

        # Exit logic
        if self.position.is_long and fast < slow:
            self.position.close()
        elif self.position.is_short and fast > slow:
            self.position.close()


class MACDStrategy(Strategy):
    """Moving Average Convergence Divergence (MACD) strategy.

    Uses MACD (fast EMA - slow EMA) and signal line for crossover signals.
    """

    fast = 12
    slow = 26
    signal = 9

    def init(self):
        price = self.data.Close

        # Calculate EMAs
        ema_fast = self.I(
            lambda x: pd.Series(x).ewm(span=self.fast, adjust=False).mean(), price
        )
        ema_slow = self.I(
            lambda x: pd.Series(x).ewm(span=self.slow, adjust=False).mean(), price
        )

        # MACD line
        self.macd = self.I(lambda f, s: f - s, ema_fast, ema_slow)

        # Signal line
        self.signal_line = self.I(
            lambda x: pd.Series(x).ewm(span=self.signal, adjust=False).mean(), self.macd
        )

    def next(self):
        if len(self.macd) < 2:
            return

        macd_val = self.macd[-1]
        signal_val = self.signal_line[-1]
        macd_prev = self.macd[-2]
        signal_prev = self.signal_line[-2]

        if np.isnan(macd_val) or np.isnan(signal_val):
            return

        if not self.position:
            # Bullish crossover
            if macd_val > signal_val and macd_prev <= signal_prev:
                self.buy()
            # Bearish crossover
            elif macd_val < signal_val and macd_prev >= signal_prev:
                self.sell()
        else:
            # Exit on opposite crossover
            if self.position.is_long and macd_val < signal_val:
                self.position.close()
            elif self.position.is_short and macd_val > signal_val:
                self.position.close()


class WilliamsRStrategy(Strategy):
    """Williams %R strategy.

    Buys when %R < -80 (oversold), sells/shorts when %R > -20 (overbought).
    """

    lookback = 14

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        self.highest_high = self.I(
            lambda x: pd.Series(x).rolling(self.lookback).max(), high
        )
        self.lowest_low = self.I(
            lambda x: pd.Series(x).rolling(self.lookback).min(), low
        )

        def calc_percent_r(close_arr, hh_arr, ll_arr):
            return -100 * (hh_arr - close_arr) / (hh_arr - ll_arr)

        self.percentR = self.I(calc_percent_r, close, self.highest_high, self.lowest_low)

    def next(self):
        perc = self.percentR[-1]
        if np.isnan(perc):
            return

        if not self.position:
            if perc < -80:
                self.buy()
            elif perc > -20:
                self.sell()
        else:
            # Exit when %R returns to mid-range
            if -60 < perc < -40:
                self.position.close()


# Baseline strategies


class RandomStrategy(Strategy):
    """Random trading strategy (baseline R0).

    When flat, randomly choose to go long, short, or do nothing.
    If holding a position, exit with 50% probability each time step.
    """

    def init(self):
        # Use numpy RandomState for reproducibility (set a seed if desired)
        self.rs = np.random.RandomState(42)  # fixed seed for repeatable random behavior

    def next(self):
        if not self.position:
            # Choose action uniformly: 0=do nothing, 1=buy, 2=sell(short)
            choice = self.rs.randint(0, 3)
            if choice == 1:
                self.buy()
            elif choice == 2:
                self.sell()
        else:
            # If currently in a trade, decide to exit with 50% probability
            if self.rs.rand() < 0.5:
                self.position.close()


class BuyAndHoldStrategy(Strategy):
    """Buy-and-Hold strategy (baseline B&H).

    Buys with all capital at the first bar and holds the position until the end.
    """

    def init(self):
        self.bought = False

    def next(self):
        if not self.bought:
            # Enter long with all available cash at the first data point
            self.buy()
            self.bought = True
        # No exit until the end (the backtesting engine will mark-to-market the position)


# Strategy registry for easy lookup by name
SEED_STRATEGIES = {
    "BollingerMeanReversion": BollingerMeanReversion,
    "CCIStrategy": CCIStrategy,
    "EMACrossover": EMACrossover,
    "MACDStrategy": MACDStrategy,
    "WilliamsRStrategy": WilliamsRStrategy,
}

BASELINE_STRATEGIES = {
    "RandomStrategy": RandomStrategy,
    "BuyAndHoldStrategy": BuyAndHoldStrategy,
}

ALL_STRATEGIES = {**SEED_STRATEGIES, **BASELINE_STRATEGIES}
