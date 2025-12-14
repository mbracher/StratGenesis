# Phase 2: Seed Strategies

## Objective

Implement the five initial seed strategies as `backtesting.Strategy` subclasses. These strategies use common technical indicators and serve as the starting point for evolutionary improvement.

## Strategy Base Class

From the design document:

> Each strategy is a subclass of `backtesting.Strategy` with an `init` to set up indicators and a `next` to execute trading logic on each new bar.

```python
from backtesting import Strategy
import pandas as pd
import numpy as np
```

---

## Strategy 1: BollingerMeanReversion

> Uses Bollinger Bands (e.g., 20-period SMA and standard deviation). If price crosses below the lower band, go long (expecting mean reversion up); if above the upper band, go short (reverting down).

### Parameters
- `bb_period = 20` - Bollinger band period
- `bb_stddev = 2` - Standard deviation multiplier

### Implementation

```python
class BollingerMeanReversion(Strategy):
    """Bollinger Bands mean-reversion strategy.
    Buys when price crosses below lower Bollinger Band and reverts, sells/shorts when above upper band.
    """
    bb_period = 20
    bb_stddev = 2

    def init(self):
        close = self.data.Close
        self.ma = self.I(pd.Series.rolling, close, self.bb_period).mean()
        self.std = self.I(pd.Series.rolling, close, self.bb_period).std(ddof=0)
        self.upper_band = self.ma + self.bb_stddev * self.std
        self.lower_band = self.ma - self.bb_stddev * self.std

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
```

---

## Strategy 2: CCIStrategy

> Uses Commodity Channel Index (CCI). If CCI falls below -100 (oversold), buy; if above +100 (overbought), short.

### Parameters
- `cci_period = 20` - CCI calculation period

### CCI Formula
```
Typical Price (TP) = (High + Low + Close) / 3
CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
```

### Implementation

```python
class CCIStrategy(Strategy):
    """Commodity Channel Index (CCI) strategy.
    Buys when CCI indicates oversold (< -100), sells/shorts when CCI indicates overbought (> +100).
    """
    cci_period = 20

    def init(self):
        TP = (self.data.High + self.data.Low + self.data.Close) / 3
        sma_tp = self.I(pd.Series.rolling, TP, self.cci_period).mean()
        mad = self.I(pd.Series.rolling, (TP - sma_tp).abs(), self.cci_period).mean()
        self.cci = (TP - sma_tp) / (0.015 * mad)

    def next(self):
        cci = self.cci[-1]
        if not self.position:
            if cci < -100:
                self.buy()
            elif cci > 100:
                self.sell()
        else:
            # Exit on mean reversion to neutral range
            if -50 < cci < 50:
                self.position.close()
```

---

## Strategy 3: EMACrossover

> A fast/slow EMA crossover strategy (e.g., 50-period and 200-period EMA). Buy when fast EMA crosses above slow EMA (bullish momentum) and sell/short when fast crosses below slow.

### Parameters
- `fast_ema = 50` - Fast EMA period
- `slow_ema = 200` - Slow EMA period

### Implementation

```python
class EMACrossover(Strategy):
    """Exponential Moving Average (EMA) Crossover strategy.
    Buys when a fast EMA crosses above a slow EMA, and sells/shorts when fast crosses below slow.
    """
    fast_ema = 50
    slow_ema = 200

    def init(self):
        price = self.data.Close
        self.ema_fast = self.I(pd.Series.ewm, price, span=self.fast_ema).mean()
        self.ema_slow = self.I(pd.Series.ewm, price, span=self.slow_ema).mean()

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
```

---

## Strategy 4: MACDStrategy

> Uses MACD (12-26-9 typical). Buy when MACD line crosses above signal line from below; sell/short when MACD crosses below signal from above.

### Parameters
- `fast = 12` - Fast EMA period
- `slow = 26` - Slow EMA period
- `signal = 9` - Signal line EMA period

### MACD Formula
```
MACD Line = EMA(fast) - EMA(slow)
Signal Line = EMA(signal) of MACD Line
```

### Implementation

```python
class MACDStrategy(Strategy):
    """Moving Average Convergence Divergence (MACD) strategy.
    Uses MACD (fast EMA - slow EMA) and signal line for crossover signals.
    """
    fast = 12
    slow = 26
    signal = 9

    def init(self):
        price = self.data.Close
        ema_fast = self.I(pd.Series.ewm, price, span=self.fast).mean()
        ema_slow = self.I(pd.Series.ewm, price, span=self.slow).mean()
        self.macd = ema_fast - ema_slow
        self.signal_line = self.I(pd.Series.ewm, self.macd, span=self.signal).mean()

    def next(self):
        macd_val = self.macd[-1]
        signal_val = self.signal_line[-1]

        if not self.position:
            # Bullish crossover
            if macd_val > signal_val and self.macd[-2] <= self.signal_line[-2]:
                self.buy()
            # Bearish crossover
            elif macd_val < signal_val and self.macd[-2] >= self.signal_line[-2]:
                self.sell()
        else:
            # Exit on opposite crossover
            if self.position.is_long and macd_val < signal_val:
                self.position.close()
            elif self.position.is_short and macd_val > signal_val:
                self.position.close()
```

---

## Strategy 5: WilliamsRStrategy

> Uses Williams %R oscillator (lookback 14). If %R < -80 (very oversold), buy; if %R > -20 (overbought), sell/short.

### Parameters
- `lookback = 14` - Williams %R lookback period

### Williams %R Formula
```
%R = (Highest High - Close) / (Highest High - Lowest Low) * -100
```

### Implementation

```python
class WilliamsRStrategy(Strategy):
    """Williams %R strategy.
    Buys when %R < -80 (oversold), sells/shorts when %R > -20 (overbought).
    """
    lookback = 14

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        self.highest_high = self.I(pd.Series.rolling, high, self.lookback).max()
        self.lowest_low = self.I(pd.Series.rolling, low, self.lookback).min()
        self.percentR = -100 * (self.highest_high - close) / (self.highest_high - self.lowest_low)

    def next(self):
        perc = self.percentR[-1]
        if not self.position:
            if perc < -80:
                self.buy()
            elif perc > -20:
                self.sell()
        else:
            # Exit when %R returns to mid-range
            if -60 < perc < -40:
                self.position.close()
```

---

## Testing Strategies

Each strategy should be testable independently:

```python
from backtesting import Backtest
import pandas as pd

# Load sample data
data = pd.read_csv("sample_data.csv", parse_dates=True, index_col="Datetime")

# Test a strategy
bt = Backtest(data, EMACrossover, cash=10000, commission=0.002)
result = bt.run()
print(result)
```

## Deliverables

- [x] `BollingerMeanReversion` class
- [x] `CCIStrategy` class
- [x] `EMACrossover` class
- [x] `MACDStrategy` class
- [x] `WilliamsRStrategy` class
- [x] All strategies testable with backtesting.py
