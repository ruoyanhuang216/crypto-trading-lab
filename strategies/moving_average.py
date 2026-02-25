"""Moving average crossover strategy."""

import pandas as pd

from .base import BaseStrategy


class MACrossover(BaseStrategy):
    """Dual moving average crossover.

    Goes long when the fast MA is above the slow MA, short when below.

    Parameters:
        fast_period: Lookback window for the fast MA (default 20).
        slow_period: Lookback window for the slow MA (default 50).
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(fast_period=fast_period, slow_period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["fast_ma"] = df["close"].rolling(self.fast_period).mean()
        df["slow_ma"] = df["close"].rolling(self.slow_period).mean()

        df["signal"] = 0
        df.loc[df["fast_ma"] > df["slow_ma"], "signal"] = 1
        df.loc[df["fast_ma"] < df["slow_ma"], "signal"] = -1

        return df


if __name__ == "__main__":
    from data import fetch_ohlcv

    df = fetch_ohlcv(since="2024-01-01", until="2024-03-01")
    strategy = MACrossover(fast_period=20, slow_period=50)
    result = strategy.generate_signals(df)
    print(result[["close", "fast_ma", "slow_ma", "signal"]].head(10))
    print("...")
    print(result[["close", "fast_ma", "slow_ma", "signal"]].tail(5))
    print(f"\nSignal counts:\n{result['signal'].value_counts().sort_index()}")
