"""RSI mean-reversion strategy."""

import pandas as pd

from .base import BaseStrategy


class RSIMeanReversion(BaseStrategy):
    """Relative Strength Index (RSI) mean-reversion strategy.

    Goes long when RSI is oversold, short when overbought, flat otherwise.
    Uses Wilder's smoothing (EWM) for the RSI calculation.

    Parameters:
        period:     RSI lookback window (default 14).
        oversold:   RSI threshold to go long (default 30).
        overbought: RSI threshold to go short (default 70).
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(period=period, oversold=oversold, overbought=overbought)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing: equivalent to EMA with alpha = 1/period
        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        # RSI = 100 * avg_gain / (avg_gain + avg_loss)
        # Handles avg_loss==0 (RSI=100) and warmup NaNs cleanly
        df["rsi"] = 100 * avg_gain / (avg_gain + avg_loss)

        df["signal"] = 0
        df.loc[df["rsi"] < self.oversold, "signal"] = 1
        df.loc[df["rsi"] > self.overbought, "signal"] = -1

        return df


if __name__ == "__main__":
    from data import fetch_ohlcv

    df = fetch_ohlcv(since="2024-01-01", until="2024-03-01")
    strategy = RSIMeanReversion(period=14, oversold=30, overbought=70)
    result = strategy.generate_signals(df)
    print(result[["close", "rsi", "signal"]].head(20))
    print("...")
    print(result[["close", "rsi", "signal"]].tail(5))
    print(f"\nSignal counts:\n{result['signal'].value_counts().sort_index()}")
