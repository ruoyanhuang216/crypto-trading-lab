"""Bollinger Bands strategies: mean reversion and breakout."""

import pandas as pd

from strategies.base import BaseStrategy


def _add_bands(df: pd.DataFrame, period: int, num_std: float) -> pd.DataFrame:
    """Add middle, upper, and lower Bollinger Bands to df in place."""
    df["bb_mid"] = df["close"].rolling(period).mean()
    df["bb_std"] = df["close"].rolling(period).std()
    df["bb_upper"] = df["bb_mid"] + num_std * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - num_std * df["bb_std"]
    return df


class BollingerMeanReversion(BaseStrategy):
    """Bollinger Bands mean-reversion strategy.

    Goes long when price closes below the lower band (oversold),
    short when price closes above the upper band (overbought),
    and flat when price is inside the bands.

    Parameters:
        period:  Lookback window for the moving average (default 20).
        num_std: Number of standard deviations for the bands (default 2.0).
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(period=period, num_std=num_std)
        self.period = period
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = _add_bands(df, self.period, self.num_std)

        df["signal"] = 0
        df.loc[df["close"] < df["bb_lower"], "signal"] = 1   # oversold → long
        df.loc[df["close"] > df["bb_upper"], "signal"] = -1  # overbought → short

        return df


class BollingerBreakout(BaseStrategy):
    """Bollinger Bands breakout strategy.

    Goes long when price closes above the upper band (bullish momentum),
    short when price closes below the lower band (bearish momentum),
    and flat when price is inside the bands.

    Parameters:
        period:  Lookback window for the moving average (default 20).
        num_std: Number of standard deviations for the bands (default 2.0).
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(period=period, num_std=num_std)
        self.period = period
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = _add_bands(df, self.period, self.num_std)

        df["signal"] = 0
        df.loc[df["close"] > df["bb_upper"], "signal"] = 1   # breakout up → long
        df.loc[df["close"] < df["bb_lower"], "signal"] = -1  # breakout down → short

        return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from data.fetch import fetch_ohlcv

    df = fetch_ohlcv(since="2024-01-01", until="2024-03-01")

    for cls in (BollingerMeanReversion, BollingerBreakout):
        strategy = cls(period=20, num_std=2.0)
        result = strategy.generate_signals(df)
        print(f"\n{strategy}")
        print(result[["close", "bb_lower", "bb_mid", "bb_upper", "signal"]].tail(10))
        print(f"Signal counts:\n{result['signal'].value_counts().sort_index()}")
