"""Moving Average Slope trend-direction signal."""

import pandas as pd

from signals.base import BaseSignal


class MASlopeTrend(BaseSignal):
    """Moving average slope — continuous trend direction and strength.

    Fits the slope of a rolling MA over a look-back window and normalises
    it by the current MA level to produce a dimensionless percentage-per-bar
    score. Positive = uptrend, negative = downtrend, near-zero = flat.

    Output columns added to df:
        ma           — rolling simple moving average.
        ma_slope     — raw slope (price units per bar).
        ma_slope_pct — normalised slope (% of MA per bar). Primary output.
        trend_dir    — +1 (slope > threshold), -1 (slope < -threshold), 0 (flat).

    Parameters:
        ma_period:     Lookback for the moving average (default 20).
        slope_window:  Number of bars over which to measure the slope (default 5).
        flat_threshold: |ma_slope_pct| below this is classified as flat (default 0.05).
    """

    def __init__(
        self,
        ma_period: int = 20,
        slope_window: int = 5,
        flat_threshold: float = 0.05,
    ):
        super().__init__(
            ma_period=ma_period,
            slope_window=slope_window,
            flat_threshold=flat_threshold,
        )
        self.ma_period = ma_period
        self.slope_window = slope_window
        self.flat_threshold = flat_threshold

    @property
    def output_col(self) -> str:
        return "ma_slope_pct"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["ma"] = df["close"].rolling(self.ma_period).mean()

        # Slope = (MA[t] - MA[t - slope_window]) / slope_window
        df["ma_slope"] = (df["ma"] - df["ma"].shift(self.slope_window)) / self.slope_window

        # Normalise by MA level → % per bar, comparable across price levels
        df["ma_slope_pct"] = df["ma_slope"] / df["ma"] * 100

        # Classify direction
        df["trend_dir"] = 0
        df.loc[df["ma_slope_pct"] >  self.flat_threshold, "trend_dir"] =  1
        df.loc[df["ma_slope_pct"] < -self.flat_threshold, "trend_dir"] = -1

        return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from data.fetch import fetch_ohlcv

    df = fetch_ohlcv(since="2024-01-01", until="2024-06-01")
    signal = MASlopeTrend(ma_period=20, slope_window=5, flat_threshold=0.05)
    result = signal.compute(df)

    print(signal)
    print(result[["close", "ma", "ma_slope_pct", "trend_dir"]].tail(10))
    print(f"\nTrend direction counts:\n{result['trend_dir'].value_counts().sort_index()}")
