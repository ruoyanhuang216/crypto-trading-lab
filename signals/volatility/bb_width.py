"""Bollinger Band width volatility signal."""

import numpy as np
import pandas as pd

from signals.base import BaseSignal


class BBWidth(BaseSignal):
    """Bollinger Band width — normalised measure of market volatility.

    Compresses when volatility is low (pre-breakout squeeze);
    expands during trending or high-volatility periods.

    Output columns added to df:
        bb_mid   — rolling SMA of close (price units)
        bb_upper — upper band: mid + num_std * rolling std (price units)
        bb_lower — lower band: mid - num_std * rolling std (price units)
        bb_width — (upper - lower) / mid  [dimensionless, primary output]

    The ``bb_width`` formula matches ``ml/features/technical.py`` exactly,
    so values are directly comparable between the signal and ML layers.

    Args:
        period:  Lookback window for the rolling mean and std. Default 20.
        num_std: Number of standard deviations for the bands. Default 2.0.
    """

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        super().__init__(period=period, num_std=num_std)
        self.period  = period
        self.num_std = num_std

    @property
    def output_col(self) -> str:
        return "bb_width"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df    = df.copy()
        mid   = df["close"].rolling(self.period).mean()
        std   = df["close"].rolling(self.period).std()
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        df["bb_mid"]   = mid
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
        return df
