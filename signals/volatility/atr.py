"""Average True Range volatility signal."""

import numpy as np
import pandas as pd

from signals.base import BaseSignal


class ATRVolatility(BaseSignal):
    """Average True Range — measures bar-level price volatility.

    Uses Wilder smoothing (EWM with alpha = 1/period, adjust=False),
    identical to ``ml/features/technical._atr()`` so values are consistent
    between the signal and ML feature layers.

    Output columns added to df:
        atr     — raw ATR in price units (useful for stop-loss sizing)
        atr_pct — ATR / close  [dimensionless, primary output]

    Args:
        period: Wilder smoothing window. Default 14.
    """

    def __init__(self, period: int = 14) -> None:
        super().__init__(period=period)
        self.period = period

    @property
    def output_col(self) -> str:
        return "atr_pct"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df   = df.copy()
        prev = df["close"].shift(1)
        tr   = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev).abs(),
                (df["low"]  - prev).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(
            alpha=1.0 / self.period,
            min_periods=self.period,
            adjust=False,
        ).mean()

        df["atr"]     = atr
        df["atr_pct"] = atr / df["close"].replace(0, np.nan)
        return df
