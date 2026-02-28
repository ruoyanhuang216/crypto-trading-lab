"""Cyclical time-of-day and day-of-week features.

Sine/cosine encoding preserves the circular nature of time:
  hour 23 is close to hour 0, and Friday is close to Monday.
"""

import numpy as np
import pandas as pd


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode temporal position as cyclical sine/cosine features.

    Features:
        hour_sin / hour_cos  — intraday position (period = 24 hours)
        dow_sin  / dow_cos   — day-of-week position (period = 7 days)

    These are always computable (no NaN from warm-up), so they never
    restrict the usable dataset size.

    Returns:
        DataFrame of 4 feature columns with the same index as df.
    """
    feats = pd.DataFrame(index=df.index)

    hour = df.index.hour
    dow  = df.index.dayofweek   # 0=Monday … 6=Sunday

    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["dow_sin"]  = np.sin(2 * np.pi * dow  / 7)
    feats["dow_cos"]  = np.cos(2 * np.pi * dow  / 7)

    return feats
