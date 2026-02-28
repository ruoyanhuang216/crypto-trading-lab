"""ML feature engineering for OHLCV time series."""

from .technical import build_technical_features
from .lag import build_lag_features
from .time import build_time_features


def build_feature_matrix(df, **kwargs) -> "pd.DataFrame":
    """Build the full feature matrix by concatenating all feature groups.

    Args:
        df:      OHLCV DataFrame with DatetimeIndex.
        **kwargs: Passed through to individual feature builders
                  (e.g. bb_period=20, rsi_period=14, lags=(1,2,3,5,10)).

    Returns:
        DataFrame of feature columns aligned to df's index.
        NaN rows (from warm-up periods) are NOT dropped here —
        callers should dropna() after aligning with labels.
    """
    import pandas as pd

    tech  = build_technical_features(df, **{
        k: v for k, v in kwargs.items()
        if k in ("bb_period", "rsi_period", "atr_period", "adx_period", "stoch_period")
    })
    lag   = build_lag_features(df, **{
        k: v for k, v in kwargs.items()
        if k in ("lags", "roll_windows")
    })
    time_ = build_time_features(df)

    return pd.concat([tech, lag, time_], axis=1)
