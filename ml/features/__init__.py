"""ML feature engineering for OHLCV time series."""

from .technical import build_technical_features
from .lag import build_lag_features
from .time import build_time_features
from .momentum import build_momentum_features
from .volume import build_volume_features


def build_feature_matrix(
    df,
    include_momentum: bool = False,
    include_volume: bool = False,
    **kwargs,
) -> "pd.DataFrame":
    """Build the full feature matrix by concatenating all feature groups.

    Args:
        df:               OHLCV DataFrame with DatetimeIndex.
        include_momentum: If True, append multi-period momentum features
                          (ret_5, ret_20, ret_60, mom_zscore_20, ret_5_minus_20).
                          Default False for backward compatibility with P-ML2–P-ML6.
        include_volume:   If True, append volume-based sentiment / institutional
                          participation features (see ml/features/volume.py).
                          Pass vol_windows, vol_zscore_win, vol_corr_win in kwargs
                          to override defaults.
                          Default False for backward compatibility with P-ML2–P-ML7.
        **kwargs:         Passed through to individual feature builders
                          (e.g. bb_period=20, rsi_period=14, lags=(1,2,3,5,10),
                          momentum_windows=(5,20,60), zscore_window=60,
                          vol_windows=(7,14,30), vol_zscore_win=30, vol_corr_win=14).

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

    parts = [tech, lag, time_]

    if include_momentum:
        mom = build_momentum_features(df, **{
            k: v for k, v in kwargs.items()
            if k in ("windows", "zscore_window")
        })
        parts.append(mom)

    if include_volume:
        vol_kwargs = {}
        if "vol_windows"    in kwargs: vol_kwargs["windows"]    = kwargs["vol_windows"]
        if "vol_zscore_win" in kwargs: vol_kwargs["zscore_win"] = kwargs["vol_zscore_win"]
        if "vol_corr_win"   in kwargs: vol_kwargs["corr_win"]   = kwargs["vol_corr_win"]
        vol = build_volume_features(df, **vol_kwargs)
        parts.append(vol)

    return pd.concat(parts, axis=1)
