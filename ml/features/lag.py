"""Lag and rolling-window features for ML models.

All features use only past data (no look-ahead):
  - Log returns at lags 1..n use close[t-lag] which is fully known at bar t.
  - Rolling statistics over window w are computed on bars [t-w+1 .. t].
"""

import numpy as np
import pandas as pd


def build_lag_features(
    df: pd.DataFrame,
    lags:         tuple = (1, 2, 3, 5, 10, 20),
    roll_windows: tuple = (5, 10, 20),
) -> pd.DataFrame:
    """Compute lag and rolling-window features from an OHLCV DataFrame.

    Features:
        ret_lag{n}       — log-return n bars ago: log(close[t-n+1] / close[t-n])
        ret_mean_{w}     — mean log-return over past w bars
        ret_std_{w}      — std of log-returns over past w bars
        ret_skew_{w}     — skewness of log-returns over past w bars (w>=5)
        bar_ret          — bar return: log(close[t] / open[t])
        hl_range         — (high - low) / close: normalised intra-bar range
        upper_wick       — (high - max(open,close)) / close: upper shadow size
        lower_wick       — (min(open,close) - low) / close: lower shadow size
        vol_log_chg      — log(volume[t] / volume[t-1]): volume momentum

    Returns:
        DataFrame of feature columns with the same index as df.
    """
    feats   = pd.DataFrame(index=df.index)
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # ── Lagged returns ────────────────────────────────────────────────────────
    for lag in lags:
        feats[f"ret_lag{lag}"] = log_ret.shift(lag - 1)
        # shift(lag-1): ret at t-1 means log(close[t-1]/close[t-2]), available at t

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in roll_windows:
        feats[f"ret_mean_{w}"]  = log_ret.shift(1).rolling(w).mean()
        feats[f"ret_std_{w}"]   = log_ret.shift(1).rolling(w).std()
        if w >= 5:
            feats[f"ret_skew_{w}"] = log_ret.shift(1).rolling(w).skew()

    # ── Bar structure ─────────────────────────────────────────────────────────
    feats["bar_ret"]    = np.log(df["close"] / df["open"])
    feats["hl_range"]   = (df["high"] - df["low"]) / df["close"]

    body_top    = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    feats["upper_wick"] = (df["high"] - body_top)    / df["close"]
    feats["lower_wick"] = (body_bottom - df["low"])  / df["close"]

    # ── Volume momentum ───────────────────────────────────────────────────────
    vol_prev = df["volume"].shift(1).replace(0, np.nan)
    feats["vol_log_chg"] = np.log(df["volume"] / vol_prev)

    return feats
