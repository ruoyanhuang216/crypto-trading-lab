"""Multi-period momentum features for ML models.

These features encode *cumulative price movement over multiple weeks/months*,
complementing the single-bar oscillators in technical.py. The key hypothesis
(P-ML7): the bull model fails to distinguish early-trend from late-trend
because the existing 12 features carry no information about whether price has
been trending for 2–8 weeks. Multi-period momentum fills that gap.

All features are:
  - Causal: log(close[t] / close[t-w]) uses only prices known at bar t.
  - Dimensionless: log-return ratios and z-scores are price-level independent.

Warmup requirements (NaN at start of series):
  - ret_5:          5 bars
  - ret_20:        20 bars
  - ret_60:        60 bars
  - mom_zscore_20: 20 + zscore_window bars  (default 20+60 = 80 bars)
  - ret_5_minus_20: 20 bars
"""

import numpy as np
import pandas as pd


def build_momentum_features(
    df:             pd.DataFrame,
    windows:        tuple = (5, 20, 60),
    zscore_window:  int   = 60,
) -> pd.DataFrame:
    """Compute multi-period momentum features from an OHLCV DataFrame.

    Features:
        ret_{w}         — log(close[t] / close[t-w]): cumulative w-bar log return.
                          Captures the full price move over w bars, unlike the
                          single-lag returns (ret_lag{n}) which capture one bar.
        mom_zscore_20   — (ret_20 - rolling_mean(ret_20, zscore_window))
                          / rolling_std(ret_20, zscore_window).
                          Normalises the 20-bar return by its own recent history,
                          flagging whether the current trend is unusually strong.
        ret_5_minus_20  — ret_5 - ret_20: short-term minus medium-term momentum.
                          Positive = recent acceleration; negative = exhaustion.

    Args:
        df:             OHLCV DataFrame with DatetimeIndex.
        windows:        Tuple of lookback periods in bars. Default (5, 20, 60).
        zscore_window:  Rolling window for mom_zscore_20 normalisation.
                        Default 60 (~3 months daily).

    Returns:
        DataFrame of feature columns aligned to df's index.
        NaN rows (warm-up) are NOT dropped — callers should dropna() after
        aligning with labels.
    """
    feats     = pd.DataFrame(index=df.index)
    log_close = np.log(df["close"])
    ret_by_w  = {}

    # ── Cumulative log returns ────────────────────────────────────────────────
    for w in windows:
        r = log_close - log_close.shift(w)
        feats[f"ret_{w}"] = r
        ret_by_w[w] = r

    # ── Normalised momentum (20-bar z-score) ──────────────────────────────────
    if 20 in ret_by_w:
        ret_20  = ret_by_w[20]
        rm      = ret_20.rolling(zscore_window, min_periods=20).mean()
        rs      = ret_20.rolling(zscore_window, min_periods=20).std().replace(0, np.nan)
        feats["mom_zscore_20"] = (ret_20 - rm) / rs

    # ── Momentum acceleration (5-bar vs 20-bar) ───────────────────────────────
    if 5 in ret_by_w and 20 in ret_by_w:
        feats["ret_5_minus_20"] = ret_by_w[5] - ret_by_w[20]

    return feats
