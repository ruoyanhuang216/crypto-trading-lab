"""Funding-rate features for daily-bar models.

Input:  raw 8h Binance perp funding-rate series (3 obs/day) from
        ``data.funding.fetch_funding_rate``.
Output: daily DataFrame indexed by UTC midnight (matching the OHLCV daily
        index convention) with funding-rate-derived features.

**Leakage discipline.** Features at daily bar ``t`` use only funding-rate
observations with timestamps in ``[t 00:00, t+1 00:00)`` — i.e. the three
fundings settled during bar ``t``. The ``t+1 00:00`` funding (which would
represent end-of-bar information that becomes the next bar's first input)
is NOT included. This matches the convention used elsewhere in the lab
(features at ``t`` are aligned with bar-``t`` close, used to predict the
``t→t+1`` forward return).

Phase 1a candidates (per P-ML19 plan in roadmap):
- ``funding_3d_mean``      — 3-day rolling mean (≈ 9 fundings)
- ``funding_zscore_30d``   — daily-mean z-score over 30 days (extremity)
- ``funding_persistence_3d`` — fraction of last 9 8h fundings same-sign
                               (crowded-trade / leverage-pressure proxy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_funding_mean(funding_8h: pd.DataFrame) -> pd.Series:
    """Aggregate 8h funding rates to UTC-day means.

    Each daily mean averages the (up to three) ``00:00 / 08:00 / 16:00`` UTC
    settlements that fall within that calendar day. Days with fewer than 3
    settlements (rare; typically the launch day) are still included.
    """
    fr = funding_8h["funding_rate"].copy()
    daily = fr.groupby(fr.index.normalize()).mean()
    daily.index = daily.index.tz_convert("UTC") if daily.index.tz is not None else daily.index
    daily.name = "funding_d_mean"
    return daily


def build_funding_features(funding_8h: pd.DataFrame) -> pd.DataFrame:
    """Compute Phase-1a funding-rate features.

    Args:
        funding_8h: DataFrame from ``fetch_funding_rate`` — index is the UTC
            settlement timestamp, columns include ``funding_rate``.

    Returns:
        DataFrame with daily index (UTC midnight) and columns
        ``[funding_3d_mean, funding_zscore_30d, funding_persistence_3d]``.
        Early rows are NaN until the rolling windows are filled — drop after
        joining with the rest of the feature matrix.
    """
    if funding_8h.empty:
        return pd.DataFrame(columns=["funding_3d_mean", "funding_zscore_30d",
                                       "funding_persistence_3d"])

    daily = _daily_funding_mean(funding_8h)

    # 3-day rolling mean of daily-mean funding (= mean of last ~9 fundings).
    f3 = daily.rolling(window=3, min_periods=3).mean().rename("funding_3d_mean")

    # 30-day z-score of daily-mean funding — relative-to-history extremity.
    rm = daily.rolling(window=30, min_periods=30).mean()
    rs = daily.rolling(window=30, min_periods=30).std().replace(0, np.nan)
    fz = ((daily - rm) / rs).rename("funding_zscore_30d")

    # Persistence: fraction of the last 9 8h fundings that are same-sign as
    # the dominant sign in the window (i.e. max(pos_frac, neg_frac)).
    fr = funding_8h["funding_rate"]
    sign = np.sign(fr).replace(0, np.nan)              # 0-rate ignored in persistence count
    pos_frac = (sign == 1).rolling(9, min_periods=9).mean()
    neg_frac = (sign == -1).rolling(9, min_periods=9).mean()
    persist = np.maximum(pos_frac, neg_frac)
    # Reduce to daily — last persistence value within the day.
    persist_daily = persist.groupby(persist.index.normalize()).last().rename("funding_persistence_3d")
    persist_daily.index = persist_daily.index.tz_convert("UTC") if persist_daily.index.tz is not None else persist_daily.index

    out = pd.concat([f3, fz, persist_daily], axis=1)
    return out
