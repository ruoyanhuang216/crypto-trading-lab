"""Volume-based features for ML models — sentiment & institutional participation.

Two theoretical frameworks motivate these features:

1. SENTIMENT THEORY: Volume validates price moves.
   High volume on an up-day = broad conviction (bullish);
   high volume on a down-day = broad panic/distribution (bearish);
   low volume on any move = lack of conviction, likely to reverse.
   Features: vol_signed_ratio (directional volume), vol_price_corr (volume-direction alignment)

2. INSTITUTIONAL PARTICIPATION THEORY: Institutions trade in size.
   BTC has seen growing institutional participation since ~2020 (Grayscale, MicroStrategy),
   accelerating with the launch of US spot ETFs in January 2024 (BlackRock, Fidelity).
   Institutional accumulation leaves footprints: steady consistent volume with up-volume bias;
   block trades show as vol_zscore spikes. As institutional share grows, these signals
   should become more predictive over time.
   Features: vol_zscore (abnormal spikes), vol_cv (consistency), vol_trend (steady accumulation)

All features are:
  - Causal: use only data available through bar t close.
  - Dimensionless: ratios, z-scores, or bounded [-1, +1] to generalise across price scales.

Existing volume features (already in the 12-feature set or full matrix):
  vol_log_chg   — log(vol[t] / vol[t-1])          — 1-day volume change (in FEATURES)
  volume_ratio  — log(vol[t] / 20d_mean)           — vs 20-day average (not in FEATURES)

New features added here target distinct information:
  - Shorter windows (7d, 14d) than the existing 20d ratio
  - Directional/signed volume not currently captured at all
  - Volume consistency (CV) — captures institutional steady vs retail spiky buying
  - Abnormal spikes over longer window (30d) — institutional block-trade signal
  - Volume momentum (7d vs 14d average trend) — is participation accelerating?
  - Volume-price correlation — is volume confirming or diverging from direction?
"""

import numpy as np
import pandas as pd


def build_volume_features(
    df:           pd.DataFrame,
    windows:      tuple = (7, 14, 30),
    zscore_win:   int   = 30,
    corr_win:     int   = 14,
) -> pd.DataFrame:
    """Compute volume-based sentiment and institutional-participation features.

    Features:
        vol_log_ratio_{w}d  — log(vol[t] / rolling_w_mean(vol)):
                              How today's volume compares to the w-day average.
                              Positive = higher than usual, negative = quieter.

        vol_cv_14d          — rolling_14d_std(vol) / rolling_14d_mean(vol):
                              Coefficient of variation of volume over 14 days.
                              Low CV = consistent volume (institutional steady accumulation);
                              High CV = spiky retail-driven volume.

        vol_zscore_30d      — (vol[t] - 30d_mean) / 30d_std:
                              How abnormal today's volume is vs the past month.
                              Captures institutional block trades (large spikes).

        vol_trend_7_14      — log(7d_mean / 14d_mean):
                              Is short-term volume higher than medium-term?
                              Positive = accelerating participation (institutions building);
                              Negative = fading participation.

        vol_signed_ratio_{w}d — sum(vol * sign(bar_ret), w days) / sum(vol, w days):
                              Directional / OBV-weighted volume over w days.
                              Bounded in [-1, +1].  +1 = all volume was on up-days
                              (accumulation); -1 = all volume on down-days (distribution).

        vol_price_corr_14d  — rolling 14d Spearman-approx correlation between
                              bar_ret and log(volume):
                              Positive = volume rises on up-days (trend confirmation);
                              Negative = volume rises on down-days (distribution signal).

    Args:
        df:         OHLCV DataFrame with DatetimeIndex.
        windows:    Lookback windows for rolling ratios and signed-volume.
        zscore_win: Window for vol_zscore (default 30 = ~1 month daily).
        corr_win:   Window for vol_price_corr (default 14 = 2 weeks).

    Returns:
        DataFrame of feature columns aligned to df.index.
        NaN rows (warm-up) are NOT dropped — callers should dropna() after
        aligning with labels.
    """
    feats   = pd.DataFrame(index=df.index)
    vol     = df["volume"]
    bar_ret = np.log(df["close"] / df["open"])      # intra-bar return
    log_vol = np.log(vol.replace(0, np.nan))

    # ── Category 1: Volume level vs rolling average ───────────────────────────
    for w in windows:
        vol_ma = vol.rolling(w, min_periods=max(1, w // 2)).mean().replace(0, np.nan)
        feats[f"vol_log_ratio_{w}d"] = np.log(vol / vol_ma)

    # ── Category 2: Volume consistency (coefficient of variation) ─────────────
    # Low CV = institutional steady flow; high CV = retail episodic spikes
    vol_mean_14 = vol.rolling(14, min_periods=7).mean().replace(0, np.nan)
    vol_std_14  = vol.rolling(14, min_periods=7).std()
    feats["vol_cv_14d"] = vol_std_14 / vol_mean_14

    # ── Category 3: Volume z-score (abnormal spike = institutional block trade) ─
    vol_mean_z = vol.rolling(zscore_win, min_periods=zscore_win // 2).mean()
    vol_std_z  = vol.rolling(zscore_win, min_periods=zscore_win // 2).std().replace(0, np.nan)
    feats["vol_zscore_30d"] = (vol - vol_mean_z) / vol_std_z

    # ── Category 4: Volume trend (7d mean vs 14d mean) ────────────────────────
    # Positive = short-term volume higher than medium-term = accelerating participation
    vol_ma_7  = vol.rolling(7,  min_periods=4).mean().replace(0, np.nan)
    vol_ma_14 = vol.rolling(14, min_periods=7).mean().replace(0, np.nan)
    feats["vol_trend_7_14"] = np.log(vol_ma_7 / vol_ma_14)

    # ── Category 5: Directional / signed volume ───────────────────────────────
    # sign(bar_ret) = +1 if close > open (buy-dominated bar), -1 otherwise
    sign_ret   = np.sign(bar_ret).replace(0, 1)   # flat bars treated as up
    signed_vol = vol * sign_ret

    for w in (7, 14):
        sv_sum  = signed_vol.rolling(w, min_periods=max(1, w // 2)).sum()
        vol_sum = vol.rolling(w,        min_periods=max(1, w // 2)).sum().replace(0, np.nan)
        feats[f"vol_signed_ratio_{w}d"] = sv_sum / vol_sum   # bounded [-1, +1]

    # ── Category 6: Volume-price correlation ──────────────────────────────────
    # Positive = volume rises on up-days (trend confirmation)
    # Negative = volume rises on down-days (distribution warning)
    feats["vol_price_corr_14d"] = (
        bar_ret.rolling(corr_win, min_periods=corr_win // 2)
               .corr(log_vol)
    )

    return feats
