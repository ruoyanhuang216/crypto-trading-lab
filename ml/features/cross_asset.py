"""Cross-asset features for ML models.

These features capture the relationship between BTC and traditional financial
markets, encoding three macro channels identified in P-ML12a:

1. **Institutional rebalancing channel** (``spy_btc_corr_30``):
   Rolling 30-day BTC-SPY return correlation. When institutional investors
   hold both equities and crypto, portfolio rebalancing creates synchronized
   moves. A high correlation regime means BTC is trading as a "levered tech
   stock" and equity signals become informative for BTC direction.
   - Era 4 (Bear): corr = +0.45. Era 5 (ETF): corr = +0.36.
   - Left tail (BTC worst 10%): corr = +0.50.

2. **Liquidity / risk-on channel** (``spy_ret_5``):
   5-day cumulative SPY return. After large SPY drops (>2%), BTC averages
   -2.7% over the next 3 days. This captures the liquidity cascade: equity
   selloffs trigger margin calls and risk reduction across asset classes.
   Not a leading indicator (concurrent), but provides regime context about
   whether the macro environment is risk-on or risk-off.

3. **Dollar / financial conditions channel** (``vix_level_zscore``):
   VIX level z-scored over a 60-day rolling window. BTC-SPY correlation
   ranges from +0.06 (VIX low) to +0.55 (VIX high). High VIX signals
   tightening financial conditions where institutional investors deleverage
   across all risk assets including crypto.

All features require aligned BTC + TradFi data (business days only).
Weekend BTC bars must be excluded prior to feature computation.

Usage::

    from ml.features.cross_asset import build_cross_asset_features

    # btc_df: OHLCV DataFrame, tradfi: dict of DataFrames from data.cross_asset
    feats = build_cross_asset_features(btc_df, tradfi)
"""

import numpy as np
import pandas as pd


CROSS_ASSET_FEATURES = ["spy_btc_corr_30", "spy_ret_5", "vix_level_zscore"]


def build_cross_asset_features(
    btc_df: pd.DataFrame,
    tradfi: dict,
    *,
    corr_window: int = 30,
    spy_ret_window: int = 5,
    vix_zscore_window: int = 60,
) -> pd.DataFrame:
    """Build cross-asset features from aligned BTC and TradFi data.

    Args:
        btc_df:           BTC OHLCV DataFrame with DatetimeIndex.
        tradfi:           Dict mapping ticker -> OHLCV DataFrame.
                          Must include "SPY" and "^VIX".
        corr_window:      Rolling window for BTC-SPY correlation (default 30).
        spy_ret_window:   Lookback for cumulative SPY return (default 5).
        vix_zscore_window: Rolling window for VIX z-score (default 60).

    Returns:
        DataFrame with columns: spy_btc_corr_30, spy_ret_5, vix_level_zscore.
        Index = common business days between BTC and all TradFi assets.
        NaN rows from warmup are NOT dropped.
    """
    if "SPY" not in tradfi:
        raise ValueError("tradfi dict must include 'SPY'")
    if "^VIX" not in tradfi:
        raise ValueError("tradfi dict must include '^VIX'")

    # Align to common business days
    btc = btc_df.copy()
    btc.index = btc.index.normalize()
    btc = btc[~btc.index.duplicated(keep="last")]

    spy = tradfi["SPY"].copy()
    spy.index = spy.index.normalize()
    spy = spy[~spy.index.duplicated(keep="last")]

    vix = tradfi["^VIX"].copy()
    vix.index = vix.index.normalize()
    vix = vix[~vix.index.duplicated(keep="last")]

    # Strip timezone for consistent comparison
    for df in [btc, spy, vix]:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    # Common dates (inner join)
    common = btc.index.intersection(spy.index).intersection(vix.index).sort_values()
    common = pd.DatetimeIndex(common)

    btc_close = btc["close"].reindex(common)
    spy_close = spy["close"].reindex(common)
    vix_close = vix["close"].reindex(common)

    # Log returns
    btc_ret = np.log(btc_close / btc_close.shift(1))
    spy_ret = np.log(spy_close / spy_close.shift(1))

    feats = pd.DataFrame(index=common)

    # ── Feature 1: spy_btc_corr_30 ──────────────────────────────────────
    # Rolling 30-day correlation of daily log returns.
    # Captures the institutional rebalancing channel: when correlation is
    # high, BTC is trading as a risk asset and equity signals are informative.
    feats["spy_btc_corr_30"] = btc_ret.rolling(
        corr_window, min_periods=corr_window // 2
    ).corr(spy_ret)

    # ── Feature 2: spy_ret_5 ────────────────────────────────────────────
    # 5-day cumulative SPY log return.
    # Captures the liquidity/risk-on channel: large equity drops signal
    # institutional deleveraging that cascades into crypto.
    feats["spy_ret_5"] = spy_ret.rolling(spy_ret_window).sum()

    # ── Feature 3: vix_level_zscore ─────────────────────────────────────
    # VIX level z-scored over 60-day rolling window.
    # Captures the financial conditions channel: high VIX = stress,
    # which amplifies BTC-equity correlation and predicts larger BTC moves.
    vix_rm = vix_close.rolling(vix_zscore_window, min_periods=20).mean()
    vix_rs = vix_close.rolling(vix_zscore_window, min_periods=20).std().replace(0, np.nan)
    feats["vix_level_zscore"] = (vix_close - vix_rm) / vix_rs

    return feats
