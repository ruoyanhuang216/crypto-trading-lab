"""Technical indicator features for ML models.

All features are:
  - Causal: only use data available at bar t to compute feature at bar t.
  - Normalised: dimensionless ratios or bounded oscillators, so that models
    trained on one price level generalise to another.
"""

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    alpha    = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder Average True Range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX (trend strength, 0-100)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0),   0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    alpha     = 1.0 / period
    atr_s     = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_p  = plus_dm.ewm(alpha=alpha,  min_periods=period, adjust=False).mean()
    smooth_m  = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di   = 100 * smooth_p / atr_s
    minus_di  = 100 * smooth_m / atr_s
    di_sum    = plus_di + minus_di
    dx        = (100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()


def build_technical_features(
    df: pd.DataFrame,
    bb_period:    int   = 20,
    rsi_period:   int   = 14,
    atr_period:   int   = 14,
    adx_period:   int   = 14,
    stoch_period: int   = 14,
) -> pd.DataFrame:
    """Compute technical indicator features from an OHLCV DataFrame.

    All output values are dimensionless (ratios, percentages, 0-100 oscillators)
    so they are directly comparable across different price levels and timeframes.

    Features:
        rsi               — RSI(rsi_period): 0-100 momentum oscillator
        bb_width          — (upper - lower) / mid: normalised band width
        bb_pct_b          — (close - lower) / (upper - lower): position in bands
        bb_zscore         — (close - mid) / std: z-score within bands
        atr_pct           — ATR / close: relative volatility
        macd_hist_norm    — MACD histogram / close: normalised momentum
        volume_ratio      — log(vol / rolling_20_mean): relative volume
        stoch_k           — Stochastic %K: 0-100 oscillator
        adx               — ADX trend strength: 0-100
        di_diff           — (+DI - -DI) normalised by ADX: directional bias

    Returns:
        DataFrame of feature columns with the same index as df.
    """
    feats = pd.DataFrame(index=df.index)

    # ── RSI ──────────────────────────────────────────────────────────────────
    feats["rsi"] = _rsi(df["close"], rsi_period) / 100.0   # scale to 0-1

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid   = df["close"].rolling(bb_period).mean()
    bb_std   = df["close"].rolling(bb_period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    band_rng = (bb_upper - bb_lower).replace(0, np.nan)

    feats["bb_width"]  = band_rng / bb_mid
    feats["bb_pct_b"]  = (df["close"] - bb_lower) / band_rng
    feats["bb_zscore"] = (df["close"] - bb_mid) / bb_std.replace(0, np.nan)

    # ── ATR ──────────────────────────────────────────────────────────────────
    feats["atr_pct"] = _atr(df, atr_period) / df["close"]

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema_fast          = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow          = df["close"].ewm(span=26, adjust=False).mean()
    macd_line         = ema_fast - ema_slow
    signal_line       = macd_line.ewm(span=9, adjust=False).mean()
    feats["macd_hist_norm"] = (macd_line - signal_line) / df["close"]

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_ma             = df["volume"].rolling(20).mean().replace(0, np.nan)
    feats["volume_ratio"] = np.log(df["volume"] / vol_ma)

    # ── Stochastic %K ────────────────────────────────────────────────────────
    lowest_low   = df["low"].rolling(stoch_period).min()
    highest_high = df["high"].rolling(stoch_period).max()
    hl_range     = (highest_high - lowest_low).replace(0, np.nan)
    feats["stoch_k"] = (df["close"] - lowest_low) / hl_range  # 0-1

    # ── ADX + directional bias ────────────────────────────────────────────────
    prev_close = df["close"].shift(1)
    tr_s = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    up_move   = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0),     0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    alpha     = 1.0 / adx_period
    atr_s     = tr_s.ewm(alpha=alpha,     min_periods=adx_period, adjust=False).mean()
    spdm      = plus_dm.ewm(alpha=alpha,  min_periods=adx_period, adjust=False).mean()
    smdm      = minus_dm.ewm(alpha=alpha, min_periods=adx_period, adjust=False).mean()
    plus_di   = 100 * spdm / atr_s
    minus_di  = 100 * smdm / atr_s
    di_sum    = (plus_di + minus_di).replace(0, np.nan)
    dx        = (100 * (plus_di - minus_di).abs() / di_sum).fillna(0)
    adx_vals  = dx.ewm(alpha=alpha, min_periods=adx_period, adjust=False).mean()

    feats["adx"]     = adx_vals / 100.0           # scale 0-1
    feats["di_diff"] = (plus_di - minus_di) / 100.0  # directional bias, ≈ -1 to +1

    return feats
