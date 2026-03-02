"""Rule-based 3-state market regime detector.

States:
    'bull'    — close > SMA(long_ma)  AND  ADX >= adx_thresh  (trending up)
    'bear'    — close <= SMA(long_ma) AND  ADX >= adx_thresh  (trending down)
    'ranging' — ADX < adx_thresh  (no clear directional trend)

Design principles:
  - Fully causal: all indicators use only past-bar rolling windows.
  - No fitting step required: rule-based with fixed hyperparameters.
  - Warmup bars (where rolling windows produce NaN) default to 'ranging'.

Typical usage::

    rc  = RegimeClassifier()
    reg = rc.transform(df)     # df is an OHLCV DataFrame
    print(reg["regime"].value_counts())
"""

import pandas as pd

from ml.features.technical import _adx


class RegimeClassifier:
    """Rule-based 3-state market regime detector.

    Combines a long simple moving average (directional bias) and the
    Average Directional Index (trend strength) to classify each bar into
    one of three states: 'bull', 'bear', or 'ranging'.

    Args:
        long_ma:    SMA period for directional bias (default 200).
        adx_period: ADX lookback period in bars (default 14).
        adx_thresh: ADX threshold above which a trend is considered active
                    (default 25; values below signal a ranging market).
    """

    def __init__(
        self,
        long_ma:    int   = 200,
        adx_period: int   = 14,
        adx_thresh: float = 25.0,
    ) -> None:
        self.long_ma    = long_ma
        self.adx_period = adx_period
        self.adx_thresh = adx_thresh

    # ── Public API ────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute regime labels for every bar in df.

        Args:
            df: OHLCV DataFrame with columns ``open``, ``high``, ``low``,
                ``close``, ``volume`` and a DatetimeIndex.

        Returns:
            DataFrame (same index as df) with columns:

            * ``regime``          — str: ``'bull'`` | ``'bear'`` | ``'ranging'``
            * ``regime_bull``     — int 0/1
            * ``regime_bear``     — int 0/1
            * ``regime_ranging``  — int 0/1
        """
        ma      = df["close"].rolling(self.long_ma).mean()
        adx_raw = _adx(df, self.adx_period)   # 0-100 (not rescaled)

        above_ma = df["close"] > ma
        trending = adx_raw >= self.adx_thresh

        # Default all bars to 'ranging' (covers warmup NaNs as well)
        regime = pd.Series("ranging", index=df.index, dtype="object")
        regime[ above_ma & trending] = "bull"
        regime[~above_ma & trending] = "bear"

        out = pd.DataFrame({"regime": regime}, index=df.index)
        out["regime_bull"]    = (regime == "bull").astype(int)
        out["regime_bear"]    = (regime == "bear").astype(int)
        out["regime_ranging"] = (regime == "ranging").astype(int)
        return out

    def __repr__(self) -> str:
        return (
            f"RegimeClassifier("
            f"long_ma={self.long_ma}, "
            f"adx_period={self.adx_period}, "
            f"adx_thresh={self.adx_thresh})"
        )
