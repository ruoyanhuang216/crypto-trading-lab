"""Average Directional Index (ADX) trend-strength signal."""

import pandas as pd

from signals.base import BaseSignal


class ADXTrend(BaseSignal):
    """Average Directional Index — measures trend strength and direction.

    Computes Wilder's ADX along with the +DI / -DI directional indicators.
    ADX measures *how strongly* the market is trending (0-100); +DI and -DI
    indicate whether the trend is up or down.

    Output columns added to df:
        adx        — trend strength (0-100). >25 = trending, <20 = ranging.
        plus_di    — positive directional indicator (upward pressure).
        minus_di   — negative directional indicator (downward pressure).
        trend_dir  — +1 (uptrend), -1 (downtrend), 0 (no clear trend).

    Parameters:
        period:           Wilder smoothing window (default 14).
        trend_threshold:  ADX level above which a trend is confirmed (default 25).
    """

    def __init__(self, period: int = 14, trend_threshold: float = 25.0):
        super().__init__(period=period, trend_threshold=trend_threshold)
        self.period = period
        self.trend_threshold = trend_threshold

    @property
    def output_col(self) -> str:
        return "adx"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.period

        # ── True Range ────────────────────────────────────────────────────────
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)

        # ── Directional Movement ──────────────────────────────────────────────
        up_move   = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]

        plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # ── Wilder smoothing (equivalent to EMA with alpha=1/period) ──────────
        alpha = 1.0 / period
        atr       = tr.ewm(alpha=alpha,       min_periods=period, adjust=False).mean()
        smooth_pdm = plus_dm.ewm(alpha=alpha,  min_periods=period, adjust=False).mean()
        smooth_mdm = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        # ── Directional Indicators ────────────────────────────────────────────
        df["plus_di"]  = 100 * smooth_pdm / atr
        df["minus_di"] = 100 * smooth_mdm / atr

        # ── ADX ───────────────────────────────────────────────────────────────
        di_sum  = df["plus_di"] + df["minus_di"]
        di_diff = (df["plus_di"] - df["minus_di"]).abs()
        dx      = (100 * di_diff / di_sum).fillna(0)
        df["adx"] = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        # ── Trend direction: only signal when ADX confirms a trend ────────────
        trending = df["adx"] >= self.trend_threshold
        df["trend_dir"] = 0
        df.loc[trending & (df["plus_di"] > df["minus_di"]), "trend_dir"] =  1
        df.loc[trending & (df["plus_di"] < df["minus_di"]), "trend_dir"] = -1

        return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from data.fetch import fetch_ohlcv

    df = fetch_ohlcv(since="2024-01-01", until="2024-06-01")
    signal = ADXTrend(period=14, trend_threshold=25)
    result = signal.compute(df)

    print(signal)
    print(result[["close", "adx", "plus_di", "minus_di", "trend_dir"]].tail(10))
    print(f"\nTrend direction counts:\n{result['trend_dir'].value_counts().sort_index()}")
    trending_pct = (result["adx"] >= 25).mean() * 100
    print(f"\nBars with ADX >= 25 (trending): {trending_pct:.1f}%")
