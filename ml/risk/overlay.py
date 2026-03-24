"""Risk overlay: drawdown brake + bull cap applied to ML strategy signals.

The overlay transforms raw position signals bar-by-bar, adjusting size based on:

1. **Drawdown brake** — when the strategy's rolling equity drawdown exceeds a
   threshold (e.g. -20%), all subsequent positions are scaled down by a
   multiplier (e.g. 0.5) until drawdown recovers above the threshold.
2. **Bull cap** — caps the maximum *long* position during bull regimes to limit
   exposure to late-bull overextension (the Fold 2 ATH+crash failure from P-ML7).

The overlay is *post-prediction*: it receives already-sized positions (binary or
scaled) and reduces them when risk conditions are triggered.  It never increases
position size.

Usage::

    from ml.risk import RiskOverlay

    overlay = RiskOverlay(
        dd_window=30, dd_threshold=-0.20, dd_multiplier=0.5,
        bull_long_cap=0.5,
    )
    adjusted = overlay.apply(signals, regimes, bar_returns)
"""

from typing import NamedTuple

import numpy as np
import pandas as pd


class OverlayResult(NamedTuple):
    """Result of applying a risk overlay."""
    signals:   pd.Series   # adjusted position Series
    equity:    pd.Series   # overlay's internal equity curve
    dd_active: pd.Series   # boolean Series: True when DD brake was active


class RiskOverlay:
    """Post-prediction risk overlay that reduces positions during drawdowns
    and caps bull-regime longs.

    Args:
        dd_window:      Rolling window (bars) for computing equity drawdown.
                        Default 30.
        dd_threshold:   Drawdown level that triggers the brake (negative float,
                        e.g. -0.20 means -20%).  Default -0.20.
        dd_multiplier:  Position multiplier when drawdown brake is active.
                        0.5 = halve all positions.  Default 0.5.
        bull_long_cap:  Maximum long position allowed during bull regime.
                        1.0 = no cap (disabled).  Default 0.5.
        enable_dd_brake: Whether to apply the drawdown brake.  Default True.
        enable_bull_cap: Whether to apply the bull cap.  Default True.
    """

    def __init__(
        self,
        *,
        dd_window:       int   = 30,
        dd_threshold:    float = -0.20,
        dd_multiplier:   float = 0.5,
        bull_long_cap:   float = 0.5,
        enable_dd_brake: bool  = True,
        enable_bull_cap: bool  = True,
    ) -> None:
        self.dd_window       = dd_window
        self.dd_threshold    = dd_threshold
        self.dd_multiplier   = dd_multiplier
        self.bull_long_cap   = bull_long_cap
        self.enable_dd_brake = enable_dd_brake
        self.enable_bull_cap = enable_bull_cap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        signals:     pd.Series,
        regimes:     pd.Series,
        bar_returns: pd.Series,
    ) -> pd.Series:
        """Apply the risk overlay to position signals.

        The overlay is computed **iteratively** because the drawdown brake
        creates a feedback loop: today's position affects tomorrow's equity,
        which affects whether the brake triggers.

        Args:
            signals:     Position Series (float), e.g. from
                         ``RegimeLGBMStrategy``.  Index = DatetimeIndex.
            regimes:     Regime label Series ('bull'/'bear'/'ranging'),
                         aligned with signals.
            bar_returns: Log-return Series for each bar (close-to-close),
                         aligned with signals.

        Returns:
            Adjusted position Series with the same index as ``signals``.
        """
        n = len(signals)
        sig_arr = signals.values.astype(float).copy()
        ret_arr = bar_returns.values.astype(float)
        reg_arr = regimes.values

        adj_arr    = np.empty(n, dtype=float)
        equity_arr = np.ones(n, dtype=float)
        dd_active  = np.zeros(n, dtype=bool)

        for t in range(n):
            pos = sig_arr[t]

            # ── Bull cap (applied before DD brake) ─────────────────────
            if self.enable_bull_cap and reg_arr[t] == "bull" and pos > self.bull_long_cap:
                pos = self.bull_long_cap

            # ── Drawdown brake ─────────────────────────────────────────
            if self.enable_dd_brake and t > 0:
                # Rolling drawdown over the overlay's own equity curve
                start = max(0, t - self.dd_window)
                window_peak = equity_arr[start : t].max()
                current_dd  = (equity_arr[t - 1] - window_peak) / window_peak
                if current_dd < self.dd_threshold:
                    pos *= self.dd_multiplier
                    dd_active[t] = True

            adj_arr[t] = pos

            # ── Update equity (shifted: position at t trades bar t+1,
            #    but for the *overlay's own* DD tracking we use the
            #    convention that position decided at t is applied to
            #    bar t's return with 1-bar shift handled externally).
            #    Here we track equity using position[t-1] * return[t]
            #    to mirror the backtest engine. ─────────────────────────
            if t == 0:
                equity_arr[t] = 1.0
            else:
                # Position from previous bar applied to this bar's return
                equity_arr[t] = equity_arr[t - 1] * (1 + adj_arr[t - 1] * ret_arr[t])

        return OverlayResult(
            signals=pd.Series(adj_arr, index=signals.index, name="signal"),
            equity=pd.Series(equity_arr, index=signals.index, name="overlay_equity"),
            dd_active=pd.Series(dd_active, index=signals.index, name="dd_brake_active"),
        )

    def __repr__(self) -> str:
        parts = []
        if self.enable_dd_brake:
            parts.append(
                f"dd_brake(window={self.dd_window}, "
                f"thresh={self.dd_threshold}, "
                f"mult={self.dd_multiplier})"
            )
        if self.enable_bull_cap:
            parts.append(f"bull_cap={self.bull_long_cap}")
        return f"RiskOverlay({', '.join(parts)})"
