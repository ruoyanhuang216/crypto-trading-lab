"""Forward-return and direction labels for ML forecasting.

Labels are constructed to predict what happens *after* bar t, using only
data available at t (close price) as the anchor.

IMPORTANT: labels are shifted into the future relative to the feature row.
  forward_return(df, horizon=1).iloc[t]
    = log(close[t+1] / close[t])      ← the return you earn by holding bar t→t+1

Rows at the tail of the series where the label cannot be computed will be NaN.
Always dropna() after aligning features with labels.
"""

import numpy as np
import pandas as pd


def forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Log return over the next `horizon` bars.

    forward_return[t] = log(close[t+horizon] / close[t])

    Args:
        df:      OHLCV DataFrame.
        horizon: Number of bars to look forward.

    Returns:
        Series of forward log-returns (NaN at tail).
    """
    log_ret = np.log(df["close"].shift(-horizon) / df["close"])
    log_ret.name = f"fwd_ret_{horizon}"
    return log_ret


def direction_label(
    df:        pd.DataFrame,
    horizon:   int   = 1,
    threshold: float = 0.0,
) -> pd.Series:
    """Ternary direction label: +1 (up), -1 (down), 0 (flat).

    direction[t] = +1  if forward_return[t] >  threshold
                 = -1  if forward_return[t] < -threshold
                 =  0  if |forward_return[t]| <= threshold

    A threshold of 0.0 collapses to binary up/down (no flat class).
    A typical small threshold (e.g. 0.001) introduces a neutral zone to
    filter out near-zero moves.

    Args:
        df:        OHLCV DataFrame.
        horizon:   Number of bars to look forward.
        threshold: Minimum absolute return to assign +1 or -1.

    Returns:
        Series of {-1, 0, +1} integer labels (NaN at tail becomes 0).
    """
    fwd = forward_return(df, horizon)
    label = pd.Series(0, index=df.index, name=f"direction_{horizon}", dtype=int)
    label[fwd >  threshold] =  1
    label[fwd < -threshold] = -1
    label[fwd.isna()]        =  0   # tail rows: assign neutral
    return label
