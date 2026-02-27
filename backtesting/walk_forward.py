"""Walk-forward validation engine.

Sequentially evaluates a strategy on non-overlapping out-of-sample windows,
giving a realistic estimate of performance that cannot be cherry-picked by regime.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Type

import numpy as np
import pandas as pd

# Make repo root importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.metrics import compute_metrics


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_idx:  int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    params:      dict
    equity:      pd.Series    # OOS equity, normalised to start at 1.0
    metrics:     dict         # compute_metrics on OOS equity


@dataclass
class WalkForwardResult:
    """Aggregated results from a complete walk-forward run."""
    windows:     list             # list[WindowResult]
    oos_equity:  pd.Series        # all OOS windows chained, starts at 1.0
    oos_metrics: dict             # compute_metrics on oos_equity
    summary_df:  pd.DataFrame     # per-window key metrics in tabular form


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_splits(
    n: int,
    n_splits: int,
    train_frac: float,
    window_type: str,
) -> list[tuple[int, int, int, int]]:
    """Return index tuples (train_start, train_end, test_start, test_end).

    Args:
        n:           Total number of bars in the dataset.
        n_splits:    Number of OOS folds.
        train_frac:  Fraction of the combined train+test period used for training
                     (only meaningful for rolling windows).
        window_type: 'rolling' or 'anchored'.

    Returns:
        List of (train_start, train_end, test_start, test_end) index tuples.
    """
    if window_type not in ("rolling", "anchored"):
        raise ValueError(f"window_type must be 'rolling' or 'anchored', got {window_type!r}")
    if not (0 < train_frac < 1):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    test_size = n // (n_splits + 1)
    if test_size < 1:
        raise ValueError(f"Not enough bars ({n}) for {n_splits} splits")

    splits = []
    for k in range(n_splits):
        test_start = (k + 1) * test_size
        test_end   = min(test_start + test_size, n)

        if window_type == "rolling":
            train_bars  = round(train_frac / (1 - train_frac) * test_size)
            train_start = max(0, test_start - train_bars)
            train_end   = test_start
        else:  # anchored
            train_start = 0
            train_end   = test_start

        splits.append((train_start, train_end, test_start, test_end))

    return splits


def _stitch_equity(pieces: list[pd.Series]) -> pd.Series:
    """Chain OOS equity pieces multiplicatively into one continuous series.

    Each piece is scaled so it starts exactly where the previous piece ended.
    """
    if not pieces:
        raise ValueError("No equity pieces to stitch")

    stitched: list[pd.Series] = []
    anchor = 1.0

    for piece in pieces:
        scaled = piece / piece.iloc[0] * anchor
        anchor = float(scaled.iloc[-1])
        stitched.append(scaled)

    return pd.concat(stitched)


# ── Main function ─────────────────────────────────────────────────────────────

def walk_forward(
    strategy_cls,
    df: pd.DataFrame,
    params: dict,
    *,
    n_splits: int = 5,
    train_frac: float = 0.6,
    window_type: str = "rolling",
    optimize_fn: Optional[Callable] = None,
) -> WalkForwardResult:
    """Run walk-forward validation for a strategy.

    Args:
        strategy_cls: Strategy class (must accept **params and have generate_signals).
        df:           Full OHLCV DataFrame with DatetimeIndex.
        params:       Default strategy parameters (used when optimize_fn is None).
        n_splits:     Number of out-of-sample folds.
        train_frac:   Training fraction for rolling windows (ignored for anchored).
        window_type:  'rolling' or 'anchored'.
        optimize_fn:  Optional callable(strategy_cls, train_df, params) → dict.
                      Reserved for P7 (Optuna). If None, params are used as-is.

    Returns:
        WalkForwardResult with per-window results and stitched OOS equity.
    """
    n = len(df)
    splits = _make_splits(n, n_splits, train_frac, window_type)

    bar_ret = df["close"].pct_change().fillna(0)

    window_results: list[WindowResult] = []
    equity_pieces:  list[pd.Series]    = []

    for k, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
        train_df = df.iloc[tr_s:tr_e]
        test_df  = df.iloc[te_s:te_e]

        # Skip windows that are too small to compute metrics
        if len(test_df) < 2:
            continue

        # Optionally optimise params on the training window
        window_params = (
            optimize_fn(strategy_cls, train_df, params)
            if optimize_fn is not None
            else params
        )

        # Generate signals on the OOS test window
        sig_df = strategy_cls(**window_params).generate_signals(test_df)

        # Build equity curve (shift signal by 1 bar to avoid look-ahead)
        test_bar_ret = bar_ret.iloc[te_s:te_e]
        position     = sig_df["signal"].shift(1).fillna(0)
        equity_raw   = (1 + position * test_bar_ret).cumprod()

        # Normalise to start at 1.0
        equity = equity_raw / equity_raw.iloc[0]

        metrics = compute_metrics(equity)

        wr = WindowResult(
            window_idx  = k,
            train_start = df.index[tr_s],
            train_end   = df.index[tr_e - 1],
            test_start  = df.index[te_s],
            test_end    = df.index[te_e - 1],
            params      = window_params,
            equity      = equity,
            metrics     = metrics,
        )
        window_results.append(wr)
        equity_pieces.append(equity)

    if not window_results:
        raise ValueError("No valid windows produced — increase data length or reduce n_splits")

    # Stitch OOS equity
    oos_equity = _stitch_equity(equity_pieces)
    oos_equity.name = "oos_equity"
    oos_metrics = compute_metrics(oos_equity)

    # Build summary DataFrame
    rows = []
    for wr in window_results:
        rows.append({
            "test_start":    wr.test_start,
            "test_end":      wr.test_end,
            "n_bars":        len(wr.equity),
            "total_return":  wr.metrics["total_return"],
            "sharpe_ratio":  wr.metrics["sharpe_ratio"],
            "sortino_ratio": wr.metrics["sortino_ratio"],
            "max_drawdown":  wr.metrics["max_drawdown"],
            "win_rate":      wr.metrics["win_rate"],
        })

    summary_df = pd.DataFrame(rows)

    return WalkForwardResult(
        windows     = window_results,
        oos_equity  = oos_equity,
        oos_metrics = oos_metrics,
        summary_df  = summary_df,
    )


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.fetch import fetch_ohlcv
    from strategies.single import MACrossover

    print("Fetching OHLCV data (2024-01-01 → 2024-03-01)...")
    df = fetch_ohlcv(since="2024-01-01", until="2024-03-01")
    print(f"  {len(df)} bars loaded\n")

    params = {"fast_period": 20, "slow_period": 50}

    print("Running walk-forward (MACrossover, 5 splits, rolling)...")
    wf = walk_forward(
        MACrossover,
        df,
        params,
        n_splits    = 5,
        train_frac  = 0.6,
        window_type = "rolling",
    )

    print("\n── Per-window summary ───────────────────────────────────────────────")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(wf.summary_df.to_string(index=False))

    print("\n── Stitched OOS metrics ────────────────────────────────────────────")
    col_w = max(len(k) for k in wf.oos_metrics) + 2
    for k, v in wf.oos_metrics.items():
        val = "nan" if (isinstance(v, float) and np.isnan(v)) else f"{v:.6f}"
        print(f"  {k:<{col_w}} {val:>12}")
