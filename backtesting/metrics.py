"""Equity-curve-based performance metrics."""

import numpy as np
import pandas as pd


def _detect_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer annualisation factor from the median bar timedelta."""
    deltas = pd.Series(index).diff().dropna()
    if deltas.empty:
        return 365  # fallback
    median_seconds = deltas.dt.total_seconds().median()
    return max(1, int(365 * 24 * 3600 / median_seconds))


def compute_metrics(
    equity: pd.Series,
    periods_per_year: int = None,
) -> dict:
    """Compute performance metrics from an equity curve.

    Args:
        equity: Portfolio value with a DatetimeIndex, starting at 1.0.
        periods_per_year: Annualisation factor. Auto-detected from the index
            if None (e.g. 8760 for 1-hour bars, 365 for daily bars).

    Returns:
        Dictionary of scalar performance metrics.
    """
    if len(equity) < 2:
        raise ValueError("equity must have at least 2 data points")

    if periods_per_year is None:
        periods_per_year = _detect_periods_per_year(equity.index)

    returns = equity.pct_change().dropna()

    # ── Return summary ────────────────────────────────────────────────────────
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    mean_ret = returns.mean()
    std_ret = returns.std()

    # ── Risk-adjusted ─────────────────────────────────────────────────────────
    ann_factor = np.sqrt(periods_per_year)
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else np.nan

    neg_returns = returns[returns < 0]
    mean_neg = neg_returns.mean() if len(neg_returns) > 0 else np.nan
    std_neg = neg_returns.std() if len(neg_returns) > 1 else np.nan
    sortino = (mean_ret / std_neg * ann_factor) if (std_neg and std_neg > 0) else np.nan

    # ── Distribution percentiles ──────────────────────────────────────────────
    p05, p25, p75, p95 = np.percentile(returns, [5, 25, 75, 95])

    # ── Drawdown ──────────────────────────────────────────────────────────────
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()  # most negative value

    n_bars = len(returns)
    annualised_return = (1 + total_return) ** (periods_per_year / n_bars) - 1
    calmar = (annualised_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan

    # ── Win rate ──────────────────────────────────────────────────────────────
    win_rate = (returns > 0).mean()

    return {
        "total_return":    float(total_return),
        "mean_return":     float(mean_ret),
        "std_return":      float(std_ret),
        "sharpe_ratio":    float(sharpe),
        "sortino_ratio":   float(sortino),
        "mean_neg_return": float(mean_neg) if not np.isnan(mean_neg) else float("nan"),
        "std_neg_return":  float(std_neg) if not np.isnan(std_neg) else float("nan"),
        "return_p05":      float(p05),
        "return_p25":      float(p25),
        "return_p75":      float(p75),
        "return_p95":      float(p95),
        "max_drawdown":    float(max_drawdown),
        "calmar_ratio":    float(calmar),
        "win_rate":        float(win_rate),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Make repo root importable when run as `python -m backtesting.metrics`
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.fetch import fetch_ohlcv
    from strategies.moving_average import MACrossover

    print("Fetching OHLCV data (2024-01-01 → 2024-03-01)...")
    df = fetch_ohlcv(since="2024-01-01", until="2024-03-01")

    strategy = MACrossover(fast_period=20, slow_period=50)
    df = strategy.generate_signals(df)

    # Build a simple long/short equity curve from signals
    # signal is position for the *next* bar (shift by 1 to avoid look-ahead)
    position = df["signal"].shift(1).fillna(0)
    bar_returns = df["close"].pct_change().fillna(0)
    strategy_returns = position * bar_returns

    equity = (1 + strategy_returns).cumprod()
    equity.name = "equity"

    print(f"Equity curve: {len(equity)} bars, "
          f"first={equity.iloc[0]:.4f}, last={equity.iloc[-1]:.4f}\n")

    metrics = compute_metrics(equity)

    col_w = max(len(k) for k in metrics) + 2
    print(f"{'Metric':<{col_w}} Value")
    print("-" * (col_w + 12))
    for k, v in metrics.items():
        if np.isnan(v):
            print(f"{k:<{col_w}} {'nan':>10}")
        else:
            print(f"{k:<{col_w}} {v:>10.6f}")
