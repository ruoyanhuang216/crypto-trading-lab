"""Fetch traditional-asset OHLCV data via yfinance with local parquet caching.

Mirrors the conventions of ``data/fetch.py``:
  - DatetimeIndex (UTC)
  - Columns: open, high, low, close, volume (lowercase)
  - Parquet cache in ``data/cache/``

Usage::

    from data.cross_asset import fetch_tradfi_ohlcv, fetch_cross_asset_panel

    spy = fetch_tradfi_ohlcv("SPY", since="2019-01-01", until="2025-01-01")
    panel = fetch_cross_asset_panel(
        ["SPY", "QQQ", "GLD", "TLT", "UUP", "^VIX"],
        since="2019-01-01", until="2025-01-01",
    )
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


_CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "").replace("/", "-")
    return _CACHE_DIR / f"yf_{safe}_daily.parquet"


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def fetch_tradfi_ohlcv(
    ticker: str,
    since: str = "2019-01-01",
    until: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV for a traditional asset from yfinance.

    Args:
        ticker:    Yahoo Finance ticker (e.g. "SPY", "^VIX", "GLD").
        since:     Start date string (inclusive).
        until:     End date string (inclusive). Defaults to today.
        use_cache: Whether to read/write the local parquet cache.

    Returns:
        DataFrame with DatetimeIndex (UTC), columns: open, high, low, close, volume.
        For indices like ^VIX, volume may be 0.
    """
    cache = _cache_path(ticker)
    cached = _load_cache(cache) if use_cache else None

    if cached is not None and not cached.empty:
        last = cached.index.max()
        # Re-fetch if cache doesn't cover the requested range
        end_ts = pd.Timestamp(until or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
                              tz="UTC")
        if last >= end_ts - pd.Timedelta(days=3):
            # Cache is fresh enough — just filter and return
            return _filter(cached, since, until)

    # Fetch from yfinance (use yf.download which is more reliable)
    end_date = until or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hist = yf.download(ticker, start=since, end=end_plus,
                       auto_adjust=True, progress=False)

    if hist.empty:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    # yfinance 1.2+ returns MultiIndex columns for single ticker: (Price, Ticker)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    # Normalize column names to lowercase
    hist.columns = [c.lower() for c in hist.columns]
    df = hist[["open", "high", "low", "close", "volume"]].copy()

    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    # Merge with cache
    if cached is not None and not cached.empty:
        df = pd.concat([cached, df])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if use_cache and not df.empty:
        _save_cache(df, cache)

    return _filter(df, since, until)


def _filter(df: pd.DataFrame, since: str, until: Optional[str]) -> pd.DataFrame:
    """Apply date range filter."""
    since_ts = pd.Timestamp(since, tz="UTC")
    result = df[df.index >= since_ts]
    if until is not None:
        until_ts = pd.Timestamp(until, tz="UTC") + pd.Timedelta(days=1)
        result = result[result.index < until_ts]
    return result


def fetch_cross_asset_panel(
    tickers: List[str],
    since: str = "2019-01-01",
    until: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers, returned as a dict.

    Args:
        tickers:   List of Yahoo Finance tickers.
        since:     Start date.
        until:     End date.
        use_cache: Cache flag.

    Returns:
        Dict mapping ticker -> OHLCV DataFrame.
    """
    panel = {}
    for ticker in tickers:
        panel[ticker] = fetch_tradfi_ohlcv(ticker, since=since, until=until,
                                            use_cache=use_cache)
    return panel


def align_to_common_dates(
    btc: pd.DataFrame,
    tradfi: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align BTC and TradFi data to common trading dates.

    BTC trades 7 days/week, TradFi only weekdays. This inner-joins on date,
    dropping BTC weekend bars.

    Returns:
        DataFrame with multi-level columns: (ticker, ohlcv_col).
        Index is the common DatetimeIndex.
    """
    # Normalize all to date-only for joining
    btc_daily = btc.copy()
    btc_daily.index = btc_daily.index.normalize()
    btc_daily = btc_daily[~btc_daily.index.duplicated(keep="last")]

    frames = {"BTC": btc_daily}
    for ticker, df in tradfi.items():
        d = df.copy()
        d.index = d.index.normalize()
        d = d[~d.index.duplicated(keep="last")]
        frames[ticker] = d

    # Inner join on common dates
    common_idx = btc_daily.index
    for df in frames.values():
        common_idx = common_idx.intersection(df.index)
    common_idx = pd.DatetimeIndex(common_idx.sort_values())

    aligned = {}
    for name, df in frames.items():
        aligned[name] = df.reindex(common_idx)

    return pd.concat(aligned, axis=1), common_idx
