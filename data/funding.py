"""Fetch Binance perpetual funding rate history from data.binance.vision archive.

The standard Binance API (api.binance.com / fapi.binance.com) is geo-blocked
in some jurisdictions (HTTP 451). The S3-fronted public archive at
``data.binance.vision`` is not, and exposes the same data as monthly /
daily zipped CSVs. This module pulls and caches the monthly archive.

Archive layout::

    https://data.binance.vision/data/futures/um/monthly/fundingRate/<SYMBOL>/
        <SYMBOL>-fundingRate-YYYY-MM.zip
        -> contains <SYMBOL>-fundingRate-YYYY-MM.csv with columns:
           calc_time (ms), funding_interval_hours, last_funding_rate

Coverage: BTCUSDT funding rate is available from **2020-01** onward
(monthly archive). Note that V2's 6yr OHLCV dataset starts 2019-03;
adding funding-rate features therefore truncates the walk-forward window
to 2020-01 onward — re-baseline V2 on the same window for apples-to-apples
comparison (see roadmap P-ML19 risk note).
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


_ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
_CACHE_DIR    = Path(__file__).parent / "cache"


def _cache_path(symbol: str) -> Path:
    return _CACHE_DIR / f"binance_funding_{symbol}.parquet"


def _month_url(symbol: str, year: int, month: int) -> str:
    fname = f"{symbol}-fundingRate-{year}-{month:02d}.zip"
    return f"{_ARCHIVE_BASE}/{symbol}/{fname}"


def _fetch_month(symbol: str, year: int, month: int, timeout: int = 15) -> Optional[pd.DataFrame]:
    """Pull one month's funding-rate archive. Returns None if not yet published."""
    url = _month_url(symbol, year, month)
    try:
        raw = urllib.request.urlopen(url, timeout=timeout).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None                              # month not (yet) archived
        raise
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
    df["timestamp"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[["last_funding_rate", "funding_interval_hours"]]
    df = df.rename(columns={"last_funding_rate": "funding_rate",
                             "funding_interval_hours": "interval_h"})
    return df


def fetch_funding_rate(
    symbol:    str  = "BTCUSDT",
    since:     Optional[str] = None,
    until:     Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch perpetual funding-rate history from data.binance.vision.

    Args:
        symbol:    Binance perp symbol (no slash), e.g. ``"BTCUSDT"``.
        since:     Start date (inclusive), e.g. ``"2020-01-01"``. Default 2020-01.
        until:     End date (exclusive), e.g. ``"2025-01-01"``. Default = today.
        use_cache: Read/write a local parquet cache for incremental fetches.

    Returns:
        DataFrame indexed by ``timestamp`` (UTC), columns
        ``[funding_rate, interval_h]``. Roughly 3 rows per UTC day
        (00:00 / 08:00 / 16:00) once 8h cadence is stable.
    """
    since_ts = pd.Timestamp(since or "2020-01-01", tz="UTC")
    until_ts = pd.Timestamp(until, tz="UTC") if until else pd.Timestamp.utcnow().tz_convert("UTC")

    cache = _cache_path(symbol)
    cached: Optional[pd.DataFrame] = None
    if use_cache and cache.exists():
        cached = pd.read_parquet(cache)
        if cached.index.tz is None:
            cached.index = cached.index.tz_localize("UTC")

    # Determine the first month we need to fetch (incremental).
    if cached is not None and not cached.empty:
        last = cached.index.max()
        # Re-pull the last cached month to catch late-arriving rows; everything before is final.
        start_y, start_m = last.year, last.month
    else:
        start_y, start_m = since_ts.year, since_ts.month

    end_y, end_m = until_ts.year, until_ts.month

    pieces = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        df_m = _fetch_month(symbol, y, m)
        if df_m is not None:
            pieces.append(df_m)
        m += 1
        if m == 13:
            y, m = y + 1, 1

    new_df = pd.concat(pieces) if pieces else pd.DataFrame()

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, new_df])
    else:
        merged = new_df

    if not merged.empty:
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    if use_cache and not merged.empty:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache, index=True)

    # Filter to the requested range.
    if not merged.empty:
        merged = merged.loc[(merged.index >= since_ts) & (merged.index < until_ts)]

    return merged
