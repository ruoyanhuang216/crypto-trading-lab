"""Fetch OHLCV candle data from crypto exchanges via ccxt, with local parquet caching."""

import time
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import yaml


_REPO_ROOT = Path(__file__).parent.parent
_CACHE_DIR = Path(__file__).parent / "cache"


def load_config() -> dict:
    config_path = _REPO_ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _cache_path(exchange_id: str, symbol: str, timeframe: str) -> Path:
    filename = f"{exchange_id}_{symbol.replace('/', '-')}_{timeframe}.parquet"
    return _CACHE_DIR / filename


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


def _fetch_range(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int],
    limit: int,
) -> pd.DataFrame:
    all_rows = []
    current_since = since_ms
    effective_page_size = None  # discovered from first response

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
        if not batch:
            break

        # Use the first response length as the exchange's real page cap
        if effective_page_size is None:
            effective_page_size = len(batch)

        original_len = len(batch)

        if until_ms is not None:
            batch = [row for row in batch if row[0] <= until_ms]

        all_rows.extend(batch)

        if not batch:
            break

        last_ts = batch[-1][0]
        if (until_ms is not None and last_ts >= until_ms) or original_len < effective_page_size:
            break

        current_since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_ohlcv(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    exchange_id: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data, using a local parquet cache for incremental updates.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT". Defaults to config value.
        timeframe: Candle size, e.g. "1h". Defaults to config value.
        since: Start date string, e.g. "2024-01-01". Defaults to config value.
        until: End date string (inclusive). Defaults to now.
        exchange_id: ccxt exchange id. Defaults to config value.
        use_cache: Whether to read/write the local parquet cache.

    Returns:
        DataFrame with DatetimeIndex (UTC) and columns: open, high, low, close, volume.
    """
    config = load_config()
    defaults = config.get("defaults", {})

    exchange_id = exchange_id or config.get("exchange", "binance")
    symbol = symbol or defaults.get("symbol", "BTC/USDT")
    timeframe = timeframe or defaults.get("timeframe", "1h")
    limit = defaults.get("limit", 1000)

    now_ms = int(time.time() * 1000)

    since_ms = None  # type: Optional[int]
    if since is not None:
        since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)

    until_ms = None  # type: Optional[int]
    if until is not None:
        until_ms = int(pd.Timestamp(until, tz="UTC").timestamp() * 1000)

    cache_path = _cache_path(exchange_id, symbol, timeframe)
    cached_df = _load_cache(cache_path) if use_cache else None

    last_cached_ms = None  # type: Optional[int]
    if cached_df is not None and not cached_df.empty:
        last_cached_ms = int(cached_df.index.max().timestamp() * 1000)

    if last_cached_ms is not None:
        fetch_since_ms = last_cached_ms + 1
    elif since_ms is not None:
        fetch_since_ms = since_ms
    else:
        # Default: fetch last 90 days if no since given and no cache
        fetch_since_ms = now_ms - 90 * 24 * 60 * 60 * 1000

    if fetch_since_ms >= now_ms:
        # Cache is already up to date; skip network call
        new_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    else:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        new_df = _fetch_range(exchange, symbol, timeframe, fetch_since_ms, until_ms, limit)

    # Merge cache and new data
    if cached_df is not None and not cached_df.empty:
        merged = pd.concat([cached_df, new_df])
    else:
        merged = new_df

    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged = merged.astype({"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"})

    if use_cache and not merged.empty:
        _save_cache(merged, cache_path)

    # Apply since/until filter for the return value
    result = merged
    if since_ms is not None:
        result = result[result.index >= pd.Timestamp(since_ms, unit="ms", tz="UTC")]
    if until_ms is not None:
        result = result[result.index <= pd.Timestamp(until_ms, unit="ms", tz="UTC")]

    return result


if __name__ == "__main__":
    df = fetch_ohlcv(since="2024-01-01", until="2024-03-01")
    print(df.head())
    print(df.tail())
    print(f"Shape: {df.shape}")
    print(f"Index dtype: {df.index.dtype}")
