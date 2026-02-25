# crypto-trading-lab

A personal lab for researching, backtesting, and tuning crypto trading strategies.

---

## Project Structure

```
crypto-trading-lab/
├── config/
│   └── config.yaml           # API keys, default symbols, timeframes
│
├── data/
│   ├── __init__.py
│   ├── fetch.py              # Pull OHLCV data from exchanges (via ccxt)
│   └── cache/                # Local parquet/csv cache to avoid re-fetching
│
├── strategies/
│   ├── __init__.py
│   ├── base.py               # Abstract base class all strategies inherit from
│   ├── moving_average.py     # Example: MA crossover strategy
│   └── rsi.py                # Example: RSI mean-reversion strategy
│
├── backtesting/
│   ├── __init__.py
│   ├── engine.py             # Runs a strategy over historical data
│   └── metrics.py            # Performance metrics: Sharpe, drawdown, win rate, etc.
│
├── tuning/
│   ├── __init__.py
│   └── optimizer.py          # Parameter search: grid search or Optuna (Bayesian)
│
├── notebooks/
│   └── exploration.ipynb     # Scratch space for interactive exploration
│
├── requirements.txt
└── README.md
```

---

## Module Responsibilities

### `data/`
- **fetch.py** — connects to exchanges via `ccxt`, pulls OHLCV (open/high/low/close/volume) data for a given symbol and timeframe, and caches results locally to avoid redundant API calls.

### `strategies/`
- **base.py** — defines a `BaseStrategy` abstract class with a standard `generate_signals(df)` interface that all strategies must implement. This keeps backtesting and tuning code exchange-agnostic.
- Concrete strategies (e.g. `moving_average.py`, `rsi.py`) each implement `generate_signals` and expose their tunable parameters.

### `backtesting/`
- **engine.py** — takes a strategy + OHLCV dataframe, runs the signals through a simulated portfolio (with configurable fees and slippage), and returns a trade log + equity curve.
- **metrics.py** — computes performance statistics from a trade log: total return, Sharpe ratio, max drawdown, win rate, profit factor.

### `tuning/`
- **optimizer.py** — wraps the backtesting engine to search over strategy parameter spaces. Supports grid search for small spaces and Optuna (Bayesian optimization) for larger ones.

### `notebooks/`
- Jupyter notebooks for interactive exploration, quick plots, and ad-hoc analysis. Not part of the importable package.

---

## Suggested Workflow

1. **Fetch data** — use `data/fetch.py` to pull and cache historical OHLCV data
2. **Build a strategy** — subclass `BaseStrategy` in `strategies/`, implement `generate_signals`
3. **Backtest** — run `backtesting/engine.py` to get an equity curve and trade log
4. **Evaluate** — pass results to `backtesting/metrics.py` to assess performance
5. **Tune** — use `tuning/optimizer.py` to find optimal strategy parameters
6. **Explore** — use notebooks for visualization and iteration

---

## Key Design Decisions

- **`ccxt`** as the data source — supports 100+ exchanges with a unified API
- **Pandas DataFrames** as the core data structure — simple, flexible, notebook-friendly
- **Abstract base class** for strategies — enforces a consistent interface, makes swapping strategies trivial
- **Decoupled tuning** — the optimizer calls the backtester as a black box, so any strategy can be tuned without extra wiring
- **Local caching** — avoids rate limits and speeds up iteration during development

---

## Dependencies

See `requirements.txt`. Core libraries:
- `ccxt` — exchange connectivity and data fetching
- `pandas` — data manipulation
- `numpy` — numerical operations
- `optuna` — Bayesian hyperparameter optimization
- `matplotlib` / `plotly` — visualization
