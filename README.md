# crypto-trading-lab

A personal lab for researching, backtesting, and tuning crypto trading strategies on BTC/USDT.

---

## Project Structure

```
crypto-trading-lab/
│
├── config/
│   └── config.yaml               # Exchange, default symbol/timeframe, API settings
│
├── data/
│   ├── fetch.py                  # OHLCV fetcher via ccxt with local parquet caching
│   └── cache/                    # Auto-managed parquet cache (one file per symbol/timeframe)
│
├── strategies/
│   ├── base.py                   # BaseStrategy: generate_signals(df) → df with 'signal' col
│   └── single/basic/
│       ├── moving_average.py     # MACrossover
│       ├── bollinger_bands.py    # BollingerMeanReversion, BollingerBreakout
│       └── rsi.py                # RSIMeanReversion
│
├── signals/
│   ├── base.py                   # BaseSignal: compute(df) → df with indicator cols
│   └── trend/
│       ├── adx.py                # ADXTrend (trend strength + direction)
│       └── ma_slope.py           # MASlopeTrend (moving-average slope)
│
├── backtesting/
│   ├── metrics.py                # compute_metrics(equity) → Sharpe, Sortino, MaxDD, etc.
│   └── walk_forward.py           # walk_forward() — sequential OOS validation engine
│
├── ml/
│   ├── features/
│   │   ├── technical.py          # 10 technical features (RSI, BB, ATR, ADX, MACD, Stoch)
│   │   ├── lag.py                # 20 lag/rolling/bar-structure features
│   │   └── time.py               # Cyclical hour-of-day and day-of-week features
│   └── labels/
│       └── returns.py            # forward_return(), direction_label()
│
├── tuning/                       # Reserved for Optuna parameter optimisation (P7)
│
├── notebooks/
│   ├── bollinger_backtest.ipynb          # Initial Bollinger strategy comparison
│   ├── trend_signals_backtest.ipynb      # ADX/MA slope signal accuracy analysis
│   ├── meanreversion_trend_filter.ipynb  # Trend filters applied to MeanReversion
│   ├── walk_forward_backtest.ipynb       # Walk-forward validation (P4)
│   ├── p1b_longonly_meanreversion.ipynb  # Long-only variant experiment (P1b)
│   └── ml_feature_engineering.ipynb     # IC analysis across 5 timeframes (P-ML1)
│
├── research_log/
│   ├── findings.md               # Distilled index of confirmed findings (F1–F7)
│   ├── roadmap.md                # Prioritised task list
│   └── daily/                    # Per-session logs
│
├── requirements.txt
└── README.md
```

---

## Module Responsibilities

### `data/`
`fetch_ohlcv(symbol, timeframe, since, until)` — fetches OHLCV bars via `ccxt`, caches
to parquet, and does incremental updates on subsequent calls. Supports any ccxt-compatible
exchange (default: OKX).

### `strategies/`
`BaseStrategy.generate_signals(df)` → returns a copy of `df` with a `signal` column:
`+1` = long, `-1` = short, `0` = flat. Strategies are stateless and fully vectorised.

Current strategies:
- `MACrossover` — dual moving-average crossover
- `BollingerMeanReversion` — long below lower band, short above upper band
- `BollingerBreakout` — long above upper band, short below lower band
- `RSIMeanReversion` — RSI-based overbought/oversold

### `signals/`
`BaseSignal.compute(df)` → returns `df` with indicator columns appended. Signals describe
market conditions; they are inputs to strategies, not trading decisions.

Current signals: `ADXTrend`, `MASlopeTrend`.

### `backtesting/`
- `compute_metrics(equity)` — total return, Sharpe, Sortino, Calmar, MaxDD, win rate
- `walk_forward(strategy_cls, df, params, *, n_splits, train_frac, window_type, optimize_fn)`
  — sequential OOS validation over N non-overlapping folds (rolling or anchored).
  Returns `WalkForwardResult` with per-window `WindowResult` objects, a stitched
  `oos_equity` series, and a `summary_df`.

### `ml/`
Foundation for ML-based price forecasting:
- `build_feature_matrix(df)` — 34 causal, normalised features (technical indicators,
  lag/rolling statistics, bar structure, cyclical time)
- `forward_return(df, horizon)` — log forward-return label at horizon *h*
- `direction_label(df, horizon, threshold)` — ternary {−1, 0, +1} label

---

## Confirmed Findings

| ID | Summary |
|---|---|
| F1 | MeanReversion outperforms Breakout on 1h BTC Jan–Mar 2024 (Sharpe +2.0 vs +0.75) |
| F2 | Trend filters degrade BollingerBreakout (lagging entry kills the edge) |
| F3 | ADX/MA slope signals have sub-random direction accuracy at 1h (~0.47) |
| F4 | MeanReversion is period-dependent: Sharpe +2.0 Jan–Mar vs −0.76 Jan–Jun 2024 |
| F5 | Trend filters do NOT fix MeanReversion in a bull market (hypothesis rejected) |
| F6 | Long-only MeanReversion reduces WF losses 5×; no variant profitable in bull 2024 |
| F7 | IC analysis: mean reversion dominates at 1h; daily IC 70% higher; GARCH confirmed |

Full details in [`research_log/findings.md`](research_log/findings.md).

---

## Workflow

```
1. Fetch data      →  data/fetch.py
2. Build features  →  ml/features/build_feature_matrix(df)
3. Build strategy  →  subclass BaseStrategy in strategies/
4. Backtest        →  backtesting/walk_forward() for OOS validation
5. Evaluate        →  backtesting/metrics.compute_metrics(equity)
6. Tune            →  tuning/ (Optuna, P7)
7. Log findings    →  research_log/
```

---

## Dependencies

See `requirements.txt`. Core libraries:
- `ccxt` — exchange connectivity and OHLCV fetching
- `pandas` / `numpy` — data manipulation
- `scipy` / `statsmodels` — statistical tests (Spearman IC, Ljung-Box, PACF)
- `lightgbm` — gradient-boosted tree models (P-ML2)
- `optuna` — Bayesian hyperparameter optimisation (P7)
- `matplotlib` / `plotly` — visualisation
- `pyarrow` — parquet caching
