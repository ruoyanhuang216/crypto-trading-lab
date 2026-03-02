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
│       ├── bollinger_bands.py    # BollingerMeanReversion, BollingerBreakout, BollingerLongOnly
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
│   ├── labels/
│   │   └── returns.py            # forward_return(), direction_label()
│   ├── regime/
│   │   └── classifier.py         # RegimeClassifier — SMA(200) + ADX(25) 3-state detector
│   ├── models/
│   │   ├── tree.py               # LGBMForecaster — LightGBM wrapper with purged WF support
│   │   └── ensemble.py           # RegimeEnsemble — separate bull/non-bull LightGBM models
│   └── validation/
│       └── purged_kfold.py       # purged_wf_splits() — leakage-safe sequential CV
│
├── tuning/                       # Reserved for Optuna parameter optimisation (P7)
│
├── notebooks/
│   ├── bollinger_backtest.ipynb              # Initial Bollinger strategy comparison
│   ├── trend_signals_backtest.ipynb          # ADX/MA slope signal accuracy analysis
│   ├── meanreversion_trend_filter.ipynb      # Trend filters applied to MeanReversion
│   ├── walk_forward_backtest.ipynb           # Walk-forward validation (P4)
│   ├── p1b_longonly_meanreversion.ipynb      # Long-only variant experiment (P1b)
│   ├── ml_feature_engineering.ipynb          # IC analysis across 5 timeframes (P-ML1)
│   ├── ml_baseline_models.ipynb              # LightGBM baseline + purged WF (P-ML2)
│   ├── ml_regime_model.ipynb                 # Regime-aware interventions, 3 experiments (P-ML3)
│   └── ml_regime_specific_models.ipynb       # Regime-specific ensemble models (P-ML4)
│
├── research_log/
│   ├── findings.md               # Distilled index of confirmed findings (F1–F10)
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
- `BollingerLongOnly` — long-only variant of MeanReversion (no short signals)
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

- **`ml/features/`** — `build_feature_matrix(df)` → 34 causal, normalised features
  (technical indicators, lag/rolling statistics, bar structure, cyclical time)
- **`ml/labels/`** — `forward_return(df, horizon)` (log) and `direction_label(df, horizon, threshold)` (ternary)
- **`ml/regime/`** — `RegimeClassifier` — rule-based 3-state market regime detector
  using SMA(200) + ADX(14) with threshold 25. States: `'bull'`, `'bear'`, `'ranging'`.
  Fully causal; NaN warmup bars default to `'ranging'`.
- **`ml/models/`** — `LGBMForecaster` (LightGBM wrapper, StandardScaler, feature importance)
  and `RegimeEnsemble` (trains separate bull/non-bull models per fold; falls back to
  sign-flipping when bull training bars < `min_bull_bars`)
- **`ml/validation/`** — `purged_wf_splits(n, purge_bars)` — leakage-safe sequential
  walk-forward splits with purge gap between train and test

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
| F8 | LightGBM IC is regime-sensitive and sign-unstable; Mean IC=−0.049, equity −32.4% |
| F9 | Regime-aware interventions yield first positive OOS equity (Exp-C +8.8%, Sharpe +0.280) |
| F10 | Regime-specific ensemble: bull model IC stays negative; P-ML4 −2.5% vs Exp-C +8.8% |

Full details in [`research_log/findings.md`](research_log/findings.md).

---

## Workflow

```
1. Fetch data      →  data/fetch.py
2. Build features  →  ml/features/build_feature_matrix(df)
3. Detect regime   →  ml/regime/RegimeClassifier.transform(df)
4. Train model     →  ml/models/LGBMForecaster or RegimeEnsemble
5. Validate        →  ml/validation/purged_wf_splits() + backtesting/walk_forward()
6. Evaluate        →  backtesting/metrics.compute_metrics(equity)
7. Tune            →  tuning/ (Optuna, P7)
8. Log findings    →  research_log/
```

---

## Dependencies

See `requirements.txt`. Core libraries:
- `ccxt` — exchange connectivity and OHLCV fetching
- `pandas` / `numpy` — data manipulation
- `scipy` / `statsmodels` — statistical tests (Spearman IC, Ljung-Box, PACF)
- `scikit-learn` — StandardScaler, train/test utilities
- `lightgbm` — gradient-boosted tree models (P-ML2+)
- `optuna` — Bayesian hyperparameter optimisation (P7)
- `matplotlib` / `plotly` — visualisation
- `pyarrow` — parquet caching
