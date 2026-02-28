# Roadmap

Current priority list. Updated at the end of each session.
_Last updated: 2026-02-28 (EOD)_

---

## Now — High Priority

### ~~P1. Apply trend filters to MeanReversion~~ ✅ COMPLETE — hypothesis rejected
Filters hurt MeanReversion in Jan–Jun 2024, same as Breakout. Root cause: in a bull
market the strategy fails on almost every trade regardless of regime. Bar-by-bar filters
cannot fix a structural directional mismatch. See F4, F5, [2026-02-25-p1](daily/2026-02-25-p1.md).

### ~~P1b. Long-only MeanReversion variant~~ ✅ COMPLETE — F6 logged
**Finding F6:** Directional bias reduces losses but does not create alpha in a bull market.
- LongOnly: WF Sharpe −0.18, Return −3.6% (vs Baseline −1.12, −20.6%) — clearly better
- TrendFiltered (200MA): WF Sharpe −0.93, barely trades (only 284/8785 bars active); MaxDD best at −12.1%
- None profitable on walk-forward basis over full 2024; signal scarcity (5.7% of bars touch lower band) is the root constraint
- **Action:** Promote `BollingerLongOnly` to `strategies/single/basic/`; park TrendFiltered for P5
See `notebooks/p1b_longonly_meanreversion.ipynb`.

### P2. Implement volatility signals
**Why:** BB width and ATR are direct inputs for position sizing and regime detection.
BB width in particular pairs naturally with the existing Bollinger strategies
(compressing bands = pre-breakout setup).
**Files:** `signals/volatility/bb_width.py`, `signals/volatility/atr.py`

---

## ML Track — new direction

### P-ML1. Feature engineering & IC analysis ✅ COMPLETE — F7 logged
`ml/` module built: `ml/features/` (technical, lag, time), `ml/labels/` (returns, direction).
34 features × 5 timeframes IC-analysed. Key findings:
- Mean reversion dominates at 1h (bar_ret IC=−0.081)
- Daily IC 70% higher than hourly (0.041 vs 0.024); upper_wick IC=0.165 at 1d
- Raw returns show no autocorrelation; squared returns show strong GARCH clustering
- 12-feature set recommended for LightGBM (drop redundant oscillator/volatility duplicates)
See `notebooks/ml_feature_engineering.ipynb`.

### P-ML2. LightGBM baseline model
**Target:** 1d forward log-return; **Features:** 12-feature set from F7;
**Validation:** purged walk-forward (5 folds, 1-day embargo);
**Success criterion:** OOS IC > 0.03 consistently across folds.

---

## Next — Medium Priority

### P3. Test signals on longer timeframes (4h, daily)
**Why:** F3 showed trend signals have sub-random accuracy at 1h. Lower-frequency
bars have less noise — same signals may have genuine predictive power at 4h/daily.
**Effort:** Low — change `timeframe` in config and re-run the existing notebook.

### ~~P4. Walk-forward / train-test split in backtesting~~ ✅ COMPLETE
Walk-forward engine implemented in `backtesting/walk_forward.py`.
Exports `walk_forward()`, `WalkForwardResult`, `WindowResult`.
Notebook `notebooks/walk_forward_backtest.ipynb` validates BollingerMeanReversion
and BollingerBreakout across 5 rolling OOS windows on full-year 2024 BTC/USDT 1h data.

---

## Later — Lower Priority

### P5. Regime-aware advanced strategy
Combine volatility signals (BB width) and trend signals (ADX) into the first
`strategies/single/advanced/` strategy that switches between mean-reversion
and breakout mode based on the detected regime.
**Depends on:** P1, P2

### P6. Per-trade metrics
Add trade-log based metrics (profit factor, avg trade duration, trade count)
to complement the equity-curve metrics in `backtesting/metrics.py`.

### P7. Parameter optimisation with Optuna
The `tuning/` folder exists but is empty. Wire up Optuna to optimise strategy
parameters (BB period, num_std, ADX threshold) with proper cross-validation.
**Depends on:** P4

---

## Parking Lot — Ideas to revisit later

- Pairs trading strategies (`strategies/pairs/`)
- Cross-exchange arbitrage (`strategies/arbitrage/`)
- ML-based regime classifier (`signals/regime/hmm.py`)
- Live paper trading integration
