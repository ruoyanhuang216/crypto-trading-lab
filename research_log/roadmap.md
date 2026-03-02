# Roadmap

Current priority list. Updated at the end of each session.
_Last updated: 2026-03-01_

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

### ~~P-ML2. LightGBM baseline model~~ ✅ COMPLETE — F8 logged
**Target:** 1d forward log-return; **Features:** 12-feature set from F7;
**Validation:** purged walk-forward (5 folds, purge=1 bar).
Key finding: Mean IC=−0.049, ICIR=−0.488. IC **sign-unstable** across regimes — model
learns mean-reversion but inverts in bull trends (Fold 3 IC=−0.224, p<0.05). LightGBM
equity −32.4% vs B&H +299.6%. Next: regime detection as meta-feature (P-ML3).
See `notebooks/ml_baseline_models.ipynb`.

### ~~P-ML3. Regime-aware LightGBM~~ ✅ COMPLETE — F9 logged
Three experiments on baseline LightGBM (P-ML2) with regime labels (SMA200 + ADX>25):
- **Exp-A** (regime as feature): marginal improvement — Mean IC −0.040 vs −0.049
- **Exp-B** (flip signal in bull): equity **+33.2% (Sharpe +0.482)** — first positive OOS equity
- **Exp-C** (skip bull entirely): equity **+8.8% (Sharpe +0.280)** — conservative positive result
Key insight: Exp-C is the first deployable-quality signal; Exp-B overfits to regime boundary.
See `ml/regime/classifier.py`, `notebooks/ml_regime_model.ipynb`, F9.

### ~~P-ML4. Regime-specific LightGBM models~~ ✅ COMPLETE — F10 logged
`RegimeEnsemble` trains bull + non-bull `LGBMForecaster` per fold; routes predictions by regime.
**Result:** Bull model IC stays negative (−0.138 to −0.044) — fails to learn trend continuation
on only 128–136 bull training bars. P-ML4 equity = −2.5% (Sharpe +0.227), slightly below
P-ML3 Exp-C (+8.8%, Sharpe +0.280). Hypothesis: need more bull training data (extend to 2019)
or longer horizon (3–5d) for the bull model to learn momentum.
See `ml/models/ensemble.py`, `notebooks/ml_regime_specific_models.ipynb`, F10.

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
