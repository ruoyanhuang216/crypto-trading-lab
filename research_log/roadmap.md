# Roadmap

Current priority list. Updated at the end of each session.
_Last updated: 2026-03-06 (P-ML7 complete; P-ML8/P-ML9 planned)_

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

### ~~P2. Implement volatility signals~~ ✅ COMPLETE
`BBWidth` and `ATRVolatility` added to `signals/volatility/`.
- `BBWidth(period, num_std)` → `bb_width = (upper−lower)/mid` (dimensionless squeeze indicator)
- `ATRVolatility(period)` → `atr_pct = ATR/close` (normalised bar volatility)
Both formulas match `ml/features/technical.py` exactly for cross-layer consistency.

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

### ~~P-ML5. Extended dataset for regime-specific models~~ ✅ COMPLETE — F12 logged
Extended dataset from 3yr (2022–2025) to **6yr (2019–2025)** giving 2,171 bars.
**Result:** Bull model IC flips positive in 3/4 fitted folds (Mean IC +0.021 vs P-ML4 −0.102).
P-ML5 equity **+630.2% (Sharpe +0.927, MaxDD −68.0%)** — dramatically outperforms P-ML4 (−2.5%)
and P-ML3 Exp-C (+8.8%). OOS covers 2020–2025 including the 2020–21 BTC bull run.
Bull bar multiplier: 2.0× vs P-ML4 (787 vs 398 total training bull bars across folds).
Hypothesis **confirmed**: more bull training data fixes the IC sign.
See `notebooks/p_ml5_extended_dataset.ipynb`, F12.

### ~~P-ML6. LSTM Forecaster~~ ✅ COMPLETE — F13 logged
**Hypothesis:** LSTM with 30-bar sliding-window input captures multi-bar temporal dependencies
that single-bar LightGBM cannot exploit, improving OOS IC and equity on the same 6yr dataset.
**Architecture:** Stacked LSTM (64 → 32 units), Dropout 0.2, EarlyStopping(patience=10).
No regime gating — pure sequence model for direct apples-to-apples comparison with P-ML5.
See `ml/models/lstm.py`, `notebooks/p_ml6_lstm.ipynb`, F13.

---

## ML Track — Scorecard & Learnings

### Experiment scoreboard (P-ML2 through P-ML7)

| Experiment | Model | OOS Sharpe | OOS Return | Max DD | Key change vs prior |
|---|---|---|---|---|---|
| P-ML2 | LightGBM baseline (3yr) | −0.046 | −32.4% | −76.8% | First model |
| P-ML3 Exp-C | LightGBM + skip-bull (3yr) | +0.280 | +8.8% | −49.8% | Regime gate |
| P-ML4 | RegimeEnsemble (3yr) | +0.227 | −2.5% | −57.3% | Separate bull model |
| P-ML5 | RegimeEnsemble (6yr) | +0.927 | +630.2% | −68.0% | Extended dataset |
| P-ML6 | LSTM 30-bar (6yr) | −0.517 | −93.2% | −94.7% | Sequence model |
| **P-ML7** | **RegimeEnsemble + momentum (6yr)** | **+1.261** | **+1997.6%** | **−77.3%** | **+4 momentum features** |
| *Buy & Hold* | *—* | *+1.379* | *+299.6%* | *−35.4%* | *Benchmark* |

**Current best: P-ML7 (Sharpe +1.261). Gap to B&H: 0.118 Sharpe points. MaxDD gap: −41.9pp.**

### Key learnings

1. **Regime gating is the single most important intervention.** Without it (P-ML2), mean-reversion
   signal inverts in bull markets. P-ML3 Exp-C (skip-bull) produced the first positive OOS equity
   (+8.8%) purely by stopping deployment during the wrong regime.

2. **Data volume matters more than model architecture.** Extending from 3yr to 6yr (P-ML4 → P-ML5)
   yielded a 2× bull-bar multiplier and turned bull-model IC from −0.102 to +0.021. Bigger gain
   than any architectural change.

3. **Feature engineering compounds on data and regime work.** P-ML7 adds 4 momentum features to
   the P-ML5 base; ICIR nearly doubles (+1.779 vs +0.888) and Sharpe closes 73% of the gap to
   B&H (0.927 → 1.261 vs target 1.379). Key insight: `ret_20` and `mom_zscore_20` give the bull
   model explicit trend-strength signal; `ret_5` and `ret_5_minus_20` add acceleration.

4. **Sequential LSTM adds no value at daily resolution with this data size.** (~330 sequences/fold
   is too few; each 1d bar already summarises intra-day dynamics.) Not worth revisiting unless
   data grows to 10+ years or frequency drops to intraday.

5. **The Fold 2 ATH+crash failure is a late-trend detection problem, not a feature gap.**
   Momentum features worsen Fold 2 bull IC (−0.128 vs −0.050) — the model sees strong `ret_20`
   at ATH and doubles down on the long, right before the crash. This is fundamentally a
   *regime-within-regime* problem: the model needs to distinguish "early bull" from "overextended
   bull", which requires either (a) a valuation signal (BTC/stock ratio, funding rate) or
   (b) a drawdown / volatility-of-momentum signal.

6. **MaxDD worsened with momentum (−77.3% vs −68.0%).** Stronger IC → stronger positions →
   larger losses on wrong calls. The signal quality improvement outweighs this on Sharpe, but
   a risk overlay (position sizing, drawdown brake) is now the most urgent next step before
   any further model improvement.

### Open hypotheses (ordered by expected impact)

| # | Hypothesis | Status | Mechanism |
|---|---|---|---|
| H1 | Momentum features improve bull IC | ✅ Confirmed (P-ML7) | `ret_20`, `mom_zscore_20` add trend-strength signal |
| H2 | Risk overlay (position sizing + drawdown brake) fixes MaxDD | Planned (P-ML9) | Scale by pred z-score; halve position on DD>20% |
| H3 | Strategy integration (MLStrategy class) | Planned (P-ML8) | Bridge `RegimeEnsemble` into backtesting engine |
| H4 | HMM regime classifier detects late-bull / overextension | Open | Latent-state model for "early vs late" bull discrimination |
| H5 | Optuna tuning on 16-feature P-ML7 model | Open (low priority) | Squeeze remaining gap vs B&H after risk overlay |

---

## ML Track — Next Planned Experiments

### ~~P-ML7. Momentum feature engineering~~ ✅ COMPLETE — F14 logged
Selected features: `ret_5`, `ret_20`, `mom_zscore_20`, `ret_5_minus_20` (4 of 5 candidates
passed |IC_bull| > 0.01). `ret_60` rejected (IC_bull ≈ 0). `ret_20` flagged as collinear with
RSI (r=0.848) but retained for bull-signal contribution.
**Result:** Sharpe +1.261 (vs P-ML5 +0.927), ICIR +1.779, Return +1997.6%. Fold 2 bull IC
worsened (−0.128) — ATH+crash failure is late-trend detection, not a feature gap.
**Caveat:** MaxDD worsened to −77.3%. Risk overlay is urgent.
See `ml/features/momentum.py`, `notebooks/p_ml7_momentum_features.ipynb`, F14.

---

### P-ML8. Strategy integration — `RegimeLGBMStrategy` class
**Priority: HIGH — prerequisite for any production use or proper backtesting.**

The P-ML7 signal (16-feature `RegimeEnsemble`) currently lives only in notebook walk-forward
loops. This experiment wraps it into the `strategies/` framework so it can be backtested with
the standard engine, combined with risk overlays, and eventually deployed.

**Design:**
- Create `strategies/ml/regime_lgbm.py` — `RegimeLGBMStrategy`
  - Constructor accepts a pre-trained `RegimeEnsemble` + `RegimeClassifier`
  - `generate_signals(df)` → position Series using the ensemble's predictions
  - Scaled position sizing: `position = clip(pred_zscore × 0.5, −1, +1)` instead of binary
- Create `strategies/ml/__init__.py`
- Notebook: `notebooks/p_ml8_strategy_integration.ipynb` — validate that the strategy
  produces the same equity curve as the P-ML7 notebook walk-forward, then test scaled sizing

**Files:**
- Create `strategies/ml/__init__.py`
- Create `strategies/ml/regime_lgbm.py`
- Create `notebooks/p_ml8_strategy_integration.ipynb`

---

### P-ML9. Risk overlay — position sizing + drawdown brake
**Priority: HIGH — MaxDD −77.3% is the biggest remaining gap to B&H (−35.4%).**

Implement a post-prediction position transformation layer that:
1. **Scales by confidence:** convert raw prediction to z-score using a rolling 60-bar window;
   `position = clip(pred_zscore × scale, −max_pos, +max_pos)` with `scale=0.5, max_pos=1.0`
2. **Drawdown brake:** when rolling 30-bar equity drawdown exceeds 20%, halve all positions
3. **Bull cap:** cap bull-regime long position at 0.5 (no leverage on overextended bull)

**Expected effect:** reduce MaxDD from −77% toward B&H −35% while preserving most of the
IC-driven return. The scaled positioning also naturally reduces whipsaw losses on marginal predictions.

Test as a wrapper around the P-ML7 `RegimeLGBMStrategy` in the P-ML8 notebook or a dedicated
`notebooks/p_ml9_risk_overlay.ipynb`.

**Files:**
- Create `ml/risk/position_sizing.py` — `ScaledPositionSizer`, `DrawdownBrake`
- Create `notebooks/p_ml9_risk_overlay.ipynb`

---

## Next — Medium Priority

### ~~P3. Test signals on longer timeframes (4h, daily)~~ ✅ COMPLETE — F11 logged
**Why:** F3 showed trend signals have sub-random accuracy at 1h. Lower-frequency
bars have less noise — same signals may have genuine predictive power at 4h/daily.
Notebook: `notebooks/p3_signals_timeframe_comparison.ipynb`.
Dataset: 2022-01-01 → 2025-01-01 (3yr). See F11 for results.

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
