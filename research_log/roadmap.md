# Roadmap

Current priority list. Updated at the end of each session.
_Last updated: 2026-03-05 (P-ML6 complete; P-ML7 planned)_

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

### Experiment scoreboard (P-ML2 through P-ML6)

| Experiment | Model | OOS Sharpe | OOS Return | Max DD | Key change vs prior |
|---|---|---|---|---|---|
| P-ML2 | LightGBM baseline (3yr) | −0.046 | −32.4% | −76.8% | First model |
| P-ML3 Exp-C | LightGBM + skip-bull (3yr) | +0.280 | +8.8% | −49.8% | Regime gate |
| P-ML4 | RegimeEnsemble (3yr) | +0.227 | −2.5% | −57.3% | Separate bull model |
| **P-ML5** | **RegimeEnsemble (6yr)** | **+0.927** | **+630.2%** | **−68.0%** | Extended dataset |
| P-ML6 | LSTM 30-bar (6yr) | −0.517 | −93.2% | −94.7% | Sequence model |
| *Buy & Hold* | *—* | *+1.379* | *+299.6%* | *−35.4%* | *Benchmark* |

**Current best: P-ML5 RegimeEnsemble (Sharpe +0.927). Gap to B&H: 0.452 Sharpe points.**

### Key learnings

1. **Regime gating is the single most important intervention.** Without it (P-ML2), mean-reversion
   signal inverts in bull markets. P-ML3 Exp-C (skip-bull) produced the first positive OOS equity
   (+8.8%) purely by stopping deployment during the wrong regime.

2. **Data volume matters more than model architecture.** Extending from 3yr to 6yr (P-ML4 → P-ML5)
   yielded a 2× bull-bar multiplier and turned bull-model IC from −0.102 to +0.021. This was a
   bigger gain (+630% vs −2.5%) than any architectural change (P-ML3→P-ML4 was nearly flat).

3. **Sequential LSTM adds no value at daily resolution with this data size.** Each 1d bar already
   summarises intra-day dynamics; ~330 training sequences per fold is insufficient for a neural
   network. LightGBM with single-bar features decisively outperforms (P-ML5 Sharpe +0.927 vs
   P-ML6 LSTM −0.517). LSTM is not worth pursuing unless training data grows to 10+ years or
   the frequency drops to intraday ticks.

4. **Remaining gap to B&H is driven by two specific weaknesses:**
   - Bull model Fold 2 IC = −0.050 (Jan 2021–Jan 2022, the ATH + crash quarter). The model
     fails to distinguish early-trend from late-trend reversal, likely because the current 12
     features are all oscillators/momentum-at-1-bar — none encodes multi-week trend continuation.
   - MaxDD −68.0% vs B&H −35.4%: the equity curve amplifies the 2022 bear market drawdown
     because the non-bull model takes short positions that compound during sharp recoveries.

### Open hypotheses (ordered by expected impact)

| # | Hypothesis | Mechanism | Risk |
|---|---|---|---|
| H1 | Multi-period momentum features fix bull Fold 2 IC | `ret_20`, `ret_60` encode trend duration that oscillators miss | Low — additive to P-ML5 |
| H2 | Longer horizon (3d / 5d) reduces mean-reversion dominance | At 3–5d, trend continuation ≫ single-bar reversal | Medium — requires new label + purge |
| H3 | HMM regime classifier outperforms rule-based SMA200+ADX | Latent-state model detects regime transitions earlier | Medium — new `ml/regime` module |
| H4 | LightGBM hyperparameter tuning (Optuna) squeezes P-ML5 | Better regularisation reduces Fold 2/3 overfitting | Low — marginal gain expected |

---

## ML Track — Next Planned Experiment

### P-ML7. Momentum feature engineering — fix bull model Fold 2 IC

**Priority: HIGH — directly addresses the primary remaining weakness of P-ML5.**

**Hypothesis (H1):** The current 12-feature set is dominated by oscillators that encode
*mean-reversion at 1-bar* (`bar_ret`, `bb_zscore`, `rsi`, `macd_hist_norm`). These features
carry no information about whether price has been trending for 2–8 weeks. Adding explicit
multi-period momentum features will give the bull model a "trend continuation" signal,
specifically fixing Fold 2 (Jan 2021–Jan 2022: ATH + crash) where bull IC = −0.050.

**Features to add (candidate set):**

| Feature | Formula | Rationale |
|---|---|---|
| `ret_5` | `log(close / close.shift(5))` | 1-week momentum |
| `ret_20` | `log(close / close.shift(20))` | 1-month momentum |
| `ret_60` | `log(close / close.shift(60))` | 1-quarter momentum |
| `mom_zscore_20` | `(ret_20 − rolling_mean(ret_20, 60)) / rolling_std(ret_20, 60)` | Normalised trend strength |
| `ret_5_minus_20` | `ret_5 − ret_20` | Short-term vs medium-term acceleration |

Selection: run IC analysis (same method as P-ML1) to retain only features with |IC| > 0.01
and low collinearity with existing features before adding to the ensemble.

**Methodology:**
1. Add candidate features to `ml/features/technical.py` (or new `ml/features/momentum.py`)
2. IC screen: compute Spearman IC per feature across all 5 folds separately for bull / non-bull bars
3. Augment feature set: `FEATURES_V2 = FEATURES + [selected momentum features]`
4. Re-run P-ML5 walk-forward with `RegimeEnsemble` on `FEATURES_V2` — same 6yr data, same splits
5. Compare per-fold bull model IC: P-ML5 vs P-ML7, focus on Fold 2
6. Compare aggregate equity: Sharpe, Return, MaxDD vs P-ML5 baseline

**Success criteria:** Bull model Fold 2 IC turns positive OR overall Mean Bull IC > +0.050
(vs P-ML5 +0.021). Aggregate Sharpe > +0.927.

**Files:**
- Create `ml/features/momentum.py` — `build_momentum_features(df)` returning `ret_5`, `ret_20`, `ret_60`, `mom_zscore_20`, `ret_5_minus_20`
- Edit `ml/features/__init__.py` — integrate momentum into `build_feature_matrix()`
- Create `notebooks/p_ml7_momentum_features.ipynb` — 5-section notebook (IC screen → augmented walk-forward → IC comparison → equity comparison → conclusions)
- Edit `research_log/findings.md` — log F14 after execution

**If hypothesis rejected:** Escalate to H2 (longer horizon). A 3d forward return would shift
the target from mean-reversion territory into trend-following territory, potentially making the
current oscillator features useful in a different way. Requires adjusting `HORIZON=3`,
`PURGE=3` and revalidating all fold IC.

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
