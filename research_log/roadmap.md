# Roadmap

Current priority list. Updated at the end of each session.
_Last updated: 2026-03-30 (P-ML14 complete; F22 logged — weekday strategy has best MaxDD)_

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

### Experiment scoreboard (P-ML2 through P-ML8)

| Experiment | Model | OOS Sharpe | OOS Return | Max DD | Key change vs prior |
|---|---|---|---|---|---|
| P-ML2 | LightGBM baseline (3yr) | −0.046 | −32.4% | −76.8% | First model |
| P-ML3 Exp-C | LightGBM + skip-bull (3yr) | +0.280 | +8.8% | −49.8% | Regime gate |
| P-ML4 | RegimeEnsemble (3yr) | +0.227 | −2.5% | −57.3% | Separate bull model |
| P-ML5 | RegimeEnsemble (6yr) | +0.927 | +630.2% | −68.0% | Extended dataset |
| P-ML6 | LSTM 30-bar (6yr) | −0.517 | −93.2% | −94.7% | Sequence model |
| **P-ML7** | **RegimeEnsemble + momentum (6yr)** | **+1.261** | **+1997.6%** | **−77.3%** | **+4 momentum features** |
| P-ML8 | RegimeEnsemble + volume (24f, 6yr) | +0.180 | −43.2% | −91.5% | +8 volume features |
| **P-ML9 binary** | **RegimeLGBMStrategy (16f, 6yr)** | **+1.261** | **+1997.6%** | **−77.3%** | **Strategy class (reproduces P-ML7)** |
| **P-ML9 scaled** | **RegimeLGBMStrategy scaled (16f, 6yr)** | **+1.583** | **+758.7%** | **−33.6%** | **pred_zscore × 0.5 positioning** |
| P-ML10 DD brake | RiskOverlay DD only (16f, 6yr) | +1.334 | +2318.8% | −68.4% | 30-bar DD brake at −20% |
| P-ML10 combined | RiskOverlay DD+bull cap (16f, 6yr) | +1.273 | +1728.8% | −68.4% | DD brake + bull cap 0.5 |
| P-ML10 comb+scaled | RiskOverlay on scaled (16f, 6yr) | +1.518 | +645.9% | −33.2% | DD+bull on P-ML9 scaled |
| P-ML11 Exp-A | HMM features (20f, 6yr) | +1.074 | +1043.0% | −77.2% | +4 HMM one-hot (H4 rejected) |
| P-ML11 Exp-B | HMM gating (gate=0.5, 6yr) | +1.055 | +884.1% | −76.8% | Block late-bull longs (H4 rejected) |
| P-ML12b V3 binary | RegimeEnsemble (19f, biz-day) | +0.680 | +99.3% | −67.3% | +3 cross-asset (hurts binary) |
| **P-ML12b V3 scaled** | **RegimeEnsemble scaled (19f, biz-day)** | **+1.118** | **+241.5%** | **−40.7%** | **+3 cross-asset (helps scaled)** |
| P-ML13 V3 / 7-day scaled | RegimeEnsemble scaled (19f, 7-day ffill) | +1.360 | +742.8% | −36.8% | V2 still wins on 7-day |
| P-ML14 V2-weekday scaled | RegimeEnsemble scaled (16f, weekday-flat) | +1.454 | +408.7% | **−25.9%** | Best MaxDD; trades Sharpe for safety |
| P-ML14 V3-weekday scaled | RegimeEnsemble scaled (19f, weekday-flat) | +1.036 | +302.3% | −38.0% | V3 loses to V2 even weekday-only |
| *Buy & Hold* | *—* | *+1.052* | *+876.6%* | *−76.6%* | *Benchmark* |

**Current best: P-ML9 scaled / V2 / 7-day (Sharpe +1.583, MaxDD −33.6%). Confirmed by P-ML13 unified comparison.**
P-ML13 tested V2 vs V3 on both 7-day and business-day datasets. V2 wins 4/6. Cross-asset features
help on business-day data but not on 7-day (weekend forward-fill noise). V2 remains the champion.

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

7. **Scaled position sizing via prediction z-score is the biggest risk improvement.** P-ML9 scaled
   mode (pred_zscore × 0.5, 60-bar window) improves Sharpe from +1.261 to +1.583 and reduces MaxDD
   from −77.3% to −33.6%, beating B&H on both metrics. The return is lower (+758.7% vs +1997.6%)
   but the risk-adjusted improvement is dramatic. Mechanism: marginal predictions get near-zero
   positions, cutting whipsaw losses while preserving high-conviction trades.

8. **Portfolio-level risk overlays are redundant when per-prediction confidence scaling is active.**
   P-ML10 DD brake (reactive, portfolio-level) improves binary signals (Sharpe +1.234 → +1.334)
   but is marginally harmful on scaled signals (+1.583 → +1.518). The z-score scaling already
   gives near-zero positions on uncertain predictions, making the reactive brake fire too late
   to add value. Bull cap is blunt — it can't distinguish early vs late bull, hurting correctly-
   predicted bull bars. Lesson: invest in *per-prediction* confidence rather than portfolio brakes.

9. **Discretized regime features add no value when the raw observations are already in the feature set.**
   P-ML11 HMM uses ret_20, atr_pct, mom_zscore_20, ret_5_minus_20 as observations — all already
   in FEATURES_V2. The one-hot state encoding is a lossy discretization that LightGBM cannot exploit
   beyond what it already learns from the continuous inputs. Sharpe dropped (+1.234 → +1.074) due to
   added noise from 4 extra binary features. Gating late-bull longs is too blunt (blocks entire
   overextension period, not just the crash). The Fold 2 failure requires *timing* signals (when the
   bull ends), not *classification* signals (that it's extended).

10. **Exogenous cross-asset features add genuine value — but only with scaled positioning.**
    P-ML12b added 3 features from SPY/VIX (institutional correlation, equity momentum, VIX stress).
    V3 scaled Sharpe improved from +0.656 to +1.118 (+70%), while V3 binary *worsened* (0.931→0.680).
    The model learned economically intuitive relationships: `spy_btc_corr_30` is the #2 bull feature
    (institutional link strength), `spy_ret_5` is the #1 non-bull feature (equity risk-off drives
    crypto bears). Scaled positioning is critical because it prevents the additional features from
    causing aggressive positions on ambiguous signals (the P-ML8 overfitting failure mode).

11. **Macro channels are real but not exploitable at daily frequency for 24/7 crypto.**
    The cross-asset research arc (P-ML12a through P-ML14) established that BTC-SPY
    correlation is driven by institutional rebalancing (+0.45 in bear), asymmetric tail
    dependence (+0.50 in crises), and VIX-mediated stress. These channels are economically
    significant but the daily-frequency features cannot overcome the weekend forward-fill
    noise in a 7-day BTC model. V3 (19f) loses to V2 (16f) in every matched comparison.
    However, the research produced an actionable finding: V2-weekday-flat achieves the
    best MaxDD (−25.9%) by avoiding weekend exposure, a genuine risk reduction strategy.

12. **Passing an IC screen is necessary but not sufficient for feature inclusion.** P-ML8 added
   8 volume features that all passed |IC_bull| > 0.01, yet ensemble Sharpe collapsed
   (+1.261 → +0.180). Root cause: 8 correlated volume features fragment LightGBM's split
   allocation across near-duplicate signals, causing overfitting. Rule: add at most 1–2 new
   features at a time, or use forward selection, not batch inclusion.

### Open hypotheses (ordered by expected impact)

| # | Hypothesis | Status | Mechanism |
|---|---|---|---|
| H1 | Momentum features improve bull IC | ✅ Confirmed (P-ML7) | `ret_20`, `mom_zscore_20` add trend-strength signal |
| H_vol | Volume features add conviction signal beyond price | ✅ Rejected at daily (P-ML8) | Signal real but too weak; 8 features → overfitting |
| H2 | Risk overlay (DD brake + bull cap) fixes MaxDD | ✅ Partially confirmed (P-ML10) | DD brake reduces binary MaxDD (−77→−68%); redundant on scaled signals |
| H3 | Strategy integration (MLStrategy class) | ✅ Confirmed (P-ML9) | `RegimeLGBMStrategy` + scaled mode beats B&H |
| H4 | HMM regime classifier detects late-bull / overextension | ✅ Rejected (P-ML11) | HMM states overlap with existing features; Fold 2 bull IC unchanged |
| H5 | Optuna tuning on 16-feature P-ML7 model | Open (low priority) | Squeeze remaining gap vs B&H after risk overlay |
| H6 | Cross-asset features improve model | ✅ Partially confirmed (P-ML12b/13) | Helps on biz-day data but not on 7-day (weekend ffill noise). V2 remains champion. |

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

### ~~P-ML8. Volume feature engineering~~ ✅ COMPLETE — F15 logged
Two theories tested: (1) volume as sentiment indicator; (2) volume as institutional participation proxy.
9 volume candidates screened: `vol_log_ratio_{7,14,30}d`, `vol_cv_14d`, `vol_zscore_30d`,
`vol_trend_7_14`, `vol_signed_ratio_{7,14}d`, `vol_price_corr_14d`.
**IC screen:** 8/9 passed |IC_bull| > 0.01. None collinear with FEATURES_V2 (all max|r| < 0.8).
**Walk-forward:** FEATURES_V3 (24 features) Sharpe **+0.180** vs P-ML7 +1.261 — adding 8 volume
features caused LightGBM to overfit (ICIR +0.747 vs +1.779). Bull IC improved (+0.120 vs +0.045)
but total model degraded.
**Institutional era analysis:** Volume IC peaked in Era 2 (2021 bull run), not Era 4 (ETF era),
inconsistent with simple institutional adoption theory. BTC volume also declined in USD-equivalent
terms over time (exchange shift artefact).
**Key learning:** IC screen is necessary but not sufficient. Batch-adding 8 correlated features
hurts LightGBM via split fragmentation. Future: add ≤2 volume features at a time.
**FEATURES_V2 (16 features from P-ML7) remains the champion feature set.**
See `ml/features/volume.py`, `notebooks/p_ml8_volume_features.ipynb`, F15.

---

### ~~P-ML9. Strategy integration — `RegimeLGBMStrategy` class~~ ✅ COMPLETE — F16 logged
`RegimeLGBMStrategy` wraps `RegimeEnsemble` into `BaseStrategy` interface.
- **Binary mode** reproduces P-ML7 exactly (Sharpe +1.261, Return +1997.6%, MaxDD −77.3%).
- **Scaled mode** (pred_zscore × 0.5, 60-bar window): **Sharpe +1.583, MaxDD −33.6%** — beats B&H on both.
- OHLCV full-pipeline (`generate_signals(df)`) validated with 250-bar warmup.
See `strategies/ml/regime_lgbm.py`, `notebooks/p_ml9_strategy_integration.ipynb`, F16.

---

### ~~P-ML10. Risk overlay — drawdown brake + bull cap~~ ✅ COMPLETE — F17 logged
`RiskOverlay` class created in `ml/risk/overlay.py` with DD brake and bull cap.
**Result:** DD brake alone is the best overlay (Sharpe +1.334, MaxDD −68.4%), improving on
binary (+1.234, −77.3%). Bull cap hurts (can't distinguish early vs late bull). Combined
overlay on scaled signals (+1.518, −33.2%) is marginally worse than P-ML9 scaled alone
(+1.583, −33.6%). **P-ML9 scaled positioning remains champion.**
See `notebooks/p_ml10_risk_overlay.ipynb`, F17.

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

## Next Planned — Post Cross-Asset

### P-ML15. Optuna hyperparameter tuning
**Priority: HIGH — last major lever before diminishing returns.**

LightGBM defaults (300 trees, depth 4, lr 0.05) and scaled positioning parameters
(window=60, scale=0.5) were hand-picked. Optuna can jointly optimise:
- Model: `n_estimators`, `max_depth`, `learning_rate`, `min_child_samples`, `reg_alpha/lambda`
- Positioning: `pred_zscore_window`, `position_scale`
- Regime-specific: separate hyperparams for bull vs non-bull models
- Objective: Sharpe ratio on inner purged CV fold

Expected impact: +0.1 to +0.3 Sharpe. Apply to both V2-24/7 and V2-weekday.

### P-ML16. Expanding window walk-forward
Current rolling window discards early training data as it moves forward.
Expanding (anchored) window keeps all history. Quick comparison to check
if more training data improves later folds.

### P-ML17. Production pipeline
Wrap V2-24/7 and V2-weekday into clean strategy classes with:
- Live OHLCV ingestion from exchange API
- Automated regime detection + prediction + position sizing
- Weekend-flat toggle for institutional variant

---

## Lower Priority

### P-ML18. Weekly-frequency cross-asset
The institutional/liquidity/dollar channels (F19) may work at weekly bars
where weekend alignment is a non-issue. Requires rebuilding the feature matrix
at weekly frequency and re-running walk-forward.

### P-ML19. On-chain / funding rate features
Crypto-native data (exchange flows, funding rates, open interest) that doesn't
have TradFi alignment issues. May help with the Fold 2 problem (crypto-specific events).

---

## Parking Lot

- Pairs trading strategies (`strategies/pairs/`)
- Cross-exchange arbitrage (`strategies/arbitrage/`)
- Live paper trading integration
