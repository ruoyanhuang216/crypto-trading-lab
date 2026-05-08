# Roadmap

Current priority list. Updated at the end of each session.
_Last updated: 2026-05-08 (P-ML19 Phase 1a complete — REJECTED; F24 logged — funding rate hurts target Fold; 2019 training data flagged as the real culprit → P-ML20 promoted)_

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
| P-ML15 Optuna-tuned scaled | RegimeEnsemble scaled (16f, 6yr, tuned) | +0.980 | +242.4% | −34.4% | **Inner CV +1.612 → outer +0.980 (overfit)** |
| P-ML19 V2 truncated scaled (re-baseline) | RegimeEnsemble (16f, 2020-03→2024-12) | +0.594 | +74.5% | −42.8% | Truncated-window baseline for V4 comparison |
| P-ML19 V4 scaled (V2 + funding_zscore_30d) | RegimeEnsemble (17f, 2020-03→2024-12) | +0.775 | +123.1% | −42.2% | **+0.181 vs V2-trunc, but Fold-2 bull IC −0.117** |
| *Buy & Hold (6yr)* | *—* | *+1.052* | *+876.6%* | *−76.6%* | *Benchmark* |
| *Buy & Hold (2020-03 truncated)* | *—* | *+1.092* | *+997.0%* | *−76.6%* | *Truncated benchmark* |

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

13. **Hand-picked defaults beat Optuna in this regime.** P-ML15 ran 100 TPE trials over
    a 7-dim search space; inner-CV best Sharpe +1.612 mapped to outer-WF Sharpe +0.980,
    a −0.604 lift vs the +1.583 hand-picked baseline. Two diagnostic root causes:
    (a) **inner-CV overfit** — 3-fold inner Sharpe overstates outer 5-fold Sharpe by
    +0.63 because TPE finds idiosyncratic optima that the outer split exposes; and
    (b) **bull-model fragility** — `min_child_samples=49` on ~130 bull bars/fold makes
    the bull sub-model degenerate (Bull IC = nan in every outer fold). Inner CV's
    different fold boundaries hid this failure mode. The remaining alpha is not in the
    hyperparameter surface; future tuning needs nested per-outer-fold CV or a
    bull-coverage validity constraint, but more impactful work lies elsewhere
    (per-regime hyperparams, on-chain features). See F23.

14. **The Fold-2 ATH+crash failure was a training-data problem, not a missing signal.**
    P-ML19 Phase 1a ran funding-rate features on a truncated 2020-03→2024-12 dataset
    (because the Binance archive starts 2020-01). The accidental V2-truncated
    re-baseline showed Fold-2 bull IC of **+0.107** vs V2 6yr's **−0.128** — a +0.235
    swing purely from removing 2019 training data. This means the 2019 recovery /
    early-2020 COVID-shock bars were teaching the bull model a pattern that
    misgeneralised to the 2021 ATH cycle. Adding funding rate on top did NOT help
    this fold (V4 dropped further to −0.011) — funding rate is a weak signal and
    actively disrupts the (now-good) bull-model on the cleaner training set.
    Implication: the largest tractable lever is now a sample-window robustness
    study (P-ML20) — testing whether dropping or down-weighting pre-2020 data
    produces a strict improvement on the V2 champion. New on-chain signals are
    deprioritised (the bull failure was solved by data, not signals). See F24.

### Open hypotheses (ordered by expected impact)

| # | Hypothesis | Status | Mechanism |
|---|---|---|---|
| H1 | Momentum features improve bull IC | ✅ Confirmed (P-ML7) | `ret_20`, `mom_zscore_20` add trend-strength signal |
| H_vol | Volume features add conviction signal beyond price | ✅ Rejected at daily (P-ML8) | Signal real but too weak; 8 features → overfitting |
| H2 | Risk overlay (DD brake + bull cap) fixes MaxDD | ✅ Partially confirmed (P-ML10) | DD brake reduces binary MaxDD (−77→−68%); redundant on scaled signals |
| H3 | Strategy integration (MLStrategy class) | ✅ Confirmed (P-ML9) | `RegimeLGBMStrategy` + scaled mode beats B&H |
| H4 | HMM regime classifier detects late-bull / overextension | ✅ Rejected (P-ML11) | HMM states overlap with existing features; Fold 2 bull IC unchanged |
| H5 | Optuna tuning on 16-feature P-ML7 model | ✅ Rejected (P-ML15) | Inner-CV overfit; tuned outer Sharpe +0.980 vs defaults +1.583 |
| H6 | Cross-asset features improve model | ✅ Partially confirmed (P-ML12b/13) | Helps on biz-day data but not on 7-day (weekend ffill noise). V2 remains champion. |
| H7 | Funding rate captures late-bull overextension | ✅ Rejected (P-ML19 Phase 1a) | Only 1/3 candidates passed IC screen; selected feature *worsens* Fold-2 bull IC −0.117 |
| H8 | Pre-2020 training data degrades bull-fold IC | Open — promoted to P-ML20 | Truncated V2 has Fold-2 bull IC +0.107 vs full V2 −0.128 (+0.235 swing) |

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

### ~~P-ML15. Optuna hyperparameter tuning~~ ✅ COMPLETE — REJECTED — F23 logged
100-trial TPE study on the 7-dim search space (LightGBM × scaled positioning) gave
**inner-CV Sharpe +1.612 but outer-WF Sharpe +0.980 vs defaults +1.583** (Δ −0.604).
Two root causes diagnosed: (a) inner-CV overfit (3-fold inner-vs-5-fold-outer gap
of +0.63 Sharpe), (b) bull-sub-model fragility — `min_child_samples=49` on ~130
bull bars/fold makes the bull model degenerate (Bull IC = nan everywhere). Inner
CV's different fold boundaries hid the failure mode.

**Hand-picked defaults remain champion.** The +0.10–0.25 Sharpe expected lift
from 2026-04-17 was not realised. Lift < +0.05 → decision rule branch fires:
P-ML15 done; remaining alpha is not in the hyperparameter surface.

If revisited (low priority): use nested per-outer-fold CV or add a bull-coverage
validity constraint that rejects param configs where the bull sub-model degenerates.
See `notebooks/p_ml15_optuna_tuning.ipynb`, F23.

### P-ML20. Sample-window robustness study  *(promoted after F24)*
**Priority: HIGH — F24's accidental discovery is the new highest-EV lever.**

**Hypothesis (H8).** Pre-2020 training data degrades the bull sub-model's
ability to predict the 2021 ATH cycle. Truncating the dataset to 2020-01
onward (P-ML19's incidental re-baseline) flipped Fold-2 bull IC from
−0.128 (V2 6yr champion) to +0.107 (V2 truncated) — a +0.235 swing from
*removing data*, the opposite of the standard "more data is better" prior
that drove P-ML5.

**Plan.**
1. **Sliding-start ablation.** Run V2 walk-forward with start dates of
   2019-03 (full), 2019-06, 2019-09, 2020-01, 2020-06, 2020-09. Report
   Sharpe, MaxDD, mean IC, ICIR, and per-fold bull IC. Identify the cutoff
   that maximises Fold-2 bull IC subject to non-degraded overall Sharpe.
2. **Sample weighting.** If the sliding-start sweep shows a clear
   breakpoint, try a weighted variant: full dataset but down-weight
   pre-cutoff bars (LightGBM `sample_weight`). Compare to hard truncation.
3. **Diagnose what's different about pre-2020 BTC.** Plot key features
   (atr_pct, mom_zscore_20, ret_20, regime distribution) for 2019-2020
   vs 2020+. The bull model's failure mode points to a specific subset
   (post-2018-bear recovery? COVID shock? both?).
4. **Outer comparison.** Re-run the sliding-start champion vs original V2
   6yr on the matched 2020-onward held-out window so the comparison is
   apples-to-apples → recommend champion variant (truncated vs weighted vs full).

**Decision rule (pre-committed).**
- Sliding-start variant strictly dominates V2 on Fold-2 bull IC AND total
  Sharpe (matched window) → adopt as new V2 baseline; rerun all downstream
  experiments (P-ML7+, P-ML9 scaled) on the new baseline.
- Improves Fold-2 bull IC but Sharpe regresses → log, document the regime
  trade-off, do not adopt.
- No improvement → H8 rejected, the +0.235 swing was a fold-boundary
  artefact rather than a 2019-data effect; revisit Phase 2 on-chain.

**Estimated effort.** 1 session for ablation + diagnosis + writeup.

---

### ~~P-ML19 Phase 1a. Funding-rate features~~ ✅ COMPLETE — REJECTED — F24 logged
Phase 1a fetched Binance perp BTCUSDT funding rate from data.binance.vision
(api.binance.com geo-blocked; S3 archive works). Three feature candidates
screened; only `funding_zscore_30d` passed the |IC|>0.01 dual-screen (1/3).
FEATURES_V4 walk-forward: **V4 scaled Sharpe +0.775 vs V2 truncated +0.594
(Δ +0.181)** — but the late-cycle Fold-2 bull IC went the **wrong way**
(V2 +0.107 → V4 −0.011, Δ −0.117). Pre-committed decision rule fires the
REJECT branch (no bull-IC improvement at the H7 target).

The accidental finding was bigger than the planned one: V2-truncated *itself*
solves the Fold-2 problem (V2 6yr Fold-2 bull IC −0.128 → V2 truncated +0.107),
implying the 2019 training data was the source of the bull-model failure.
That insight is what promoted P-ML20 to top of queue.

**Phase 1b (open interest) — PARKED.** P-ML19's premise was that the bull
failure required new signal types; F24 shows the failure was primarily a
training-data issue. New on-chain signals deserve revisit only after P-ML20
clarifies what the V2 champion should look like on the right training window.

See `notebooks/p_ml19_funding_features.ipynb`, F24.

---

### P-ML19 Phase 2. On-chain features (parked)
Originally hypothesised as the missing signal type for late-bull
overextension. F24 redirects effort to P-ML20 before revisiting on-chain.
If P-ML20 establishes a stable V2 champion and the late-cycle problem still
exists, return here with realised cap, MVRV, exchange flows, OI from a paid
archive (Coinglass / Glassnode).

---

### P-ML19 Phase 1a — original plan (kept for reference)
**Priority: HIGH — F23 explicitly redirects effort from "more fits to existing
data" toward "new data signals". Funding rates are the highest-EV candidate
because they capture leverage / sentiment dynamics that price-derived features
cannot — directly targeting the Fold 2 ATH+crash failure mode (F14, F23).**

**Hypothesis (H7).** Persistently elevated funding in a bull regime is a leading
indicator of overextension and impending drawdown. Adding funding-rate-derived
features to FEATURES_V2 should improve Fold 2 bull IC (currently −0.128 with
defaults — the structural failure flagged in learning #5).

**Phase 1 — Funding rate + open interest (this experiment).**
Data sources (Binance perp BTCUSDT):
- Funding rate — 8h cadence (3 obs/day) from 2019-09 onward
- Open interest — daily snapshot from ~2020 onward (verify exact start)

Feature candidates:
- `funding_rate_3d_mean` — recent leverage skew (rolling 9-funding mean)
- `funding_rate_zscore_30d` — relative-to-history extremity
- `funding_persistence_3d` — fraction of last 9 fundings same-sign (crowded-trade proxy)
- `oi_log_chg_7d` — leverage build-up rate
- `oi_zscore_30d` — relative OI level
- `funding_x_oi_zscore` — interaction (high funding × high OI = squeeze setup)

**Validation methodology** (mirrors P-ML8 / F15 — the volume-features failure
that produced learning #12):
1. **Data availability check.** Confirm coverage from 2019-09 onward. If the
   dataset shortens vs 6yr V2, re-baseline V2 on the same window so the
   comparison is apples-to-apples.
2. **IC screen.** |IC_total| > 0.01 AND |IC_bull| > 0.01. Reject features
   that pass total but not bull — Fold 2 is the target.
3. **Collinearity.** max|r| ≤ 0.8 against any FEATURES_V2 column and
   against each other.
4. **Add at most 2 features at a time** (learning #12) — avoid the
   8-features-at-once split-fragmentation that tanked P-ML8.
5. **FEATURES_V4** = FEATURES_V2 + ≤2 selected funding/OI features.
6. **Walk-forward.** Same outer 5-fold purged WF (TRAIN_FRAC=0.6, PURGE=1).
   Run scaled and binary modes; compare against V2 champion (Sharpe +1.583,
   MaxDD −33.6%) and B&H.
7. **Per-fold IC inspection.** Primary success criterion is **Fold 2 bull IC
   improvement**, not just aggregate Sharpe. Aggregate Sharpe can rise from
   non-Fold-2 folds even if the structural problem is unsolved.

**Decision rule (pre-committed).**
- Fold 2 bull IC improves (e.g. from −0.128 to ≥ 0) AND outer Sharpe ≥ +1.40
  → ship FEATURES_V4 as new champion candidate; queue P-ML15b (per-regime tune).
- Fold 2 bull IC improves but outer Sharpe < +1.40 → useful diagnostic only;
  log and stay on V2 champion; consider Phase 2.
- No Fold 2 bull IC improvement → hypothesis rejected; pivot to Phase 2
  on-chain, or accept Fold 2 as structurally unfixable at daily resolution.

**Open risks.**
- **Dataset truncation.** Funding/OI may not cover the full 2019 V2 window.
  Mitigation: re-baseline V2 on the truncated window before comparison.
- **Timestamp / leakage.** Funding settled at bar t reflects positioning over
  bar t−1. Need to confirm conventions in the Binance API and shift if needed.
- **Regime overlap with momentum features.** `funding_rate_zscore_30d` may
  correlate with `mom_zscore_20` in trending regimes. Collinearity screen
  catches this; if borderline, choose the one with higher IC_bull.

**Phase 2 (parked, only if Phase 1 promising) — On-chain metrics.**
Glassnode-style realised cap, MVRV, exchange flows. Heavier data engineering
(API integration, backfill, alignment). Defer until Phase 1 result is in.

**Estimated effort.** 1 session for data ingestion + IC screen; 1 session for
WF + reporting. Cadence comparable to P-ML7 / P-ML8.

### P-ML16. Expanding window walk-forward
Current rolling window discards early training data as it moves forward.
Expanding (anchored) window keeps all history. Quick comparison to check
if more training data improves later folds. Lower priority post-F23 —
mechanical change unlikely to fix the Fold 2 structural problem.

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

---

## Parking Lot

- Pairs trading strategies (`strategies/pairs/`)
- Cross-exchange arbitrage (`strategies/arbitrage/`)
- Live paper trading integration
