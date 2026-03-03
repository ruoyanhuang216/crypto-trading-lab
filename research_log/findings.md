# Confirmed Findings

A distilled index of key findings across all sessions.
Each entry references the daily log where it was first observed.

---

## F11 — Longer timeframes do NOT rescue ADXTrend / MASlopeTrend directional accuracy
**Date:** 2026-03-03 | **Notebook:** `p3_signals_timeframe_comparison.ipynb`

Hypothesis (from F3): sub-random accuracy at 1h is due to noise; 4h/1d bars should yield
genuine predictive power for `ADXTrend` and `MASlopeTrend`.

**Results — accuracy at h=1 and Spearman IC vs 1-bar forward log-return:**

| Signal | Timeframe | Acc (h=1) | IC | p-value | Verdict |
|---|---|---|---|---|---|
| ADXTrend | 1h | 0.481 | −0.0191 | 0.074 | Sub-random |
| ADXTrend | 4h | 0.500 | +0.0093 | 0.449 | Weak (not significant) |
| ADXTrend | 1d | 0.491 | +0.0014 | 0.963 | Sub-random |
| MASlopeTrend | 1h | 0.480 | −0.0209 | 0.050 | Sub-random |
| MASlopeTrend | 4h | 0.504 | +0.0101 | 0.415 | Weak (not significant) |
| MASlopeTrend | 1d | 0.480 | −0.0203 | 0.502 | Sub-random |

Dataset: BTC/USDT 2022-01-01→2025-01-01 for 4h/1d (~6,577 / ~1,097 bars);
1h data limited to 2024-01-01→2025-01-01 (~8,785 bars) due to cache.

**Key findings:**
- **Hypothesis partially confirmed at 4h only:** both signals flip from negative IC
  (1h) to weakly positive IC at 4h (+0.009 / +0.010), and accuracy nudges above 0.50.
  However, p-values (~0.42–0.45) are far from significance — the improvement is noise.
- **1d is worse than 4h:** accuracy drops back below 0.50 and IC collapses near zero.
  Reduced sample size (~1,097 bars) also widens confidence intervals.
- **ADX threshold sweep at 1d (15→35):** tighter thresholds (higher ADX) reduce
  coverage from ~60% to ~35% without any consistent accuracy improvement.
  No threshold rescues the signal at daily granularity.

**Conclusion:** Hypothesis **rejected**. Lower timeframes do not produce meaningful
predictive power for these trend signals on BTC/USDT. The signals remain best used as
**regime classifiers** (trending vs ranging), not as directional forecasters at any timeframe.

**Implication for ML track:** Confirms that `trend_dir` as a raw feature is a poor
directional predictor. The regime-classifier role (ADX+SMA200 gate in P-ML3 Exp-C)
remains the recommended use. Discretionary direction forecasting should rely on
the full 12-feature ML model (F7/F8/F9) rather than these heuristic signals.

---

## F10 — Regime-specific models fail to learn trend continuation on 3yr data
**Date:** 2026-03-01 | **Notebook:** `ml_regime_specific_models.ipynb`

`RegimeEnsemble` trains separate `LGBMForecaster` for bull and non-bull (bear+ranging) bars.
Bull model available in Folds 3–5 (128–136 bull training bars); Folds 1–2 fall back to sign-flip.

**Per-fold IC:**

| Fold | Period | Non-bull IC | Bull IC | P-ML4 IC | Bull model? |
|---|---|---|---|---|---|
| 1 | Jul 2022–Jan 2023 | −0.066 | +1.00† | −0.051 | FALLBACK |
| 2 | Jan 2023–Jul 2023 | −0.017 | −0.162 | −0.053 | FALLBACK |
| 3 | Jul 2023–Jan 2024 | −0.131 | −0.138 | −0.148 | YES |
| 4 | Jan 2024–Jul 2024 | −0.087 | −0.063 | −0.056 | YES |
| 5 | Jul 2024–Dec 2024 | +0.042 | −0.044 | −0.003 | YES |

†Fold 1 bull: only 2 test bars, degenerate.

**Equity:** Baseline −30.2% → **P-ML4 −2.5% (Sharpe +0.227)** → P-ML3 Exp-C +8.8% (Sharpe +0.280)

**Root cause:** Bull model IC remains negative in all 3 fitted folds. With 128–136 bull
training bars and 1-day horizon, the model learns the same mean-reversion pattern as the
non-bull model, not trend continuation. P-ML3 Exp-C (skip bull) still wins — it avoids
deploying a broken bull model.

**Feature importance:** `atr_pct` importance drops sharply in bull model (20 vs 114 for non-bull),
but `di_diff`/`adx` don't increase enough to indicate trend-following was learned.

**Implication:** More training data (extend to 2019, 6yr) or longer bull horizon (3–5d)
required before a separate bull model can consistently outperform regime-gating.

---

## F9 — Regime-aware interventions produce first positive OOS equity
**Date:** 2026-03-01 | **Notebook:** `ml_regime_model.ipynb`

`RegimeClassifier` (SMA200 + ADX>25) classifies 1,075 daily bars into:
bull=31.9%, ranging=50.5%, bear=17.6%.

Three experiments on the P-ML2 LightGBM baseline (purged WF, 5 folds, 3yr daily BTC):

| Approach | Mean IC | ICIR | Return | Sharpe | MaxDD |
|---|---|---|---|---|---|
| Baseline (P-ML2) | −0.049 | −0.488 | −30.2% | −0.046 | −76.1% |
| Exp-A (regime feat) | −0.040 | −0.364 | −28.3% | −0.023 | −76.1% |
| **Exp-B (flip in bull)** | **−0.015** | **−0.282** | **+33.2%** | **+0.482** | **−46.0%** |
| **Exp-C (skip bull)** | — | — | **+8.8%** | **+0.280** | **−49.8%** |
| Buy-and-Hold | — | — | +299.6% | +1.379 | −35.4% |

**IC-by-regime (§4):** Most features don't flip sign between bull and bear — they weaken.
Strongest sign-flippers: `vol_log_chg` (bear=+0.125 vs bull=−0.021) and `di_diff`
(bear=−0.114 vs bull=+0.027). `bar_ret` consistently negative across all regimes.

**Exp-B caveat:** Signal flip hurts Fold 2 (bull-heavy recovery where model was correct).
The model is not always wrong in bull — only when trend is sustained and strong.

**Implication:** Exp-C (regime-gated) is the first deployable strategy with positive OOS
Sharpe. Next step: P-ML4 — separate bull/non-bull models.

---

## F1 — MeanReversion outperforms Breakout on 1h BTC (Jan–Jun 2024)
**Date:** 2026-02-25 | **Ref:** [2026-02-25](daily/2026-02-25.md)

`BollingerMeanReversion` (Sharpe ~2.0) significantly outperformed `BollingerBreakout`
(Sharpe ~0.75) on 1h BTC/USDT over Jan–Jun 2024.

**Hypothesis:** BTC was range-bound in Jan–Feb 2024 before the bull run,
favouring mean-reversion entries. Result may not hold in strongly trending periods.

**To validate:** rerun on a different time window (e.g. 2023 bear market, 2024 Q3 bull run).

---

## F2 — Lagging trend filters hurt BollingerBreakout
**Date:** 2026-02-25 | **Ref:** [2026-02-25](daily/2026-02-25.md)

Applying `ADXTrend` and `MASlopeTrend` filters to `BollingerBreakout` degraded
performance across all variants (unfiltered was the best).

**Root cause:** Breakout strategies need to enter at the *start* of a move.
ADX/slope are lagging — they confirm trends after the move is already underway,
removing the entry edge.

**Implication:** These signals should be tested as filters for *mean-reversion*
strategies instead (sit out when strongly trending), where the logic is aligned.

---

## F4 — BollingerMeanReversion is period-dependent, not structurally robust
**Date:** 2026-02-25 | **Ref:** [2026-02-25-p1](daily/2026-02-25-p1.md)

MeanReversion returned -9.1% over Jan–Jun 2024 vs +15.9% over Jan–Mar 2024.
The Q2 2024 BTC bull run (42k → 70k) caused repeated short losses on overbought
signals that never reversed. The Jan–Mar result (F1) was period-specific, not structural.

**Implication:** strategy results must always be validated across multiple time windows
before drawing conclusions. Walk-forward testing (P4) is essential.

---

## F5 — Trend filters did NOT improve BollingerMeanReversion (hypothesis rejected)
**Date:** 2026-02-25 | **Ref:** [2026-02-25-p1](daily/2026-02-25-p1.md)

Applying ADX ranging and slope filters to MeanReversion made performance worse, not better.
In a strongly trending bull market, the strategy loses on almost every trade regardless
of regime — filtering reduces trade count but does not fix the core directional failure.

**Exception:** the `ADX + aligned` combined filter reduced max drawdown (-11.9% vs -17.1%),
suggesting some value for risk management even when it cannot fix returns.

**Implication:** the correct fix for MeanReversion in a bull market is directional bias
(long-only mode) or a high-level regime switch, not a bar-by-bar filter.

---

## F6 — Long-only MeanReversion reduces losses but does not create alpha in a bull market
**Date:** 2026-02-28 | **Ref:** [2026-02-28](daily/2026-02-28.md) | **Notebook:** `p1b_longonly_meanreversion.ipynb`

Three variants of BollingerMeanReversion tested on full-year 2024 BTC/USDT 1h via walk-forward (5 rolling folds):

| Variant | WF Sharpe | WF Return | Max DD |
|---|---|---|---|
| Baseline (long+short) | −1.12 | −20.6% | −33.5% |
| **LongOnly** | **−0.18** | **−3.6%** | −18.9% |
| TrendFiltered (200MA) | −0.93 | −7.0% | −12.1% |

Removing short signals eliminates 5× the losses. TrendFiltered reduces MaxDD further but barely
trades (284 vs 1040 active bars with MA=200). Signal scarcity (5.7% of bars touch lower band)
is the binding constraint — not direction.

**Implication:** LongOnly is strictly better than baseline for bull-market deployment.
No variant is profitable on a WF basis in 2024 — the mean-reversion edge is too small
to overcome a strong trending year. Promote `BollingerLongOnly` to strategies module.

---

## F7 — IC analysis: mean reversion dominates at 1h; daily timeframe has strongest signal
**Date:** 2026-02-28 | **Ref:** [2026-02-28](daily/2026-02-28.md) | **Notebook:** `ml_feature_engineering.ipynb`

34 features × 5 timeframes (5m, 15m, 1h, 4h, 1d) analysed via Spearman IC against 1-bar-ahead forward return.

**Top features at 1h (all negative — mean reversion):**
- `bar_ret` / `ret_lag1`: IC = −0.081 (current bar return predicts reversal)
- `stoch_k`: IC = −0.063 | `bb_zscore`: IC = −0.049 | `rsi`: IC = −0.045

**IC by timeframe (mean |IC|):** 5m=0.015 → 15m=0.020 → 1h=0.024 → 4h=0.024 → **1d=0.041**

Daily IC is 70% higher than hourly. `upper_wick` at 1d reaches IC = 0.165 — best
single-feature signal found. **Model building should target 1d data.**

**Structural findings:**
- Raw returns: no autocorrelation (Ljung-Box p > 0.05) — consistent with weak-form EMH
- Squared returns: strong GARCH clustering (p ≈ 0) — volatility is predictable, direction is not
- 27 highly correlated feature pairs; `ret_lag1` = `bar_ret` exactly (|r| = 1.0)
- Recommended 12-feature set for LightGBM (deduplicated oscillator + volatility groups)

**Implication:** Skip 5m modelling. Start LightGBM on 1d data (P-ML2) targeting forward
log-return regression with purged walk-forward CV.

---

## F8 — LightGBM IC is regime-sensitive and sign-unstable on 1d BTC
**Date:** 2026-03-01 | **Ref:** [2026-02-28](daily/2026-02-28.md) | **Notebook:** `ml_baseline_models.ipynb`

LightGBM regressor trained on 12-feature set (F7), evaluated via purged walk-forward
(5 folds, 3 years daily BTC/USDT 2022–2024, purge=1 bar).

**Per-fold OOS IC:**

| Fold | Period | Regime | IC | Hit rate | Significant? |
|---|---|---|---|---|---|
| 1 | Jul 2022 – Jan 2023 | Bear | −0.074 | 0.514 | No |
| 2 | Jan 2023 – Jul 2023 | Recovery | +0.044 | 0.564 | No |
| 3 | Jul 2023 – Jan 2024 | Recovery→Bull | **−0.224** | 0.413 | **Yes (p<0.05)** |
| 4 | Jan 2024 – Jul 2024 | Bull | +0.049 | 0.486 | No |
| 5 | Jul 2024 – Dec 2024 | Bull | −0.039 | 0.508 | No |

**Aggregate:** Mean IC=−0.049, ICIR=−0.488, Pooled OOS IC=−0.017

**Equity:** LightGBM −32.4% (Sharpe −0.075, MaxDD −76.8%) vs B&H +299.6%

**Top features:** `vol_log_chg` (185), `adx` (174), `upper_wick` (169), `bb_width` (149)

**Root cause of failure:** IC sign alternates across regimes. The model learns mean-reversion
(negative-IC features from F7) which works in bear/ranging but inverts in strong bull trends.
Fold 3 is statistically significant (p<0.05) — the model has skill but in the wrong direction.

**Implication:** Regime detection is prerequisite for deployment. Next step: P-ML3 — add a
regime classifier (bull/bear/ranging) as meta-feature or fold-selection gate.

---

## F3 — Trend signals have sub-random directional accuracy at 1h
**Date:** 2026-02-25 | **Ref:** [2026-02-25](daily/2026-02-25.md)

Both `ADXTrend` and `MASlopeTrend` predicted next-bar direction with ~0.47 accuracy
on 1h BTC data — below the 0.5 random baseline. Accuracy degraded further at 4, 12,
and 24-bar horizons. Tightening thresholds reduced coverage without improving precision.

**Implication:** These signals should not be used as standalone directional forecasters
on 1h data. Their value is as **regime classifiers** (trending vs ranging market),
not as price direction predictors.

**To validate:** test on 4h or daily bars to check if signal quality improves at
lower frequencies where noise is reduced.
