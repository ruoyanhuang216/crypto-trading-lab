# Confirmed Findings

A distilled index of key findings across all sessions.
Each entry references the daily log where it was first observed.

---

## ML Track — Summary (P-ML2 through P-ML8)

| Finding | Experiment | Result | Verdict |
|---|---|---|---|
| F8 | P-ML2 LightGBM baseline | Sharpe −0.046, IC sign-unstable across regimes | Baseline established |
| F9 | P-ML3 Regime-aware LightGBM | Exp-C skip-bull: Sharpe +0.280 — first positive OOS | Regime gate essential |
| F10 | P-ML4 RegimeEnsemble (3yr) | Sharpe +0.227; bull IC negative — too few training bars | Data volume is bottleneck |
| F12 | P-ML5 RegimeEnsemble (6yr) | **Sharpe +0.927, Return +630%** — bull IC turns positive in 3/4 folds | Best model to date |
| F13 | P-ML6 LSTM 30-bar (6yr) | Sharpe −0.517, Return −93% — worse than LightGBM on every metric | Sequential model rejected |
| F14 | P-ML7 RegimeEnsemble + momentum (6yr) | **Sharpe +1.261, Return +1998%** — approaches B&H Sharpe +1.379 | New best model |
| F15 | P-ML8 Volume features (6yr) | Sharpe +0.180 — **worse than P-ML7** despite 8/9 features passing IC | Volume redundant at daily resolution |

**Current champion: P-ML7 RegimeEnsemble + momentum features (Sharpe +1.261 vs B&H +1.379).**

Dominant improvement axes confirmed: **regime gating + data volume + feature engineering**.
Remaining gap to B&H: 0.118 Sharpe points. MaxDD −77.3% — risk overlay is the next priority.

Next planned work: **P-ML9** — wrap `RegimeEnsemble` into a proper `RegimeLGBMStrategy` class;
**P-ML10** — position sizing + drawdown brake to address MaxDD −77.3%.

---

## F14 — Momentum features (P-ML7) push Sharpe to +1.261, approaching Buy & Hold
**Date:** 2026-03-06 | **Notebook:** `p_ml7_momentum_features.ipynb`

Hypothesis (H1): Adding multi-period momentum features (`ret_5`, `ret_20`, `mom_zscore_20`,
`ret_5_minus_20`) to the 12-feature P-ML5 set gives the bull model an explicit
trend-continuation signal, improving OOS IC and Sharpe.

**IC screen results (Spearman IC on bull bars):**

| Feature | IC_all | IC_bull | IC_nonbull | Collinear with | Selected? |
|---|---|---|---|---|---|
| `ret_5` | −0.018 | +0.012 | −0.044 | `bb_zscore` (r=0.748) | YES |
| `ret_20` | +0.008 | +0.010 | −0.021 | `rsi` (r=0.848) ← high | YES |
| `ret_60` | +0.041 | +0.002 | +0.025 | — | no (|IC_bull|<0.01) |
| `mom_zscore_20` | +0.000 | +0.040 | −0.031 | `macd_hist_norm` (r=0.721) | YES |
| `ret_5_minus_20` | +0.000 | −0.012 | +0.026 | `rsi` (r=0.618) | YES |

`ret_20` flagged as collinear with RSI (r=0.848) but selected anyway (|IC_bull|=0.010 > threshold).
`ret_60` rejected — IC_bull ≈ 0 despite strong IC_all, suggesting the 60-bar signal is
non-bull-specific and already captured by `di_diff` / `adx`.

**FEATURES_V2 (16 features):** original 12 + [`ret_5`, `ret_20`, `mom_zscore_20`, `ret_5_minus_20`]

**Per-fold OOS IC (P-ML5 vs P-ML7):**

| Fold | Test period | P-ML5 IC | P-ML7 IC | Δ IC | Bull IC P5 | Bull IC P7 |
|---|---|---|---|---|---|---|
| 1 | Feb 2020–Feb 2021 | +0.0612 | +0.0721 | +0.011 | +0.031 | nan† |
| 2 | Feb 2021–Jan 2022 | −0.0536 | **+0.0091** | **+0.063** | −0.050 | −0.128‡ |
| 3 | Jan 2022–Jan 2023 | +0.1295 | +0.1283 | −0.001 | nan | +0.179 |
| 4 | Jan 2023–Jan 2024 | +0.0462 | +0.0537 | +0.008 | +0.042 | +0.058 |
| 5 | Jan 2024–Dec 2024 | +0.0861 | +0.1069 | +0.021 | +0.061 | +0.071 |

†Fold 1: OOS window shifted (ret_60 adds ~40 extra warmup bars), fewer bull test bars available.
‡Fold 2: Aggregate IC improved (+0.0091 vs −0.0536) but **bull-specific IC worsened** (−0.128 vs −0.050). The momentum features make the bull model more aggressive long just before the ATH crash.

**Aggregate metrics:**

| Metric | P-ML5 (12 feats) | P-ML7 (16 feats) |
|---|---|---|
| Mean OOS IC | +0.054 | **+0.074** |
| ICIR | +0.888 | **+1.779** |
| Mean Bull IC | +0.021 | **+0.045** |
| Fold 2 Bull IC | −0.050 | −0.128 (worse) |
| OOS Sharpe | +0.927 | **+1.261** |
| OOS Return | +630.2% | **+1997.6%** |
| Max Drawdown | −68.0% | **−77.3%** (worse) |

**Verdict: HYPOTHESIS SUPPORTED.** Momentum features improve Mean IC (+0.074 vs +0.054),
ICIR (+1.779 vs +0.888), Mean Bull IC (+0.045 vs +0.021), and Sharpe (+1.261 vs +0.927).
P-ML7 approaches Buy & Hold (Sharpe +1.379) to within 0.118 Sharpe points.

**Key nuance — Fold 2 bull IC worsens despite aggregate IC improving:**
The ATH+crash period (Jan 2021–Jan 2022) is not fixed by momentum features. In fact it gets
worse: `ret_20` ≈ +40–60% log-return at ATH makes the bull model strongly predict continuation
just before the crash. The aggregate Fold 2 IC improves (+0.0091 vs −0.0536) because the
*non-bull* predictions improve, not the bull predictions.

**Implication:**
- Momentum features are a genuine signal improvement (ICIR nearly doubles)
- The Fold 2 bull failure is a *late-trend detection* problem, not a *feature availability* problem
  — the model correctly identifies strong momentum but cannot predict the reversal
- MaxDD worsened to −77.3% because stronger bull predictions amplify losses when they're wrong
- **Next priority: P-ML8 (strategy integration) + P-ML9 (risk overlay / position sizing)**
  to exploit the improved IC without amplifying drawdowns

---

## F15 — Volume features hurt P-ML7 performance despite passing IC screen
**Date:** 2026-03-06 | **Notebook:** `p_ml8_volume_features.ipynb`

**Motivation:** Two theories justify volume features: (1) **sentiment** — volume validates price
conviction; (2) **institutional participation** — growing BTC institutional presence since 2020
(Grayscale/MicroStrategy) and ETF era (Jan 2024) leaves footprints in volume patterns.

**Setup:** 9 volume candidates across 6 categories: `vol_log_ratio_{7,14,30}d` (level vs average),
`vol_cv_14d` (consistency), `vol_zscore_30d` (block-trade spike), `vol_trend_7_14` (acceleration),
`vol_signed_ratio_{7,14}d` (directional OBV-style), `vol_price_corr_14d` (volume-price alignment).
IC screen on bull bars (|IC_bull| > 0.01). Augmented walk-forward: FEATURES_V3 = FEATURES_V2 (16) + selected.

**IC screen results (Spearman IC on bull bars):**

| Feature | IC_all | IC_bull | IC_nonbull | max corr with V2 | Selected? |
|---|---|---|---|---|---|
| `vol_log_ratio_7d` | +0.040 | +0.035 | +0.043 | 0.656 (vol_log_chg) | YES |
| `vol_log_ratio_14d` | +0.044 | +0.036 | +0.047 | 0.592 (vol_log_chg) | YES |
| `vol_log_ratio_30d` | +0.048 | +0.029 | +0.054 | 0.555 (hl_range) | YES |
| `vol_cv_14d` | −0.019 | −0.043 | −0.007 | 0.333 (ret_20) | YES |
| `vol_zscore_30d` | +0.034 | +0.026 | +0.036 | 0.581 (hl_range) | YES |
| `vol_trend_7_14` | +0.025 | **+0.005** | +0.033 | 0.430 | **no** |
| `vol_signed_ratio_7d` | −0.001 | −0.036 | +0.000 | 0.691 (bb_zscore) | YES |
| `vol_signed_ratio_14d` | +0.019 | +0.023 | +0.001 | 0.739 (rsi) | YES |
| `vol_price_corr_14d` | +0.027 | +0.046 | +0.010 | 0.624 (mom_zscore_20) | YES |

8/9 features passed IC screen. All max correlations with FEATURES_V2 below 0.8 (not redundant
by correlation alone). `vol_trend_7_14` rejected (|IC_bull| = 0.005 < 0.01).

**Era-stratified IC — institutional participation theory:**

| Feature | Era1 Retail | Era2 Instit | Era3 Bear | Era4 ETF |
|---|---|---|---|---|
| `vol_log_ratio_7d` | +0.014 | **+0.153** | +0.008 | +0.012 |
| `vol_zscore_30d` | −0.013 | **+0.133** | +0.028 | +0.012 |
| `vol_cv_14d` | −0.074 | +0.029 | +0.063 | +0.030 |
| `vol_signed_ratio_7d` | −0.009 | +0.003 | −0.003 | **−0.024** |
| `vol_price_corr_14d` | +0.008 | **+0.076** | +0.005 | +0.051 |

Most features peaked in Era 2 (2021 — first institutional wave), not Era 4 (ETF era). The 2021
bull market had unusually strong volume-price correlation. Era 4 IC is not systematically higher
than Era 1, suggesting the institutional adoption signal (if real) does not manifest at daily
granularity within this 6yr window.

**Volume landscape:** BTC daily volume actually *declined* 57% from Era 1 (mean 40,733 BTC/day)
to Era 2 (17,683 BTC/day) — likely reflecting USD-denominated vs BTC-denominated exchange shifts
rather than true institutional exit. Era 3 and 4 are similarly lower volume in BTC terms,
which complicates the institutional theory.

**FEATURES_V3 (24 features) walk-forward results:**

| Fold | Test period | P-ML7 IC | P-ML8 IC | Δ IC |
|---|---|---|---|---|
| 1 | Feb 2020–Feb 2021 | +0.0721 | **−0.0391** | −0.111 |
| 2 | Feb 2021–Jan 2022 | +0.0091 | +0.0027 | −0.006 |
| 3 | Jan 2022–Jan 2023 | +0.1283 | **+0.1434** | +0.015 |
| 4 | Jan 2023–Jan 2024 | +0.0537 | **+0.0701** | +0.016 |
| 5 | Jan 2024–Dec 2024 | +0.1069 | +0.0545 | −0.052 |

**Aggregate:**

| Metric | P-ML7 (16f) | P-ML8 (24f) |
|---|---|---|
| Mean OOS IC | +0.074 | +0.046 |
| ICIR | **+1.779** | +0.747 |
| Mean Bull IC | +0.045 | **+0.120** ← improved |
| OOS Sharpe | **+1.261** | +0.180 |
| OOS Return | **+1997.6%** | −43.2% |
| Max Drawdown | −77.3% | −91.5% |

**Verdict: HYPOTHESIS NOT SUPPORTED.** Volume features hurt equity despite 8/9 passing IC
screen. Sharpe collapses (+1.261 → +0.180), ICIR halved (+1.779 → +0.747), return goes negative.
Mean Bull IC *improved* (+0.120 vs +0.045) but the extra 8 features caused overfitting in 3/5
folds (Fold 1, 2, 5) and increased non-bull noise. FEATURES_V2 (P-ML7, 16 features) remains
the champion feature set.

**Root causes:**

1. **Dimensionality curse with LightGBM:** Adding 8 correlated volume features (all related to
   the single underlying volume signal) fragments information gain across similar splits. The
   boost trees see 8 near-duplicate "votes" for volume and overfit that direction.
2. **Volume IC is real but weak:** ICs 0.026–0.046 on bull bars are genuine signals (above 0.01
   threshold) but too weak to survive the noise amplification of going from 16 to 24 features.
3. **Feature selection was too permissive:** The |IC_bull| > 0.01 threshold with 9 candidates
   selects 88% of them. A stricter threshold (0.03) or forward selection would have kept 1–2
   volume features rather than 8. Volume ratio, z-score, and price-correlation are largely
   correlated and redundant with each other.

**Institutional participation theory:** *Partially confirmed.* Era 2 (2021) shows the strongest
volume IC across most features, but Era 4 (ETF era) is not systematically stronger than Era 1
(retail era). The theory may hold, but the signal is too diffuse at daily resolution.

**Key learning (generalises to future feature work):**
> Passing an IC screen is necessary but not sufficient. Adding too many features with
> moderate, correlated IC can hurt a tree model by splitting information gain across redundant
> splits. Future feature addition should be done incrementally (1–2 at a time) or via
> forward selection rather than adding all IC-positive candidates at once.

**Recommended volume feature strategy for future experiments:**
- If testing volume: add at most 2 features (e.g., `vol_price_corr_14d` + `vol_cv_14d`)
- Better leverage: revisit volume at **1h frequency** where intra-day conviction matters more
- Alternative: test volume as a **regime conditioning variable** (high-vol regime vs low-vol
  regime) rather than a direct predictive feature

---

## F13 — LSTM Forecaster does NOT outperform LightGBM at 1-day BTC horizon (hypothesis rejected)
**Date:** 2026-03-05 | **Notebook:** `p_ml6_lstm.ipynb`

Hypothesis: A 30-bar LSTM captures multi-bar temporal dependencies that single-bar LightGBM
(P-ML5) cannot, improving OOS IC and Sharpe on 6yr BTC/USDT daily.

**Setup:** Same dataset (2019–2025, 2,171 bars), same 12 features, same purged walk-forward
(5 folds, train_frac=0.6, purge=1). LSTM: seq_len=30, units=64, dropout=0.2, EarlyStopping
(patience=10). No regime gating — pure sequence model, direct comparison to P-ML5 LightGBM.

**Per-fold OOS IC (after excluding 29-bar warmup from each test fold):**

| Fold | Test period | P-ML5 IC | P-ML6 IC | Epochs | LSTM better? |
|---|---|---|---|---|---|
| 1 | Jan 2020–Jan 2021 | +0.0612 | −0.0451 | 23 | no |
| 2 | Jan 2021–Jan 2022 | −0.0536 | +0.0223 | 23 | YES |
| 3 | Jan 2022–Jan 2023 | +0.1295 | +0.0011 | 24 | no |
| 4 | Jan 2023–Dec 2023 | +0.0462 | −0.0061 | 24 | no |
| 5 | Jan 2024–Dec 2024 | +0.0861 | +0.0271 | 15 | no |

EarlyStopping triggered early in all folds (15–24 epochs of max 100) — model converges quickly.

**Aggregate:**

| Metric | P-ML5 LightGBM | P-ML6 LSTM |
|---|---|---|
| Mean OOS IC | +0.054 | −0.000 |
| ICIR | +0.888 | −0.006 |
| Negative-IC folds | 1/5 | 2/5 |
| OOS Return | +630.2% | **−93.2%** |
| OOS Sharpe | +0.927 | **−0.517** |
| Max Drawdown | −68.0% | **−94.7%** |

**Verdict: HYPOTHESIS REJECTED.** LSTM performs dramatically worse than LightGBM on both IC
and equity. Mean IC ≈ 0 (random), Sharpe −0.517, equity −93.2% vs LightGBM +630.2%.
LSTM beats LightGBM in only 1/5 folds (Fold 2).

**Root cause analysis:**

1. **Daily bar already integrates intra-day sequence:** Each 1d OHLCV bar (open, high, low,
   close, volume, and derived features) already captures the within-day price path. The LSTM's
   30-bar window provides inter-day history, but the key mean-reversion and momentum signals
   encoded in the 12 features are already available in the snapshot at bar t. The LSTM adds
   sequence-of-summaries rather than sequence-of-raw-ticks.

2. **Insufficient training sequences per fold:** With seq_len=30 and ~360 training bars per fold,
   the LSTM trains on only ~330 sequences — far fewer than the ~1,200 samples LightGBM uses.
   Neural networks typically require at least 10× the parameter count in samples; with 64+32
   LSTM units and Dense(1), this training set is marginal.

3. **EarlyStopping fires at 15–24 epochs:** Training loss ≈ 0.0019, val loss ≈ 0.0012 at Fold 3
   termination. The gap suggests mild overfitting within the first 20 epochs on only ~300 training
   sequences — the model learns some noise before early stopping kicks in.

4. **30-bar window may be too long:** The dominant signals in this 12-feature set (mean-reversion:
   `bar_ret`, `bb_zscore`, `rsi`) are concentrated in the most recent 1–5 bars. A 30-bar window
   adds 25+ bars of older, lower-IC history that likely dilutes the signal.

**Implication:** Sequential LSTM architecture is not a reliable upgrade over LightGBM for
daily BTC directional forecasting with this feature set and data size. P-ML5 RegimeEnsemble
(Sharpe +0.927) remains the best ML model. Next directions:
- Try shorter seq_len (5–10 bars) targeting recent momentum signal
- Extend to tick or minute-level data where true intra-sequence patterns exist
- Combine LSTM prediction with P-ML3 Exp-C regime gating to reduce MaxDD
- Try 3–5d horizon where trend memory is more relevant than mean-reversion

---

## F12 — Extended 6yr dataset fixes bull model IC; P-ML5 equity +630% (Sharpe +0.927)
**Date:** 2026-03-03 | **Notebook:** `p_ml5_extended_dataset.ipynb`

Hypothesis (from F10): extending the dataset from 3yr (2022–2025) to **6yr (2019–2025)**
gives ~600+ bull training bars per fold, enabling the bull model to learn trend continuation.

**Dataset:** 2,171 usable bars (2019-01-22 → 2024-12-31) | 5-fold purged walk-forward

**Regime distribution (6yr vs 3yr P-ML4):**
| Period | Bull | Bear | Ranging | Total |
|---|---|---|---|---|
| P-ML4 (3yr 2022–2025) | 31.9% | 17.6% | 50.5% | 1,075 bars |
| P-ML5 (6yr 2019–2025) | 31.6% | 25.6% | 42.8% | 2,171 bars |

**Bull training bars per fold:**
| Fold | Train period | P-ML4 bull bars | P-ML5 bull bars | Bull model? |
|---|---|---|---|---|
| 1 | 2019-01 → 2020-01 | 0 | 15 | FALLBACK |
| 2 | 2019-07 → 2021-01 | 1 | 199 | YES |
| 3 | 2020-07 → 2022-01 | 128 | 273 | YES |
| 4 | 2021-07 → 2023-01 | 133 | 86 | YES |
| 5 | 2022-07 → 2023-12 | 136 | 214 | YES |

Total bull training bars: **787 (P-ML5) vs 398 (P-ML4) — 2.0× multiplier**

**Per-fold OOS IC (P-ML5 ensemble, all test bars):**
| Fold | Test period | Bull train | Bull IC | NB IC | P-ML5 IC | Bull>0? |
|---|---|---|---|---|---|---|
| 1 | Jan 2020–Jan 2021 | 15 | +0.031 (FLIP) | +0.158 | +0.061 | YES |
| 2 | Jan 2021–Jan 2022 | 199 | −0.050 | −0.053 | −0.054 | no |
| 3 | Jan 2022–Jan 2023 | 273 | NaN (0 bull test) | +0.130 | +0.130 | — |
| 4 | Jan 2023–Dec 2023 | 86 | +0.042 | +0.063 | +0.046 | YES |
| 5 | Jan 2024–Dec 2024 | 214 | +0.061 | +0.089 | +0.086 | YES |

**Aggregate:** Mean IC=+0.054, ICIR=+0.888 (vs P-ML4: Mean IC=−0.062, ICIR=−1.319)
**Bull model IC:** Mean=+0.021 (positive in 3/4 folds with bull test bars)
**P-ML4 bull IC:** Mean=−0.102 (negative in all fitted folds) — **IC sign reversed**

**Equity comparison:**
| Strategy | Return | Sharpe | MaxDD |
|---|---|---|---|
| Buy & Hold | +299.6% | +1.379 | −35.4% |
| P-ML3 Exp-C (best prior, 3yr OOS) | +8.8% | +0.280 | −49.8% |
| P-ML4 (3yr RegimeEnsemble) | −2.5% | +0.227 | −57.3% |
| **P-ML5 (6yr RegimeEnsemble)** | **+630.2%** | **+0.927** | **−68.0%** |

Note: P-ML5 OOS covers 2020–2025 (5yr), including the 2020–21 BTC bull run (~$8k→$65k),
which the ensemble (with sign-flipped bull model in Fold 1) largely captured.

**Feature importance — key difference vs P-ML4:**
- `di_diff` importance narrows significantly in bull model vs non-bull (149.8 vs 154.4) —
  still not dramatically divergent, but bull model now has enough data to use all features.
- Importance totals are lower for bull model (gap from 6yr non-bull model is expected
  since non-bull has 3–5× more training bars per fold).

**Key findings:**

1. **Hypothesis confirmed:** Extending to 6yr data fixed the bull model IC sign.
   With 86–273 bull training bars (vs 128–136 in P-ML4), the bull model achieves positive
   IC in 3/4 applicable folds. The 2020–21 bull run in the training set teaches the model
   trend continuation that 2022–2024 alone could not.

2. **P-ML5 equity +630.2% (Sharpe +0.927)** dramatically exceeds P-ML4 (−2.5%) and
   P-ML3 Exp-C (+8.8%), and approaches Buy & Hold (Sharpe +1.379). The strong result is
   partly explained by the OOS window spanning the 2020–21 BTC bull run.

3. **Fold 1 caveat:** Only 15 bull training bars — bull model falls back to sign-flip.
   The Fold 1 OOS (Jan 2020–Jan 2021) covers the 2020 COVID crash + recovery + 2021 bull
   run. The sign-flipped prediction still achieves IC +0.031 on bull bars (by design).

4. **MaxDD worsens to −68.0%** (vs P-ML4 −57.3%, P-ML3 Exp-C −49.8%). The longer OOS
   window includes the 2022 bear market drawdown on top of accumulated bull gains.

5. **Remaining limitation:** Fold 2 (Jan 2021–Jan 2022) bull IC = −0.050, the only
   fitted bull fold with negative IC. The 2021 Q4 sideways-then-crash pattern after ATH
   ($65k) may be hard to distinguish from ranging on 1-day horizon.

**Implication:** 6yr data is the recommended training window for `RegimeEnsemble`.
The bull model now meaningfully contributes positive IC. Next direction: momentum
feature engineering (ret_mean_20, quarterly momentum) and/or longer bull horizon (3–5d)
to further improve bull model consistency across all folds.

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
