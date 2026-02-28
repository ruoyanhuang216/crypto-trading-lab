# Confirmed Findings

A distilled index of key findings across all sessions.
Each entry references the daily log where it was first observed.

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
