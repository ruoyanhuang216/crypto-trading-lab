# Roadmap

Current priority list. Updated at the end of each session.
_Last updated: 2026-02-25_

---

## Now — High Priority

### P1. Apply trend filters to MeanReversion (not Breakout)
**Why:** F2 showed filters hurt Breakout due to lag mismatch. The natural use case
is the opposite: filter *out* trending regimes from a mean-reversion strategy.
Apply `ADXTrend` (sit out when ADX > 25) to `BollingerMeanReversion` and compare.
**Effort:** Low — all code exists, just a new notebook cell or short script.

### P2. Implement volatility signals
**Why:** BB width and ATR are direct inputs for position sizing and regime detection.
BB width in particular pairs naturally with the existing Bollinger strategies
(compressing bands = pre-breakout setup).
**Files:** `signals/volatility/bb_width.py`, `signals/volatility/atr.py`

---

## Next — Medium Priority

### P3. Test signals on longer timeframes (4h, daily)
**Why:** F3 showed trend signals have sub-random accuracy at 1h. Lower-frequency
bars have less noise — same signals may have genuine predictive power at 4h/daily.
**Effort:** Low — change `timeframe` in config and re-run the existing notebook.

### P4. Walk-forward / train-test split in backtesting
**Why:** All results so far are in-sample. Before trusting any strategy, need
out-of-sample validation. Add a walk-forward engine to `backtesting/`.
**Files:** `backtesting/walk_forward.py`

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
