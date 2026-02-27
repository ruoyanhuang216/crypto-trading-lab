# Roadmap

Current priority list. Updated at the end of each session.
_Last updated: 2026-02-27 (EOD)_

---

## Now — High Priority

### ~~P1. Apply trend filters to MeanReversion~~ ✅ COMPLETE — hypothesis rejected
Filters hurt MeanReversion in Jan–Jun 2024, same as Breakout. Root cause: in a bull
market the strategy fails on almost every trade regardless of regime. Bar-by-bar filters
cannot fix a structural directional mismatch. See F4, F5, [2026-02-25-p1](daily/2026-02-25-p1.md).

### P1b. Long-only MeanReversion variant ← NEW
**Why:** F5 suggests the fix for MeanReversion in a bull market is directional bias,
not a regime filter. A long-only variant (only buy oversold dips, never short overbought)
combined with a high-level trend switch (e.g. 200-bar MA) may preserve the ranging-market
edge while avoiding bull-run losses.
**Effort:** Low — small extension to `bollinger_bands.py` or a notebook experiment.

### P1b. Long-only MeanReversion variant ← NEW
**Why:** F5 showed filters cannot fix MeanReversion in a bull market at the bar level.
The structural fix is directional bias: only take long signals (buy oversold dips),
combined with a high-level trend switch (e.g. 200-bar MA — if price is above the 200MA,
go long-only; if below, go short-only or flat). This preserves ranging-market edge
while avoiding shorting a bull run.
**Effort:** Low — notebook experiment first, no strategy code changes yet.

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
