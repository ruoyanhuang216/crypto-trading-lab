"""Regime-aware LightGBM strategy.

Wraps a pre-trained ``RegimeEnsemble`` into the ``BaseStrategy`` interface so
the ML signal can be backtested with the standard engine and combined with
any downstream risk overlay.

Design principles:
  - The ensemble is trained externally (in a walk-forward notebook loop) and
    passed to the constructor already fitted. This separates the *training*
    concern (handled by the notebook / pipeline) from the *signal generation*
    concern (handled by this class).
  - ``generate_signals(df)`` accepts a raw OHLCV DataFrame, builds features
    and regime labels internally, and outputs a copy of df with added columns.
  - ``predict(X, regime)`` is a thin wrapper that accepts pre-computed features
    and regime labels — useful when features are computed globally for efficiency
    in a walk-forward validation loop.
  - Two position-sizing modes:
      binary   (default): signal ∈ {−1, 0, +1} = np.sign(pred)
      scaled:             signal = clip(pred_zscore × scale, −max_pos, +max_pos)
                          using a rolling 60-bar z-score of raw predictions.

Usage::

    from ml.models import RegimeEnsemble
    from ml.regime import RegimeClassifier
    from strategies.ml import RegimeLGBMStrategy

    # Train ensemble externally (e.g. in a walk-forward loop)
    rc  = RegimeClassifier()
    ens = RegimeEnsemble(min_bull_bars=30)
    ens.fit(X_train, y_train, regime_train)

    # Wrap in strategy (binary mode)
    strat = RegimeLGBMStrategy(
        ensemble=ens, regime_classifier=rc, feature_columns=FEATURES_V2
    )
    sig_df = strat.generate_signals(df_ohlcv_test)
    print(sig_df["signal"].value_counts())

    # Scaled-position mode
    strat_scaled = RegimeLGBMStrategy(
        ensemble=ens, regime_classifier=rc, feature_columns=FEATURES_V2,
        scale_positions=True, position_scale=0.5, max_position=1.0,
        pred_zscore_window=60,
    )
    sig_df_scaled = strat_scaled.generate_signals(df_ohlcv_test)
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from ml.features import build_feature_matrix
from ml.features.momentum import build_momentum_features
from ml.regime import RegimeClassifier
from ml.models import RegimeEnsemble


class RegimeLGBMStrategy(BaseStrategy):
    """Regime-aware LightGBM strategy wrapping a pre-trained ``RegimeEnsemble``.

    Args:
        ensemble:           A fitted ``RegimeEnsemble`` instance.
        regime_classifier:  ``RegimeClassifier`` used to compute live regime labels.
        feature_columns:    Ordered list of feature column names expected by the
                            ensemble (must match the feature set used at training time).
        scale_positions:    If False (default), output binary signals ∈ {−1, 0, +1}.
                            If True, scale by a rolling z-score of raw predictions.
        pred_zscore_window: Rolling window (bars) for computing the prediction
                            z-score used in scaled mode (default 60).
        position_scale:     Multiplier applied to pred z-score before clipping
                            (default 0.5; keeps most positions below ±0.5 leverage).
        max_position:       Maximum absolute position allowed after scaling
                            (default 1.0 = fully invested, no leverage).
    """

    def __init__(
        self,
        ensemble:           RegimeEnsemble,
        regime_classifier:  RegimeClassifier,
        feature_columns:    list,
        *,
        scale_positions:    bool  = False,
        pred_zscore_window: int   = 60,
        position_scale:     float = 0.5,
        max_position:       float = 1.0,
        include_momentum:   bool  = True,
    ) -> None:
        super().__init__(
            scale_positions=scale_positions,
            pred_zscore_window=pred_zscore_window,
            position_scale=position_scale,
            max_position=max_position,
        )
        self.ensemble           = ensemble
        self.regime_classifier  = regime_classifier
        self.feature_columns    = list(feature_columns)
        self.scale_positions    = scale_positions
        self.pred_zscore_window = pred_zscore_window
        self.position_scale     = position_scale
        self.max_position       = max_position
        self.include_momentum   = include_momentum

    # ── Core signal generation ─────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features internally from raw OHLCV and return signal column.

        Warmup bars (where rolling windows produce NaN) will have signal = 0.
        To avoid look-ahead bias the returned 'signal' should be shifted by the
        caller (the backtesting engine does this automatically via
        ``position = sig_df["signal"].shift(1)``).

        Args:
            df: OHLCV DataFrame with DatetimeIndex. Should include at least 250
                leading warmup bars to allow convergence of SMA(200) in the
                regime classifier, ADX(14), and momentum rolling windows (up to
                80 bars). The signal column for warmup bars will be 0.

        Returns:
            Copy of df with added columns:
                'pred'         — raw ensemble prediction (float log-return)
                'regime'       — regime label ('bull'/'bear'/'ranging')
                'signal'       — position in {−1, 0, +1} (binary) or
                                 float ∈ [−max_pos, +max_pos] (scaled)
                'pred_zscore'  — rolling z-score of pred (only if scaled mode)
        """
        df = df.copy()

        # ── Build features ────────────────────────────────────────────────
        feats_base = build_feature_matrix(df)   # technical + lag + time (12 base)
        if self.include_momentum:
            feats_mom = build_momentum_features(df)
            feats_all = pd.concat([feats_base, feats_mom], axis=1)
        else:
            feats_all = feats_base

        # Restrict to the feature columns expected by the trained ensemble
        missing = [c for c in self.feature_columns if c not in feats_all.columns]
        if missing:
            raise ValueError(
                f"RegimeLGBMStrategy: feature columns missing from built features: {missing}\n"
                f"Available: {list(feats_all.columns)}"
            )
        X = feats_all[self.feature_columns]

        # ── Regime labels ─────────────────────────────────────────────────
        regime_df = self.regime_classifier.transform(df)
        regime    = regime_df["regime"].fillna("ranging")

        # ── Find valid (non-NaN) rows and predict ─────────────────────────
        valid_mask = X.notna().all(axis=1)
        preds_full = np.full(len(df), np.nan)

        if valid_mask.sum() > 0:
            X_valid      = X[valid_mask]
            regime_valid = regime[valid_mask]
            preds_full[valid_mask] = self.ensemble.predict(X_valid, regime_valid)

        pred_series = pd.Series(preds_full, index=df.index)

        # ── Position sizing ───────────────────────────────────────────────
        if self.scale_positions:
            rm = pred_series.rolling(self.pred_zscore_window, min_periods=1).mean()
            rs = pred_series.rolling(self.pred_zscore_window, min_periods=1).std()
            rs = rs.replace(0, np.nan)
            pred_z  = (pred_series - rm) / rs
            signal  = (pred_z * self.position_scale).clip(
                -self.max_position, self.max_position
            )
            df["pred_zscore"] = pred_z
        else:
            signal = np.sign(pred_series)

        df["pred"]   = pred_series
        df["regime"] = regime
        df["signal"] = signal.fillna(0)

        return df

    # ── Convenience: predict from pre-computed features ────────────────────

    def predict(
        self,
        X:      pd.DataFrame,
        regime: pd.Series,
    ) -> np.ndarray:
        """Return raw ensemble predictions from pre-computed features.

        This skips the internal feature-building step and is useful when features
        are computed globally once (e.g. in a walk-forward loop).

        Args:
            X:      Feature DataFrame (bars × features), same columns as
                    ``self.feature_columns``.
            regime: Regime label Series aligned with X.

        Returns:
            1-D numpy array of predicted log-returns.
        """
        return self.ensemble.predict(X[self.feature_columns], regime)

    def signal_from_predictions(
        self,
        preds:  np.ndarray,
        index:  pd.Index,
    ) -> pd.Series:
        """Convert raw predictions to a position Series.

        Applies the same binary / scaled logic as ``generate_signals`` but
        starting from already-computed predictions.  Useful for unit-testing
        the position-sizing logic independently of feature building.

        Args:
            preds: Raw ensemble predictions (same order as ``index``).
            index: DatetimeIndex for the resulting Series.

        Returns:
            Position Series aligned to ``index``.
        """
        pred_s = pd.Series(preds, index=index, dtype=float)

        if self.scale_positions:
            rm = pred_s.rolling(self.pred_zscore_window, min_periods=1).mean()
            rs = pred_s.rolling(self.pred_zscore_window, min_periods=1).std()
            rs = rs.replace(0, np.nan)
            pred_z = (pred_s - rm) / rs
            signal = (pred_z * self.position_scale).clip(
                -self.max_position, self.max_position
            )
        else:
            signal = np.sign(pred_s)

        return signal.fillna(0)

    def __repr__(self) -> str:
        mode = (
            f"scaled(scale={self.position_scale}, max={self.max_position}, "
            f"window={self.pred_zscore_window})"
            if self.scale_positions
            else "binary"
        )
        return (
            f"RegimeLGBMStrategy("
            f"n_features={len(self.feature_columns)}, "
            f"positioning={mode}, "
            f"ensemble={self.ensemble!r})"
        )
