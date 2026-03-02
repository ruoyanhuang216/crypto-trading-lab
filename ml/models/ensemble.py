"""Regime-specific LightGBM ensemble.

Trains separate forecasters for bull and non-bull (bear + ranging) market regimes,
then routes each bar to the model appropriate for its detected regime.

Design principles:
  - Non-bull model is always trained (sufficient bars in every fold).
  - Bull model is trained only when `min_bull_bars` of bull training data exist;
    otherwise the non-bull model's sign is flipped for bull bars (P-ML3 Exp-B fallback).
  - Each sub-model uses its own in-fold StandardScaler (inherited from LGBMForecaster),
    so no cross-regime leakage occurs during normalisation.
  - Predict routes each bar to its regime model; regime is passed as a Series at call time
    so the caller (notebook/live-trading) controls regime detection independently.
"""

import numpy as np
import pandas as pd

from .tree import LGBMForecaster


class RegimeEnsemble:
    """Two-model regime-specific ensemble.

    Trains separate ``LGBMForecaster`` instances for:

    * **non-bull regime** (bear + ranging) — learns mean-reversion relationships
    * **bull regime** — learns trend-continuation relationships

    When a fold has fewer than ``min_bull_bars`` bull training bars, the bull model
    is skipped and predictions for bull test bars are obtained by sign-flipping the
    non-bull model (equivalent to P-ML3 Experiment B).

    Usage::

        ensemble = RegimeEnsemble(min_bull_bars=30)
        ensemble.fit(X_train, y_train, regime_train)
        preds = ensemble.predict(X_test, regime_test)
        print("bull model fitted:", ensemble.has_bull_model)
        print(ensemble.bull_importance.head())

    Args:
        min_bull_bars: Minimum number of bull training bars required to fit a
                       dedicated bull model. Folds below this threshold fall back
                       to sign-flipped non-bull predictions. Default 30.
        **params:      Override any default LightGBM hyperparameter (forwarded to
                       both sub-model ``LGBMForecaster`` instances).
    """

    def __init__(self, min_bull_bars: int = 30, **params) -> None:
        self.min_bull_bars  = min_bull_bars
        self.params         = params
        self.non_bull_model: LGBMForecaster | None = None
        self.bull_model:     LGBMForecaster | None = None
        self._has_bull_model: bool = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X:      pd.DataFrame,
        y:      pd.Series,
        regime: pd.Series,
    ) -> "RegimeEnsemble":
        """Fit non-bull model (always) and bull model (if enough data).

        Args:
            X:      Feature DataFrame (training bars × features).
            y:      Forward log-return Series aligned with X.
            regime: String Series of ``'bull'`` | ``'bear'`` | ``'ranging'``
                    labels aligned with X. Bear and ranging are pooled into the
                    non-bull training set.

        Returns:
            self (for chaining).
        """
        bull_mask     = (regime == "bull")
        non_bull_mask = ~bull_mask

        if non_bull_mask.sum() < 1:
            raise ValueError("No non-bull training bars — cannot fit non-bull model.")

        # Non-bull model (bear + ranging pooled) — always trained
        self.non_bull_model = LGBMForecaster(**self.params)
        self.non_bull_model.fit(X[non_bull_mask], y[non_bull_mask])

        # Bull model — trained only when sufficient data exist
        n_bull = int(bull_mask.sum())
        self._has_bull_model = n_bull >= self.min_bull_bars
        if self._has_bull_model:
            self.bull_model = LGBMForecaster(**self.params)
            self.bull_model.fit(X[bull_mask], y[bull_mask])

        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame, regime: pd.Series) -> np.ndarray:
        """Predict using the regime-appropriate model for each bar.

        Args:
            X:      Feature DataFrame (test bars × features).
            regime: String Series of ``'bull'`` | ``'bear'`` | ``'ranging'``
                    labels aligned with X.

        Returns:
            1-D numpy array of predicted log-returns.
            * Non-bull bars → non-bull model prediction.
            * Bull bars (bull model available) → bull model prediction.
            * Bull bars (no bull model) → sign-flipped non-bull prediction.
        """
        if self.non_bull_model is None:
            raise RuntimeError("Call fit() before predict().")

        preds     = np.empty(len(X))
        bull_mask = (regime == "bull").values

        if (~bull_mask).any():
            preds[~bull_mask] = self.non_bull_model.predict(X.iloc[~bull_mask])

        if bull_mask.any():
            if self._has_bull_model:
                preds[bull_mask] = self.bull_model.predict(X.iloc[bull_mask])
            else:
                # Fallback: sign-flip the non-bull model's prediction for bull bars
                preds[bull_mask] = -self.non_bull_model.predict(X.iloc[bull_mask])

        return preds

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def has_bull_model(self) -> bool:
        """True if a dedicated bull model was trained in the last fit() call."""
        return self._has_bull_model

    @property
    def bull_importance(self) -> pd.Series:
        """LightGBM gain-based feature importance for the bull model."""
        if self.bull_model is None:
            raise RuntimeError("No bull model fitted (insufficient bull training bars).")
        return self.bull_model.feature_importance

    @property
    def non_bull_importance(self) -> pd.Series:
        """LightGBM gain-based feature importance for the non-bull model."""
        if self.non_bull_model is None:
            raise RuntimeError("Call fit() first.")
        return self.non_bull_model.feature_importance

    def __repr__(self) -> str:
        bull_status = "fitted" if self._has_bull_model else "fallback(flip)"
        nb_status   = "fitted" if self.non_bull_model is not None else "unfitted"
        return (
            f"RegimeEnsemble("
            f"non_bull={nb_status}, bull={bull_status}, "
            f"min_bull_bars={self.min_bull_bars})"
        )
