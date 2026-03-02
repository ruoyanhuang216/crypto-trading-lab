"""LightGBM forecasting model with in-fold feature scaling.

Design principles:
  - Scaler is fit on training data only (no global normalisation → no leakage).
  - Conservative defaults: shallow trees, regularisation, subsampling.
  - `feature_importance` property returns a tidy pandas Series for inspection.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


# Default hyperparameters — tunable via Optuna in P7
_DEFAULT_PARAMS = {
    "objective":          "regression",
    "metric":             "rmse",
    "n_estimators":       300,
    "learning_rate":      0.05,
    "max_depth":          4,
    "num_leaves":         15,
    "min_child_samples":  20,    # guards against overfitting on small folds
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         0.1,
    "random_state":       42,
    "n_jobs":             -1,
    "verbose":            -1,
}


class LGBMForecaster:
    """LightGBM regressor that predicts forward log-returns.

    Wraps `lgb.LGBMRegressor` with a `StandardScaler` that is fit inside
    each training fold to avoid data leakage.

    Usage::

        model = LGBMForecaster()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(model.feature_importance.head())
    """

    def __init__(self, **params):
        """
        Args:
            **params: Override any key in the default LightGBM parameter dict.
        """
        self.params = {**_DEFAULT_PARAMS, **params}
        self.model:  lgb.LGBMRegressor | None = None
        self.scaler: StandardScaler | None     = None
        self._feature_names: list[str]         = []

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMForecaster":
        """Fit scaler + LightGBM on training data.

        Args:
            X: Feature DataFrame (rows = bars, columns = features).
            y: Forward log-return series aligned with X.

        Returns:
            self (for chaining).
        """
        self._feature_names = list(X.columns)
        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X.values)

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_sc, y.values)
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict forward log-returns for new bars.

        Args:
            X: Feature DataFrame with the same columns used in fit().

        Returns:
            1-D numpy array of predicted log-returns.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit() before predict()")
        X_sc = self.scaler.transform(X[self._feature_names].values)
        return self.model.predict(X_sc)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def feature_importance(self) -> pd.Series:
        """LightGBM gain-based feature importance, sorted descending."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        return pd.Series(
            self.model.feature_importances_,
            index=self._feature_names,
            name="importance",
        ).sort_values(ascending=False)

    def __repr__(self) -> str:
        status = "fitted" if self.model is not None else "unfitted"
        return f"LGBMForecaster({status}, params={self.params})"
