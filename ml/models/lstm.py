"""LSTM forecasting model for multi-bar temporal sequence learning.

Design principles:
  - Mirrors LGBMForecaster interface (fit/predict) for drop-in walk-forward compatibility.
  - Scaler is fit on training data only (no global normalisation → no leakage).
  - Sequences are built with a sliding window (seq_len bars → predict bar seq_len-1).
  - predict() prepends (seq_len-1) zeros so output length matches input length,
    consistent with LGBMForecaster. Callers exclude warmup bars from IC.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class LSTMForecaster:
    """LSTM regressor that predicts forward log-returns from a sequence of bars.

    Builds two stacked LSTM layers with dropout and early stopping.
    Mirrors `LGBMForecaster` interface for compatibility with the walk-forward loop.

    Usage::

        model = LSTMForecaster(seq_len=30, units=64)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)   # length == len(X_test)
    """

    def __init__(
        self,
        seq_len:       int   = 30,
        units:         int   = 64,
        dropout:       float = 0.2,
        epochs:        int   = 100,
        batch_size:    int   = 32,
        learning_rate: float = 1e-3,
        verbose:       int   = 0,
    ):
        """
        Args:
            seq_len:       Number of consecutive bars fed into the LSTM as one input window.
            units:         Number of units in the first LSTM layer (second = units // 2).
            dropout:       Dropout rate applied after each LSTM layer.
            epochs:        Maximum training epochs (EarlyStopping will terminate earlier).
            batch_size:    Mini-batch size.
            learning_rate: Adam learning rate.
            verbose:       Keras verbosity (0 = silent).
        """
        self.seq_len       = seq_len
        self.units         = units
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.verbose       = verbose

        self.scaler:          StandardScaler | None = None
        self.model                                   = None   # Keras model
        self.feature_names_:  list[str]              = []
        self.epochs_stopped_: int                    = 0

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMForecaster":
        """Fit scaler + LSTM on training data.

        Args:
            X: Feature DataFrame (rows = bars, columns = features).
            y: Forward log-return series aligned with X.

        Returns:
            self (for chaining).
        """
        import tensorflow as tf
        from tensorflow import keras

        # Silence TF logs unless verbose > 0
        if self.verbose == 0:
            tf.get_logger().setLevel("ERROR")

        self.feature_names_ = list(X.columns)

        # Step 1: fit scaler on training data only
        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X.values)

        # Step 2: build sliding-window sequences
        X_seq, y_seq = self._make_sequences(X_sc, y.values)

        # Step 3: build Keras model
        n_feats = X_sc.shape[1]
        inp = keras.Input(shape=(self.seq_len, n_feats))
        x   = keras.layers.LSTM(self.units, return_sequences=True)(inp)
        x   = keras.layers.Dropout(self.dropout)(x)
        x   = keras.layers.LSTM(self.units // 2)(x)
        x   = keras.layers.Dropout(self.dropout)(x)
        out = keras.layers.Dense(1)(x)

        self.model = keras.Model(inp, out)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        # Step 4: fit with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        hist = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=self.verbose,
        )

        # Step 5: record how many epochs actually ran
        self.epochs_stopped_ = len(hist.history["loss"])
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict forward log-returns for new bars.

        Prepends (seq_len - 1) zeros so the output array length equals len(X),
        matching LGBMForecaster convention. Callers should exclude the first
        (seq_len - 1) warmup bars from IC computation.

        Args:
            X: Feature DataFrame with the same columns used in fit().

        Returns:
            1-D numpy array of predicted log-returns, length == len(X).
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit() before predict()")

        X_sc  = self.scaler.transform(X[self.feature_names_].values)
        X_seq = self._make_sequences(X_sc)          # (n - seq_len + 1, seq_len, n_feats)
        raw   = self.model.predict(X_seq, verbose=0).flatten()

        warmup = np.zeros(self.seq_len - 1)
        return np.concatenate([warmup, raw])

    # ── Sequence builder ──────────────────────────────────────────────────────

    def _make_sequences(self, X_arr: np.ndarray, y_arr=None):
        """Build sliding-window sequences.

        Args:
            X_arr: Scaled feature array, shape (n, n_features).
            y_arr: Optional label array, shape (n,).
                   If provided, label[i] = y[i + seq_len - 1]
                   (aligned to the last bar of the window).

        Returns:
            (X_seq, y_seq) if y_arr provided, else X_seq alone.
            X_seq shape: (n - seq_len + 1, seq_len, n_features).
        """
        n       = len(X_arr)
        n_win   = n - self.seq_len + 1
        n_feats = X_arr.shape[1]

        X_seq = np.empty((n_win, self.seq_len, n_feats), dtype=np.float32)
        for i in range(n_win):
            X_seq[i] = X_arr[i : i + self.seq_len]

        if y_arr is not None:
            y_seq = y_arr[self.seq_len - 1 :].astype(np.float32)
            return X_seq, y_seq

        return X_seq

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def feature_importance(self):
        """Not available for LSTMs. Returns None with a warning."""
        warnings.warn(
            "LSTMForecaster has no built-in gain-based feature importance. "
            "Use SHAP or permutation importance for post-hoc attribution.",
            UserWarning,
            stacklevel=2,
        )
        return None

    def __repr__(self) -> str:
        status = (
            f"fitted (epochs={self.epochs_stopped_})"
            if self.model is not None
            else "unfitted"
        )
        return (
            f"LSTMForecaster({status}, seq_len={self.seq_len}, "
            f"units={self.units}, dropout={self.dropout})"
        )
