"""Hidden Markov Model regime classifier for late-bull detection.

Uses a Gaussian HMM to discover latent market states from observable
momentum/volatility features. The key motivation (H4): the rule-based
``RegimeClassifier`` cannot distinguish "early bull" from "late/overextended
bull", causing the LightGBM bull model to go aggressively long right before
ATH crashes (Fold 2 bull IC = -0.128 in P-ML7).

The HMM is trained on 4 observable features that capture trend strength,
volatility, overextension, and momentum acceleration:
  1. ret_20          — 20-bar cumulative log return (trend direction)
  2. atr_pct         — ATR / close (realised volatility)
  3. mom_zscore_20   — z-score of 20-bar return (overextension signal)
  4. ret_5_minus_20  — short minus medium momentum (acceleration vs exhaustion)

Usage::

    from ml.regime.hmm import HMMRegimeClassifier

    hmm_rc = HMMRegimeClassifier(n_states=4)
    hmm_rc.fit(df_train)               # fit on training OHLCV
    states = hmm_rc.predict(df_test)    # predict states on test OHLCV
    feats  = hmm_rc.build_features(df_test)  # one-hot + probabilities
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# Observation features used by the HMM
_OBS_FEATURES = ["ret_20", "atr_pct", "mom_zscore_20", "ret_5_minus_20"]


class HMMRegimeClassifier:
    """Gaussian HMM regime classifier.

    Discovers latent market states from momentum/volatility observations.
    States are deterministically sorted by mean ``ret_20`` after fitting
    so that state numbering is consistent across folds (lowest = most
    bearish, highest = most bullish).

    Args:
        n_states:        Number of hidden states. Default 4.
        covariance_type: HMM covariance type. Default ``"full"``.
        n_iter:          Maximum EM iterations. Default 100.
        random_state:    Random seed for reproducibility.
    """

    def __init__(
        self,
        n_states:        int = 4,
        covariance_type: str = "full",
        n_iter:          int = 100,
        random_state:    int = 42,
    ) -> None:
        self.n_states        = n_states
        self.covariance_type = covariance_type
        self.n_iter          = n_iter
        self.random_state    = random_state

        self._model:   GaussianHMM | None    = None
        self._scaler:  StandardScaler | None  = None
        self._sort_map: np.ndarray | None     = None  # old→new state mapping

    # ------------------------------------------------------------------
    # Feature computation (static, causal)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_observations(df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 4 HMM observation features from OHLCV.

        Returns DataFrame with columns: ret_20, atr_pct, mom_zscore_20,
        ret_5_minus_20. NaN rows at start (warmup).
        """
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        log_c = np.log(close)

        # ret_20: 20-bar cumulative log return
        ret_20 = log_c - log_c.shift(20)

        # atr_pct: ATR(14) / close
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()
        atr_pct = atr / close

        # mom_zscore_20: z-score of ret_20 over 60-bar window
        rm = ret_20.rolling(60, min_periods=20).mean()
        rs = ret_20.rolling(60, min_periods=20).std().replace(0, np.nan)
        mom_zscore_20 = (ret_20 - rm) / rs

        # ret_5_minus_20: short - medium momentum
        ret_5 = log_c - log_c.shift(5)
        ret_5_minus_20 = ret_5 - ret_20

        return pd.DataFrame({
            "ret_20":         ret_20,
            "atr_pct":        atr_pct,
            "mom_zscore_20":  mom_zscore_20,
            "ret_5_minus_20": ret_5_minus_20,
        }, index=df.index)

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "HMMRegimeClassifier":
        """Fit the HMM on training OHLCV data.

        Computes observation features internally, scales them, and fits
        the Gaussian HMM. States are sorted by ascending mean ``ret_20``
        so that numbering is deterministic across folds.

        Args:
            df: OHLCV DataFrame with DatetimeIndex (training data only).

        Returns:
            self (for chaining).
        """
        obs = self._compute_observations(df)
        valid = obs.dropna()
        if len(valid) < self.n_states * 10:
            raise ValueError(
                f"Only {len(valid)} valid observations after warmup; "
                f"need at least {self.n_states * 10} for {self.n_states}-state HMM."
            )

        # Scale observations
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(valid.values)

        # Fit HMM
        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            tol=1e-4,
        )
        self._model.fit(X_scaled)

        # Sort states by mean ret_20 (first observation feature)
        # so state 0 = most bearish, state n-1 = most bullish
        raw_states = self._model.predict(X_scaled)
        state_mean_ret = np.array([
            valid["ret_20"].values[raw_states == s].mean()
            if (raw_states == s).any() else -np.inf
            for s in range(self.n_states)
        ])
        self._sort_map = np.argsort(np.argsort(state_mean_ret))
        # _sort_map[old_state] = new_state (sorted rank)

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict HMM state for each bar (Viterbi decoding).

        Args:
            df: OHLCV DataFrame with DatetimeIndex.

        Returns:
            Integer Series of state labels (0 = most bearish, n-1 = most
            bullish). NaN for warmup bars.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        obs = self._compute_observations(df)
        valid_mask = obs.notna().all(axis=1)
        states = pd.Series(np.nan, index=df.index, name="hmm_state", dtype=float)

        if valid_mask.sum() > 0:
            X = self._scaler.transform(obs[valid_mask].values)
            raw = self._model.predict(X)
            mapped = self._sort_map[raw]
            states[valid_mask] = mapped.astype(float)

        return states

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return posterior state probabilities for each bar.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.

        Returns:
            DataFrame with columns ``hmm_prob_0`` ... ``hmm_prob_{n-1}``
            (sorted states). NaN for warmup bars.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")

        obs = self._compute_observations(df)
        valid_mask = obs.notna().all(axis=1)

        cols = [f"hmm_prob_{s}" for s in range(self.n_states)]
        proba_df = pd.DataFrame(np.nan, index=df.index, columns=cols)

        if valid_mask.sum() > 0:
            X = self._scaler.transform(obs[valid_mask].values)
            raw_proba = self._model.predict_proba(X)
            # Reorder columns according to sorted state mapping
            sorted_proba = np.zeros_like(raw_proba)
            for old_s in range(self.n_states):
                new_s = self._sort_map[old_s]
                sorted_proba[:, new_s] = raw_proba[:, old_s]
            proba_df.loc[valid_mask] = sorted_proba

        return proba_df

    def build_features(
        self,
        df: pd.DataFrame,
        *,
        use_proba: bool = False,
    ) -> pd.DataFrame:
        """Build HMM-derived features for the ML pipeline.

        Args:
            df:        OHLCV DataFrame with DatetimeIndex.
            use_proba: If False (default), return one-hot state columns.
                       If True, return posterior probability columns.

        Returns:
            DataFrame with ``n_states`` columns, aligned to df's index.
        """
        if use_proba:
            return self.predict_proba(df)

        states = self.predict(df)
        feats = pd.DataFrame(index=df.index)
        for s in range(self.n_states):
            feats[f"hmm_state_{s}"] = (states == s).astype(float)
            # NaN propagation: warmup bars (state=NaN) → 0 in all columns
            feats.loc[states.isna(), f"hmm_state_{s}"] = np.nan
        return feats

    def state_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Print interpretable summary of each state's characteristics.

        Args:
            df: OHLCV DataFrame (ideally the training data).

        Returns:
            DataFrame with mean observation values per state.
        """
        obs = self._compute_observations(df)
        states = self.predict(df)
        valid = obs.notna().all(axis=1) & states.notna()

        summary = obs[valid].copy()
        summary["state"] = states[valid].astype(int)
        return summary.groupby("state")[_OBS_FEATURES].mean()

    def __repr__(self) -> str:
        status = "fitted" if self._model is not None else "unfitted"
        return (
            f"HMMRegimeClassifier(n_states={self.n_states}, "
            f"cov={self.covariance_type}, status={status})"
        )
