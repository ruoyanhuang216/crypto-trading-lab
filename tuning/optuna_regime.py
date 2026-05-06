"""Optuna hyperparameter tuning for ``RegimeEnsemble`` + scaled positioning.

Implements the P-ML15 design agreed 2026-04-17:

  * **Search space (7 dims):** LightGBM (``n_estimators``, ``max_depth``,
    ``learning_rate``, ``min_child_samples``, ``reg_alpha``, ``reg_lambda``)
    + positioning (``pred_zscore_window``, ``position_scale``).
    Bull and non-bull sub-models share hyperparameters in this first pass.
  * **CV:** shared-study 3-fold inner purged walk-forward. Each trial
    reports the mean Sharpe across folds as its objective value, with
    per-fold intermediate values so ``MedianPruner`` can short-circuit
    clearly-bad trials.
  * **Objective:** Sharpe ratio on the OOS equity curve produced by the
    scaled ``RegimeLGBMStrategy`` pipeline (fit → predict → pred z-score
    positioning → equity → ``compute_metrics``).

The outer 5-fold walk-forward used for final reporting lives in the
notebook — this module is purely the tuning inner loop so Optuna never
sees outer test bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from backtesting.metrics import compute_metrics
from ml.models import RegimeEnsemble
from ml.validation import purged_wf_splits


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class TuningResult:
    """Outcome of an Optuna study."""
    study:       optuna.Study
    best_params: dict
    best_value:  float            # mean inner-CV Sharpe for best trial
    n_trials:    int


# ── Search space ─────────────────────────────────────────────────────────────

def _suggest_params(trial: optuna.Trial) -> dict:
    """Sample one hyperparameter configuration from the 7-dim medium space."""
    return {
        # LightGBM (shared across bull and non-bull sub-models)
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        # Scaled positioning
        "pred_zscore_window": trial.suggest_int("pred_zscore_window", 20, 120, step=10),
        "position_scale":     trial.suggest_float("position_scale", 0.2, 1.0),
    }


def _split_params(params: dict) -> tuple[dict, dict]:
    """Separate model params (for ``LGBMForecaster``) from positioning params."""
    positioning_keys = {"pred_zscore_window", "position_scale"}
    model_params = {k: v for k, v in params.items() if k not in positioning_keys}
    pos_params   = {k: params[k] for k in positioning_keys}
    return model_params, pos_params


# ── Equity + Sharpe for one fold ─────────────────────────────────────────────

def _fold_sharpe(
    X:        pd.DataFrame,
    y:        pd.Series,
    regime:   pd.Series,
    bar_ret:  pd.Series,
    train_idx: np.ndarray,
    test_idx:  np.ndarray,
    model_params: dict,
    pos_params:   dict,
    min_bull_bars: int,
    max_position:  float,
) -> float:
    """Fit on train, predict on test, apply scaled positioning, return Sharpe.

    Mirrors the walk-forward loop in ``p_ml9_strategy_integration.ipynb`` so
    the inner-CV metric is comparable to the outer reporting.
    """
    ens = RegimeEnsemble(min_bull_bars=min_bull_bars, **model_params)
    ens.fit(X.iloc[train_idx], y.iloc[train_idx], regime.iloc[train_idx])

    preds = ens.predict(X.iloc[test_idx], regime.iloc[test_idx])
    pred_s = pd.Series(preds, index=X.index[test_idx], dtype=float)

    # Scaled positioning: rolling z-score × scale, clipped to ±max_position
    w  = pos_params["pred_zscore_window"]
    sc = pos_params["position_scale"]
    rm = pred_s.rolling(w, min_periods=1).mean()
    rs = pred_s.rolling(w, min_periods=1).std().replace(0, np.nan)
    pred_z = (pred_s - rm) / rs
    signal = (pred_z * sc).clip(-max_position, max_position).fillna(0)

    # Equity curve: shift signal by 1 to avoid look-ahead
    test_ret  = bar_ret.iloc[test_idx]
    position  = signal.shift(1).fillna(0)
    equity    = (1 + position * test_ret).cumprod()
    equity    = equity / equity.iloc[0]

    if len(equity) < 2:
        return float("nan")

    return float(compute_metrics(equity)["sharpe_ratio"])


# ── Objective function (closure factory) ─────────────────────────────────────

def _make_objective(
    X:        pd.DataFrame,
    y:        pd.Series,
    regime:   pd.Series,
    bar_ret:  pd.Series,
    splits:   Sequence[tuple[np.ndarray, np.ndarray]],
    min_bull_bars: int,
    max_position:  float,
):
    """Build the Optuna objective closure bound to a dataset and splits."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        model_params, pos_params = _split_params(params)

        fold_sharpes: list[float] = []
        for k, (tr_idx, te_idx) in enumerate(splits):
            s = _fold_sharpe(
                X, y, regime, bar_ret,
                tr_idx, te_idx,
                model_params, pos_params,
                min_bull_bars=min_bull_bars,
                max_position=max_position,
            )
            # NaN Sharpe (e.g. degenerate fold with std=0) penalises the trial
            if np.isnan(s):
                s = -5.0
            fold_sharpes.append(s)

            # Report intermediate value so MedianPruner can short-circuit.
            # We report the running mean — trials substantially below the
            # median of past trials' running means at the same step are pruned.
            trial.report(float(np.mean(fold_sharpes)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_sharpes))

    return objective


# ── Entry point ──────────────────────────────────────────────────────────────

def tune_regime_ensemble(
    X:         pd.DataFrame,
    y:         pd.Series,
    regime:    pd.Series,
    bar_ret:   pd.Series,
    *,
    n_trials:      int   = 100,
    n_inner_folds: int   = 3,
    train_frac:    float = 0.6,
    purge_bars:    int   = 1,
    min_bull_bars: int   = 30,
    max_position:  float = 1.0,
    seed:          int   = 42,
    study_name:    str   | None = None,
    show_progress_bar: bool = True,
) -> TuningResult:
    """Run an Optuna study tuning ``RegimeEnsemble`` + scaled positioning.

    Args:
        X:             Aligned feature matrix (rows = bars, cols = FEATURES_V2).
        y:             Forward log-return labels aligned with X.
        regime:        Regime label Series ('bull'/'bear'/'ranging') aligned with X.
        bar_ret:       Bar-over-bar simple returns aligned with X
                       (``close.pct_change()`` reindexed to X).
        n_trials:      Optuna trial budget (default 100).
        n_inner_folds: Purged walk-forward folds used for each trial (default 3).
        train_frac:    Rolling training fraction for inner folds (default 0.6).
        purge_bars:    Purge window at end of each train fold (default 1 = 1d horizon).
        min_bull_bars: Min bull training bars required to fit a dedicated bull model.
        max_position:  Position clip applied after pred_zscore × position_scale.
        seed:          Random seed for TPE sampler reproducibility.
        study_name:    Optional Optuna study name (useful when persisting studies).
        show_progress_bar: Whether Optuna displays its progress bar.

    Returns:
        ``TuningResult`` containing the study, best params, and best value.
    """
    if not (len(X) == len(y) == len(regime) == len(bar_ret)):
        raise ValueError(
            f"Aligned inputs required: len(X)={len(X)}, len(y)={len(y)}, "
            f"len(regime)={len(regime)}, len(bar_ret)={len(bar_ret)}"
        )

    splits = list(purged_wf_splits(
        len(X), n_splits=n_inner_folds,
        train_frac=train_frac, window_type="rolling",
        purge_bars=purge_bars,
    ))
    if len(splits) < 2:
        raise ValueError(
            f"Only {len(splits)} inner folds produced from {len(X)} bars — "
            "shorten train_frac or use more data."
        )

    sampler = TPESampler(seed=seed, multivariate=True)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )

    objective = _make_objective(
        X, y, regime, bar_ret, splits,
        min_bull_bars=min_bull_bars,
        max_position=max_position,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        gc_after_trial=True,
    )

    return TuningResult(
        study       = study,
        best_params = dict(study.best_params),
        best_value  = float(study.best_value),
        n_trials    = len(study.trials),
    )
