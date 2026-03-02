"""Purged walk-forward splits for financial ML.

Standard k-fold cross-validation leaks future information in financial time series
because nearby samples have overlapping labels (multi-bar-ahead targets). The fix:

  Purge  — remove the last `purge_bars` from each training fold.
           These bars have forward-return labels that span into the test period,
           so their targets are partially determined by test-period prices.

           For a label horizon h, set purge_bars=h.
           e.g. 1-day-ahead label → purge_bars=1 (drop last training bar).

Note: full embargo (as in López de Prado ch. 7) is the cross-validation variant
where samples are drawn non-sequentially. For *sequential* walk-forward the only
leakage risk is label overlap, which purge handles completely.
"""

import numpy as np

from backtesting.walk_forward import _make_splits


def purged_wf_splits(
    n:           int,
    n_splits:    int,
    train_frac:  float,
    window_type: str = "rolling",
    purge_bars:  int = 1,
):
    """Yield (train_idx, test_idx) arrays for purged sequential walk-forward CV.

    Args:
        n:            Total number of bars in the dataset.
        n_splits:     Number of OOS test folds.
        train_frac:   Rolling training window fraction (ignored for anchored).
        window_type:  'rolling' or 'anchored'.
        purge_bars:   Bars to drop from the end of each training set
                      (set equal to the label horizon to eliminate label leakage).

    Yields:
        (train_idx, test_idx): 1-D integer numpy arrays of bar positions.
        Folds where the purged training set has fewer than 20 bars are skipped.
    """
    base_splits = _make_splits(n, n_splits, train_frac, window_type)

    for tr_s, tr_e, te_s, te_e in base_splits:
        # Purge: drop the last purge_bars whose labels overlap with the test period
        tr_e_purged = tr_e - purge_bars
        if tr_e_purged - tr_s < 20:
            continue   # too little training data after purge

        train_idx = np.arange(tr_s, tr_e_purged)
        test_idx  = np.arange(te_s, te_e)
        yield train_idx, test_idx
