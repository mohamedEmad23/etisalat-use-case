"""Task 3.4 — Isotonic calibration fitted on train-fold (OOF) predictions.

The chat surface serves the CALIBRATED probability; calibration is fitted on
out-of-fold train predictions only, never on the test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(y_oof: np.ndarray, proba_oof: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression on out-of-fold train predictions."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_oof, y_oof)
    return iso


def calibrate(iso: IsotonicRegression, proba: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to fresh probabilities."""
    return iso.predict(proba)
