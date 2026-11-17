"""Probability calibration: make a predicted 10% really mean 10% before applying the cost threshold.

Class weights (balanced / pos_weight) push raw probabilities far above the 1.8% default rate, so
uncalibrated scores must never be compared with the cost threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Platt scaling: a one-feature logistic regression on the logit of the raw score.

    Used for deployment because it is smooth and never returns exactly 0 or 1. Isotonic regression
    is a step function, and its lowest step is 0.0 whenever no default falls in it, which would tell a
    brand-new borrower they carry no risk at all.
    """

    def fit(self, raw_prob, y_true) -> PlattCalibrator:
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_prob), np.asarray(y_true, dtype=int))
        return self

    def predict(self, raw_prob) -> np.ndarray:
        return self.lr_.predict_proba(_logit(raw_prob))[:, 1]


def make_calibrator(method: str):
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    raise ValueError(f"Unknown calibration method '{method}'; use 'platt' or 'isotonic'.")


def calibrate_oof(y_true, y_prob, folds, method: str = "platt") -> np.ndarray:
    """Calibrate out-of-fold predictions without leakage: fold k's calibrator never sees fold k."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=np.float64)
    f_arr = np.asarray(folds)
    if len(np.unique(f_arr)) < 2:
        raise ValueError("Out-of-fold calibration needs at least 2 folds (temporal has 1: calibrate on a slice of train).")
    out = np.zeros_like(yp)
    for f in np.unique(f_arr):
        tr, va = f_arr != f, f_arr == f
        cal = make_calibrator(method)
        cal.fit(yp[tr], yt[tr])
        out[va] = cal.predict(yp[va])
    return np.clip(out, 0.0, 1.0)
