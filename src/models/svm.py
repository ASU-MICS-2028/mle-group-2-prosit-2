"""Margin model: SVM with an RBF kernel (Task 2), trained on a stratified subsample."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from .. import config as C


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, max_rows: int, seed: int = C.SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample of at most `max_rows` rows that keeps the default rate."""
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = min(len(pos_idx), max(1, int(round(max_rows * len(pos_idx) / len(y)))))
    n_neg = min(len(neg_idx), max_rows - n_pos)
    chosen = np.concatenate(
        [rng.choice(pos_idx, size=n_pos, replace=False), rng.choice(neg_idx, size=n_neg, replace=False)]
    )
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


class SubsampledSVC(ClassifierMixin, BaseEstimator):
    """SVC(kernel='rbf') fitted on at most `max_train_rows` rows.

    Kernel SVMs cost O(n^2) memory and time, so 55,000 training rows are out of reach on a laptop.
    The subsample keeps the default rate; the model still predicts every validation row.
    """

    def __init__(self, C: float = 1.0, gamma: str | float = "scale", max_train_rows: int = 12000):
        self.C = C
        self.gamma = gamma
        self.max_train_rows = max_train_rows

    def fit(self, X: np.ndarray, y: np.ndarray) -> SubsampledSVC:
        X_fit, y_fit = stratified_subsample(np.asarray(X), np.asarray(y), self.max_train_rows)
        svc = SVC(kernel="rbf", C=self.C, gamma=self.gamma, class_weight="balanced", random_state=C.SEED)
        # SVMs output a margin, not a probability; Platt scaling (sigmoid) on internal folds maps it to one.
        self.svc_ = CalibratedClassifierCV(svc, method="sigmoid", cv=3, ensemble=False).fit(X_fit, y_fit)
        self.classes_ = self.svc_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.predict(X)


def svm_rbf_classifier(params: dict[str, Any], pos_weight: float | None = None) -> SubsampledSVC:
    return SubsampledSVC(**params)
