"""Instance-based model: k-nearest neighbours (Task 2)."""
from __future__ import annotations

from typing import Any

from sklearn.neighbors import KNeighborsClassifier


def knn_classifier(params: dict[str, Any], pos_weight: float | None = None) -> KNeighborsClassifier:
    """Scores a loan by the default rate of its k most similar training loans.

    kNN has no class weights; with 1.8% defaults its probabilities are coarse (multiples of 1/k),
    so compare it on PR-AUC, not on F1 at a fixed threshold.
    """
    return KNeighborsClassifier(**params)
