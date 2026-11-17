"""Baselines the brief requires us to always report: majority class (Task 2) and mean (Task 1)."""
from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor


def majority_classifier(params: dict[str, Any], pos_weight: float | None = None) -> DummyClassifier:
    """Predicts 'repays' for everyone: 98% accurate and useless, which is the point."""
    return DummyClassifier(strategy="most_frequent", **params)


def mean_regressor(params: dict[str, Any], pos_weight: float | None = None) -> DummyRegressor:
    """Predicts the average (log) loan amount for everyone."""
    return DummyRegressor(strategy="mean", **params)
