"""Linear models: logistic regression (Task 2) and OLS / Ridge / Lasso (Task 1)."""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

from .. import config as C


def logistic_regression(params: dict[str, Any], pos_weight: float | None = None) -> LogisticRegression:
    """Log-odds model fitted by maximum likelihood; class_weight makes each default count ~54x."""
    return LogisticRegression(class_weight="balanced", random_state=C.SEED, **params)


def ols(params: dict[str, Any], pos_weight: float | None = None) -> LinearRegression:
    """Ordinary least squares (normal equations)."""
    return LinearRegression(**params)


def ridge(params: dict[str, Any], pos_weight: float | None = None) -> Ridge:
    """OLS plus an L2 penalty on the weights."""
    return Ridge(random_state=C.SEED, **params)


def lasso(params: dict[str, Any], pos_weight: float | None = None) -> Lasso:
    """OLS plus an L1 penalty, which can set weights exactly to zero."""
    return Lasso(random_state=C.SEED, **params)
