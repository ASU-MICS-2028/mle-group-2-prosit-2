"""Single decision trees with cost-complexity pruning (Task 2 classifier, Task 1 regressor)."""
from __future__ import annotations

from typing import Any

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .. import config as C


def decision_tree_classifier(params: dict[str, Any], pos_weight: float | None = None) -> DecisionTreeClassifier:
    """Pruned by max_depth and ccp_alpha; class_weight balances the 1.8% defaults."""
    return DecisionTreeClassifier(class_weight="balanced", random_state=C.SEED, **params)


def decision_tree_regressor(params: dict[str, Any], pos_weight: float | None = None) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(random_state=C.SEED, **params)
