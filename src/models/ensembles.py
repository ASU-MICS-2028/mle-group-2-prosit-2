"""Tree ensembles: Random Forest (bagging) and LightGBM (gradient boosting)."""
from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier

from .. import config as C


def random_forest_classifier(params: dict[str, Any], pos_weight: float | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(class_weight="balanced", random_state=C.SEED, n_jobs=-1, **params)


def lightgbm_classifier(params: dict[str, Any], pos_weight: float | None = None) -> LGBMClassifier:
    """scale_pos_weight = negatives / positives in the training fold (about 53.6 on all of Train)."""
    return LGBMClassifier(
        scale_pos_weight=pos_weight if pos_weight is not None else 1.0,
        random_state=C.SEED,
        verbose=-1,
        n_jobs=-1,
        **params,
    )


def lightgbm_regressor(params: dict[str, Any], pos_weight: float | None = None) -> LGBMRegressor:
    """LightGBM regressor for Task 1; supports quantile regression when `objective='quantile'` and `alpha` are set (D-30)."""
    return LGBMRegressor(random_state=C.SEED, verbose=-1, n_jobs=-1, **params)
