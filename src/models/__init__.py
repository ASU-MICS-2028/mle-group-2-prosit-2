"""Model registry: every model the team trains, built from one place with settings from configs/models.yaml.

    from src.models import build_estimator
    est = build_estimator("logreg", task="task2", pos_weight=53.6)

Lightning models (lit_logreg, lit_mlp, lit_linear) are trained by src.training.lightning_runner,
because they need the DataModule and Trainer rather than a scikit-learn `.fit()`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from .. import config as C
from . import baselines, ensembles, linear, neighbors, svm, trees

TASK2_MODELS = (
    "majority",
    "logreg",
    "knn",
    "svm_rbf",
    "decision_tree",
    "random_forest",
    "lightgbm",
    "lit_logreg",
    "lit_mlp",
)
TASK1_MODELS = ("mean", "ols", "ridge", "lasso", "decision_tree_reg", "lightgbm_reg", "lit_linear")
LIGHTNING_MODELS = ("lit_logreg", "lit_mlp", "lit_linear")

# name -> function(params, pos_weight) returning an unfitted estimator
_TASK2_BUILDERS = {
    "majority": baselines.majority_classifier,
    "logreg": linear.logistic_regression,
    "knn": neighbors.knn_classifier,
    "svm_rbf": svm.svm_rbf_classifier,
    "decision_tree": trees.decision_tree_classifier,
    "random_forest": ensembles.random_forest_classifier,
    "lightgbm": ensembles.lightgbm_classifier,
}
_TASK1_BUILDERS = {
    "mean": baselines.mean_regressor,
    "ols": linear.ols,
    "ridge": linear.ridge,
    "lasso": linear.lasso,
    "decision_tree_reg": trees.decision_tree_regressor,
    "lightgbm_reg": ensembles.lightgbm_regressor,
}


@lru_cache(maxsize=1)
def _load_yaml() -> dict[str, Any]:
    with open(C.MODELS_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def model_params(model_name: str, task: str = "task2") -> dict[str, Any]:
    """Hyperparameters for one model from configs/models.yaml (a copy, safe to modify)."""
    cfg = _load_yaml()
    if model_name == "lightning":
        return dict(cfg["lightning"])
    section = cfg.get(task) or {}
    if model_name not in section:
        raise ValueError(f"No settings for '{model_name}' under '{task}' in {C.MODELS_YAML.name}.")
    return dict(section[model_name] or {})


def build_estimator(model_name: str, task: str = "task2", pos_weight: float | None = None) -> Any:
    """Return an unfitted scikit-learn / LightGBM estimator for `task`.

    `pos_weight` = negatives / positives in the training fold; only LightGBM uses it directly,
    the scikit-learn classifiers use class_weight="balanced", which is the same idea.
    """
    builders = _TASK2_BUILDERS if task == "task2" else _TASK1_BUILDERS if task == "task1" else None
    if builders is None:
        raise ValueError(f"Unknown task '{task}'.")
    if model_name in LIGHTNING_MODELS:
        raise ValueError(f"'{model_name}' is a Lightning model; use src.training.lightning_runner.")
    if model_name not in builders:
        raise ValueError(f"Unknown {task} model '{model_name}'. Options: {sorted(builders)}")
    return builders[model_name](model_params(model_name, task), pos_weight)


__all__ = [
    "LIGHTNING_MODELS",
    "TASK1_MODELS",
    "TASK2_MODELS",
    "build_estimator",
    "model_params",
]
