"""Every registered model builds from configs/models.yaml and trains through the shared CV loop."""
from __future__ import annotations

import numpy as np
import pytest

from src import config as C
from src.models import LIGHTNING_MODELS, TASK1_MODELS, TASK2_MODELS, build_estimator
from src.training import run_cv


@pytest.mark.parametrize("name", [m for m in TASK2_MODELS if m not in LIGHTNING_MODELS])
def test_task2_models_build(name: str) -> None:
    assert hasattr(build_estimator(name, "task2", pos_weight=10.0), "fit")


@pytest.mark.parametrize("name", [m for m in TASK1_MODELS if m not in LIGHTNING_MODELS])
def test_task1_models_build(name: str) -> None:
    assert hasattr(build_estimator(name, "task1"), "fit")


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_estimator("not_a_model", "task2")


@pytest.mark.parametrize("name", ["majority", "logreg", "decision_tree", "lightgbm"])
def test_run_cv_task2_shared_schema(synthetic_features, name: str) -> None:
    preds = run_cv(synthetic_features, name, task="task2", scheme="fold_group", save_preds=False)
    assert list(preds.columns) == C.PRED_COLUMNS
    assert len(preds) == len(synthetic_features)  # every row predicted exactly once
    assert preds["y_pred"].between(0, 1).all()


def test_run_cv_task1_predicts_usd_within_training_range(synthetic_features) -> None:
    preds = run_cv(synthetic_features, "ols", task="task1", scheme="fold_group", save_preds=False)
    amounts = np.expm1(synthetic_features["log_amount_usd"])
    assert preds["y_pred"].between(amounts.min() - 1e-6, amounts.max() + 1e-6).all()
