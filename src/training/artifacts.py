"""Fit the deployment models on all of Train and save them for the scoring service (Part E)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .. import config as C
from ..evaluation.calibration import PlattCalibrator
from ..evaluation.cost_matrix import cost_matrix_threshold
from ..models import LIGHTNING_MODELS, build_estimator
from ..processing import TargetLogScaler, get_feature_lists, make_preprocessor
from .cross_validation import run_cv, task_target

CLF_ARTIFACT = "task2_clf.joblib"
REG_ARTIFACT = "task1_reg.joblib"


def fit_and_save_deployment_artifacts(
    df: pd.DataFrame,
    clf_model: str = "logreg",
    reg_model: str = "lightgbm_reg",
    threshold: float | None = None,
) -> dict[str, Path]:
    """Save a calibrated Task 2 pipeline and a Task 1 pipeline (target in USD) as joblib bundles.

    Calibration: Platt scaling fitted on out-of-fold (GroupKFold) raw probabilities, so the
    calibrator never sees a score the model made on its own training rows.
    Threshold: from configs/cost_matrix.yaml unless given.
    """
    if clf_model in LIGHTNING_MODELS or reg_model in LIGHTNING_MODELS:
        raise ValueError("Deployment bundles use scikit-learn/LightGBM pipelines (fast, picklable).")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    num_t2, cat_t2 = get_feature_lists("task2")
    num_t1, cat_t1 = get_feature_lists("task1")

    oof = run_cv(df, model_name=clf_model, task="task2", scheme="fold_group", save_preds=False)
    calibrator = PlattCalibrator().fit(oof["y_pred"].to_numpy(), oof["y_true"].to_numpy())

    y_t2 = task_target(train_df, "task2")
    pos_weight = float(np.sum(y_t2 == 0)) / max(float(np.sum(y_t2 == 1)), 1.0)
    clf_pipe = Pipeline(
        [("pre", make_preprocessor("task2")), ("est", build_estimator(clf_model, "task2", pos_weight=pos_weight))]
    ).fit(train_df[num_t2 + cat_t2], y_t2)

    y_t1 = task_target(train_df, "task1")  # USD
    scaler = TargetLogScaler().fit(y_t1)
    reg_pipe = Pipeline([("pre", make_preprocessor("task1")), ("est", build_estimator(reg_model, "task1"))]).fit(
        train_df[num_t1 + cat_t1], scaler.transform(y_t1)
    )

    cost = cost_matrix_threshold()
    amount_usd = np.expm1(train_df["log_amount_usd"].to_numpy())
    training_scope = {
        "countries": sorted(train_df["country_id"].unique().tolist()),
        "lenders": sorted(train_df["lender_id"].astype(str).unique().tolist()),
        "loan_types": sorted(train_df["loan_type"].astype(str).unique().tolist()),
        "amount_usd_range": [float(np.quantile(amount_usd, 0.005)), float(np.quantile(amount_usd, 0.995))],
    }

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    clf_path, reg_path = C.ARTIFACTS / CLF_ARTIFACT, C.ARTIFACTS / REG_ARTIFACT
    joblib.dump(
        {
            "pipeline": clf_pipe,
            "calibrator": calibrator,
            "model_name": clf_model,
            "threshold": float(threshold if threshold is not None else cost["threshold"]),
            "cost_matrix": cost,
            "training_scope": training_scope,
        },
        clf_path,
    )
    joblib.dump({"pipeline": reg_pipe, "target_scaler": scaler, "model_name": reg_model, "target_unit": "USD"}, reg_path)
    return {"clf_path": clf_path, "reg_path": reg_path}
