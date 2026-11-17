"""Unified training script for Task 1 (regression) and Task 2 (classification).

Every model is trained on the exact same shared feature file (`data/processed/features.parquet`)
and fold schemes, with preprocessing fitted strictly inside each training fold.

CLI usage:
    python -m src.train --task task2 --models majority logreg lightgbm --schemes fold_group
    python -m src.train --task task1 --models mean ridge lightgbm_reg --schemes fold_group --save-artifacts
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config as C
from .datamodule import fit_predict_lightning_clf, fit_predict_lightning_reg
from .evaluate import cost_matrix_threshold
from .preprocess import TargetLogScaler, get_feature_lists, load_features, make_preprocessor

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


def build_estimator(model_name: str, task: str = "task2", pos_weight: float = 53.6) -> Any:
    """Return an unfitted scikit-learn/LightGBM estimator for `task`."""
    if task == "task2":
        if model_name == "majority":
            return DummyClassifier(strategy="most_frequent")
        if model_name == "logreg":
            return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=C.SEED)
        if model_name == "knn":
            return KNeighborsClassifier(n_neighbors=25, weights="distance")
        if model_name == "svm_rbf":
            return SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=C.SEED)
        if model_name == "decision_tree":
            return DecisionTreeClassifier(
                max_depth=6, ccp_alpha=1e-4, class_weight="balanced", random_state=C.SEED
            )
        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=C.SEED,
                n_jobs=-1,
            )
        if model_name == "lightgbm":
            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=pos_weight,
                random_state=C.SEED,
                verbose=-1,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown Task 2 estimator '{model_name}'.")

    if task == "task1":
        if model_name == "mean":
            return DummyRegressor(strategy="mean")
        if model_name == "ols":
            return LinearRegression()
        if model_name == "ridge":
            return Ridge(alpha=10.0, random_state=C.SEED)
        if model_name == "lasso":
            return Lasso(alpha=0.01, max_iter=2000, random_state=C.SEED)
        if model_name == "decision_tree_reg":
            return DecisionTreeRegressor(max_depth=6, ccp_alpha=1e-4, random_state=C.SEED)
        if model_name == "lightgbm_reg":
            return LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=C.SEED,
                verbose=-1,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown Task 1 estimator '{model_name}'.")

    raise ValueError(f"Unknown task '{task}'.")


def _stratified_subsample(
    X: np.ndarray, y: np.ndarray, max_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = min(len(pos_idx), max(1, int(round(max_rows * len(pos_idx) / len(y)))))
    n_neg = max_rows - n_pos
    chosen = np.concatenate(
        [
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=min(len(neg_idx), n_neg), replace=False),
        ]
    )
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def _iter_splits(train_df: pd.DataFrame, scheme: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if scheme in ("fold_group", "fold_strat"):
        return [
            (k, (train_df[scheme] != k).to_numpy(), (train_df[scheme] == k).to_numpy())
            for k in range(C.N_FOLDS)
        ]
    if scheme == "temporal":
        return [
            (
                0,
                (train_df["temporal"] == "train").to_numpy(),
                (train_df["temporal"] == "valid").to_numpy(),
            )
        ]
    raise ValueError(f"Unknown validation scheme '{scheme}'.")


def _predict_positive_proba(est: Any, X: np.ndarray) -> np.ndarray:
    probs = est.predict_proba(X)
    if probs.shape[1] == 1:
        cls = int(est.classes_[0])
        return np.full(len(X), float(cls), dtype=np.float64)
    pos_col = list(est.classes_).index(1)
    return probs[:, pos_col].astype(np.float64)


def run_cv(
    df: pd.DataFrame,
    model_name: str,
    task: str = "task2",
    scheme: str = "fold_group",
    save_preds: bool = True,
    svm_max_rows: int = 12000,
) -> pd.DataFrame:
    """Train and generate out-of-fold predictions using the shared schema."""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    num_cols, cat_cols = get_feature_lists(task)
    X_all = train_df[num_cols + cat_cols]

    if task == "task2":
        y_all = train_df["target"].astype(int).to_numpy()
    elif task == "task1":
        y_all = np.expm1(train_df["log_amount_local"].to_numpy(dtype=np.float64))
    else:
        raise ValueError(f"Unknown task '{task}'.")

    fold_frames: list[pd.DataFrame] = []
    for fold_id, tr_mask, va_mask in _iter_splits(train_df, scheme):
        X_tr_df, X_va_df = X_all.iloc[tr_mask], X_all.iloc[va_mask]
        y_tr, y_va = y_all[tr_mask], y_all[va_mask]
        groups_tr = train_df.loc[tr_mask, "customer_id"].to_numpy()

        pre = make_preprocessor(task=task, sparse_output=False)
        X_tr = pre.fit_transform(X_tr_df)
        X_va = pre.transform(X_va_df)

        if task == "task2":
            if model_name in ("lit_logreg", "lit_mlp"):
                y_pred = fit_predict_lightning_clf(
                    model_name, X_tr, y_tr, X_va, groups_train=groups_tr, seed=C.SEED
                )
            else:
                n_pos = max(int(np.sum(y_tr == 1)), 1)
                pos_weight = float(np.sum(y_tr == 0)) / float(n_pos)
                est = build_estimator(model_name, task="task2", pos_weight=pos_weight)
                if model_name == "svm_rbf":
                    X_fit, y_fit = _stratified_subsample(X_tr, y_tr, svm_max_rows, C.SEED)
                    est.fit(X_fit, y_fit)
                else:
                    est.fit(X_tr, y_tr)
                y_pred = _predict_positive_proba(est, X_va)
        else:
            y_scaler = TargetLogScaler().fit(y_tr)
            z_tr = y_scaler.transform(y_tr)
            if model_name == "lit_linear":
                z_pred = fit_predict_lightning_reg(
                    X_tr, z_tr, X_va, groups_train=groups_tr, seed=C.SEED
                )
            else:
                est = build_estimator(model_name, task="task1")
                est.fit(X_tr, z_tr)
                z_pred = est.predict(X_va)
            y_pred = y_scaler.inverse_transform(z_pred)

        va_meta = train_df.loc[
            va_mask,
            ["ID", "customer_id", "country_id", "lender_id", "loan_type", "is_new_borrower", "split"],
        ].copy()
        va_meta["scheme"] = scheme
        va_meta["fold"] = fold_id
        va_meta["model"] = f"{task}:{model_name}"
        va_meta["y_true"] = y_va.astype(np.float64)
        va_meta["y_pred"] = np.asarray(y_pred, dtype=np.float64)
        fold_frames.append(va_meta[C.PRED_COLUMNS])

    preds_df = pd.concat(fold_frames, ignore_index=True)
    if save_preds:
        C.PREDS.mkdir(parents=True, exist_ok=True)
        out_path = C.PREDS / f"{task}_{model_name}_{scheme}.parquet"
        preds_df.to_parquet(out_path, index=False)
    return preds_df


def predict_test_split(
    df: pd.DataFrame,
    model_name: str,
    task: str = "task2",
    save_preds: bool = True,
) -> pd.DataFrame:
    """Fit on all labeled Kenya training rows and predict on Test (Kenya + Ghana) rows."""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    num_cols, cat_cols = get_feature_lists(task)

    pre = make_preprocessor(task=task, sparse_output=False)
    X_tr = pre.fit_transform(train_df[num_cols + cat_cols])
    X_te = pre.transform(test_df[num_cols + cat_cols])
    groups_tr = train_df["customer_id"].to_numpy()

    if task == "task2":
        y_tr = train_df["target"].astype(int).to_numpy()
        if model_name in ("lit_logreg", "lit_mlp"):
            y_pred = fit_predict_lightning_clf(
                model_name, X_tr, y_tr, X_te, groups_train=groups_tr, seed=C.SEED
            )
        else:
            pos_weight = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)
            est = build_estimator(model_name, task="task2", pos_weight=pos_weight)
            est.fit(X_tr, y_tr)
            y_pred = _predict_positive_proba(est, X_te)
    else:
        y_tr = np.expm1(train_df["log_amount_local"].to_numpy(dtype=np.float64))
        y_scaler = TargetLogScaler().fit(y_tr)
        z_tr = y_scaler.transform(y_tr)
        if model_name == "lit_linear":
            z_pred = fit_predict_lightning_reg(
                X_tr, z_tr, X_te, groups_train=groups_tr, seed=C.SEED
            )
        else:
            est = build_estimator(model_name, task="task1")
            est.fit(X_tr, z_tr)
            z_pred = est.predict(X_te)
        y_pred = y_scaler.inverse_transform(z_pred)

    out = test_df[
        ["ID", "customer_id", "country_id", "lender_id", "loan_type", "is_new_borrower", "split"]
    ].copy()
    out["scheme"] = "test_inference"
    out["fold"] = -1
    out["model"] = f"{task}:{model_name}"
    out["y_true"] = np.nan
    out["y_pred"] = np.asarray(y_pred, dtype=np.float64)
    out = out[C.PRED_COLUMNS]
    if save_preds:
        C.PREDS.mkdir(parents=True, exist_ok=True)
        out.to_parquet(C.PREDS / f"{task}_{model_name}_test.parquet", index=False)
    return out


def fit_and_save_deployment_artifacts(
    df: pd.DataFrame,
    clf_model: str = "logreg",
    reg_model: str = "ridge",
    threshold: float | None = None,
) -> dict[str, Path]:
    """Fit calibrated Task 2 and Task 1 sklearn pipelines on Train and save joblib bundles."""
    if clf_model in ("lit_logreg", "lit_mlp") or reg_model == "lit_linear":
        raise ValueError("Deployment joblib artifacts use sklearn/LightGBM Pipelines for low-latency scoring.")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    num_t2, cat_t2 = get_feature_lists("task2")
    num_t1, cat_t1 = get_feature_lists("task1")

    # 1. Out-of-fold raw probabilities for leak-free isotonic calibration
    oof_clf = run_cv(df, model_name=clf_model, task="task2", scheme="fold_group", save_preds=False)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(oof_clf["y_pred"].to_numpy(), oof_clf["y_true"].to_numpy())

    # 2. Fit final Task 2 pipeline on all Kenya training data
    y_t2 = train_df["target"].astype(int).to_numpy()
    pos_weight = float(np.sum(y_t2 == 0)) / max(float(np.sum(y_t2 == 1)), 1.0)
    clf_pipe = Pipeline(
        steps=[
            ("pre", make_preprocessor("task2", sparse_output=False)),
            ("est", build_estimator(clf_model, task="task2", pos_weight=pos_weight)),
        ]
    )
    clf_pipe.fit(train_df[num_t2 + cat_t2], y_t2)

    # 3. Fit final Task 1 pipeline + target scaler on all Kenya training data
    y_t1 = np.expm1(train_df["log_amount_local"].to_numpy(dtype=np.float64))
    y_scaler = TargetLogScaler().fit(y_t1)
    z_t1 = y_scaler.transform(y_t1)
    reg_pipe = Pipeline(
        steps=[
            ("pre", make_preprocessor("task1", sparse_output=False)),
            ("est", build_estimator(reg_model, task="task1")),
        ]
    )
    reg_pipe.fit(train_df[num_t1 + cat_t1], z_t1)

    op_threshold = threshold if threshold is not None else cost_matrix_threshold()["threshold"]
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    clf_path = C.ARTIFACTS / "task2_clf.joblib"
    reg_path = C.ARTIFACTS / "task1_reg.joblib"

    joblib.dump(
        {
            "pipeline": clf_pipe,
            "calibrator": calibrator,
            "model_name": clf_model,
            "threshold": float(op_threshold),
            "num_features": num_t2,
            "cat_features": cat_t2,
        },
        clf_path,
    )
    joblib.dump(
        {
            "pipeline": reg_pipe,
            "target_scaler": y_scaler,
            "model_name": reg_model,
            "num_features": num_t1,
            "cat_features": cat_t1,
        },
        reg_path,
    )
    return {"clf_path": clf_path, "reg_path": reg_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared Task 1 and Task 2 models.")
    parser.add_argument("--task", choices=["task1", "task2"], default="task2")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["fold_group"],
        choices=["fold_group", "fold_strat", "temporal"],
    )
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--clf-artifact-model", default="logreg")
    parser.add_argument("--reg-artifact-model", default="ridge")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_features()
    default_models = ["majority", "logreg"] if args.task == "task2" else ["mean", "ridge"]
    models = args.models or default_models

    for model_name in models:
        for scheme in args.schemes:
            preds = run_cv(df, model_name=model_name, task=args.task, scheme=scheme, save_preds=True)
            print(f"Saved preds for {args.task}:{model_name} ({scheme}) -> rows={len(preds)}")
        if args.predict_test:
            test_preds = predict_test_split(df, model_name=model_name, task=args.task, save_preds=True)
            print(f"Saved test preds for {args.task}:{model_name} -> rows={len(test_preds)}")

    if args.save_artifacts:
        paths = fit_and_save_deployment_artifacts(
            df,
            clf_model=args.clf_artifact_model,
            reg_model=args.reg_artifact_model,
        )
        print(f"Saved deployment artifacts: {paths['clf_path']} and {paths['reg_path']}")


if __name__ == "__main__":
    main()
