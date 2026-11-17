"""Train any model on any validation scheme and write predictions in one shared format.

Preprocessing is fitted inside each training fold, never on validation rows.
Every model returns the same columns (config.PRED_COLUMNS), so evaluation never needs to know
which library is inside.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config as C
from ..models import LIGHTNING_MODELS, build_estimator
from ..processing import TargetLogScaler, get_feature_lists, make_preprocessor
from .lightning_runner import fit_predict_lightning_clf, fit_predict_lightning_reg

_META = ["ID", "customer_id", "country_id", "lender_id", "loan_type", "is_new_borrower", "split"]


def task_target(df: pd.DataFrame, task: str) -> np.ndarray:
    """Task 2: 0/1 default label. Task 1: loan amount in USD (so Kenya and Ghana share a unit)."""
    if task == "task2":
        return df["target"].astype(int).to_numpy()
    if task == "task1":
        return np.expm1(df[C.TARGET_T1].to_numpy(dtype=np.float64))
    raise ValueError(f"Unknown task '{task}'.")


def iter_splits(train_df: pd.DataFrame, scheme: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(fold id, training mask, validation mask) for 'fold_group', 'fold_strat' or 'temporal'."""
    if scheme in ("fold_group", "fold_strat"):
        return [(k, (train_df[scheme] != k).to_numpy(), (train_df[scheme] == k).to_numpy()) for k in range(C.N_FOLDS)]
    if scheme == "temporal":
        return [(0, (train_df["temporal"] == "train").to_numpy(), (train_df["temporal"] == "valid").to_numpy())]
    raise ValueError(f"Unknown validation scheme '{scheme}'.")


def positive_proba(est: Any, X: np.ndarray) -> np.ndarray:
    """P(default) from a fitted classifier, also when a fold had only one class."""
    probs = est.predict_proba(X)
    if probs.shape[1] == 1:
        return np.full(len(X), float(est.classes_[0]))
    return probs[:, list(est.classes_).index(1)].astype(np.float64)


def fit_predict(
    model_name: str,
    task: str,
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    X_pred_df: pd.DataFrame,
    groups_tr: np.ndarray,
) -> np.ndarray:
    """Fit preprocessor + model on training rows; return predictions for X_pred_df.

    Task 2 returns raw (uncalibrated) default probabilities. Task 1 returns amounts in USD.
    """
    pre = make_preprocessor(task=task)
    X_tr, X_pred = pre.fit_transform(X_tr_df), pre.transform(X_pred_df)

    if task == "task2":
        if model_name in LIGHTNING_MODELS:
            return fit_predict_lightning_clf(model_name, X_tr, y_tr, X_pred, groups_train=groups_tr)
        pos_weight = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)
        est = build_estimator(model_name, task="task2", pos_weight=pos_weight).fit(X_tr, y_tr)
        return positive_proba(est, X_pred)

    scaler = TargetLogScaler().fit(y_tr)
    z_tr = scaler.transform(y_tr)
    if model_name in LIGHTNING_MODELS:
        z_pred = fit_predict_lightning_reg(X_tr, z_tr, X_pred, groups_train=groups_tr)
    else:
        z_pred = build_estimator(model_name, task="task1").fit(X_tr, z_tr).predict(X_pred)
    return scaler.inverse_transform(z_pred)


def _frame(meta: pd.DataFrame, scheme: str, fold: int, task: str, model_name: str, y_true, y_pred) -> pd.DataFrame:
    out = meta[_META].copy()
    out["scheme"], out["fold"], out["model"] = scheme, fold, f"{task}:{model_name}"
    out["y_true"] = np.asarray(y_true, dtype=np.float64)
    out["y_pred"] = np.asarray(y_pred, dtype=np.float64)
    return out[C.PRED_COLUMNS]


def run_cv(
    df: pd.DataFrame,
    model_name: str,
    task: str = "task2",
    scheme: str = "fold_group",
    save_preds: bool = True,
) -> pd.DataFrame:
    """Out-of-fold predictions for every labelled Train row under one validation scheme."""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    num_cols, cat_cols = get_feature_lists(task)
    X_all, y_all = train_df[num_cols + cat_cols], task_target(train_df, task)
    groups = train_df["customer_id"].to_numpy()

    frames = []
    for fold, tr, va in iter_splits(train_df, scheme):
        y_pred = fit_predict(model_name, task, X_all[tr], y_all[tr], X_all[va], groups[tr])
        frames.append(_frame(train_df[va], scheme, fold, task, model_name, y_all[va], y_pred))
    preds = pd.concat(frames, ignore_index=True)
    if save_preds:
        C.PREDS.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(C.PREDS / f"{task}_{model_name}_{scheme}.parquet", index=False)
    return preds


def predict_test_split(df: pd.DataFrame, model_name: str, task: str = "task2", save_preds: bool = True) -> pd.DataFrame:
    """Fit on all labelled Kenya rows and predict the Test rows (Kenya and Ghana, no labels)."""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    num_cols, cat_cols = get_feature_lists(task)
    y_pred = fit_predict(
        model_name,
        task,
        train_df[num_cols + cat_cols],
        task_target(train_df, task),
        test_df[num_cols + cat_cols],
        train_df["customer_id"].to_numpy(),
    )
    out = _frame(test_df, "test_inference", -1, task, model_name, np.full(len(test_df), np.nan), y_pred)
    if save_preds:
        C.PREDS.mkdir(parents=True, exist_ok=True)
        out.to_parquet(C.PREDS / f"{task}_{model_name}_test.parquet", index=False)
    return out
