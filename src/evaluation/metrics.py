"""Metrics every model is judged by: classification (Task 2) and regression (Task 1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from .. import config as C


def classification_metrics(y_true, y_prob, threshold: float) -> dict[str, float]:
    """Task 2 metrics at one decision threshold (p > threshold means decline)."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=np.float64)
    y_hat = (yp > threshold).astype(int)
    has_both = len(np.unique(yt)) >= 2
    tn, fp, fn, tp = confusion_matrix(yt, y_hat, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(yt, yp)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(yt, yp)) if has_both else float("nan"),
        "precision": float(precision_score(yt, y_hat, zero_division=0)),
        "recall": float(recall_score(yt, y_hat, zero_division=0)),
        "f1": float(f1_score(yt, y_hat, zero_division=0)),
        "selection_rate": float(np.mean(y_hat == 0)),  # share approved (the brief's "selection rate")
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sweep_thresholds(y_true, y_prob, c_fp: float, c_fn: float, thresholds=None) -> pd.DataFrame:
    """Precision, recall, F1, approval rate and expected cost per loan across thresholds."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=np.float64)
    grid = thresholds if thresholds is not None else np.round(np.linspace(0.005, 0.5, 100), 4)
    rows = []
    for t in grid:
        m = classification_metrics(yt, yp, threshold=float(t))
        rows.append({"threshold": float(t), **m, "expected_cost_per_loan": (m["fp"] * c_fp + m["fn"] * c_fn) / len(yt)})
    return pd.DataFrame(rows)


def reliability_table(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile") -> tuple[pd.DataFrame, float]:
    """Binned predicted vs actual default rate, and Expected Calibration Error (ECE).

    Quantile bins by default: with 1.8% defaults, equal-width bins put almost every loan in bin 0.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if strategy == "quantile":
        edges = np.unique(np.quantile(yp, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.searchsorted(edges, yp, side="right") - 1, 0, len(edges) - 2)
    rows, ece = [], 0.0
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        mean_pred, frac_pos = float(yp[mask].mean()), float(yt[mask].mean())
        ece += mask.mean() * abs(frac_pos - mean_pred)
        rows.append({"bin": b, "count": int(mask.sum()), "mean_predicted": mean_pred, "fraction_positive": frac_pos})
    return pd.DataFrame(rows), float(ece)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Task 1 RMSE and MAE on the log1p scale and on the original (USD) scale."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.maximum(0.0, np.asarray(y_pred, dtype=np.float64))
    yt_log, yp_log = np.log1p(yt), np.log1p(yp)
    return {
        "rmse_orig": float(root_mean_squared_error(yt, yp)),
        "mae_orig": float(mean_absolute_error(yt, yp)),
        "rmse_log": float(root_mean_squared_error(yt_log, yp_log)),
        "mae_log": float(mean_absolute_error(yt_log, yp_log)),
        "r2_log": float(r2_score(yt_log, yp_log)),
        "median_abs_error_orig": float(np.median(np.abs(yt - yp))),
    }


def summarize_predictions_dir(threshold: float, preds_dir: Path | None = None) -> pd.DataFrame:
    """One row per (task, model, scheme) prediction file in preds/, scored with the shared metrics.

    Task 2: ROC-AUC and PR-AUC use the raw scores; precision, recall, F1 and approval rate use
    out-of-fold Platt-calibrated probabilities, because the cost threshold only means something on
    calibrated probabilities. The temporal scheme has one fold, so it cannot be calibrated out of
    fold: its threshold metrics are left empty.
    """
    from .calibration import calibrate_oof

    rows = []
    for f in sorted((preds_dir or C.PREDS).glob("*.parquet")):
        df = pd.read_parquet(f)
        if df["y_true"].isna().all():
            continue  # Test-set predictions have no labels
        model_tag, scheme = str(df["model"].iloc[0]), str(df["scheme"].iloc[0])
        task, name = model_tag.split(":")
        row = {"task": task, "model": name, "scheme": scheme, "n": len(df)}
        if task == "task1":
            rows.append({**row, **regression_metrics(df["y_true"], df["y_pred"])})
            continue
        m = classification_metrics(df["y_true"], df["y_pred"], threshold=threshold)
        row.update(roc_auc=m["roc_auc"], pr_auc=m["pr_auc"])
        if df["fold"].nunique() >= 2 and df["y_pred"].nunique() > 1:
            cal = calibrate_oof(df["y_true"], df["y_pred"], df["fold"], method="platt")
            mc = classification_metrics(df["y_true"], cal, threshold=threshold)
            row.update({k: mc[k] for k in ("precision", "recall", "f1", "selection_rate", "tn", "fp", "fn", "tp")})
        rows.append(row)
    return pd.DataFrame(rows)
