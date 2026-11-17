"""Shared evaluation, calibration, cost-matrix, bootstrap CI, and diagnostic utilities.

CLI usage:
    python -m src.evaluate --threshold 0.145
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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

from . import config as C


def cost_matrix_threshold(
    principal: float = C.DEFAULT_PRINCIPAL_GHS,
    fee_rate: float = C.DEFAULT_FEE_RATE,
    recovery_rate: float = C.DEFAULT_RECOVERY_RATE,
    collection_cost: float = C.DEFAULT_COLLECTION_COST_GHS,
) -> dict[str, float]:
    """Compute money-denominated FP/FN costs and optimal calibrated threshold.

    Declining is cheaper than approving whenever p * C_FN > (1 - p) * C_FP,
    yielding p* = C_FP / (C_FP + C_FN).
    """
    c_fp = float(principal * fee_rate)
    c_fn = float(principal * (1.0 - recovery_rate) + collection_cost)
    threshold = c_fp / (c_fp + c_fn) if (c_fp + c_fn) > 0 else 0.5
    return {
        "principal": float(principal),
        "fee_rate": float(fee_rate),
        "recovery_rate": float(recovery_rate),
        "collection_cost": float(collection_cost),
        "c_fp": c_fp,
        "c_fn": c_fn,
        "cost_ratio": c_fn / max(c_fp, 1e-9),
        "threshold": float(threshold),
    }


def cost_sensitivity_table(
    principal: float = C.DEFAULT_PRINCIPAL_GHS,
    fee_rate: float = C.DEFAULT_FEE_RATE,
    collection_cost: float = C.DEFAULT_COLLECTION_COST_GHS,
    recovery_rates: tuple[float, ...] = (0.0, 0.20, 0.40),
) -> pd.DataFrame:
    """Return operating-point sensitivity across alternative collection recovery assumptions."""
    rows = [
        cost_matrix_threshold(
            principal=principal,
            fee_rate=fee_rate,
            recovery_rate=r,
            collection_cost=collection_cost,
        )
        for r in recovery_rates
    ]
    return pd.DataFrame(rows)


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.145,
) -> dict[str, float]:
    """Compute Task 2 classification metrics at a decision threshold (y_prob > threshold => decline)."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=np.float64)
    y_hat = (yp > threshold).astype(int)

    has_both = len(np.unique(yt)) >= 2
    roc_auc = float(roc_auc_score(yt, yp)) if has_both else float("nan")
    pr_auc = float(average_precision_score(yt, yp)) if has_both else float("nan")
    tn, fp, fn, tp = confusion_matrix(yt, y_hat, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision_score(yt, y_hat, zero_division=0)),
        "recall": float(recall_score(yt, y_hat, zero_division=0)),
        "f1": float(f1_score(yt, y_hat, zero_division=0)),
        "selection_rate": float(np.mean(yp <= threshold)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sweep_thresholds(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    thresholds: np.ndarray | None = None,
    c_fp: float = 45.0,
    c_fn: float = 265.0,
) -> pd.DataFrame:
    """Evaluate precision, recall, F1, approval rate, and expected cost across thresholds."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=np.float64)
    grid = thresholds if thresholds is not None else np.round(np.linspace(0.02, 0.90, 45), 4)
    rows = []
    n = max(len(yt), 1)
    for t in grid:
        m = classification_metrics(yt, yp, threshold=float(t))
        expected_cost = (m["fp"] * c_fp + m["fn"] * c_fn) / n
        rows.append({"threshold": float(t), **m, "expected_cost_per_loan": float(expected_cost)})
    return pd.DataFrame(rows)


def calibrate_oof(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    folds: np.ndarray | pd.Series,
    method: str = "isotonic",
) -> np.ndarray:
    """Apply out-of-fold isotonic or Platt scaling calibration without target leakage."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    f_arr = np.asarray(folds)
    unique_folds = np.unique(f_arr)
    if len(unique_folds) < 2:
        raise ValueError("Out-of-fold calibration requires at least 2 distinct folds.")

    cal_probs = np.zeros_like(yp, dtype=np.float64)
    for f in unique_folds:
        tr = f_arr != f
        va = f_arr == f
        if method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(yp[tr], yt[tr])
            cal_probs[va] = iso.predict(yp[va])
        elif method == "platt":
            logits_tr = np.log(yp[tr] / (1.0 - yp[tr])).reshape(-1, 1)
            logits_va = np.log(yp[va] / (1.0 - yp[va])).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(logits_tr, yt[tr])
            cal_probs[va] = lr.predict_proba(logits_va)[:, 1]
        else:
            raise ValueError(f"Unknown calibration method '{method}'; use 'isotonic' or 'platt'.")
    return np.clip(cal_probs, 0.0, 1.0)


def reliability_table(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """Return binned reliability table and Expected Calibration Error (ECE)."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(yp, edges[1:-1], right=True), n_bins - 1)

    rows = []
    ece = 0.0
    n = max(len(yt), 1)
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        mean_pred = float(np.mean(yp[mask]))
        frac_pos = float(np.mean(yt[mask]))
        ece += (cnt / n) * abs(frac_pos - mean_pred)
        rows.append(
            {
                "bin": b,
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "count": cnt,
                "mean_predicted": mean_pred,
                "fraction_positive": frac_pos,
            }
        )
    return pd.DataFrame(rows), float(ece)


def bootstrap_ci(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.145,
    n_boot: int = 500,
    seed: int = C.SEED,
    baseline_pr_auc: float = 0.0183,
    noise_floor: float = 0.0263,
) -> dict[str, float | bool]:
    """Compute 95% bootstrap confidence intervals for PR-AUC and F1 and check noise floor."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(yt)

    pr_scores = np.empty(n_boot, dtype=np.float64)
    f1_scores = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt_b, yp_b = yt[idx], yp[idx]
        if len(np.unique(yt_b)) < 2:
            pr_scores[i] = np.nan
            f1_scores[i] = 0.0
        else:
            pr_scores[i] = average_precision_score(yt_b, yp_b)
            f1_scores[i] = f1_score(yt_b, (yp_b > threshold).astype(int), zero_division=0)

    pr_low, pr_high = np.nanpercentile(pr_scores, [2.5, 97.5])
    f1_low, f1_high = np.nanpercentile(f1_scores, [2.5, 97.5])
    pr_point = float(average_precision_score(yt, yp)) if len(np.unique(yt)) >= 2 else float("nan")
    f1_point = float(f1_score(yt, (yp > threshold).astype(int), zero_division=0))

    return {
        "pr_auc": pr_point,
        "pr_auc_ci_low": float(pr_low),
        "pr_auc_ci_high": float(pr_high),
        "f1": f1_point,
        "f1_ci_low": float(f1_low),
        "f1_ci_high": float(f1_high),
        "beats_noise_floor": bool(pr_low > (baseline_pr_auc + noise_floor)),
    }


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute Task 1 RMSE and MAE on both log1p and original currency scales."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.maximum(0.0, np.asarray(y_pred, dtype=np.float64))
    yt_log = np.log1p(yt)
    yp_log = np.log1p(yp)
    return {
        "rmse_orig": float(root_mean_squared_error(yt, yp)),
        "mae_orig": float(mean_absolute_error(yt, yp)),
        "rmse_log": float(root_mean_squared_error(yt_log, yp_log)),
        "mae_log": float(mean_absolute_error(yt_log, yp_log)),
        "r2_log": float(r2_score(yt_log, yp_log)),
        "mean_residual_orig": float(np.mean(yt - yp)),
    }


def summarize_predictions_dir(
    preds_dir: Path | None = None,
    threshold: float = 0.145,
) -> pd.DataFrame:
    """Aggregate all saved parquet prediction files in `preds/` into one comparison table."""
    target_dir = preds_dir or C.PREDS
    files = sorted(target_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    rows = []
    for f in files:
        df = pd.read_parquet(f)
        if df["y_true"].isna().all():
            continue
        model_tag = str(df["model"].iloc[0])
        scheme = str(df["scheme"].iloc[0])
        if model_tag.startswith("task2:"):
            m = classification_metrics(df["y_true"], df["y_pred"], threshold=threshold)
            rows.append({"task": "task2", "model": model_tag, "scheme": scheme, "n": len(df), **m})
        elif model_tag.startswith("task1:"):
            m = regression_metrics(df["y_true"], df["y_pred"])
            rows.append({"task": "task1", "model": model_tag, "scheme": scheme, "n": len(df), **m})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all saved predictions in preds/.")
    parser.add_argument("--threshold", type=float, default=cost_matrix_threshold()["threshold"])
    args = parser.parse_args()
    summary = summarize_predictions_dir(threshold=args.threshold)
    if summary.empty:
        print("No labeled prediction files found in preds/. Run `python -m src.train` first.")
        return
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
