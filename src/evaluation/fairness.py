"""Disaggregated evaluation for the fairness audit (Part D): metrics and selection rates per group.

Always pass calibrated probabilities together with the cost threshold; raw class-weighted scores
sit far above the true default rate and make every group look heavily declined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import classification_metrics


def add_loan_size_band(df: pd.DataFrame, amount_col: str = "log_amount_usd") -> pd.Series:
    """Quartile bands of loan size, to check whether the tool is regressive (hurts small loans)."""
    labels = ["Q1_small", "Q2_medium", "Q3_large", "Q4_very_large"]
    try:
        return pd.qcut(df[amount_col].astype(float), q=4, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series("all", index=df.index, dtype=str)


def disaggregated_evaluation(
    df: pd.DataFrame,
    group_col: str,
    threshold: float,
    pred_col: str = "y_pred",
    target_col: str = "y_true",
) -> pd.DataFrame:
    """Selection rate, error rates and ranking metrics for each value of `group_col`.

    Works on labelled Kenya validation predictions and on unlabelled Test predictions
    (Ghana has no labels, so only selection rate and mean score are available there).
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")
    rows = []
    for grp, gdf in df.groupby(group_col, dropna=False):
        yp = gdf[pred_col].to_numpy(dtype=np.float64)
        row = {
            "group_col": group_col,
            "group": str(grp),
            "n": len(gdf),
            "selection_rate": float(np.mean(yp <= threshold)),
            "mean_pred_prob": float(np.mean(yp)),
        }
        labelled = gdf[target_col].notna() if target_col in gdf else pd.Series(False, index=gdf.index)
        if labelled.any():
            yt = gdf.loc[labelled, target_col].astype(int).to_numpy()
            m = classification_metrics(yt, gdf.loc[labelled, pred_col].to_numpy(), threshold=threshold)
            row.update(
                {
                    "actual_default_rate": float(yt.mean()),
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "f1": m["f1"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "fnr": m["fn"] / max(m["fn"] + m["tp"], 1),  # defaulters we approved
                    "fpr": m["fp"] / max(m["fp"] + m["tn"], 1),  # good borrowers we declined
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    best = out["selection_rate"].max() if len(out) else 1.0
    out["selection_rate_gap"] = best - out["selection_rate"]
    out["disparate_impact_ratio"] = out["selection_rate"] / max(best, 1e-9)
    return out


def decision_cost_by_group(
    df: pd.DataFrame,
    group_col: str,
    approve_col: str,
    c_fp: float,
    c_fn: float,
    target_col: str = "y_true",
) -> pd.DataFrame:
    """Money cost of a set of approve/decline decisions, per group (judges a mitigation by its price).

    Approving a loan that defaults costs `c_fn`; declining a borrower who would have repaid costs
    `c_fp` in forgone fees. Same cost matrix as the operating threshold (`cost_matrix_threshold`).
    """
    rows = []
    for grp, g in df.groupby(group_col, dropna=False):
        approve = g[approve_col].astype(bool)
        y = g[target_col].astype(int)
        fn = int((approve & (y == 1)).sum())
        fp = int((~approve & (y == 0)).sum())
        total = fn * c_fn + fp * c_fp
        rows.append({
            "group": str(grp),
            "n": len(g),
            "approval_rate": float(approve.mean()),
            "defaults_approved": fn,
            "good_borrowers_declined": fp,
            "cost_fn": fn * c_fn,
            "cost_fp": fp * c_fp,
            "total_cost": total,
            "cost_per_applicant": total / max(len(g), 1),
        })
    return pd.DataFrame(rows)


def proxy_strength(df: pd.DataFrame, group_col: str, features: list[str]) -> pd.DataFrame:
    """How well each feature on its own identifies a group (proxy check for #26).

    Numeric features are scored as they are; a categorical feature is replaced by the group's share
    within each category. Strength = |AUC - 0.5| * 2: 0 says nothing about the group, 1 identifies it.
    """
    from sklearn.metrics import roc_auc_score

    y = df[group_col].astype(int).to_numpy()
    rows = []
    for f in features:
        x = df[f]
        if not pd.api.types.is_numeric_dtype(x):
            x = df.groupby(f, dropna=False, observed=True)[group_col].transform("mean")
        x = x.astype(float)
        auc = float(roc_auc_score(y, x.fillna(x.median())))
        rows.append({"feature": f, "auc": auc, "proxy_strength": abs(auc - 0.5) * 2})
    return pd.DataFrame(rows).sort_values("proxy_strength", ascending=False).reset_index(drop=True)
