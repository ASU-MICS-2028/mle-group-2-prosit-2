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
