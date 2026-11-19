"""Feature-group ablation study (#52): retrain Task 2 models with one group removed (or macro added back)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import config as C
from ..models import build_estimator
from ..training.cross_validation import iter_splits

# Exact partition of NUM_FEATURES_T2 + CAT_FEATURES into interpretable domain groups.
ABLATION_GROUPS: dict[str, list[str]] = {
    "borrower_history": [
        "is_new_borrower",
        "n_past_loans",
        "days_since_last_loan",
        "n_past_labelled",
        "n_past_defaults",
        "past_default_rate",
        "days_since_last_default",
    ],
    "loan_terms": [
        "duration",
        "Lender_portion_Funded",
        "n_lenders_on_loan",
    ],
    "calendar": [
        "disb_month",
        "disb_dow",
        "disb_dom",
    ],
    "lender_product": [
        "lender_id",
        "loan_type",
    ],
    "amount": [
        "log_amount_usd",
    ],
}


def make_subset_preprocessor(num_cols: Sequence[str], cat_cols: Sequence[str]) -> ColumnTransformer:
    """Fold-safe preprocessor matching `src.processing.make_preprocessor`, for any feature subset."""
    transformers = []
    if num_cols:
        num_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, list(num_cols)))
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols))
        )
    if not transformers:
        raise ValueError("Cannot build a preprocessor with zero numeric and zero categorical columns.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _ablation_variants(include_macro: bool = True) -> list[tuple[str, str, list[str], list[str]]]:
    """Return (variant_label, action, num_cols, cat_cols) tuples."""
    base_num = list(C.NUM_FEATURES_T2)
    base_cat = list(C.CAT_FEATURES)
    variants: list[tuple[str, str, list[str], list[str]]] = [
        ("full", "baseline", base_num, base_cat),
    ]
    for group_name, cols in ABLATION_GROUPS.items():
        drop_set = set(cols)
        num_sub = [c for c in base_num if c not in drop_set]
        cat_sub = [c for c in base_cat if c not in drop_set]
        variants.append((f"- {group_name}", "remove", num_sub, cat_sub))
    if include_macro:
        variants.append(("+ macro", "add", base_num + list(C.MACRO_FEATURES), base_cat))
    return variants


def _evaluate_variant(
    train_df: pd.DataFrame,
    model_name: str,
    scheme: str,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[float, float, int]:
    """Run fold-safe CV/temporal fit for one feature subset and return (roc_auc, pr_auc, n_val)."""
    feature_cols = num_cols + cat_cols
    y_all = train_df["target"].astype(int).to_numpy()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for _fold, tr_idx, va_idx in iter_splits(train_df, scheme):
        X_tr = train_df.iloc[tr_idx][feature_cols]
        y_tr = y_all[tr_idx]
        X_va = train_df.iloc[va_idx][feature_cols]
        y_va = y_all[va_idx]

        pre = make_subset_preprocessor(num_cols, cat_cols)
        X_tr_t = pre.fit_transform(X_tr)
        X_va_t = pre.transform(X_va)

        pos_weight = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)
        est = build_estimator(model_name, task="task2", pos_weight=pos_weight)
        est.fit(X_tr_t, y_tr)
        scores = est.predict_proba(X_va_t)[:, 1]

        y_true_parts.append(y_va)
        y_pred_parts.append(scores)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    roc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    pr = float(average_precision_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return roc, pr, len(y_true)


def run_ablation(
    df: pd.DataFrame,
    models: Sequence[str] = ("logreg", "lightgbm"),
    schemes: Sequence[str] = ("fold_strat", "fold_group", "temporal"),
) -> pd.DataFrame:
    """Retrain each model across schemes with each feature group removed (and macro added back)."""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    include_macro = all(c in train_df.columns for c in C.MACRO_FEATURES)
    variants = _ablation_variants(include_macro=include_macro)

    rows: list[dict[str, object]] = []
    for model_name in models:
        for scheme in schemes:
            base_roc: float | None = None
            base_pr: float | None = None
            for label, action, num_cols, cat_cols in variants:
                roc, pr, n_val = _evaluate_variant(train_df, model_name, scheme, num_cols, cat_cols)
                if label == "full":
                    base_roc, base_pr = roc, pr
                rows.append(
                    {
                        "model": model_name,
                        "scheme": scheme,
                        "variant": label,
                        "action": action,
                        "n_features": len(num_cols) + len(cat_cols),
                        "n_val": n_val,
                        "roc_auc": roc,
                        "pr_auc": pr,
                        "delta_roc_auc": roc - base_roc if base_roc is not None else 0.0,
                        "delta_pr_auc": pr - base_pr if base_pr is not None else 0.0,
                    }
                )
    return pd.DataFrame(rows)
