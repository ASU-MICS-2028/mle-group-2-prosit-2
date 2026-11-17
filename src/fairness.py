"""Cross-market shift diagnostics and disaggregated fairness evaluation (Parts C and D)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .evaluate import classification_metrics
from .preprocess import get_feature_lists, make_preprocessor


def add_loan_size_band(
    df: pd.DataFrame,
    amount_col: str = "log_amount_usd",
) -> pd.Series:
    """Partition loans into quartile size bands for regressive-impact auditing."""
    col = amount_col if amount_col in df.columns else "log_amount_local"
    vals = df[col].astype(float)
    labels = ["Q1_Small", "Q2_Medium", "Q3_Large", "Q4_VeryLarge"]
    try:
        bands = pd.qcut(vals, q=4, labels=labels, duplicates="drop")
        return bands.astype(str)
    except ValueError:
        return pd.Series(["All"] * len(df), index=df.index, dtype=str)


def disaggregated_evaluation(
    df: pd.DataFrame,
    group_col: str,
    threshold: float = 0.145,
    pred_col: str = "y_pred",
    target_col: str = "y_true",
) -> pd.DataFrame:
    """Compute selection rate, error rates, and performance metrics per subgroup.

    Works on both labeled Kenya validation predictions (`y_true` present) and
    unlabeled Test predictions including Ghana (`y_true` all NaN).
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")

    rows: list[dict[str, float | int | str]] = []
    for grp, gdf in df.groupby(group_col, dropna=False):
        yp = gdf[pred_col].to_numpy(dtype=np.float64)
        sel_rate = float(np.mean(yp <= threshold))
        row: dict[str, float | int | str] = {
            "group_col": group_col,
            "group": str(grp),
            "n": int(len(gdf)),
            "selection_rate": sel_rate,
            "decline_rate": float(1.0 - sel_rate),
            "mean_pred_prob": float(np.mean(yp)),
        }

        labeled = gdf[target_col].notna() if target_col in gdf.columns else pd.Series(False, index=gdf.index)
        if labeled.any():
            yt = gdf.loc[labeled, target_col].astype(int).to_numpy()
            yp_lab = gdf.loc[labeled, pred_col].to_numpy(dtype=np.float64)
            m = classification_metrics(yt, yp_lab, threshold=threshold)
            fnr = m["fn"] / max(m["fn"] + m["tp"], 1)
            fpr = m["fp"] / max(m["fp"] + m["tn"], 1)
            row.update(
                {
                    "actual_default_rate": float(np.mean(yt)),
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "f1": m["f1"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "fnr": float(fnr),
                    "fpr": float(fpr),
                }
            )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    max_sel = out["selection_rate"].max() if not out.empty else 1.0
    out["selection_rate_gap"] = max_sel - out["selection_rate"]
    out["disparate_impact_ratio"] = out["selection_rate"] / max(max_sel, 1e-9)
    return out


def run_domain_classifier(
    df: pd.DataFrame,
    include_categoricals: bool = False,
    include_macro: bool = False,
) -> dict[str, object]:
    """Train a Kenya-vs-Ghana domain classifier to quantify covariate shift and importance weights.

    By default evaluates behavioral/loan-structure shift (excluding country-identifying
    macro rates and lender/loan_type IDs), and supports full-feature shift diagnostics
    via `include_categoricals=True, include_macro=True`.
    """
    if "country_id" not in df.columns or df["country_id"].nunique() < 2:
        raise ValueError("Dataframe must contain at least two countries in `country_id`.")

    y_dom = (df["country_id"] == "Ghana").astype(int).to_numpy()
    num_cols, cat_cols = get_feature_lists("task2")
    if not include_macro:
        num_cols = [c for c in num_cols if not c.startswith("macro_")]
    used_cats = cat_cols if include_categoricals else []

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = [
        (
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            num_cols,
        )
    ]
    if used_cats:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), used_cats)
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    X_df = df[num_cols + used_cats]

    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    oof_probs = np.zeros(len(df), dtype=np.float64)
    coefs: list[np.ndarray] = []
    feature_names: list[str] = []

    for tr, va in skf.split(X_df, y_dom):
        X_tr = pre.fit_transform(X_df.iloc[tr])
        X_va = pre.transform(X_df.iloc[va])
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=C.SEED)
        clf.fit(X_tr, y_dom[tr])
        oof_probs[va] = clf.predict_proba(X_va)[:, 1]
        coefs.append(clf.coef_.ravel())
        if not feature_names:
            feature_names = list(pre.get_feature_names_out())

    domain_auc = float(roc_auc_score(y_dom, oof_probs))
    clipped = np.clip(oof_probs, 0.01, 0.99)
    # Importance weights w(x) = P(Ghana|x) / P(Kenya|x) normalized on Train rows
    weights = np.clip(clipped / (1.0 - clipped), 0.05, 20.0)

    # Align coefficient lengths in case folds saw slightly different OHE categories
    min_len = min(len(c) for c in coefs)
    mean_coef = np.mean([c[:min_len] for c in coefs], axis=0)
    top_drivers = (
        pd.DataFrame({"feature": feature_names[:min_len], "coef": mean_coef, "abs_coef": np.abs(mean_coef)})
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "domain_roc_auc": domain_auc,
        "oof_ghana_prob": oof_probs,
        "importance_weights": weights,
        "top_shift_drivers": top_drivers,
        "preprocessor_reference": make_preprocessor("task2") if include_categoricals else pre,
    }
