"""Kenya -> Ghana covariate shift (Part C): a domain classifier and importance weights.

Idea: train a model to tell Kenyan from Ghanaian applications. AUC near 0.5 means the two look
alike; AUC near 1.0 means Ghana is a different population, and its predictions rest on extrapolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import config as C


def run_domain_classifier(
    df: pd.DataFrame,
    include_categoricals: bool = False,
    include_macro: bool = False,
    drop_cols: tuple[str, ...] = (),
) -> dict[str, object]:
    """Out-of-fold Kenya-vs-Ghana classifier on the model's numeric features.

    Defaults to borrower and loan behaviour only. `include_categoricals` adds lender and loan type;
    `include_macro` adds the country-level macro columns, which separate the countries trivially.
    Only one amount column (USD) is used: USD and local amounts together reveal the exchange rate,
    and therefore the country, which would make any shift look total (AUC 1.0).
    `drop_cols` removes features, e.g. label-derived history, which is structurally 0 in Ghana.
    `effective_sample_share` = ESS / n of the Kenyan importance weights: near 0 means reweighting
    Kenyan validation rests on a handful of rows and cannot stand in for Ghana.
    """
    if "country_id" not in df.columns or df["country_id"].nunique() < 2:
        raise ValueError("Need rows from at least two countries in `country_id`.")
    y_dom = (df["country_id"] == "Ghana").astype(int).to_numpy()
    num_cols = [c for c in list(C.NUM_FEATURES_T2) + (C.MACRO_FEATURES if include_macro else []) if c not in drop_cols]
    cat_cols = [c for c in C.CAT_FEATURES if c not in drop_cols] if include_categoricals else []

    steps = [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols)]
    if cat_cols:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    X_df = df[num_cols + cat_cols]

    oof = np.zeros(len(df))
    coefs, names = [], []
    for tr, va in StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED).split(X_df, y_dom):
        pre = ColumnTransformer(steps, remainder="drop")
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=C.SEED)
        clf.fit(pre.fit_transform(X_df.iloc[tr]), y_dom[tr])
        oof[va] = clf.predict_proba(pre.transform(X_df.iloc[va]))[:, 1]
        if not names:
            names = list(pre.get_feature_names_out())
        coefs.append(clf.coef_.ravel())

    # Importance weight w(x) = P(Ghana | x) / P(Kenya | x), clipped, rescaled to mean 1 on Kenya rows.
    p = np.clip(oof, 0.01, 0.99)
    w = np.clip(p / (1.0 - p), 0.05, 20.0)
    kenya = y_dom == 0
    w = w / w[kenya].mean()

    k = min(len(c) for c in coefs)
    mean_coef = np.mean([c[:k] for c in coefs], axis=0)
    drivers = (
        pd.DataFrame({"feature": names[:k], "coef": mean_coef, "abs_coef": np.abs(mean_coef)})
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "domain_roc_auc": float(roc_auc_score(y_dom, oof)),
        "oof_ghana_prob": oof,
        "importance_weights": w,
        "effective_sample_share": float(w[kenya].sum() ** 2 / (w[kenya] ** 2).sum() / kenya.sum()),
        "top_shift_drivers": drivers,
    }
