"""Score many applications at once from feature rows (e.g. the whole Test set) with the saved bundle.

Same decision rule as `score_application`, vectorised: used to show how the product behaves on
Kenya and Ghana applications (decision mix, limits, reasons, review flags) without 18,000 single calls.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config as C
from ..processing import get_feature_lists
from .reasons import explain
from .schema import exchange_rate
from .scorer import load_bundle


def score_frame(
    features: pd.DataFrame,
    bundle: dict[str, Any] | None = None,
    explain_sample: int = 0,
    seed: int = C.SEED,
) -> pd.DataFrame:
    """One row per application: probability, decision, local-currency limit, review flags, and
    (for `explain_sample` random rows) the top-3 reasons."""
    bundle = bundle or load_bundle()
    clf, reg = bundle["clf_bundle"], bundle["reg_bundle"]
    num2, cat2 = get_feature_lists("task2")
    num1, cat1 = get_feature_lists("task1")
    X2 = features[num2 + cat2].copy()
    for c in C.CAT_FEATURES:
        X2[c] = X2[c].astype(str)

    pipe = clf["pipeline"]
    raw = pipe.predict_proba(X2)[:, 1]
    prob = clf["calibrator"].predict(raw) if clf.get("calibrator") is not None else raw

    fx = features["macro_fx"].copy() if "macro_fx" in features else pd.Series(np.nan, index=features.index)
    missing = fx.isna()
    if missing.any():
        years = pd.to_datetime(features.loc[missing, "disbursement_date"]).dt.year
        fx.loc[missing] = [exchange_rate(c, int(y)) for c, y in zip(features.loc[missing, "country_id"], years)]
    limit_usd = reg["target_scaler"].inverse_transform(reg["pipeline"].predict(X2[num1 + cat1]))
    limit_local = limit_usd * fx.to_numpy()
    requested = np.expm1(features["log_amount_local"].to_numpy())

    threshold = float(clf["threshold"])
    decision = np.where(prob > threshold, "decline", np.where(requested > limit_local, "limit", "approve"))

    out = pd.DataFrame(
        {
            "ID": features["ID"].to_numpy(),
            "country_id": features["country_id"].to_numpy(),
            "is_new_borrower": features["is_new_borrower"].to_numpy(),
            "probability_of_default": prob,
            "decision": decision,
            "requested_local": requested,
            "limit_local": np.where(decision == "decline", 0.0, limit_local),
            "limit_usd": limit_usd,
        },
        index=features.index,
    )
    scope = clf.get("training_scope")
    if scope:
        amount_usd = np.expm1(features["log_amount_usd"].to_numpy())
        lo, hi = scope["amount_usd_range"]
        out["flag_country"] = ~features["country_id"].isin(scope["countries"]).to_numpy()
        out["flag_lender"] = ~X2["lender_id"].isin(scope["lenders"]).to_numpy()
        out["flag_product"] = ~X2["loan_type"].isin(scope["loan_types"]).to_numpy()
        out["flag_amount"] = (amount_usd < lo) | (amount_usd > hi)
        out["needs_review"] = out[["flag_country", "flag_lender", "flag_product", "flag_amount"]].any(axis=1)

    if explain_sample:
        idx = out.sample(min(explain_sample, len(out)), random_state=seed).index
        reasons = {i: explain(pipe, X2.loc[[i]], decision=out.at[i, "decision"]) for i in idx}
        out["top_3_reasons"] = pd.Series(reasons)
    return out
