"""Plain-language reason codes: the top factors behind one decision, worded for the borrower.

Only factors a borrower can understand are ever shown. Calendar effects (day of week, month) and
internal loan-structure fields still count in the score, but are never given as a reason.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config as C

# feature: (wording when it raised the risk, wording when it lowered the risk)
REASON_TEMPLATES: dict[str, tuple[str, str]] = {
    "log_amount_usd": (
        "The amount requested is large compared with similar loans",
        "The amount requested is in a range that is usually repaid",
    ),
    "duration": (
        "The repayment period is longer than most loans we see repaid",
        "The short repayment period",
    ),
    "is_new_borrower": (
        "This is your first loan with us, so there is no repayment record yet",
        "You are a returning borrower",
    ),
    "n_past_loans": (
        "You have few earlier loans with us",
        "Your record of earlier loans with us",
    ),
    "days_since_last_loan": (
        "The time since your last loan",
        "The time since your last loan",
    ),
    "n_past_labelled": (
        "Few of your earlier loans have reached their due date yet",
        "Several of your earlier loans have reached their due date",
    ),
    "n_past_defaults": (
        "An earlier loan was not repaid",
        "No missed repayments on earlier loans",
    ),
    "past_default_rate": (
        "The share of earlier loans that were not repaid",
        "Your earlier loans were repaid",
    ),
    "days_since_last_default": (
        "A recent missed repayment",
        "No recent missed repayment",
    ),
    "loan_type": (
        "This loan product has a higher rate of missed repayments",
        "This loan product is usually repaid",
    ),
    "lender_id": (
        "Loans through this lender have a higher rate of missed repayments",
        "Loans through this lender are usually repaid",
    ),
}
NEVER_SHOWN = {"disb_month", "disb_dow", "disb_dom", "Lender_portion_Funded", "n_lenders_on_loan"}
FALLBACK_REASON = "Your overall application profile"

# A blank history field means "no such event yet", so credit its missing-indicator to the event itself:
# no previous loan -> first-time borrower; no previous default -> no missed repayments.
MISSING_MEANS = {
    "days_since_last_loan": "is_new_borrower",
    "days_since_last_default": "n_past_defaults",
    "past_default_rate": "n_past_labelled",
}


def _base_feature(raw_name: str, num_cols: list[str]) -> str:
    """Map a transformed column name (num__x, num__missingindicator_x, cat__lender_id_123) to its feature."""
    name = raw_name.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        base = name.removeprefix("missingindicator_")
        return MISSING_MEANS.get(base, base)
    for cat in C.CAT_FEATURES:
        if name.startswith(f"{cat}_"):
            return cat
    return name if name in num_cols else raw_name


def _is_true_of_borrower(feature: str, raises_risk: bool, raw: dict[str, Any] | None) -> bool:
    """Never state something that contradicts the borrower's own record.

    A linear model can give a 0 past-default rate a small positive push (correlated features share
    credit), which would otherwise produce "an earlier loan was not repaid" for someone who never missed one.
    """
    if raw is None or feature not in {"n_past_defaults", "past_default_rate", "days_since_last_default", "is_new_borrower"}:
        return True
    value = raw.get(feature)
    missing = value is None or (isinstance(value, float) and np.isnan(value))
    if feature == "days_since_last_default":
        has_bad_history = not missing
    elif feature == "is_new_borrower":
        has_bad_history = not missing and value == 1
    else:
        has_bad_history = not missing and value > 0
    return has_bad_history == raises_risk


def feature_contributions(pipe: Any, x_trans: np.ndarray, num_cols: list[str]) -> dict[str, float]:
    """Each feature's push on the log-odds for one row (positive = riskier).

    Linear models: weight x standardized value. Tree models: SHAP values.
    """
    est = pipe.named_steps["est"]
    row = np.asarray(x_trans, dtype=np.float64).ravel()
    if hasattr(est, "coef_"):
        contribs = np.asarray(est.coef_, dtype=np.float64).ravel() * row
    elif hasattr(est, "feature_importances_"):
        import shap

        sv = shap.TreeExplainer(est).shap_values(row.reshape(1, -1))
        contribs = np.asarray(sv[1] if isinstance(sv, list) else sv, dtype=np.float64).ravel()
    else:
        raise TypeError(f"No explanation method for {type(est).__name__}.")
    grouped: dict[str, float] = {}
    for raw_name, val in zip(pipe.named_steps["pre"].get_feature_names_out(), contribs):
        base = _base_feature(raw_name, num_cols)
        grouped[base] = grouped.get(base, 0.0) + float(val)
    return grouped


def top_reasons(
    contributions: dict[str, float], decision: str, top_k: int = 3, raw: dict[str, Any] | None = None
) -> list[str]:
    """Declines and limits: the factors that raised the risk most. Approvals: those that lowered it most."""
    shown = {f: v for f, v in contributions.items() if f in REASON_TEMPLATES and f not in NEVER_SHOWN}
    if decision == "approve":
        candidates = [(f, v) for f, v in sorted(shown.items(), key=lambda kv: kv[1]) if v < 0]
    else:
        candidates = [(f, v) for f, v in sorted(shown.items(), key=lambda kv: -kv[1]) if v > 0]
    reasons: list[str] = []
    for feat, val in candidates:
        if not _is_true_of_borrower(feat, raises_risk=val > 0, raw=raw):
            continue
        text = REASON_TEMPLATES[feat][0 if val > 0 else 1]
        if text not in reasons:
            reasons.append(text)
        if len(reasons) == top_k:
            break
    while len(reasons) < top_k:
        reasons.append(FALLBACK_REASON)
    return reasons


def explain(pipe: Any, X_row: pd.DataFrame, decision: str, top_k: int = 3, x_trans=None) -> list[str]:
    num_cols = [c for c in X_row.columns if c not in C.CAT_FEATURES]
    x = x_trans if x_trans is not None else pipe.named_steps["pre"].transform(X_row)
    return top_reasons(feature_contributions(pipe, x, num_cols), decision, top_k, raw=X_row.iloc[0].to_dict())
