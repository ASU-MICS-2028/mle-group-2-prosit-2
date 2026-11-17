"""Part E deployable credit scoring service with plain-language reason codes and latency benchmark.

CLI usage:
    python -m src.score
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .evaluate import cost_matrix_threshold
from .preprocess import get_feature_lists

REQUIRED_APPLICATION_FIELDS: dict[str, tuple[str, str, bool]] = {
    "customer_id": ("str", "Unique borrower identifier", True),
    "country_id": ("str", "Operating country ('Kenya' or 'Ghana')", True),
    "lender_id": ("str", "Funding lender identifier", True),
    "loan_type": ("str", "Loan product type (e.g. 'Type_1')", True),
    "Total_Amount": ("float", "Requested loan principal in local currency (> 0)", True),
    "duration": ("float", "Requested loan duration in days (> 0)", True),
    "disbursement_date": ("str", "Application/disbursement ISO date (YYYY-MM-DD)", True),
    "New_versus_Repeat": ("str", "'New Loan' or 'Repeat Loan'", True),
    "Lender_portion_Funded": ("float", "Share funded by lender in [0, 1] (default 0.30)", False),
    "n_lenders_on_loan": ("int", "Number of funding lenders on loan (default 1)", False),
    "n_past_loans": ("int", "Count of prior loans before this application (default 0)", False),
    "days_since_last_loan": ("float", "Days since most recent prior loan (optional)", False),
    "n_past_labelled": ("int", "Prior loans whose due date has passed (default 0)", False),
    "n_past_defaults": ("int", "Prior defaulted loans with passed due date (default 0)", False),
    "past_default_rate": ("float", "Historical default ratio among matured loans (optional)", False),
    "days_since_last_default": ("float", "Days since last recorded default (optional)", False),
}

# Fallback macro figures (prior-year official values) when MACRO_CSV is unavailable at runtime.
_DEFAULT_MACRO: dict[str, dict[str, float]] = {
    "Kenya": {
        "macro_inflation": 7.66,
        "macro_fx": 117.87,
        "macro_deposit_rate": 7.14,
        "macro_unemployment": 5.50,
    },
    "Ghana": {
        "macro_inflation": 31.26,
        "macro_fx": 8.27,
        "macro_deposit_rate": 11.80,
        "macro_unemployment": 3.87,
    },
}

_REASON_TEMPLATES: dict[str, tuple[str, str]] = {
    "log_amount_usd": (
        "Requested loan amount (USD equivalent) increases repayment burden",
        "Requested loan amount (USD equivalent) is within manageable range",
    ),
    "log_amount_local": (
        "Requested loan principal in local currency increases risk",
        "Requested loan principal in local currency supports approval",
    ),
    "duration": (
        "Longer loan repayment period increases uncertainty",
        "Short loan repayment period lowers default risk",
    ),
    "Lender_portion_Funded": (
        "Lender funding share structure raises exposure",
        "Lender funding share aligns with lower-risk loans",
    ),
    "is_new_borrower": (
        "First-time borrower with no established repayment track record",
        "Established repeat borrower status lowers risk",
    ),
    "n_lenders_on_loan": (
        "Multiple-lender loan structure is associated with higher default risk",
        "Single-lender loan structure lowers risk",
    ),
    "n_past_loans": (
        "Limited prior borrowing history on NkwaPay",
        "Consistent history of completed prior loans",
    ),
    "days_since_last_loan": (
        "Timing since previous loan increases risk",
        "Healthy spacing since previous loan",
    ),
    "n_past_labelled": (
        "Few prior loans have reached maturity for repayment verification",
        "Strong record of matured prior loans",
    ),
    "n_past_defaults": (
        "Previous missed payment(s) on record",
        "Zero prior defaults on mature loans",
    ),
    "past_default_rate": (
        "High historical default rate on prior loans",
        "Clean historical repayment ratio",
    ),
    "days_since_last_default": (
        "Recent prior default history",
        "No recent default history",
    ),
    "disb_month": (
        "Seasonal timing of application adds risk",
        "Seasonal timing of application is favorable",
    ),
    "disb_dow": (
        "Application day-of-week pattern adds risk",
        "Application timing pattern is favorable",
    ),
    "disb_dom": (
        "Application day-of-month cash-flow timing adds risk",
        "Application day-of-month cash-flow timing is favorable",
    ),
    "macro_inflation": (
        "Elevated local inflation increases household repayment pressure",
        "Macroeconomic inflation environment is relatively stable",
    ),
    "macro_fx": (
        "Local currency exchange-rate conditions add risk",
        "Exchange-rate environment supports repayment stability",
    ),
    "macro_deposit_rate": (
        "Prevailing interest-rate environment increases risk",
        "Prevailing interest-rate environment supports repayment",
    ),
    "macro_unemployment": (
        "Local unemployment conditions raise repayment risk",
        "Local employment conditions support repayment",
    ),
    "lender_id": (
        "Partner lender channel carries higher historical risk",
        "Partner lender channel shows strong repayment performance",
    ),
    "loan_type": (
        "Selected loan product tier has elevated historical default risk",
        "Selected loan product tier has strong repayment history",
    ),
}

_ARTIFACT_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_FEATURE_NAME_CACHE: dict[int, list[str]] = {}


@lru_cache(maxsize=1)
def _load_macro_lookup() -> dict[tuple[str, int], dict[str, float]]:
    if not C.MACRO_CSV.exists():
        return {}
    m = pd.read_csv(C.MACRO_CSV)
    m = m[m["Indicator"].isin(C.MACRO_INDICATORS)]
    long = m.melt(id_vars=["Country", "Indicator"], var_name="year", value_name="v")
    long["year"] = long["year"].astype(str).str.replace("YR", "").astype(int)
    long["Indicator"] = long["Indicator"].map(C.MACRO_INDICATORS)
    wide = long.pivot_table(index=["Country", "year"], columns="Indicator", values="v").reset_index()
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    for _, row in wide.iterrows():
        key = (str(row["Country"]), int(row["year"]))
        lookup[key] = {
            col: float(row[col]) for col in C.MACRO_INDICATORS.values() if pd.notna(row.get(col))
        }
    return lookup


def _resolve_macro(country: str, disb_year: int, app: dict[str, Any]) -> dict[str, float]:
    defaults = _DEFAULT_MACRO.get(country, _DEFAULT_MACRO["Kenya"])
    lookup = _load_macro_lookup()
    # Match training logic: join previous year's macro indicators, falling back to latest known year.
    matched: dict[str, float] = {}
    for yr in (disb_year - 1, disb_year - 2, 2023, 2022, 2021):
        if (country, yr) in lookup:
            matched = lookup[(country, yr)]
            break
    out: dict[str, float] = {}
    for col in C.MACRO_INDICATORS.values():
        if col in app and app[col] is not None:
            out[col] = float(app[col])
        elif col in matched:
            out[col] = float(matched[col])
        else:
            out[col] = float(defaults[col])
    return out


def prepare_single_application(app: dict[str, Any]) -> pd.DataFrame:
    """Validate and convert one raw application payload into the shared feature DataFrame."""
    required = [k for k, (_, _, is_req) in REQUIRED_APPLICATION_FIELDS.items() if is_req]
    missing = [k for k in required if k not in app or app[k] is None or str(app[k]).strip() == ""]
    if missing:
        raise ValueError(f"Missing required application fields: {missing}")

    amount = float(app["Total_Amount"])
    duration = float(app["duration"])
    if amount <= 0.0:
        raise ValueError("Total_Amount must be strictly positive.")
    if duration <= 0.0:
        raise ValueError("duration must be strictly positive.")

    disb_dt = pd.Timestamp(app["disbursement_date"])
    country = str(app["country_id"])
    macro = _resolve_macro(country, int(disb_dt.year), app)

    n_past_labelled = int(app.get("n_past_labelled", 0))
    n_past_defaults = int(app.get("n_past_defaults", 0))
    past_default_rate = app.get("past_default_rate")
    if past_default_rate is None:
        past_default_rate = (
            float(n_past_defaults) / float(n_past_labelled) if n_past_labelled > 0 else np.nan
        )

    row: dict[str, Any] = {
        "lender_id": str(app["lender_id"]),
        "loan_type": str(app["loan_type"]),
        "log_amount_usd": float(np.log1p(amount / max(macro["macro_fx"], 1e-6))),
        "log_amount_local": float(np.log1p(amount)),
        "duration": duration,
        "Lender_portion_Funded": float(app.get("Lender_portion_Funded", 0.30)),
        "is_new_borrower": int(str(app["New_versus_Repeat"]) == "New Loan"),
        "n_lenders_on_loan": int(app.get("n_lenders_on_loan", 1)),
        "n_past_loans": int(app.get("n_past_loans", 0)),
        "days_since_last_loan": (
            float(app["days_since_last_loan"])
            if app.get("days_since_last_loan") is not None
            else np.nan
        ),
        "n_past_labelled": n_past_labelled,
        "n_past_defaults": n_past_defaults,
        "past_default_rate": float(past_default_rate) if pd.notna(past_default_rate) else np.nan,
        "days_since_last_default": (
            float(app["days_since_last_default"])
            if app.get("days_since_last_default") is not None
            else np.nan
        ),
        "disb_month": int(disb_dt.month),
        "disb_dow": int(disb_dt.day_of_week),
        "disb_dom": int(disb_dt.day),
        **macro,
    }
    num_t2, cat_t2 = get_feature_lists("task2")
    return pd.DataFrame([row])[num_t2 + cat_t2]


def _base_feature_name(raw_name: str, num_cols: list[str]) -> str:
    if raw_name.startswith("num__missingindicator_"):
        suffix = raw_name.removeprefix("num__missingindicator_")
        if suffix in num_cols:
            return suffix
        if suffix.isdigit():
            idx = int(suffix)
            if 0 <= idx < len(num_cols):
                return num_cols[idx]
    if raw_name.startswith("num__"):
        return raw_name.removeprefix("num__")
    if raw_name.startswith("cat__"):
        rest = raw_name.removeprefix("cat__")
        for c in C.CAT_FEATURES:
            if rest.startswith(f"{c}_"):
                return c
    return raw_name


def explain_prediction(
    clf_or_bundle: Any,
    X_row: pd.DataFrame,
    top_k: int = 3,
    decision: str = "decline",
    x_trans: np.ndarray | None = None,
) -> list[str]:
    """Return top `k` plain-language borrower reasons explaining the risk score."""
    pipe = clf_or_bundle["pipeline"] if isinstance(clf_or_bundle, dict) else clf_or_bundle
    pre = pipe.named_steps["pre"]
    est = pipe.named_steps.get("est") or pipe.named_steps.get("lr")

    arr_trans = (
        np.asarray(x_trans, dtype=np.float64).ravel()
        if x_trans is not None
        else np.asarray(pre.transform(X_row), dtype=np.float64).ravel()
    )
    cache_key = id(pre)
    if cache_key not in _FEATURE_NAME_CACHE:
        _FEATURE_NAME_CACHE[cache_key] = list(pre.get_feature_names_out())
    raw_names = _FEATURE_NAME_CACHE[cache_key]
    num_cols, _ = get_feature_lists("task2")

    if est is not None and hasattr(est, "coef_"):
        weights = np.asarray(est.coef_, dtype=np.float64).ravel()
        contribs = weights * arr_trans
    elif est is not None and hasattr(est, "feature_importances_"):
        try:
            import shap

            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(arr_trans.reshape(1, -1))
            arr = sv[1] if isinstance(sv, list) else sv
            contribs = np.asarray(arr, dtype=np.float64).ravel()
        except Exception:
            contribs = np.asarray(est.feature_importances_, dtype=np.float64).ravel() * np.abs(arr_trans)
    else:
        contribs = np.abs(arr_trans)

    grouped: dict[str, float] = {}
    for raw_name, val in zip(raw_names, contribs):
        base = _base_feature_name(raw_name, num_cols)
        grouped[base] = grouped.get(base, 0.0) + float(val)

    # For declines/limits prioritize largest risk-increasing factors; for approvals prioritize protective ones.
    if decision in ("decline", "limit"):
        ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
    else:
        ranked = sorted(grouped.items(), key=lambda kv: kv[1])

    reasons: list[str] = []
    for feat, score in ranked:
        risk_msg, safe_msg = _REASON_TEMPLATES.get(
            feat,
            (
                f"Application factor ({feat.replace('_', ' ')}) increases risk",
                f"Application factor ({feat.replace('_', ' ')}) supports approval",
            ),
        )
        text = risk_msg if score >= 0 else safe_msg
        if text not in reasons:
            reasons.append(text)
        if len(reasons) == top_k:
            break

    while len(reasons) < top_k:
        reasons.append("Overall credit profile evaluated under NkwaPay lending guidelines")
    return reasons[:top_k]


def _load_bundle(clf_path: Path | None = None, reg_path: Path | None = None) -> dict[str, Any]:
    cp = str(clf_path or (C.ARTIFACTS / "task2_clf.joblib"))
    rp = str(reg_path or (C.ARTIFACTS / "task1_reg.joblib"))
    key = (cp, rp)
    if key not in _ARTIFACT_CACHE:
        if not Path(cp).exists() or not Path(rp).exists():
            raise FileNotFoundError(
                f"Missing saved artifacts ({cp}, {rp}). Run `python -m src.train --save-artifacts` first."
            )
        _ARTIFACT_CACHE[key] = {
            "clf_bundle": joblib.load(cp),
            "reg_bundle": joblib.load(rp),
        }
    return _ARTIFACT_CACHE[key]


def score_application(
    app: dict[str, Any],
    clf_path: Path | None = None,
    reg_path: Path | None = None,
    threshold: float | None = None,
    _bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one loan application and return default probability, decision, limit, and top 3 reasons."""
    t0 = time.perf_counter()
    bundle = _bundle if _bundle is not None else _load_bundle(clf_path, reg_path)
    clf_bundle = bundle["clf_bundle"]
    reg_bundle = bundle["reg_bundle"]

    X_t2 = prepare_single_application(app)
    num_t1, cat_t1 = get_feature_lists("task1")
    X_t1 = X_t2[num_t1 + cat_t1]

    clf_pipe = clf_bundle["pipeline"]
    pre_t2 = clf_pipe.named_steps["pre"]
    est_t2 = clf_pipe.named_steps.get("est") or clf_pipe.named_steps.get("lr")
    x_t2_trans = pre_t2.transform(X_t2)
    raw_prob = (
        float(est_t2.predict_proba(x_t2_trans)[0, 1])
        if est_t2 is not None
        else float(clf_pipe.predict_proba(X_t2)[0, 1])
    )
    calibrator = clf_bundle.get("calibrator")
    prob = float(calibrator.predict([raw_prob])[0]) if calibrator is not None else raw_prob
    prob = float(np.clip(prob, 0.0, 1.0))

    z_limit = float(reg_bundle["pipeline"].predict(X_t1)[0])
    recommended_limit = float(reg_bundle["target_scaler"].inverse_transform([z_limit])[0])
    recommended_limit = round(max(0.0, recommended_limit), 2)

    active_threshold = (
        float(threshold)
        if threshold is not None
        else float(clf_bundle.get("threshold", cost_matrix_threshold()["threshold"]))
    )
    requested_amount = float(app["Total_Amount"])

    if prob > active_threshold:
        decision = "decline"
    elif requested_amount > recommended_limit:
        decision = "limit"
    else:
        decision = "approve"

    reasons = explain_prediction(
        clf_bundle, X_t2, top_k=3, decision=decision, x_trans=x_t2_trans
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "customer_id": str(app["customer_id"]),
        "probability_of_default": round(prob, 6),
        "threshold": round(active_threshold, 6),
        "decision": decision,
        "requested_amount": round(requested_amount, 2),
        "recommended_limit": recommended_limit,
        "top_3_reasons": reasons,
        "latency_ms": latency_ms,
    }


def benchmark_latency(
    n_calls: int = 1000,
    sample_app: dict[str, Any] | None = None,
    _bundle: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Measure median, p95, and mean single-application scoring latency across `n_calls`."""
    app = sample_app or {
        "customer_id": "CUST_BENCH_001",
        "country_id": "Ghana",
        "lender_id": "297271",
        "loan_type": "Type_1",
        "Total_Amount": 300.0,
        "duration": 30.0,
        "disbursement_date": "2023-02-15",
        "New_versus_Repeat": "Repeat Loan",
        "n_past_loans": 4,
        "days_since_last_loan": 21.0,
        "n_past_labelled": 3,
        "n_past_defaults": 0,
    }
    # Warm up artifact cache once before measuring steady-state latency
    score_application(app, _bundle=_bundle)
    timings = np.empty(n_calls, dtype=np.float64)
    for i in range(n_calls):
        res = score_application(app, _bundle=_bundle)
        timings[i] = float(res["latency_ms"])

    return {
        "n_calls": float(n_calls),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(np.mean(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
    }


def main() -> None:
    sample_ghana = {
        "customer_id": "GHA_DEMO_001",
        "country_id": "Ghana",
        "lender_id": "GH_LENDER_NEW",
        "loan_type": "Type_3",
        "Total_Amount": 300.0,
        "duration": 30.0,
        "disbursement_date": "2023-02-15",
        "New_versus_Repeat": "New Loan",
        "n_past_loans": 0,
    }
    result = score_application(sample_ghana)
    print("Sample Ghana application score:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    stats = benchmark_latency(n_calls=1000)
    print("\nLatency benchmark (1,000 calls):")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
