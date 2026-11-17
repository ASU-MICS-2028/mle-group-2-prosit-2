"""Part E scoring service: one application in; probability, decision, limit and three reasons out.

    from src.serving import score_application
    score_application({"customer_id": "C1", "country_id": "Ghana", ...})

Decision rule: decline if calibrated P(default) > the cost-matrix threshold; otherwise approve,
or approve with a lower limit if the request exceeds the Task 1 limit.
`needs_review` is True when the application is outside what the model was trained on
(another country, an unseen lender or product, an unusual amount): a person should look at it.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .. import config as C
from ..processing import get_feature_lists
from ..training.artifacts import CLF_ARTIFACT, REG_ARTIFACT
from .reasons import explain
from .schema import prepare_single_application

_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def load_bundle(clf_path: Path | None = None, reg_path: Path | None = None) -> dict[str, Any]:
    cp, rp = Path(clf_path or C.ARTIFACTS / CLF_ARTIFACT), Path(reg_path or C.ARTIFACTS / REG_ARTIFACT)
    key = (str(cp), str(rp))
    if key not in _CACHE:
        if not cp.exists() or not rp.exists():
            raise FileNotFoundError(f"Missing {cp.name} / {rp.name}. Run `make train` (it saves the artifacts).")
        _CACHE[key] = {"clf_bundle": joblib.load(cp), "reg_bundle": joblib.load(rp)}
    return _CACHE[key]


def review_flags(app: dict[str, Any], amount_usd: float, scope: dict[str, Any] | None) -> list[str]:
    """Reasons a person should review this score before it is used."""
    if not scope:
        return []
    flags = []
    if str(app["country_id"]) not in scope["countries"]:
        flags.append(f"Model trained on {', '.join(scope['countries'])} loans only; {app['country_id']} scores are unvalidated")
    if str(app["lender_id"]) not in scope["lenders"]:
        flags.append("Lender not seen in training data")
    if str(app["loan_type"]) not in scope["loan_types"]:
        flags.append("Loan product not seen in training data")
    lo, hi = scope["amount_usd_range"]
    if not lo <= amount_usd <= hi:
        flags.append(f"Requested amount (USD {amount_usd:,.0f}) outside the training range")
    return flags


def score_application(
    app: dict[str, Any],
    clf_path: Path | None = None,
    reg_path: Path | None = None,
    threshold: float | None = None,
    _bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    bundle = _bundle or load_bundle(clf_path, reg_path)
    clf, reg = bundle["clf_bundle"], bundle["reg_bundle"]

    X_t2, fx = prepare_single_application(app)
    num_t1, cat_t1 = get_feature_lists("task1")

    pipe = clf["pipeline"]
    x_trans = pipe.named_steps["pre"].transform(X_t2)
    raw_prob = float(pipe.named_steps["est"].predict_proba(x_trans)[0, 1])
    calibrator = clf.get("calibrator")
    prob = float(calibrator.predict(np.array([raw_prob]))[0]) if calibrator is not None else raw_prob

    # Task 1 predicts USD; convert with the same exchange rate used for the features.
    z = reg["pipeline"].predict(X_t2[num_t1 + cat_t1])
    limit_local = round(float(reg["target_scaler"].inverse_transform(z)[0]) * fx, 2)

    cutoff = float(threshold if threshold is not None else clf["threshold"])
    requested = float(app["Total_Amount"])
    if prob > cutoff:
        decision = "decline"
    elif requested > limit_local:
        decision = "limit"
    else:
        decision = "approve"

    flags = review_flags(app, requested / fx, clf.get("training_scope"))
    return {
        "customer_id": str(app["customer_id"]),
        "probability_of_default": round(prob, 6),
        "threshold": round(cutoff, 6),
        "decision": decision,
        "requested_amount": round(requested, 2),
        "recommended_limit": limit_local if decision != "decline" else 0.0,
        "currency": {"Kenya": "KES", "Ghana": "GHS"}.get(str(app["country_id"]), "local"),
        "top_3_reasons": explain(pipe, X_t2, decision=decision, x_trans=x_trans),
        "needs_review": bool(flags),
        "review_flags": flags,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }
