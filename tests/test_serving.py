"""Part E scorer contract: decision, limit in local currency, reasons a borrower can read, review flags."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from src.evaluation import PlattCalibrator
from src.processing import TargetLogScaler, get_feature_lists, make_preprocessor
from src.serving import REASON_TEMPLATES, exchange_rate, schema, score_application
from src.serving.reasons import NEVER_SHOWN


@pytest.fixture(autouse=True)
def fx_table(monkeypatch):
    """Real 2022 rates from economic_indicators.csv, so these tests also run in CI (no data there)."""
    monkeypatch.setattr(schema, "_fx_table", lambda: {("Kenya", 2022): 117.866, ("Ghana", 2022): 8.2724})

KENYA = {
    "customer_id": "K1",
    "country_id": "Kenya",
    "lender_id": "267278",
    "loan_type": "Type_1",
    "Total_Amount": 5000.0,
    "duration": 7.0,
    "disbursement_date": "2023-02-15",
    "New_versus_Repeat": "New Loan",
}
GHANA = {**KENYA, "customer_id": "G1", "country_id": "Ghana", "lender_id": "NEW_GH", "loan_type": "Type_3",
         "Total_Amount": 350.0, "duration": 30.0}


@pytest.fixture(scope="module")
def bundle(synthetic_features):
    df = synthetic_features
    num2, cat2 = get_feature_lists("task2")
    num1, cat1 = get_feature_lists("task1")
    y = df["target"].astype(int)
    clf = Pipeline([("pre", make_preprocessor("task2")), ("est", LogisticRegression(class_weight="balanced", max_iter=500))])
    clf.fit(df[num2 + cat2], y)
    cal = PlattCalibrator().fit(clf.predict_proba(df[num2 + cat2])[:, 1], y)
    amount_usd = np.expm1(df["log_amount_usd"].to_numpy())
    scaler = TargetLogScaler().fit(amount_usd)
    reg = Pipeline([("pre", make_preprocessor("task1")), ("est", Ridge())]).fit(df[num1 + cat1], scaler.transform(amount_usd))
    scope = {"countries": ["Kenya"], "lenders": ["251804", "267278"], "loan_types": ["Type_1", "Type_14", "Type_7"],
             "amount_usd_range": [float(amount_usd.min()), float(amount_usd.max())]}
    return {
        "clf_bundle": {"pipeline": clf, "calibrator": cal, "threshold": 0.05, "training_scope": scope},
        "reg_bundle": {"pipeline": reg, "target_scaler": scaler},
    }


def test_contract(bundle) -> None:
    res = score_application(KENYA, _bundle=bundle)
    assert 0.0 < res["probability_of_default"] < 1.0
    assert res["decision"] in {"approve", "limit", "starter", "decline"}
    assert len(res["top_3_reasons"]) == 3 and all(r.strip() for r in res["top_3_reasons"])
    assert res["currency"] == "KES"
    assert res["latency_ms"] < 1000


def test_limit_is_in_local_currency(bundle) -> None:
    """Same application, same model: the Ghana limit in GHS should equal the Kenya limit in KES
    divided by the KES/GHS rate (the model predicts USD; only the conversion differs)."""
    k = score_application(KENYA, threshold=1.0, _bundle=bundle)
    g = score_application({**KENYA, "country_id": "Ghana"}, threshold=1.0, _bundle=bundle)
    fx_k, fx_g = exchange_rate("Kenya", 2023), exchange_rate("Ghana", 2023)
    # amount_usd differs (5000 KES vs 5000 GHS), but Task 1 does not use the amount, so the USD limit is equal
    assert np.isclose(k["recommended_limit"] / fx_k, g["recommended_limit"] / fx_g, rtol=1e-3)
    assert g["currency"] == "GHS"


def test_out_of_scope_application_is_flagged(bundle) -> None:
    res = score_application(GHANA, _bundle=bundle)
    assert res["needs_review"]
    text = " ".join(res["review_flags"])
    assert "Ghana" in text and "Lender" in text and "product" in text
    assert not score_application({**KENYA, "New_versus_Repeat": "Repeat Loan"}, _bundle=bundle)["review_flags"]


def test_reasons_never_mention_calendar_or_internal_fields(bundle) -> None:
    hidden = {REASON_TEMPLATES[f][i] for f in NEVER_SHOWN if f in REASON_TEMPLATES for i in (0, 1)}
    for app in (KENYA, GHANA):
        res = score_application(app, _bundle=bundle)
        assert not hidden & set(res["top_3_reasons"])
        assert not any(w in " ".join(res["top_3_reasons"]).lower() for w in ("day-of", "month", "exchange", "inflation"))


def test_missing_field_is_rejected(bundle) -> None:
    with pytest.raises(ValueError, match="Missing required"):
        score_application({k: v for k, v in KENYA.items() if k != "duration"}, _bundle=bundle)


def test_reasons_never_contradict_the_record() -> None:
    """A positive push from past_default_rate must not become 'not repaid' for someone with 0 defaults."""
    from src.serving.reasons import top_reasons

    contribs = {"past_default_rate": 0.9, "n_past_defaults": 0.5, "is_new_borrower": 0.4, "n_past_loans": 0.3,
                "loan_type": 0.2, "duration": 0.1}
    clean = {"past_default_rate": 0.0, "n_past_defaults": 0, "is_new_borrower": 0, "n_past_loans": 2}
    reasons = top_reasons(contribs, "decline", raw=clean)
    assert REASON_TEMPLATES["past_default_rate"][0] not in reasons
    assert REASON_TEMPLATES["n_past_defaults"][0] not in reasons
    assert REASON_TEMPLATES["is_new_borrower"][0] not in reasons
    assert reasons[0] == REASON_TEMPLATES["n_past_loans"][0]


# ----------------------------------------------------------------- starter loan (D-24) and borrower wording
STARTER = {"max_probability": 0.30, "share_of_request": 0.25}


def test_decide_offers_a_starter_loan_only_to_first_time_borrowers_in_the_band() -> None:
    from src.serving import decide

    prob = np.array([0.05, 0.20, 0.20, 0.50, 0.05])
    is_new = np.array([1, 1, 0, 1, 0])
    requested = np.array([1000.0, 1000.0, 1000.0, 1000.0, 3000.0])
    limit = np.array([2000.0, 2000.0, 2000.0, 2000.0, 2000.0])
    decision, offered = decide(prob, requested, limit, is_new, threshold=0.10, starter=STARTER)
    assert decision.tolist() == ["approve", "starter", "decline", "decline", "limit"]
    assert offered.tolist() == [2000.0, 250.0, 0.0, 0.0, 2000.0]


def test_starter_amount_never_exceeds_the_task1_limit() -> None:
    from src.serving import decide

    decision, offered = decide(0.2, 10_000.0, 1_000.0, 1, threshold=0.10, starter=STARTER)
    assert str(decision) == "starter" and float(offered) == 1_000.0


def test_without_a_starter_policy_the_rule_is_unchanged() -> None:
    from src.serving import decide

    decision, offered = decide(0.2, 1000.0, 2000.0, 1, threshold=0.10)
    assert str(decision) == "decline" and float(offered) == 0.0


def test_scorer_and_batch_agree_on_the_starter_rule(bundle, synthetic_features) -> None:
    from src.serving import score_frame

    b = {**bundle, "clf_bundle": {**bundle["clf_bundle"], "threshold": 0.0, "starter_loan": {"max_probability": 1.0, "share_of_request": 0.25}}}
    single = score_application(KENYA, _bundle=b)  # KENYA is a first-time borrower
    assert single["decision"] == "starter"
    assert single["recommended_limit"] <= 0.25 * KENYA["Total_Amount"] + 1e-6
    rows = synthetic_features.assign(log_amount_local=np.log1p(1000.0), macro_fx=117.866).head(50)
    out = score_frame(rows, bundle=b)
    new = out["is_new_borrower"] == 1
    assert (out.loc[new, "decision"] == "starter").all() and (out.loc[~new, "decision"] == "decline").all()
    assert (out.loc[new, "offered_local"] <= 250.0 + 1e-6).all()


def test_borrower_reasons_reword_lender_and_drop_filler() -> None:
    from src.serving import borrower_reasons
    from src.serving.reasons import FALLBACK_REASON

    lender = REASON_TEMPLATES["lender_id"][0]
    shown = borrower_reasons([REASON_TEMPLATES["duration"][0], lender, FALLBACK_REASON])
    assert lender not in shown and FALLBACK_REASON not in shown
    assert len(shown) == 2 and "product" in shown[1]
