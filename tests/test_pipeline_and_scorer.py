"""Automated contract tests for preprocessing, metrics, calibration, and Part E scorer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from src import config as C
from src.evaluate import (
    bootstrap_ci,
    calibrate_oof,
    classification_metrics,
    cost_matrix_threshold,
    cost_sensitivity_table,
    regression_metrics,
)
from src.preprocess import TargetLogScaler, get_feature_lists, make_preprocessor
from src.score import prepare_single_application, score_application


def _make_synthetic_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        is_default = int(i % 5 == 0)
        amount = float(rng.uniform(150.0, 1200.0))
        app = {
            "customer_id": f"CUST_{i // 2}",
            "country_id": "Kenya",
            "lender_id": "297271" if i % 3 else "268908",
            "loan_type": "Type_1" if i % 4 else "Type_14",
            "Total_Amount": amount,
            "duration": 7.0 if i % 3 else 30.0,
            "disbursement_date": "2022-06-15",
            "New_versus_Repeat": "New Loan" if is_default else "Repeat Loan",
            "n_past_loans": 0 if is_default else int(rng.integers(1, 10)),
            "days_since_last_loan": None if is_default else float(rng.integers(5, 60)),
            "n_past_labelled": 0 if is_default else 3,
            "n_past_defaults": is_default,
            "days_since_last_default": 45.0 if (is_default and i % 2 == 0) else None,
        }
        row_df = prepare_single_application(app)
        row_df["target"] = is_default
        row_df["Total_Amount"] = amount
        rows.append(row_df)
    return pd.concat(rows, ignore_index=True)


def test_preprocessor_handles_unseen_ghana_categories() -> None:
    df = _make_synthetic_df(40)
    for task in ("task1", "task2"):
        num_cols, cat_cols = get_feature_lists(task)
        pre = make_preprocessor(task=task, sparse_output=False)
        X_tr = pre.fit_transform(df[num_cols + cat_cols])
        assert not np.isnan(X_tr).any()

        ghana_app = prepare_single_application(
            {
                "customer_id": "GH_999",
                "country_id": "Ghana",
                "lender_id": "UNSEEN_GH_LENDER",
                "loan_type": "Type_3",
                "Total_Amount": 300.0,
                "duration": 30.0,
                "disbursement_date": "2023-02-01",
                "New_versus_Repeat": "New Loan",
            }
        )
        X_gh = pre.transform(ghana_app[num_cols + cat_cols])
        assert X_gh.shape[1] == X_tr.shape[1]
        assert not np.isnan(X_gh).any()


def test_target_log_scaler_round_trip() -> None:
    y = np.array([100.0, 300.0, 1500.0, 25000.0])
    scaler = TargetLogScaler().fit(y)
    z = scaler.transform(y)
    y_back = scaler.inverse_transform(z)
    assert np.allclose(y, y_back, rtol=1e-6)


def test_cost_matrix_and_evaluation_metrics() -> None:
    cm = cost_matrix_threshold(principal=300.0, fee_rate=0.15, recovery_rate=0.20, collection_cost=25.0)
    assert np.isclose(cm["c_fp"], 45.0)
    assert np.isclose(cm["c_fn"], 265.0)
    assert np.isclose(cm["threshold"], 45.0 / 310.0)

    sens = cost_sensitivity_table()
    assert len(sens) == 3

    y_true = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.05, 0.08, 0.12, 0.45, 0.20, 0.65, 0.03, 0.07, 0.55, 0.10])
    folds = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])

    m = classification_metrics(y_true, y_prob, threshold=0.145)
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["selection_rate"] <= 1.0

    cal_p = calibrate_oof(y_true, y_prob, folds, method="isotonic")
    assert cal_p.shape == y_prob.shape
    assert np.all((cal_p >= 0.0) & (cal_p <= 1.0))

    ci = bootstrap_ci(y_true, y_prob, threshold=0.145, n_boot=50, seed=C.SEED)
    assert ci["pr_auc_ci_low"] <= ci["pr_auc_ci_high"]

    rm = regression_metrics(np.array([100.0, 200.0, 300.0]), np.array([110.0, 190.0, 310.0]))
    assert rm["rmse_orig"] >= rm["mae_orig"] >= 0.0


def test_score_application_contract_and_latency() -> None:
    df = _make_synthetic_df(60)
    num_t2, cat_t2 = get_feature_lists("task2")
    num_t1, cat_t1 = get_feature_lists("task1")

    clf_pipe = Pipeline(
        [("pre", make_preprocessor("task2")), ("est", LogisticRegression(class_weight="balanced", max_iter=500))]
    )
    clf_pipe.fit(df[num_t2 + cat_t2], df["target"].astype(int))

    y_scaler = TargetLogScaler().fit(df["Total_Amount"].to_numpy())
    reg_pipe = Pipeline([("pre", make_preprocessor("task1")), ("est", Ridge(alpha=1.0))])
    reg_pipe.fit(df[num_t1 + cat_t1], y_scaler.transform(df["Total_Amount"].to_numpy()))

    bundle = {
        "clf_bundle": {"pipeline": clf_pipe, "calibrator": None, "threshold": 0.145},
        "reg_bundle": {"pipeline": reg_pipe, "target_scaler": y_scaler},
    }

    payload = {
        "customer_id": "GH_TEST_01",
        "country_id": "Ghana",
        "lender_id": "NEW_GH_LENDER",
        "loan_type": "Type_3",
        "Total_Amount": 350.0,
        "duration": 30.0,
        "disbursement_date": "2023-03-10",
        "New_versus_Repeat": "New Loan",
    }
    # Warm up macro CSV cache once (matching steady-state scoring service behavior)
    score_application(payload, _bundle=bundle)
    res = score_application(payload, _bundle=bundle)

    assert 0.0 <= res["probability_of_default"] <= 1.0
    assert res["decision"] in {"approve", "limit", "decline"}
    assert res["recommended_limit"] >= 0.0
    assert len(res["top_3_reasons"]) == 3
    for reason in res["top_3_reasons"]:
        assert isinstance(reason, str) and len(reason.strip()) > 0
        assert "num__" not in reason and "cat__" not in reason
    assert res["latency_ms"] < 100.0
