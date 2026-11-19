"""The whole pipeline, raw CSVs to a scored application, on fake data shaped like the real files.

Order matters: each test builds on the previous one's outputs inside the shared sandbox, exactly as
`make features && make train && make eval && make eda && make figures && make score` would.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C


@pytest.fixture(scope="module")
def features(sandbox):
    from src import features as F

    df = F.build()
    F.validate(df)  # leakage, fold and history checks on the fake data
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.FEATURES_PARQUET, index=False)
    return df


def test_features_build_and_pass_their_own_leakage_checks(features) -> None:
    assert features["ID"].is_unique
    assert set(features["split"]) == {"train", "test"}
    assert features.loc[features["split"] == "test", "target"].isna().all()
    assert {"fold_group", "fold_strat", "temporal"} <= set(features.columns)
    assert not features["log_amount_usd"].isna().any(), "every row needs an exchange rate"


@pytest.fixture(scope="module")
def trained(features):
    from src.training import fit_and_save_deployment_artifacts, predict_test_split, run_cv

    for model in ("majority", "logreg", "lightgbm", "decision_tree"):
        for scheme in ("fold_group", "fold_strat", "temporal"):
            run_cv(features, model, task="task2", scheme=scheme)
    predict_test_split(features, "logreg", task="task2")
    for model in ("mean", "ridge", "lightgbm_reg"):
        run_cv(features, model, task="task1", scheme="fold_group")
    return fit_and_save_deployment_artifacts(features, clf_model="logreg", reg_model="lightgbm_reg")


def test_every_model_writes_the_shared_prediction_format(trained) -> None:
    files = sorted(C.PREDS.glob("*.parquet"))
    assert len(files) == 4 * 3 + 1 + 3
    for f in files:
        df = pd.read_parquet(f)
        assert list(df.columns) == C.PRED_COLUMNS, f.name
        assert df["y_pred"].notna().all(), f.name


def test_scorer_artifacts_load_and_score_both_countries(trained, features) -> None:
    from src.serving import score_application, score_frame
    from src.serving.scorer import _CACHE

    _CACHE.clear()
    base = {"customer_id": "X", "lender_id": "267278", "loan_type": "Type_1", "Total_Amount": 5000.0,
            "duration": 7, "disbursement_date": "2023-02-01", "New_versus_Repeat": "Repeat Loan"}
    kenya = score_application({**base, "country_id": "Kenya"})
    ghana = score_application({**base, "country_id": "Ghana", "lender_id": "NEW", "loan_type": "Type_3", "Total_Amount": 500.0})
    for res in (kenya, ghana):
        assert 0 < res["probability_of_default"] < 1
        assert res["decision"] in {"approve", "limit", "starter", "decline"}
        assert len(res["top_3_reasons"]) == 3
    assert not kenya["needs_review"] and ghana["needs_review"]
    assert (kenya["currency"], ghana["currency"]) == ("KES", "GHS")

    scores = score_frame(features[features["split"] == "test"], explain_sample=20)
    assert len(scores) == (features["split"] == "test").sum()
    assert scores.loc[scores["country_id"] == "Ghana", "needs_review"].all()
    assert scores["top_3_reasons"].notna().sum() == 20


def test_evaluation_tables(trained) -> None:
    from scripts import evaluate

    evaluate.main()
    t2 = pd.read_csv(C.TABLES / "task2_model_summary.csv")
    t1 = pd.read_csv(C.TABLES / "task1_model_summary.csv")
    assert {"fold_group", "fold_strat", "temporal"} <= set(t2["scheme"])
    assert t2.loc[t2["model"] == "majority", "roc_auc"].eq(0.5).all()
    real = t2[(t2["scheme"] != "temporal") & (t2["model"] != "majority")]  # a constant baseline has nothing to calibrate
    assert real["f1"].notna().all(), "threshold metrics must be computed on calibrated folds"
    assert {"rmse_orig", "mae_orig", "rmse_log", "mae_log"} <= set(t1.columns)
    assert (C.TABLES / "cost_sensitivity.csv").exists()


def test_eda_report_builds(features) -> None:
    from scripts import make_eda_report

    make_eda_report.main()
    text = (C.REPORTS / "EDA_REPORT.md").read_text(encoding="utf-8")
    assert text.count("## ") == 10
    assert text.count("**Decision it supports:**") == 10
    for link in [line.split("](")[1].split(")")[0] for line in text.splitlines() if line.startswith("![")]:
        assert (C.REPORTS / link).exists(), link


def test_figure_pack_and_catalogue(trained, monkeypatch) -> None:
    from scripts import make_figures
    from src.serving import benchmark_latency

    monkeypatch.setattr(make_figures, "benchmark_latency", lambda n_calls, return_samples: benchmark_latency(30, return_samples=True))
    make_figures.main()
    catalogue = (C.REPORTS / "FIGURES.md").read_text(encoding="utf-8")
    linked = [part.split("](")[1].split(")")[0] for part in catalogue.split("[`")[1:]]
    assert len(linked) >= 25
    for link in linked:
        assert (C.REPORTS / link).stat().st_size > 5_000, f"{link} missing or empty"
    for section in ("Preprocessing", "Validation", "Evaluation", "Interpretation", "Fairness", "transfer", "product"):
        assert section in catalogue


def test_target_eda_script(features) -> None:
    from scripts import run_target_eda

    run_target_eda.main()
    assert (C.TABLES / "eda_target_by_year.csv").exists()


def test_latency_benchmark_is_reported(trained) -> None:
    from src.serving import benchmark_latency

    stats = benchmark_latency(n_calls=25, return_samples=True)
    assert len(stats["samples_ms"]) == 25
    assert 0 < stats["median_ms"] <= stats["p95_ms"] <= stats["max_ms"]
    assert np.isfinite(stats["mean_ms"])
