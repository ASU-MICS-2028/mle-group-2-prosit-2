"""Automated guardrails ensuring zero data leakage and strict fold integrity."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config as C


def test_no_leaky_or_target_columns_in_feature_lists() -> None:
    forbidden = set(C.LEAKY_COLUMNS) | set(C.REDUNDANT_COLUMNS) | set(C.KEY_COLUMNS) | {"country_id"}
    t2_cols = set(C.NUM_FEATURES_T2) | set(C.CAT_FEATURES)
    t1_cols = set(C.NUM_FEATURES_T1) | set(C.CAT_FEATURES)

    assert not (forbidden & t2_cols), f"Forbidden columns in Task 2 features: {forbidden & t2_cols}"
    assert not (forbidden & t1_cols), f"Forbidden columns in Task 1 features: {forbidden & t1_cols}"


def test_task1_excludes_amount_derived_features() -> None:
    t1_cols = set(C.NUM_FEATURES_T1) | set(C.CAT_FEATURES)
    assert "log_amount_usd" not in t1_cols
    assert "log_amount_local" not in t1_cols


@pytest.mark.skipif(not C.FEATURES_PARQUET.exists(), reason="features.parquet not built locally")
def test_features_parquet_fold_and_temporal_integrity() -> None:
    df = pd.read_parquet(C.FEATURES_PARQUET)
    expected = set(C.KEY_COLUMNS) | set(C.NUM_FEATURES_T2) | set(C.CAT_FEATURES) | {
        "country_id",
        "fold_group",
        "fold_strat",
        "temporal",
    }
    missing = expected - set(df.columns)
    assert not missing, f"Missing expected columns in features.parquet: {missing}"

    tr = df[df["split"] == "train"]
    for k in range(C.N_FOLDS):
        va_custs = set(tr.loc[tr["fold_group"] == k, "customer_id"])
        tr_custs = set(tr.loc[tr["fold_group"] != k, "customer_id"])
        assert not (va_custs & tr_custs), f"Customer overlap detected in fold_group={k}"

    assert (tr.groupby("tbl_loan_id")["fold_group"].nunique() == 1).all()
    max_tr_date = tr.loc[tr["temporal"] == "train", "disbursement_date"].max()
    min_va_date = tr.loc[tr["temporal"] == "valid", "disbursement_date"].min()
    assert max_tr_date < min_va_date, "Temporal split violates chronological ordering"


def test_only_four_deliverable_notebooks_exist() -> None:
    nb_files = {p.name for p in (C.ROOT / "notebooks").glob("*.ipynb")}
    expected = {
        "01_eda.ipynb",
        "02_classification.ipynb",
        "03_regression.ipynb",
        "04_transfer_and_fairness.ipynb",
    }
    assert nb_files == expected, f"Expected exact 4 deliverable notebooks, found: {sorted(nb_files)}"


def test_eda_target_computation_on_synthetic_data() -> None:
    from src.eda_target import compute_target_eda

    df = pd.DataFrame(
        {
            "lender_id": ["267278", "267278", "251804", "251804"],
            "loan_type": ["Type_1", "Type_1", "Type_7", "Type_7"],
            "Total_Amount": [1000.0, 2000.0, 3000.0, 4000.0],
            "Total_Amount_to_Repay": [1010.0, 2500.0, 3030.0, 5000.0],
            "disbursement_date": ["2022-05-01", "2022-06-01", "2023-03-01", "2023-04-01"],
            "duration": [7, 7, 14, 30],
            "New_versus_Repeat": ["Repeat Loan", "New Loan", "Repeat Loan", "New Loan"],
            "target": [0, 1, 0, 1],
        }
    )
    res = compute_target_eda(df)
    assert res["n_rows"] == 4
    assert res["n_defaults"] == 2
    assert res["extra_pct_by_target"] is not None
    assert len(res["shift_2023_summary"]) == 2
