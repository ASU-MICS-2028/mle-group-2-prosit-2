"""Guardrails on the shared feature lists, the feature file and the fold splits."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config as C

needs_features = pytest.mark.skipif(not C.FEATURES_PARQUET.exists(), reason="features.parquet not built locally")


def test_no_leaky_key_or_country_columns_in_model_features() -> None:
    forbidden = (
        set(C.LEAKY_COLUMNS) | set(C.REDUNDANT_COLUMNS) | set(C.KEY_COLUMNS) | {"country_id", "log_amount_local"}
    )
    for name, cols in (("task2", C.NUM_FEATURES_T2), ("task1", C.NUM_FEATURES_T1)):
        bad = forbidden & (set(cols) | set(C.CAT_FEATURES))
        assert not bad, f"Forbidden columns in {name} features: {bad}"


def test_macro_columns_are_not_model_inputs() -> None:
    """D-18: Kenya-only training gives 4 macro values; Ghana's are far outside them."""
    assert not set(C.MACRO_FEATURES) & set(C.NUM_FEATURES_T2)


def test_task1_excludes_its_own_target() -> None:
    assert C.TARGET_T1 not in C.NUM_FEATURES_T1
    assert "log_amount_local" not in C.NUM_FEATURES_T1


@needs_features
def test_features_parquet_fold_and_temporal_integrity() -> None:
    df = pd.read_parquet(C.FEATURES_PARQUET)
    expected = set(C.KEY_COLUMNS) | set(C.NUM_FEATURES_T2) | set(C.CAT_FEATURES) | set(C.MACRO_FEATURES) | {
        "country_id",
        "fold_group",
        "fold_strat",
        "temporal",
    }
    assert not expected - set(df.columns), f"Missing columns: {expected - set(df.columns)}"

    tr = df[df["split"] == "train"]
    for k in range(C.N_FOLDS):
        va_custs = set(tr.loc[tr["fold_group"] == k, "customer_id"])
        assert not va_custs & set(tr.loc[tr["fold_group"] != k, "customer_id"]), f"customer overlap in fold {k}"
    assert (tr.groupby("tbl_loan_id")["fold_group"].nunique() == 1).all(), "a loan straddles folds"
    last_train = tr.loc[tr["temporal"] == "train", "disbursement_date"].max()
    first_valid = tr.loc[tr["temporal"] == "valid", "disbursement_date"].min()
    assert last_train < first_valid, "temporal split is not ordered in time"


def test_deliverable_notebooks_exist() -> None:
    """The brief requires these four names; exploration notebooks may live in notebooks/exploration/."""
    names = {p.name for p in (C.ROOT / "notebooks").glob("*.ipynb")}
    required = {"01_eda.ipynb", "02_classification.ipynb", "03_regression.ipynb", "04_transfer_and_fairness.ipynb"}
    assert required <= names, f"Missing deliverable notebooks: {required - names}"


def test_target_eda_on_synthetic_data() -> None:
    from src.analysis import compute_target_eda

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
    assert res["n_rows"] == 4 and res["n_defaults"] == 2
    assert res["extra_pct_by_target"] is not None
    assert len(res["shift_2023_summary"]) == 2
