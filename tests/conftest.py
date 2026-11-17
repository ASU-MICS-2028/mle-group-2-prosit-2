"""Shared synthetic data so tests run in CI, where the real data is not available."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C


@pytest.fixture(scope="session")
def synthetic_features() -> pd.DataFrame:
    """A small fake features.parquet: Kenyan training rows with GroupKFold folds and a temporal split."""
    rng = np.random.default_rng(C.SEED)
    n = 600
    customers = rng.integers(0, 150, size=n)
    is_new = (rng.random(n) < 0.1).astype(int)
    target = (rng.random(n) < 0.05 + 0.3 * is_new).astype(float)
    df = pd.DataFrame(
        {
            "ID": [f"ID_{i}" for i in range(n)],
            "split": "train",
            "target": target,
            "customer_id": customers,
            "tbl_loan_id": np.arange(n),
            "disbursement_date": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
            "country_id": "Kenya",
            "lender_id": rng.choice(["267278", "251804"], size=n),
            "loan_type": rng.choice(["Type_1", "Type_7", "Type_14"], size=n),
            "log_amount_usd": np.log1p(rng.lognormal(3.8, 1.0, n)),
            "duration": rng.choice([7, 14, 30], size=n).astype(float),
            "Lender_portion_Funded": rng.uniform(0.1, 0.5, n),
            "is_new_borrower": is_new,
            "n_lenders_on_loan": 1,
            "n_past_loans": rng.integers(0, 20, n),
            "days_since_last_loan": np.where(is_new == 1, np.nan, rng.integers(1, 60, n)),
            "n_past_labelled": rng.integers(0, 15, n),
            "n_past_defaults": rng.integers(0, 2, n),
            "past_default_rate": rng.uniform(0, 0.2, n),
            "days_since_last_default": np.where(rng.random(n) < 0.05, rng.integers(10, 300, n), np.nan),
            "disb_month": rng.integers(1, 13, n),
            "disb_dow": rng.integers(0, 7, n),
            "disb_dom": rng.integers(1, 29, n),
        }
    )
    df["fold_group"] = df["customer_id"] % C.N_FOLDS
    df["fold_strat"] = np.arange(n) % C.N_FOLDS
    df["temporal"] = np.where(df["disbursement_date"] < C.TEMPORAL_CUTOFF, "train", "valid")
    return df


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """A throw-away project root with fake raw files; every config path points into it.

    Module-scoped: the end-to-end module builds the pipeline once; other modules see the real paths.
    """
    from tests.synthetic import write_raw

    from src.serving import schema

    root = tmp_path_factory.mktemp("project")
    paths = write_raw(root, np.random.default_rng(7))
    mp = pytest.MonkeyPatch()
    mp.setattr(C, "TRAIN_CSV", paths["train"])
    mp.setattr(C, "TEST_CSV", paths["test"])
    mp.setattr(C, "MACRO_CSV", paths["macro"])
    mp.setattr(C, "FINDEX_CSV", paths["findex"])
    for name, sub in {"PROCESSED": "data/processed", "PREDS": "preds", "ARTIFACTS": "artifacts", "REPORTS": "reports",
                      "FIGURES": "reports/figures", "TABLES": "reports/tables"}.items():
        mp.setattr(C, name, root / sub)
    mp.setattr(C, "FEATURES_PARQUET", root / "data/processed/features.parquet")
    schema._fx_table.cache_clear()
    yield root
    mp.undo()
    schema._fx_table.cache_clear()
