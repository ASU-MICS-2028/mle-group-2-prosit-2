"""Fake raw files shaped exactly like the real ones, so the whole pipeline can be tested in CI.

The data is small but keeps the structures the code depends on: repeat customers, two-lender
loans, defaults in every fold and on both sides of the 2023 cutoff, new borrowers, Kenyan Test rows,
and Ghanaian Test rows with unseen lenders and loan types.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MACRO_INDICATORS = [
    "Inflation, consumer prices (annual %)",
    "Official exchange rate (LCU per US$, period average)",
    "Deposit interest rate (%)",
    "Unemployment rate",
    "Lending interest rate (%)",
]


def _loans(rng, n_customers: int, country: str, lenders, types, start: str, end: str, id_offset: int) -> pd.DataFrame:
    rows = []
    loan_id = id_offset
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    span = (t1 - t0).days
    for c in range(n_customers):
        cust = id_offset + c
        n = int(rng.integers(1, 9))
        days = np.sort(rng.integers(0, span, n))
        risky = rng.random() < 0.15
        for k, d in enumerate(days):
            loan_id += 1
            disb = t0 + pd.Timedelta(days=int(d))
            duration = int(rng.choice([7, 7, 7, 14, 30, 60]))
            amount = float(np.round(rng.lognormal(8.5 if country == "Kenya" else 7.8, 1.0)))
            late = disb >= pd.Timestamp("2023-01-01")
            p = 0.02 + 0.25 * risky + 0.15 * (k == 0) + 0.10 * late + 0.05 * (duration > 14)
            target = int(rng.random() < min(p, 0.9))
            fee = rng.uniform(0.0, 0.03) if target == 0 else rng.uniform(0.1, 0.5)
            n_lenders = 2 if rng.random() < 0.04 else 1
            for j in range(n_lenders):
                portion = 0.3 if n_lenders == 1 else 0.2
                lender = str(lenders[j % len(lenders)] if n_lenders == 2 else rng.choice(lenders))
                rows.append(
                    {
                        "ID": f"ID_{cust}{loan_id}{lender}",
                        "customer_id": cust,
                        "country_id": country,
                        "tbl_loan_id": loan_id,
                        "lender_id": lender,
                        "loan_type": str(rng.choice(types)),
                        "Total_Amount": amount,
                        "Total_Amount_to_Repay": round(amount * (1 + fee)),
                        "disbursement_date": disb.date().isoformat(),
                        "due_date": (disb + pd.Timedelta(days=duration)).date().isoformat(),
                        "duration": duration,
                        "New_versus_Repeat": "New Loan" if k == 0 and rng.random() < 0.5 else "Repeat Loan",
                        "Amount_Funded_By_Lender": amount * portion,
                        "Lender_portion_Funded": portion,
                        "Lender_portion_to_be_repaid": round(amount * portion * (1 + fee)),
                        "target": target,
                    }
                )
    df = pd.DataFrame(rows)
    # two-lender loans share the borrower, amount, date and type
    first = df.groupby("tbl_loan_id")[["loan_type"]].transform("first")
    df["loan_type"] = first["loan_type"]
    return df


def make_raw(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, test, macro, findex) DataFrames with the real files' columns."""
    kenya = _loans(rng, 260, "Kenya", ["267278", "251804", "245684"], ["Type_1", "Type_1", "Type_7", "Type_14"],
                   "2021-10-01", "2024-11-01", 100_000)
    is_test = kenya["customer_id"].isin(kenya["customer_id"].drop_duplicates().sample(frac=0.2, random_state=1))
    train = kenya[~is_test].reset_index(drop=True)
    ghana = _loans(rng, 40, "Ghana", ["296540", "297182"], ["Type_3", "Type_8", "Type_1"], "2021-11-01", "2023-03-01", 900_000)
    test = pd.concat([kenya[is_test], ghana], ignore_index=True).drop(columns="target")

    years = [f"YR{y}" for y in range(2001, 2024)]
    base = {
        "Kenya": [7.0, 115.0, 7.0, 5.7, 12.0],
        "Ghana": [20.0, 7.0, 12.0, 3.2, np.nan],
    }
    macro_rows = []
    for country, values in base.items():
        for ind, v in zip(MACRO_INDICATORS, values):
            row = {"Country": country, "Indicator": ind}
            row.update({y: (np.nan if np.isnan(v) else v * (1 + 0.05 * (int(y[2:]) - 2015))) for y in years})
            macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)

    n = 300
    findex = pd.DataFrame(
        {
            "wgt": rng.uniform(0.5, 2.0, n),
            "female": rng.integers(1, 3, n),
            "urbanicity_f2f": rng.integers(1, 3, n),
            "inc_q": rng.integers(1, 6, n),
            "educ": rng.integers(1, 4, n),
            "account": rng.integers(1, 3, n),
            "account_mob": rng.integers(1, 3, n),
            "fin22a": rng.integers(1, 3, n),
            "borrowed": rng.integers(1, 3, n),
        }
    )
    return train, test, macro, findex


def write_raw(root: Path, rng: np.random.Generator) -> dict[str, Path]:
    train, test, macro, findex = make_raw(rng)
    credit = root / "data" / "raw" / "african-credit-scoring-challenge"
    fx_dir = root / "data" / "raw" / "GHA_2021_FINDEX_v02_M_csv"
    credit.mkdir(parents=True)
    fx_dir.mkdir(parents=True)
    paths = {
        "train": credit / "Train.csv",
        "test": credit / "Test.csv",
        "macro": credit / "economic_indicators.csv",
        "findex": fx_dir / "micro_gha.csv",
    }
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    macro.to_csv(paths["macro"], index=False)
    findex.to_csv(paths["findex"], index=False)
    return paths
