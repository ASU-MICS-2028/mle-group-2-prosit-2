"""Data profiling for the EDA report: one function per question, each returns a small table.

Every number in reports/EDA_REPORT.md comes from these functions, so the report can be rebuilt
from the raw files at any time (`make eda`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import config as C

RAW_NUMERIC = [
    "Total_Amount",
    "Total_Amount_to_Repay",
    "duration",
    "Amount_Funded_By_Lender",
    "Lender_portion_Funded",
    "Lender_portion_to_be_repaid",
]


# ---------------------------------------------------------------- loading
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and Test exactly as delivered, with parsed dates and string ids."""
    kw = {"parse_dates": ["disbursement_date", "due_date"], "dtype": {"lender_id": str}}
    return pd.read_csv(C.TRAIN_CSV, **kw), pd.read_csv(C.TEST_CSV, **kw)


def load_macro_long() -> pd.DataFrame:
    """economic_indicators.csv as (country, indicator, year, value)."""
    m = pd.read_csv(C.MACRO_CSV)
    long = m.melt(id_vars=["Country", "Indicator"], var_name="year", value_name="value")
    long["year"] = long["year"].str.replace("YR", "").astype(int)
    return long.dropna(subset=["value"])


# ---------------------------------------------------------------- structure
def overview(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Rows, loans, customers, dates and default rate per split and country."""
    both = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    out = both.groupby(["split", "country_id"]).agg(
        rows=("ID", "size"),
        loans=("tbl_loan_id", "nunique"),
        customers=("customer_id", "nunique"),
        first_loan=("disbursement_date", "min"),
        last_loan=("disbursement_date", "max"),
        default_rate=("target", "mean"),
    )
    out["first_loan"] = out["first_loan"].dt.date
    out["last_loan"] = out["last_loan"].dt.date
    return out.reset_index()


def data_quality(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Missing values, duplicate rows/ids and exact-formula columns."""
    rows = []
    for name, df in (("train", train), ("test", test)):
        rows.append(
            {
                "split": name,
                "missing_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_ids": int(df["ID"].duplicated().sum()),
                "due_eq_disb_plus_duration": float(
                    ((df["due_date"] - df["disbursement_date"]).dt.days == df["duration"]).mean()
                ),
                "funded_eq_amount_x_portion": float(
                    np.isclose(df["Amount_Funded_By_Lender"], df["Total_Amount"] * df["Lender_portion_Funded"], atol=1).mean()
                ),
                "zero_amount_funded": int((df["Amount_Funded_By_Lender"] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def cardinality(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Distinct values per categorical column, the top value's share, and Test values unseen in Train."""
    rows = []
    for col in ("lender_id", "loan_type", "New_versus_Repeat", "country_id"):
        unseen = set(test[col]) - set(train[col])
        rows.append(
            {
                "column": col,
                "n_train": train[col].nunique(),
                "n_test": test[col].nunique(),
                "top_value": train[col].value_counts().index[0],
                "top_share": float(train[col].value_counts(normalize=True).iloc[0]),
                "unseen_in_test": ", ".join(sorted(map(str, unseen))) or "-",
                "test_rows_unseen": float(test[col].isin(unseen).mean()),
            }
        )
    return pd.DataFrame(rows)


def loans_per_customer(train: pd.DataFrame) -> pd.Series:
    return train.drop_duplicates("tbl_loan_id").groupby("customer_id").size()


def two_lender_loans(train: pd.DataFrame) -> dict[str, float]:
    rows_per_loan = train.groupby("tbl_loan_id").size()
    multi = rows_per_loan[rows_per_loan > 1].index
    agree = train[train["tbl_loan_id"].isin(multi)].groupby("tbl_loan_id")["target"].nunique().eq(1).mean()
    is_multi = train["tbl_loan_id"].isin(multi)
    return {
        "loans": int(rows_per_loan.size),
        "two_lender_loans": int(len(multi)),
        "labels_agree": float(agree),
        "default_rate_two_lender": float(train.loc[is_multi, "target"].mean()),
        "default_rate_single": float(train.loc[~is_multi, "target"].mean()),
    }


def train_test_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    shared_loans = set(train["tbl_loan_id"]) & set(test["tbl_loan_id"])
    return {
        "shared_customers": len(set(train["customer_id"]) & set(test["customer_id"])),
        "shared_loans": len(shared_loans),
        "ghana_customers_in_train": int(test.loc[test["country_id"] == "Ghana", "customer_id"].isin(train["customer_id"]).sum()),
    }


# ---------------------------------------------------------------- target
def rate_table(df: pd.DataFrame, by, min_count: int = 1) -> pd.DataFrame:
    """Loans, defaults and default rate for each value of `by` (a column name or a Series)."""
    out = df.groupby(by, observed=True)["target"].agg(loans="size", defaults="sum", default_rate="mean").reset_index()
    return out[out["loans"] >= min_count]


def amount_deciles(df: pd.DataFrame) -> pd.DataFrame:
    band = pd.qcut(df["Total_Amount"], 10, duplicates="drop")
    out = rate_table(df.assign(amount_decile=band), "amount_decile")
    out["amount_decile"] = out["amount_decile"].astype(str)
    return out


def defaulter_followup(train: pd.DataFrame) -> dict[str, float]:
    """Do defaulters get another loan? (If not, labels only cover approved, non-defaulting history.)"""
    loans = train.drop_duplicates("tbl_loan_id").sort_values("disbursement_date")
    first_default = loans[loans["target"] == 1].groupby("customer_id")["due_date"].min()
    later = loans.merge(first_default.rename("first_default_due"), on="customer_id")
    got_another = later[later["disbursement_date"] > later["first_default_due"]].groupby("customer_id").size()
    return {
        "customers_who_defaulted": int(first_default.size),
        "share_given_another_loan": float(got_another.size / max(first_default.size, 1)),
    }


# ---------------------------------------------------------------- leakage
def univariate_auc(train: pd.DataFrame) -> pd.DataFrame:
    """How well each raw numeric column alone separates defaulters (0.5 = no signal, 1 = perfect)."""
    df = train.assign(
        markup=train["Total_Amount_to_Repay"] / train["Total_Amount"].replace(0, np.nan) - 1,
        lender_owed_over_funded=train["Lender_portion_to_be_repaid"] / train["Amount_Funded_By_Lender"].replace(0, np.nan),
    )
    verdicts = {
        "markup": "leak: repay / principal - 1 (D-2)",
        "lender_owed_over_funded": "leak: lender owed / funded (D-2)",
        "Total_Amount_to_Repay": "dropped: its ratio to principal leaks (D-2)",
        "Lender_portion_to_be_repaid": "dropped: its ratio to funded leaks (D-2)",
        "Amount_Funded_By_Lender": "dropped: = amount x portion (redundant)",
        "Total_Amount": "kept (log, USD); Task 1 target",
        "duration": "kept",
        "Lender_portion_Funded": "kept",
    }
    rows = []
    for col in verdicts:
        x = df[col].fillna(df[col].median())
        auc = roc_auc_score(df["target"], x)
        rows.append({"column": col, "auc": max(auc, 1 - auc), "verdict": verdicts[col]})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def markup_by_outcome(train: pd.DataFrame) -> pd.DataFrame:
    df = train.assign(markup_pct=(train["Total_Amount_to_Repay"] / train["Total_Amount"].replace(0, np.nan) - 1) * 100)
    df["outcome"] = np.where(df["target"] == 1, "defaulted", "repaid")
    return df.groupby("outcome")["markup_pct"].describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%", "mean"]].reset_index()


# ---------------------------------------------------------------- engineered features
def history_signal(features: pd.DataFrame) -> pd.DataFrame:
    """Default rate by number of earlier loans (the strongest engineered signal)."""
    tr = features[features["split"] == "train"]
    bucket = pd.cut(tr["n_past_loans"], [-1, 0, 2, 5, 10, 20, 50, 10_000], labels=["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51+"])
    out = rate_table(tr.assign(past_loans=bucket), "past_loans")
    out["past_loans"] = out["past_loans"].astype(str)
    return out


def feature_missingness(features: pd.DataFrame) -> pd.DataFrame:
    """Share of rows where each engineered feature is empty, by split and country."""
    cols = ["days_since_last_loan", "past_default_rate", "days_since_last_default"]
    grp = features.assign(group=features["split"] + " " + features["country_id"]).groupby("group")[cols]
    return grp.apply(lambda g: g.isna().mean()).reset_index()


# ---------------------------------------------------------------- Kenya vs Ghana
def kenya_ghana_shift(features: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side summary of the populations the model learns from and is applied to."""
    df = features.assign(amount_usd=np.expm1(features["log_amount_usd"]))
    df["group"] = np.where(df["split"] == "train", "Train (Kenya)", "Test " + df["country_id"])
    train_lenders = set(df.loc[df["split"] == "train", "lender_id"])
    train_types = set(df.loc[df["split"] == "train", "loan_type"])
    out = df.groupby("group").agg(
        rows=("ID", "size"),
        median_amount_usd=("amount_usd", "median"),
        share_7_day=("duration", lambda s: float((s == 7).mean())),
        mean_duration_days=("duration", "mean"),
        share_new_borrower=("is_new_borrower", "mean"),
        median_past_loans=("n_past_loans", "median"),
        share_no_labelled_history=("n_past_labelled", lambda s: float((s == 0).mean())),
        share_unseen_lender=("lender_id", lambda s: float((~s.isin(train_lenders)).mean())),
        share_unseen_loan_type=("loan_type", lambda s: float((~s.isin(train_types)).mean())),
    )
    return out.T.reset_index().rename(columns={"index": "measure"})


# ---------------------------------------------------------------- Findex (Part D)
def findex_access(findex: pd.DataFrame) -> pd.DataFrame:
    """Weighted % with an account, mobile money, and formal borrowing, by group (Ghana 2021)."""
    def pct(sub: pd.DataFrame, col: str) -> float:
        ok = sub[col].notna()
        return float(np.average((sub.loc[ok, col] == 1).astype(float), weights=sub.loc[ok, "wgt"]) * 100)

    groups = {
        "All adults": findex,
        "Women": findex[findex["female"] == 1],
        "Men": findex[findex["female"] == 2],
        "Rural": findex[findex["urbanicity_f2f"] == 1],
        "Urban": findex[findex["urbanicity_f2f"] == 2],
        "Poorest 40%": findex[findex["inc_q"] <= 2],
        "Richest 60%": findex[findex["inc_q"] >= 3],
        "Primary or less": findex[findex["educ"] == 1],
        "Tertiary": findex[findex["educ"] == 3],
    }
    return pd.DataFrame(
        [
            {
                "group": g,
                "n": len(s),
                "has_account": pct(s, "account"),
                "mobile_money": pct(s, "account_mob"),
                "borrowed_formally": pct(s, "fin22a"),
                "borrowed_any": pct(s, "borrowed"),
            }
            for g, s in groups.items()
        ]
    )
