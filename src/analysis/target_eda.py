"""Reproducible target and leakage exploration (Henry's explore_target_henry.ipynb as a script).

Covers Issue #6: overall default rate, missing values, repayment markup leak check (`extra_pct`),
default rate by tenure, year, month, lender_id, loan_type, and 2023+ lender-mix vs population shift.

Run:  python -m scripts.run_target_eda   (or: make eda-target)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config as C


def load_target_eda_source() -> tuple[pd.DataFrame, str]:
    """Load raw Train.csv if present, otherwise fall back to processed features.parquet."""
    if C.TRAIN_CSV.exists():
        df = pd.read_csv(C.TRAIN_CSV, parse_dates=["disbursement_date", "due_date"])
        return df, "raw:Train.csv"
    if C.FEATURES_PARQUET.exists():
        df = pd.read_parquet(C.FEATURES_PARQUET)
        df = df[df["split"] == "train"].reset_index(drop=True)
        return df, "processed:features.parquet"
    raise FileNotFoundError(
        f"Neither {C.TRAIN_CSV} nor {C.FEATURES_PARQUET} exists. Download raw data or build features first."
    )


def _rate_table(df: pd.DataFrame, group_col: str | pd.Series) -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)["target"]
        .agg(count="count", defaults="sum", default_rate="mean")
        .reset_index()
    )
    out["defaults"] = out["defaults"].astype(int)
    out["default_rate_pct"] = (out["default_rate"] * 100.0).round(3)
    return out


def compute_target_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Compute target distributions, leak checks, and 2023+ shift tables."""
    work = df.copy()
    if "split" in work.columns:
        work = work[work["split"] == "train"].copy()
    work["target"] = work["target"].astype(int)
    work["disbursement_date"] = pd.to_datetime(work["disbursement_date"])
    work["lender_id"] = work["lender_id"].astype(str)
    work["loan_type"] = work["loan_type"].astype(str)

    n_rows = int(len(work))
    n_defaults = int(work["target"].sum())
    default_rate = float(work["target"].mean()) if n_rows > 0 else 0.0
    missing_counts = work.isna().sum()

    extra_pct_by_target: pd.DataFrame | None = None
    if {"Total_Amount_to_Repay", "Total_Amount"}.issubset(work.columns):
        extra = (work["Total_Amount_to_Repay"] / work["Total_Amount"].replace(0, np.nan) - 1.0) * 100.0
        extra_pct_by_target = (
            work.assign(extra_pct=extra)
            .groupby("target")["extra_pct"]
            .agg(
                count="count",
                median="median",
                q25=lambda s: float(s.quantile(0.25)),
                q75=lambda s: float(s.quantile(0.75)),
            )
            .reset_index()
        )

    if "New_versus_Repeat" in work.columns:
        tenure_series = work["New_versus_Repeat"].astype(str)
    elif "is_new_borrower" in work.columns:
        tenure_series = pd.Series(
            np.where(work["is_new_borrower"] == 1, "New Loan", "Repeat Loan"),
            index=work.index,
        )
    else:
        tenure_series = pd.Series("Unknown", index=work.index)

    by_tenure = _rate_table(work, tenure_series.rename("tenure"))
    by_year = _rate_table(work, work["disbursement_date"].dt.year.rename("year"))
    by_month = _rate_table(
        work, work["disbursement_date"].dt.to_period("M").astype(str).rename("month")
    )
    by_lender = _rate_table(work, "lender_id").sort_values("count", ascending=False).reset_index(drop=True)
    by_loan_type = _rate_table(work, "loan_type").sort_values("count", ascending=False).reset_index(drop=True)

    period = pd.Series(
        np.where(work["disbursement_date"] < pd.Timestamp(C.TEMPORAL_CUTOFF), "before_2023", "2023_onwards"),
        index=work.index,
        name="period",
    )
    main_lender_id = str(work["lender_id"].value_counts().index[0]) if n_rows > 0 else "267278"
    shift_summary = (
        work.assign(
            period=period,
            is_main_lender=(work["lender_id"] == main_lender_id).astype(float),
            is_new=(tenure_series == "New Loan").astype(float),
        )
        .groupby("period")
        .agg(
            n_loans=("target", "count"),
            default_rate_pct=("target", lambda s: float(s.mean() * 100.0)),
            main_lender_share=("is_main_lender", "mean"),
            new_borrower_share=("is_new", "mean"),
            mean_duration=("duration", "mean"),
        )
        .round(3)
        .reset_index()
    )

    shift_by_lender = (
        work.assign(period=period)
        .groupby(["period", "lender_id"])["target"]
        .agg(count="count", defaults="sum", default_rate="mean")
        .reset_index()
    )
    shift_by_lender["default_rate_pct"] = (shift_by_lender["default_rate"] * 100.0).round(3)

    return {
        "n_rows": n_rows,
        "n_defaults": n_defaults,
        "default_rate": default_rate,
        "missing_counts": missing_counts,
        "extra_pct_by_target": extra_pct_by_target,
        "by_tenure": by_tenure,
        "by_year": by_year,
        "by_month": by_month,
        "by_lender": by_lender,
        "by_loan_type": by_loan_type,
        "shift_2023_summary": shift_summary,
        "shift_2023_by_lender": shift_by_lender,
    }
