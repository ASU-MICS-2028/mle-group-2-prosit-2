"""What an application must contain, and how it becomes the model's feature row.

Borrower history (past loans, past defaults) comes from NkwaPay's own records, not the applicant.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .. import config as C
from ..processing import get_feature_lists

# field: (type, description, required)
REQUIRED_APPLICATION_FIELDS: dict[str, tuple[str, str, bool]] = {
    "customer_id": ("str", "Borrower identifier", True),
    "country_id": ("str", "'Kenya' or 'Ghana'", True),
    "lender_id": ("str", "Funding lender identifier", True),
    "loan_type": ("str", "Loan product, e.g. 'Type_1'", True),
    "Total_Amount": ("float", "Requested principal in local currency (KES or GHS), > 0", True),
    "duration": ("float", "Requested term in days, > 0", True),
    "disbursement_date": ("str", "Application date, YYYY-MM-DD", True),
    "New_versus_Repeat": ("str", "'New Loan' or 'Repeat Loan'", True),
    "Lender_portion_Funded": ("float", "Lender's share of the loan, 0-1 (default 0.30, the Train median)", False),
    "n_lenders_on_loan": ("int", "Lenders funding this loan (default 1)", False),
    "n_past_loans": ("int", "From records: earlier loans (default 0)", False),
    "days_since_last_loan": ("float", "From records: days since the last loan", False),
    "n_past_labelled": ("int", "From records: earlier loans already past their due date (default 0)", False),
    "n_past_defaults": ("int", "From records: of those, how many defaulted (default 0)", False),
    "days_since_last_default": ("float", "From records: days since the last default's due date", False),
}


@lru_cache(maxsize=1)
def _fx_table() -> dict[tuple[str, int], float]:
    """(country, year) -> local currency per USD, from economic_indicators.csv."""
    m = pd.read_csv(C.MACRO_CSV)
    row = m[m["Indicator"] == "Official exchange rate (LCU per US$, period average)"]
    table = {}
    for _, r in row.iterrows():
        for col in [c for c in m.columns if c.startswith("YR")]:
            if pd.notna(r[col]):
                table[(str(r["Country"]), int(col[2:]))] = float(r[col])
    return table


def exchange_rate(country: str, disb_year: int) -> float:
    """Previous year's average rate, as in features.py (annual figures arrive after year-end).

    Falls back to the latest year available for the country, for applications after the data ends.
    """
    table = _fx_table()
    years = sorted(y for (c, y) in table if c == country)
    if not years:
        raise ValueError(f"No exchange rate for country '{country}' in {C.MACRO_CSV.name}.")
    year = disb_year - 1 if (country, disb_year - 1) in table else years[-1]
    return table[(country, year)]


def _optional_float(app: dict[str, Any], key: str) -> float:
    value = app.get(key)
    return float(value) if value is not None else np.nan


def prepare_single_application(app: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    """Validate one application; return (Task 2 feature row, exchange rate used)."""
    required = [k for k, (_, _, req) in REQUIRED_APPLICATION_FIELDS.items() if req]
    missing = [k for k in required if app.get(k) is None or str(app[k]).strip() == ""]
    if missing:
        raise ValueError(f"Missing required application fields: {missing}")
    amount, duration = float(app["Total_Amount"]), float(app["duration"])
    if amount <= 0 or duration <= 0:
        raise ValueError("Total_Amount and duration must be positive.")

    disb = pd.Timestamp(app["disbursement_date"])
    fx = exchange_rate(str(app["country_id"]), int(disb.year))
    n_labelled = int(app.get("n_past_labelled", 0))
    n_defaults = int(app.get("n_past_defaults", 0))

    row = {
        "lender_id": str(app["lender_id"]),
        "loan_type": str(app["loan_type"]),
        "log_amount_usd": float(np.log1p(amount / fx)),
        "duration": duration,
        "Lender_portion_Funded": float(app.get("Lender_portion_Funded", 0.30)),
        "is_new_borrower": int(str(app["New_versus_Repeat"]) == "New Loan"),
        "n_lenders_on_loan": int(app.get("n_lenders_on_loan", 1)),
        "n_past_loans": int(app.get("n_past_loans", 0)),
        "days_since_last_loan": _optional_float(app, "days_since_last_loan"),
        "n_past_labelled": n_labelled,
        "n_past_defaults": n_defaults,
        "past_default_rate": n_defaults / n_labelled if n_labelled > 0 else np.nan,
        "days_since_last_default": _optional_float(app, "days_since_last_default"),
        "disb_month": int(disb.month),
        "disb_dow": int(disb.day_of_week),
        "disb_dom": int(disb.day),
    }
    num, cat = get_feature_lists("task2")
    return pd.DataFrame([row])[num + cat], fx
