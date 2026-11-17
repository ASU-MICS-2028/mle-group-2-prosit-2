"""Time safety of the borrower-history features (src/features.py), on a hand-made customer.

A past default may only be counted once the earlier loan's DUE date has passed (D-13).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import borrower_history


def _loan(id_, loan, disb, due, target, cust=1):
    return {"ID": id_, "customer_id": cust, "tbl_loan_id": loan, "disbursement_date": pd.Timestamp(disb),
            "due_date": pd.Timestamp(due), "target": target}


def test_history_counts_only_what_was_known_at_disbursement() -> None:
    df = pd.DataFrame(
        [
            _loan("a", 1, "2022-01-01", "2022-01-08", 1.0),  # defaults; outcome known from Jan 8
            _loan("b", 2, "2022-01-05", "2022-01-12", 0.0),  # before A is due: A's default is NOT known yet
            _loan("c", 3, "2022-01-10", "2022-01-17", 0.0),  # after A is due: A's default is known
            _loan("c2", 3, "2022-01-10", "2022-01-17", 0.0),  # second lender row of loan 3: same history
            _loan("d", 4, "2022-01-10", "2022-01-17", np.nan, cust=2),  # another customer, Test row
        ]
    )
    h = borrower_history(df).set_index("ID")
    assert h.loc["a", "n_past_loans"] == 0 and np.isnan(h.loc["a", "days_since_last_loan"])
    assert h.loc["b", "n_past_loans"] == 1 and h.loc["b", "n_past_defaults"] == 0 and h.loc["b", "n_past_labelled"] == 0
    assert h.loc["c", "n_past_loans"] == 2 and h.loc["c", "n_past_defaults"] == 1 and h.loc["c", "n_past_labelled"] == 1
    assert h.loc["c", "days_since_last_default"] == 2  # Jan 10 - Jan 8
    assert h.loc["c", "past_default_rate"] == 1.0
    assert h.loc["c"].equals(h.loc["c2"]), "both lender rows of one loan get the same history"
    assert h.loc["d", "n_past_loans"] == 0, "history never crosses customers"


def test_same_day_loans_do_not_see_each_other() -> None:
    df = pd.DataFrame([_loan("x", 1, "2022-03-01", "2022-03-08", 0.0), _loan("y", 2, "2022-03-01", "2022-03-08", 1.0)])
    h = borrower_history(df).set_index("ID")
    assert (h["n_past_loans"] == 0).all(), "backward merge_asof must exclude exact matches"
