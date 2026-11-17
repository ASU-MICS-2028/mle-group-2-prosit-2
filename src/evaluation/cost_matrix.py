"""Money-denominated cost matrix and the threshold it implies (Part A; issue #11, owner Henry).

Values live in configs/cost_matrix.yaml. Principal and fee are measured from Train.csv
(`measure_cost_inputs` recomputes them); recovery, collection cost and future loans are assumptions.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .. import config as C

_COST_KEYS = ("principal", "fee_rate", "future_loans", "recovery_rate", "collection_cost")


def load_cost_config() -> dict[str, Any]:
    with open(C.COST_MATRIX_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cost_matrix_threshold(**overrides: float) -> dict[str, float]:
    """C_FP, C_FN and the cost-minimising threshold p* = C_FP / (C_FP + C_FN).

    With calibrated probabilities, declining costs less than approving whenever p * C_FN > (1 - p) * C_FP.
    Any value from the config can be overridden, e.g. cost_matrix_threshold(recovery_rate=0.4).
    """
    cfg = load_cost_config()
    v = {k: float(overrides.get(k, cfg[k])) for k in _COST_KEYS}
    c_fp = v["principal"] * v["fee_rate"] * v["future_loans"]
    c_fn = v["principal"] * (1.0 - v["recovery_rate"]) + v["collection_cost"]
    return {
        **v,
        "currency": cfg.get("currency", "KES"),
        "c_fp": c_fp,
        "c_fn": c_fn,
        "cost_ratio": c_fn / max(c_fp, 1e-9),
        "threshold": c_fp / (c_fp + c_fn),
    }


def cost_sensitivity_table() -> pd.DataFrame:
    """Threshold under every combination of the alternative assumptions in the config."""
    grid = load_cost_config().get("sensitivity", {})
    keys = list(grid)
    rows = [cost_matrix_threshold(**dict(zip(keys, combo))) for combo in itertools.product(*grid.values())]
    return pd.DataFrame(rows)


def measure_cost_inputs(train_raw: pd.DataFrame) -> dict[str, float]:
    """Recompute the measured inputs from raw Train.csv, to check the config against the data.

    The fee comes from Total_Amount_to_Repay on *repaid* loans only: for those it is the agreed fee.
    (On defaulted loans it includes penalties, which is why it is a leak and never a model feature.)
    """
    repaid = train_raw[train_raw["target"] == 0]
    markup = repaid["Total_Amount_to_Repay"] / repaid["Total_Amount"].replace(0, np.nan) - 1.0
    loans = train_raw.drop_duplicates("tbl_loan_id")
    return {
        "principal_median": float(train_raw["Total_Amount"].median()),
        "fee_rate_mean_repaid": float(markup.mean()),
        "fee_rate_median_repaid": float(markup.median()),
        "loans_per_customer_median": float(loans.groupby("customer_id").size().median()),
    }


def measure_market_cost_inputs(raw: pd.DataFrame, country: str = "Ghana") -> dict[str, float]:
    """Measure principal and fee for a market that has no labels yet (e.g. Ghana in Test.csv; #11, D-22).

    Without labels we cannot keep only repaid loans, so the fee is the *median* markup on the
    market's most common loan term: the agreed fee schedule, robust to the few defaulted loans
    whose amount owed includes penalties. Repayment columns stay out of every model (leak).
    """
    df = raw[raw["country_id"] == country]
    if df.empty:
        raise ValueError(f"no loans for country {country!r}")
    term = int(df["duration"].mode().iloc[0])
    on_term = df[df["duration"] == term]
    markup = on_term["Total_Amount_to_Repay"] / on_term["Total_Amount"].replace(0, np.nan) - 1.0
    return {
        "principal_median": float(df["Total_Amount"].median()),
        "fee_rate_median_on_term": float(markup.median()),
        "term_days": float(term),
        "share_on_term": float(len(on_term) / len(df)),
        "n_loans": float(len(df)),
    }

