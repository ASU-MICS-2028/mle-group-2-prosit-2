"""Latency of the scoring service over repeated single-application calls (brief: median and p95)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .scorer import score_application

SAMPLE_APPLICATION = {
    "customer_id": "BENCH_001",
    "country_id": "Kenya",
    "lender_id": "267278",
    "loan_type": "Type_1",
    "Total_Amount": 5000.0,
    "duration": 7.0,
    "disbursement_date": "2023-02-15",
    "New_versus_Repeat": "Repeat Loan",
    "n_past_loans": 4,
    "days_since_last_loan": 21.0,
    "n_past_labelled": 3,
    "n_past_defaults": 0,
}


def benchmark_latency(n_calls: int = 1000, app: dict[str, Any] | None = None, _bundle=None) -> dict[str, float]:
    """Median, p95, mean, min and max milliseconds per call, after one warm-up call."""
    app = app or SAMPLE_APPLICATION
    score_application(app, _bundle=_bundle)  # warm-up: loads artifacts and the exchange-rate table
    ms = np.array([score_application(app, _bundle=_bundle)["latency_ms"] for _ in range(n_calls)])
    return {
        "n_calls": n_calls,
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
    }
