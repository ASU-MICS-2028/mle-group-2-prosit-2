"""Score sample applications with the saved scorer and benchmark latency over 1,000 calls.

    python -m scripts.score                 (or: make score; needs `make train` first)
"""
from __future__ import annotations

import json

import pandas as pd

from src import config as C
from src.serving import SAMPLE_APPLICATION, benchmark_latency, score_application

SAMPLES = {
    "kenya_repeat": SAMPLE_APPLICATION,
    "kenya_new": {**SAMPLE_APPLICATION, "customer_id": "KEN_NEW", "New_versus_Repeat": "New Loan",
                  "n_past_loans": 0, "n_past_labelled": 0, "days_since_last_loan": None},
    "ghana_new": {"customer_id": "GHA_NEW", "country_id": "Ghana", "lender_id": "UNSEEN_GH_LENDER",
                  "loan_type": "Type_3", "Total_Amount": 350.0, "duration": 30.0,
                  "disbursement_date": "2023-02-15", "New_versus_Repeat": "New Loan"},
}


def main() -> None:
    for name, app in SAMPLES.items():
        result = score_application(app)
        print(f"\n--- {name} ---")
        print(json.dumps({k: v for k, v in result.items() if k != "latency_ms"}, indent=2))
    stats = benchmark_latency(n_calls=1000)
    print("\nLatency over 1,000 calls (ms):", {k: round(v, 2) for k, v in stats.items()})
    C.TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(C.TABLES / "scorer_latency.csv", index=False)


if __name__ == "__main__":
    main()
