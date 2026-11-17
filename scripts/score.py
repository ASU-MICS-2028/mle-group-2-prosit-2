"""Score sample applications with the saved scorer and benchmark latency over 1,000 calls.

    python -m scripts.score                 (or: make score; needs `make train` first)
"""
from __future__ import annotations

import pandas as pd

from src import config as C
from src.console import console
from src.serving import SAMPLE_APPLICATION, benchmark_latency, score_application

SAMPLES = {
    "Kenya, repeat borrower": SAMPLE_APPLICATION,
    "Kenya, first loan": {**SAMPLE_APPLICATION, "customer_id": "KEN_NEW", "New_versus_Repeat": "New Loan",
                          "n_past_loans": 0, "n_past_labelled": 0, "days_since_last_loan": None},
    "Ghana, first loan": {"customer_id": "GHA_NEW", "country_id": "Ghana", "lender_id": "UNSEEN_GH_LENDER",
                          "loan_type": "Type_3", "Total_Amount": 350.0, "duration": 30.0,
                          "disbursement_date": "2023-02-15", "New_versus_Repeat": "New Loan"},
}


def main() -> None:
    console.title("Score sample applications")
    for name, app in SAMPLES.items():
        with console.step(name):
            r = score_application(app)
        cur = r["currency"]
        console.info(f"P(default) {r['probability_of_default']:.1%} (threshold {r['threshold']:.1%})  ->  "
                     f"{r['decision'].upper()}   requested {r['requested_amount']:,.0f} {cur}, limit {r['recommended_limit']:,.0f} {cur}")
        for reason in r["top_3_reasons"]:
            console.info(f"  · {reason}")
        for flag in r["review_flags"]:
            console.info(f"  ! review: {flag}")

    console.title("Latency")
    with console.step("1,000 single-application calls"):
        stats = benchmark_latency(n_calls=1000)
    console.info(f"median {stats['median_ms']:.0f} ms   p95 {stats['p95_ms']:.0f} ms   max {stats['max_ms']:.0f} ms")
    C.TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(C.TABLES / "scorer_latency.csv", index=False)
    console.done(f"latency saved to {C.rel(C.TABLES / 'scorer_latency.csv')}")


if __name__ == "__main__":
    main()
