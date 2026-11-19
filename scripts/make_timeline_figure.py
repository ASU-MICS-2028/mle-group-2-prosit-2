"""Draw the Part A timeline of one loan and where each Train.csv column sits on it.

Run:  python -m scripts.make_timeline_figure      (writes reports/figures/part_a_loan_timeline.png)

Facts used (all checked in docs/feature_audit.md and reports/EDA_REPORT.md):
  - there is no application or approval date in the data; `disbursement_date` is the closest column;
  - `due_date` = `disbursement_date` + `duration` on 100% of rows; 95% of loans last 7 days;
  - `target` is only known once the loan is due.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config as C  # noqa: E402

OUT = C.ROOT / "reports" / "figures" / "part_a_loan_timeline.png"
BLUE, RED, GREY, GREEN = "#1f4e79", "#b03a2e", "#6b7280", "#2e7d32"

EVENTS = [  # x position, label, sublabel, is the data column known?
    (0.0, "Application", "not in the data", False),
    (2.2, "Approval", "not in the data", False),
    (4.4, "Disbursement", "column:\ndisbursement_date", True),
    (8.0, "Due date", "= disbursement_date + duration\n(date known at the decision;\n7 days for 95% of loans)", True),
    (11.0, "Repaid or defaulted", "column: target\n(the label)", True),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(-1.0, 12.4)
    ax.set_ylim(-3.3, 3.3)
    ax.axis("off")

    ax.axhline(0, color=GREY, lw=2, zorder=1)
    ax.axvspan(-1.0, 4.4, ymin=0.08, ymax=0.92, color=GREEN, alpha=0.08, zorder=0)
    ax.axvspan(4.4, 12.4, ymin=0.08, ymax=0.92, color=RED, alpha=0.07, zorder=0)
    ax.plot([4.4, 4.4], [-3.0, 2.75], color=BLUE, lw=2.5, ls="--", zorder=2)
    ax.text(4.4, 3.05, "THE DECISION MOMENT: score the borrower before this line",
            ha="center", va="center", fontsize=11, color=BLUE, weight="bold")
    ax.text(1.7, 2.35, "Known when we decide: usable as features", ha="center", fontsize=10, color=GREEN, weight="bold")
    ax.text(8.4, 2.35, "Not known yet: anything here is invalid as a feature", ha="center", fontsize=10, color=RED, weight="bold")

    for x, name, sub, in_data in EVENTS:
        ax.scatter([x], [0], s=150, zorder=3, color=BLUE if in_data else "white",
                   edgecolor=BLUE if in_data else GREY, linewidth=2, linestyle="-" if in_data else "--")
        ax.text(x, 0.55, name, ha="center", fontsize=11, weight="bold", color="#111827",
                bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.9), zorder=4)
        ax.text(x, -0.5, sub, ha="center", va="top", fontsize=8.5, color=GREY,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9), zorder=4)

    ax.text(1.5, -1.9,
            "Known at the decision (features):\ncountry_id, lender_id, loan_type, Total_Amount, duration,\n"
            "Lender_portion_Funded, New_versus_Repeat, borrower history\n(earlier loans already due), calendar and macro derived from the date",
            ha="center", va="top", fontsize=9, color=GREEN)
    ax.text(8.4, -1.9,
            "Recorded after the loan is issued (invalid):\nTotal_Amount_to_Repay and Lender_portion_to_be_repaid\n"
            "(a loan that defaults owes a median 22% extra, repaid loans 0.8%),\nand target itself, which only exists once the loan is due",
            ha="center", va="top", fontsize=9, color=RED)
    ax.text(5.7, -3.15, "Keys and groupers, not features: ID, customer_id, tbl_loan_id.   "
            "Exact copies, dropped: due_date, Amount_Funded_By_Lender.",
            ha="center", fontsize=8.5, color=GREY)

    ax.set_title("One NkwaPay loan: when the decision is made and what is known then", fontsize=13, weight="bold", pad=6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Saved {C.rel(OUT)}")


if __name__ == "__main__":
    main()
