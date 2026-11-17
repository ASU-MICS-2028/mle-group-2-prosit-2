"""Score every prediction file in preds/ and write the comparison tables to reports/tables/.

    python -m scripts.evaluate              (or: make eval)
"""
from __future__ import annotations

import pandas as pd

from src import config as C
from src.evaluation import cost_matrix_threshold, cost_sensitivity_table, measure_cost_inputs, summarize_predictions_dir


def main() -> None:
    C.TABLES.mkdir(parents=True, exist_ok=True)
    cost = cost_matrix_threshold()
    print(
        f"Cost matrix ({cost['currency']}): C_FP={cost['c_fp']:,.0f}  C_FN={cost['c_fn']:,.0f}  "
        f"-> decline if P(default) > {cost['threshold']:.4f}   (configs/cost_matrix.yaml, draft)"
    )
    if C.TRAIN_CSV.exists():
        measured = measure_cost_inputs(pd.read_csv(C.TRAIN_CSV))
        print("Measured from Train.csv:", {k: round(v, 4) for k, v in measured.items()})

    sens = cost_sensitivity_table()
    sens.to_csv(C.TABLES / "cost_sensitivity.csv", index=False)
    print("\nThreshold under alternative assumptions:")
    print(sens[["recovery_rate", "future_loans", "c_fp", "c_fn", "threshold"]].round(4).to_string(index=False))

    summary = summarize_predictions_dir(threshold=cost["threshold"])
    if summary.empty:
        print("\nNo labelled prediction files in preds/. Run `make train` first.")
        return
    for task, part in summary.groupby("task"):
        part = part.dropna(axis=1, how="all").sort_values(["scheme", "model"])
        part.to_csv(C.TABLES / f"{task}_model_summary.csv", index=False)
        print(f"\n=== {task} ===")
        print(part.drop(columns=["task"]).round(4).to_string(index=False))
    print(f"\nTables saved to {C.TABLES.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    main()
