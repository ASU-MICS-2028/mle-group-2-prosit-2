"""Score every prediction file in preds/ and write the comparison tables to reports/tables/.

    python -m scripts.evaluate              (or: make eval)
"""
from __future__ import annotations

import pandas as pd

from src import config as C
from src.console import console
from src.evaluation import cost_matrix_threshold, cost_sensitivity_table, measure_cost_inputs, summarize_predictions_dir

T2_COLS = ["model", "scheme", "roc_auc", "pr_auc", "precision", "recall", "f1", "selection_rate"]
T1_COLS = ["model", "scheme", "rmse_log", "mae_log", "r2_log", "rmse_orig", "mae_orig", "median_abs_error_orig"]


def main() -> None:
    console.title("Evaluate")
    C.TABLES.mkdir(parents=True, exist_ok=True)
    with console.step("cost matrix and sensitivity (configs/cost_matrix.yaml)"):
        cost = cost_matrix_threshold()
        sens = cost_sensitivity_table()
        sens.to_csv(C.TABLES / "cost_sensitivity.csv", index=False)
        measured = measure_cost_inputs(pd.read_csv(C.TRAIN_CSV)) if C.TRAIN_CSV.exists() else None
    console.info(f"C_FP {cost['c_fp']:,.0f} {cost['currency']}   C_FN {cost['c_fn']:,.0f} {cost['currency']}   "
                 f"-> decline if P(default) > {cost['threshold']:.4f}")
    if measured:
        console.info(f"measured: median loan {measured['principal_median']:,.0f}, mean fee on repaid loans "
                     f"{measured['fee_rate_mean_repaid']:.2%}, median loans per customer {measured['loans_per_customer_median']:.0f}")

    with console.step(f"score every file in {C.rel(C.PREDS)} (Task 2 calibrated out of fold)"):
        summary = summarize_predictions_dir(threshold=cost["threshold"])
    if summary.empty:
        console.done("no labelled prediction files yet: run `make train` first")
        return
    for task, cols, key in (("task2", T2_COLS, "pr_auc"), ("task1", T1_COLS, "r2_log")):
        part = summary[summary["task"] == task].dropna(axis=1, how="all")
        if part.empty:
            continue
        part.sort_values(["scheme", "model"]).to_csv(C.TABLES / f"{task}_model_summary.csv", index=False)
        shown = part[[c for c in cols if c in part]].sort_values(["scheme", key], ascending=[True, False])
        console.table(shown, f"{task} ({'default' if task == 'task2' else 'loan amount, USD'})", max_rows=40)
    console.table(sens[["recovery_rate", "future_loans", "c_fp", "c_fn", "threshold"]], "threshold under alternative costs")
    console.done(f"tables in {C.rel(C.TABLES)}: task2_model_summary.csv, task1_model_summary.csv, cost_sensitivity.csv")


if __name__ == "__main__":
    main()
