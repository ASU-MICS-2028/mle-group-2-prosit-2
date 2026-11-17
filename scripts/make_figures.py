"""Draw the report figures from the saved predictions into reports/figures/.

    python -m scripts.make_figures          (or: make figures; needs `make train` first)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src.evaluation import calibrate_oof, cost_matrix_threshold, sweep_thresholds  # noqa: E402
from src.processing import load_features  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_amount_distribution,
    plot_feature_shift,
    plot_pr_curves,
    plot_reliability,
    plot_residuals_vs_fitted,
    plot_threshold_sweep,
    save_figure,
)


def _load(task: str, model: str, scheme: str) -> pd.DataFrame | None:
    path = C.PREDS / f"{task}_{model}_{scheme}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def main() -> None:
    saved = []
    df = load_features()

    # Task 2: PR curves, reliability after calibration, cost-threshold sweep (GroupKFold predictions)
    t2 = {m: p for m in ("logreg", "lit_mlp", "random_forest", "lightgbm") if (p := _load("task2", m, "fold_group")) is not None}
    if t2:
        y = next(iter(t2.values()))["y_true"].to_numpy()
        saved.append(save_figure(plot_pr_curves(y, {m: p["y_pred"].to_numpy() for m, p in t2.items()}), "t2_pr_curves"))
        cal = {m: calibrate_oof(p["y_true"], p["y_pred"], p["fold"]) for m, p in t2.items()}
        saved.append(save_figure(plot_reliability(y, cal), "t2_reliability_calibrated"))
        cost = cost_matrix_threshold()
        first = next(iter(cal))
        sweep = sweep_thresholds(y, cal[first], c_fp=cost["c_fp"], c_fn=cost["c_fn"])
        saved.append(save_figure(plot_threshold_sweep(sweep, cost["threshold"]), f"t2_threshold_sweep_{first}"))

    # Task 1: amount distribution and residuals
    train = df[df["split"] == "train"]
    saved.append(save_figure(plot_amount_distribution(np.expm1(train["log_amount_usd"]), title="Loan amount, USD (log)"),
                             "t1_amount_distribution"))
    for m in ("ridge", "lightgbm_reg"):
        if (p := _load("task1", m, "fold_group")) is not None:
            saved.append(save_figure(plot_residuals_vs_fitted(p["y_true"], p["y_pred"], title=f"{m}: residuals"),
                                     f"t1_residuals_{m}"))

    # Part C: Kenya vs Ghana on the Test rows
    test = df[df["split"] == "test"]
    for feat in ("log_amount_usd", "duration", "n_past_loans"):
        saved.append(save_figure(plot_feature_shift(test, feat), f"shift_{feat}"))

    plt.close("all")
    print("\n".join(str(p.relative_to(C.ROOT)) for p in saved))


if __name__ == "__main__":
    main()
