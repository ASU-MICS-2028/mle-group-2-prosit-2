"""Feature-group ablation study (#52): retrain deployed and tree models without each feature group.

    python -m scripts.run_ablation              (or: make ablation)

Outputs:
  - reports/tables/ablation.csv                 PR-AUC, ROC-AUC and delta vs full D-18 model
  - reports/figures/interp_ablation.png         grouped bar chart across random, by-customer and temporal splits
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src import config as C  # noqa: E402
from src.console import console  # noqa: E402
from src.evaluation import run_ablation  # noqa: E402
from src.processing import load_features  # noqa: E402
from src.visualization import plot_ablation, save_figure  # noqa: E402


def main() -> None:
    console.setup()
    console.title("Ablation study (#52): feature groups on fold_strat, fold_group and temporal")
    df = load_features()
    with console.step("Retraining logreg (deployed, D-29) and lightgbm across 7 feature subsets x 3 schemes"):
        ablation = run_ablation(df, models=("logreg", "lightgbm"), schemes=("fold_strat", "fold_group", "temporal"))

    C.TABLES.mkdir(parents=True, exist_ok=True)
    out_csv = C.TABLES / "ablation.csv"
    ablation.to_csv(out_csv, index=False)

    console.table(ablation.round(4), title="Ablation results (delta = variant - full D-18 model)", max_rows=50)
    console.info("D-18 seed result: removing the 4 macro columns raised lit_mlp by-time ROC-AUC from 0.48 to 0.65.")

    fig = plot_ablation(ablation, metric="delta_pr_auc")
    save_figure(fig, "interp_ablation")
    plt.close(fig)

    fig_roc = plot_ablation(ablation, metric="delta_roc_auc")
    save_figure(fig_roc, "interp_ablation_roc")
    plt.close(fig_roc)

    console.done(f"saved {C.rel(out_csv)} and reports/figures/interp_ablation*.png")


if __name__ == "__main__":
    main()
