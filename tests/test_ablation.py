"""Tests for the feature-group ablation study (#52)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src import config as C  # noqa: E402
from src.evaluation import ABLATION_GROUPS, run_ablation  # noqa: E402
from src.visualization import plot_ablation  # noqa: E402


def test_ablation_groups_partition_all_task2_features() -> None:
    grouped = [col for cols in ABLATION_GROUPS.values() for col in cols]
    expected = list(C.NUM_FEATURES_T2) + list(C.CAT_FEATURES)
    assert len(grouped) == len(set(grouped)), "feature listed in more than one ablation group"
    assert set(grouped) == set(expected), "ablation groups must cover every Task 2 input feature"


def test_run_ablation_and_plot_on_synthetic_data(synthetic_features) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    df = synthetic_features.copy()
    for col in C.MACRO_FEATURES:
        df[col] = rng.normal(loc=10.0, scale=2.0, size=len(df))

    res = run_ablation(df, models=("logreg",), schemes=("fold_group", "temporal"))
    expected_variants = {"full", "+ macro"} | {f"- {g}" for g in ABLATION_GROUPS}
    assert set(res["variant"]) == expected_variants
    assert set(res["scheme"]) == {"fold_group", "temporal"}
    full_rows = res[res["variant"] == "full"]
    assert (full_rows["delta_pr_auc"] == 0.0).all()
    assert (full_rows["delta_roc_auc"] == 0.0).all()
    assert res["roc_auc"].notna().all() and res["pr_auc"].notna().all()

    fig = plot_ablation(res, metric="delta_pr_auc")
    assert fig is not None
    plt.close(fig)
