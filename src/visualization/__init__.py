"""Reusable plots. Each function draws on an optional matplotlib Axes and returns the Figure;
`save_figure` writes it to reports/figures/ so the report and slides use the same images."""
from __future__ import annotations

from pathlib import Path

import matplotlib

from .. import config as C

matplotlib.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig, name: str) -> Path:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


from .classification import plot_pr_curves, plot_reliability, plot_threshold_sweep  # noqa: E402
from .fairness import plot_group_selection_rates  # noqa: E402
from .regression import plot_amount_distribution, plot_residuals_vs_fitted  # noqa: E402
from .transfer import plot_feature_shift  # noqa: E402

__all__ = [
    "plot_amount_distribution",
    "plot_feature_shift",
    "plot_group_selection_rates",
    "plot_pr_curves",
    "plot_reliability",
    "plot_residuals_vs_fitted",
    "plot_threshold_sweep",
    "save_figure",
]
