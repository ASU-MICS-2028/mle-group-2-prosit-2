"""Reusable plots. Each function draws on an optional matplotlib Axes and returns the Figure;
`save_figure` writes it to reports/figures/ so the report and slides use the same images."""
from __future__ import annotations

from pathlib import Path

import matplotlib

from .. import config as C

SAVE_DPI = 200  # print quality: a 6.5-inch figure is 1,300 px wide

# One house style for every figure in the report and slides.
matplotlib.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": SAVE_DPI,
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(
            color=["#1f5f9f", "#e07b24", "#2e8b57", "#c0392b", "#7d5ba6", "#8c6d31", "#d4589c", "#5f6b6d"]
        ),
    }
)


def save_figure(fig, name: str) -> Path:
    """Save to reports/figures/<name>.png at print resolution with a tight layout."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / f"{name}.png"
    try:
        fig.tight_layout()
    except Exception:  # some layouts (twin axes, colour bars) are already final
        pass
    fig.savefig(path, bbox_inches="tight", dpi=SAVE_DPI)
    return path


from .classification import (  # noqa: E402
    plot_calibration_by_group,
    plot_confusion,
    plot_cost_sensitivity,
    plot_interval_forest,
    plot_pr_curves,
    plot_reliability,
    plot_roc_curves,
    plot_scheme_comparison,
    plot_threshold_sweep,
)
from .fairness import plot_group_selection_rates  # noqa: E402
from .interpretation import plot_coefficients, plot_shap_importance  # noqa: E402
from .product import (  # noqa: E402
    plot_counts,
    plot_decision_mix,
    plot_latency,
    plot_limit_vs_request,
    plot_score_distribution,
)
from .regression import (  # noqa: E402
    plot_amount_distribution,
    plot_error_by_decile,
    plot_pred_vs_actual,
    plot_residuals_vs_fitted,
)
from .transfer import plot_domain_scores, plot_feature_shift, plot_shift_drivers  # noqa: E402

__all__ = [
    "plot_amount_distribution",
    "plot_calibration_by_group",
    "plot_coefficients",
    "plot_confusion",
    "plot_cost_sensitivity",
    "plot_counts",
    "plot_decision_mix",
    "plot_domain_scores",
    "plot_error_by_decile",
    "plot_feature_shift",
    "plot_group_selection_rates",
    "plot_interval_forest",
    "plot_latency",
    "plot_limit_vs_request",
    "plot_pr_curves",
    "plot_pred_vs_actual",
    "plot_reliability",
    "plot_residuals_vs_fitted",
    "plot_roc_curves",
    "plot_scheme_comparison",
    "plot_score_distribution",
    "plot_shap_importance",
    "plot_shift_drivers",
    "plot_threshold_sweep",
    "save_figure",
]
