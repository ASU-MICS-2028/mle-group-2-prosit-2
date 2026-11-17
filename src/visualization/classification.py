"""Task 2 plots: precision-recall curves, reliability diagram and threshold sweep."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from ..evaluation.metrics import reliability_table


def _axes(ax, figsize=(5.5, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def plot_pr_curves(y_true, probs: dict[str, np.ndarray], ax=None):
    """One precision-recall curve per model; the dashed line is the no-skill default rate."""
    fig, ax = _axes(ax)
    for name, p in probs.items():
        precision, recall, _ = precision_recall_curve(y_true, p)
        ax.plot(recall, precision, label=name)
    ax.axhline(np.mean(y_true), ls="--", c="grey", lw=1, label="no skill")
    ax.set(xlabel="Recall (defaults caught)", ylabel="Precision (flagged that defaulted)", title="Precision-recall")
    ax.legend(frameon=False)
    return fig


def plot_reliability(y_true, probs: dict[str, np.ndarray], n_bins: int = 10, ax=None):
    """Predicted vs actual default rate per quantile bin; on the diagonal = calibrated."""
    fig, ax = _axes(ax)
    top = 0.0
    for name, p in probs.items():
        table, ece = reliability_table(y_true, p, n_bins=n_bins)
        ax.plot(table["mean_predicted"], table["fraction_positive"], "o-", label=f"{name} (ECE {ece:.3f})")
        top = max(top, table["mean_predicted"].max(), table["fraction_positive"].max())
    ax.plot([0, top], [0, top], ls="--", c="grey", lw=1)
    ax.set(xlabel="Predicted default rate", ylabel="Actual default rate", title="Reliability")
    ax.legend(frameon=False)
    return fig


def plot_threshold_sweep(sweep: pd.DataFrame, threshold: float | None = None, ax=None):
    """Expected cost per loan and approval rate across thresholds, with the chosen operating point."""
    fig, ax = _axes(ax, figsize=(6.5, 4))
    ax.plot(sweep["threshold"], sweep["expected_cost_per_loan"], label="expected cost per loan")
    ax.set(xlabel="Decline if P(default) >", ylabel="Expected cost per loan")
    ax2 = ax.twinx()
    ax2.plot(sweep["threshold"], sweep["selection_rate"], c="tab:orange", label="approval rate")
    ax2.set_ylabel("Approval rate")
    if threshold is not None:
        ax.axvline(threshold, ls="--", c="grey", lw=1)
    ax.set_title("Threshold sweep")
    return fig
