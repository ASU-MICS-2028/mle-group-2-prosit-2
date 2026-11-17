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


def plot_roc_curves(y_true, probs: dict[str, np.ndarray], ax=None):
    """ROC curves; the diagonal is a coin flip."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = _axes(ax)
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xlabel="False positive rate (good borrowers declined)", ylabel="True positive rate (defaults caught)", title="ROC")
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_confusion(tn: int, fp: int, fn: int, tp: int, title: str = "Confusion matrix", ax=None):
    """Counts with row percentages, laid out as the lender sees them."""
    fig, ax = _axes(ax, figsize=(4.6, 4))
    m = np.array([[tn, fp], [fn, tp]])
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:,}\n({v / m[i].sum():.0%} of row)", ha="center", va="center", fontsize=9)
    ax.set_xticks([0, 1], ["approved", "declined"])
    ax.set_yticks([0, 1], ["repaid", "defaulted"])
    ax.set_title(title)
    return fig


def plot_interval_forest(table: pd.DataFrame, value: str, low: str, high: str, label: str, title: str, ref: float | None = None, ax=None):
    """Point estimate and 95% interval per row; overlapping intervals mean the gap is not certain."""
    fig, ax = _axes(ax, figsize=(6.5, 0.45 * len(table) + 1.5))
    t = table.sort_values(value).reset_index(drop=True)
    y = np.arange(len(t))
    ax.errorbar(t[value], y, xerr=[t[value] - t[low], t[high] - t[value]], fmt="o", capsize=3)
    ax.set_yticks(y, t[label])
    if ref is not None:
        ax.axvline(ref, ls="--", c="grey", lw=1, label="baseline + noise floor")
        ax.legend(frameon=False, fontsize=8)
    ax.set(xlabel=value.replace("_", " "), title=title)
    return fig


def plot_scheme_comparison(table: pd.DataFrame, metric: str = "pr_auc", ax=None):
    """One group of bars per model, one bar per validation scheme: the gap is how much a split flatters."""
    fig, ax = _axes(ax, figsize=(9, 4))
    schemes = [s for s in ("fold_strat", "fold_group", "temporal") if s in set(table["scheme"])]
    labels = {"fold_strat": "random (naive)", "fold_group": "by customer", "temporal": "by time (2023+)"}
    wide = table.pivot(index="model", columns="scheme", values=metric)[schemes].sort_values(schemes[-1], ascending=False)
    pos = np.arange(len(wide))
    width = 0.8 / len(schemes)
    for i, s in enumerate(schemes):
        ax.bar(pos + i * width - 0.4 + width / 2, wide[s], width, label=labels[s])
    ax.set_xticks(pos, wide.index, rotation=30, ha="right")
    ax.set(ylabel=metric.replace("_", "-").upper(), title=f"{metric.replace('_', '-').upper()} by validation scheme")
    ax.legend(frameon=False)
    return fig


def plot_cost_sensitivity(table: pd.DataFrame, ax=None):
    """Cost-minimising threshold under each recovery / future-loss assumption."""
    fig, ax = _axes(ax)
    for fl, g in table.groupby("future_loans"):
        ax.plot(g["recovery_rate"] * 100, g["threshold"], "o-", label=f"{int(fl)} future loan(s) lost per wrong decline")
    ax.set(xlabel="Share of principal recovered after default (%)", ylabel="Decline if P(default) >",
           title="Operating threshold under different cost assumptions")
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_calibration_by_group(y_true, prob, groups, ax=None):
    """Mean predicted vs actual default rate per group: is the model honest for every group?"""
    fig, ax = _axes(ax)
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(prob), "g": np.asarray(groups)})
    t = df.groupby("g").agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()
    pos = np.arange(len(t))
    ax.bar(pos - 0.2, t["predicted"] * 100, 0.4, label="predicted")
    ax.bar(pos + 0.2, t["actual"] * 100, 0.4, label="actual")
    ax.set_xticks(pos, [f"{g}\n(n={n:,})" for g, n in zip(t["g"], t["n"])])
    ax.set(ylabel="Default rate (%)", title="Calibration by group")
    ax.legend(frameon=False)
    return fig
