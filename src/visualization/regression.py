"""Task 1 plots: amount distribution and residual-vs-fitted diagnostics."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_amount_distribution(amounts, ax=None, title: str = "Loan amount (log scale)"):
    """Histogram of log1p(amount): the raw amounts span 2 to 23 million, so only the log is readable."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.figure
    ax.hist(np.log1p(np.asarray(amounts, dtype=float)), bins=60)
    ax.set(xlabel="log(1 + amount)", ylabel="Loans", title=title)
    return fig


def plot_residuals_vs_fitted(y_true, y_pred, ax=None, title: str = "Residuals vs fitted (log scale)"):
    """A cloud centred on zero with no funnel or curve means the model's errors have no pattern left."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.figure
    fitted = np.log1p(np.maximum(0.0, np.asarray(y_pred, dtype=float)))
    resid = np.log1p(np.asarray(y_true, dtype=float)) - fitted
    ax.scatter(fitted, resid, s=3, alpha=0.25)
    ax.axhline(0, c="grey", lw=1)
    ax.set(xlabel="Fitted log(1 + amount)", ylabel="Residual", title=title)
    return fig
