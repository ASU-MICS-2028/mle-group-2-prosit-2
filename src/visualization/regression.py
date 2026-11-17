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


def plot_pred_vs_actual(y_true, y_pred, ax=None, title: str = "Predicted vs actual amount (log scale)"):
    """Points on the diagonal are perfect; a flat cloud means the model predicts close to the average."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    a, p = np.log1p(np.asarray(y_true, float)), np.log1p(np.asarray(y_pred, float))
    ax.scatter(a, p, s=2, alpha=0.2)
    lim = [min(a.min(), p.min()), max(a.max(), p.max())]
    ax.plot(lim, lim, c="grey", lw=1)
    ax.set(xlabel="actual log(1 + USD)", ylabel="predicted log(1 + USD)", title=title)
    return fig


def plot_error_by_decile(y_true, y_pred, ax=None, title: str = "Median absolute error by loan-size decile"):
    """Where the limit model goes wrong: small loans, big loans, or everywhere?"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    else:
        fig = ax.figure
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    dec = np.asarray(np.ceil(np.argsort(np.argsort(yt)) / len(yt) * 10), dtype=int).clip(1, 10)
    med = [np.median(np.abs(yt[dec == d] - yp[dec == d]) / np.maximum(yt[dec == d], 1e-9)) * 100 for d in range(1, 11)]
    ax.bar(range(1, 11), med)
    ax.set(xlabel="actual amount decile (1 = smallest)", ylabel="median |error| as % of the loan", title=title)
    return fig
