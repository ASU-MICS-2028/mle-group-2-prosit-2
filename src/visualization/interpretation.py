"""Global interpretation (#22): which features drive the default model overall."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _barh(values: pd.Series, title: str, xlabel: str, ax=None, signed: bool = False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(values) + 1.4))
    else:
        fig = ax.figure
    v = values.sort_values()
    colors = np.where(v > 0, "tab:red", "tab:blue") if signed else "tab:blue"
    ax.barh(v.index, v.to_numpy(), color=colors)
    if signed:
        ax.axvline(0, c="grey", lw=1)
    ax.set(title=title, xlabel=xlabel)
    return fig


def plot_coefficients(names, coefs, top: int = 15, ax=None):
    """Logistic regression weights on standardized inputs: red raises the log-odds of default, blue lowers it."""
    s = pd.Series(np.asarray(coefs, float).ravel(), index=[n.split("__", 1)[-1] for n in names])
    s = s.reindex(s.abs().sort_values(ascending=False).index[:top])
    return _barh(s, "Logistic regression: largest weights", "weight on the standardized feature (log-odds)", ax, signed=True)


def plot_shap_importance(names, shap_values, top: int = 15, ax=None):
    """Mean |SHAP| per feature: average push on the log-odds, whatever the direction."""
    s = pd.Series(np.abs(np.asarray(shap_values, float)).mean(axis=0), index=[n.split("__", 1)[-1] for n in names])
    s = s.sort_values(ascending=False)[:top]
    return _barh(s, "Tree model: mean |SHAP| per feature", "mean |SHAP value| (log-odds)", ax)
