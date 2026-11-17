"""Part C plot: one feature's distribution in Kenya vs Ghana."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feature_shift(df: pd.DataFrame, feature: str, ax=None, bins: int = 40):
    """Overlaid density histograms of `feature` for each country in `df`."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.figure
    values = df[feature].astype(float)
    edges = np.histogram_bin_edges(values.dropna(), bins=bins)
    for country, sub in df.groupby("country_id"):
        ax.hist(sub[feature].dropna(), bins=edges, density=True, alpha=0.5, label=f"{country} (n={len(sub):,})")
    ax.set(xlabel=feature, ylabel="Density", title=f"{feature}: Kenya vs Ghana")
    ax.legend(frameon=False)
    return fig


def plot_domain_scores(prob_ghana, is_ghana, auc: float, ax=None):
    """How well a classifier tells Ghana from Kenya: two separate humps = a different population."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    else:
        fig = ax.figure
    p, g = np.asarray(prob_ghana), np.asarray(is_ghana, bool)
    bins = np.linspace(0, 1, 41)
    ax.hist(p[~g], bins=bins, density=True, alpha=0.5, label="Kenya")
    ax.hist(p[g], bins=bins, density=True, alpha=0.5, label="Ghana")
    ax.set(xlabel="P(Ghana | features), out of fold", ylabel="density", title=f"Domain classifier (AUC {auc:.3f})")
    ax.legend(frameon=False)
    return fig


def plot_shift_drivers(drivers: pd.DataFrame, top: int = 10, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 0.35 * top + 1.4))
    else:
        fig = ax.figure
    d = drivers.head(top).iloc[::-1]
    ax.barh(d["feature"].str.split("__").str[-1], d["coef"], color=np.where(d["coef"] > 0, "tab:orange", "tab:blue"))
    ax.axvline(0, c="grey", lw=1)
    ax.set(xlabel="weight (positive = more typical of Ghana)", title="What separates Ghana from Kenya")
    return fig
