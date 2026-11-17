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
