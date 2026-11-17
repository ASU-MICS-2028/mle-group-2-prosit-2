"""Part D plot: approval (selection) rate per group, next to the group's actual default rate."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_group_selection_rates(table: pd.DataFrame, ax=None):
    """Bars from src.evaluation.disaggregated_evaluation output: approval rate (and default rate if labelled)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    else:
        fig = ax.figure
    x = np.arange(len(table))
    has_labels = "actual_default_rate" in table
    width = 0.4 if has_labels else 0.7
    ax.bar(x - (width / 2 if has_labels else 0), table["selection_rate"], width, label="approval rate")
    if has_labels:
        ax.bar(x + width / 2, table["actual_default_rate"], width, label="actual default rate")
    ax.set_xticks(x, table["group"], rotation=30, ha="right")
    ax.set(ylabel="Rate", title=f"By {table['group_col'].iloc[0]}")
    ax.legend(frameon=False)
    return fig
