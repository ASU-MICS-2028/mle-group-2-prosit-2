"""EDA plots: generic, so the same few functions draw every exploration figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _axes(ax, figsize=(6.5, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def plot_rate_bars(table: pd.DataFrame, x: str, title: str, ax=None, overall: float | None = None):
    """Default rate per category as bars, with the number of loans written on each bar."""
    fig, ax = _axes(ax)
    pos = np.arange(len(table))
    ax.bar(pos, table["default_rate"] * 100)
    for p, (rate, n) in enumerate(zip(table["default_rate"], table["loans"])):
        ax.text(p, rate * 100, f"n={n:,}", ha="center", va="bottom", fontsize=7)
    if overall is not None:
        ax.axhline(overall * 100, ls="--", c="grey", lw=1, label=f"overall {overall:.1%}")
        ax.legend(frameon=False)
    ax.set_xticks(pos, table[x].astype(str), rotation=30, ha="right")
    ax.set(ylabel="Default rate (%)", title=title)
    return fig


def plot_rate_over_time(table: pd.DataFrame, x: str, title: str, ax=None):
    """Default rate (line) and loan volume (bars) over time."""
    fig, ax = _axes(ax, figsize=(8, 4))
    pos = np.arange(len(table))
    ax.bar(pos, table["loans"], color="lightgrey", label="loans")
    ax.set_ylabel("Loans")
    ax2 = ax.twinx()
    ax2.plot(pos, table["default_rate"] * 100, "o-", c="tab:red", label="default rate")
    ax2.set_ylabel("Default rate (%)")
    step = max(1, len(pos) // 12)
    ax.set_xticks(pos[::step], table[x].astype(str).iloc[::step], rotation=45, ha="right")
    ax.set_title(title)
    return fig


def plot_raw_vs_log(values, title: str, ax=None):
    """Raw values next to log(1 + value): why every money column is log-transformed."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6)) if ax is None else (ax[0].figure, ax)
    v = np.asarray(values, dtype=float)
    axes[0].hist(v, bins=80)
    axes[0].set(title=f"{title}: raw", xlabel="value", ylabel="rows")
    axes[0].set_yscale("log")
    axes[1].hist(np.log1p(v), bins=80)
    axes[1].set(title=f"{title}: log(1 + value)", xlabel="log(1 + value)")
    return fig


def plot_univariate_auc(table: pd.DataFrame, ax=None):
    """Single-column AUC; anything near 1 records the outcome (leak)."""
    fig, ax = _axes(ax, figsize=(7, 3.8))
    t = table.sort_values("auc")
    colors = ["tab:red" if "leak" in v else ("tab:orange" if "dropped" in v else "tab:blue") for v in t["verdict"]]
    ax.barh(t["column"], t["auc"], color=colors)
    ax.axvline(0.5, c="grey", lw=1)
    ax.set(xlim=(0.45, 1.0), xlabel="AUC of the column alone (0.5 = no signal)", title="Leakage check: one column at a time")
    return fig


def plot_box_by_group(df: pd.DataFrame, value: str, group: str, title: str, ax=None, clip_q: float = 0.99):
    """Boxplots of `value` per group, clipped at a quantile so a few extreme rows do not hide the shape."""
    fig, ax = _axes(ax)
    upper = df[value].quantile(clip_q)
    groups = list(df[group].dropna().unique())
    ax.boxplot([df.loc[df[group] == g, value].clip(upper=upper) for g in groups], showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1), groups)
    ax.set(ylabel=value, title=title)
    return fig


def plot_histogram(values, title: str, xlabel: str, ax=None, bins: int = 50, log_y: bool = False):
    fig, ax = _axes(ax)
    ax.hist(np.asarray(values, dtype=float), bins=bins)
    if log_y:
        ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel="count")
    return fig


def plot_macro_lines(macro_long: pd.DataFrame, indicators: list[str], ax_row=None):
    """One small panel per indicator, one line per country."""
    fig, axes = plt.subplots(1, len(indicators), figsize=(4.2 * len(indicators), 3.4))
    for ax, ind in zip(np.atleast_1d(axes), indicators):
        sub = macro_long[(macro_long["Indicator"] == ind) & (macro_long["year"] >= 2015)]
        for country, g in sub.groupby("Country"):
            ax.plot(g["year"], g["value"], marker="o", ms=3, label=country)
        ax.axvspan(2020.5, 2023.5, color="grey", alpha=0.1)
        ax.set_title(ind.split("(")[0].strip(), fontsize=9)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    return fig


def plot_grouped_bars(table: pd.DataFrame, label_col: str, value_cols: list[str], title: str, ylabel: str, ax=None):
    fig, ax = _axes(ax, figsize=(8, 4))
    pos = np.arange(len(table))
    width = 0.8 / len(value_cols)
    for i, col in enumerate(value_cols):
        ax.bar(pos + i * width - 0.4 + width / 2, table[col], width, label=col.replace("_", " "))
    ax.set_xticks(pos, table[label_col], rotation=30, ha="right")
    ax.set(ylabel=ylabel, title=title)
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_correlation(df: pd.DataFrame, columns: list[str], title: str = "Correlation between model inputs", ax=None):
    fig, ax = _axes(ax, figsize=(7.5, 6.5))
    corr = df[columns].corr(method="spearman")
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(columns)), columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title(title)
    return fig
