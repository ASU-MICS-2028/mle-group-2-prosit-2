"""Part E plots: how the scoring service behaves on real applications."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_decision_mix(scores: pd.DataFrame, by: str = "country_id", ax=None):
    """Share of approve / limit / decline per group."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    t = pd.crosstab(scores[by], scores["decision"], normalize="index").reindex(columns=["approve", "limit", "decline"], fill_value=0)
    counts = scores[by].value_counts()
    labels = [f"{g}\n(n={counts[g]:,})" for g in t.index]
    bottom = np.zeros(len(t))
    for col, color in zip(t.columns, ["#2e8b57", "#e0a224", "#c0392b"]):
        vals = t[col].to_numpy() * 100
        ax.bar(labels, vals, bottom=bottom, label=col, color=color, width=0.6)
        for x, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 4:
                ax.text(x, b + v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        bottom += vals
    ax.set(ylabel="% of applications", ylim=(0, 100), title=f"Scorer decisions by {by.replace('_id', '')}")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    return fig


def plot_score_distribution(scores: pd.DataFrame, threshold: float, by: str = "country_id", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    else:
        fig = ax.figure
    bins = np.linspace(0, min(1, scores["probability_of_default"].quantile(0.995) * 1.1), 60)
    for g, sub in scores.groupby(by):
        ax.hist(sub["probability_of_default"], bins=bins, density=True, alpha=0.5, label=str(g))
    ax.axvline(threshold, c="k", ls="--", lw=1, label=f"threshold {threshold:.3f}")
    ax.set(xlabel="calibrated P(default)", ylabel="density", title="Scores by group")
    ax.legend(frameon=False)
    return fig


def plot_limit_vs_request(scores: pd.DataFrame, ax=None, title: str = "Requested amount vs model limit (USD)"):
    """Above the diagonal the request exceeds the limit and is cut."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    req_usd = scores["requested_local"] / (scores["limit_local"].where(scores["limit_local"] > 0) / scores["limit_usd"])
    ok = req_usd.notna()
    for g, sub in scores[ok].groupby("country_id"):
        ax.scatter(np.log10(sub["limit_usd"] + 1), np.log10(req_usd[sub.index] + 1), s=3, alpha=0.3, label=str(g))
    lim = ax.get_xlim()
    ax.plot(lim, lim, c="grey", lw=1)
    ax.set(xlabel="limit, log10(1 + USD)", ylabel="request, log10(1 + USD)", title=title)
    ax.legend(frameon=False)
    return fig


def plot_counts(counts: pd.Series, title: str, xlabel: str = "applications", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(counts) + 1.4))
    else:
        fig = ax.figure
    c = counts.sort_values()
    ax.barh(c.index.astype(str), c.to_numpy())
    ax.set(title=title, xlabel=xlabel)
    return fig


def plot_latency(samples_ms, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.8))
    else:
        fig = ax.figure
    ms = np.asarray(samples_ms, float)
    ax.hist(np.clip(ms, None, np.percentile(ms, 99.5)), bins=50)
    for q, ls in ((50, "-"), (95, "--")):
        ax.axvline(np.percentile(ms, q), c="k", ls=ls, lw=1, label=f"p{q} {np.percentile(ms, q):.0f} ms")
    ax.set(xlabel="ms per application", ylabel="calls", title="Scoring latency (1,000 calls)")
    ax.legend(frameon=False)
    return fig
