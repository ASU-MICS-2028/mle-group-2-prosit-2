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


def plot_ablation(ablation_df: pd.DataFrame, metric: str = "delta_pr_auc", models: list[str] | None = None):
    """Grouped bar chart of metric change when each feature group is removed or macro is added (#52)."""
    sub = ablation_df[ablation_df["variant"] != "full"].copy()
    chosen_models = models or list(dict.fromkeys(sub["model"]))
    schemes = [s for s in ("fold_strat", "fold_group", "temporal") if s in set(sub["scheme"])]
    scheme_labels = {"fold_strat": "Random folds", "fold_group": "By customer (GroupKFold)", "temporal": "By time (2023+)"}
    scheme_colors = {"fold_strat": "#9aa0a6", "fold_group": "#1f5f9f", "temporal": "#e07b24"}

    n_panels = len(chosen_models)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.4), sharey=True)
    axes_list = [axes] if n_panels == 1 else list(axes)
    variants = list(dict.fromkeys(sub["variant"]))
    y_pos = np.arange(len(variants))
    width = 0.78 / max(len(schemes), 1)
    xlabel = "Δ PR-AUC vs full D-18 model" if "pr_auc" in metric else "Δ ROC-AUC vs full D-18 model"

    for ax, model_name in zip(axes_list, chosen_models):
        m_df = sub[sub["model"] == model_name]
        for idx, scheme in enumerate(schemes):
            s_df = m_df[m_df["scheme"] == scheme].set_index("variant").reindex(variants)
            offset = (idx - (len(schemes) - 1) / 2) * width
            ax.barh(
                y_pos + offset,
                s_df[metric].to_numpy(),
                height=width,
                label=scheme_labels.get(scheme, scheme),
                color=scheme_colors.get(scheme),
            )
        ax.axvline(0.0, color="black", lw=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variants)
        ax.invert_yaxis()
        ax.set(title=f"{model_name}: {xlabel}", xlabel=xlabel)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)

    axes_list[-1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Feature-group ablation (negative = dropping group hurts; D-18 seed: removing macro raised MLP temporal ROC-AUC 0.48 -> 0.65)",
        fontsize=9.5,
        y=1.02,
    )
    return fig
