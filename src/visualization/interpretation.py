"""Global interpretation (#22): which features drive the default model overall."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PRETTY_NAMES: dict[str, str] = {
    "n_past_loans": "Prior loans count",
    "n_past_labelled": "Prior completed loans",
    "days_since_last_loan": "Days since last loan",
    "n_past_defaults": "Past defaults count",
    "past_default_rate": "Past default rate",
    "days_since_last_default": "Days since last default",
    "is_new_borrower": "First-time borrower",
    "Lender_portion_Funded": "Lender funding share",
    "n_lenders_on_loan": "Lenders on loan",
    "log_amount_usd": "Loan amount (log USD)",
    "duration": "Repayment term (days)",
    "disb_month": "Disbursement month",
    "disb_dom": "Day of month",
    "disb_dow": "Day of week",
}

ABLATION_LABELS: dict[str, str] = {
    "- borrower_history": "Without borrower history",
    "- loan_terms": "Without loan terms",
    "- calendar": "Without calendar dates",
    "- lender_product": "Without lender & product",
    "- amount": "Without loan amount",
    "+ macro": "Adding macro indicators",
}


def _pretty_feature(raw_name: str) -> str:
    clean = raw_name.split("__", 1)[-1]
    if clean in PRETTY_NAMES:
        return PRETTY_NAMES[clean]
    if clean.startswith("loan_type_"):
        return f"Loan product: {clean.removeprefix('loan_type_')}"
    if clean.startswith("lender_id_"):
        return f"Lender: {clean.removeprefix('lender_id_')}"
    if clean.startswith("missingindicator_"):
        base = clean.removeprefix("missingindicator_")
        return f"Missing: {PRETTY_NAMES.get(base, base)}"
    return clean


def _barh(values: pd.Series, title: str, xlabel: str, ax=None, signed: bool = False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 0.34 * len(values) + 1.6))
    else:
        fig = ax.figure
    v = values.sort_values()
    colors = np.where(v > 0, "#c0392b", "#1f5f9f") if signed else "#1f5f9f"
    ax.barh(v.index, v.to_numpy(), color=colors)
    if signed:
        ax.axvline(0, c="grey", lw=1)
        ax.legend(
            handles=[
                Patch(facecolor="#c0392b", label="Raises default risk"),
                Patch(facecolor="#1f5f9f", label="Lowers default risk"),
            ],
            frameon=False,
            loc="lower right",
        )
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    return fig


def plot_coefficients(names, coefs, top: int = 15, ax=None):
    """Logistic regression weights on standardized inputs: red raises the log-odds of default, blue lowers it."""
    s = pd.Series(np.asarray(coefs, float).ravel(), index=[_pretty_feature(n) for n in names])
    s = s.reindex(s.abs().sort_values(ascending=False).index[:top])
    return _barh(s, "Deployed logistic regression: top risk drivers", "Weight on standardized input (log-odds)", ax, signed=True)


def plot_shap_importance(names, shap_values, top: int = 15, ax=None):
    """Mean |SHAP| per feature: average push on the log-odds, whatever the direction."""
    s = pd.Series(np.abs(np.asarray(shap_values, float)).mean(axis=0), index=[_pretty_feature(n) for n in names])
    s = s.sort_values(ascending=False)[:top]
    return _barh(s, "LightGBM: average feature impact (mean |SHAP|)", "Mean |SHAP value| (log-odds)", ax)


def plot_ablation(ablation_df: pd.DataFrame, metric: str = "delta_pr_auc", models: list[str] | None = None):
    """Grouped bar chart of metric change when each feature group is removed or macro is added (#52)."""
    sub = ablation_df[ablation_df["variant"] != "full"].copy()
    chosen_models = models or list(dict.fromkeys(sub["model"]))
    schemes = [s for s in ("fold_strat", "fold_group", "temporal") if s in set(sub["scheme"])]
    scheme_labels = {"fold_strat": "Random folds", "fold_group": "By customer (GroupKFold)", "temporal": "By time (2023+)"}
    scheme_colors = {"fold_strat": "#9aa0a6", "fold_group": "#1f5f9f", "temporal": "#e07b24"}
    model_labels = {"logreg": "Deployed Logistic Regression", "lightgbm": "LightGBM (Tree Ensemble)"}

    n_panels = len(chosen_models)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.4 * n_panels, 4.8), sharey=True)
    axes_list = [axes] if n_panels == 1 else list(axes)
    variants = list(dict.fromkeys(sub["variant"]))
    pretty_variants = [ABLATION_LABELS.get(v, v) for v in variants]
    y_pos = np.arange(len(variants))
    width = 0.76 / max(len(schemes), 1)
    xlabel = "Change in PR-AUC vs full model" if "pr_auc" in metric else "Change in ROC-AUC vs full model"

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
        ax.set_yticklabels(pretty_variants)
        ax.invert_yaxis()
        ax.set(title=model_labels.get(model_name, model_name), xlabel=xlabel)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=len(schemes), frameon=False)
    fig.suptitle(
        "What each feature group adds (negative = dropping group hurts; positive = dropping group helps)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    return fig
