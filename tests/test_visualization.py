"""Every plot function draws without error, labels its axes, and saves a sharp PNG."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import src.visualization as V  # noqa: E402
from src.visualization import eda as E  # noqa: E402

rng = np.random.default_rng(0)
Y = (rng.random(500) < 0.1).astype(int)
P = np.clip(Y * 0.3 + rng.random(500) * 0.5, 0, 1)
GROUPS = pd.DataFrame({"group_col": "g", "group": ["a", "b"], "selection_rate": [0.8, 0.6], "actual_default_rate": [0.02, 0.05]})
SCORES = pd.DataFrame({
    "country_id": ["Kenya"] * 50 + ["Ghana"] * 50,
    "decision": rng.choice(["approve", "limit", "decline"], 100),
    "probability_of_default": rng.random(100) * 0.2,
    "requested_local": rng.lognormal(8, 1, 100),
    "limit_local": rng.lognormal(8, 1, 100),
    "limit_usd": rng.lognormal(4, 1, 100),
})
RATE = pd.DataFrame({"k": ["x", "y", "z"], "loans": [100, 50, 10], "defaults": [2, 3, 4], "default_rate": [0.02, 0.06, 0.4]})

CASES = {
    "pr_curves": lambda: V.plot_pr_curves(Y, {"m": P}),
    "roc_curves": lambda: V.plot_roc_curves(Y, {"m": P}),
    "reliability": lambda: V.plot_reliability(Y, {"m": P}),
    "threshold_sweep": lambda: V.plot_threshold_sweep(pd.DataFrame({"threshold": [0.1, 0.2], "expected_cost_per_loan": [3, 2], "selection_rate": [0.9, 0.95]}), 0.15),
    "confusion": lambda: V.plot_confusion(90, 5, 2, 3),
    "interval_forest": lambda: V.plot_interval_forest(pd.DataFrame({"m": ["a", "b"], "v": [0.2, 0.3], "lo": [0.1, 0.2], "hi": [0.3, 0.4]}), "v", "lo", "hi", "m", "t", ref=0.05),
    "scheme_comparison": lambda: V.plot_scheme_comparison(pd.DataFrame({"model": ["a"] * 3 + ["b"] * 3, "scheme": ["fold_strat", "fold_group", "temporal"] * 2, "pr_auc": [0.3, 0.2, 0.1, 0.25, 0.2, 0.15]})),
    "cost_sensitivity": lambda: V.plot_cost_sensitivity(pd.DataFrame({"future_loans": [1, 1, 6, 6], "recovery_rate": [0, 0.4, 0, 0.4], "threshold": [0.01, 0.02, 0.08, 0.1]})),
    "calibration_by_group": lambda: V.plot_calibration_by_group(Y, P, np.where(rng.random(500) < 0.5, "new", "repeat")),
    "amount_distribution": lambda: V.plot_amount_distribution(rng.lognormal(8, 1, 300)),
    "residuals": lambda: V.plot_residuals_vs_fitted(rng.lognormal(4, 1, 300), rng.lognormal(4, 1, 300)),
    "pred_vs_actual": lambda: V.plot_pred_vs_actual(rng.lognormal(4, 1, 300), rng.lognormal(4, 1, 300)),
    "error_by_decile": lambda: V.plot_error_by_decile(rng.lognormal(4, 1, 300), rng.lognormal(4, 1, 300)),
    "coefficients": lambda: V.plot_coefficients(["num__a", "num__b", "cat__c"], [0.5, -0.2, 0.1]),
    "shap": lambda: V.plot_shap_importance(["num__a", "num__b"], rng.normal(size=(50, 2))),
    "group_selection": lambda: V.plot_group_selection_rates(GROUPS),
    "feature_shift": lambda: V.plot_feature_shift(SCORES.assign(x=rng.random(100)), "x"),
    "domain_scores": lambda: V.plot_domain_scores(rng.random(100), SCORES["country_id"].eq("Ghana"), 0.9),
    "shift_drivers": lambda: V.plot_shift_drivers(pd.DataFrame({"feature": ["num__a", "num__b"], "coef": [1.0, -0.5]})),
    "decision_mix": lambda: V.plot_decision_mix(SCORES),
    "score_distribution": lambda: V.plot_score_distribution(SCORES, 0.05),
    "limit_vs_request": lambda: V.plot_limit_vs_request(SCORES),
    "counts": lambda: V.plot_counts(pd.Series({"a": 3, "b": 5}), "t"),
    "latency": lambda: V.plot_latency(rng.gamma(5, 10, 200)),
    "eda_rate_bars": lambda: E.plot_rate_bars(RATE, "k", "t", overall=0.05),
    "eda_rate_over_time": lambda: E.plot_rate_over_time(RATE, "k", "t"),
    "eda_raw_vs_log": lambda: E.plot_raw_vs_log(rng.lognormal(8, 2, 300), "amount"),
    "eda_univariate_auc": lambda: E.plot_univariate_auc(pd.DataFrame({"column": ["a", "b"], "auc": [0.98, 0.6], "verdict": ["leak", "kept"]})),
    "eda_box": lambda: E.plot_box_by_group(pd.DataFrame({"v": rng.random(100), "g": ["x", "y"] * 50}), "v", "g", "t"),
    "eda_hist": lambda: E.plot_histogram(rng.random(100), "t", "x"),
    "eda_grouped": lambda: E.plot_grouped_bars(pd.DataFrame({"g": ["a", "b"], "u": [1, 2], "v": [3, 4]}), "g", ["u", "v"], "t", "y"),
    "eda_corr": lambda: E.plot_correlation(pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc")), list("abc")),
    "eda_macro": lambda: E.plot_macro_lines(pd.DataFrame({"Country": ["K", "K", "G", "G"], "Indicator": ["I (%)"] * 4, "year": [2021, 2022] * 2, "value": [1, 2, 3, 4]}), ["I (%)"]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_plot_draws_a_titled_labelled_figure(name: str) -> None:
    fig = CASES[name]()
    assert isinstance(fig, plt.Figure)
    axes = [ax for ax in fig.axes if ax.has_data() and ax.get_label() != "<colorbar>"]
    assert axes, "nothing drawn"
    assert any(ax.get_title() for ax in fig.axes), "every figure needs a title"
    assert all(ax.get_xlabel() or ax.get_xticklabels() for ax in axes), "every axis needs a label or ticks"
    plt.close(fig)


def test_saved_figures_are_high_resolution(tmp_path, monkeypatch) -> None:
    from src import config as C

    monkeypatch.setattr(C, "FIGURES", tmp_path)
    fig = V.plot_pr_curves(Y, {"m": P})
    path = V.save_figure(fig, "check")
    from PIL import Image

    width, height = Image.open(path).size
    assert width >= 1000, f"figure only {width}px wide; report figures must be print quality"
    assert path.stat().st_size > 20_000
    plt.close(fig)


def test_every_plot_function_has_a_test() -> None:
    tested = {fn.__code__.co_names for fn in CASES.values()}
    used = set().union(*tested)
    missing = {n for n in V.__all__ if n.startswith("plot_") and n not in used}
    assert not missing, f"untested plot functions: {missing}"
