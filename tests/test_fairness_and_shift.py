"""Fairness audit and Kenya -> Ghana shift helpers (Parts C and D, owner P4), on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calibrate_oof,
    decision_cost_by_group,
    disaggregated_evaluation,
    proxy_strength,
    run_domain_classifier,
)


def _two_markets(synthetic_features: pd.DataFrame, shift: float) -> pd.DataFrame:
    """Kenya rows plus a Ghana copy whose loan amounts are moved by `shift` (log USD)."""
    ke = synthetic_features.assign(country_id="Kenya")
    gh = synthetic_features.assign(country_id="Ghana", ID=lambda d: "GH_" + d["ID"])
    gh["log_amount_usd"] = gh["log_amount_usd"] + shift
    return pd.concat([ke, gh], ignore_index=True)


# ----------------------------------------------------------------- cost of decisions
def test_decision_cost_counts_both_errors_in_money() -> None:
    df = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 2, "y_true": [1, 0, 1, 0, 0, 0],
                       "approve": [True, True, False, False, True, True]})
    out = decision_cost_by_group(df, "g", "approve", c_fp=90.0, c_fn=2416.0).set_index("group")
    a = out.loc["a"]
    assert (a["defaults_approved"], a["good_borrowers_declined"]) == (1, 1)
    assert a["total_cost"] == pytest.approx(2416.0 + 90.0)
    assert a["cost_per_applicant"] == pytest.approx((2416.0 + 90.0) / 4)
    assert a["approval_rate"] == pytest.approx(0.5)
    assert out.loc["b", "total_cost"] == 0.0  # all good borrowers approved: no cost


def test_approving_everyone_costs_exactly_the_defaults() -> None:
    y = np.array([1, 0, 0, 1, 0])
    df = pd.DataFrame({"g": "all", "y_true": y, "approve": True})
    out = decision_cost_by_group(df, "g", "approve", c_fp=10.0, c_fn=100.0).iloc[0]
    assert out["total_cost"] == pytest.approx(y.sum() * 100.0)


# ----------------------------------------------------------------- proxies
def test_proxy_strength_ranks_a_perfect_categorical_proxy_first() -> None:
    df = pd.DataFrame({"is_g": [1, 1, 0, 0], "lender": ["x", "x", "y", "y"], "noise": [1.0, 2.0, 1.0, 2.0]})
    out = proxy_strength(df, "is_g", ["noise", "lender"])
    assert out["feature"].tolist() == ["lender", "noise"]
    assert out.set_index("feature")["proxy_strength"].tolist() == [1.0, 0.0]


def test_proxy_strength_is_symmetric_for_inverse_proxies() -> None:
    """A feature that is *lower* for the group is just as strong a proxy as one that is higher."""
    df = pd.DataFrame({"is_g": [1, 1, 0, 0], "x": [0.0, 1.0, 2.0, 3.0]})
    row = proxy_strength(df, "is_g", ["x"]).iloc[0]
    assert row["auc"] == 0.0 and row["proxy_strength"] == 1.0


def test_proxy_strength_handles_missing_values() -> None:
    df = pd.DataFrame({"is_g": [1, 1, 0, 0], "x": [np.nan, 5.0, 1.0, np.nan]})
    assert np.isfinite(proxy_strength(df, "is_g", ["x"])["proxy_strength"]).all()


# ----------------------------------------------------------------- disaggregated metrics
def test_disaggregated_evaluation_selection_rates_and_ratio() -> None:
    df = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "y_true": [0, 0, 0, 1, 0, 1, 1, 0],
                       "y_pred": [0.01, 0.01, 0.01, 0.9, 0.9, 0.9, 0.01, 0.01]})
    out = disaggregated_evaluation(df, "g", threshold=0.5).set_index("group")
    assert out.loc["a", "selection_rate"] == 0.75 and out.loc["b", "selection_rate"] == 0.5
    assert out.loc["b", "disparate_impact_ratio"] == pytest.approx(0.5 / 0.75)
    assert out.loc["a", "selection_rate_gap"] == 0.0


def test_disaggregated_evaluation_works_without_labels() -> None:
    """Ghana Test rows have no label: only selection rate and mean score are reported."""
    df = pd.DataFrame({"g": ["gh"] * 3, "y_true": np.nan, "y_pred": [0.01, 0.02, 0.9]})
    row = disaggregated_evaluation(df, "g", threshold=0.5).iloc[0]
    assert row["selection_rate"] == pytest.approx(2 / 3)
    assert "pr_auc" not in row or pd.isna(row.get("pr_auc"))


def test_calibration_fixes_the_selection_rate_read_off_class_weighted_scores() -> None:
    """The bug #54 fixes: thresholding inflated raw scores declines almost everyone."""
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.02).astype(int)
    raw = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, n), 0.001, 0.999)
    cal = calibrate_oof(y, raw, np.arange(n) % 5, method="isotonic")
    thr = 0.036
    assert np.mean(raw <= thr) < 0.05
    assert np.mean(cal <= thr) > 0.9
    assert abs(cal.mean() - y.mean()) < 0.01


# ----------------------------------------------------------------- domain classifier
def test_identical_markets_are_indistinguishable(synthetic_features) -> None:
    dom = run_domain_classifier(_two_markets(synthetic_features, shift=0.0))
    assert dom["domain_roc_auc"] < 0.6
    assert dom["effective_sample_share"] > 0.8  # weights near 1: Kenya already looks like Ghana


def test_shifted_market_is_detected_and_weights_collapse(synthetic_features) -> None:
    dom = run_domain_classifier(_two_markets(synthetic_features, shift=3.0))
    assert dom["domain_roc_auc"] > 0.95
    assert dom["top_shift_drivers"]["feature"].iloc[0].endswith("log_amount_usd")
    assert dom["effective_sample_share"] < 0.3
    w = dom["importance_weights"]
    kenya = _two_markets(synthetic_features, 3.0)["country_id"].eq("Kenya").to_numpy()
    assert w[kenya].mean() == pytest.approx(1.0)


def test_drop_cols_removes_the_shifted_feature(synthetic_features) -> None:
    dom = run_domain_classifier(_two_markets(synthetic_features, shift=3.0), drop_cols=("log_amount_usd",))
    assert not dom["top_shift_drivers"]["feature"].str.contains("log_amount_usd").any()
    assert dom["domain_roc_auc"] < 0.6


def test_domain_classifier_needs_two_countries(synthetic_features) -> None:
    with pytest.raises(ValueError, match="two countries"):
        run_domain_classifier(synthetic_features)
