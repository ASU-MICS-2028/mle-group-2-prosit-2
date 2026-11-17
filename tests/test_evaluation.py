"""Cost matrix, calibration, metrics and the customer-level bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import (
    PlattCalibrator,
    bootstrap_ci,
    calibrate_oof,
    classification_metrics,
    cost_matrix_threshold,
    cost_sensitivity_table,
    measure_market_cost_inputs,
    regression_metrics,
)


def test_cost_threshold_formula() -> None:
    cm = cost_matrix_threshold(principal=1000, fee_rate=0.02, future_loans=1, recovery_rate=0.2, collection_cost=100)
    assert np.isclose(cm["c_fp"], 20.0)
    assert np.isclose(cm["c_fn"], 900.0)
    assert np.isclose(cm["threshold"], 20.0 / 920.0)


def test_cost_config_is_not_the_brief_example() -> None:
    """The brief says not to reuse Appendix A (GHS 300, 15% fee)."""
    cm = cost_matrix_threshold()
    assert (cm["principal"], cm["fee_rate"]) != (300.0, 0.15)


def test_market_cost_inputs_use_the_main_term() -> None:
    """Ghana has no labels: the fee is the median markup on its most common loan term (#11)."""
    raw = pd.DataFrame({
        "country_id": ["Ghana"] * 5 + ["Kenya"],
        "duration": [7, 7, 7, 180, 7, 7],
        "Total_Amount": [1000.0, 2000.0, 3000.0, 5000.0, 4000.0, 9.0],
        "Total_Amount_to_Repay": [1030.0, 2060.0, 3090.0, 6000.0, 6000.0, 9.0],  # one defaulted 7-day loan
    })
    m = measure_market_cost_inputs(raw, "Ghana")
    assert m["principal_median"] == 3000.0
    assert np.isclose(m["fee_rate_median_on_term"], 0.03)   # the defaulted loan's 50% does not move the median
    assert (m["term_days"], m["share_on_term"], m["n_loans"]) == (7.0, 0.8, 5.0)


def test_sensitivity_has_at_least_two_alternatives() -> None:
    assert len(cost_sensitivity_table()) >= 3


def test_platt_never_returns_exactly_zero() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < raw * 0.1).astype(int)
    p = PlattCalibrator().fit(raw, y).predict(np.array([0.0, 1e-9, 0.5, 1.0]))
    assert (p > 0).all() and (p < 1).all()


def test_calibrate_oof_shapes_and_range() -> None:
    y = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1])
    p = np.linspace(0.05, 0.9, 12)
    folds = np.arange(12) % 3
    for method in ("platt", "isotonic"):
        cal = calibrate_oof(y, p, folds, method=method)
        assert cal.shape == p.shape and ((cal >= 0) & (cal <= 1)).all()


def test_classification_and_regression_metrics() -> None:
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9], threshold=0.25)
    assert m["tp"] == 2 and m["fp"] == 0 and np.isclose(m["selection_rate"], 0.5)
    r = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
    assert r["rmse_orig"] >= r["mae_orig"] > 0


def test_customer_bootstrap_is_wider_than_row_bootstrap() -> None:
    """Correlated loans per customer: resampling customers must not give a narrower interval."""
    rng = np.random.default_rng(1)
    customers = np.repeat(np.arange(200), 10)
    risk = rng.uniform(0, 0.3, 200)
    y = np.repeat((rng.uniform(0, 1, 200) < risk).astype(int), 10)  # a customer's loans share one outcome
    p = np.repeat(risk, 10) + rng.normal(0, 0.02, 2000)
    rows = bootstrap_ci(y, p, threshold=0.15, n_boot=200)
    clustered = bootstrap_ci(y, p, threshold=0.15, groups=customers, n_boot=200)
    assert clustered["resampled_unit"] == "customer"
    width = lambda r: r["pr_auc_ci_high"] - r["pr_auc_ci_low"]  # noqa: E731
    assert width(clustered) >= width(rows)
