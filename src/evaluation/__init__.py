"""Evaluation shared by the whole team: metrics, calibration, cost matrix, bootstrap, fairness, shift and ablation."""
from .ablation import ABLATION_GROUPS, run_ablation
from .bootstrap import bootstrap_ci
from .calibration import PlattCalibrator, calibrate_oof, make_calibrator
from .cost_matrix import (
    cost_matrix_threshold,
    cost_sensitivity_table,
    load_cost_config,
    measure_cost_inputs,
    measure_market_cost_inputs,
)
from .fairness import add_loan_size_band, decision_cost_by_group, disaggregated_evaluation, proxy_strength
from .metrics import (
    classification_metrics,
    regression_metrics,
    reliability_table,
    summarize_predictions_dir,
    sweep_thresholds,
)
from .shift import run_domain_classifier

__all__ = [
    "ABLATION_GROUPS",
    "PlattCalibrator",
    "add_loan_size_band",
    "bootstrap_ci",
    "calibrate_oof",
    "classification_metrics",
    "cost_matrix_threshold",
    "cost_sensitivity_table",
    "decision_cost_by_group",
    "disaggregated_evaluation",
    "load_cost_config",
    "make_calibrator",
    "measure_cost_inputs",
    "measure_market_cost_inputs",
    "proxy_strength",
    "regression_metrics",
    "reliability_table",
    "run_ablation",
    "run_domain_classifier",
    "summarize_predictions_dir",
    "sweep_thresholds",
]
