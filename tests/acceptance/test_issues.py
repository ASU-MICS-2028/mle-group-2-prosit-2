"""Acceptance tests: one per GitHub issue. Each passes only when that issue's "done when" is true.

    make acceptance                 # scoreboard for every issue (open issues are expected to fail)
    make acceptance ISSUE=53        # one issue
    python -m pytest tests/acceptance --issues 53,54

They read only committed files (docs, notebooks, decision log, reports/tables), so they run anywhere.
GitHub runs them automatically:
  - on a pull request that says "Closes #N": the PR check fails until issue N's test passes;
  - when issue N is closed: if its test fails on main, a bot reopens the issue with the reason.
The first line of each docstring is shown on the issue's card in the README.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pandas as pd
import pytest

from src import config as C

issue = pytest.mark.issue
DOCS, NB, TABLES = C.ROOT / "docs", C.ROOT / "notebooks", C.TABLES


# ------------------------------------------------------------------ helpers
def read(rel: str) -> str:
    path = C.ROOT / rel
    if not path.exists():
        pytest.fail(f"{rel} does not exist yet")
    return path.read_text(encoding="utf-8")


def notebook(name: str) -> tuple[str, list[dict]]:
    nb = json.loads(read(f"notebooks/{name}.ipynb"))
    return "\n".join("".join(c["source"]) for c in nb["cells"]), nb["cells"]


def decision_status(decision: str) -> str:
    for line in read("docs/DECISIONS.md").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and cells[1] == decision:
            return cells[3].lower()
    pytest.fail(f"{decision} not found in docs/DECISIONS.md")


def assert_decided(*decisions: str) -> None:
    pending = {d: decision_status(d) for d in decisions if not decision_status(d).startswith("decided")}
    assert not pending, f"still not decided in docs/DECISIONS.md: {pending}"


def table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        pytest.fail(f"reports/tables/{name} does not exist yet")
    return pd.read_csv(path)


def assert_sections(rel: str, sections: list[str], min_chars: int = 800) -> None:
    text = read(rel)
    missing = [s for s in sections if s.lower() not in text.lower()]
    assert not missing, f"{rel} is missing sections: {missing}"
    assert len(text) >= min_chars, f"{rel} is only {len(text)} characters; write it out fully"


def files_in(rel: str, patterns: tuple[str, ...]) -> list:
    folder = C.ROOT / rel
    return [p for pat in patterns for p in folder.glob(pat)] if folder.exists() else []


def executed_without_errors(cells: list[dict]) -> bool:
    code = [c for c in cells if c["cell_type"] == "code" and "".join(c["source"]).strip()]
    ran = all(c.get("execution_count") for c in code)
    errors = any(o.get("output_type") == "error" for c in code for o in c.get("outputs", []))
    return ran and not errors


# ------------------------------------------------------------------ setup and Part A
@issue(2)
def test_issue_2_datasets_profiled():
    """The data guide documents all three datasets and every Train column."""
    text = read("docs/DATA_AND_PIPELINE_GUIDE.md")
    for pattern in ("Train.csv", "economic_indicators", "micro_gha", r"financial.inclusion"):
        assert re.search(pattern, text, re.I), f"data guide does not cover {pattern}"


@issue(3)
def test_issue_3_roles_in_readme():
    """The README names all four members with their roles."""
    text = read("README.md")
    for name in ("Perfect", "Henry", "Bernard", "Eugene"):
        assert name in text


@issue(4)
def test_issue_4_repo_skeleton():
    """Repo skeleton, .gitignore that excludes data, and a shared config exist."""
    assert "data/raw/" in read(".gitignore")
    assert C.SEED == 42 and C.N_FOLDS == 5


@issue(5)
def test_issue_5_train_test_structure():
    """01_eda explains two-lender loans and the Train/Test overlap, with outputs."""
    src, cells = notebook("01_eda")
    for word in ("two-lender", "overlap", "Ghana"):
        assert word.lower() in src.lower(), f"01_eda does not discuss {word}"
    assert any(c.get("outputs") for c in cells if c["cell_type"] == "code")


@issue(6)
def test_issue_6_target_eda():
    """Default rate by year, month, lender, loan type and tenure is reproducible."""
    from src.analysis import compute_target_eda  # noqa: F401

    report = read("reports/EDA_REPORT.md")
    assert "How rare is default" in report


@issue(7)
def test_issue_7_leak_ruling():
    """The leak-suspect ruling (D-2) is decided, backed by the leakage section of the EDA report."""
    assert "Which columns leak the answer" in read("reports/EDA_REPORT.md")
    assert_decided("D-2")


@issue(8)
def test_issue_8_ghana_and_context_explored():
    """Kenya vs Ghana, macro indicators and Findex are explored in the EDA report."""
    report = read("reports/EDA_REPORT.md")
    for section in ("Is Ghana like Kenya", "macro indicators", "Findex"):
        assert section in report


@issue(9)
def test_issue_9_part_a_session():
    """Part A notes exist (timeline, label, decision moment) and decisions D-2 to D-7 are decided."""
    assert_sections("docs/part_a.md", ["timeline", "label", "disbursement_date", "decision"], min_chars=600)
    assert_decided("D-2", "D-3", "D-4", "D-5", "D-6", "D-7")


@issue(10)
def test_issue_10_tasks_and_goal():
    """Task 1, Task 2 and the business goal are written down (02 section 0, or docs/part_a.md)."""
    src, cells = notebook("02_classification")
    framing = [c for c in cells if c["cell_type"] == "markdown" and "two tasks" in "".join(c["source"]).lower()]
    text = src if framing else read("docs/part_a.md")
    for word in ("Task 1", "Task 2", "goal", "approv"):
        assert word.lower() in text.lower(), f"the problem framing does not state {word}"


@issue(11)
def test_issue_11_cost_matrix():
    """A money cost matrix (KES or GHS) is decided (D-19 or D-22), with at least two alternative assumptions."""
    statuses = {d: decision_status(d) for d in ("D-19", "D-22")}
    assert any(v.startswith("decided") for v in statuses.values()), f"no cost-matrix decision is decided yet: {statuses}"
    sens = table("cost_sensitivity.csv")
    assert len(sens) >= 3 and {"c_fp", "c_fn", "threshold"} <= set(sens.columns)
    src, _ = notebook("02_classification")
    assert "cost_matrix_threshold" in src and ("GHS" in src or "KES" in src), "02 does not show the cost matrix in money"


# ------------------------------------------------------------------ Part B
@issue(12)
def test_issue_12_features_and_folds():
    """Shared features with GroupKFold, stratified and temporal folds, and leakage validation."""
    from src import features

    assert callable(features.validate) and callable(features.add_folds)


@issue(13)
def test_issue_13_shared_train_and_evaluate():
    """Shared training, evaluation and Lightning DataModule code exist."""
    from scripts import evaluate, train  # noqa: F401
    from src.models.data_module import LoanDataModule  # noqa: F401


@issue(14)
def test_issue_14_validation_study():
    """All models are scored under random, by-customer and by-time validation."""
    t2 = table("task2_model_summary.csv")
    assert {"fold_strat", "fold_group", "temporal"} <= set(t2["scheme"])


@issue(15)
def test_issue_15_baselines_knn_svm():
    """Majority baseline, kNN and SVM results are in the model summary."""
    t2 = table("task2_model_summary.csv")
    assert {"majority", "knn", "svm_rbf"} <= set(t2["model"])


@issue(16)
def test_issue_16_logreg_and_mlp():
    """Logistic regression (scikit-learn and Lightning) and the MLP are trained and compared."""
    src, _ = notebook("02_classification")
    assert "LitLogReg" in src and "LitMLP" in src


@issue(17)
def test_issue_17_p3_models():
    """03 compares OLS, Ridge, Lasso and the regression tree, and discusses the Task 2 decision tree."""
    src, _ = notebook("03_regression")
    for model in ("ols", "ridge", "lasso", "decision_tree_reg"):
        assert model in src
    assert re.search(r"[\"']decision_tree[\"']", src), "03 does not discuss the Task 2 decision tree (decision_tree)"


@issue(18)
def test_issue_18_random_forest_and_lightgbm():
    """Random Forest and LightGBM results are in the model summary."""
    t2 = table("task2_model_summary.csv")
    assert {"random_forest", "lightgbm"} <= set(t2["model"])


@issue(19)
def test_issue_19_calibration_and_threshold():
    """02 calibrates the probabilities and picks the cost-based threshold with a confusion matrix."""
    src, _ = notebook("02_classification")
    for word in ("calibrate_oof", "cost_matrix_threshold", "sweep_thresholds", "classification_metrics"):
        assert word in src, f"02 does not use {word}"
    assert "approved" in src and "declined" in src, "02 has no confusion matrix (approved/declined vs repaid/defaulted)"


@issue(20)
def test_issue_20_bootstrap_ci():
    """02 reports customer-level bootstrap confidence intervals against the baseline."""
    src, _ = notebook("02_classification")
    assert "bootstrap_ci" in src and "groups=" in src


@issue(21)
def test_issue_21_regression_metrics():
    """Task 1 metrics on both scales and residual plots exist."""
    t1 = table("task1_model_summary.csv")
    assert {"rmse_orig", "mae_orig", "rmse_log", "mae_log"} <= set(t1.columns)
    assert files_in("reports/figures", ("t1_residuals_*.png",))


@issue(22)
def test_issue_22_interpretation():
    """docs/interpretation.md explains the main drivers in plain language, using the coefficient and SHAP figures."""
    assert_sections("docs/interpretation.md", ["coefficient", "SHAP", "month", "borrower"])


@issue(48)
def test_issue_48_eda_complete_and_linked():
    """The EDA report covers imbalance, missing values, skew and cardinality, and 01_eda links to it."""
    report = read("reports/EDA_REPORT.md")
    for section in ("How rare is default", "What data do we have", "How are amounts distributed", "How many categories"):
        assert section in report
    src, _ = notebook("01_eda")
    assert "EDA_REPORT" in src, "01_eda does not link reports/EDA_REPORT.md yet"


@issue(51)
def test_issue_51_deployed_model_chosen():
    """The deployed classifier and limit model are chosen and recorded as a decided decision."""
    rows = [line for line in read("docs/DECISIONS.md").splitlines() if "deployed model" in line.lower()]
    assert rows, "no decision about the deployed model in docs/DECISIONS.md"
    assert any("| decided" in r.lower() for r in rows), "the deployed-model decision is not marked decided"


@issue(52)
def test_issue_52_ablation():
    """An ablation table shows the effect of removing each feature group."""
    found = files_in("reports/tables", ("ablation*.csv",))
    assert found, "no reports/tables/ablation*.csv yet"
    assert len(pd.read_csv(found[0])) >= 3


@issue(55)
def test_issue_55_temporal_calibration():
    """02 checks calibration of pre-2023 models on 2023+ loans and recommends recalibration."""
    src, _ = notebook("02_classification")
    assert re.search(r"temporal", src) and re.search(r"plot_reliability|reliability_table", src)
    assert re.search(r"recalibrat", src, re.I), "02 has no recalibration recommendation yet"
    assert re.search(r"2023\+?.{0,80}calibrat|calibrat.{0,80}2023", src, re.I | re.S), "no 2023+ calibration section yet"


# ------------------------------------------------------------------ Part C
@issue(23)
def test_issue_23_shift_analysis():
    """04 compares Kenya and Ghana, runs the domain classifier and reports the missing-history artefact."""
    src, _ = notebook("04_transfer_and_fairness")
    for word in ("run_domain_classifier", "importance_weights", "n_past_labelled"):
        assert word in src, f"04 does not cover {word}"


@issue(24)
def test_issue_24_trust_evidence():
    """docs/audit/ghana_trust.md lists the evidence NkwaPay needs before trusting Ghana scores."""
    assert_sections("docs/audit/ghana_trust.md", ["pilot", "calibrat", "monitor"], min_chars=500)


@issue(54)
def test_issue_54_notebook_04_owned_and_fixed():
    """04 has no draft banner, calibrates before thresholds, and ran top to bottom without errors."""
    src, cells = notebook("04_transfer_and_fairness")
    assert "DRAFT for Eugene" not in src, "the AI-draft banner is still at the top of 04"
    assert "calibrate_oof" in src, "04 still applies thresholds to uncalibrated probabilities"
    assert executed_without_errors(cells), "04 has unexecuted cells or error outputs; rerun it top to bottom"


# ------------------------------------------------------------------ Part D
@issue(25)
def test_issue_25_model_card():
    """docs/audit/model_card.md has every Model Card section (Mitchell et al.)."""
    assert_sections("docs/audit/model_card.md", ["Model details", "Intended use", "Factors", "Metrics",
                                                 "Evaluation data", "Training data", "Ethical considerations", "Caveats"])


@issue(26)
def test_issue_26_fairness_and_proxies():
    """04 reports metrics per group and a proxy analysis."""
    src, cells = notebook("04_transfer_and_fairness")
    assert "disaggregated_evaluation" in src
    headings = [line for c in cells if c["cell_type"] == "markdown" for line in "".join(c["source"]).splitlines()
                if line.lstrip().startswith("#")]
    assert any("proxy" in h.lower() for h in headings), "04 has no proxy-analysis section (a heading with 'proxy') yet"


@issue(27)
def test_issue_27_regulatory_map():
    """docs/audit/regulatory_map.md maps the design to the BoG Directive, Act 843 and Act 726, with gaps."""
    assert_sections("docs/audit/regulatory_map.md", ["Directive", "Act 843", "Act 726", "cannot"])


@issue(28)
def test_issue_28_findex():
    """Findex analysis says who in Ghana lacks credit and whether the model narrows or widens the gap."""
    text = read("docs/audit/findex.md") if (DOCS / "audit" / "findex.md").exists() else notebook("04_transfer_and_fairness")[0]
    assert "findex" in text.lower()
    assert re.search(r"widen|narrow", text, re.I), "no statement on whether the model widens or narrows the gap"


@issue(29)
def test_issue_29_decline_explanation():
    """docs/audit/decline_explanation.md has a plain-language decline message with how to ask for review."""
    assert_sections("docs/audit/decline_explanation.md", ["reason", "review"], min_chars=300)


@issue(35)
def test_issue_35_audit_document():
    """docs/audit/audit.md brings together the Model Card, fairness, proxy analysis and regulatory map."""
    assert_sections("docs/audit/audit.md", ["Model Card", "fairness", "proxy", "regulat"], min_chars=1500)


@issue(49)
def test_issue_49_datasheet():
    """docs/audit/datasheet.md answers every Datasheet section (Gebru et al.)."""
    assert_sections("docs/audit/datasheet.md", ["Motivation", "Composition", "Collection", "Preprocessing",
                                                "Uses", "Distribution", "Maintenance"])


@issue(50)
def test_issue_50_mitigation():
    """A mitigation is applied with before/after metrics per group (reports/tables/fairness_mitigation.csv)."""
    t = table("fairness_mitigation.csv")
    cols = " ".join(t.columns).lower()
    assert ("before" in cols and "after" in cols) or "stage" in cols, "table needs before and after values"


# ------------------------------------------------------------------ Part E
@issue(30)
def test_issue_30_scoring_service():
    """The scorer returns probability, decision, local-currency limit, three reasons and review flags."""
    import inspect

    from src.serving import scorer

    src = inspect.getsource(scorer.score_application)
    for key in ("probability_of_default", "decision", "recommended_limit", "top_3_reasons", "needs_review", "currency"):
        assert f'"{key}"' in src


@issue(31)
def test_issue_31_latency_and_inputs():
    """Latency over 1,000 calls is recorded and the required inputs are specified."""
    from src.serving import REQUIRED_APPLICATION_FIELDS

    lat = table("scorer_latency.csv")
    assert int(lat["n_calls"].iloc[0]) == 1000 and {"median_ms", "p95_ms"} <= set(lat.columns)
    assert len(REQUIRED_APPLICATION_FIELDS) >= 8


@issue(53)
def test_issue_53_limit_rule():
    """The limit is a cap: at most 25% of Kenyan Test applications get their amount cut."""
    t = table("product_decisions_by_country.csv").set_index("country_id")
    share = float(t.loc["Kenya", "limit"])
    assert share <= 0.25, f"{share:.0%} of Kenyan applications are still cut by the limit"


# ------------------------------------------------------------------ deliverables and process
@issue(32)
def test_issue_32_derivations():
    """All four derivations exist: SVM, logistic regression, linear regression, gradient boosting."""
    missing = [n for n in ("svm", "logistic", "linear", "boosting")
               if not (DOCS / "derivations" / f"{n}.md").exists() or len(read(f"docs/derivations/{n}.md")) < 1500]
    assert not missing, f"derivations missing or too short in docs/derivations/: {missing}"


@issue(33)
def test_issue_33_notebooks_run_clean():
    """All four notebooks are saved fully executed with no error outputs."""
    bad = [n for n in ("01_eda", "02_classification", "03_regression", "04_transfer_and_fairness")
           if not executed_without_errors(notebook(n)[1])]
    assert not bad, f"rerun top to bottom and save: {bad}"


@issue(34)
def test_issue_34_report():
    """The technical report is in docs/report/."""
    assert files_in("docs/report", ("*.pdf", "*.docx", "*.md")), "no report in docs/report/ yet"


@issue(36)
def test_issue_36_slides():
    """Debrief slides are in slides/."""
    assert files_in("slides", ("*.pptx", "*.pdf", "*.key")), "no slides in slides/ yet"


@issue(37)
def test_issue_37_reflections():
    """Four individual reflections are in docs/reflections/."""
    found = files_in("docs/reflections", ("*.md", "*.pdf", "*.docx"))
    assert len(found) >= 4, f"{len(found)} of 4 reflections in docs/reflections/"


@issue(56)
def test_issue_56_main_is_protected():
    """main requires a pull request with one approving review (checked through the GitHub API)."""
    if not shutil.which("gh"):
        pytest.skip("gh CLI not available")
    out = subprocess.run(["gh", "api", "repos/ASU-MICS-2028/mle-group-2-prosit-2/branches/main/protection"],
                         capture_output=True, text=True)
    assert out.returncode == 0, "main is not protected"
    reviews = json.loads(out.stdout).get("required_pull_request_reviews") or {}
    assert reviews.get("required_approving_review_count", 0) >= 1, "main does not require an approving review"
