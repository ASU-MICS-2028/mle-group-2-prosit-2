# Progress dashboard

**Read this first.** Update it at the end of every work session (2 minutes), then push. Last updated: 2026-11-19 by Bernard (branch `p3/52-ablation-study`).

Status: ⬜ not started · 🟨 in progress · ✅ done · 🟥 blocked

## At a glance

| Part | What | Owner | Status |
| --- | --- | --- | --- |
| Setup | README, data guide, roadmap | Bernard | ✅ (2026-10-21) |
| Setup | Roles, docs, tracking | Perfect | ✅ pushed 2026-11-07 |
| A | Explore Train/Test structure | Perfect | ✅ `notebooks/01_eda.ipynb` §1-5, merged (#38, #39) |
| A | Explore the target | Henry | ✅ `src/analysis/target_eda.py` + `python -m scripts.run_target_eda` (tables in `reports/tables/eda_target_*.csv`) |
| A | Explore amounts/durations (P3 slice) | Bernard | 🟨 amounts, terms, leakage and history in `reports/EDA_REPORT.md` §4-6 (PR #57); leak-suspect ruling at Part A session |
| A | Explore Ghana and context files (P4 slice) | Eugene | ✅ EDA §8-10 (PR #57), interpreted in `04` §2.1 and §4 |
| A | Timeline, label, feature audit table | Perfect + Henry | 🟨 timeline drawn (`reports/figures/part_a_loan_timeline.png`, PR pending); label, window and decision moment in `02_classification` §0 (PR #60); audit table drafted (`docs/feature_audit.md`). All three await confirmation at the Part A session: agenda in `docs/PART_A_SESSION.md` |
| A | Two tasks + cost matrix in cedis/shillings | Henry | ✅ tasks and goal in `02` §0; cedis from Ghana's loans in §9b (D-22): p\* 0.036, approve 88.6%, −44% expected loss (PR #60). Team config still KES (D-19) |
| B | Shared feature file and fold splits | Perfect | ✅ merged (#39); Henry's baseline already uses it |
| B | Preprocessor for unseen lenders and loan types (D-5) | Perfect + Bernard | ✅ `src/processing/preprocess.py` (PR #46; target scaler clipped in PR #47) |
| B | Finish `01_eda`: missing values, skew, cardinality | Perfect | 🟨 §6 added: links `reports/EDA_REPORT.md`, re-derives its headline numbers, records 2 corrections (branch `p1/32-svm-derivation`, awaiting review); issue #48 to close after review |
| B | Perfect's models: baselines, kNN, SVM | Perfect | 🟨 trained in the full run (`reports/tables/task2_model_summary.csv`); explained in `docs/learning/P1_models_explained.md` (branch `p1/32-svm-derivation`, awaiting review); issue #15 to close after review |
| Deliverable | Report: data, leakage, features and validation section (#34) | Perfect | 🟨 `docs/report/p1_data_section.md` drafted (branch `p1/34-report-data-section`); Bernard assembles the report |
| B | Four derivations (#32): SVM, logistic, linear, boosting | all | ✅ all four complete (`docs/derivations/{svm,logistic,linear,boosting}.md`) with numerical tests against scikit-learn / PyTorch (`tests/test_*_derivation.py`) |
| B | Shared train/evaluate scripts, Lightning DataModule | Henry + Bernard | ✅ PR #46, restructured in PR #47: `src/models/`, `src/training/`, `src/evaluation/`, `scripts/`, 41 tests |
| B | Logistic regression (scikit-learn + Lightning) | Henry | ✅ ROC-AUC 0.844, PR-AUC 0.180 on the D-18 feature list (was 0.845 / 0.170) |
| B | MLP | Henry | ✅ ROC-AUC 0.867, PR-AUC 0.209; by time 0.649 (was 0.48 with macro columns) |
| B | Models, validation study, calibration, CIs | per [Team & Ownership](../README.md#team--ownership) | ✅ all 16 models run (`reports/tables/`); calibration (Platt, ECE 0.30 → 0.002) and customer-level CIs in `02_classification` §8–10 (added by Bernard) |
| B | Calibration over time (#55) | Henry | ✅ pre-2023 calibration under-predicts 2024 risk by half (LogReg 4.5% vs 8.7%); recalibrating on 2023 halves ECE but puts every 2024 applicant above break-even; recalibrate + monthly monitoring (D-24, PR #61) |
| C | Kenya → Ghana transfer study | Eugene | 🟨 `04` §2 rebuilt (domain classifier in 5 variants, importance weights, scorer on Ghana, pilot plan); in review |
| D | Model Card, Datasheet, fairness, regulation, Findex | Eugene | 🟨 `04` §3–4 and `docs/audit/` written; in review. Act 843/726 section numbers to confirm |
| E | Scoring service, reason codes, deployed models & quantile cap (#30, #31, #51, #53) | Bernard | ✅ D-29 (`logreg`) + D-30 (`lightgbm_reg_q85` 85th-quantile cap): Kenya Test cut drops 48.9% → 14.6% (15.7% of repaid Train loans cut); latency median 21 ms, p95 26 ms |
| E | Borrower app over the scorer | Eugene | 🟨 `app/streamlit_app.py` calls `score_application`; shows approve/limit/starter/decline, borrower reasons, credit-team panel (#65, branch `p4/65-borrower-app`, stacked on PR #64) |
| Final | Report, audit, slides, reflections, AI declaration | all | ⬜ |

Per-issue progress is on GitHub (Roadmap issue #1). Closed on 2026-11-17 with evidence: #3, #4, #5, #12, #14. Reopened: #30 (scorer bugs, now fixed in PR #47).

## Done so far

| When | Who | What | Where |
| --- | --- | --- | --- |
| 2026-10-21 | Bernard | README: data profiling, framework, pipeline design, roles, rules | `main` |
| 2026-11-01 | Henry | EDA: default rate, leak check, new vs repeat, rate by year | branch `explore/henry-eda`, `01_eda.ipynb` |
| 2026-11-07 | Perfect | Roles assigned; this docs system | `docs/`, commit d207a1f pushed straight to `main` |
| 2026-11-09 | Perfect | Train/Test structure EDA: two-lender loans, sibling-row leak, Ghana shift | `notebooks/01_eda.ipynb`, PR #38 |
| 2026-11-10 | Perfect | Shared feature file, 3 fold schemes, leakage checks, feature audit, P3/P4 roles | `src/features.py`, `docs/feature_audit.md`, PR #39 (merged by Bernard) |
| 2026-11-17 | Bernard | Converted `explore_target_henry.ipynb` to a script; made `02`–`04` script-backed notebooks (02 later restored, D-20) | PR #46 (merged without review) |
| 2026-11-17 | Bernard | Restructured `src/` into packages + `scripts/`, `configs/`, `reports/`; fixed scorer bugs (D-18, D-19); restored and completed Henry's `02` (§7–10); ran all 16 models | PR #47 |
| 2026-11-16 | Bernard | Shared `src/` pipeline, `tests/` CI guardrails, `Makefile`, streamlined README | PR #46 |
| 2026-11-10 | Henry | Logistic regression (raw + shared + Lightning) and MLP + temporal check | `notebooks/02_classification.ipynb`, `docs/learning/`, PRs #40, #41 |
| 2026-11-19 | Bernard | Feature-group ablation study (`logreg` + `lightgbm` across 7 subsets × 3 schemes): proves calendar only helps random/in-period folds (+0.020 LightGBM temporal ROC-AUC when removed), history/terms/product carry signal, macro hurts temporal (#52) | `src/evaluation/ablation.py`, `scripts/run_ablation.py`, `reports/tables/ablation.csv`, `reports/figures/interp_ablation*.png` |
| 2026-11-19 | Bernard | Added Task 2 pruned `decision_tree` comparison & pruning settings (`max_depth=6`, `ccp_alpha=0.0001`) to `03_regression.ipynb`, updated to D-29/D-30 scorer, re-executed end-to-end (#17) | `notebooks/03_regression.ipynb` |
| 2026-11-19 | Bernard | High-quantile limit cap (`lightgbm_reg_q85`, alpha=0.85, D-30): Kenya Test cut 48.9% → 14.6%, repaid Train cut 50.1% → 15.7%; tests, figures & latency (median 21 ms) updated (#53) | `configs/models.yaml`, `src/models/`, `src/training/artifacts.py`, `tests/`, `reports/` |
| 2026-11-19 | Bernard | Deployed model decision (D-29: calibrated `logreg` + `lightgbm_reg`), refitted scorer on D-28 cedi config + D-26 starter loan, regenerated `eval`, 35 `figures`, `score` (median 24 ms, p95 35 ms) (#51) | `docs/DECISIONS.md`, `reports/`, `scripts/make_figures.py` |
| 2026-11-19 | Bernard | Linear regression derivation (Gaussian MLE → normal equations → Ridge MAP) + 5 numerical tests vs scikit-learn & PyTorch (#32) | `docs/derivations/linear.md`, `tests/test_linear_derivation.py` |
| 2026-11-19 | Perfect | Report section for #34: data, leakage audit, time-safe features, three validation schemes with the inflation numbers, baseline/kNN/SVM, limitations; two deliberate-bug runs re-confirm `validate()` fires | `docs/report/p1_data_section.md`, branch `p1/34-report-data-section` |
| 2026-11-19 | Perfect | Part A prep (#9): loan timeline figure with a script, session agenda and decision sheet for the 8 open drafts, caveat on D-3's evidence, D-5 path fixed, feature audit links the figure | `docs/PART_A_SESSION.md`, `scripts/make_timeline_figure.py`, `reports/figures/part_a_loan_timeline.png`, branch `p1/9-part-a-session-prep` |
| 2026-11-19 | Perfect | `01_eda` §6 (#48): report linked, numbers re-derived from raw data; found the report's "loans" columns count rows, and that two-lender loans' higher default rate is mostly a time effect; fixed the same claim in `01_eda` §1 | `notebooks/01_eda.ipynb`, branch `p1/32-svm-derivation` |
| 2026-11-19 | Perfect | SVM derivation: margin, hinge loss, dual, kernel, Platt, with 8 tests against `SVC`; reads our SVM results (ROC-AUC 0.805, PR-AUC 0.073 by customer) | `docs/derivations/svm.md`, `tests/test_svm_derivation.py`, branch `p1/32-svm-derivation` |
| 2026-11-18 | Eugene | `04` rebuilt on the shipped scorer; cedi cost matrix (D-28); starter loan and borrower reasons in the scorer (D-26); audit docs, boosting derivation, fairness/shift helpers + 18 tests; D-25 to D-28 | `04`, `docs/audit/`, `docs/derivations/boosting.md`; branch `p4/54-transfer-fairness-audit` |
| 2026-11-17 | Henry | Problem framing, Ghana cost matrix in cedis, conclusions; verified calibration and CIs; D-7, D-22 | `02_classification` §0, §9b, §11; `src/evaluation/cost_matrix.py`; PR #60 |
| 2026-11-17 | Henry | Calibration over time (#55): stale vs recalibrated Platt on 2023+ loans, quarterly monitoring, recommendation (D-24) | `02_classification` §11, PR #61 |
| 2026-11-17 | Henry | Logistic regression derivation: log-odds → likelihood → BCE → gradient, `pos_weight`, convexity; 4 autograd tests | `docs/derivations/logistic.md`, `tests/test_logistic_derivation.py`, PR #62 |

Findings so far:
- Default rate by year is 1.3% (2022, 64,405 loans), 11.5% (2023), 8.7% (2024), so random splits will mislead; the 2023+ validation set defaults at 9.5%.
- The two repayment columns look like leaks (AUC 0.98 on the markup); both dropped.
- The 920 "shared" Train/Test loans are sibling lender rows, so Train labels leak Test labels; group every split by customer.
- Ghana: 0% of customers, 97% of rows by lender and 87% by loan type are unseen in Train.
- Defaulters get cut off (3.5% ever get another loan), so labels cover only approved borrowers.
- Logistic regression on shared features: ROC-AUC 0.844, PR-AUC 0.180; MLP 0.867 / 0.209 (Henry, `02`, D-18 feature list).
- Naive random folds inflate scores: LightGBM PR-AUC 0.303 random vs 0.195 by customer vs 0.148 by time (2023+). Best by customer: MLP 0.207, Random Forest 0.204. On 2023+ loans every model drops to ROC-AUC 0.56–0.65.
- (Superseded by D-28.) Draft cost matrix (D-19): declining a good borrower costs 88 KES, approving a default 4,449 KES, so the cost threshold is 0.019 and the MLP approves 79%; with 6 future loans lost per wrong decline the threshold is 0.106.
- Task 1 (USD): every model beats the mean on the log scale (LightGBM R² 0.21 by customer, 0.61 by time), but on dollars the linear models are worse than the mean; trees are best.
- Kenya vs Ghana is almost perfectly separable (domain AUC 0.997). Without the label-history artefact (`n_past_labelled` is 0 for all of Ghana) it is still 0.985, so importance weighting is unusable (ESS share 0.04, D-27) (`04` §2.2).
- Team cost matrix is now cedis with 3 future loans lost (D-28): threshold 0.10, 97.5% of Kenyan applicants approved. First-time borrowers 49% vs 98%; the scorer's 25% starter loan up to p 0.30 lifts them to 81% for GHS 18 per first-time applicant (D-26, `04` §3.4). No Ghanaian is approved in full (`04` §2.3).
- The deployed calibrated logistic regression under-predicts new borrowers (2.2% for a profile whose real rate is 5.7%).

## What's left, in order

1. **Part A session (all four):** follow the agenda in [PART_A_SESSION](PART_A_SESSION.md) and mark D-2 to D-6, D-13, D-14 and D-18 `decided` (drafts that the code already relies on). The timeline is drawn and the label and cost matrix are written; the session confirms them.
2. **Henry:** personal reflection (#37); agree with Bernard and Eugene whether the team config switches to the D-22 cedi costs.
3. **Perfect:** finish `01_eda`, the unseen-category preprocessor (D-5), baselines, kNN, SVM and the SVM derivation.
4. **Bernard:** get PR #47 reviewed and merged; #7 amounts/leak-suspect EDA, Task 2 decision tree (#17), linear regression derivation, final scorer model.
   **Eugene:** get the `04` + audit PR reviewed (Bernard); confirm Act 843/726 section numbers; personal reflection.
5. Report, audit, slides, reflections, AI-use declaration.
6. **Deadline: 19 Nov 2026.**

## Blockers and risks

| # | Severity | Item | Next step |
| --- | --- | --- | --- |
| 1 | 🟧 | **Deadline confirmed: 19 Nov 2026.** The brief's Nov 2–6 dates are superseded. Nine days left as of 2026-11-10 | Plan backwards from 19 Nov: models and results by about 15 Nov, report, audit and slides after |
| 2 | 🟨 | Cost matrix switched to cedis, 3 lost loans (D-28); committed `reports/` tables and `02`/`03` outputs predate it | Re-run `make all`, `02` and `03` after the PR merges |
| 2b | 🟧 | PR #46 was merged without review and rewrote Henry's notebook (since restored, D-20) | Branch protection on `main`; every PR reviewed by the rotation |
| 3 | 🟨 | Review rotation: P1→P2→P3→P4→P1 (Perfect reviews Henry, Henry reviews Bernard, Bernard reviews Eugene, Eugene reviews Perfect) | Confirm everyone has repo access and is assigned issues |
| 4 | 🟧 | D-2 to D-6 are drafts (D-7 decided in PR #60), yet results depend on them | Hold the Part A session; mark each decided or changed |
| 9 | 🟨 | `pytest tests/` segfaults on macOS (LightGBM/PyTorch OpenMP clash); Linux CI passes | Mac users run tests in CI or with Linux; fix the import order in `tests/` |
| 5 | 🟧 | Raw data is git-ignored; each person downloads the zips | See Getting started |
| 6 | 🟨 | Brief says key `TransactionId`; file uses `ID` | Use `ID`; note in report |
| 7 | 🟨 | README plan omits graded items: Datasheet, proxy analysis, a mitigation, AI declaration | See [DELIVERABLES](DELIVERABLES.md#gaps-in-the-readme-plan) |

## Session log (newest first)

| Date | Who | Did | Next |
| --- | --- | --- | --- |
| 2026-11-18 | Eugene | Borrower app wired to the live scorer; app smoke test | Merge after PR #64 |
| 2026-11-18 | Eugene | `04` rebuilt (#54); cedi cost matrix (D-28); starter loan + borrower reasons in the scorer (D-26); `docs/audit/`; boosting derivation; 18 tests | Bernard reviews; re-run `02`/`03`/`make all` on the cedi config |
| 2026-11-17 | Henry | Logistic regression derivation with autograd check (`docs/derivations/logistic.md`, #32, PR #62) | Reflection (#37) |
| 2026-11-17 | Henry | Calibration over time on 2023+ loans and recalibration plan, `02` §11 (#55, PR #61) | Derivation (#32); reflection |
| 2026-11-17 | Bernard | Acceptance test per issue + Action that blocks unfinished closes (D-23) | Owners close issues whose checks pass |
| 2026-11-17 | Henry | Reviewed PR #47; framing (#10), Ghana cost matrix (#11), conclusions; verified #19, #20 (PR #60) | Derivation (#32); align cost config |
| 2026-11-17 | Bernard | PRs #47, #57 merged; README status now auto-generated by a GitHub Action (D-21) | Each owner works their cards |
| 2026-11-17 | Bernard | All open issues assigned on GitHub; README issue cards per person | Everyone works their cards |
| 2026-11-17 | Bernard | EDA report, defence figure pack + catalogue, end-to-end and architecture/normalization tests (127), issues #48-#56 | Review PR #47 and the stacked PR #57 PR |
| 2026-11-17 | Bernard | Restructure (packages, scripts, configs, reports), scorer bug fixes, Henry's `02` restored and completed, all 16 models run | Review and merge PR #47; Eugene takes `04` |
| 2026-11-17 | Bernard | Script-first refactor (`src/eda_target.py`, thin `02`–`04` deliverable notebooks, D-17, 4-notebook test) | Merge PR; Part A session; audit & derivations |
| 2026-11-16 | Bernard | Senior ML scaffolding (`preprocess`, `datamodule`, `train`, `evaluate`, `fairness`, `score`, `tests/`, CI, `Makefile`, streamlined `README.md`) on `p3/13-30-senior-scaffolding` | Review & merge PR; run all models via `src.train`, finish `03_regression.ipynb` |
| 2026-11-10 | Perfect | Docs sync after #39–#41; docs-freshness rule and check | Part A session; finish `01_eda` |
| 2026-11-10 | Henry | Logistic regression: scikit-learn + Lightning on shared features (0.845 / 0.170) | MLP; train/evaluate scripts; validation study |
| 2026-11-10 | Perfect | Shared feature file, folds, feature audit, P3/P4 roles (#39) | Part A session |
| 2026-11-09 | Perfect | Train/Test structure EDA (#38) | Shared feature file |
| 2026-11-07 | Perfect | Roles, docs | Push; Part A session |
