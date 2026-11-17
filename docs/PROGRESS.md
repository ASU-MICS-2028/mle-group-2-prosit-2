# Progress dashboard

**Read this first.** Update it at the end of every work session (2 minutes), then push. Last updated: 2026-11-17 by Henry (PR #XX).

Status: ⬜ not started · 🟨 in progress · ✅ done · 🟥 blocked

## At a glance

| Part | What | Owner | Status |
| --- | --- | --- | --- |
| Setup | README, data guide, roadmap | Bernard | ✅ (2026-10-21) |
| Setup | Roles, docs, tracking | Perfect | ✅ pushed 2026-11-07 |
| A | Explore Train/Test structure | Perfect | ✅ `notebooks/01_eda.ipynb` §1-5, merged (#38, #39) |
| A | Explore the target | Henry | ✅ `src/analysis/target_eda.py` + `python -m scripts.run_target_eda` (tables in `reports/tables/eda_target_*.csv`) |
| A | Explore amounts/durations (P3 slice) | Bernard | 🟨 amounts, terms, leakage and history in `reports/EDA_REPORT.md` §4-6 (PR #57); leak-suspect ruling at Part A session |
| A | Explore Ghana and context files (P4 slice) | Eugene | 🟨 Kenya vs Ghana, macro and Findex in `reports/EDA_REPORT.md` §8-10 (PR #57); Eugene to interpret for Part C/D |
| A | Timeline, label, feature audit table | Perfect + Henry | 🟨 label, window and decision moment in `02_classification` §0 (PR #XX); audit table drafted (`docs/feature_audit.md`); timeline drawing to do |
| A | Two tasks + cost matrix in cedis/shillings | Henry | ✅ tasks and goal in `02` §0; cedis from Ghana's loans in §9b (D-22): p\* 0.036, approve 88.6%, −44% expected loss (PR #XX). Team config still KES (D-19) |
| B | Shared feature file and fold splits | Perfect | ✅ merged (#39); Henry's baseline already uses it |
| B | Preprocessor for unseen lenders and loan types (D-5) | Perfect + Bernard | ✅ `src/processing/preprocess.py` (PR #46; target scaler clipped in PR #47) |
| B | Finish `01_eda`: missing values, skew, cardinality | Perfect | ⬜ |
| B | Perfect's models: baselines, kNN, SVM | Perfect | ⬜ |
| B | Shared train/evaluate scripts, Lightning DataModule | Henry + Bernard | ✅ PR #46, restructured in PR #47: `src/models/`, `src/training/`, `src/evaluation/`, `scripts/`, 41 tests |
| B | Logistic regression (scikit-learn + Lightning) | Henry | ✅ ROC-AUC 0.844, PR-AUC 0.180 on the D-18 feature list (was 0.845 / 0.170) |
| B | MLP | Henry | ✅ ROC-AUC 0.867, PR-AUC 0.209; by time 0.649 (was 0.48 with macro columns) |
| B | Models, validation study, calibration, CIs | per [Team & Ownership](../README.md#team--ownership) | ✅ all 16 models run (`reports/tables/`); calibration (Platt, ECE 0.30 → 0.002) and customer-level CIs in `02_classification` §8–10 (added by Bernard) |
| C | Kenya → Ghana transfer study | Eugene | 🟥 `04` is an AI draft (PR #46) Eugene must own; helpers in `src/evaluation/shift.py` |
| D | Model Card, Datasheet, fairness, regulation, Findex | Eugene | 🟥 not started by Eugene; helpers in `src/evaluation/fairness.py` |
| E | Scoring service, reason codes | Bernard | 🟨 #30 bugs fixed in PR #47 (Ghana limit currency, 0% risk, false reasons, review flags); latency median 44 ms, p95 75 ms |
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
| 2026-11-17 | Henry | Problem framing, Ghana cost matrix in cedis, conclusions; verified calibration and CIs; D-7, D-22 | `02_classification` §0, §9b, §11; `src/evaluation/cost_matrix.py`; PR #XX |

Findings so far:
- Default rate by year is 1.3% (2022, 64,405 loans), 11.5% (2023), 8.7% (2024), so random splits will mislead; the 2023+ validation set defaults at 9.5%.
- The two repayment columns look like leaks (AUC 0.98 on the markup); both dropped.
- The 920 "shared" Train/Test loans are sibling lender rows, so Train labels leak Test labels; group every split by customer.
- Ghana: 0% of customers, 97% of rows by lender and 87% by loan type are unseen in Train.
- Defaulters get cut off (3.5% ever get another loan), so labels cover only approved borrowers.
- Logistic regression on shared features: ROC-AUC 0.844, PR-AUC 0.180; MLP 0.867 / 0.209 (Henry, `02`, D-18 feature list).
- Naive random folds inflate scores: LightGBM PR-AUC 0.303 random vs 0.195 by customer vs 0.148 by time (2023+). Best by customer: MLP 0.207, Random Forest 0.204. On 2023+ loans every model drops to ROC-AUC 0.56–0.65.
- Draft cost matrix (D-19): declining a good borrower costs 88 KES, approving a default 4,449 KES, so the cost threshold is 0.019 and the MLP approves 79%; with 6 future loans lost per wrong decline the threshold is 0.106.
- Task 1 (USD): every model beats the mean on the log scale (LightGBM R² 0.21 by customer, 0.61 by time), but on dollars the linear models are worse than the mean; trees are best.
- Kenya vs Ghana is almost perfectly separable (domain AUC 0.996) even on behaviour, mainly because no Ghana customer has labelled history (`n_past_labelled` is 0 for all of Ghana): a data artefact Eugene must report in Part C.
- The deployed calibrated logistic regression under-predicts new borrowers (2.2% for a profile whose real rate is 5.7%).

## What's left, in order

1. **Part A session (all four):** confirm [DECISIONS](DECISIONS.md) D-2 to D-7, which are drafts that the code already relies on. Draw the loan timeline, define the label, write the cost matrix.
2. **Henry:** logistic regression derivation (#32); temporal calibration (#55); agree with Bernard and Eugene whether the team config switches to the D-22 cedi costs; personal reflection.
3. **Perfect:** finish `01_eda`, the unseen-category preprocessor (D-5), baselines, kNN, SVM and the SVM derivation.
4. **Bernard:** get PR #47 reviewed and merged; #7 amounts/leak-suspect EDA, Task 2 decision tree (#17), linear regression derivation, final scorer model.
   **Eugene:** take ownership of `04` (fix the 4 issues listed at its top), Model Card, Datasheet, regulatory map, proxy analysis, mitigation.
5. Report, audit, slides, reflections, AI-use declaration.
6. **Deadline: 19 Nov 2026.**

## Blockers and risks

| # | Severity | Item | Next step |
| --- | --- | --- | --- |
| 1 | 🟧 | **Deadline confirmed: 19 Nov 2026.** The brief's Nov 2–6 dates are superseded. Nine days left as of 2026-11-10 | Plan backwards from 19 Nov: models and results by about 15 Nov, report, audit and slides after |
| 2 | 🟥 | **Eugene (P4) has no commits; all of Part C/D so far is an AI draft (PR #46).** Parts C and D are 20+ points | Eugene owns `04` and `docs/audit/` today; deadline 19 Nov |
| 2b | 🟧 | PR #46 was merged without review and rewrote Henry's notebook (since restored, D-20) | Branch protection on `main`; every PR reviewed by the rotation |
| 3 | 🟨 | Review rotation: P1→P2→P3→P4→P1 (Perfect reviews Henry, Henry reviews Bernard, Bernard reviews Eugene, Eugene reviews Perfect) | Confirm everyone has repo access and is assigned issues |
| 4 | 🟧 | D-2 to D-6 are drafts (D-7 decided in PR #XX), yet results depend on them | Hold the Part A session; mark each decided or changed |
| 9 | 🟨 | `pytest tests/` segfaults on macOS (LightGBM/PyTorch OpenMP clash); Linux CI passes | Mac users run tests in CI or with Linux; fix the import order in `tests/` |
| 5 | 🟧 | Raw data is git-ignored; each person downloads the zips | See Getting started |
| 6 | 🟨 | Brief says key `TransactionId`; file uses `ID` | Use `ID`; note in report |
| 7 | 🟨 | README plan omits graded items: Datasheet, proxy analysis, a mitigation, AI declaration | See [DELIVERABLES](DELIVERABLES.md#gaps-in-the-readme-plan) |

## Session log (newest first)

| Date | Who | Did | Next |
| --- | --- | --- | --- |
| 2026-11-17 | Henry | Reviewed PR #47; framing (#10), Ghana cost matrix (#11), conclusions; verified #19, #20 (PR #XX) | Derivation (#32); align cost config |
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
