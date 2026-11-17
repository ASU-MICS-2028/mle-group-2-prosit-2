# Progress dashboard

**Read this first.** Update it at the end of every work session (2 minutes), then push. Last updated: 2026-11-17 by Bernard on branch `p3/13-30-senior-scaffolding`.

Status: ⬜ not started · 🟨 in progress · ✅ done · 🟥 blocked

## At a glance

| Part | What | Owner | Status |
| --- | --- | --- | --- |
| Setup | README, data guide, roadmap | Bernard | ✅ (2026-10-21) |
| Setup | Roles, docs, tracking | Perfect | ✅ pushed 2026-11-07 |
| A | Explore Train/Test structure | Perfect | ✅ `notebooks/01_eda.ipynb` §1-5, merged (#38, #39) |
| A | Explore the target | Henry + Bernard | ✅ Converted `explore_target_henry.ipynb` to `src/eda_target.py` (year, month, lender, loan type, tenure, 2023+ shift, `extra_pct`) |
| A | Explore amounts/durations (P3 slice) | Bernard | ⬜ |
| A | Explore Ghana and context files (P4 slice) | Eugene | ⬜ |
| A | Timeline, label, feature audit table | Perfect + Henry | 🟨 audit table drafted (`docs/feature_audit.md`); timeline and label to do |
| A | Two tasks + cost matrix in cedis/shillings | Henry (draft) | ⬜ |
| B | Shared feature file and fold splits | Perfect | ✅ merged (#39); Henry's baseline already uses it |
| B | Preprocessor for unseen lenders and loan types (D-5) | Perfect + Bernard | 🟨 `src/preprocess.py` (`make_preprocessor`, `TargetLogScaler`) on `p3/13-30-senior-scaffolding` |
| B | Finish `01_eda`: missing values, skew, cardinality | Perfect | ⬜ |
| B | Perfect's models: baselines, kNN, SVM | Perfect | ⬜ |
| B | Shared train/evaluate scripts, Lightning DataModule | Henry + Bernard | 🟨 `src/datamodule.py`, `src/train.py`, `src/evaluate.py` + `tests/` on `p3/13-30-senior-scaffolding` |
| B | Logistic regression (scikit-learn + Lightning) | Henry | ✅ ROC-AUC 0.845, PR-AUC 0.170 on shared features (`02_classification`, PR #41) |
| B | MLP | Henry | ✅ ROC-AUC 0.865, PR-AUC 0.215 (`02_classification.ipynb`) |
| B | Models, validation study, calibration, CIs | per [Team & Ownership](../README.md#team--ownership) | 🟨 Validation study run; calibration & CIs next |
| C | Kenya → Ghana transfer study | Eugene | 🟨 `src/fairness.py` helpers ready; `04_transfer_and_fairness.ipynb` next |
| D | Model Card, Datasheet, fairness, regulation, Findex | Eugene | 🟨 `src/fairness.py` disaggregated evaluation ready; `docs/audit/` next |
| E | Scoring service, reason codes | Bernard | 🟨 `src/score.py` + `tests/test_pipeline_and_scorer.py` on `p3/13-30-senior-scaffolding` |
| Final | Report, audit, slides, reflections, AI declaration | all | ⬜ |

Per-issue progress is on GitHub (Roadmap issue #1). I could not read the issues from here (no `gh` CLI), so check their state there.

## Done so far

| When | Who | What | Where |
| --- | --- | --- | --- |
| 2026-10-21 | Bernard | README: data profiling, framework, pipeline design, roles, rules | `main` |
| 2026-11-01 | Henry | EDA: default rate, leak check, new vs repeat, rate by year | branch `explore/henry-eda`, `01_eda.ipynb` |
| 2026-11-07 | Perfect | Roles assigned; this docs system | `docs/`, commit d207a1f pushed straight to `main` |
| 2026-11-09 | Perfect | Train/Test structure EDA: two-lender loans, sibling-row leak, Ghana shift | `notebooks/01_eda.ipynb`, PR #38 |
| 2026-11-10 | Perfect | Shared feature file, 3 fold schemes, leakage checks, feature audit, P3/P4 roles | `src/features.py`, `docs/feature_audit.md`, PR #39 (merged by Bernard) |
| 2026-11-17 | Bernard | Converted `explore_target_henry.ipynb` to `src/eda_target.py`; made `02`, `03`, `04` thin script-backed deliverable notebooks; enforced 4-notebook rule in `tests/` | `p3/13-30-senior-scaffolding` |
| 2026-11-16 | Bernard | Shared `src/` pipeline (`preprocess`, `datamodule`, `train`, `evaluate`, `fairness`, `score`), `tests/` CI guardrails, `Makefile`, streamlined README | `p3/13-30-senior-scaffolding` |
| 2026-11-10 | Henry | Logistic regression (raw + shared + Lightning) and MLP + temporal check | `notebooks/02_classification.ipynb`, `docs/learning/`, PRs #40, #41 |

Findings so far:
- Default rate by year is 1.3% (2022, 64,405 loans), 11.5% (2023), 8.7% (2024), so random splits will mislead; the 2023+ validation set defaults at 9.5%.
- The two repayment columns look like leaks (AUC 0.98 on the markup); both dropped.
- The 920 "shared" Train/Test loans are sibling lender rows, so Train labels leak Test labels; group every split by customer.
- Ghana: 0% of customers, 97% of rows by lender and 87% by loan type are unseen in Train.
- Defaulters get cut off (3.5% ever get another loan), so labels cover only approved borrowers.
- Logistic regression on shared features: ROC-AUC 0.845, PR-AUC 0.170 (Henry, #41).

## What's left, in order

1. **Part A session (all four):** confirm [DECISIONS](DECISIONS.md) D-2 to D-7, which are drafts that the code already relies on. Draw the loan timeline, define the label, write the cost matrix.
2. **Henry:** MLP, `src/train.py` and `src/evaluate.py`, validation study (`fold_strat` vs `fold_group` vs `temporal`), calibration, cost threshold, bootstrap CIs.
3. **Perfect:** finish `01_eda`, the unseen-category preprocessor (D-5), baselines, kNN, SVM and the SVM derivation.
4. **Bernard (P3) and Eugene (P4):** start their parts using `data/processed/features.parquet` and `docs/feature_audit.md`.
5. Report, audit, slides, reflections, AI-use declaration.
6. **Deadline: 19 Nov 2026.**

## Blockers and risks

| # | Severity | Item | Next step |
| --- | --- | --- | --- |
| 1 | 🟧 | **Deadline confirmed: 19 Nov 2026.** The brief's Nov 2–6 dates are superseded. Nine days left as of 2026-11-10 | Plan backwards from 19 Nov: models and results by about 15 Nov, report, audit and slides after |
| 2 | 🟨 | **Bernard (P3) shared scaffolding & Part E service built on `p3/13-30-senior-scaffolding`; Eugene (P4) work starting** | Merge `p3/13-30-senior-scaffolding`; Bernard finishes `03_regression.ipynb`, Eugene runs `04_transfer_and_fairness.ipynb` & `docs/audit/` |
| 3 | 🟨 | Review rotation: P1→P2→P3→P4→P1 (Perfect reviews Henry, Henry reviews Bernard, Bernard reviews Eugene, Eugene reviews Perfect) | Confirm everyone has repo access and is assigned issues |
| 4 | 🟧 | D-2 to D-7 are drafts, yet Henry's results already depend on them | Hold the Part A session; mark each decided or changed |
| 5 | 🟧 | Raw data is git-ignored; each person downloads the zips | See Getting started |
| 6 | 🟨 | Brief says key `TransactionId`; file uses `ID` | Use `ID`; note in report |
| 7 | 🟨 | README plan omits graded items: Datasheet, proxy analysis, a mitigation, AI declaration | See [DELIVERABLES](DELIVERABLES.md#gaps-in-the-readme-plan) |

## Session log (newest first)

| Date | Who | Did | Next |
| --- | --- | --- | --- |
| 2026-11-17 | Bernard | Script-first refactor (`src/eda_target.py`, thin `02`–`04` deliverable notebooks, D-17, 4-notebook test) | Merge PR; Part A session; audit & derivations |
| 2026-11-16 | Bernard | Senior ML scaffolding (`preprocess`, `datamodule`, `train`, `evaluate`, `fairness`, `score`, `tests/`, CI, `Makefile`, streamlined `README.md`) on `p3/13-30-senior-scaffolding` | Review & merge PR; run all models via `src.train`, finish `03_regression.ipynb` |
| 2026-11-10 | Perfect | Docs sync after #39–#41; docs-freshness rule and check | Part A session; finish `01_eda` |
| 2026-11-10 | Henry | Logistic regression: scikit-learn + Lightning on shared features (0.845 / 0.170) | MLP; train/evaluate scripts; validation study |
| 2026-11-10 | Perfect | Shared feature file, folds, feature audit, P3/P4 roles (#39) | Part A session |
| 2026-11-09 | Perfect | Train/Test structure EDA (#38) | Shared feature file |
| 2026-11-07 | Perfect | Roles, docs | Push; Part A session |
