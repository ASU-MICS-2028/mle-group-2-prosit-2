# Progress dashboard

**Read this first.** Update it at the end of every work session (2 minutes), then push. Last updated: 2026-11-10 by Perfect, covering everything merged to `main` up to PR #41.

Status: ⬜ not started · 🟨 in progress · ✅ done · 🟥 blocked

## At a glance

| Part | What | Owner | Status |
| --- | --- | --- | --- |
| Setup | README, data guide, roadmap | Bernard | ✅ (2026-10-21) |
| Setup | Roles, docs, tracking | Perfect | ✅ pushed 2026-11-07 |
| A | Explore Train/Test structure | Perfect | ✅ `notebooks/01_eda.ipynb` §1-5, merged (#38, #39) |
| A | Explore the target | Henry | 🟨 first EDA merged (`notebooks/explore_target_henry.ipynb`): by year only; month, lender, loan type, tenure still to do |
| A | Explore amounts/durations (P3 slice) | Bernard | ⬜ |
| A | Explore Ghana and context files (P4 slice) | Eugene | ⬜ |
| A | Timeline, label, feature audit table | Perfect + Henry | 🟨 audit table drafted (`docs/feature_audit.md`); timeline and label to do |
| A | Two tasks + cost matrix in cedis/shillings | Henry (draft) | ⬜ |
| B | Shared feature file and fold splits | Perfect | ✅ merged (#39); Henry's baseline already uses it |
| B | Preprocessor for unseen lenders and loan types (D-5) | Perfect + Henry | ⬜ |
| B | Finish `01_eda`: missing values, skew, cardinality | Perfect | ⬜ |
| B | Perfect's models: baselines, kNN, SVM | Perfect | ⬜ |
| B | Shared train/evaluate scripts, Lightning DataModule | Henry | ⬜ |
| B | Logistic regression (scikit-learn + Lightning) | Henry | ✅ ROC-AUC 0.845, PR-AUC 0.170 on shared features (`02_classification`, PR #41) |
| B | MLP | Henry | ⬜ next |
| B | Models, validation study, calibration, CIs | per [model table](../README.md#5-models-and-who-owns-them) | ⬜ |
| C | Kenya → Ghana transfer study | Eugene | ⬜ |
| D | Model Card, Datasheet, fairness, regulation, Findex | Eugene | ⬜ |
| E | Scoring service, reason codes | Bernard | ⬜ |
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
| 2026-11-10 | Henry | Logistic regression baseline (raw columns), then on shared features + Lightning version; plain-language explainer | `notebooks/02_classification.ipynb`, `docs/learning/`, PRs #40, #41 |

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
| 2 | 🟧 | **All four roles are filled, but Bernard (P3) and Eugene (P4) have not pushed any work yet** (Bernard merged PR #39). They own Task 1, Parts C/D/E and 4 of 9 models, about half the grade | Perfect or Henry brief them; they pick up issues and add a WORKLOG entry |
| 3 | 🟨 | Review rotation: P1→P2→P3→P4→P1 (Perfect reviews Henry, Henry reviews Bernard, Bernard reviews Eugene, Eugene reviews Perfect) | Confirm everyone has repo access and is assigned issues |
| 4 | 🟧 | D-2 to D-7 are drafts, yet Henry's results already depend on them | Hold the Part A session; mark each decided or changed |
| 5 | 🟧 | Raw data is git-ignored; each person downloads the zips | See Getting started |
| 6 | 🟨 | Brief says key `TransactionId`; file uses `ID` | Use `ID`; note in report |
| 7 | 🟨 | README plan omits graded items: Datasheet, proxy analysis, a mitigation, AI declaration | See [DELIVERABLES](DELIVERABLES.md#gaps-in-the-readme-plan) |

## Session log (newest first)

| Date | Who | Did | Next |
| --- | --- | --- | --- |
| 2026-11-10 | Perfect | Docs sync after #39–#41; docs-freshness rule and check | Part A session; finish `01_eda` |
| 2026-11-10 | Henry | Logistic regression: scikit-learn + Lightning on shared features (0.845 / 0.170) | MLP; train/evaluate scripts; validation study |
| 2026-11-10 | Perfect | Shared feature file, folds, feature audit, P3/P4 roles (#39) | Part A session |
| 2026-11-09 | Perfect | Train/Test structure EDA (#38) | Shared feature file |
| 2026-11-07 | Perfect | Roles, docs | Push; Part A session |
