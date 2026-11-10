# Progress dashboard

**Read this first.** Update it at the end of every work session (2 minutes), then push. Last updated: 2026-11-07.

Status: ⬜ not started · 🟨 in progress · ✅ done · 🟥 blocked

## At a glance

| Part | What | Owner | Status |
| --- | --- | --- | --- |
| Setup | README, data guide, roadmap | Bernard | ✅ (2026-10-21) |
| Setup | Roles, docs, tracking | Perfect | ✅ pushed 2026-11-07 |
| A | Explore Train/Test structure | Perfect | ✅ `notebooks/01_eda.ipynb` §1-4, in PR |
| A | Explore the target | Henry | 🟨 first EDA merged to `main` (`notebooks/explore_target_henry.ipynb`) |
| A | Explore amounts/durations (P3 slice) | Bernard | ⬜ |
| A | Explore Ghana and context files (P4 slice) | Eugene | ⬜ |
| A | Timeline, label, feature audit table | Perfect + Henry | 🟨 audit table drafted (`docs/feature_audit.md`); timeline and label to do |
| A | Two tasks + cost matrix in cedis/shillings | Henry (draft) | ⬜ |
| B | Shared feature file and fold splits | Perfect | 🟨 built and checked (`src/features.py`); awaiting review |
| B | Shared train/evaluate scripts, Lightning DataModule | Henry | ⬜ |
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
| 2026-11-07 | Perfect | Roles assigned; this docs system | this folder |

Findings so far: default rate by year is 1.3% (2022, 64,405 loans), 11.5% (2023), 8.7% (2024), so random splits will mislead. The two repayment columns look like leaks (AUC 0.98 each, per README).

## What's left, in order

1. Push the setup via PR.
2. Part A session (Perfect + Henry): settle [DECISIONS](DECISIONS.md) D-2 to D-7.
3. Perfect builds the shared feature file and folds; Henry builds train/evaluate scripts on a rough file meanwhile.
4. Each trains their models; Henry runs the validation study, calibration and CIs.
5. Bernard (P3) and Eugene (P4) start their parts: they need the feature file, `docs/feature_audit.md` and their role sections in the README.
6. Report, audit, slides, reflections, AI-use declaration.

## Blockers and risks

| # | Severity | Item | Next step |
| --- | --- | --- | --- |
| 1 | 🟥 | **Deadline.** The brief lists the debrief and portfolio for Nov 2–6; today is Nov 7 | Confirm the real due date with the instructor; record it here |
| 2 | 🟧 | **All four roles are filled, but Bernard (P3) and Eugene (P4) have not pushed anything yet.** They own Task 1, Parts C/D/E and 4 of 9 models, about half the grade | Perfect or Henry brief them; they pick up issues and add a WORKLOG entry |
| 3 | 🟨 | Review rotation: P1→P2→P3→P4→P1 (Perfect reviews Henry, Henry reviews Bernard, Bernard reviews Eugene, Eugene reviews Perfect) | Confirm everyone has repo access and is assigned issues |
| 5 | 🟧 | Raw data is git-ignored; each person downloads the zips | See Getting started |
| 6 | 🟨 | Brief says key `TransactionId`; file uses `ID` | Use `ID`; note in report |
| 7 | 🟨 | README plan omits graded items: Datasheet, proxy analysis, a mitigation, AI declaration | See [DELIVERABLES](DELIVERABLES.md#gaps-in-the-readme-plan) |

## Session log (newest first)

| Date | Who | Did | Next |
| --- | --- | --- | --- |
| 2026-11-09 | Perfect | Train/Test structure EDA | Part A session; rough feature file |
| 2026-11-07 | Perfect | Roles, docs | Push; Part A session |
