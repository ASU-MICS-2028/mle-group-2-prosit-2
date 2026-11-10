# Work log (handoff notes)

Add an entry at the **top** at the end of each session, 5–10 lines. Link to notebooks or PRs instead of pasting results.

```
## YYYY-MM-DD · Name (role) · branch or PR
**Did:** …   **Found:** … (numbers + where)   **Next:** …   **Watch out:** …
```

---

## 2026-11-10 · Perfect (P1) · `p1/docs-sync`
**Did:** Synced every doc to `main` after #39–#41: README Status table, PROGRESS (done table, findings, what's left, blockers), WORKLOG titles that still said "uncommitted" or "not yet pushed", LEARNING_OUTCOMES, P1 and P2 plans. Added the rule **docs ship with code**: README section "Keeping the docs current", `CLAUDE.md`, an extended PR template and the `Docs freshness` GitHub Action (D-15).
**Found:** The README Status table was stale because it was never part of a PR checklist. Henry's #41 now does use `features.parquet` and `fold_group`, so an earlier note that it did not is out of date.
**Next:** Part A session to confirm D-2 to D-7; finish `01_eda`.
**Watch out:** The new check needs the label `no-docs-needed` for PRs that really need no doc update. It only runs once this PR is merged.

## 2026-11-10 · Henry (P2) · `p2/logreg-baseline`, PRs #40 and #41
**Did:** Logistic regression in `notebooks/02_classification.ipynb`. First on 7 raw columns from `Train.csv` (merged in #40), then rerun on `features.parquet` with `fold_group`, then rebuilt in PyTorch Lightning (#41). Plain-language explainer in `docs/learning/P2_logistic_regression_baseline.md`.
**Found:** (1) Shared features: ROC-AUC 0.845, PR-AUC 0.170; scikit-learn and Lightning agree exactly, so the Lightning loop is verified. (2) Borrower history lifts ROC-AUC 0.79 → 0.84 but not PR-AUC (0.19 → 0.19). (3) Calendar and macro columns slightly lower PR-AUC (all features: 0.17). Logistic regression weights every column, so weak ones add noise; macro barely varies because Train is all Kenya.
**Next:** MLP (rest of #16); `src/train.py` and `src/evaluate.py` (#13); validation study with `fold_strat` and `temporal` (#14).
**Watch out:** #40 was merged before the shared-features and Lightning commits were pushed, so they're in #41. Run the notebook with the `.venv` kernel (others may lack Lightning). On a Mac, Lightning needs `accelerator="cpu"` and float32. `src/config.py` expects the CSVs in `data/raw/african-credit-scoring-challenge/`.

## 2026-11-10 · Perfect (P1) · roles
**Did:** P3 = Bernard (`adjeibernardmarfo`), P4 = Eugene (`eugene-sew`); updated README, PROGRESS, DECISIONS (D-1, D-9, D-10), LEARNING_OUTCOMES, DELIVERABLES.
**Next:** Brief Bernard and Eugene; they read README §7 and `docs/feature_audit.md`.
**Watch out:** Neither has pushed anything yet.

## 2026-11-10 · Perfect (P1) · `p1/shared-features-and-folds`, PR #39 (merged by Bernard)
**Did:** `src/config.py`, `src/features.py` (shared feature file + 3 fold schemes + `validate()` leakage checks), `docs/feature_audit.md` (all 16 columns), EDA section 5, D-2 to D-7 marked draft/implemented, new D-13, D-14.
**Found:** (1) Repay markup alone gives AUC 0.98: 7-day loans owe 0.7% if repaid vs 32% if defaulted. (2) Defaulters get cut off: only 3.5% ever get another loan, so past-default features are nearly empty (152 rows) and labels cover only approved people. (3) Defaulted loans had a median 1.5 earlier loans vs 9 for repaid. (4) Temporal validation set (2023+) defaults at 9.5% vs 1.3% before, so scores will drop.
**Next:** Henry builds train/evaluate on `data/processed/features.parquet`; Part A session; preprocessor (D-5).
**Watch out:** `python -m src.features` builds the file (4 s) and fails loudly if folds or history leak. Fold columns: `fold_group` (trust), `fold_strat` (naive, for the inflation comparison), `temporal`.

## 2026-11-09 · Perfect (P1) · `p1/explore-train-test-structure`, PR #38 (merged)
**Did:** `notebooks/01_eda.ipynb` section on Train/Test structure; filled evidence into D-3, D-4, D-5.
**Found:** (1) Two-lender loans = same borrower, amount, date, labels agree 2,132/2,134, default 8.3% vs 1.4%. (2) The 920 "shared loans" are the other lender's row of a two-lender loan: 100% same customer/amount/date, 0% same lender, so Train labels leak Test labels. (3) Ghana: 0% customers seen, 97% rows with unseen lender, 87% with unseen loan type. (4) Kenyan Test spans 2021-2024 like Train, so it is not temporal. (5) 57 loans over 1M include 27 defaults.
**Next:** Part A session with Henry (D-2 to D-7); then rough feature file and the feature audit table.
**Watch out:** Data is symlinked into `data/raw/` from the folder above the repo (git-ignored). The Findex and inclusion files are not yet used.

## 2026-11-07 · Perfect (P1) · setup (merged to `main`)
**Did:** Assigned P1/P2; created `docs/`, README tracking section. Reviewed Henry's branch.
**Found:** The brief's debrief window was Nov 2–6. README plan misses Datasheet, proxy analysis, a mitigation and the AI declaration.
**Next:** PR for setup; Part A session with Henry (D-2 to D-7).
**Watch out:** P3 and P4 had nobody at the time (Bernard and Eugene were added on 2026-11-10). Confirm the deadline.

## 2026-11-01 · Henry · `explore/henry-eda`
**Did:** `01_eda.ipynb` at repo root: shapes, default rate, missing values, repayment markup by outcome, new vs repeat, default rate by year.
**Found:** Rate by year: 2021 53.8% (39 loans), 2022 1.3% (64,405), 2023 11.5% (1,240), 2024 8.7% (2,970). Only this output is saved.
**Next:** Rate by month, lender, loan type; compare random vs customer vs time splits.
**Watch out:** Merged 2026-11-07 by Perfect; moved to `notebooks/explore_target_henry.ipynb` and path changed to `data/raw/`. Not yet rerun after the path change, and only the by-year output is saved.

## 2026-10-21 · Bernard Adjei · `main`
**Did:** README: data guide for all three zips, framework, pipeline, roles, rules, roadmap issues #1–#37.
**Found:** Leak suspects, drift by year, unseen lenders and loan types, Findex access gaps.
**Next:** Assign roles; explore by slice.
**Watch out:** No code or requirements yet.
