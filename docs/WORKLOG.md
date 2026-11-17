# Work log (handoff notes)

Add an entry at the **top** at the end of each session, 5–10 lines. Link to notebooks or PRs instead of pasting results.

```
## YYYY-MM-DD · Name (role) · branch or PR
**Did:** …   **Found:** … (numbers + where)   **Next:** …   **Watch out:** …
```

---

## 2026-11-17 (night) · Eugene (P4) · `p4/54-transfer-fairness-audit`
**Did:** Rebuilt `04_transfer_and_fairness` on the restructured pipeline and fixed its four known problems (#54): calibrate before every threshold, mitigation with its cost, proxy analysis, audit docs. Added `decision_cost_by_group`, `proxy_strength`, `run_domain_classifier(drop_cols=...)` and `effective_sample_share` to `src/evaluation/` with `tests/test_fairness_and_shift.py` (12 tests). Wrote `docs/audit/` (Model Card, Datasheet, regulatory map, decline message, `audit.md` summary) and `docs/derivations/boosting.md`. D-23 to D-25.
**Found:** (1) The shipped scorer approves **0% of first-time borrowers** (ratio 0; real default 21%). At the cedi threshold it is 12%, and at 6 lost future loans 49%: the cost assumption drives it. (2) A 25% starter loan at p ≤ 0.05 → 36% approved for +KES 8.7k on 567 applicants (D-24). (3) Ghana is separable even without the label-history artefact (AUC 0.985), and the importance weights have an ESS share of 0.04, so weighting is unusable (D-25). (4) No Ghanaian is approved in full: 70% are limited to ~16% of the request, and declines move 30% → 79% if the unknown history is treated as clean. (5) RF best by customer (PR-AUC 0.203), but every model is ≈ 0.6 ROC-AUC by time.
**Next:** Bernard reviews; team settles the cost matrix (KES vs cedis, future loans) and D-24; I confirm the Act 843/726 section numbers on GhaLII.
**Watch out:** `04` needs `make features artifacts` first (it uses the saved scorer for Ghana). Every number in its readings assumes threshold 0.019: if the config changes, re-run and re-read §3.

## 2026-11-17 (late night) · Henry (P2) · `p2/32-logistic-derivation`, PR #62
**Did:** `docs/derivations/logistic.md`: log-odds → sigmoid → Bernoulli likelihood → BCE (and its stable form) → gradient (1/n)Xᵀ(σ(Xw) − y); Hessian and convexity; the `pos_weight` version our Lightning model trains. `tests/test_logistic_derivation.py` checks the analytic gradients against PyTorch autograd.
**Found:** Weighting defaulters by c ≈ 54 shifts every log-odds by about log c ≈ 4.0, which explains why raw scores averaged 0.28–0.32 against a true 0.018 and why we calibrate afterwards (`02` §8).
**Next:** Personal reflection (#37). #32 stays open until Perfect (SVM), Bernard (linear) and Eugene (boosting) add theirs.
**Watch out:** The #32 acceptance check needs all four files in `docs/derivations/`, each ≥ 1,500 characters.

## 2026-11-17 (night) · Henry (P2) · `p2/55-temporal-calibration`, PR #61
**Did:** New `02_classification` §11 (#55): trained on pre-2023 loans with Platt fitted on their out-of-fold scores, checked calibration on 2023+ loans, then recalibrated on 2023 and checked on 2024; quarterly monitoring table; recommendation (D-24). Conclusions renumbered to §12.
**Found:** (1) Stale calibration under-predicts: LogReg 4.5% predicted vs 8.7% actual on 2024 loans (ECE 0.076; MLP 7.1% vs 8.7%). (2) Believing it, the Ghana rule approves 89% at an 8.1% default rate: 183 GHS loss per application. (3) Recalibrating on 2023 halves ECE (0.038 / 0.029) but every 2024 applicant is then above the 3.6% break-even (lowest 6.2% / 4.9%), so the rule declines everyone; ROC-AUC unchanged (0.64–0.66). (4) Quarterly gaps reach 7.8 pp, visible within one quarter.
**Next:** Logistic regression derivation (#32); personal reflection (#37); raise the pricing question (fee vs risk) for the report and #24.
**Watch out:** §11 retrains both models 6 times on pre-2023 loans; the notebook now takes about 1 min 45 s.

## 2026-11-17 (later still) · Bernard (P3 / PM) · `p3/58-readme-status-after-merge` (PR #58)
**Did:** Acceptance tests for every issue (`tests/acceptance/test_issues.py`, `make acceptance`), the *Issue acceptance* Action (PR must pass the tests of the issues it closes; a closed issue whose test fails is reopened), and an "Automated check" row on every README card (D-23). Fixed the README status Action failing before its script reached main. Repo About and topics set. No more Claude co-author trailers in commits.
**Found:** scoreboard today: 18 checks pass; open issues whose check already passes (#8, #15, #18, #19, #20, #33) can be closed by their owners after review; 28 fail as expected for open work.

## 2026-11-17 (evening) · Henry (P2) · `p2/11-ghana-cost-and-conclusions`, PR #60
**Did:** Reviewed and approved PR #47 (tested on macOS). Added to `02_classification`: §0 problem framing (#10), §9b cost matrix in cedis from Ghana's own loans (#11; `src.evaluation.measure_market_cost_inputs` + unit test), §11 conclusions. Verified Bernard's §7–10 (#19, #20) by rerunning them. Decided D-7, added D-22.
**Found:** (1) Ghana: median loan 3,002 GHS, fee 3.0% → C_FP 90, C_FN 2,416 GHS, **decline if p > 0.036** (0.029–0.047 across recovery; 0.15–0.23 if 6 future loans are lost). (2) At 0.036 the calibrated MLP approves 88.6% and cuts expected loss 44.3 → 25.0 GHS per application (−44%); the KES config's 0.019 approves only 78.7%, below the proposed 80% floor. (3) With customer-level bootstrap the MLP vs LogReg PR-AUC intervals overlap, so the MLP's lead is not established.
**Next:** Logistic regression derivation (#32); agree with Bernard and Eugene whether `configs/cost_matrix.yaml` switches to the D-22 cedi values; temporal calibration (#55); personal reflection.
**Watch out:** On macOS `pytest tests/` segfaults inside `tests/test_end_to_end.py` (LightGBM and PyTorch OpenMP clash; passes on Linux CI). Notebook `02` itself runs fine.

## 2026-11-17 (late) · Bernard (P3 / PM) · `p3/58-readme-status-after-merge`
**Did:** PRs #47 and #57 merged; closed #21. The README status is now generated: `scripts/update_readme.py` rebuilds the issue badge, progress table and per-person cards from live issues and PRs; the *README status* Action runs it on every issue/PR change, daily and on demand (D-21). Card text moved to `docs/issue_cards.yaml`. Static notebook statuses updated to merged.
**Watch out:** when `main` is protected (#56), allow GitHub Actions to push, or the README sync fails.

## 2026-11-17 (night) · Bernard (P3 / PM) · `p3/48-eda-report-figures-tests` (PR #57)
**Did:** Assigned every open issue on GitHub to its owner (P1 Perfect, P2 Henry, P3 Bernard, P4 Eugene; team issues to all four). README "What's left, by person": a summary table per person, then one expandable card per open issue (what it is, why it matters for the grade, what already exists, where to work, how to run it, when it is done, numbered steps), plus how to pick up an issue.
**Next:** each person opens their cards and starts; Eugene first (#54, #9 for all).

## 2026-11-17 (evening) · Bernard (P3 / PM) · `p3/48-eda-report-figures-tests` (stacked on PR #47)
**Did:** `reports/EDA_REPORT.md` (`make eda`): 10 data-science questions, each with figures, tables, computed findings and the decision it supports. Defence figure pack (`make figures`) with catalogue `reports/FIGURES.md`: preprocessing, validation, evaluation (PR/ROC, reliability before/after, customer-level CIs, cost sensitivity, threshold sweep, confusion matrix, calibration by tenure), Task 1 diagnostics, interpretation (coefficients, SHAP), fairness by group, Kenya-Ghana shift, and the product on all Test applications (`score_frame`). One house style at 200 dpi. Tests: 127 pass, including an end-to-end run from fake raw CSVs to scored applications, reports and figures; model architecture, normalization/denormalization, feature time-safety, and every plot. Issues #48-#56.
**Found:** the README's leakage claim was imprecise: the repayment columns alone score AUC 0.66-0.70; their ratio to principal scores 0.98. Longer terms are much riskier (1.2% at 7 days, 31% above 60). The limit rule cuts about half of requests by construction (Kenya 42% limited; Ghana 70% limited, none fully approved) (#53). Tests caught two real bugs: scripts crashed when outputs lived outside the repo, and the latency chart crashed (the earlier figure run had silently stopped).
**Next:** Henry reviews PR #47 then this PR; P1 links the EDA report from `01_eda` (#48); Eugene takes #54; Bernard #51 and #53.
**Watch out:** `make figures` needs `make train` first; `make test-notebooks` runs the notebooks with real data (slow).

## 2026-11-17 (later) · Bernard (P3 / PM) · `p3/30-scorer-fixes-and-structure`
**Did:** Reviewed PR #46 (merged without review). Restructured `src/` into packages (`processing`, `models` one file per family + registry, `training`, `evaluation`, `serving`, `visualization`, `analysis`) with entry points in `scripts/`, settings in `configs/` and committed outputs in `reports/`; Python 3.12 + cross-platform `requirements-lock.txt` in CI. Fixed #30: Task 1 now predicts USD and the scorer converts with the exchange rate; macro columns and `log_amount_local` are no longer model inputs (D-18); Platt calibration (never 0%); reason codes limited to what a borrower can act on and never contradicting their record; `needs_review` flags for out-of-scope applications; cost matrix from data instead of the brief's example (D-19); target scaler clipped; customer-level bootstrap. Restored Henry's original `02_classification` (D-20), fixed its data path and feature list, added headings and §7–10 (calibration, cost threshold, CIs). Ran all 16 models on all schemes (`make train`, `make eval`, `make figures`). 41 tests pass.
**Found:** New Ghana borrower now scores 8.9% (was 0%) and is flagged for review; Kenya and Ghana limits share one USD model. MLP by-time ROC-AUC rose from 0.48 to 0.65 once the macro year-markers were removed. Random folds inflate LightGBM PR-AUC to 0.303 vs 0.195 by customer. Linear Task 1 models lose to the mean in dollars (heavy tail); trees win. Domain classifier AUC 0.996 is driven by `n_past_labelled` = 0 for every Ghana row (labels exist only in Train). The deployed logistic regression under-predicts new borrowers (2.2% vs 5.7% observed).
**Next:** Review and merge PR #47 (Henry reviews, per the rotation). Henry settles the future-loss assumption (D-19). Eugene owns `04`. Bernard: #7 EDA, Task 2 tree (#17), linear derivation, choose the deployed classifier (MLP / RF vs LogReg).
**Watch out:** Commands are now `python -m scripts.<name>` or `make <target>`; `python -m src.features` still works. The SVM is trained on a 12,000-row subsample and is the weakest model (PR-AUC 0.07). Notebook 04 still has PR #46 outputs; Eugene should rerun it.

## 2026-11-17 · Bernard (P3 / PM) · PR #46
**Did:** Converted non-deliverable `notebooks/explore_target_henry.ipynb` into `src/eda_target.py` (with full Issue `#6` breakdowns: tenure, year, month, lender, loan type, `extra_pct`, and 2023+ lender/population shift) and deleted `notebooks/explore_target_henry.ipynb`; refactored `notebooks/02_classification.ipynb` and created `notebooks/03_regression.ipynb` and `notebooks/04_transfer_and_fairness.ipynb` as thin, top-to-bottom executable wrappers importing 100% from `src/`; added `eda-target` to `Makefile`, enforced the exact 4-deliverable-notebooks rule in `tests/test_leakage_and_splits.py`, and recorded Decision D-17.
**Found:** `python -m src.eda_target` shows the 2023+ jump (1.33% -> 9.53% default) is driven by **both** lender mix (`267278` share dropped 97.1% -> 49.0%) **and** nearly 10x higher defaults inside `267278` itself (0.95% -> 9.12%).
**Next:** Merge `p3/13-30-senior-scaffolding` PR (`Closes #13, refs #6, #30`), hold Part A session (#9) to confirm D-2..D-7, and complete `docs/audit/`, `docs/derivations/`, technical report, slides, and individual reflections.
**Watch out:** Non-deliverable notebooks must be written as scripts in `src/`, not added to `notebooks/` — CI `test_only_four_deliverable_notebooks_exist` enforces this.

## 2026-11-16 · Bernard (P3 / PM) · PR #46
**Did:** Built shared senior-level ML engineering scaffolding on `p3/13-30-senior-scaffolding` without modifying P1 (`src/features.py`, `01_eda.ipynb`) or P2 (`02_classification.ipynb`, `explore_target_henry.ipynb`) files: added `NUM_FEATURES_T1`, `NUM_FEATURES_T2`, `PRED_COLUMNS`, and cost constants to `src/config.py`; created `src/preprocess.py` (`transforms.py` alias), `src/datamodule.py`, `src/train.py`, `src/evaluate.py`, `src/fairness.py`, and `src/score.py` (Part E scorer + 1,000-call latency benchmark); added automated `pytest` guardrails (`tests/`), `ci.yml`, `pr-issue-check.yml`, `.pre-commit-config.yaml`, `pyproject.toml`, and `Makefile`; streamlined `README.md` (moving full data dictionary/glossary to `docs/DATA_AND_PIPELINE_GUIDE.md`) and added `docs/P3_BERNARD_PLAN.md` & `docs/P4_EUGENE_PLAN.md`.
**Found:** `data/processed/features.parquet` has shape `(87248, 31)` matching `src/features.py`. Excluding `log_amount_usd` and `log_amount_local` from `NUM_FEATURES_T1` prevents Task 1 target leakage while keeping all 19 numeric features for Task 2.
**Next:** Henry reviews `p3/13-30-senior-scaffolding` PR (`Closes #13, refs #30`); team runs `make train` / `make eval`, holds Part A session (#9) to finalize D-2..D-7, and finishes `02_classification.ipynb`, `03_regression.ipynb`, and `04_transfer_and_fairness.ipynb`.
**Watch out:** `.venv` is managed with `uv` (`uv pip install`); added `pytest>=7.4` and `ruff>=0.3` to `requirements.txt`. Run `pytest tests/ -v` and `ruff check src/ tests/` before every PR.

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
