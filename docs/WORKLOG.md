# Work log (handoff notes)

Add an entry at the **top** at the end of each session, 5–10 lines. Link to notebooks or PRs instead of pasting results.

```
## YYYY-MM-DD · Name (role) · branch or PR
**Did:** …   **Found:** … (numbers + where)   **Next:** …   **Watch out:** …
```

---

## 2026-11-19 · Eugene (P4) · `p4/27-37-act843-and-reflection`
**Did:** Read Act 843 from the Data Protection Commission's official PDF and put section numbers in `docs/audit/regulatory_map.md` (s.17–28, s.32–35, s.41), plus a new s.18(2) row (Kenyan training data → Kenya's DPA 2019, not assessed). Re-ran `04` after Bernard's D-30 limit cap and refreshed the Ghana figures in 04 and `docs/audit/`.
**Found:** s.41 (automated decisions: notify, reconsider within 21 days) has a contract exemption (s.41(4)(a)–(b)) that arguably covers loan applications; we keep the review route because the BoG Directive requires one. After D-30, 7% of Ghanaians are approved in full and 79% limited to a median 46% of the request (was 0% and 15%).
**Next:** My reflection is submitted separately, not in the repo. Act 726 section numbers need a printed copy (no official text reachable online).

## 2026-11-19 (late night) · Bernard (P3) · `p3/36-slides-and-clear-figures` (#36)
**Did:** Built the 10-slide executive debrief PowerPoint deck (`slides/NkwaPay_Prosit2_Debrief.pptx`) with embedded 200-dpi figures, per-slide presenter notes, and a run-of-show + 12-question mock Q&A guide (`slides/SPEAKER_NOTES.md`). Upgraded `src/visualization/interpretation.py` (`plot_coefficients`, `plot_shap_importance`, `plot_ablation`) with plain-English feature labels, a risk-direction legend, and non-overlapping legend placement, then regenerated `interp_coefficients.png`, `interp_shap_lightgbm.png`, and `interp_ablation*.png`.
**Found:** Visual audit caught a legend overlap on the initial `interp_ablation.png`; moving the legend below the x-axes and using plain-English labels (`Without borrower history`, `Without calendar dates`, `Adding macro indicators`) makes every slide chart readable for non-technical committee members. `make acceptance ISSUE=36` passes.
**Next:** Rehearse the 15-min presentation and mock Q&A without notes; complete Part A session rulings (#7/#9), branch protection (#56), and personal reflection (#37).

## 2026-11-19 (late night) · Bernard (P3) · `p3/17-polish-03-regression-tone` (Refs #17)
**Did:** Rewrote the narrative markdown cells of `notebooks/03_regression.ipynb` to match the plain-English, question-driven structure and `**Reading:**` breakdown style of `01_eda.ipynb` (opening questions list, plain-English intro before every code block explaining *why* and technical terms when needed, `**Reading:**` takeaways after every chart/table, and a closing `Findings for the report` summary table); re-executed all 17 cells end-to-end with zero errors.
**Found:** `02_classification.ipynb` (Henry) and `04_transfer_and_fairness.ipynb` (Eugene) already follow the same `01_eda.ipynb` question-and-Reading tone and were left untouched in line with notebook ownership (D-20). `make acceptance ISSUE=17` and `ISSUE=33` both pass.
**Next:** Part A session rulings (#7/#9), branch protection (#56), debrief slides (#36), and personal reflection (#37).

## 2026-11-19 (late night) · Bernard (P3) · `p3/22-interpretation` (#22)
**Did:** Wrote `docs/interpretation.md` explaining in plain language what drives NkwaPay's default and loan-limit models: deployed logistic regression standardized weights (`interp_coefficients.png`), LightGBM mean |SHAP| values (`interp_shap_lightgbm.png`), the period-marker finding (`Lender_portion_Funded`, `disb_month`), the feature-group ablation evidence (`reports/tables/ablation.csv`, `interp_ablation*.png`, #52), and how `src/serving/reasons.py` filters non-actionable fields and produces borrower-facing reasons.
**Found:** Tying coefficients, SHAP, and ablation together gives a cohesive defence narrative: borrower history, product type and loan terms carry real signal, while calendar features inflate LightGBM random-fold PR-AUC (+0.044) yet hurt 2023+ temporal accuracy (−0.020 ROC-AUC). `make acceptance ISSUE=22` passes.
**Next:** Confirm Part A session rulings (#7 / #9) with the team, configure branch protection (#56), assemble final report (#34), slides (#36), and personal reflection (#37).

## 2026-11-19 (late night) · Bernard (P3) · `p3/52-ablation-study` (#52)
**Did:** Built the feature-group ablation module (`src/evaluation/ablation.py`, `plot_ablation` in `src/visualization/interpretation.py`, `scripts/run_ablation.py`, `make ablation`, `tests/test_ablation.py`) and registered `interp_ablation.png` / `interp_ablation_roc.png` in `scripts/make_figures.py` and `reports/FIGURES.md`. Retrained deployed `logreg` (D-29) and `lightgbm` across all 7 feature variants (`full`, `- borrower_history`, `- loan_terms`, `- calendar`, `- lender_product`, `- amount`, `+ macro`) on `fold_strat`, `fold_group`, and `temporal` (`reports/tables/ablation.csv`).
**Found:** (1) **Calendar (`disb_month/dow/dom`) is a period-marker artefact for trees:** dropping it hurts LightGBM PR-AUC by −0.044 on random folds (`0.303 → 0.259`), yet **improves** LightGBM on 2023+ `temporal` (ROC-AUC +0.020 to 0.602, PR-AUC +0.005 to 0.153) and improves `logreg` GroupKFold PR-AUC (+0.008). (2) **Borrower history** drives ROC-AUC on every split (`logreg` −0.053 GroupKFold / −0.036 temporal; `lightgbm` −0.059 / −0.028). (3) **Lender/product** drives PR-AUC (`logreg` −0.031 GroupKFold; `lightgbm` −0.020 temporal), and **loan terms** carry genuine temporal ROC-AUC (−0.018 `logreg`, −0.024 `lightgbm`). (4) **Adding macro back (`+ macro`)** hurts `logreg` PR-AUC on every scheme and temporal ROC-AUC (−0.005), adds 0.000 temporal value for LightGBM, and reinforces the D-18 seed result (removing macro raised `lit_mlp` temporal ROC-AUC 0.48 → 0.65).
**Next:** Write plain-language interpretation doc `docs/interpretation.md` (#22) tying together coefficients, LightGBM SHAP, period markers, and these ablation results.

## 2026-11-19 (late night) · Bernard (P3) · `p3/17-task2-decision-tree` (#17)
**Did:** Added Section 4 to `notebooks/03_regression.ipynb` comparing the P3 Task 2 pruned `decision_tree` against all other Task 2 classifiers using `reports/tables/task2_model_summary.csv`, inspecting its pruning settings (`max_depth=6`, `ccp_alpha=0.0001` in `configs/models.yaml`), depth/leaves and top splits; updated Section 5 to fit/demo the D-29/D-30 scorer (`logreg` + `lightgbm_reg_q85`) and re-executed `03_regression.ipynb` top to bottom via `nbconvert --execute`.
**Found:** `decision_tree` achieves `fold_group` ROC-AUC 0.849 / PR-AUC 0.157 (beating `knn` 0.155 and `svm_rbf` 0.073) but drops hardest on `temporal` (ROC-AUC 0.559, PR-AUC 0.113) because top splits track period-sensitive features. `test_issue_17_p3_models` passes and all 14 cells execute cleanly.
**Next:** Run the feature-group ablation study (#52) and write the plain-language interpretation doc (#22).

## 2026-11-19 (late night) · Bernard (P3) · `p3/53-quantile-limit-rule` (#53)
**Did:** Registered `lightgbm_reg_q85` (`objective="quantile", alpha=0.85`) in `configs/models.yaml`, `src/models/__init__.py`, and `src/models/ensembles.py`; switched `fit_and_save_deployment_artifacts` (`src/training/artifacts.py`) and `scripts/train.py` to use it as the deployed limit cap (D-30); added unit tests in `tests/test_architecture.py` and `tests/test_serving.py`; refitted the bundle and regenerated `make figures` and `make score`.
**Found:** On `fold_group` OOF, `lightgbm_reg_q85` cuts only **15.7%** of repaid Kenyan loans (vs 50.1% under mean regression, and 19.6% of defaulted loans). On Test (`reports/tables/product_decisions_by_country.csv`), Kenyan `"limit"` drops from **48.9% → 14.6%** (83.0% approved in full; ≤25% acceptance target met), Ghana gets 6.9% approved / 79.4% limited / 100% flagged for review, and 1,000-call latency is median 21 ms, p95 26 ms.
**Next:** Add Task 2 pruned `decision_tree` section to `03_regression.ipynb` and re-execute (#17), run feature-group ablation (#52), and write plain-language interpretation (#22).

## 2026-11-19 (late night) · Bernard (P3) · `p3/51-choose-deployed-models` (#51)
**Did:** Recorded **D-29** in `docs/DECISIONS.md` choosing **Platt-calibrated logistic regression (`logreg`)** for Task 2 and **LightGBM (`lightgbm_reg`)** for Task 1. Refitted the scoring bundle on the D-28 cedi cost matrix (threshold 0.1006) and D-26 starter loan (`python -m scripts.train --save-artifacts --clf-artifact-model logreg --reg-artifact-model lightgbm_reg`), updated `scripts/make_figures.py` to record `starter` in `product_decisions_by_country.csv`, and regenerated `make eval`, `make figures`, and `make score`.
**Found:** `logreg` overlaps `random_forest` and `lightgbm` on GroupKFold PR-AUC 95% CIs (0.175 [0.154, 0.199] vs 0.203 / 0.194), beats both on temporal 2023+ ROC-AUC (0.633 vs 0.619 / 0.582), matches `random_forest` on new-borrower calibration (−2.4 pp gap; `lightgbm` is −6.3 pp), and runs ~31× faster with exact linear reason codes (median 24 ms, p95 35 ms over 1,000 calls vs ~289 ms for RF TreeSHAP). All `reports/tables/` and 35 figures now reflect the D-28 cedi cost matrix.
**Next:** Upgrade the Task 1 limit rule to an 85th-quantile cap (#53), add Task 2 decision tree to `03_regression` (#17), run ablation (#52), and write interpretation (#22).

## 2026-11-19 (late night) · Bernard (P3) · `p3/32-linear-derivation` (#32)
**Did:** Wrote `docs/derivations/linear.md` (Gaussian likelihood → OLS/MSE → gradient & convexity → normal equations → Ridge as MAP with Gaussian prior → connection to Lasso/`LitLinear` → why linear models beat `mean` on log USD but lose on dollar RMSE) and `tests/test_linear_derivation.py` (5 numerical checks against scikit-learn `LinearRegression`, `Ridge(alpha=10.0)`, and PyTorch autograd, plus collinearity condition-number reduction and Jensen's inequality).
**Found:** All four derivations (`svm.md`, `logistic.md`, `linear.md`, `boosting.md`) are now complete and pass `make acceptance ISSUE=32`. Ridge (`alpha=10`) cuts `fold_group` dollar RMSE from $2,089 (OLS) to $1,745 (−16%) at almost zero cost in log RMSE (1.1094 → 1.1115).
**Next:** Choose deployed models (#51), switch limit rule to high quantile (#53), add Task 2 decision tree to `03_regression` (#17), ablation (#52), and interpretation (#22).

## 2026-11-19 (night) · Perfect (P1) · `p1/34-report-data-section` (#34)
**Did:** Drafted P1's section of the technical report, `docs/report/p1_data_section.md` (about 2 pages): data and the decision it supports, leakage audit, time-safe features, three validation schemes, baseline/kNN/SVM, limitations, with a sources list. Every number comes from committed tables or the EDA; none was newly estimated.
**Found:** Random folds overstate LightGBM's PR-AUC by 55% (0.303 vs 0.195 by customer); logistic regression is barely affected (0.186 vs 0.175), so the inflation grows with model flexibility. By time every model's ROC-AUC is 0.56 to 0.65. Re-ran two deliberate bugs on the full feature build (exact-date matches allowed; outcomes counted before the due date): `validate()` caught both (`n_past_loans`, `n_past_labelled`).
**Also did:** Added Perfect's four missing rows to `docs/AI_USE_LOG.md` (the log had one row since 7 Nov). Two columns are left as `_Perfect to fill in_` / `_Perfect to confirm_` on purpose: only Perfect can say what he ran himself and whether he understands it.
**Next:** Bernard assembles the report (#34); someone reads this section against the others for overlap. My reflection (#37) is still to write, in my own words.
**Watch out:** the report is a draft for review; the Ghana and fairness numbers belong to Eugene's and Henry's sections and are not repeated here.

## 2026-11-19 (later) · Perfect (P1) · `p1/9-part-a-session-prep` (#9)
**Did:** Drew the Part A loan timeline (`scripts/make_timeline_figure.py` → `reports/figures/part_a_loan_timeline.png`, linked from `docs/feature_audit.md`). Wrote `docs/PART_A_SESSION.md`: where each Part A item lives, and for the 8 open drafts (D-2, D-3, D-4, D-5, D-6, D-13, D-14, D-18) the question, evidence, recommendation and what changes if we disagree. Added a dated caveat to D-3's evidence and fixed D-5's path (`src/processing/preprocess.py`).
**Found:** Issue #9's card is stale: it lists D-7, D-19 and D-20 as drafts, but D-7 and D-20 are decided and D-19 is superseded by D-28. The drafts really still open are the 8 above. The data has no application or approval date, so `disbursement_date` is the closest column to the decision moment.
**Next:** hold the session (about 45 minutes), then mark the confirmed rows `decided` and close #9 and #7.
**Watch out:** Nothing was changed in code the models use; no results were re-run. The session itself has not happened, so every Part A decision is still a draft.

## 2026-11-19 · Perfect (P1) · `p1/32-svm-derivation` (#32)
**Did:** Wrote `docs/derivations/svm.md` (widest street, soft margin and hinge loss, class weights, Lagrangian, dual, KKT cases, RBF kernel, Platt scaling, why 12,000 rows) and `tests/test_svm_derivation.py`, 8 checks of each claim against scikit-learn `SVC` on 2-D toy data (margin = 2/‖w‖ by brute force, w = Σαyx, Σαy = 0, primal = dual, box constraint with class weights, RBF decision function rebuilt by hand).
**Found:** SVM by customer ROC-AUC 0.805 (kNN 0.769; logistic 0.843; Random Forest 0.871) but PR-AUC 0.073, the weakest learner (kNN 0.155), see `reports/tables/task2_model_summary.csv`. By time ROC-AUC 0.570. The subsample leaves about 220 defaulters; that it explains the weak PR-AUC is a guess, not tested.
**Also did:** `docs/learning/P1_models_explained.md` for #15: majority, kNN and SVM in plain language, with a side-by-side table and report text. Caught that `selection_rate` in the summary CSV is the share *approved*, and that the decline cut-off moved from 0.036 (D-22) to about 0.10 (D-28), so the CSV's confusion counts are illustrative only.
**Also did (#48):** `01_eda` §6 links `reports/EDA_REPORT.md` and re-derives its headline numbers from `Train.csv`. Two corrections: (1) the report's by-year, by-group and by-history tables label a *row* count as `loans` (2023: 1,240 rows, 848 loans); (2) two-lender loans' 8.3% vs 1.4% default gap is mostly timing (0.9% of 2022 loans, 46% of 2023, 64% of 2024; within 2021-22 they default more, within 2023+ less), so `n_lenders_on_loan` is probably a period marker. My own `01_eda` §1 reading had repeated the claim; fixed. Report text and generator (`scripts/make_eda_report.py`, Bernard's) are **not** changed; Bernard should fix the labels.
**Next:** review, then merge. #32 stays open until the linear regression derivation (Bernard) exists. Then #15 write-up, #48 closing section of `01_eda`.
**Watch out:** I cited Henry's by-time numbers from an older doc at first; they changed after D-18 (MLP 0.48 → 0.648). Cite `task2_model_summary.csv`, not older notes. Local `main` had to be reset because Bernard's rewrite changed commit hashes (backup branch `backup/main-before-reset`).

## 2026-11-18 (later) · Eugene (P4) · `p4/65-borrower-app` (#65, stacked on PR #64)
**Did:** Borrower Streamlit app (`app/streamlit_app.py`) now calls `src.serving.score_application`: approve / limit / starter / decline cards, `borrower_reasons` for the borrower, model reasons + review flags in a credit-team panel. Demo controls gained a lender dropdown (default 267278) and repeat-borrower history inputs. `streamlit` added to `requirements.txt` and the lock (no existing pins changed). `tests/test_app.py` renders the app headless.
**Found:** Ghana repeat borrower, GHS 500 / 7 days → limit GHS 327; first-time → starter GHS 125; Type_14 → decline (p 0.75).
**Next:** merge PR #64 first, then this one. **Watch out:** run `make features artifacts` before `streamlit run app/streamlit_app.py`.

## 2026-11-18 · Eugene (P4) · `p4/54-transfer-fairness-audit`
**Did:** Rebuilt `04_transfer_and_fairness` on the restructured pipeline and fixed its four known problems (#54). Switched the team cost matrix to **cedis with 3 future loans lost** (D-28, threshold 0.10) and put the **starter loan** in the scorer (D-26): `decide()` in `src/serving/scorer.py`, shared by `score_application` and `score_frame`, new decision `starter`, `offered_local` in batch output; `borrower_reasons` rewords the lender reason and drops filler. Added `decision_cost_by_group`, `proxy_strength`, `run_domain_classifier(drop_cols=...)`, `effective_sample_share`. Tests: `tests/test_fairness_and_shift.py` (12) + 6 serving/config tests. Wrote `docs/audit/` and `docs/derivations/boosting.md`. D-25 to D-28.
**Found:** (1) First-time borrowers: 49% approved vs 98% repeat at 0.10 (0% under the old KES 0.019); the cost assumption drives it (12% → 70% across 1–6 lost loans). (2) Starter loan up to p 0.30 → 81% (ratio 0.83) for +GHS 10.0k on 567 applicants. (3) Ghana separable even without the label-history artefact (AUC 0.985); importance weights ESS share 0.04 (D-27). (4) No Ghanaian approved in full: 86% limited to ~15% of the request. (5) RF best by customer (PR-AUC 0.203), every model ≈ 0.6 ROC-AUC by time.
**Next:** Bernard reviews (scorer changes); Henry and Bernard re-run `02`/`03` and `make all` so `reports/` matches the cedi config; I confirm the Act 843/726 section numbers.
**Watch out:** `configs/cost_matrix.yaml` is now GHS: `reports/tables/cost_sensitivity.csv`, `fairness_*.csv`, `product_decisions_by_country.csv` and `02` §9 outputs predate it until regenerated. `04` needs `make features artifacts` first.

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
